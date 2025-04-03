# Code derived from stdlib.functools
# Credit and huge thanks to:
# Nick Coghlan <ncoghlan at gmail.com>,
# Raymond Hettinger <python at rcn.com>,
# Lukasz Langa <lukasz at langa.pl>.
# Copyright (C) 2006 Python Software Foundation.

__all__ = ["singledispatch", "singledispatchmethod"]

from abc import get_cache_token
import weakref
from types import GenericAlias, MappingProxyType, UnionType
import typing

from functools import update_wrapper

from .mro import *

################################################################################
### update_wrapper() and wraps() decorator
################################################################################

# update_wrapper() and wraps() are tools to help write
# wrapper functions that can handle naive introspection

WRAPPER_ASSIGNMENTS = ("__module__", "__name__", "__qualname__", "__doc__",
                       "__annotate__", "__type_params__")
WRAPPER_UPDATES = ("__dict__",)

from functools import update_wrapper
from complexdispatch.mro import *
from complexdispatch.find_impl import _find_impl
from complexdispatch.utils import _pep585_registry_matches

################################################################################
### singledispatch() - single-dispatch generic function decorator
################################################################################

def singledispatch(func):
    """functools.singledispatch with support for PEP-585

    Transforms a function into a generic function, which can have different
    behaviours depending upon the type of its first argument. The decorated
    function acts as the default implementation, and additional
    implementations can be registered using the register() attribute of the
    generic function. Adds support for collections typing such as `list[str]`.
    """

    registry = {}
    dispatch_cache = weakref.WeakKeyDictionary()
    cache_token = None

    def _fetch_dispatch_with_cache(cls):
        try:
            impl = dispatch_cache[cls]
        except KeyError:
            try:
                impl = registry[cls]
            except KeyError:
                impl = _find_impl(cls, registry)
            dispatch_cache[cls] = impl
        return impl


    def dispatch(cls_obj):
        """generic_func.dispatch(cls) -> <function implementation>

        Runs the dispatch algorithm to return the best available implementation
        for the given *cls* registered on *generic_func*.

        """
        cls = (cls_obj if isinstance(cls_obj, type) else cls_obj.__class__)
        nonlocal cache_token
        if cache_token is not None:
            current_token = get_cache_token()
            if cache_token != current_token:
                dispatch_cache.clear()
                cache_token = current_token

        # if PEP-585 types are not registered for the given *cls*,
        # then we can use the cache. Otherwise, the cache cannot be used
        # because we need to confirm every item matches first
        if not any(_pep585_registry_matches(cls, registry)):
            return _fetch_dispatch_with_cache(cls)

        return _find_impl(cls_obj, registry)

    def _is_union_type(cls):
        from typing import get_origin, Union
        return get_origin(cls) in {Union, UnionType}

    def _is_valid_dispatch_type(cls):
        if isinstance(cls, type):
            return True

        if isinstance(cls, GenericAlias):
            from typing import get_args
            return all(isinstance(arg, (type, UnionType)) for arg in get_args(cls))

        return (_is_union_type(cls) and
                all(isinstance(arg, (type, GenericAlias)) for arg in cls.__args__))


    def register(cls, func=None):
        """generic_func.register(cls, func) -> func

        Registers a new implementation for the given *cls* on a *generic_func*.

        """
        nonlocal cache_token
        if _is_valid_dispatch_type(cls):
            if func is None:
                return lambda f: register(cls, f)
        else:
            if func is not None:
                raise TypeError(
                    f"Invalid first argument to `register()`. "
                    f"{cls!r} is not a class or union type."
                )

            #ann = getattr(cls, "__annotate__", None) # 3.14 only
            ann = getattr(cls, "__annotations__", {})
            if not ann:
                raise TypeError(
                    f"Invalid first argument to `register()`: {cls!r}. "
                    f"Use either `@register(some_class)` or plain `@register` "
                    f"on an annotated function."
                )
            func = cls

            # only import typing if annotation parsing is necessary
            from typing import get_type_hints
            #from annotationlib import Format, ForwardRef # 3.14 only
            argname, cls = next(iter(get_type_hints(func).items()))
            if not _is_valid_dispatch_type(cls):
                if _is_union_type(cls):
                    raise TypeError(
                        f"Invalid annotation for {argname!r}. "
                        f"{cls!r} not all arguments are classes."
                    )
                # 3.14 only
                #elif isinstance(cls, ForwardRef):
                    #raise TypeError(
                        #f"Invalid annotation for {argname!r}. "
                        #f"{cls!r} is an unresolved forward reference."
                    #)
                else:
                    raise TypeError(
                        f"Invalid annotation for {argname!r}. "
                        f"{cls!r} is not a class."
                    )

        if _is_union_type(cls):
            for arg in cls.__args__:
                registry[arg] = func
        else:
            registry[cls] = func
        if cache_token is None and hasattr(cls, "__abstractmethods__"):
            cache_token = get_cache_token()
        dispatch_cache.clear()
        return func

    def wrapper(*args, **kw):
        if not args:
            raise TypeError(f"{funcname} requires at least "
                            "1 positional argument")
        return dispatch(args[0])(*args, **kw)

    funcname = getattr(func, "__name__", "singledispatch function")
    registry[object] = func
    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = MappingProxyType(registry)
    wrapper._clear_cache = dispatch_cache.clear
    update_wrapper(wrapper, func)
    return wrapper


# Descriptor version
class singledispatchmethod:
    """Single-dispatch generic method descriptor.

    Supports wrapping existing descriptors and handles non-descriptor
    callables as instance methods.
    """

    def __init__(self, func):
        if not callable(func) and not hasattr(func, "__get__"):
            raise TypeError(f"{func!r} is not callable or a descriptor")

        self.dispatcher = singledispatch(func)
        self.func = func

    def register(self, cls, method=None):
        """generic_method.register(cls, func) -> func

        Registers a new implementation for the given *cls* on a *generic_method*.
        """
        return self.dispatcher.register(cls, func=method)

    def __get__(self, obj, cls=None):
        return _singledispatchmethod_get(self, obj, cls)

    @property
    def __isabstractmethod__(self):
        return getattr(self.func, "__isabstractmethod__", False)

    def __repr__(self):
        try:
            name = self.func.__qualname__
        except AttributeError:
            try:
                name = self.func.__name__
            except AttributeError:
                name = "?"
        return f"<single dispatch method descriptor {name}>"

class _singledispatchmethod_get:
    def __init__(self, unbound, obj, cls):
        self._unbound = unbound
        self._dispatch = unbound.dispatcher.dispatch
        self._obj = obj
        self._cls = cls
        # Set instance attributes which cannot be handled in __getattr__()
        # because they conflict with type descriptors.
        func = unbound.func
        try:
            self.__module__ = func.__module__
        except AttributeError:
            pass
        try:
            self.__doc__ = func.__doc__
        except AttributeError:
            pass

    def __repr__(self):
        try:
            name = self.__qualname__
        except AttributeError:
            try:
                name = self.__name__
            except AttributeError:
                name = "?"
        if self._obj is not None:
            return f"<bound single dispatch method {name} of {self._obj!r}>"
        else:
            return f"<single dispatch method {name}>"

    def __call__(self, /, *args, **kwargs):
        if not args:
            funcname = getattr(self._unbound.func, "__name__",
                               "singledispatchmethod method")
            raise TypeError(f"{funcname} requires at least "
                            "1 positional argument")
        return self._dispatch(args[0]).__get__(self._obj, self._cls)(*args, **kwargs)

    def __getattr__(self, name):
        # Resolve these attributes lazily to speed up creation of
        # the _singledispatchmethod_get instance.
        if name not in {"__name__", "__qualname__", "__isabstractmethod__",
                        "__annotations__", "__type_params__"}:
            raise AttributeError
        return getattr(self._unbound.func, name)

    @property
    def __wrapped__(self):
        return self._unbound.func

    @property
    def register(self):
        return self._unbound.register

