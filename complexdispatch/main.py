# Code derived from stdlib.functools
# Credit and huge thanks to:
# Nick Coghlan <ncoghlan at gmail.com>,
# Raymond Hettinger <python at rcn.com>,
# Łukasz Langa <lukasz at langa.pl>.
# Copyright (C) 2006 Python Software Foundation.

__all__ = ["singledispatch", "singledispatchmethod"]

from abc import get_cache_token
from collections import namedtuple
# import weakref  # Deferred to single_dispatch()
from operator import itemgetter
from reprlib import recursive_repr
from types import GenericAlias, MethodType, MappingProxyType, UnionType
from _thread import RLock

from functools import update_wrapper, wraps

from .mro import *

################################################################################
### update_wrapper() and wraps() decorator
################################################################################

# update_wrapper() and wraps() are tools to help write
# wrapper functions that can handle naive introspection

WRAPPER_ASSIGNMENTS = ("__module__", "__name__", "__qualname__", "__doc__",
                       "__annotate__", "__type_params__")
WRAPPER_UPDATES = ("__dict__",)


################################################################################
### singledispatch() - single-dispatch generic function decorator
################################################################################


def _pep585_registry_matches(cls, registry):
    from typing import get_origin
    return (i for i in registry.keys() if get_origin(i) == cls)

def _find_impl_match(cls_obj, registry):
    """Returns the best matching implementation from *registry* for type *cls_obj*.

    Where there is no registered implementation for a specific type, its method
    resolution order is used to find a more generic implementation.

    Note: if *registry* does not contain an implementation for the base
    *object* type, this function may return None.

    """
    cls = cls_obj if isinstance(cls_obj, type) else cls_obj.__class__
    mro = _compose_mro(cls, registry.keys())
    match = None

    from typing import get_origin, get_args

    if (not isinstance(cls_obj, type) and
        len(cls_obj) > 0 and # dont try to match the types of empty containers
        any(_pep585_registry_matches(cls, registry))):
        # check containers that match cls first
        for t in _pep585_registry_matches(cls, registry):
            if not all((isinstance(i, get_args(t)) for i in cls_obj)):
                continue

            if match is None:
                match = t

            else:
                match_args = get_args(get_args(match)[0])
                t_args = get_args(get_args(t)[0])
                if len(match_args) == len(t_args):
                    raise RuntimeError(f"Ambiguous dispatch: {match} or {t}")

                elif len(t_args)<len(match_args):
                    match = t

    if match:
        return match

    for t in mro:
        if match is not None:
            # If *match* is an implicit ABC but there is another unrelated,
            # equally matching implicit ABC, refuse the temptation to guess.
            if (t in registry and t not in cls.__mro__
                              and match not in cls.__mro__
                              and not issubclass(match, t)):
                raise RuntimeError(f"Ambiguous dispatch: {match} or {t}")
            break
        if t in registry:
            match = t

    return match

def _find_impl(cls_obj, registry):
    return registry.get(
        _find_impl_match(cls_obj, registry)
    )


def singledispatch(func):
    """Single-dispatch generic function decorator.

    Transforms a function into a generic function, which can have different
    behaviours depending upon the type of its first argument. The decorated
    function acts as the default implementation, and additional
    implementations can be registered using the register() attribute of the
    generic function.
    """
    # There are many programs that use functools without singledispatch, so we
    # trade-off making singledispatch marginally slower for the benefit of
    # making start-up of such applications slightly faster.
    import weakref

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

    def _is_valid_dispatch_type(cls):
        if isinstance(cls, type):
            return True

        if isinstance(cls, GenericAlias):
            from typing import get_args
            return all(isinstance(arg, (type, UnionType)) for arg in get_args(cls))

        return (isinstance(cls, UnionType) and
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

            ann = getattr(cls, "__annotate__", None)
            if ann is None:
                raise TypeError(
                    f"Invalid first argument to `register()`: {cls!r}. "
                    f"Use either `@register(some_class)` or plain `@register` "
                    f"on an annotated function."
                )
            func = cls

            # only import typing if annotation parsing is necessary
            from typing import get_type_hints
            from annotationlib import Format, ForwardRef
            argname, cls = next(iter(get_type_hints(func, format=Format.FORWARDREF).items()))
            if not _is_valid_dispatch_type(cls):
                if isinstance(cls, UnionType):
                    raise TypeError(
                        f"Invalid annotation for {argname!r}. "
                        f"{cls!r} not all arguments are classes."
                    )
                elif isinstance(cls, ForwardRef):
                    raise TypeError(
                        f"Invalid annotation for {argname!r}. "
                        f"{cls!r} is an unresolved forward reference."
                    )
                else:
                    raise TypeError(
                        f"Invalid annotation for {argname!r}. "
                        f"{cls!r} is not a class."
                    )

        if isinstance(cls, UnionType):
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

