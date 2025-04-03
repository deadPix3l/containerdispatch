from functools import _compose_mro
from complexdispatch.utils import _pep585_registry_matches

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

    from typing import get_args

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
