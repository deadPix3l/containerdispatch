def _pep585_registry_matches(cls, registry):
    from typing import get_origin
    return (i for i in registry.keys() if get_origin(i) == cls)

