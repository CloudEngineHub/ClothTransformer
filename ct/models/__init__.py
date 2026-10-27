"""Model registry.

To add an architecture, create a module in this package, decorate the class with
``@register("name")``, import the module below, and select it with
``model.name=name``. The class receives the ``model`` config section and must
implement the same ``forward`` signature as ``ClothTransformer``.
"""
MODELS = {}


def register(name):
    def deco(cls):
        if name in MODELS:
            raise ValueError(f"Model '{name}' already registered (existing: {MODELS[name]}).")
        MODELS[name] = cls
        return cls
    return deco


def build_model(cfg):
    if cfg.name not in MODELS:
        raise KeyError(f"Unknown model '{cfg.name}'. Registered: {sorted(MODELS)}")
    return MODELS[cfg.name](cfg)


from . import clothtransformer  # noqa: F401,E402
