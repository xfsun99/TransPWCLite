from .Transpwclite import TransPWCLite

def get_model(cfg):
    if cfg.type == 'Transpwclite':
        model = TransPWCLite(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return model
