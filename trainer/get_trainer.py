from . import pre_trainer, ft_trainer


def get_trainer(name):
    if name == 'Sintel':
        TrainFramework = pre_trainer.TrainFramework
    elif name == 'Ultrasound':
        TrainFramework = ft_trainer.TrainFramework
    else:
        raise NotImplementedError(name)

    return TrainFramework
