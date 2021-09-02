import os


def get_config_file(config_path):
    if not os.path.exists(os.path.join("config", config_path)):
        raise RuntimeError("{} not available in config!".format(config_path))

    return os.path.join("config", config_path)


def get_model_weights(model_wheights):
    if not os.path.exists(os.path.join("weights", model_wheights)):
        raise RuntimeError("{} not available in weights!".format(model_wheights))

    return os.path.join("weights", model_wheights)
