import config


def get_movement_command(offset_x):
    """
    Translate horizontal pixel offset from frame center into a steering command.

    Returns: "CENTERED", "TURN LEFT", or "TURN RIGHT"
    """
    if abs(offset_x) < config.CENTER_DEAD_ZONE:
        return "CENTERED"
    elif offset_x > config.CENTER_DEAD_ZONE:
        return "TURN RIGHT"
    else:
        return "TURN LEFT"
