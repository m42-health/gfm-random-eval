"""Configuration class for the NT benchmark."""


class Config:  # noqa
    def __init__(self: "Config", **kwargs: object) -> None:  # noqa
        for key, value in kwargs.items():
            setattr(self, key, value)
