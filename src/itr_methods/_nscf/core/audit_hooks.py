"""Read-only observers for the isolated pilot copy. Never choose/update a model."""
observer = None


def call(name, *args, **kwargs):
    if observer is not None:
        return getattr(observer, name)(*args, **kwargs)
    return None
