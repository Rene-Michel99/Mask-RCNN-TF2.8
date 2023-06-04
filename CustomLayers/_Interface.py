class Interface(object):
    def __int__(self, config):
        raise NotImplementedError

    def to_dict(self):
        return {a: getattr(self, a)
                for a in sorted(dir(self))
                if not a.startswith("__") and not callable(getattr(self, a))}
