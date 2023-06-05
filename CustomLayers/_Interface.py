class Interface(object):
    def __int__(self, *args):
        raise NotImplementedError

    def to_dict(self):
        return {a: getattr(self, a)
                for a in sorted(dir(self))
                if not a.startswith("__") and not callable(getattr(self, a))}
