from inspect import isabstract


def child_subclasses(base):
    """
    Return all non-abstract subclasses of a base class.

    Parameters
    ----------
    base : class
        high level class object that is inherited by the
        desired subclasses

    Returns
    -------
    children : list
        list of non-abstract subclasses

    """
    family = base.__subclasses__() + [
        g for s in base.__subclasses__()
        for g in child_subclasses(s)
    ]
    children = [g for g in family if not isabstract(g)]

    unique = []
    unique_names = []
    for c in children:
        if c.__name__ not in unique_names:
            unique.append(c)
            unique_names.append(c.__name__)

    return unique
