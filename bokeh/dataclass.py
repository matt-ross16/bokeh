#-----------------------------------------------------------------------------
# Copyright (c) 2012 - 2020, Anaconda, Inc., and Bokeh Contributors.
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
#-----------------------------------------------------------------------------
"""
Copyright (C) 2020 biqqles.
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

#-----------------------------------------------------------------------------
# Boilerplate
#-----------------------------------------------------------------------------

import logging # isort:skip
log = logging.getLogger(__name__)

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Standard library imports
from types import FunctionType
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

#-----------------------------------------------------------------------------
# Globals and constants
#-----------------------------------------------------------------------------

__all__ = (
    "as_dict",
    "as_tuple",
    "dataclass",
    "fields",
    "is_dataclass",
)

DataClass = Any
T = TypeVar('T')

#-----------------------------------------------------------------------------
# General API
#-----------------------------------------------------------------------------

def dataclass(cls: Optional[type] = None, *, meta=None, **options):
    """The decorator used to apply DataClassMeta, or optionally a subclass of that metaclass, to a class."""
    if meta is None:
        meta = DataClassMeta
    else:
        assert issubclass(meta, DataClassMeta)

    def apply_metaclass(to_class, metaclass=meta):
        """Apply a metaclass to a class."""
        namespace = dict(vars(to_class), __metaclass__=metaclass)
        return metaclass(to_class.__name__, to_class.__bases__, namespace, **options)

    if cls is not None:  # if decorator used with no arguments, apply metaclass to the class immediately
        if not isinstance(cls, type):
            raise TypeError("This decorator takes no explicit positional arguments")
        return apply_metaclass(cls)
    return apply_metaclass  # otherwise, return function for later evaluation

def is_dataclass(obj: Any) -> bool:
    """Return True if the given object is a data class as implemented in this package, otherwise False."""
    return getattr(obj, "__metaclass__", None) is DataClassMeta

def is_dataclass_instance(obj: Any) -> bool:
    """Return True if the given object is an instance of a data class, otherwise False."""
    return is_dataclass(obj) and type(obj) is not DataClassMeta

def fields(dataclass: DataClass, internals: bool = False) -> Dict[str, Any]:
    """Return a dict of `dataclass`"s fields and their values. `internals` selects whether to include internal fields.
    A field is defined as a class-level variable with a type annotation."""
    assert is_dataclass_instance(dataclass)
    return {f: getattr(dataclass, f) for f in _filter_annotations(dataclass.__annotations__, internals)}

def as_dict(dataclass: DataClass, dict_factory=dict) -> Dict[str, Any]:
    """Recursively create a dict of a dataclass instance"s fields and their values.
    This function is recursively called on data classes, named tuples and iterables."""
    assert is_dataclass_instance(dataclass)
    return _recurse_structure(dataclass, dict_factory)

def as_tuple(dataclass: DataClass) -> Tuple:
    """Recursively create a tuple of the values of a dataclass instance"s fields, in definition order.
    This function is recursively called on data classes, named tuples and iterables."""
    assert is_dataclass_instance(dataclass)
    return _recurse_structure(dataclass, lambda k_v: tuple(v for k, v in k_v))

def replace(dataclass: DataClass, **changes) -> DataClass:
    """Return a new copy of `dataclass` with field values replaced as specified in `changes`."""
    return type(dataclass)(**dict(fields(dataclass, internals=True), **changes))

#-----------------------------------------------------------------------------
# Dev API
#-----------------------------------------------------------------------------

class DataClassMeta(type):
    """The metaclass for a data class."""

    _DEFAULT_OPTIONS = dict(
      init=True,
      repr=True,
      eq=True,
      iter=False,
      frozen=False,
      kwargs=False,
      slots=False,
      hide_internals=True,
    )

    def __new__(cls, name, bases, namespace, **kwargs):
        # collect annotations, defaults, slots and options from this class" ancestors, in definition order
        all_annotations = {}
        all_defaults = {}
        all_slots = set()
        options = dict(cls._DEFAULT_OPTIONS)

        dataclass_bases = [vars(b) for b in bases if hasattr(b, "__dataclass__")]
        for b in dataclass_bases + [namespace]:
            all_annotations.update(b.get("__annotations__", {}))
            all_defaults.update(b.get("__defaults__", namespace))
            all_slots.update(b.get("__slots__", set()))
            options.update(b.get("__dataclass__", {}))
        options.update(kwargs)

        # fill out this class" dict and store defaults, annotations and decorator options for future subclasses
        namespace.update(all_defaults)
        namespace["__defaults__"] = all_defaults
        namespace["__annotations__"] = all_annotations
        namespace["__dataclass__"] = options

        # delete what will become stale references so that Python creates new ones
        del namespace["__dict__"], namespace["__weakref__"]

        # warn the user if they try to use __post_init__
        if "__post_init__" in namespace:
            raise TypeError("dataclassy does not use __post_init__. You should rename this method __init__")

        # create/apply generated methods and attributes
        if options["slots"]:
            # values with default values must only be present in slots, not dict, otherwise Python will interpret them
            # as read only
            for d in all_annotations.keys() & all_defaults.keys():
                del namespace[d]
            namespace["__slots__"] = all_annotations.keys() - all_slots
        elif "__slots__" in namespace:
            # if the slots option has been removed from an inheriting dataclass we must remove descriptors and __slots__
            for d in all_annotations.keys() - all_defaults.keys() & namespace.keys():
                del namespace[d]
            del namespace["__slots__"]
        if options["init"]:
            namespace.setdefault("__new__", _generate_new(all_annotations, all_defaults,
                                                      options["kwargs"], options["frozen"]))
        if options["repr"]:
            namespace.setdefault("__repr__", __repr__)
        if options["eq"]:
            namespace.setdefault("__eq__", __eq__)
        if options["iter"]:
            namespace.setdefault("__iter__", __iter__)
        if options["frozen"]:
            namespace["__delattr__"] = namespace["__setattr__"] = __setattr__

        return type.__new__(cls, name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        """Remove arguments used by __new__ before calling __init__."""
        instance = cls.__new__(cls, *args, **kwargs)

        args = args[cls.__new__.__code__.co_argcount - 1:]  # -1 for "cls"
        for parameter in kwargs.keys() & cls.__annotations__.keys():
            del kwargs[parameter]

        instance.__init__(*args, **kwargs)
        return instance

    @property
    def __signature__(cls):
        """Defining a __call__ breaks inspect.signature. Lazily generate a Signature object ourselves."""
        import inspect
        parameters = tuple(inspect.signature(cls.__new__).parameters.values())
        return inspect.Signature(parameters[1:])  # remove "cls" to transform parameters of __new__ into those of class

#-----------------------------------------------------------------------------
# Private API
#-----------------------------------------------------------------------------

def _generate_new(annotations: Dict[str, Any], defaults: Dict[str, Any], gen_kwargs: bool, frozen: bool) -> FunctionType:
    """Generate and return a __new__ method for a data class which has as parameters all fields of the data class.
    When the data class is initialised, arguments to this function are applied to the fields of the new instance. Using
    __new__ frees up __init__, allowing it to be defined by the user to perform additional, custom initialisation."""
    user_init = "__init__" in defaults

    # determine arguments for initialiser
    arguments = [a for a in annotations if a not in defaults]
    default_arguments = [f"{a}={a}" for a in annotations if a in defaults]
    args = ["*args"] if user_init else []  # if init is defined, new"s arguments must be kw-only to avoid ambiguity
    kwargs = ["**kwargs"] if gen_kwargs or user_init else []

    parameters = ", ".join(arguments + args + default_arguments + kwargs)

    # determine what to do with arguments before assignment. If the argument matches a mutable default, make a copy
    references = {n: f"{n}.copy() if {n} is self.__defaults__[{n!r}] else {n}"
                  if n in defaults and hasattr(defaults[n], "copy") else n for n in annotations}

    # if the class is frozen, use the necessary but slightly slower object.__setattr__
    assignments = [f"object.__setattr__(self, {n!r}, {r})" if frozen else f"self.{n} = {r}"
                   for n, r in references.items()]

    # generate the function
    signature = f"def __new__(cls, {parameters}):"
    body = ["self = object.__new__(cls)", *assignments, "return self"]

    exec("\n\t".join([signature, *body]), {}, defaults)
    function = defaults.pop("__new__")
    function.__annotations__ = annotations
    return function

def __eq__(self: DataClass, other: DataClass):
    return type(self) is type(other) and as_tuple(self) == as_tuple(other)

def __iter__(self: DataClass):
    return iter(as_tuple(self))

def __repr__(self):
    show_internals = not self.__dataclass__["hide_internals"]
    args = ", ".join(f"{f}={v!r}" for f, v in fields(self, show_internals).items())
    return f"{type(self).__name__}({args})"

def __setattr__(self: DataClass, *args):
    raise AttributeError("frozen class")

def _filter_annotations(annotations: Dict[str, Any], internals: bool) -> Dict[str, Any]:
    """Filter an annotations dict for to remove or keep internal fields."""
    return annotations if internals else {f: a for f, a in annotations.items()
                                          if not f.startswith("_") and not Internal.is_internal(a)}

def _recurse_structure(var: Any, iter_proc: Callable) -> Any:
    """Recursively convert an arbitrarily nested structure beginning at `var`, copying and processing any iterables
    encountered with `iter_proc`."""
    if is_dataclass(var):
        var = fields(var, internals=True)
    if hasattr(var, "_asdict"):  # handle named tuples
        # noinspection PyCallingNonCallable, PyProtectedMember
        var = var._asdict()
    if isinstance(var, dict):
        return iter_proc((_recurse_structure(k, iter_proc), _recurse_structure(v, iter_proc)) for k, v in var.items())
    if isinstance(var, (list, tuple)):
        return type(var)(_recurse_structure(e, iter_proc) for e in var)
    return var

class Internal(Generic[T]):
    """ This type hint wrapper represents a field that is internal to the data class and is,
        for example, not to be shown in a repr.
    """
    @classmethod
    def is_internal(cls, annotation):
        return getattr(annotation, '__origin__', None) is cls or \
               (type(annotation) is str and cls.__name__ in annotation)

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------
