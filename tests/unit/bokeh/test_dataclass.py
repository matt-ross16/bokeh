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

import pytest ; pytest

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Standard library imports
from collections import namedtuple
from inspect import signature
from sys import getsizeof
from typing import Dict, List, Tuple

# Module under test
from bokeh.dataclass import as_dict, as_tuple, dataclass, fields, replace, Internal # isort:skip

#-----------------------------------------------------------------------------
# Setup
#-----------------------------------------------------------------------------

@dataclass(slots=True)
class Alpha:
    a: int
    b: int = 2
    c: int

class Beta(Alpha):
    d: int = 4
    e: Internal[Dict[int, str]] = {}
    f: int

@dataclass(slots=False, iter=True)  # test option inheritance and overriding
class Gamma(Beta):
    pass

@dataclass  # same fields as Beta but without inheritance or slots
class Delta:
    a: int
    b: int = 2
    c: int
    d: int = 4
    e: Internal[Dict[int, str]] = {}
    f: str

NT = namedtuple("NT", "x y z")

@dataclass  # a complex (nested) class
class Epsilon:
    g: Tuple[NT]
    h: List["Epsilon"]
    _i: int = 0

b = Beta(1, 2, 3)
e = Epsilon((NT(1, 2, 3)), [Epsilon(4, 5, 6)])

#-----------------------------------------------------------------------------
# General API
#-----------------------------------------------------------------------------

def test_decorator_options() -> None:
    """Test decorator options are inherited and overridden correctly."""
    assert Beta.__dataclass__["slots"]
    assert not Delta.__dataclass__["slots"]

def test_invalid_decorator_use() -> None:
    """Test invalid use of the decorator."""
    with pytest.raises(TypeError):
        dataclass(1)

    with pytest.raises(AssertionError):
        @dataclass(meta=int)
        class Dummy:
            pass

def test_readme() -> None:
    """Test all examples from the project readme."""
    @dataclass  # with default parameters
    class Pet:
        name: str
        age: int
        species: str
        foods: List[str] = []
        SOME_CONSTANT = 232

    assert str(signature(Pet)) == "(name: str, age: int, species: str, foods: List[str] = [])"

def test_init() -> None:
    """Test correct generation of an __init__ method."""
    assert str(signature(Beta)) == "(a: int, c: int, f: int, b: int = 2, d: int = 4, e: bokeh.dataclass.Internal[typing.Dict[int, str]] = {})"

def test_repr() -> None:
    """Test correct generation of a __repr__ method."""
    assert repr(b) == "Beta(a=1, b=2, c=2, d=4, f=3)"

def test_iter() -> None:
    """Test correct generation of an __iter__ method."""
    iterable = Gamma(0, 1, [2, 3])
    a, b, *_, f = iterable
    assert a == 0
    assert b == 2
    assert f == [2, 3]

def test_eq() -> None:
    """Test correct generation of an __eq__ method."""
    assert b == b
    unequal_b = Beta(10, 20, 30)
    assert b != unequal_b
    assert b != [0]  # test comparisons with non-dataclasses

def test_slots() -> None:
    """Test correct generation and efficacy of a __slots__ attribute."""
    assert hasattr(b, "__slots__")
    assert not hasattr(b, "__dict__")
    e = Epsilon(1, 2, 3)
    assert getsizeof(e) + getsizeof(e.__dict__) > getsizeof(b)

def test_frozen() -> None:
    """Test correct generation of __setattr__ and __delattr__ for a frozen class."""
    @dataclass(frozen=True)
    class Frozen:
        a: int
        b: int

    f = Frozen(1, 2)
    with pytest.raises(AttributeError):
        f.a = 3
    with pytest.raises(AttributeError):
        del f.b

def test_empty_dataclass() -> None:
    """Test data classes with no fields and data classes with only class fields."""
    @dataclass
    class Empty:
        pass

    @dataclass(kwargs=False)
    class ClassVarOnly:
        class_var = 0

    assert str(signature(ClassVarOnly)) == "()"

def test_mutable_defaults() -> None:
    """Test mutable defaults are copied and not mutated between instances."""
    @dataclass
    class MutableDefault:
        mutable: List[int] = []

    a = MutableDefault()
    a.mutable.append(2)
    b = MutableDefault()
    b.mutable.append(3)
    c = MutableDefault(4)  # incorrect types should still be OK (shouldn"t try to call copy)
    assert a.mutable == [2]
    assert b.mutable == [3]
    assert c.mutable == 4

def test_custom_init() -> None:
    """Test user-defined __init__ used for post-initialisation logic."""
    @dataclass
    class CustomInit:
        a: int
        b: int

        def __init__(self, c):
            self.d = (self.a + self.b) / c

    custom = CustomInit(1, 2, 3)
    assert custom.d == 1.0

    @dataclass
    class CustomInitKwargs:
        a: int
        b: int

        def __init__(self, *args, **kwargs):
            self.c = kwargs

    custom_kwargs = CustomInitKwargs(1, 2, c=3)
    assert custom_kwargs.c == {"c": 3}

def test_fields() -> None:
    """Test fields()."""
    assert repr(fields(e)) == """{'g': NT(x=1, y=2, z=3), 'h': [Epsilon(g=4, h=5)]}"""
    assert repr(fields(e, True)) == """{'g': NT(x=1, y=2, z=3), 'h': [Epsilon(g=4, h=5)], '_i': 0}"""

def test_as_tuple() -> None:
    """Test as_tuple()."""
    assert as_tuple(e) == ((1, 2, 3), [(4, 5, 6)], 0)

def test_as_dict() -> None:
    """Test as_dict()."""
    assert as_dict(e) == {"g": {"x": 1, "y": 2, "z": 3}, "h": [{"g": 4, "h": 5, "_i": 6}], "_i": 0}

#def test_make_dataclass() -> None:
#    """Test functional creation of a data class using make_dataclass."""
#    dynamic = make_dataclass("Dynamic", dict(name=str), {})
#    dynamic(name="Dynamic")

def test_replace() -> None:
    """Test replace()."""
    assert replace(b, f=4) == Beta(1, 2, 4)
    assert b == Beta(1, 2, 3)  # assert that the original instance remains unchanged

def test_post_init_warning() -> None:
    """Test that the user is warned if a __post_init__ is defined."""
    with pytest.raises(TypeError):
        @dataclass
        class Deprecated:
            def __post_init__(self):
                pass

#-----------------------------------------------------------------------------
# Dev API
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Private API
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------
