#-----------------------------------------------------------------------------
# Copyright (c) 2012 - 2020, Anaconda, Inc., and Bokeh Contributors.
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Boilerplate
#-----------------------------------------------------------------------------
import logging # isort:skip
log = logging.getLogger(__name__)

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Standard library imports
import collections
import datetime as dt
import decimal
import json
from typing import Any, Dict, List, Literal, Set

# External imports
import numpy as np

# Bokeh imports
from ..dataclass import DataClass, fields, valueclass, is_dataclass, is_valueclass
from ..settings import settings
from ..util.dependencies import import_optional
from ..util.serialization import (
    convert_datetime_type,
    convert_timedelta_type,
    is_datetime_type,
    is_timedelta_type,
    make_id,
    transform_array,
    transform_series,
)
from ..util.version import __version__

#-----------------------------------------------------------------------------
# Globals and constants
#-----------------------------------------------------------------------------

__all__ = (
    'Serializer',
)

rd = import_optional("dateutil.relativedelta")
pd = import_optional("pandas")

_pos_infinity = float("+inf")
_neg_infinity = float("-inf")

#-----------------------------------------------------------------------------
# General API
#-----------------------------------------------------------------------------

@valueclass
class Ref:
    id: str

ByteOrder = Literal["little", "big"]
DataType = Literal["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"]

@valueclass
class NDArrayDef:
    buffer: bytes
    shape: List[int]
    dtype: DataType
    order: ByteOrder

class Serializer:

    def __init__(self):
        self._refs: Dict[Any, Ref] = {}
        self._defs: Dict[Any, Any] = {}
        self._bufs = {}
        self._objs: Set[Any] = set()

    def get_ref(self, obj: Any) -> Ref:
        return self._refs.get(obj, None)

    def add_ref(self, obj: Any, obj_ref: Optional[Ref] = None):
        if obj not in self._refs:
            if obj_ref is None:
                obj_ref = Ref(make_id())
            self._refs[obj] = obj_ref
            return obj_ref
        else:
            raise ValueError("object already known")

    def add_def(self, obj: Any, obj_def: Any) -> None:
        if obj in self._refs:
            self._defs[obj] = obj_def
        else:
            raise ValueError("object not known")

    def add_buf(self, obj: bytes) -> Ref:
        pass

    def encode(self, obj: Any) -> Any:
        if obj is None:
            return obj
        if obj is True:
            return obj
        if obj is False:
            return obj
        if isinstance(obj, str):
            return obj
        if isinstance(obj, int):
            return obj
        if isinstance(obj, float):
            return self._encode_float(obj)
        if isinstance(obj, (list, tuple, set)):
            return self._encode_list(obj)
        if isinstance(obj, dict):
            return self._encode_dict(obj)
        if isinstance(obj, bytes):
            ref = self.get_ref(obj)
            if ref is None:
                ref = self.add_buf(obj)
            return ref

        if is_valueclass(obj):
            return self._encode_valueclass(obj)
        if is_dataclass(obj):
            return self._encode_dataclass(obj)

        from ..model import Model
        if isinstance(obj, Model):
            return self._encode_Model(obj)

        #from .has_props import HasProps
        #if isinstance(obj, HasProps):
        #    return self._encode_HasProps(obj)

        from ..colors import Color
        if isinstance(obj, Color):
            return self._encode_Color(obj)

        if pd and isinstance(obj, (pd.Series, pd.Index)):
            return self._encode_series(obj)
        if isinstance(obj, np.ndarray):
            return self._encode_ndarray(obj)

        if isinstance(obj, collections.deque):
            return self._encode_deque(obj)

        # date/time values that get serialized as milliseconds
        if is_datetime_type(obj):
            return self._encode_datetime(obj)

        if is_timedelta_type(obj):
            return self._encode_timedelta(obj)

        # Date
        if isinstance(obj, dt.date):
            return self.encode(obj.isoformat())

        # slice objects
        if isinstance(obj, slice):
            return self._encode_slice(obj)

        # NumPy scalars
        if np.issubdtype(type(obj), np.floating):
            return self.encode(float(obj))
        if np.issubdtype(type(obj), np.integer):
            return self.encode(int(obj))
        if np.issubdtype(type(obj), np.bool_):
            return self.encode(bool(obj))

        # Decimal values
        if isinstance(obj, decimal.Decimal):
            return self.encode(float(obj)) # TODO: this is incorrect

        # RelativeDelta gets serialized as a dict
        if rd and isinstance(obj, rd.relativedelta):
            return dict(
                years=self.encode(obj.years),
                months=self.encode(obj.months),
                days=self.encode(obj.days),
                hours=self.encode(obj.hours),
                minutes=self.encode(obj.minutes),
                seconds=self.encode(obj.seconds),
                microseconds=self.encode(obj.microseconds),
            )

        # TODO: display XPath-style selector pointing to the offending object in the graph
        raise ValueError(f"object of type '{type(obj)}' is not serializable")

    def _encode_float(self, obj: float):
        if obj != obj:
            return {"$": "NaN"}
        elif obj == _pos_infinity:
            return {"$": "Infinity"}
        elif obj == _neg_infinity:
            return {"$": "-Infinity"}
        else:
            return obj

    def _encode_list(self, obj: List[Any]):
        return [ self.encode(item) for item in obj ]

    def _encode_dict(self, obj: Dict[Any, Any]):
        items = []
        plain = True
        for key, val in obj.items():
            items.push((self.encode(key), self.encode(val)))
            if not isinstance(key, str):
                plain = False
        if plain:
            return dict(items)
        else:
            return {"$": "Map", "items": itmes}

    def _encode_slice(self, obj: slice):
        return dict(
            start=self.encode(obj.start),
            stop=self.encode(obj.stop),
            step=self.encode(obj.step),
        )

    def _encode_valueclass(self, obj: DataClass):
        if obj not in self._objs:
            self._objs.add(obj)
            return { self.encode(key): self.encode(val) for key, val in fields(obj).items() }
        else:
            raise ValueError("circular object")

    def _encode_dataclass(self, obj: DataClass):
        ref = self.get_ref(obj)
        if ref is None:
            ref = self.add_ref(obj)
            struct = dict(
                name=obj.__name__,
                module=obj.__module__,
                fields=self._encode_valueclass(obj),
            )
            self.add_def(obj, struct)
        return ref

    def _encode_Document(self, obj): #: Document):
        @valueclass
        class DocStruct:
            roots: List[Model]
            title: str
            version: str

        ref = self.get_ref(obj)
        if ref is None:
            ref = self.add_ref(obj)
            struct = DocStruct(doc.roots, doc.title, __version__)
            self.add_def(obj, struct)

        return ref

    def _encode_Model(self, obj): #: Model):
        ref = self.get_ref(obj)
        if ref is None:
            ref = self.add_ref(obj, obj.ref)
            struct = obj.struct
            model_ref = self._encode_ModelType(obj.__class__)
            if model_ref is not None:
                struct["model"] = model_ref
            struct["attributes"] = obj._to_json_like(include_defaults=False)
            self.add_def(obj, struct)
        return ref

    def _encode_ModelType(self, obj): #: Type[Model]):
        from .has_props import is_DataModel
        if is_DataModel(obj):
            ref = self.get_ref(obj)
            if ref is not None:
                ref = self.add_ref(obj)
                #self.add_def(obj, ???)
            return ref
        else:
            return None

    #def _encode_HasProps(self, obj): #: HasProps):
    #    return obj.properties_with_values(include_defaults=False)

    def _encode_Color(self, obj): #: Color):
        return obj.to_css()

    def _encode_series(self, obj):
        if isinstance(series, pd.PeriodIndex):
            values = obj.to_timestamp().values
        else:
            values = obj.values
        return self._encode_ndarray(values)

    def _encode_ndarray(self, obj: np.ndarray):
        if isinstance(obj, np.ma.MaskedArray):
            obj = obj.filled(np.nan)
        if not obj.flags["C_CONTIGUOUS"]:
            obj = np.ascontiguousarray(obj)

        ref = self.get_ref(obj)
        if ref is None:
            ref = self.add_ref(obj)
            struct = self.encode(NDArrayDef(
                buffer=array.tobytes(),
                shape=array.shape,
                dtype=array.dtype.name,
                order=sys.byteorder,
            ))
            self.add_ref(obj, ref, struct)
            #__buffer__ = buffer_id
            #__ndarray__ = base64.b64encode(array.data).decode('utf-8')

        return ref

    def _encode_deque(self, obj: collections.deque):
        # TODO: encode this as a proper object
        return [ self.encode(item) for item in obj ]

    def _encode_datetime(self, obj):
        return convert_datetime_type(obj)

    def _encode_timedelta(self, obj):
        return convert_timedelta_type(obj)
