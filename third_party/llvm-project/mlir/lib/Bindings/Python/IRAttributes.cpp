//===- IRAttributes.cpp - Exports builtin and standard attributes ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "IRModule.h"

#include "PybindUtils.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"

namespace py = pybind11;
using namespace mlir;
using namespace mlir::python;

using llvm::Optional;
using llvm::SmallVector;
using llvm::Twine;

//------------------------------------------------------------------------------
// Docstrings (trivial, non-duplicated docstrings are included inline).
//------------------------------------------------------------------------------

static const char kDenseElementsAttrGetDocstring[] =
    R"(Gets a DenseElementsAttr from a Python buffer or array.

When `type` is not provided, then some limited type inferencing is done based
on the buffer format. Support presently exists for 8/16/32/64 signed and
unsigned integers and float16/float32/float64. DenseElementsAttrs of these
types can also be converted back to a corresponding buffer.

For conversions outside of these types, a `type=` must be explicitly provided
and the buffer contents must be bit-castable to the MLIR internal
representation:

  * Integer types (except for i1): the buffer must be byte aligned to the
    next byte boundary.
  * Floating point types: Must be bit-castable to the given floating point
    size.
  * i1 (bool): Bit packed into 8bit words where the bit pattern matches a
    row major ordering. An arbitrary Numpy `bool_` array can be bit packed to
    this specification with: `np.packbits(ary, axis=None, bitorder='little')`.

If a single element buffer is passed (or for i1, a single byte with value 0
or 255), then a splat will be created.

Args:
  array: The array or buffer to convert.
  signless: If inferring an appropriate MLIR type, use signless types for
    integers (defaults True).
  type: Skips inference of the MLIR element type and uses this instead. The
    storage size must be consistent with the actual contents of the buffer.
  shape: Overrides the shape of the buffer when constructing the MLIR
    shaped type. This is needed when the physical and logical shape differ (as
    for i1).
  context: Explicit context, if not from context manager.

Returns:
  DenseElementsAttr on success.

Raises:
  ValueError: If the type of the buffer or array cannot be matched to an MLIR
    type or if the buffer does not meet expectations.
)";

namespace {

static MlirStringRef toMlirStringRef(const std::string &s) {
  return mlirStringRefCreate(s.data(), s.size());
}

class PyAffineMapAttribute : public PyConcreteAttribute<PyAffineMapAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAAffineMap;
  static constexpr const char *pyClassName = "AffineMapAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyAffineMap &affineMap) {
          MlirAttribute attr = mlirAffineMapAttrGet(affineMap.get());
          return PyAffineMapAttribute(affineMap.getContext(), attr);
        },
        py::arg("affine_map"), "Gets an attribute wrapping an AffineMap.");
  }
};

template <typename T>
static T pyTryCast(py::handle object) {
  try {
    return object.cast<T>();
  } catch (py::cast_error &err) {
    std::string msg =
        std::string(
            "Invalid attribute when attempting to create an ArrayAttribute (") +
        err.what() + ")";
    throw py::cast_error(msg);
  } catch (py::reference_cast_error &err) {
    std::string msg = std::string("Invalid attribute (None?) when attempting "
                                  "to create an ArrayAttribute (") +
                      err.what() + ")";
    throw py::cast_error(msg);
  }
}

class PyArrayAttribute : public PyConcreteAttribute<PyArrayAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAArray;
  static constexpr const char *pyClassName = "ArrayAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  class PyArrayAttributeIterator {
  public:
    PyArrayAttributeIterator(PyAttribute attr) : attr(std::move(attr)) {}

    PyArrayAttributeIterator &dunderIter() { return *this; }

    PyAttribute dunderNext() {
      if (nextIndex >= mlirArrayAttrGetNumElements(attr.get())) {
        throw py::stop_iteration();
      }
      return PyAttribute(attr.getContext(),
                         mlirArrayAttrGetElement(attr.get(), nextIndex++));
    }

    static void bind(py::module &m) {
      py::class_<PyArrayAttributeIterator>(m, "ArrayAttributeIterator",
                                           py::module_local())
          .def("__iter__", &PyArrayAttributeIterator::dunderIter)
          .def("__next__", &PyArrayAttributeIterator::dunderNext);
    }

  private:
    PyAttribute attr;
    int nextIndex = 0;
  };

  PyAttribute getItem(intptr_t i) {
    return PyAttribute(getContext(), mlirArrayAttrGetElement(*this, i));
  }

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](py::list attributes, DefaultingPyMlirContext context) {
          SmallVector<MlirAttribute> mlirAttributes;
          mlirAttributes.reserve(py::len(attributes));
          for (auto attribute : attributes) {
            mlirAttributes.push_back(pyTryCast<PyAttribute>(attribute));
          }
          MlirAttribute attr = mlirArrayAttrGet(
              context->get(), mlirAttributes.size(), mlirAttributes.data());
          return PyArrayAttribute(context->getRef(), attr);
        },
        py::arg("attributes"), py::arg("context") = py::none(),
        "Gets a uniqued Array attribute");
    c.def("__getitem__",
          [](PyArrayAttribute &arr, intptr_t i) {
            if (i >= mlirArrayAttrGetNumElements(arr))
              throw py::index_error("ArrayAttribute index out of range");
            return arr.getItem(i);
          })
        .def("__len__",
             [](const PyArrayAttribute &arr) {
               return mlirArrayAttrGetNumElements(arr);
             })
        .def("__iter__", [](const PyArrayAttribute &arr) {
          return PyArrayAttributeIterator(arr);
        });
    c.def("__add__", [](PyArrayAttribute arr, py::list extras) {
      std::vector<MlirAttribute> attributes;
      intptr_t numOldElements = mlirArrayAttrGetNumElements(arr);
      attributes.reserve(numOldElements + py::len(extras));
      for (intptr_t i = 0; i < numOldElements; ++i)
        attributes.push_back(arr.getItem(i));
      for (py::handle attr : extras)
        attributes.push_back(pyTryCast<PyAttribute>(attr));
      MlirAttribute arrayAttr = mlirArrayAttrGet(
          arr.getContext()->get(), attributes.size(), attributes.data());
      return PyArrayAttribute(arr.getContext(), arrayAttr);
    });
  }
};

/// Float Point Attribute subclass - FloatAttr.
class PyFloatAttribute : public PyConcreteAttribute<PyFloatAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAFloat;
  static constexpr const char *pyClassName = "FloatAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &type, double value, DefaultingPyLocation loc) {
          MlirAttribute attr = mlirFloatAttrDoubleGetChecked(loc, type, value);
          // TODO: Rework error reporting once diagnostic engine is exposed
          // in C API.
          if (mlirAttributeIsNull(attr)) {
            throw SetPyError(PyExc_ValueError,
                             Twine("invalid '") +
                                 py::repr(py::cast(type)).cast<std::string>() +
                                 "' and expected floating point type.");
          }
          return PyFloatAttribute(type.getContext(), attr);
        },
        py::arg("type"), py::arg("value"), py::arg("loc") = py::none(),
        "Gets an uniqued float point attribute associated to a type");
    c.def_static(
        "get_f32",
        [](double value, DefaultingPyMlirContext context) {
          MlirAttribute attr = mlirFloatAttrDoubleGet(
              context->get(), mlirF32TypeGet(context->get()), value);
          return PyFloatAttribute(context->getRef(), attr);
        },
        py::arg("value"), py::arg("context") = py::none(),
        "Gets an uniqued float point attribute associated to a f32 type");
    c.def_static(
        "get_f64",
        [](double value, DefaultingPyMlirContext context) {
          MlirAttribute attr = mlirFloatAttrDoubleGet(
              context->get(), mlirF64TypeGet(context->get()), value);
          return PyFloatAttribute(context->getRef(), attr);
        },
        py::arg("value"), py::arg("context") = py::none(),
        "Gets an uniqued float point attribute associated to a f64 type");
    c.def_property_readonly(
        "value",
        [](PyFloatAttribute &self) {
          return mlirFloatAttrGetValueDouble(self);
        },
        "Returns the value of the float point attribute");
  }
};

/// Integer Attribute subclass - IntegerAttr.
class PyIntegerAttribute : public PyConcreteAttribute<PyIntegerAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAInteger;
  static constexpr const char *pyClassName = "IntegerAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &type, int64_t value) {
          MlirAttribute attr = mlirIntegerAttrGet(type, value);
          return PyIntegerAttribute(type.getContext(), attr);
        },
        py::arg("type"), py::arg("value"),
        "Gets an uniqued integer attribute associated to a type");
    c.def_property_readonly(
        "value",
        [](PyIntegerAttribute &self) {
          return mlirIntegerAttrGetValueInt(self);
        },
        "Returns the value of the integer attribute");
  }
};

/// Bool Attribute subclass - BoolAttr.
class PyBoolAttribute : public PyConcreteAttribute<PyBoolAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsABool;
  static constexpr const char *pyClassName = "BoolAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](bool value, DefaultingPyMlirContext context) {
          MlirAttribute attr = mlirBoolAttrGet(context->get(), value);
          return PyBoolAttribute(context->getRef(), attr);
        },
        py::arg("value"), py::arg("context") = py::none(),
        "Gets an uniqued bool attribute");
    c.def_property_readonly(
        "value",
        [](PyBoolAttribute &self) { return mlirBoolAttrGetValue(self); },
        "Returns the value of the bool attribute");
  }
};

class PyFlatSymbolRefAttribute
    : public PyConcreteAttribute<PyFlatSymbolRefAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAFlatSymbolRef;
  static constexpr const char *pyClassName = "FlatSymbolRefAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::string value, DefaultingPyMlirContext context) {
          MlirAttribute attr =
              mlirFlatSymbolRefAttrGet(context->get(), toMlirStringRef(value));
          return PyFlatSymbolRefAttribute(context->getRef(), attr);
        },
        py::arg("value"), py::arg("context") = py::none(),
        "Gets a uniqued FlatSymbolRef attribute");
    c.def_property_readonly(
        "value",
        [](PyFlatSymbolRefAttribute &self) {
          MlirStringRef stringRef = mlirFlatSymbolRefAttrGetValue(self);
          return py::str(stringRef.data, stringRef.length);
        },
        "Returns the value of the FlatSymbolRef attribute as a string");
  }
};

class PyStringAttribute : public PyConcreteAttribute<PyStringAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAString;
  static constexpr const char *pyClassName = "StringAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::string value, DefaultingPyMlirContext context) {
          MlirAttribute attr =
              mlirStringAttrGet(context->get(), toMlirStringRef(value));
          return PyStringAttribute(context->getRef(), attr);
        },
        py::arg("value"), py::arg("context") = py::none(),
        "Gets a uniqued string attribute");
    c.def_static(
        "get_typed",
        [](PyType &type, std::string value) {
          MlirAttribute attr =
              mlirStringAttrTypedGet(type, toMlirStringRef(value));
          return PyStringAttribute(type.getContext(), attr);
        },
        py::arg("type"), py::arg("value"),
        "Gets a uniqued string attribute associated to a type");
    c.def_property_readonly(
        "value",
        [](PyStringAttribute &self) {
          MlirStringRef stringRef = mlirStringAttrGetValue(self);
          return py::str(stringRef.data, stringRef.length);
        },
        "Returns the value of the string attribute");
  }
};

// TODO: Support construction of string elements.
class PyDenseElementsAttribute
    : public PyConcreteAttribute<PyDenseElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseElements;
  static constexpr const char *pyClassName = "DenseElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static PyDenseElementsAttribute
  getFromBuffer(py::buffer array, bool signless, Optional<PyType> explicitType,
                Optional<std::vector<int64_t>> explicitShape,
                DefaultingPyMlirContext contextWrapper) {
    // Request a contiguous view. In exotic cases, this will cause a copy.
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
    Py_buffer *view = new Py_buffer();
    if (PyObject_GetBuffer(array.ptr(), view, flags) != 0) {
      delete view;
      throw py::error_already_set();
    }
    py::buffer_info arrayInfo(view);
    SmallVector<int64_t> shape;
    if (explicitShape) {
      shape.append(explicitShape->begin(), explicitShape->end());
    } else {
      shape.append(arrayInfo.shape.begin(),
                   arrayInfo.shape.begin() + arrayInfo.ndim);
    }

    MlirAttribute encodingAttr = mlirAttributeGetNull();
    MlirContext context = contextWrapper->get();

    // Detect format codes that are suitable for bulk loading. This includes
    // all byte aligned integer and floating point types up to 8 bytes.
    // Notably, this excludes, bool (which needs to be bit-packed) and
    // other exotics which do not have a direct representation in the buffer
    // protocol (i.e. complex, etc).
    Optional<MlirType> bulkLoadElementType;
    if (explicitType) {
      bulkLoadElementType = *explicitType;
    } else if (arrayInfo.format == "f") {
      // f32
      assert(arrayInfo.itemsize == 4 && "mismatched array itemsize");
      bulkLoadElementType = mlirF32TypeGet(context);
    } else if (arrayInfo.format == "d") {
      // f64
      assert(arrayInfo.itemsize == 8 && "mismatched array itemsize");
      bulkLoadElementType = mlirF64TypeGet(context);
    } else if (arrayInfo.format == "e") {
      // f16
      assert(arrayInfo.itemsize == 2 && "mismatched array itemsize");
      bulkLoadElementType = mlirF16TypeGet(context);
    } else if (isSignedIntegerFormat(arrayInfo.format)) {
      if (arrayInfo.itemsize == 4) {
        // i32
        bulkLoadElementType = signless ? mlirIntegerTypeGet(context, 32)
                                       : mlirIntegerTypeSignedGet(context, 32);
      } else if (arrayInfo.itemsize == 8) {
        // i64
        bulkLoadElementType = signless ? mlirIntegerTypeGet(context, 64)
                                       : mlirIntegerTypeSignedGet(context, 64);
      } else if (arrayInfo.itemsize == 1) {
        // i8
        bulkLoadElementType = signless ? mlirIntegerTypeGet(context, 8)
                                       : mlirIntegerTypeSignedGet(context, 8);
      } else if (arrayInfo.itemsize == 2) {
        // i16
        bulkLoadElementType = signless ? mlirIntegerTypeGet(context, 16)
                                       : mlirIntegerTypeSignedGet(context, 16);
      }
    } else if (isUnsignedIntegerFormat(arrayInfo.format)) {
      if (arrayInfo.itemsize == 4) {
        // unsigned i32
        bulkLoadElementType = signless
                                  ? mlirIntegerTypeGet(context, 32)
                                  : mlirIntegerTypeUnsignedGet(context, 32);
      } else if (arrayInfo.itemsize == 8) {
        // unsigned i64
        bulkLoadElementType = signless
                                  ? mlirIntegerTypeGet(context, 64)
                                  : mlirIntegerTypeUnsignedGet(context, 64);
      } else if (arrayInfo.itemsize == 1) {
        // i8
        bulkLoadElementType = signless ? mlirIntegerTypeGet(context, 8)
                                       : mlirIntegerTypeUnsignedGet(context, 8);
      } else if (arrayInfo.itemsize == 2) {
        // i16
        bulkLoadElementType = signless
                                  ? mlirIntegerTypeGet(context, 16)
                                  : mlirIntegerTypeUnsignedGet(context, 16);
      }
    }
    if (bulkLoadElementType) {
      auto shapedType = mlirRankedTensorTypeGet(
          shape.size(), shape.data(), *bulkLoadElementType, encodingAttr);
      size_t rawBufferSize = arrayInfo.size * arrayInfo.itemsize;
      MlirAttribute attr = mlirDenseElementsAttrRawBufferGet(
          shapedType, rawBufferSize, arrayInfo.ptr);
      if (mlirAttributeIsNull(attr)) {
        throw std::invalid_argument(
            "DenseElementsAttr could not be constructed from the given buffer. "
            "This may mean that the Python buffer layout does not match that "
            "MLIR expected layout and is a bug.");
      }
      return PyDenseElementsAttribute(contextWrapper->getRef(), attr);
    }

    throw std::invalid_argument(
        std::string("unimplemented array format conversion from format: ") +
        arrayInfo.format);
  }

  static PyDenseElementsAttribute getSplat(const PyType &shapedType,
                                           PyAttribute &elementAttr) {
    auto contextWrapper =
        PyMlirContext::forContext(mlirTypeGetContext(shapedType));
    if (!mlirAttributeIsAInteger(elementAttr) &&
        !mlirAttributeIsAFloat(elementAttr)) {
      std::string message = "Illegal element type for DenseElementsAttr: ";
      message.append(py::repr(py::cast(elementAttr)));
      throw SetPyError(PyExc_ValueError, message);
    }
    if (!mlirTypeIsAShaped(shapedType) ||
        !mlirShapedTypeHasStaticShape(shapedType)) {
      std::string message =
          "Expected a static ShapedType for the shaped_type parameter: ";
      message.append(py::repr(py::cast(shapedType)));
      throw SetPyError(PyExc_ValueError, message);
    }
    MlirType shapedElementType = mlirShapedTypeGetElementType(shapedType);
    MlirType attrType = mlirAttributeGetType(elementAttr);
    if (!mlirTypeEqual(shapedElementType, attrType)) {
      std::string message =
          "Shaped element type and attribute type must be equal: shaped=";
      message.append(py::repr(py::cast(shapedType)));
      message.append(", element=");
      message.append(py::repr(py::cast(elementAttr)));
      throw SetPyError(PyExc_ValueError, message);
    }

    MlirAttribute elements =
        mlirDenseElementsAttrSplatGet(shapedType, elementAttr);
    return PyDenseElementsAttribute(contextWrapper->getRef(), elements);
  }

  intptr_t dunderLen() { return mlirElementsAttrGetNumElements(*this); }

  py::buffer_info accessBuffer() {
    if (mlirDenseElementsAttrIsSplat(*this)) {
      // TODO: Currently crashes the program.
      // Reported as https://github.com/pybind/pybind11/issues/3336
      throw std::invalid_argument(
          "unsupported data type for conversion to Python buffer");
    }

    MlirType shapedType = mlirAttributeGetType(*this);
    MlirType elementType = mlirShapedTypeGetElementType(shapedType);
    std::string format;

    if (mlirTypeIsAF32(elementType)) {
      // f32
      return bufferInfo<float>(shapedType);
    }
    if (mlirTypeIsAF64(elementType)) {
      // f64
      return bufferInfo<double>(shapedType);
    }
    if (mlirTypeIsAF16(elementType)) {
      // f16
      return bufferInfo<uint16_t>(shapedType, "e");
    }
    if (mlirTypeIsAInteger(elementType) &&
        mlirIntegerTypeGetWidth(elementType) == 32) {
      if (mlirIntegerTypeIsSignless(elementType) ||
          mlirIntegerTypeIsSigned(elementType)) {
        // i32
        return bufferInfo<int32_t>(shapedType);
      }
      if (mlirIntegerTypeIsUnsigned(elementType)) {
        // unsigned i32
        return bufferInfo<uint32_t>(shapedType);
      }
    } else if (mlirTypeIsAInteger(elementType) &&
               mlirIntegerTypeGetWidth(elementType) == 64) {
      if (mlirIntegerTypeIsSignless(elementType) ||
          mlirIntegerTypeIsSigned(elementType)) {
        // i64
        return bufferInfo<int64_t>(shapedType);
      }
      if (mlirIntegerTypeIsUnsigned(elementType)) {
        // unsigned i64
        return bufferInfo<uint64_t>(shapedType);
      }
    } else if (mlirTypeIsAInteger(elementType) &&
               mlirIntegerTypeGetWidth(elementType) == 8) {
      if (mlirIntegerTypeIsSignless(elementType) ||
          mlirIntegerTypeIsSigned(elementType)) {
        // i8
        return bufferInfo<int8_t>(shapedType);
      }
      if (mlirIntegerTypeIsUnsigned(elementType)) {
        // unsigned i8
        return bufferInfo<uint8_t>(shapedType);
      }
    } else if (mlirTypeIsAInteger(elementType) &&
               mlirIntegerTypeGetWidth(elementType) == 16) {
      if (mlirIntegerTypeIsSignless(elementType) ||
          mlirIntegerTypeIsSigned(elementType)) {
        // i16
        return bufferInfo<int16_t>(shapedType);
      }
      if (mlirIntegerTypeIsUnsigned(elementType)) {
        // unsigned i16
        return bufferInfo<uint16_t>(shapedType);
      }
    }

    // TODO: Currently crashes the program.
    // Reported as https://github.com/pybind/pybind11/issues/3336
    throw std::invalid_argument(
        "unsupported data type for conversion to Python buffer");
  }

  static void bindDerived(ClassTy &c) {
    c.def("__len__", &PyDenseElementsAttribute::dunderLen)
        .def_static("get", PyDenseElementsAttribute::getFromBuffer,
                    py::arg("array"), py::arg("signless") = true,
                    py::arg("type") = py::none(), py::arg("shape") = py::none(),
                    py::arg("context") = py::none(),
                    kDenseElementsAttrGetDocstring)
        .def_static("get_splat", PyDenseElementsAttribute::getSplat,
                    py::arg("shaped_type"), py::arg("element_attr"),
                    "Gets a DenseElementsAttr where all values are the same")
        .def_property_readonly("is_splat",
                               [](PyDenseElementsAttribute &self) -> bool {
                                 return mlirDenseElementsAttrIsSplat(self);
                               })
        .def_buffer(&PyDenseElementsAttribute::accessBuffer);
  }

private:
  static bool isUnsignedIntegerFormat(const std::string &format) {
    if (format.empty())
      return false;
    char code = format[0];
    return code == 'I' || code == 'B' || code == 'H' || code == 'L' ||
           code == 'Q';
  }

  static bool isSignedIntegerFormat(const std::string &format) {
    if (format.empty())
      return false;
    char code = format[0];
    return code == 'i' || code == 'b' || code == 'h' || code == 'l' ||
           code == 'q';
  }

  template <typename Type>
  py::buffer_info bufferInfo(MlirType shapedType,
                             const char *explicitFormat = nullptr) {
    intptr_t rank = mlirShapedTypeGetRank(shapedType);
    // Prepare the data for the buffer_info.
    // Buffer is configured for read-only access below.
    Type *data = static_cast<Type *>(
        const_cast<void *>(mlirDenseElementsAttrGetRawData(*this)));
    // Prepare the shape for the buffer_info.
    SmallVector<intptr_t, 4> shape;
    for (intptr_t i = 0; i < rank; ++i)
      shape.push_back(mlirShapedTypeGetDimSize(shapedType, i));
    // Prepare the strides for the buffer_info.
    SmallVector<intptr_t, 4> strides;
    intptr_t strideFactor = 1;
    for (intptr_t i = 1; i < rank; ++i) {
      strideFactor = 1;
      for (intptr_t j = i; j < rank; ++j) {
        strideFactor *= mlirShapedTypeGetDimSize(shapedType, j);
      }
      strides.push_back(sizeof(Type) * strideFactor);
    }
    strides.push_back(sizeof(Type));
    std::string format;
    if (explicitFormat) {
      format = explicitFormat;
    } else {
      format = py::format_descriptor<Type>::format();
    }
    return py::buffer_info(data, sizeof(Type), format, rank, shape, strides,
                           /*readonly=*/true);
  }
}; // namespace

/// Refinement of the PyDenseElementsAttribute for attributes containing integer
/// (and boolean) values. Supports element access.
class PyDenseIntElementsAttribute
    : public PyConcreteAttribute<PyDenseIntElementsAttribute,
                                 PyDenseElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseIntElements;
  static constexpr const char *pyClassName = "DenseIntElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  /// Returns the element at the given linear position. Asserts if the index is
  /// out of range.
  py::int_ dunderGetItem(intptr_t pos) {
    if (pos < 0 || pos >= dunderLen()) {
      throw SetPyError(PyExc_IndexError,
                       "attempt to access out of bounds element");
    }

    MlirType type = mlirAttributeGetType(*this);
    type = mlirShapedTypeGetElementType(type);
    assert(mlirTypeIsAInteger(type) &&
           "expected integer element type in dense int elements attribute");
    // Dispatch element extraction to an appropriate C function based on the
    // elemental type of the attribute. py::int_ is implicitly constructible
    // from any C++ integral type and handles bitwidth correctly.
    // TODO: consider caching the type properties in the constructor to avoid
    // querying them on each element access.
    unsigned width = mlirIntegerTypeGetWidth(type);
    bool isUnsigned = mlirIntegerTypeIsUnsigned(type);
    if (isUnsigned) {
      if (width == 1) {
        return mlirDenseElementsAttrGetBoolValue(*this, pos);
      }
      if (width == 8) {
        return mlirDenseElementsAttrGetUInt8Value(*this, pos);
      }
      if (width == 16) {
        return mlirDenseElementsAttrGetUInt16Value(*this, pos);
      }
      if (width == 32) {
        return mlirDenseElementsAttrGetUInt32Value(*this, pos);
      }
      if (width == 64) {
        return mlirDenseElementsAttrGetUInt64Value(*this, pos);
      }
    } else {
      if (width == 1) {
        return mlirDenseElementsAttrGetBoolValue(*this, pos);
      }
      if (width == 8) {
        return mlirDenseElementsAttrGetInt8Value(*this, pos);
      }
      if (width == 16) {
        return mlirDenseElementsAttrGetInt16Value(*this, pos);
      }
      if (width == 32) {
        return mlirDenseElementsAttrGetInt32Value(*this, pos);
      }
      if (width == 64) {
        return mlirDenseElementsAttrGetInt64Value(*this, pos);
      }
    }
    throw SetPyError(PyExc_TypeError, "Unsupported integer type");
  }

  static void bindDerived(ClassTy &c) {
    c.def("__getitem__", &PyDenseIntElementsAttribute::dunderGetItem);
  }
};

class PyDictAttribute : public PyConcreteAttribute<PyDictAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADictionary;
  static constexpr const char *pyClassName = "DictAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  intptr_t dunderLen() { return mlirDictionaryAttrGetNumElements(*this); }

  bool dunderContains(const std::string &name) {
    return !mlirAttributeIsNull(
        mlirDictionaryAttrGetElementByName(*this, toMlirStringRef(name)));
  }

  static void bindDerived(ClassTy &c) {
    c.def("__contains__", &PyDictAttribute::dunderContains);
    c.def("__len__", &PyDictAttribute::dunderLen);
    c.def_static(
        "get",
        [](py::dict attributes, DefaultingPyMlirContext context) {
          SmallVector<MlirNamedAttribute> mlirNamedAttributes;
          mlirNamedAttributes.reserve(attributes.size());
          for (auto &it : attributes) {
            auto &mlirAttr = it.second.cast<PyAttribute &>();
            auto name = it.first.cast<std::string>();
            mlirNamedAttributes.push_back(mlirNamedAttributeGet(
                mlirIdentifierGet(mlirAttributeGetContext(mlirAttr),
                                  toMlirStringRef(name)),
                mlirAttr));
          }
          MlirAttribute attr =
              mlirDictionaryAttrGet(context->get(), mlirNamedAttributes.size(),
                                    mlirNamedAttributes.data());
          return PyDictAttribute(context->getRef(), attr);
        },
        py::arg("value") = py::dict(), py::arg("context") = py::none(),
        "Gets an uniqued dict attribute");
    c.def("__getitem__", [](PyDictAttribute &self, const std::string &name) {
      MlirAttribute attr =
          mlirDictionaryAttrGetElementByName(self, toMlirStringRef(name));
      if (mlirAttributeIsNull(attr)) {
        throw SetPyError(PyExc_KeyError,
                         "attempt to access a non-existent attribute");
      }
      return PyAttribute(self.getContext(), attr);
    });
    c.def("__getitem__", [](PyDictAttribute &self, intptr_t index) {
      if (index < 0 || index >= self.dunderLen()) {
        throw SetPyError(PyExc_IndexError,
                         "attempt to access out of bounds attribute");
      }
      MlirNamedAttribute namedAttr = mlirDictionaryAttrGetElement(self, index);
      return PyNamedAttribute(
          namedAttr.attribute,
          std::string(mlirIdentifierStr(namedAttr.name).data));
    });
  }
};

/// Refinement of PyDenseElementsAttribute for attributes containing
/// floating-point values. Supports element access.
class PyDenseFPElementsAttribute
    : public PyConcreteAttribute<PyDenseFPElementsAttribute,
                                 PyDenseElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseFPElements;
  static constexpr const char *pyClassName = "DenseFPElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  py::float_ dunderGetItem(intptr_t pos) {
    if (pos < 0 || pos >= dunderLen()) {
      throw SetPyError(PyExc_IndexError,
                       "attempt to access out of bounds element");
    }

    MlirType type = mlirAttributeGetType(*this);
    type = mlirShapedTypeGetElementType(type);
    // Dispatch element extraction to an appropriate C function based on the
    // elemental type of the attribute. py::float_ is implicitly constructible
    // from float and double.
    // TODO: consider caching the type properties in the constructor to avoid
    // querying them on each element access.
    if (mlirTypeIsAF32(type)) {
      return mlirDenseElementsAttrGetFloatValue(*this, pos);
    }
    if (mlirTypeIsAF64(type)) {
      return mlirDenseElementsAttrGetDoubleValue(*this, pos);
    }
    throw SetPyError(PyExc_TypeError, "Unsupported floating-point type");
  }

  static void bindDerived(ClassTy &c) {
    c.def("__getitem__", &PyDenseFPElementsAttribute::dunderGetItem);
  }
};

class PyTypeAttribute : public PyConcreteAttribute<PyTypeAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAType;
  static constexpr const char *pyClassName = "TypeAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType value, DefaultingPyMlirContext context) {
          MlirAttribute attr = mlirTypeAttrGet(value.get());
          return PyTypeAttribute(context->getRef(), attr);
        },
        py::arg("value"), py::arg("context") = py::none(),
        "Gets a uniqued Type attribute");
    c.def_property_readonly("value", [](PyTypeAttribute &self) {
      return PyType(self.getContext()->getRef(),
                    mlirTypeAttrGetValue(self.get()));
    });
  }
};

/// Unit Attribute subclass. Unit attributes don't have values.
class PyUnitAttribute : public PyConcreteAttribute<PyUnitAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAUnit;
  static constexpr const char *pyClassName = "UnitAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return PyUnitAttribute(context->getRef(),
                                 mlirUnitAttrGet(context->get()));
        },
        py::arg("context") = py::none(), "Create a Unit attribute.");
  }
};

} // namespace

void mlir::python::populateIRAttributes(py::module &m) {
  PyAffineMapAttribute::bind(m);
  PyArrayAttribute::bind(m);
  PyArrayAttribute::PyArrayAttributeIterator::bind(m);
  PyBoolAttribute::bind(m);
  PyDenseElementsAttribute::bind(m);
  PyDenseFPElementsAttribute::bind(m);
  PyDenseIntElementsAttribute::bind(m);
  PyDictAttribute::bind(m);
  PyFlatSymbolRefAttribute::bind(m);
  PyFloatAttribute::bind(m);
  PyIntegerAttribute::bind(m);
  PyStringAttribute::bind(m);
  PyTypeAttribute::bind(m);
  PyUnitAttribute::bind(m);
}
