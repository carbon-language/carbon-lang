//===- DialectQuant.cpp - 'quant' dialect submodule -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Quant.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace llvm;
using namespace mlir;
using namespace mlir::python::adaptors;

static void populateDialectQuantSubmodule(const py::module &m) {
  //===-------------------------------------------------------------------===//
  // QuantizedType
  //===-------------------------------------------------------------------===//

  auto quantizedType =
      mlir_type_subclass(m, "QuantizedType", mlirTypeIsAQuantizedType);
  quantizedType.def_staticmethod(
      "default_minimum_for_integer",
      [](bool isSigned, unsigned integralWidth) {
        return mlirQuantizedTypeGetDefaultMinimumForInteger(isSigned,
                                                            integralWidth);
      },
      "Default minimum value for the integer with the specified signedness and "
      "bit width.",
      py::arg("is_signed"), py::arg("integral_width"));
  quantizedType.def_staticmethod(
      "default_maximum_for_integer",
      [](bool isSigned, unsigned integralWidth) {
        return mlirQuantizedTypeGetDefaultMaximumForInteger(isSigned,
                                                            integralWidth);
      },
      "Default maximum value for the integer with the specified signedness and "
      "bit width.",
      py::arg("is_signed"), py::arg("integral_width"));
  quantizedType.def_property_readonly(
      "expressed_type",
      [](MlirType type) { return mlirQuantizedTypeGetExpressedType(type); },
      "Type expressed by this quantized type.");
  quantizedType.def_property_readonly(
      "flags", [](MlirType type) { return mlirQuantizedTypeGetFlags(type); },
      "Flags of this quantized type (named accessors should be preferred to "
      "this)");
  quantizedType.def_property_readonly(
      "is_signed",
      [](MlirType type) { return mlirQuantizedTypeIsSigned(type); },
      "Signedness of this quantized type.");
  quantizedType.def_property_readonly(
      "storage_type",
      [](MlirType type) { return mlirQuantizedTypeGetStorageType(type); },
      "Storage type backing this quantized type.");
  quantizedType.def_property_readonly(
      "storage_type_min",
      [](MlirType type) { return mlirQuantizedTypeGetStorageTypeMin(type); },
      "The minimum value held by the storage type of this quantized type.");
  quantizedType.def_property_readonly(
      "storage_type_max",
      [](MlirType type) { return mlirQuantizedTypeGetStorageTypeMax(type); },
      "The maximum value held by the storage type of this quantized type.");
  quantizedType.def_property_readonly(
      "storage_type_integral_width",
      [](MlirType type) {
        return mlirQuantizedTypeGetStorageTypeIntegralWidth(type);
      },
      "The bitwidth of the storage type of this quantized type.");
  quantizedType.def(
      "is_compatible_expressed_type",
      [](MlirType type, MlirType candidate) {
        return mlirQuantizedTypeIsCompatibleExpressedType(type, candidate);
      },
      "Checks whether the candidate type can be expressed by this quantized "
      "type.",
      py::arg("candidate"));
  quantizedType.def_property_readonly(
      "quantized_element_type",
      [](MlirType type) {
        return mlirQuantizedTypeGetQuantizedElementType(type);
      },
      "Element type of this quantized type expressed as quantized type.");
  quantizedType.def(
      "cast_from_storage_type",
      [](MlirType type, MlirType candidate) {
        MlirType castResult =
            mlirQuantizedTypeCastFromStorageType(type, candidate);
        if (!mlirTypeIsNull(castResult))
          return castResult;
        throw py::type_error("Invalid cast.");
      },
      "Casts from a type based on the storage type of this quantized type to a "
      "corresponding type based on the quantized type. Raises TypeError if the "
      "cast is not valid.",
      py::arg("candidate"));
  quantizedType.def_staticmethod(
      "cast_to_storage_type",
      [](MlirType type) {
        MlirType castResult = mlirQuantizedTypeCastToStorageType(type);
        if (!mlirTypeIsNull(castResult))
          return castResult;
        throw py::type_error("Invalid cast.");
      },
      "Casts from a type based on a quantized type to a corresponding type "
      "based on the storage type of this quantized type. Raises TypeError if "
      "the cast is not valid.",
      py::arg("type"));
  quantizedType.def(
      "cast_from_expressed_type",
      [](MlirType type, MlirType candidate) {
        MlirType castResult =
            mlirQuantizedTypeCastFromExpressedType(type, candidate);
        if (!mlirTypeIsNull(castResult))
          return castResult;
        throw py::type_error("Invalid cast.");
      },
      "Casts from a type based on the expressed type of this quantized type to "
      "a corresponding type based on the quantized type. Raises TypeError if "
      "the cast is not valid.",
      py::arg("candidate"));
  quantizedType.def_staticmethod(
      "cast_to_expressed_type",
      [](MlirType type) {
        MlirType castResult = mlirQuantizedTypeCastToExpressedType(type);
        if (!mlirTypeIsNull(castResult))
          return castResult;
        throw py::type_error("Invalid cast.");
      },
      "Casts from a type based on a quantized type to a corresponding type "
      "based on the expressed type of this quantized type. Raises TypeError if "
      "the cast is not valid.",
      py::arg("type"));
  quantizedType.def(
      "cast_expressed_to_storage_type",
      [](MlirType type, MlirType candidate) {
        MlirType castResult =
            mlirQuantizedTypeCastExpressedToStorageType(type, candidate);
        if (!mlirTypeIsNull(castResult))
          return castResult;
        throw py::type_error("Invalid cast.");
      },
      "Casts from a type based on the expressed type of this quantized type to "
      "a corresponding type based on the storage type. Raises TypeError if the "
      "cast is not valid.",
      py::arg("candidate"));

  quantizedType.get_class().attr("FLAG_SIGNED") =
      mlirQuantizedTypeGetSignedFlag();

  //===-------------------------------------------------------------------===//
  // AnyQuantizedType
  //===-------------------------------------------------------------------===//

  auto anyQuantizedType =
      mlir_type_subclass(m, "AnyQuantizedType", mlirTypeIsAAnyQuantizedType,
                         quantizedType.get_class());
  anyQuantizedType.def_classmethod(
      "get",
      [](py::object cls, unsigned flags, MlirType storageType,
         MlirType expressedType, int64_t storageTypeMin,
         int64_t storageTypeMax) {
        return cls(mlirAnyQuantizedTypeGet(flags, storageType, expressedType,
                                           storageTypeMin, storageTypeMax));
      },
      "Gets an instance of AnyQuantizedType in the same context as the "
      "provided storage type.",
      py::arg("cls"), py::arg("flags"), py::arg("storage_type"),
      py::arg("expressed_type"), py::arg("storage_type_min"),
      py::arg("storage_type_max"));

  //===-------------------------------------------------------------------===//
  // UniformQuantizedType
  //===-------------------------------------------------------------------===//

  auto uniformQuantizedType = mlir_type_subclass(
      m, "UniformQuantizedType", mlirTypeIsAUniformQuantizedType,
      quantizedType.get_class());
  uniformQuantizedType.def_classmethod(
      "get",
      [](py::object cls, unsigned flags, MlirType storageType,
         MlirType expressedType, double scale, int64_t zeroPoint,
         int64_t storageTypeMin, int64_t storageTypeMax) {
        return cls(mlirUniformQuantizedTypeGet(flags, storageType,
                                               expressedType, scale, zeroPoint,
                                               storageTypeMin, storageTypeMax));
      },
      "Gets an instance of UniformQuantizedType in the same context as the "
      "provided storage type.",
      py::arg("cls"), py::arg("flags"), py::arg("storage_type"),
      py::arg("expressed_type"), py::arg("scale"), py::arg("zero_point"),
      py::arg("storage_type_min"), py::arg("storage_type_max"));
  uniformQuantizedType.def_property_readonly(
      "scale",
      [](MlirType type) { return mlirUniformQuantizedTypeGetScale(type); },
      "The scale designates the difference between the real values "
      "corresponding to consecutive quantized values differing by 1.");
  uniformQuantizedType.def_property_readonly(
      "zero_point",
      [](MlirType type) { return mlirUniformQuantizedTypeGetZeroPoint(type); },
      "The storage value corresponding to the real value 0 in the affine "
      "equation.");
  uniformQuantizedType.def_property_readonly(
      "is_fixed_point",
      [](MlirType type) { return mlirUniformQuantizedTypeIsFixedPoint(type); },
      "Fixed point values are real numbers divided by a scale.");

  //===-------------------------------------------------------------------===//
  // UniformQuantizedPerAxisType
  //===-------------------------------------------------------------------===//
  auto uniformQuantizedPerAxisType = mlir_type_subclass(
      m, "UniformQuantizedPerAxisType", mlirTypeIsAUniformQuantizedPerAxisType,
      quantizedType.get_class());
  uniformQuantizedPerAxisType.def_classmethod(
      "get",
      [](py::object cls, unsigned flags, MlirType storageType,
         MlirType expressedType, std::vector<double> scales,
         std::vector<int64_t> zeroPoints, int32_t quantizedDimension,
         int64_t storageTypeMin, int64_t storageTypeMax) {
        if (scales.size() != zeroPoints.size())
          throw py::value_error(
              "Mismatching number of scales and zero points.");
        auto nDims = static_cast<intptr_t>(scales.size());
        return cls(mlirUniformQuantizedPerAxisTypeGet(
            flags, storageType, expressedType, nDims, scales.data(),
            zeroPoints.data(), quantizedDimension, storageTypeMin,
            storageTypeMax));
      },
      "Gets an instance of UniformQuantizedPerAxisType in the same context as "
      "the provided storage type.",
      py::arg("cls"), py::arg("flags"), py::arg("storage_type"),
      py::arg("expressed_type"), py::arg("scales"), py::arg("zero_points"),
      py::arg("quantized_dimension"), py::arg("storage_type_min"),
      py::arg("storage_type_max"));
  uniformQuantizedPerAxisType.def_property_readonly(
      "scales",
      [](MlirType type) {
        intptr_t nDim = mlirUniformQuantizedPerAxisTypeGetNumDims(type);
        std::vector<double> scales;
        scales.reserve(nDim);
        for (intptr_t i = 0; i < nDim; ++i) {
          double scale = mlirUniformQuantizedPerAxisTypeGetScale(type, i);
          scales.push_back(scale);
        }
      },
      "The scales designate the difference between the real values "
      "corresponding to consecutive quantized values differing by 1. The ith "
      "scale corresponds to the ith slice in the quantized_dimension.");
  uniformQuantizedPerAxisType.def_property_readonly(
      "zero_points",
      [](MlirType type) {
        intptr_t nDim = mlirUniformQuantizedPerAxisTypeGetNumDims(type);
        std::vector<int64_t> zeroPoints;
        zeroPoints.reserve(nDim);
        for (intptr_t i = 0; i < nDim; ++i) {
          int64_t zeroPoint =
              mlirUniformQuantizedPerAxisTypeGetZeroPoint(type, i);
          zeroPoints.push_back(zeroPoint);
        }
      },
      "the storage values corresponding to the real value 0 in the affine "
      "equation. The ith zero point corresponds to the ith slice in the "
      "quantized_dimension.");
  uniformQuantizedPerAxisType.def_property_readonly(
      "quantized_dimension",
      [](MlirType type) {
        return mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(type);
      },
      "Specifies the dimension of the shape that the scales and zero points "
      "correspond to.");
  uniformQuantizedPerAxisType.def_property_readonly(
      "is_fixed_point",
      [](MlirType type) {
        return mlirUniformQuantizedPerAxisTypeIsFixedPoint(type);
      },
      "Fixed point values are real numbers divided by a scale.");

  //===-------------------------------------------------------------------===//
  // CalibratedQuantizedType
  //===-------------------------------------------------------------------===//

  auto calibratedQuantizedType = mlir_type_subclass(
      m, "CalibratedQuantizedType", mlirTypeIsACalibratedQuantizedType,
      quantizedType.get_class());
  calibratedQuantizedType.def_classmethod(
      "get",
      [](py::object cls, MlirType expressedType, double min, double max) {
        return cls(mlirCalibratedQuantizedTypeGet(expressedType, min, max));
      },
      "Gets an instance of CalibratedQuantizedType in the same context as the "
      "provided expressed type.",
      py::arg("cls"), py::arg("expressed_type"), py::arg("min"),
      py::arg("max"));
  calibratedQuantizedType.def_property_readonly("min", [](MlirType type) {
    return mlirCalibratedQuantizedTypeGetMin(type);
  });
  calibratedQuantizedType.def_property_readonly("max", [](MlirType type) {
    return mlirCalibratedQuantizedTypeGetMax(type);
  });
}

PYBIND11_MODULE(_mlirDialectsQuant, m) {
  m.doc() = "MLIR Quantization dialect";

  populateDialectQuantSubmodule(m);
}
