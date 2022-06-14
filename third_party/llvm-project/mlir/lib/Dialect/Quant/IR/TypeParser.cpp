//===- TypeParser.h - Quantization Type Parser ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace quant;

static IntegerType parseStorageType(DialectAsmParser &parser, bool &isSigned) {
  auto typeLoc = parser.getCurrentLocation();
  IntegerType type;

  // Parse storage type (alpha_ident, integer_literal).
  StringRef identifier;
  unsigned storageTypeWidth = 0;
  OptionalParseResult result = parser.parseOptionalType(type);
  if (result.hasValue()) {
    if (!succeeded(*result))
      return nullptr;
    isSigned = !type.isUnsigned();
    storageTypeWidth = type.getWidth();
  } else if (succeeded(parser.parseKeyword(&identifier))) {
    // Otherwise, this must be an unsigned integer (`u` integer-literal).
    if (!identifier.consume_front("u")) {
      parser.emitError(typeLoc, "illegal storage type prefix");
      return nullptr;
    }
    if (identifier.getAsInteger(10, storageTypeWidth)) {
      parser.emitError(typeLoc, "expected storage type width");
      return nullptr;
    }
    isSigned = false;
    type = parser.getBuilder().getIntegerType(storageTypeWidth);
  } else {
    return nullptr;
  }

  if (storageTypeWidth == 0 ||
      storageTypeWidth > QuantizedType::MaxStorageBits) {
    parser.emitError(typeLoc, "illegal storage type size: ")
        << storageTypeWidth;
    return nullptr;
  }

  return type;
}

static ParseResult parseStorageRange(DialectAsmParser &parser,
                                     IntegerType storageType, bool isSigned,
                                     int64_t &storageTypeMin,
                                     int64_t &storageTypeMax) {
  int64_t defaultIntegerMin = QuantizedType::getDefaultMinimumForInteger(
      isSigned, storageType.getWidth());
  int64_t defaultIntegerMax = QuantizedType::getDefaultMaximumForInteger(
      isSigned, storageType.getWidth());
  if (failed(parser.parseOptionalLess())) {
    storageTypeMin = defaultIntegerMin;
    storageTypeMax = defaultIntegerMax;
    return success();
  }

  // Explicit storage min and storage max.
  SMLoc minLoc = parser.getCurrentLocation(), maxLoc;
  if (parser.parseInteger(storageTypeMin) || parser.parseColon() ||
      parser.getCurrentLocation(&maxLoc) ||
      parser.parseInteger(storageTypeMax) || parser.parseGreater())
    return failure();
  if (storageTypeMin < defaultIntegerMin) {
    return parser.emitError(minLoc, "illegal storage type minimum: ")
           << storageTypeMin;
  }
  if (storageTypeMax > defaultIntegerMax) {
    return parser.emitError(maxLoc, "illegal storage type maximum: ")
           << storageTypeMax;
  }
  return success();
}

static FloatType parseExpressedTypeAndRange(DialectAsmParser &parser,
                                            double &min, double &max) {
  auto typeLoc = parser.getCurrentLocation();
  FloatType type;

  if (failed(parser.parseType(type))) {
    parser.emitError(typeLoc, "expecting float expressed type");
    return nullptr;
  }

  // Calibrated min and max values.
  if (parser.parseLess() || parser.parseFloat(min) || parser.parseColon() ||
      parser.parseFloat(max) || parser.parseGreater()) {
    parser.emitError(typeLoc, "calibrated values must be present");
    return nullptr;
  }
  return type;
}

/// Parses an AnyQuantizedType.
///
///   any ::= `any<` storage-spec (expressed-type-spec)?`>`
///   storage-spec ::= storage-type (`<` storage-range `>`)?
///   storage-range ::= integer-literal `:` integer-literal
///   storage-type ::= (`i` | `u`) integer-literal
///   expressed-type-spec ::= `:` `f` integer-literal
static Type parseAnyType(DialectAsmParser &parser) {
  IntegerType storageType;
  FloatType expressedType;
  unsigned typeFlags = 0;
  int64_t storageTypeMin;
  int64_t storageTypeMax;

  // Type specification.
  if (parser.parseLess())
    return nullptr;

  // Storage type.
  bool isSigned = false;
  storageType = parseStorageType(parser, isSigned);
  if (!storageType) {
    return nullptr;
  }
  if (isSigned) {
    typeFlags |= QuantizationFlags::Signed;
  }

  // Storage type range.
  if (parseStorageRange(parser, storageType, isSigned, storageTypeMin,
                        storageTypeMax)) {
    return nullptr;
  }

  // Optional expressed type.
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseType(expressedType)) {
      return nullptr;
    }
  }

  if (parser.parseGreater()) {
    return nullptr;
  }

  return parser.getChecked<AnyQuantizedType>(
      typeFlags, storageType, expressedType, storageTypeMin, storageTypeMax);
}

static ParseResult parseQuantParams(DialectAsmParser &parser, double &scale,
                                    int64_t &zeroPoint) {
  // scale[:zeroPoint]?
  // scale.
  if (parser.parseFloat(scale))
    return failure();

  // zero point.
  zeroPoint = 0;
  if (failed(parser.parseOptionalColon())) {
    // Default zero point.
    return success();
  }

  return parser.parseInteger(zeroPoint);
}

/// Parses a UniformQuantizedType.
///
///   uniform_type ::= uniform_per_layer
///                  | uniform_per_axis
///   uniform_per_layer ::= `uniform<` storage-spec expressed-type-spec
///                          `,` scale-zero `>`
///   uniform_per_axis ::= `uniform<` storage-spec expressed-type-spec
///                        axis-spec `,` scale-zero-list `>`
///   storage-spec ::= storage-type (`<` storage-range `>`)?
///   storage-range ::= integer-literal `:` integer-literal
///   storage-type ::= (`i` | `u`) integer-literal
///   expressed-type-spec ::= `:` `f` integer-literal
///   axis-spec ::= `:` integer-literal
///   scale-zero ::= float-literal `:` integer-literal
///   scale-zero-list ::= `{` scale-zero (`,` scale-zero)* `}`
static Type parseUniformType(DialectAsmParser &parser) {
  IntegerType storageType;
  FloatType expressedType;
  unsigned typeFlags = 0;
  int64_t storageTypeMin;
  int64_t storageTypeMax;
  bool isPerAxis = false;
  int32_t quantizedDimension;
  SmallVector<double, 1> scales;
  SmallVector<int64_t, 1> zeroPoints;

  // Type specification.
  if (parser.parseLess()) {
    return nullptr;
  }

  // Storage type.
  bool isSigned = false;
  storageType = parseStorageType(parser, isSigned);
  if (!storageType) {
    return nullptr;
  }
  if (isSigned) {
    typeFlags |= QuantizationFlags::Signed;
  }

  // Storage type range.
  if (parseStorageRange(parser, storageType, isSigned, storageTypeMin,
                        storageTypeMax)) {
    return nullptr;
  }

  // Expressed type.
  if (parser.parseColon() || parser.parseType(expressedType)) {
    return nullptr;
  }

  // Optionally parse quantized dimension for per-axis quantization.
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseInteger(quantizedDimension))
      return nullptr;
    isPerAxis = true;
  }

  // Comma leading into range_spec.
  if (parser.parseComma()) {
    return nullptr;
  }

  // Parameter specification.
  // For per-axis, ranges are in a {} delimitted list.
  if (isPerAxis) {
    if (parser.parseLBrace()) {
      return nullptr;
    }
  }

  // Parse scales/zeroPoints.
  SMLoc scaleZPLoc = parser.getCurrentLocation();
  do {
    scales.resize(scales.size() + 1);
    zeroPoints.resize(zeroPoints.size() + 1);
    if (parseQuantParams(parser, scales.back(), zeroPoints.back())) {
      return nullptr;
    }
  } while (isPerAxis && succeeded(parser.parseOptionalComma()));

  if (isPerAxis) {
    if (parser.parseRBrace()) {
      return nullptr;
    }
  }

  if (parser.parseGreater()) {
    return nullptr;
  }

  if (!isPerAxis && scales.size() > 1) {
    return (parser.emitError(scaleZPLoc,
                             "multiple scales/zeroPoints provided, but "
                             "quantizedDimension wasn't specified"),
            nullptr);
  }

  if (isPerAxis) {
    ArrayRef<double> scalesRef(scales.begin(), scales.end());
    ArrayRef<int64_t> zeroPointsRef(zeroPoints.begin(), zeroPoints.end());
    return parser.getChecked<UniformQuantizedPerAxisType>(
        typeFlags, storageType, expressedType, scalesRef, zeroPointsRef,
        quantizedDimension, storageTypeMin, storageTypeMax);
  }

  return parser.getChecked<UniformQuantizedType>(
      typeFlags, storageType, expressedType, scales.front(), zeroPoints.front(),
      storageTypeMin, storageTypeMax);
}

/// Parses an CalibratedQuantizedType.
///
///   calibrated ::= `calibrated<` expressed-spec `>`
///   expressed-spec ::= expressed-type `<` calibrated-range `>`
///   expressed-type ::= `f` integer-literal
///   calibrated-range ::= float-literal `:` float-literal
static Type parseCalibratedType(DialectAsmParser &parser) {
  FloatType expressedType;
  double min;
  double max;

  // Type specification.
  if (parser.parseLess())
    return nullptr;

  // Expressed type.
  expressedType = parseExpressedTypeAndRange(parser, min, max);
  if (!expressedType) {
    return nullptr;
  }

  if (parser.parseGreater()) {
    return nullptr;
  }

  return parser.getChecked<CalibratedQuantizedType>(expressedType, min, max);
}

/// Parse a type registered to this dialect.
Type QuantizationDialect::parseType(DialectAsmParser &parser) const {
  // All types start with an identifier that we switch on.
  StringRef typeNameSpelling;
  if (failed(parser.parseKeyword(&typeNameSpelling)))
    return nullptr;

  if (typeNameSpelling == "uniform")
    return parseUniformType(parser);
  if (typeNameSpelling == "any")
    return parseAnyType(parser);
  if (typeNameSpelling == "calibrated")
    return parseCalibratedType(parser);

  parser.emitError(parser.getNameLoc(),
                   "unknown quantized type " + typeNameSpelling);
  return nullptr;
}

static void printStorageType(QuantizedType type, DialectAsmPrinter &out) {
  // storage type
  unsigned storageWidth = type.getStorageTypeIntegralWidth();
  bool isSigned = type.isSigned();
  if (isSigned) {
    out << "i" << storageWidth;
  } else {
    out << "u" << storageWidth;
  }

  // storageTypeMin and storageTypeMax if not default.
  int64_t defaultIntegerMin =
      QuantizedType::getDefaultMinimumForInteger(isSigned, storageWidth);
  int64_t defaultIntegerMax =
      QuantizedType::getDefaultMaximumForInteger(isSigned, storageWidth);
  if (defaultIntegerMin != type.getStorageTypeMin() ||
      defaultIntegerMax != type.getStorageTypeMax()) {
    out << "<" << type.getStorageTypeMin() << ":" << type.getStorageTypeMax()
        << ">";
  }
}

static void printQuantParams(double scale, int64_t zeroPoint,
                             DialectAsmPrinter &out) {
  out << scale;
  if (zeroPoint != 0) {
    out << ":" << zeroPoint;
  }
}

/// Helper that prints a AnyQuantizedType.
static void printAnyQuantizedType(AnyQuantizedType type,
                                  DialectAsmPrinter &out) {
  out << "any<";
  printStorageType(type, out);
  if (Type expressedType = type.getExpressedType()) {
    out << ":" << expressedType;
  }
  out << ">";
}

/// Helper that prints a UniformQuantizedType.
static void printUniformQuantizedType(UniformQuantizedType type,
                                      DialectAsmPrinter &out) {
  out << "uniform<";
  printStorageType(type, out);
  out << ":" << type.getExpressedType() << ", ";

  // scheme specific parameters
  printQuantParams(type.getScale(), type.getZeroPoint(), out);
  out << ">";
}

/// Helper that prints a UniformQuantizedPerAxisType.
static void printUniformQuantizedPerAxisType(UniformQuantizedPerAxisType type,
                                             DialectAsmPrinter &out) {
  out << "uniform<";
  printStorageType(type, out);
  out << ":" << type.getExpressedType() << ":";
  out << type.getQuantizedDimension();
  out << ", ";

  // scheme specific parameters
  ArrayRef<double> scales = type.getScales();
  ArrayRef<int64_t> zeroPoints = type.getZeroPoints();
  out << "{";
  llvm::interleave(
      llvm::seq<size_t>(0, scales.size()), out,
      [&](size_t index) {
        printQuantParams(scales[index], zeroPoints[index], out);
      },
      ",");
  out << "}>";
}

/// Helper that prints a CalibratedQuantizedType.
static void printCalibratedQuantizedType(CalibratedQuantizedType type,
                                         DialectAsmPrinter &out) {
  out << "calibrated<" << type.getExpressedType();
  out << "<" << type.getMin() << ":" << type.getMax() << ">";
  out << ">";
}

/// Print a type registered to this dialect.
void QuantizationDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (auto anyType = type.dyn_cast<AnyQuantizedType>())
    printAnyQuantizedType(anyType, os);
  else if (auto uniformType = type.dyn_cast<UniformQuantizedType>())
    printUniformQuantizedType(uniformType, os);
  else if (auto perAxisType = type.dyn_cast<UniformQuantizedPerAxisType>())
    printUniformQuantizedPerAxisType(perAxisType, os);
  else if (auto calibratedType = type.dyn_cast<CalibratedQuantizedType>())
    printCalibratedQuantizedType(calibratedType, os);
  else
    llvm_unreachable("Unhandled quantized type");
}
