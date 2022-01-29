//===- VectorOps.cpp - MLIR Vector Dialect Operations ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements convenience types for working with super-vectorization
// operations, in particular super-vector loads and stores.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/VectorUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/bit.h"
#include <numeric>

#include "mlir/Dialect/Vector/VectorOpsDialect.cpp.inc"
// Pull in all enum type and utility function definitions.
#include "mlir/Dialect/Vector/VectorOpsEnums.cpp.inc"

using namespace mlir;
using namespace mlir::vector;

/// Helper enum to classify mask value.
enum class MaskFormat {
  AllTrue = 0,
  AllFalse = 1,
  Unknown = 2,
};

/// Helper method to classify a 1-D mask value. Currently, the method
/// looks "under the hood" of a constant value with dense attributes
/// and a constant mask operation (since the client may be called at
/// various stages during progressive lowering).
static MaskFormat get1DMaskFormat(Value mask) {
  if (auto c = mask.getDefiningOp<arith::ConstantOp>()) {
    // Inspect constant dense values. We count up for bits that
    // are set, count down for bits that are cleared, and bail
    // when a mix is detected.
    if (auto denseElts = c.getValue().dyn_cast<DenseIntElementsAttr>()) {
      int64_t val = 0;
      for (bool b : denseElts.getValues<bool>())
        if (b && val >= 0)
          val++;
        else if (!b && val <= 0)
          val--;
        else
          return MaskFormat::Unknown;
      if (val > 0)
        return MaskFormat::AllTrue;
      if (val < 0)
        return MaskFormat::AllFalse;
    }
  } else if (auto m = mask.getDefiningOp<ConstantMaskOp>()) {
    // Inspect constant mask index. If the index exceeds the
    // dimension size, all bits are set. If the index is zero
    // or less, no bits are set.
    ArrayAttr masks = m.mask_dim_sizes();
    assert(masks.size() == 1);
    int64_t i = masks[0].cast<IntegerAttr>().getInt();
    int64_t u = m.getType().getDimSize(0);
    if (i >= u)
      return MaskFormat::AllTrue;
    if (i <= 0)
      return MaskFormat::AllFalse;
  }
  return MaskFormat::Unknown;
}

// Helper for verifying combining kinds in contractions and reductions.
static bool isSupportedCombiningKind(CombiningKind combiningKind,
                                     Type elementType) {
  switch (combiningKind) {
  case CombiningKind::ADD:
  case CombiningKind::MUL:
    return elementType.isIntOrIndexOrFloat();
  case CombiningKind::MINUI:
  case CombiningKind::MINSI:
  case CombiningKind::MAXUI:
  case CombiningKind::MAXSI:
  case CombiningKind::AND:
  case CombiningKind::OR:
  case CombiningKind::XOR:
    return elementType.isIntOrIndex();
  case CombiningKind::MINF:
  case CombiningKind::MAXF:
    return elementType.isa<FloatType>();
  }
  return false;
}

/// Return true if the last dimension of the MemRefType has unit stride. Also
/// return true for memrefs with no strides.
bool mlir::vector::isLastMemrefDimUnitStride(MemRefType type) {
  int64_t offset;
  SmallVector<int64_t> strides;
  auto successStrides = getStridesAndOffset(type, strides, offset);
  return succeeded(successStrides) && (strides.empty() || strides.back() == 1);
}

//===----------------------------------------------------------------------===//
// CombiningKindAttr
//===----------------------------------------------------------------------===//

namespace mlir {
namespace vector {
namespace detail {
struct BitmaskEnumStorage : public AttributeStorage {
  using KeyTy = uint64_t;

  BitmaskEnumStorage(KeyTy val) : value(val) {}

  bool operator==(const KeyTy &key) const { return value == key; }

  static BitmaskEnumStorage *construct(AttributeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<BitmaskEnumStorage>())
        BitmaskEnumStorage(key);
  }

  KeyTy value = 0;
};
} // namespace detail
} // namespace vector
} // namespace mlir

CombiningKindAttr CombiningKindAttr::get(CombiningKind kind,
                                         MLIRContext *context) {
  return Base::get(context, static_cast<uint64_t>(kind));
}

CombiningKind CombiningKindAttr::getKind() const {
  return static_cast<CombiningKind>(getImpl()->value);
}

static constexpr const CombiningKind combiningKindsList[] = {
    // clang-format off
    CombiningKind::ADD,
    CombiningKind::MUL,
    CombiningKind::MINUI,
    CombiningKind::MINSI,
    CombiningKind::MINF,
    CombiningKind::MAXUI,
    CombiningKind::MAXSI,
    CombiningKind::MAXF,
    CombiningKind::AND,
    CombiningKind::OR,
    CombiningKind::XOR,
    // clang-format on
};

void CombiningKindAttr::print(AsmPrinter &printer) const {
  printer << "<";
  auto kinds = llvm::make_filter_range(combiningKindsList, [&](auto kind) {
    return bitEnumContains(this->getKind(), kind);
  });
  llvm::interleaveComma(kinds, printer,
                        [&](auto kind) { printer << stringifyEnum(kind); });
  printer << ">";
}

Attribute CombiningKindAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};

  StringRef elemName;
  if (failed(parser.parseKeyword(&elemName)))
    return {};

  auto kind = symbolizeCombiningKind(elemName);
  if (!kind) {
    parser.emitError(parser.getNameLoc(), "Unknown combining kind: ")
        << elemName;
    return {};
  }

  if (failed(parser.parseGreater()))
    return {};

  return CombiningKindAttr::get(kind.getValue(), parser.getContext());
}

Attribute VectorDialect::parseAttribute(DialectAsmParser &parser,
                                        Type type) const {
  StringRef attrKind;
  if (parser.parseKeyword(&attrKind))
    return {};

  if (attrKind == "kind")
    return CombiningKindAttr::parse(parser, {});

  parser.emitError(parser.getNameLoc(), "Unknown attribute type: ") << attrKind;
  return {};
}

void VectorDialect::printAttribute(Attribute attr,
                                   DialectAsmPrinter &os) const {
  if (auto ck = attr.dyn_cast<CombiningKindAttr>()) {
    os << "kind";
    ck.print(os);
    return;
  }
  llvm_unreachable("Unknown attribute type");
}

//===----------------------------------------------------------------------===//
// VectorDialect
//===----------------------------------------------------------------------===//

void VectorDialect::initialize() {
  addAttributes<CombiningKindAttr>();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Vector/VectorOps.cpp.inc"
      >();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *VectorDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return builder.create<arith::ConstantOp>(loc, type, value);
}

IntegerType vector::getVectorSubscriptType(Builder &builder) {
  return builder.getIntegerType(64);
}

ArrayAttr vector::getVectorSubscriptAttr(Builder &builder,
                                         ArrayRef<int64_t> values) {
  return builder.getI64ArrayAttr(values);
}

//===----------------------------------------------------------------------===//
// MultiDimReductionOp
//===----------------------------------------------------------------------===//

void vector::MultiDimReductionOp::build(OpBuilder &builder,
                                        OperationState &result, Value source,
                                        ArrayRef<bool> reductionMask,
                                        CombiningKind kind) {
  result.addOperands(source);
  auto sourceVectorType = source.getType().cast<VectorType>();
  auto targetType = MultiDimReductionOp::inferDestType(
      sourceVectorType.getShape(), reductionMask,
      sourceVectorType.getElementType());
  result.addTypes(targetType);

  SmallVector<int64_t> reductionDims;
  for (const auto &en : llvm::enumerate(reductionMask))
    if (en.value())
      reductionDims.push_back(en.index());
  result.addAttribute(getReductionDimsAttrName(),
                      builder.getI64ArrayAttr(reductionDims));
  result.addAttribute(getKindAttrName(),
                      CombiningKindAttr::get(kind, builder.getContext()));
}

static LogicalResult verify(MultiDimReductionOp op) {
  auto reductionMask = op.getReductionMask();
  auto targetType = MultiDimReductionOp::inferDestType(
      op.getSourceVectorType().getShape(), reductionMask,
      op.getSourceVectorType().getElementType());
  // TODO: update to support 0-d vectors when available.
  if (targetType != op.getDestType())
    return op.emitError("invalid output vector type: ")
           << op.getDestType() << " (expected: " << targetType << ")";
  return success();
}

OpFoldResult MultiDimReductionOp::fold(ArrayRef<Attribute> operands) {
  // Single parallel dim, this is a noop.
  if (getSourceVectorType().getRank() == 1 && !isReducedDim(0))
    return source();
  return {};
}

//===----------------------------------------------------------------------===//
// ReductionOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ReductionOp op) {
  // Verify for 1-D vector.
  int64_t rank = op.getVectorType().getRank();
  if (rank != 1)
    return op.emitOpError("unsupported reduction rank: ") << rank;

  // Verify supported reduction kind.
  StringRef strKind = op.kind();
  auto maybeKind = symbolizeCombiningKind(strKind);
  if (!maybeKind)
    return op.emitOpError("unknown reduction kind: ") << strKind;

  Type eltType = op.dest().getType();
  if (!isSupportedCombiningKind(*maybeKind, eltType))
    return op.emitOpError("unsupported reduction type '")
           << eltType << "' for kind '" << op.kind() << "'";

  // Verify optional accumulator.
  if (!op.acc().empty()) {
    if (strKind != "add" && strKind != "mul")
      return op.emitOpError("no accumulator for reduction kind: ") << strKind;
    if (!eltType.isa<FloatType>())
      return op.emitOpError("no accumulator for type: ") << eltType;
  }

  return success();
}

static ParseResult parseReductionOp(OpAsmParser &parser,
                                    OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> operandsInfo;
  Type redType;
  Type resType;
  Attribute attr;
  if (parser.parseAttribute(attr, "kind", result.attributes) ||
      parser.parseComma() || parser.parseOperandList(operandsInfo) ||
      parser.parseColonType(redType) ||
      parser.parseKeywordType("into", resType) ||
      (!operandsInfo.empty() &&
       parser.resolveOperand(operandsInfo[0], redType, result.operands)) ||
      (operandsInfo.size() > 1 &&
       parser.resolveOperand(operandsInfo[1], resType, result.operands)) ||
      parser.addTypeToList(resType, result.types))
    return failure();
  if (operandsInfo.empty() || operandsInfo.size() > 2)
    return parser.emitError(parser.getNameLoc(),
                            "unsupported number of operands");
  return success();
}

static void print(OpAsmPrinter &p, ReductionOp op) {
  p << " \"" << op.kind() << "\", " << op.vector();
  if (!op.acc().empty())
    p << ", " << op.acc();
  p << " : " << op.vector().getType() << " into " << op.dest().getType();
}

Value mlir::vector::getVectorReductionOp(arith::AtomicRMWKind op,
                                         OpBuilder &builder, Location loc,
                                         Value vector) {
  Type scalarType = vector.getType().cast<ShapedType>().getElementType();
  switch (op) {
  case arith::AtomicRMWKind::addf:
  case arith::AtomicRMWKind::addi:
    return builder.create<vector::ReductionOp>(vector.getLoc(), scalarType,
                                               builder.getStringAttr("add"),
                                               vector, ValueRange{});
  case arith::AtomicRMWKind::mulf:
  case arith::AtomicRMWKind::muli:
    return builder.create<vector::ReductionOp>(vector.getLoc(), scalarType,
                                               builder.getStringAttr("mul"),
                                               vector, ValueRange{});
  case arith::AtomicRMWKind::minf:
    return builder.create<vector::ReductionOp>(vector.getLoc(), scalarType,
                                               builder.getStringAttr("minf"),
                                               vector, ValueRange{});
  case arith::AtomicRMWKind::mins:
    return builder.create<vector::ReductionOp>(vector.getLoc(), scalarType,
                                               builder.getStringAttr("minsi"),
                                               vector, ValueRange{});
  case arith::AtomicRMWKind::minu:
    return builder.create<vector::ReductionOp>(vector.getLoc(), scalarType,
                                               builder.getStringAttr("minui"),
                                               vector, ValueRange{});
  case arith::AtomicRMWKind::maxf:
    return builder.create<vector::ReductionOp>(vector.getLoc(), scalarType,
                                               builder.getStringAttr("maxf"),
                                               vector, ValueRange{});
  case arith::AtomicRMWKind::maxs:
    return builder.create<vector::ReductionOp>(vector.getLoc(), scalarType,
                                               builder.getStringAttr("maxsi"),
                                               vector, ValueRange{});
  case arith::AtomicRMWKind::maxu:
    return builder.create<vector::ReductionOp>(vector.getLoc(), scalarType,
                                               builder.getStringAttr("maxui"),
                                               vector, ValueRange{});
  // TODO: Add remaining reduction operations.
  default:
    (void)emitOptionalError(loc, "Reduction operation type not supported");
    break;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ContractionOp
//===----------------------------------------------------------------------===//

void vector::ContractionOp::build(OpBuilder &builder, OperationState &result,
                                  Value lhs, Value rhs, Value acc,
                                  ArrayRef<ArrayRef<AffineExpr>> indexingExprs,
                                  ArrayRef<StringRef> iteratorTypes) {
  result.addOperands({lhs, rhs, acc});
  result.addTypes(acc.getType());
  result.addAttribute(getIndexingMapsAttrName(),
                      builder.getAffineMapArrayAttr(
                          AffineMap::inferFromExprList(indexingExprs)));
  result.addAttribute(getIteratorTypesAttrName(),
                      builder.getStrArrayAttr(iteratorTypes));
}

void vector::ContractionOp::build(OpBuilder &builder, OperationState &result,
                                  Value lhs, Value rhs, Value acc,
                                  ArrayAttr indexingMaps,
                                  ArrayAttr iteratorTypes) {
  result.addOperands({lhs, rhs, acc});
  result.addTypes(acc.getType());
  result.addAttribute(getIndexingMapsAttrName(), indexingMaps);
  result.addAttribute(getIteratorTypesAttrName(), iteratorTypes);
  result.addAttribute(ContractionOp::getKindAttrName(),
                      CombiningKindAttr::get(ContractionOp::getDefaultKind(),
                                             builder.getContext()));
}

static ParseResult parseContractionOp(OpAsmParser &parser,
                                      OperationState &result) {
  OpAsmParser::OperandType lhsInfo;
  OpAsmParser::OperandType rhsInfo;
  OpAsmParser::OperandType accInfo;
  SmallVector<OpAsmParser::OperandType, 2> masksInfo;
  SmallVector<Type, 2> types;
  Type resultType;
  auto loc = parser.getCurrentLocation();
  DictionaryAttr dictAttr;
  // TODO: Unify linalg op attribute parsing.
  if (parser.parseAttribute(dictAttr, "_", result.attributes) ||
      parser.parseOperand(lhsInfo) || parser.parseComma() ||
      parser.parseOperand(rhsInfo) || parser.parseComma() ||
      parser.parseOperand(accInfo) ||
      parser.parseTrailingOperandList(masksInfo) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.parseKeywordType("into", resultType) ||
      parser.resolveOperand(lhsInfo, types[0], result.operands) ||
      parser.resolveOperand(rhsInfo, types[1], result.operands) ||
      parser.resolveOperand(accInfo, resultType, result.operands) ||
      parser.addTypeToList(resultType, result.types))
    return failure();
  result.attributes.assign(dictAttr.getValue().begin(),
                           dictAttr.getValue().end());
  if (!result.attributes.get(ContractionOp::getKindAttrName())) {
    result.addAttribute(ContractionOp::getKindAttrName(),
                        CombiningKindAttr::get(ContractionOp::getDefaultKind(),
                                               result.getContext()));
  }
  if (masksInfo.empty())
    return success();
  if (masksInfo.size() != 2)
    return parser.emitError(parser.getNameLoc(),
                            "expected zero or exactly 2 vector mask operands");
  auto lhsType = types[0].cast<VectorType>();
  auto rhsType = types[1].cast<VectorType>();
  auto maskElementType = parser.getBuilder().getI1Type();
  std::array<Type, 2> maskTypes = {
      VectorType::Builder(lhsType).setElementType(maskElementType),
      VectorType::Builder(rhsType).setElementType(maskElementType)};
  if (parser.resolveOperands(masksInfo, maskTypes, loc, result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, ContractionOp op) {
  // TODO: Unify printing code with linalg ops.
  auto attrNames = op.getTraitAttrNames();
  llvm::StringSet<> traitAttrsSet;
  traitAttrsSet.insert(attrNames.begin(), attrNames.end());
  SmallVector<NamedAttribute, 8> attrs;
  for (auto attr : op->getAttrs())
    if (traitAttrsSet.count(attr.getName().strref()) > 0)
      attrs.push_back(attr);

  auto dictAttr = DictionaryAttr::get(op.getContext(), attrs);
  p << " " << dictAttr << " " << op.lhs() << ", ";
  p << op.rhs() << ", " << op.acc();
  if (op.masks().size() == 2)
    p << ", " << op.masks();

  p.printOptionalAttrDict(op->getAttrs(), attrNames);
  p << " : " << op.lhs().getType() << ", " << op.rhs().getType() << " into "
    << op.getResultType();
}

static bool verifyDimMap(VectorType lhsType, VectorType rhsType,
                         const std::vector<std::pair<int64_t, int64_t>> &map) {
  for (auto &dimPair : map) {
    if (dimPair.first < 0 || dimPair.first >= lhsType.getRank() ||
        dimPair.second < 0 || dimPair.second >= rhsType.getRank() ||
        lhsType.getDimSize(dimPair.first) != rhsType.getDimSize(dimPair.second))
      return false;
  }
  return true;
}

static LogicalResult verifyOutputShape(
    ContractionOp op, VectorType lhsType, VectorType rhsType, Type accType,
    Type resType,
    const std::vector<std::pair<int64_t, int64_t>> &contractingDimMap,
    const std::vector<std::pair<int64_t, int64_t>> &batchDimMap) {
  DenseSet<int64_t> lhsContractingDimSet;
  DenseSet<int64_t> rhsContractingDimSet;
  for (auto &dimPair : contractingDimMap) {
    lhsContractingDimSet.insert(dimPair.first);
    rhsContractingDimSet.insert(dimPair.second);
  }
  DenseSet<int64_t> rhsBatchDimSet;
  for (auto &dimPair : batchDimMap)
    rhsBatchDimSet.insert(dimPair.second);

  // Add free and batch dimensions from 'lhsType' to 'expectedResultDims'.
  SmallVector<int64_t, 4> expectedResultDims;
  for (int64_t i = 0, e = lhsType.getRank(); i < e; ++i) {
    if (lhsContractingDimSet.count(i) > 0)
      continue;
    expectedResultDims.push_back(lhsType.getDimSize(i));
  }

  // Add free dimensions from 'rhsType' to 'expectedResultDims'.
  for (int64_t i = 0, e = rhsType.getRank(); i < e; ++i) {
    if (rhsContractingDimSet.count(i) > 0 || rhsBatchDimSet.count(i) > 0)
      continue;
    expectedResultDims.push_back(rhsType.getDimSize(i));
  }

  // Verify 'expectedResultDims'.
  if (expectedResultDims.empty()) {
    // No batch or free dimension implies a scalar result.
    if (resType.isa<VectorType>() || accType.isa<VectorType>())
      return op.emitOpError("invalid accumulator/result vector shape");
  } else {
    // At least one batch or free dimension implies a vector result.
    auto resVectorType = resType.dyn_cast<VectorType>();
    auto accVectorType = accType.dyn_cast<VectorType>();
    if (!resVectorType || !accVectorType)
      return op.emitOpError("invalid accumulator/result vector shape");

    // Infer expected result vector type. Lhs + rhs map and lhs + rhs vector
    // types fully define the result vector type. This assumes the affine maps
    // are well-formed, which must have been verified already.
    MLIRContext *ctx = op.getContext();
    AffineMap lhsMap = op.getIndexingMaps()[0];
    AffineMap rhsMap = op.getIndexingMaps()[1];
    SmallVector<AffineExpr, 4> extents(lhsMap.getNumInputs());
    for (auto pair :
         {std::make_pair(lhsType, lhsMap), std::make_pair(rhsType, rhsMap)}) {
      VectorType v = pair.first;
      auto map = pair.second;
      for (unsigned idx = 0, e = v.getRank(); idx < e; ++idx) {
        unsigned pos = map.getDimPosition(idx);
        if (!extents[pos])
          extents[pos] = getAffineConstantExpr(v.getShape()[idx], ctx);
      }
    }
    assert(llvm::all_of(extents, [](AffineExpr e) { return e; }) &&
           "expected extent along all dimensions.");

    AffineMap resMap = op.getIndexingMaps()[2];
    auto extentsMap = AffineMap::get(/*dimCount=*/extents.size(),
                                     /*symCount=*/0, extents, ctx);
    // Compose the resMap with the extentsMap, which is a constant map.
    AffineMap expectedMap = simplifyAffineMap(resMap.compose(extentsMap));
    assert(llvm::all_of(
               expectedMap.getResults(),
               [](AffineExpr e) { return e.isa<AffineConstantExpr>(); }) &&
           "expected constant extent along all dimensions.");
    // Extract the expected shape and build the type.
    auto expectedShape = llvm::to_vector<4>(
        llvm::map_range(expectedMap.getResults(), [](AffineExpr e) {
          return e.cast<AffineConstantExpr>().getValue();
        }));
    auto expected =
        VectorType::get(expectedShape, resVectorType.getElementType());
    if (resVectorType != expected || accVectorType != expected)
      return op.emitOpError(
                 "invalid accumulator/result vector shape, expected: ")
             << expected;
  }
  return success();
}

static LogicalResult verify(ContractionOp op) {
  auto lhsType = op.getLhsType();
  auto rhsType = op.getRhsType();
  auto accType = op.getAccType();
  auto resType = op.getResultType();

  // Verify that an indexing map was specified for each vector operand.
  if (op.indexing_maps().size() != 3)
    return op.emitOpError("expected an indexing map for each vector operand");

  // Verify that each index map has 'numIterators' inputs, no symbols, and
  // that the number of map outputs equals the rank of its associated
  // vector operand.
  unsigned numIterators = op.iterator_types().getValue().size();
  for (const auto &it : llvm::enumerate(op.indexing_maps())) {
    auto index = it.index();
    auto map = it.value().cast<AffineMapAttr>().getValue();
    if (map.getNumSymbols() != 0)
      return op.emitOpError("expected indexing map ")
             << index << " to have no symbols";
    auto vectorType = op.getOperand(index).getType().dyn_cast<VectorType>();
    unsigned rank = vectorType ? vectorType.getShape().size() : 0;
    // Verify that the map has the right number of inputs, outputs, and indices.
    // This also correctly accounts for (..) -> () for rank-0 results.
    if (map.getNumDims() != numIterators)
      return op.emitOpError("expected indexing map ")
             << index << " to have " << numIterators << " number of inputs";
    if (map.getNumResults() != rank)
      return op.emitOpError("expected indexing map ")
             << index << " to have " << rank << " number of outputs";
    if (!map.isProjectedPermutation())
      return op.emitOpError("expected indexing map ")
             << index << " to be a projected permutation of its inputs";
  }

  auto contractingDimMap = op.getContractingDimMap();
  auto batchDimMap = op.getBatchDimMap();

  // Verify at least one contracting dimension pair was specified.
  if (contractingDimMap.empty())
    return op.emitOpError("expected at least one contracting dimension pair");

  // Verify contracting dimension map was properly constructed.
  if (!verifyDimMap(lhsType, rhsType, contractingDimMap))
    return op.emitOpError("invalid contracting dimension map");

  // Verify batch dimension map was properly constructed.
  if (!verifyDimMap(lhsType, rhsType, batchDimMap))
    return op.emitOpError("invalid batch dimension map");

  // Verify 'accType' and 'resType' shape.
  if (failed(verifyOutputShape(op, lhsType, rhsType, accType, resType,
                               contractingDimMap, batchDimMap)))
    return failure();

  // Verify that either two vector masks are set or none are set.
  auto lhsMaskType = op.getLHSVectorMaskType();
  auto rhsMaskType = op.getRHSVectorMaskType();
  if ((lhsMaskType && !rhsMaskType) || (!lhsMaskType && rhsMaskType))
    return op.emitOpError("invalid number of vector masks specified");
  if (lhsMaskType && rhsMaskType) {
    // Verify mask rank == argument rank.
    if (lhsMaskType.getShape().size() != lhsType.getShape().size() ||
        rhsMaskType.getShape().size() != rhsType.getShape().size())
      return op.emitOpError("invalid vector mask rank");
  }

  // Verify supported combining kind.
  auto vectorType = resType.dyn_cast<VectorType>();
  auto elementType = vectorType ? vectorType.getElementType() : resType;
  if (!isSupportedCombiningKind(op.kind(), elementType))
    return op.emitOpError("unsupported contraction type");

  return success();
}

ArrayRef<StringRef> ContractionOp::getTraitAttrNames() {
  static constexpr StringRef names[3] = {getIndexingMapsAttrName(),
                                         getIteratorTypesAttrName(),
                                         ContractionOp::getKindAttrName()};
  return llvm::makeArrayRef(names);
}

static int64_t getResultIndex(AffineMap map, AffineExpr targetExpr) {
  for (int64_t i = 0, e = map.getNumResults(); i < e; ++i)
    if (targetExpr == map.getResult(i))
      return i;
  return -1;
}

static std::vector<std::pair<int64_t, int64_t>>
getDimMap(ArrayRef<AffineMap> indexingMaps, ArrayAttr iteratorTypes,
          StringRef targetIteratorTypeName, MLIRContext *context) {
  std::vector<std::pair<int64_t, int64_t>> dimMap;
  for (const auto &it : llvm::enumerate(iteratorTypes)) {
    auto iteratorTypeName = it.value().cast<StringAttr>().getValue();
    if (iteratorTypeName != targetIteratorTypeName)
      continue;
    // Search lhs/rhs map results for 'targetExpr'.
    auto targetExpr = getAffineDimExpr(it.index(), context);
    int64_t lhsDim = getResultIndex(indexingMaps[0], targetExpr);
    int64_t rhsDim = getResultIndex(indexingMaps[1], targetExpr);
    if (lhsDim >= 0 && rhsDim >= 0)
      dimMap.emplace_back(lhsDim, rhsDim);
  }
  return dimMap;
}

void ContractionOp::getIterationBounds(
    SmallVectorImpl<int64_t> &iterationBounds) {
  auto lhsShape = getLhsType().getShape();
  auto resVectorType = getResultType().dyn_cast<VectorType>();
  SmallVector<AffineMap, 4> indexingMaps(getIndexingMaps());
  SmallVector<int64_t, 2> iterationShape;
  for (const auto &it : llvm::enumerate(iterator_types())) {
    // Search lhs/rhs map results for 'targetExpr'.
    auto targetExpr = getAffineDimExpr(it.index(), getContext());
    auto iteratorTypeName = it.value().cast<StringAttr>().getValue();
    if (iteratorTypeName == getReductionIteratorTypeName()) {
      // Get reduction dim size from lhs shape (same size in rhsShape).
      int64_t lhsDimIndex = getResultIndex(indexingMaps[0], targetExpr);
      assert(lhsDimIndex >= 0);
      iterationBounds.push_back(lhsShape[lhsDimIndex]);
      continue;
    }
    // Get parallel dimension size from result shape.
    int64_t resDimIndex = getResultIndex(indexingMaps[2], targetExpr);
    assert(resDimIndex >= 0);
    assert(resVectorType != nullptr);
    iterationBounds.push_back(resVectorType.getShape()[resDimIndex]);
  }
}

void ContractionOp::getIterationIndexMap(
    std::vector<DenseMap<int64_t, int64_t>> &iterationIndexMap) {
  unsigned numMaps = indexing_maps().getValue().size();
  iterationIndexMap.resize(numMaps);
  for (const auto &it : llvm::enumerate(indexing_maps())) {
    auto index = it.index();
    auto map = it.value().cast<AffineMapAttr>().getValue();
    for (unsigned i = 0, e = map.getNumResults(); i < e; ++i) {
      auto dim = map.getResult(i).cast<AffineDimExpr>();
      iterationIndexMap[index][dim.getPosition()] = i;
    }
  }
}

std::vector<std::pair<int64_t, int64_t>> ContractionOp::getContractingDimMap() {
  SmallVector<AffineMap, 4> indexingMaps(getIndexingMaps());
  return getDimMap(indexingMaps, iterator_types(),
                   getReductionIteratorTypeName(), getContext());
}

std::vector<std::pair<int64_t, int64_t>> ContractionOp::getBatchDimMap() {
  SmallVector<AffineMap, 4> indexingMaps(getIndexingMaps());
  return getDimMap(indexingMaps, iterator_types(),
                   getParallelIteratorTypeName(), getContext());
}

SmallVector<AffineMap, 4> ContractionOp::getIndexingMaps() {
  return llvm::to_vector<4>(
      llvm::map_range(indexing_maps().getValue(), [](Attribute mapAttr) {
        return mapAttr.cast<AffineMapAttr>().getValue();
      }));
}

Optional<SmallVector<int64_t, 4>> ContractionOp::getShapeForUnroll() {
  SmallVector<int64_t, 4> shape;
  getIterationBounds(shape);
  return shape;
}

/// Return a fused vector::ContractionOp which represents a patterns such as:
///
/// ```mlir
///    %c0 = vector.constant 0: ...
///    %c = vector.contract %a, %b, %c0: ...
///    %e = add %c, %d: ...
/// ```
///
/// by:
///
/// ```mlir
///    %e = vector.contract %a, %b, %d: ...
/// ```
///
/// Return null if the canonicalization does not apply.
// TODO: This should be a folding of Add into Contract in core but while they
// live in different dialects, it is not possible without unnatural
// dependencies.
template <typename AddOpType>
struct CanonicalizeContractAdd : public OpRewritePattern<AddOpType> {
  using OpRewritePattern<AddOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOpType addOp,
                                PatternRewriter &rewriter) const override {
    auto canonicalize = [&](Value maybeContraction,
                            Value otherOperand) -> vector::ContractionOp {
      vector::ContractionOp contractionOp =
          dyn_cast_or_null<vector::ContractionOp>(
              maybeContraction.getDefiningOp());
      if (!contractionOp)
        return vector::ContractionOp();
      if (auto maybeZero = dyn_cast_or_null<arith::ConstantOp>(
              contractionOp.acc().getDefiningOp())) {
        if (maybeZero.getValue() ==
            rewriter.getZeroAttr(contractionOp.acc().getType())) {
          BlockAndValueMapping bvm;
          bvm.map(contractionOp.acc(), otherOperand);
          auto newContraction =
              cast<vector::ContractionOp>(rewriter.clone(*contractionOp, bvm));
          rewriter.replaceOp(addOp, newContraction.getResult());
          return newContraction;
        }
      }
      return vector::ContractionOp();
    };

    Value a = addOp->getOperand(0), b = addOp->getOperand(1);
    vector::ContractionOp contract = canonicalize(a, b);
    contract = contract ? contract : canonicalize(b, a);
    return contract ? success() : failure();
  }
};

void ContractionOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<CanonicalizeContractAdd<arith::AddIOp>,
              CanonicalizeContractAdd<arith::AddFOp>>(context);
}

//===----------------------------------------------------------------------===//
// ExtractElementOp
//===----------------------------------------------------------------------===//

void vector::ExtractElementOp::build(OpBuilder &builder, OperationState &result,
                                     Value source) {
  result.addOperands({source});
  result.addTypes(source.getType().cast<VectorType>().getElementType());
}

void vector::ExtractElementOp::build(OpBuilder &builder, OperationState &result,
                                     Value source, Value position) {
  result.addOperands({source, position});
  result.addTypes(source.getType().cast<VectorType>().getElementType());
}

static LogicalResult verify(vector::ExtractElementOp op) {
  VectorType vectorType = op.getVectorType();
  if (vectorType.getRank() == 0) {
    if (op.position())
      return op.emitOpError("expected position to be empty with 0-D vector");
    return success();
  }
  if (vectorType.getRank() != 1)
    return op.emitOpError("unexpected >1 vector rank");
  if (!op.position())
    return op.emitOpError("expected position for 1-D vector");
  return success();
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

static Type inferExtractOpResultType(VectorType vectorType,
                                     ArrayAttr position) {
  if (static_cast<int64_t>(position.size()) == vectorType.getRank())
    return vectorType.getElementType();
  return VectorType::get(vectorType.getShape().drop_front(position.size()),
                         vectorType.getElementType());
}

void vector::ExtractOp::build(OpBuilder &builder, OperationState &result,
                              Value source, ArrayRef<int64_t> position) {
  result.addOperands(source);
  auto positionAttr = getVectorSubscriptAttr(builder, position);
  result.addTypes(inferExtractOpResultType(source.getType().cast<VectorType>(),
                                           positionAttr));
  result.addAttribute(getPositionAttrName(), positionAttr);
}

// Convenience builder which assumes the values are constant indices.
void vector::ExtractOp::build(OpBuilder &builder, OperationState &result,
                              Value source, ValueRange position) {
  SmallVector<int64_t, 4> positionConstants =
      llvm::to_vector<4>(llvm::map_range(position, [](Value pos) {
        return pos.getDefiningOp<arith::ConstantIndexOp>().value();
      }));
  build(builder, result, source, positionConstants);
}

static void print(OpAsmPrinter &p, vector::ExtractOp op) {
  p << " " << op.vector() << op.position();
  p.printOptionalAttrDict(op->getAttrs(), {"position"});
  p << " : " << op.vector().getType();
}

static ParseResult parseExtractOp(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc attributeLoc, typeLoc;
  NamedAttrList attrs;
  OpAsmParser::OperandType vector;
  Type type;
  Attribute attr;
  if (parser.parseOperand(vector) || parser.getCurrentLocation(&attributeLoc) ||
      parser.parseAttribute(attr, "position", attrs) ||
      parser.parseOptionalAttrDict(attrs) ||
      parser.getCurrentLocation(&typeLoc) || parser.parseColonType(type))
    return failure();

  auto vectorType = type.dyn_cast<VectorType>();
  if (!vectorType)
    return parser.emitError(typeLoc, "expected vector type");

  auto positionAttr = attr.dyn_cast<ArrayAttr>();
  if (!positionAttr ||
      static_cast<int64_t>(positionAttr.size()) > vectorType.getRank())
    return parser.emitError(
        attributeLoc,
        "expected position attribute of rank smaller than vector rank");

  Type resType = inferExtractOpResultType(vectorType, positionAttr);
  result.attributes = attrs;
  return failure(parser.resolveOperand(vector, type, result.operands) ||
                 parser.addTypeToList(resType, result.types));
}

static LogicalResult verify(vector::ExtractOp op) {
  auto positionAttr = op.position().getValue();
  if (positionAttr.size() > static_cast<unsigned>(op.getVectorType().getRank()))
    return op.emitOpError(
        "expected position attribute of rank smaller than vector rank");
  for (const auto &en : llvm::enumerate(positionAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    if (!attr || attr.getInt() < 0 ||
        attr.getInt() >= op.getVectorType().getDimSize(en.index()))
      return op.emitOpError("expected position attribute #")
             << (en.index() + 1)
             << " to be a non-negative integer smaller than the corresponding "
                "vector dimension";
  }
  return success();
}

template <typename IntType>
static SmallVector<IntType, 4> extractVector(ArrayAttr arrayAttr) {
  return llvm::to_vector<4>(llvm::map_range(
      arrayAttr.getAsRange<IntegerAttr>(),
      [](IntegerAttr attr) { return static_cast<IntType>(attr.getInt()); }));
}

/// Fold the result of chains of ExtractOp in place by simply concatenating the
/// positions.
static LogicalResult foldExtractOpFromExtractChain(ExtractOp extractOp) {
  if (!extractOp.vector().getDefiningOp<ExtractOp>())
    return failure();

  SmallVector<int64_t, 4> globalPosition;
  ExtractOp currentOp = extractOp;
  auto extractedPos = extractVector<int64_t>(currentOp.position());
  globalPosition.append(extractedPos.rbegin(), extractedPos.rend());
  while (ExtractOp nextOp = currentOp.vector().getDefiningOp<ExtractOp>()) {
    currentOp = nextOp;
    auto extractedPos = extractVector<int64_t>(currentOp.position());
    globalPosition.append(extractedPos.rbegin(), extractedPos.rend());
  }
  extractOp.setOperand(currentOp.vector());
  // OpBuilder is only used as a helper to build an I64ArrayAttr.
  OpBuilder b(extractOp.getContext());
  std::reverse(globalPosition.begin(), globalPosition.end());
  extractOp->setAttr(ExtractOp::getPositionAttrName(),
                     b.getI64ArrayAttr(globalPosition));
  return success();
}

/// Fold the result of an ExtractOp in place when it comes from a TransposeOp.
static LogicalResult foldExtractOpFromTranspose(ExtractOp extractOp) {
  auto transposeOp = extractOp.vector().getDefiningOp<vector::TransposeOp>();
  if (!transposeOp)
    return failure();

  auto permutation = extractVector<unsigned>(transposeOp.transp());
  auto extractedPos = extractVector<int64_t>(extractOp.position());

  // If transposition permutation is larger than the ExtractOp, all minor
  // dimensions must be an identity for folding to occur. If not, individual
  // elements within the extracted value are transposed and this is not just a
  // simple folding.
  unsigned minorRank = permutation.size() - extractedPos.size();
  MLIRContext *ctx = extractOp.getContext();
  AffineMap permutationMap = AffineMap::getPermutationMap(permutation, ctx);
  AffineMap minorMap = permutationMap.getMinorSubMap(minorRank);
  if (minorMap && !minorMap.isMinorIdentity())
    return failure();

  //   %1 = transpose %0[x, y, z] : vector<axbxcxf32>
  //   %2 = extract %1[u, v] : vector<..xf32>
  // may turn into:
  //   %2 = extract %0[w, x] : vector<..xf32>
  // iff z == 2 and [w, x] = [x, y]^-1 o [u, v] here o denotes composition and
  // -1 denotes the inverse.
  permutationMap = permutationMap.getMajorSubMap(extractedPos.size());
  // The major submap has fewer results but the same number of dims. To compose
  // cleanly, we need to drop dims to form a "square matrix". This is possible
  // because:
  //   (a) this is a permutation map and
  //   (b) the minor map has already been checked to be identity.
  // Therefore, the major map cannot contain dims of position greater or equal
  // than the number of results.
  assert(llvm::all_of(permutationMap.getResults(),
                      [&](AffineExpr e) {
                        auto dim = e.dyn_cast<AffineDimExpr>();
                        return dim && dim.getPosition() <
                                          permutationMap.getNumResults();
                      }) &&
         "Unexpected map results depend on higher rank positions");
  // Project on the first domain dimensions to allow composition.
  permutationMap = AffineMap::get(permutationMap.getNumResults(), 0,
                                  permutationMap.getResults(), ctx);

  extractOp.setOperand(transposeOp.vector());
  // Compose the inverse permutation map with the extractedPos.
  auto newExtractedPos =
      inversePermutation(permutationMap).compose(extractedPos);
  // OpBuilder is only used as a helper to build an I64ArrayAttr.
  OpBuilder b(extractOp.getContext());
  extractOp->setAttr(ExtractOp::getPositionAttrName(),
                     b.getI64ArrayAttr(newExtractedPos));

  return success();
}

/// Fold an ExtractOp that is fed by a chain of InsertOps and TransposeOps. The
/// result is always the input to some InsertOp.
static Value foldExtractOpFromInsertChainAndTranspose(ExtractOp extractOp) {
  MLIRContext *context = extractOp.getContext();
  AffineMap permutationMap;
  auto extractedPos = extractVector<unsigned>(extractOp.position());
  // Walk back a chain of InsertOp/TransposeOp until we hit a match.
  // Compose TransposeOp permutations as we walk back.
  auto insertOp = extractOp.vector().getDefiningOp<vector::InsertOp>();
  auto transposeOp = extractOp.vector().getDefiningOp<vector::TransposeOp>();
  while (insertOp || transposeOp) {
    if (transposeOp) {
      // If it is transposed, compose the map and iterate.
      auto permutation = extractVector<unsigned>(transposeOp.transp());
      AffineMap newMap = AffineMap::getPermutationMap(permutation, context);
      if (!permutationMap)
        permutationMap = newMap;
      else if (newMap.getNumInputs() != permutationMap.getNumResults())
        return Value();
      else
        permutationMap = newMap.compose(permutationMap);
      // Compute insert/transpose for the next iteration.
      Value transposed = transposeOp.vector();
      insertOp = transposed.getDefiningOp<vector::InsertOp>();
      transposeOp = transposed.getDefiningOp<vector::TransposeOp>();
      continue;
    }

    assert(insertOp);
    Value insertionDest = insertOp.dest();
    // If it is inserted into, either the position matches and we have a
    // successful folding; or we iterate until we run out of
    // InsertOp/TransposeOp. This is because `vector.insert %scalar, %vector`
    // produces a new vector with 1 modified value/slice in exactly the static
    // position we need to match.
    auto insertedPos = extractVector<unsigned>(insertOp.position());
    // Trivial permutations are solved with position equality checks.
    if (!permutationMap || permutationMap.isIdentity()) {
      if (extractedPos == insertedPos)
        return insertOp.source();
      // Fallthrough: if the position does not match, just skip to the next
      // producing `vector.insert` / `vector.transpose`.
      // Compute insert/transpose for the next iteration.
      insertOp = insertionDest.getDefiningOp<vector::InsertOp>();
      transposeOp = insertionDest.getDefiningOp<vector::TransposeOp>();
      continue;
    }

    // More advanced permutations require application of the permutation.
    // However, the rank of `insertedPos` may be different from that of the
    // `permutationMap`. To support such case, we need to:
    //   1. apply on the `insertedPos.size()` major dimensions
    //   2. check the other dimensions of the permutation form a minor identity.
    assert(permutationMap.isPermutation() && "expected a permutation");
    if (insertedPos.size() == extractedPos.size()) {
      bool fold = true;
      for (unsigned idx = 0, sz = extractedPos.size(); idx < sz; ++idx) {
        auto pos = permutationMap.getDimPosition(idx);
        if (pos >= sz || insertedPos[pos] != extractedPos[idx]) {
          fold = false;
          break;
        }
      }
      if (fold) {
        assert(permutationMap.getNumResults() >= insertedPos.size() &&
               "expected map of rank larger than insert indexing");
        unsigned minorRank =
            permutationMap.getNumResults() - insertedPos.size();
        AffineMap minorMap = permutationMap.getMinorSubMap(minorRank);
        if (!minorMap || minorMap.isMinorIdentity())
          return insertOp.source();
      }
    }

    // If we haven't found a match, just continue to the next producing
    // `vector.insert` / `vector.transpose`.
    // Compute insert/transpose for the next iteration.
    insertOp = insertionDest.getDefiningOp<vector::InsertOp>();
    transposeOp = insertionDest.getDefiningOp<vector::TransposeOp>();
  }
  return Value();
}

/// Fold extractOp with scalar result coming from BroadcastOp or SplatOp.
static Value foldExtractFromBroadcast(ExtractOp extractOp) {
  Operation *defOp = extractOp.vector().getDefiningOp();
  if (!defOp || !isa<vector::BroadcastOp, SplatOp>(defOp))
    return Value();
  Value source = defOp->getOperand(0);
  if (extractOp.getType() == source.getType())
    return source;
  auto getRank = [](Type type) {
    return type.isa<VectorType>() ? type.cast<VectorType>().getRank() : 0;
  };
  unsigned broadcastSrcRank = getRank(source.getType());
  unsigned extractResultRank = getRank(extractOp.getType());
  if (extractResultRank < broadcastSrcRank) {
    auto extractPos = extractVector<int64_t>(extractOp.position());
    unsigned rankDiff = broadcastSrcRank - extractResultRank;
    extractPos.erase(
        extractPos.begin(),
        std::next(extractPos.begin(), extractPos.size() - rankDiff));
    extractOp.setOperand(source);
    // OpBuilder is only used as a helper to build an I64ArrayAttr.
    OpBuilder b(extractOp.getContext());
    extractOp->setAttr(ExtractOp::getPositionAttrName(),
                       b.getI64ArrayAttr(extractPos));
    return extractOp.getResult();
  }
  return Value();
}

// Fold extractOp with source coming from ShapeCast op.
static Value foldExtractFromShapeCast(ExtractOp extractOp) {
  auto shapeCastOp = extractOp.vector().getDefiningOp<vector::ShapeCastOp>();
  if (!shapeCastOp)
    return Value();
  // Get the nth dimension size starting from lowest dimension.
  auto getDimReverse = [](VectorType type, int64_t n) {
    return type.getShape().take_back(n + 1).front();
  };
  int64_t destinationRank =
      extractOp.getType().isa<VectorType>()
          ? extractOp.getType().cast<VectorType>().getRank()
          : 0;
  if (destinationRank > shapeCastOp.getSourceVectorType().getRank())
    return Value();
  if (destinationRank > 0) {
    auto destinationType = extractOp.getResult().getType().cast<VectorType>();
    for (int64_t i = 0; i < destinationRank; i++) {
      // The lowest dimension of of the destination must match the lowest
      // dimension of the shapecast op source.
      // TODO: This case could be support in a canonicalization pattern.
      if (getDimReverse(shapeCastOp.getSourceVectorType(), i) !=
          getDimReverse(destinationType, i))
        return Value();
    }
  }
  // Extract the strides associated with the extract op vector source. Then use
  // this to calculate a linearized position for the extract.
  auto extractedPos = extractVector<int64_t>(extractOp.position());
  std::reverse(extractedPos.begin(), extractedPos.end());
  SmallVector<int64_t, 4> strides;
  int64_t stride = 1;
  for (int64_t i = 0, e = extractedPos.size(); i < e; i++) {
    strides.push_back(stride);
    stride *= getDimReverse(extractOp.getVectorType(), i + destinationRank);
  }

  int64_t position = linearize(extractedPos, strides);
  // Then extract the strides associated to the shapeCast op vector source and
  // delinearize the position using those strides.
  SmallVector<int64_t, 4> newStrides;
  int64_t numDimension =
      shapeCastOp.getSourceVectorType().getRank() - destinationRank;
  stride = 1;
  for (int64_t i = 0; i < numDimension; i++) {
    newStrides.push_back(stride);
    stride *=
        getDimReverse(shapeCastOp.getSourceVectorType(), i + destinationRank);
  }
  std::reverse(newStrides.begin(), newStrides.end());
  SmallVector<int64_t, 4> newPosition = delinearize(newStrides, position);
  // OpBuilder is only used as a helper to build an I64ArrayAttr.
  OpBuilder b(extractOp.getContext());
  extractOp->setAttr(ExtractOp::getPositionAttrName(),
                     b.getI64ArrayAttr(newPosition));
  extractOp.setOperand(shapeCastOp.source());
  return extractOp.getResult();
}

OpFoldResult ExtractOp::fold(ArrayRef<Attribute>) {
  if (position().empty())
    return vector();
  if (succeeded(foldExtractOpFromExtractChain(*this)))
    return getResult();
  if (succeeded(foldExtractOpFromTranspose(*this)))
    return getResult();
  if (auto val = foldExtractOpFromInsertChainAndTranspose(*this))
    return val;
  if (auto val = foldExtractFromBroadcast(*this))
    return val;
  if (auto val = foldExtractFromShapeCast(*this))
    return val;
  return OpFoldResult();
}

namespace {

// Pattern to rewrite a ExtractOp(Broadcast) -> Broadcast.
class ExtractOpFromBroadcast final : public OpRewritePattern<ExtractOp> {
public:
  using OpRewritePattern<ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    Operation *defOp = extractOp.vector().getDefiningOp();
    if (!defOp || !isa<vector::BroadcastOp, SplatOp>(defOp))
      return failure();
    Value source = defOp->getOperand(0);
    if (extractOp.getType() == source.getType())
      return failure();
    auto getRank = [](Type type) {
      return type.isa<VectorType>() ? type.cast<VectorType>().getRank() : 0;
    };
    unsigned broadcastSrcRank = getRank(source.getType());
    unsigned extractResultRank = getRank(extractOp.getType());
    // We only consider the case where the rank of the source is smaller than
    // the rank of the extract dst. The other cases are handled in the folding
    // patterns.
    if (extractResultRank <= broadcastSrcRank)
      return failure();
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        extractOp, extractOp.getType(), source);
    return success();
  }
};

// Pattern to rewrite a ExtractOp(splat ConstantOp) -> ConstantOp.
class ExtractOpConstantFolder final : public OpRewritePattern<ExtractOp> {
public:
  using OpRewritePattern<ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    // Return if 'extractStridedSliceOp' operand is not defined by a
    // ConstantOp.
    auto constantOp = extractOp.vector().getDefiningOp<arith::ConstantOp>();
    if (!constantOp)
      return failure();
    auto dense = constantOp.getValue().dyn_cast<SplatElementsAttr>();
    if (!dense)
      return failure();
    Attribute newAttr = dense.getSplatValue<Attribute>();
    if (auto vecDstType = extractOp.getType().dyn_cast<VectorType>())
      newAttr = DenseElementsAttr::get(vecDstType, newAttr);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(extractOp, newAttr);
    return success();
  }
};

} // namespace

void ExtractOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ExtractOpConstantFolder, ExtractOpFromBroadcast>(context);
}

static void populateFromInt64AttrArray(ArrayAttr arrayAttr,
                                       SmallVectorImpl<int64_t> &results) {
  for (auto attr : arrayAttr)
    results.push_back(attr.cast<IntegerAttr>().getInt());
}

//===----------------------------------------------------------------------===//
// ExtractMapOp
//===----------------------------------------------------------------------===//

void ExtractMapOp::build(OpBuilder &builder, OperationState &result,
                         Value vector, ValueRange ids,
                         ArrayRef<int64_t> multiplicity,
                         AffineMap permutationMap) {
  assert(ids.size() == multiplicity.size() &&
         ids.size() == permutationMap.getNumResults());
  assert(permutationMap.isProjectedPermutation());
  VectorType type = vector.getType().cast<VectorType>();
  SmallVector<int64_t, 4> newShape(type.getShape().begin(),
                                   type.getShape().end());
  for (unsigned i = 0, e = permutationMap.getNumResults(); i < e; i++) {
    AffineExpr expr = permutationMap.getResult(i);
    auto dim = expr.cast<AffineDimExpr>();
    newShape[dim.getPosition()] = newShape[dim.getPosition()] / multiplicity[i];
  }
  VectorType resultType = VectorType::get(newShape, type.getElementType());
  ExtractMapOp::build(builder, result, resultType, vector, ids);
}

static LogicalResult verify(ExtractMapOp op) {
  if (op.getSourceVectorType().getRank() != op.getResultType().getRank())
    return op.emitOpError(
        "expected source and destination vectors of same rank");
  unsigned numId = 0;
  for (unsigned i = 0, e = op.getSourceVectorType().getRank(); i < e; ++i) {
    if (op.getSourceVectorType().getDimSize(i) %
            op.getResultType().getDimSize(i) !=
        0)
      return op.emitOpError("source vector dimensions must be a multiple of "
                            "destination vector dimensions");
    if (op.getSourceVectorType().getDimSize(i) !=
        op.getResultType().getDimSize(i))
      numId++;
  }
  if (numId != op.ids().size())
    return op.emitOpError("expected number of ids must match the number of "
                          "dimensions distributed");
  return success();
}

OpFoldResult ExtractMapOp::fold(ArrayRef<Attribute> operands) {
  auto insert = vector().getDefiningOp<vector::InsertMapOp>();
  if (insert == nullptr || getType() != insert.vector().getType() ||
      ids() != insert.ids())
    return {};
  return insert.vector();
}

void ExtractMapOp::getMultiplicity(SmallVectorImpl<int64_t> &multiplicity) {
  assert(multiplicity.empty());
  for (unsigned i = 0, e = getSourceVectorType().getRank(); i < e; i++) {
    if (getSourceVectorType().getDimSize(i) != getResultType().getDimSize(i))
      multiplicity.push_back(getSourceVectorType().getDimSize(i) /
                             getResultType().getDimSize(i));
  }
}

template <typename MapOp>
AffineMap calculateImplicitMap(MapOp op) {
  SmallVector<AffineExpr, 4> perm;
  // Check which dimension have a multiplicity greater than 1 and associated
  // them to the IDs in order.
  for (unsigned i = 0, e = op.getSourceVectorType().getRank(); i < e; i++) {
    if (op.getSourceVectorType().getDimSize(i) !=
        op.getResultType().getDimSize(i))
      perm.push_back(getAffineDimExpr(i, op.getContext()));
  }
  auto map = AffineMap::get(op.getSourceVectorType().getRank(), 0, perm,
                            op.getContext());
  return map;
}

AffineMap ExtractMapOp::map() { return calculateImplicitMap(*this); }

//===----------------------------------------------------------------------===//
// FmaOp
//===----------------------------------------------------------------------===//

Optional<SmallVector<int64_t, 4>> FMAOp::getShapeForUnroll() {
  return llvm::to_vector<4>(getVectorType().getShape());
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

BroadcastableToResult
mlir::vector::isBroadcastableTo(Type srcType, VectorType dstVectorType,
                                std::pair<int, int> *mismatchingDims) {
  // Broadcast scalar to vector of the same element type.
  if (srcType.isIntOrIndexOrFloat() && dstVectorType &&
      getElementTypeOrSelf(srcType) == getElementTypeOrSelf(dstVectorType))
    return BroadcastableToResult::Success;
  // From now on, only vectors broadcast.
  VectorType srcVectorType = srcType.dyn_cast<VectorType>();
  if (!srcVectorType)
    return BroadcastableToResult::SourceTypeNotAVector;

  int64_t srcRank = srcVectorType.getRank();
  int64_t dstRank = dstVectorType.getRank();
  if (srcRank > dstRank)
    return BroadcastableToResult::SourceRankHigher;
  // Source has an exact match or singleton value for all trailing dimensions
  // (all leading dimensions are simply duplicated).
  int64_t lead = dstRank - srcRank;
  for (int64_t r = 0; r < srcRank; ++r) {
    int64_t srcDim = srcVectorType.getDimSize(r);
    int64_t dstDim = dstVectorType.getDimSize(lead + r);
    if (srcDim != 1 && srcDim != dstDim) {
      if (mismatchingDims) {
        mismatchingDims->first = srcDim;
        mismatchingDims->second = dstDim;
      }
      return BroadcastableToResult::DimensionMismatch;
    }
  }

  return BroadcastableToResult::Success;
}

static LogicalResult verify(BroadcastOp op) {
  std::pair<int, int> mismatchingDims;
  BroadcastableToResult res = isBroadcastableTo(
      op.getSourceType(), op.getVectorType(), &mismatchingDims);
  if (res == BroadcastableToResult::Success)
    return success();
  if (res == BroadcastableToResult::SourceRankHigher)
    return op.emitOpError("source rank higher than destination rank");
  if (res == BroadcastableToResult::DimensionMismatch)
    return op.emitOpError("dimension mismatch (")
           << mismatchingDims.first << " vs. " << mismatchingDims.second << ")";
  if (res == BroadcastableToResult::SourceTypeNotAVector)
    return op.emitOpError("source type is not a vector");
  llvm_unreachable("unexpected vector.broadcast op error");
}

OpFoldResult BroadcastOp::fold(ArrayRef<Attribute> operands) {
  if (getSourceType() == getVectorType())
    return source();
  if (!operands[0])
    return {};
  auto vectorType = getVectorType();
  if (operands[0].getType().isIntOrIndexOrFloat())
    return DenseElementsAttr::get(vectorType, operands[0]);
  if (auto attr = operands[0].dyn_cast<SplatElementsAttr>())
    return DenseElementsAttr::get(vectorType, attr.getSplatValue<Attribute>());
  return {};
}

namespace {

// Fold broadcast1(broadcast2(x)) into broadcast1(x).
struct BroadcastFolder : public OpRewritePattern<BroadcastOp> {
  using OpRewritePattern<BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastOp broadcastOp,
                                PatternRewriter &rewriter) const override {
    auto srcBroadcast = broadcastOp.source().getDefiningOp<BroadcastOp>();
    if (!srcBroadcast)
      return failure();
    rewriter.replaceOpWithNewOp<BroadcastOp>(
        broadcastOp, broadcastOp.getVectorType(), srcBroadcast.source());
    return success();
  }
};
} // namespace

void BroadcastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  // BroadcastToShapeCast is not a default canonicalization, it is opt-in by
  // calling `populateCastAwayVectorLeadingOneDimPatterns`
  results.add<BroadcastFolder>(context);
}

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//

void ShuffleOp::build(OpBuilder &builder, OperationState &result, Value v1,
                      Value v2, ArrayRef<int64_t> mask) {
  result.addOperands({v1, v2});
  auto maskAttr = getVectorSubscriptAttr(builder, mask);
  auto v1Type = v1.getType().cast<VectorType>();
  auto shape = llvm::to_vector<4>(v1Type.getShape());
  shape[0] = mask.size();
  result.addTypes(VectorType::get(shape, v1Type.getElementType()));
  result.addAttribute(getMaskAttrName(), maskAttr);
}

static void print(OpAsmPrinter &p, ShuffleOp op) {
  p << " " << op.v1() << ", " << op.v2() << " " << op.mask();
  p.printOptionalAttrDict(op->getAttrs(), {ShuffleOp::getMaskAttrName()});
  p << " : " << op.v1().getType() << ", " << op.v2().getType();
}

static LogicalResult verify(ShuffleOp op) {
  VectorType resultType = op.getVectorType();
  VectorType v1Type = op.getV1VectorType();
  VectorType v2Type = op.getV2VectorType();
  // Verify ranks.
  int64_t resRank = resultType.getRank();
  int64_t v1Rank = v1Type.getRank();
  int64_t v2Rank = v2Type.getRank();
  if (resRank != v1Rank || v1Rank != v2Rank)
    return op.emitOpError("rank mismatch");
  // Verify all but leading dimension sizes.
  for (int64_t r = 1; r < v1Rank; ++r) {
    int64_t resDim = resultType.getDimSize(r);
    int64_t v1Dim = v1Type.getDimSize(r);
    int64_t v2Dim = v2Type.getDimSize(r);
    if (resDim != v1Dim || v1Dim != v2Dim)
      return op.emitOpError("dimension mismatch");
  }
  // Verify mask length.
  auto maskAttr = op.mask().getValue();
  int64_t maskLength = maskAttr.size();
  if (maskLength != resultType.getDimSize(0))
    return op.emitOpError("mask length mismatch");
  // Verify all indices.
  int64_t indexSize = v1Type.getDimSize(0) + v2Type.getDimSize(0);
  for (const auto &en : llvm::enumerate(maskAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    if (!attr || attr.getInt() < 0 || attr.getInt() >= indexSize)
      return op.emitOpError("mask index #")
             << (en.index() + 1) << " out of range";
  }
  return success();
}

static ParseResult parseShuffleOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType v1, v2;
  Attribute attr;
  VectorType v1Type, v2Type;
  if (parser.parseOperand(v1) || parser.parseComma() ||
      parser.parseOperand(v2) ||
      parser.parseAttribute(attr, ShuffleOp::getMaskAttrName(),
                            result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(v1Type) || parser.parseComma() ||
      parser.parseType(v2Type) ||
      parser.resolveOperand(v1, v1Type, result.operands) ||
      parser.resolveOperand(v2, v2Type, result.operands))
    return failure();
  // Construct resulting type: leading dimension matches mask length,
  // all trailing dimensions match the operands.
  auto maskAttr = attr.dyn_cast<ArrayAttr>();
  if (!maskAttr)
    return parser.emitError(parser.getNameLoc(), "missing mask attribute");
  int64_t maskLength = maskAttr.size();
  if (maskLength <= 0)
    return parser.emitError(parser.getNameLoc(), "invalid mask length");
  int64_t v1Rank = v1Type.getRank();
  SmallVector<int64_t, 4> shape;
  shape.reserve(v1Rank);
  shape.push_back(maskLength);
  for (int64_t r = 1; r < v1Rank; ++r)
    shape.push_back(v1Type.getDimSize(r));
  VectorType resType = VectorType::get(shape, v1Type.getElementType());
  parser.addTypeToList(resType, result.types);
  return success();
}

//===----------------------------------------------------------------------===//
// InsertElementOp
//===----------------------------------------------------------------------===//

void InsertElementOp::build(OpBuilder &builder, OperationState &result,
                            Value source, Value dest) {
  result.addOperands({source, dest});
  result.addTypes(dest.getType());
}

void InsertElementOp::build(OpBuilder &builder, OperationState &result,
                            Value source, Value dest, Value position) {
  result.addOperands({source, dest, position});
  result.addTypes(dest.getType());
}

static LogicalResult verify(InsertElementOp op) {
  auto dstVectorType = op.getDestVectorType();
  if (dstVectorType.getRank() == 0) {
    if (op.position())
      return op.emitOpError("expected position to be empty with 0-D vector");
    return success();
  }
  if (dstVectorType.getRank() != 1)
    return op.emitOpError("unexpected >1 vector rank");
  if (!op.position())
    return op.emitOpError("expected position for 1-D vector");
  return success();
}

//===----------------------------------------------------------------------===//
// InsertOp
//===----------------------------------------------------------------------===//

void InsertOp::build(OpBuilder &builder, OperationState &result, Value source,
                     Value dest, ArrayRef<int64_t> position) {
  result.addOperands({source, dest});
  auto positionAttr = getVectorSubscriptAttr(builder, position);
  result.addTypes(dest.getType());
  result.addAttribute(getPositionAttrName(), positionAttr);
}

// Convenience builder which assumes the values are constant indices.
void InsertOp::build(OpBuilder &builder, OperationState &result, Value source,
                     Value dest, ValueRange position) {
  SmallVector<int64_t, 4> positionConstants =
      llvm::to_vector<4>(llvm::map_range(position, [](Value pos) {
        return pos.getDefiningOp<arith::ConstantIndexOp>().value();
      }));
  build(builder, result, source, dest, positionConstants);
}

static LogicalResult verify(InsertOp op) {
  auto positionAttr = op.position().getValue();
  auto destVectorType = op.getDestVectorType();
  if (positionAttr.size() > static_cast<unsigned>(destVectorType.getRank()))
    return op.emitOpError(
        "expected position attribute of rank smaller than dest vector rank");
  auto srcVectorType = op.getSourceType().dyn_cast<VectorType>();
  if (srcVectorType &&
      (static_cast<unsigned>(srcVectorType.getRank()) + positionAttr.size() !=
       static_cast<unsigned>(destVectorType.getRank())))
    return op.emitOpError("expected position attribute rank + source rank to "
                          "match dest vector rank");
  if (!srcVectorType &&
      (positionAttr.size() != static_cast<unsigned>(destVectorType.getRank())))
    return op.emitOpError(
        "expected position attribute rank to match the dest vector rank");
  for (const auto &en : llvm::enumerate(positionAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    if (!attr || attr.getInt() < 0 ||
        attr.getInt() >= destVectorType.getDimSize(en.index()))
      return op.emitOpError("expected position attribute #")
             << (en.index() + 1)
             << " to be a non-negative integer smaller than the corresponding "
                "dest vector dimension";
  }
  return success();
}

namespace {

// If insertOp is only inserting unit dimensions it can be transformed to a
// broadcast.
class InsertToBroadcast final : public OpRewritePattern<InsertOp> {
public:
  using OpRewritePattern<InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp insertOp,
                                PatternRewriter &rewriter) const override {
    auto srcVecType = insertOp.getSourceType().dyn_cast<VectorType>();
    if (!srcVecType || insertOp.getDestVectorType().getNumElements() !=
                           srcVecType.getNumElements())
      return failure();
    rewriter.replaceOpWithNewOp<BroadcastOp>(
        insertOp, insertOp.getDestVectorType(), insertOp.source());
    return success();
  }
};

} // namespace

void InsertOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<InsertToBroadcast, BroadcastFolder>(context);
}

// Eliminates insert operations that produce values identical to their source
// value. This happens when the source and destination vectors have identical
// sizes.
OpFoldResult vector::InsertOp::fold(ArrayRef<Attribute> operands) {
  if (position().empty())
    return source();
  return {};
}

//===----------------------------------------------------------------------===//
// InsertMapOp
//===----------------------------------------------------------------------===//

void InsertMapOp::build(OpBuilder &builder, OperationState &result,
                        Value vector, Value dest, ValueRange ids) {
  InsertMapOp::build(builder, result, dest.getType(), vector, dest, ids);
}

static LogicalResult verify(InsertMapOp op) {
  if (op.getSourceVectorType().getRank() != op.getResultType().getRank())
    return op.emitOpError(
        "expected source and destination vectors of same rank");
  unsigned numId = 0;
  for (unsigned i = 0, e = op.getResultType().getRank(); i < e; i++) {
    if (op.getResultType().getDimSize(i) %
            op.getSourceVectorType().getDimSize(i) !=
        0)
      return op.emitOpError(
          "destination vector size must be a multiple of source vector size");
    if (op.getResultType().getDimSize(i) !=
        op.getSourceVectorType().getDimSize(i))
      numId++;
  }
  if (numId != op.ids().size())
    return op.emitOpError("expected number of ids must match the number of "
                          "dimensions distributed");
  return success();
}

AffineMap InsertMapOp::map() { return calculateImplicitMap(*this); }

//===----------------------------------------------------------------------===//
// InsertStridedSliceOp
//===----------------------------------------------------------------------===//

void InsertStridedSliceOp::build(OpBuilder &builder, OperationState &result,
                                 Value source, Value dest,
                                 ArrayRef<int64_t> offsets,
                                 ArrayRef<int64_t> strides) {
  result.addOperands({source, dest});
  auto offsetsAttr = getVectorSubscriptAttr(builder, offsets);
  auto stridesAttr = getVectorSubscriptAttr(builder, strides);
  result.addTypes(dest.getType());
  result.addAttribute(getOffsetsAttrName(), offsetsAttr);
  result.addAttribute(getStridesAttrName(), stridesAttr);
}

// TODO: Should be moved to Tablegen Confined attributes.
template <typename OpType>
static LogicalResult isIntegerArrayAttrSmallerThanShape(OpType op,
                                                        ArrayAttr arrayAttr,
                                                        ArrayRef<int64_t> shape,
                                                        StringRef attrName) {
  if (arrayAttr.size() > shape.size())
    return op.emitOpError("expected ")
           << attrName << " attribute of rank smaller than vector rank";
  return success();
}

// Returns true if all integers in `arrayAttr` are in the half-open [min, max}
// interval. If `halfOpen` is true then the admissible interval is [min, max).
// Otherwise, the admissible interval is [min, max].
template <typename OpType>
static LogicalResult
isIntegerArrayAttrConfinedToRange(OpType op, ArrayAttr arrayAttr, int64_t min,
                                  int64_t max, StringRef attrName,
                                  bool halfOpen = true) {
  for (auto attr : arrayAttr) {
    auto val = attr.cast<IntegerAttr>().getInt();
    auto upper = max;
    if (!halfOpen)
      upper += 1;
    if (val < min || val >= upper)
      return op.emitOpError("expected ") << attrName << " to be confined to ["
                                         << min << ", " << upper << ")";
  }
  return success();
}

// Returns true if all integers in `arrayAttr` are in the half-open [min, max}
// interval. If `halfOpen` is true then the admissible interval is [min, max).
// Otherwise, the admissible interval is [min, max].
template <typename OpType>
static LogicalResult
isIntegerArrayAttrConfinedToShape(OpType op, ArrayAttr arrayAttr,
                                  ArrayRef<int64_t> shape, StringRef attrName,
                                  bool halfOpen = true, int64_t min = 0) {
  assert(arrayAttr.size() <= shape.size());
  unsigned index = 0;
  for (auto it : llvm::zip(arrayAttr, shape)) {
    auto val = std::get<0>(it).cast<IntegerAttr>().getInt();
    auto max = std::get<1>(it);
    if (!halfOpen)
      max += 1;
    if (val < min || val >= max)
      return op.emitOpError("expected ")
             << attrName << " dimension " << index << " to be confined to ["
             << min << ", " << max << ")";
    ++index;
  }
  return success();
}

// Returns true if all integers in `arrayAttr` are in the interval [min, max}.
// interval. If `halfOpen` is true then the admissible interval is [min, max).
// Otherwise, the admissible interval is [min, max].
template <typename OpType>
static LogicalResult isSumOfIntegerArrayAttrConfinedToShape(
    OpType op, ArrayAttr arrayAttr1, ArrayAttr arrayAttr2,
    ArrayRef<int64_t> shape, StringRef attrName1, StringRef attrName2,
    bool halfOpen = true, int64_t min = 1) {
  assert(arrayAttr1.size() <= shape.size());
  assert(arrayAttr2.size() <= shape.size());
  unsigned index = 0;
  for (auto it : llvm::zip(arrayAttr1, arrayAttr2, shape)) {
    auto val1 = std::get<0>(it).cast<IntegerAttr>().getInt();
    auto val2 = std::get<1>(it).cast<IntegerAttr>().getInt();
    auto max = std::get<2>(it);
    if (!halfOpen)
      max += 1;
    if (val1 + val2 < 0 || val1 + val2 >= max)
      return op.emitOpError("expected sum(")
             << attrName1 << ", " << attrName2 << ") dimension " << index
             << " to be confined to [" << min << ", " << max << ")";
    ++index;
  }
  return success();
}

static ArrayAttr makeI64ArrayAttr(ArrayRef<int64_t> values,
                                  MLIRContext *context) {
  auto attrs = llvm::map_range(values, [context](int64_t v) -> Attribute {
    return IntegerAttr::get(IntegerType::get(context, 64), APInt(64, v));
  });
  return ArrayAttr::get(context, llvm::to_vector<8>(attrs));
}

static LogicalResult verify(InsertStridedSliceOp op) {
  auto sourceVectorType = op.getSourceVectorType();
  auto destVectorType = op.getDestVectorType();
  auto offsets = op.offsets();
  auto strides = op.strides();
  if (offsets.size() != static_cast<unsigned>(destVectorType.getRank()))
    return op.emitOpError(
        "expected offsets of same size as destination vector rank");
  if (strides.size() != static_cast<unsigned>(sourceVectorType.getRank()))
    return op.emitOpError(
        "expected strides of same size as source vector rank");
  if (sourceVectorType.getRank() > destVectorType.getRank())
    return op.emitOpError(
        "expected source rank to be smaller than destination rank");

  auto sourceShape = sourceVectorType.getShape();
  auto destShape = destVectorType.getShape();
  SmallVector<int64_t, 4> sourceShapeAsDestShape(
      destShape.size() - sourceShape.size(), 0);
  sourceShapeAsDestShape.append(sourceShape.begin(), sourceShape.end());
  auto offName = InsertStridedSliceOp::getOffsetsAttrName();
  auto stridesName = InsertStridedSliceOp::getStridesAttrName();
  if (failed(
          isIntegerArrayAttrConfinedToShape(op, offsets, destShape, offName)) ||
      failed(isIntegerArrayAttrConfinedToRange(op, strides, 1, 1, stridesName,
                                               /*halfOpen=*/false)) ||
      failed(isSumOfIntegerArrayAttrConfinedToShape(
          op, offsets,
          makeI64ArrayAttr(sourceShapeAsDestShape, op.getContext()), destShape,
          offName, "source vector shape",
          /*halfOpen=*/false, /*min=*/1)))
    return failure();

  return success();
}

OpFoldResult InsertStridedSliceOp::fold(ArrayRef<Attribute> operands) {
  if (getSourceVectorType() == getDestVectorType())
    return source();
  return {};
}

//===----------------------------------------------------------------------===//
// OuterProductOp
//===----------------------------------------------------------------------===//

/// Build an op without mask, use the type of `acc` as the return type.
void OuterProductOp::build(OpBuilder &builder, OperationState &result,
                           Value lhs, Value rhs, Value acc) {
  result.addOperands({lhs, rhs, acc});
  result.addTypes(acc.getType());
}

static void print(OpAsmPrinter &p, OuterProductOp op) {
  p << " " << op.lhs() << ", " << op.rhs();
  if (!op.acc().empty()) {
    p << ", " << op.acc();
    p.printOptionalAttrDict(op->getAttrs());
  }
  p << " : " << op.lhs().getType() << ", " << op.rhs().getType();
}

static ParseResult parseOuterProductOp(OpAsmParser &parser,
                                       OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> operandsInfo;
  Type tLHS, tRHS;
  if (parser.parseOperandList(operandsInfo) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(tLHS) || parser.parseComma() ||
      parser.parseType(tRHS))
    return failure();
  if (operandsInfo.size() < 2)
    return parser.emitError(parser.getNameLoc(),
                            "expected at least 2 operands");
  VectorType vLHS = tLHS.dyn_cast<VectorType>();
  VectorType vRHS = tRHS.dyn_cast<VectorType>();
  if (!vLHS)
    return parser.emitError(parser.getNameLoc(),
                            "expected vector type for operand #1");
  VectorType resType =
      vRHS ? VectorType::get({vLHS.getDimSize(0), vRHS.getDimSize(0)},
                             vLHS.getElementType())
           : VectorType::get({vLHS.getDimSize(0)}, vLHS.getElementType());

  if (!result.attributes.get(OuterProductOp::getKindAttrName())) {
    result.attributes.append(
        OuterProductOp::getKindAttrName(),
        CombiningKindAttr::get(OuterProductOp::getDefaultKind(),
                               result.getContext()));
  }

  return failure(
      parser.resolveOperand(operandsInfo[0], tLHS, result.operands) ||
      parser.resolveOperand(operandsInfo[1], tRHS, result.operands) ||
      (operandsInfo.size() > 2 &&
       parser.resolveOperand(operandsInfo[2], resType, result.operands)) ||
      parser.addTypeToList(resType, result.types));
}

static LogicalResult verify(OuterProductOp op) {
  Type tRHS = op.getOperandTypeRHS();
  VectorType vLHS = op.getOperandVectorTypeLHS(),
             vRHS = tRHS.dyn_cast<VectorType>(),
             vACC = op.getOperandVectorTypeACC(), vRES = op.getVectorType();

  if (vLHS.getRank() != 1)
    return op.emitOpError("expected 1-d vector for operand #1");

  if (vRHS) {
    // Proper OUTER operation.
    if (vRHS.getRank() != 1)
      return op.emitOpError("expected 1-d vector for operand #2");
    if (vRES.getRank() != 2)
      return op.emitOpError("expected 2-d vector result");
    if (vLHS.getDimSize(0) != vRES.getDimSize(0))
      return op.emitOpError("expected #1 operand dim to match result dim #1");
    if (vRHS.getDimSize(0) != vRES.getDimSize(1))
      return op.emitOpError("expected #2 operand dim to match result dim #2");
  } else {
    // An AXPY operation.
    if (vRES.getRank() != 1)
      return op.emitOpError("expected 1-d vector result");
    if (vLHS.getDimSize(0) != vRES.getDimSize(0))
      return op.emitOpError("expected #1 operand dim to match result dim #1");
  }

  if (vACC && vACC != vRES)
    return op.emitOpError("expected operand #3 of same type as result type");

  // Verify supported combining kind.
  if (!isSupportedCombiningKind(op.kind(), vRES.getElementType()))
    return op.emitOpError("unsupported outerproduct type");

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ReshapeOp op) {
  // Verify that rank(numInputs/outputs) + numFixedVec dim matches vec rank.
  auto inputVectorType = op.getInputVectorType();
  auto outputVectorType = op.getOutputVectorType();
  int64_t inputShapeRank = op.getNumInputShapeSizes();
  int64_t outputShapeRank = op.getNumOutputShapeSizes();
  SmallVector<int64_t, 4> fixedVectorSizes;
  op.getFixedVectorSizes(fixedVectorSizes);
  int64_t numFixedVectorSizes = fixedVectorSizes.size();

  if (inputVectorType.getRank() != inputShapeRank + numFixedVectorSizes)
    return op.emitError("invalid input shape for vector type ")
           << inputVectorType;

  if (outputVectorType.getRank() != outputShapeRank + numFixedVectorSizes)
    return op.emitError("invalid output shape for vector type ")
           << outputVectorType;

  // Verify that the 'fixedVectorSizes' match an input/output vector shape
  // suffix.
  unsigned inputVectorRank = inputVectorType.getRank();
  for (unsigned i = 0; i < numFixedVectorSizes; ++i) {
    unsigned index = inputVectorRank - numFixedVectorSizes - i;
    if (fixedVectorSizes[i] != inputVectorType.getShape()[index])
      return op.emitError("fixed vector size must match input vector for dim ")
             << i;
  }

  unsigned outputVectorRank = outputVectorType.getRank();
  for (unsigned i = 0; i < numFixedVectorSizes; ++i) {
    unsigned index = outputVectorRank - numFixedVectorSizes - i;
    if (fixedVectorSizes[i] != outputVectorType.getShape()[index])
      return op.emitError("fixed vector size must match output vector for dim ")
             << i;
  }

  // If all shape operands are produced by constant ops, verify that product
  // of dimensions for input/output shape match.
  auto isDefByConstant = [](Value operand) {
    return isa_and_nonnull<arith::ConstantIndexOp>(operand.getDefiningOp());
  };
  if (llvm::all_of(op.input_shape(), isDefByConstant) &&
      llvm::all_of(op.output_shape(), isDefByConstant)) {
    int64_t numInputElements = 1;
    for (auto operand : op.input_shape())
      numInputElements *=
          cast<arith::ConstantIndexOp>(operand.getDefiningOp()).value();
    int64_t numOutputElements = 1;
    for (auto operand : op.output_shape())
      numOutputElements *=
          cast<arith::ConstantIndexOp>(operand.getDefiningOp()).value();
    if (numInputElements != numOutputElements)
      return op.emitError("product of input and output shape sizes must match");
  }
  return success();
}

void ReshapeOp::getFixedVectorSizes(SmallVectorImpl<int64_t> &results) {
  populateFromInt64AttrArray(fixed_vector_sizes(), results);
}

//===----------------------------------------------------------------------===//
// ExtractStridedSliceOp
//===----------------------------------------------------------------------===//

// Inference works as follows:
//   1. Add 'sizes' from prefix of dims in 'offsets'.
//   2. Add sizes from 'vectorType' for remaining dims.
static Type inferStridedSliceOpResultType(VectorType vectorType,
                                          ArrayAttr offsets, ArrayAttr sizes,
                                          ArrayAttr strides) {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size());
  SmallVector<int64_t, 4> shape;
  shape.reserve(vectorType.getRank());
  unsigned idx = 0;
  for (unsigned e = offsets.size(); idx < e; ++idx)
    shape.push_back(sizes[idx].cast<IntegerAttr>().getInt());
  for (unsigned e = vectorType.getShape().size(); idx < e; ++idx)
    shape.push_back(vectorType.getShape()[idx]);

  return VectorType::get(shape, vectorType.getElementType());
}

void ExtractStridedSliceOp::build(OpBuilder &builder, OperationState &result,
                                  Value source, ArrayRef<int64_t> offsets,
                                  ArrayRef<int64_t> sizes,
                                  ArrayRef<int64_t> strides) {
  result.addOperands(source);
  auto offsetsAttr = getVectorSubscriptAttr(builder, offsets);
  auto sizesAttr = getVectorSubscriptAttr(builder, sizes);
  auto stridesAttr = getVectorSubscriptAttr(builder, strides);
  result.addTypes(
      inferStridedSliceOpResultType(source.getType().cast<VectorType>(),
                                    offsetsAttr, sizesAttr, stridesAttr));
  result.addAttribute(getOffsetsAttrName(), offsetsAttr);
  result.addAttribute(getSizesAttrName(), sizesAttr);
  result.addAttribute(getStridesAttrName(), stridesAttr);
}

static LogicalResult verify(ExtractStridedSliceOp op) {
  auto type = op.getVectorType();
  auto offsets = op.offsets();
  auto sizes = op.sizes();
  auto strides = op.strides();
  if (offsets.size() != sizes.size() || offsets.size() != strides.size()) {
    op.emitOpError(
        "expected offsets, sizes and strides attributes of same size");
    return failure();
  }

  auto shape = type.getShape();
  auto offName = ExtractStridedSliceOp::getOffsetsAttrName();
  auto sizesName = ExtractStridedSliceOp::getSizesAttrName();
  auto stridesName = ExtractStridedSliceOp::getStridesAttrName();
  if (failed(isIntegerArrayAttrSmallerThanShape(op, offsets, shape, offName)) ||
      failed(isIntegerArrayAttrSmallerThanShape(op, sizes, shape, sizesName)) ||
      failed(isIntegerArrayAttrSmallerThanShape(op, strides, shape,
                                                stridesName)) ||
      failed(isIntegerArrayAttrConfinedToShape(op, offsets, shape, offName)) ||
      failed(isIntegerArrayAttrConfinedToShape(op, sizes, shape, sizesName,
                                               /*halfOpen=*/false,
                                               /*min=*/1)) ||
      failed(isIntegerArrayAttrConfinedToRange(op, strides, 1, 1, stridesName,
                                               /*halfOpen=*/false)) ||
      failed(isSumOfIntegerArrayAttrConfinedToShape(op, offsets, sizes, shape,
                                                    offName, sizesName,
                                                    /*halfOpen=*/false)))
    return failure();

  auto resultType = inferStridedSliceOpResultType(
      op.getVectorType(), op.offsets(), op.sizes(), op.strides());
  if (op.getResult().getType() != resultType) {
    op.emitOpError("expected result type to be ") << resultType;
    return failure();
  }

  return success();
}

// When the source of ExtractStrided comes from a chain of InsertStrided ops try
// to use the source of the InsertStrided ops if we can detect that the
// extracted vector is a subset of one of the vector inserted.
static LogicalResult
foldExtractStridedOpFromInsertChain(ExtractStridedSliceOp op) {
  // Helper to extract integer out of ArrayAttr.
  auto getElement = [](ArrayAttr array, int idx) {
    return array[idx].cast<IntegerAttr>().getInt();
  };
  ArrayAttr extractOffsets = op.offsets();
  ArrayAttr extractStrides = op.strides();
  ArrayAttr extractSizes = op.sizes();
  auto insertOp = op.vector().getDefiningOp<InsertStridedSliceOp>();
  while (insertOp) {
    if (op.getVectorType().getRank() !=
        insertOp.getSourceVectorType().getRank())
      return failure();
    ArrayAttr insertOffsets = insertOp.offsets();
    ArrayAttr insertStrides = insertOp.strides();
    // If the rank of extract is greater than the rank of insert, we are likely
    // extracting a partial chunk of the vector inserted.
    if (extractOffsets.size() > insertOffsets.size())
      return failure();
    bool patialoverlap = false;
    bool disjoint = false;
    SmallVector<int64_t, 4> offsetDiffs;
    for (unsigned dim = 0, e = extractOffsets.size(); dim < e; ++dim) {
      if (getElement(extractStrides, dim) != getElement(insertStrides, dim))
        return failure();
      int64_t start = getElement(insertOffsets, dim);
      int64_t end = start + insertOp.getSourceVectorType().getDimSize(dim);
      int64_t offset = getElement(extractOffsets, dim);
      int64_t size = getElement(extractSizes, dim);
      // Check if the start of the extract offset is in the interval inserted.
      if (start <= offset && offset < end) {
        // If the extract interval overlaps but is not fully included we may
        // have a partial overlap that will prevent any folding.
        if (offset + size > end)
          patialoverlap = true;
        offsetDiffs.push_back(offset - start);
        continue;
      }
      disjoint = true;
      break;
    }
    // The extract element chunk is a subset of the insert element.
    if (!disjoint && !patialoverlap) {
      op.setOperand(insertOp.source());
      // OpBuilder is only used as a helper to build an I64ArrayAttr.
      OpBuilder b(op.getContext());
      op->setAttr(ExtractStridedSliceOp::getOffsetsAttrName(),
                  b.getI64ArrayAttr(offsetDiffs));
      return success();
    }
    // If the chunk extracted is disjoint from the chunk inserted, keep looking
    // in the insert chain.
    if (disjoint)
      insertOp = insertOp.dest().getDefiningOp<InsertStridedSliceOp>();
    else {
      // The extracted vector partially overlap the inserted vector, we cannot
      // fold.
      return failure();
    }
  }
  return failure();
}

OpFoldResult ExtractStridedSliceOp::fold(ArrayRef<Attribute> operands) {
  if (getVectorType() == getResult().getType())
    return vector();
  if (succeeded(foldExtractStridedOpFromInsertChain(*this)))
    return getResult();
  return {};
}

void ExtractStridedSliceOp::getOffsets(SmallVectorImpl<int64_t> &results) {
  populateFromInt64AttrArray(offsets(), results);
}

namespace {

// Pattern to rewrite an ExtractStridedSliceOp(ConstantMaskOp) to
// ConstantMaskOp.
class StridedSliceConstantMaskFolder final
    : public OpRewritePattern<ExtractStridedSliceOp> {
public:
  using OpRewritePattern<ExtractStridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractStridedSliceOp extractStridedSliceOp,
                                PatternRewriter &rewriter) const override {
    // Return if 'extractStridedSliceOp' operand is not defined by a
    // ConstantMaskOp.
    auto *defOp = extractStridedSliceOp.vector().getDefiningOp();
    auto constantMaskOp = dyn_cast_or_null<ConstantMaskOp>(defOp);
    if (!constantMaskOp)
      return failure();
    // Return if 'extractStridedSliceOp' has non-unit strides.
    if (llvm::any_of(extractStridedSliceOp.strides(), [](Attribute attr) {
          return attr.cast<IntegerAttr>().getInt() != 1;
        }))
      return failure();
    // Gather constant mask dimension sizes.
    SmallVector<int64_t, 4> maskDimSizes;
    populateFromInt64AttrArray(constantMaskOp.mask_dim_sizes(), maskDimSizes);
    // Gather strided slice offsets and sizes.
    SmallVector<int64_t, 4> sliceOffsets;
    populateFromInt64AttrArray(extractStridedSliceOp.offsets(), sliceOffsets);
    SmallVector<int64_t, 4> sliceSizes;
    populateFromInt64AttrArray(extractStridedSliceOp.sizes(), sliceSizes);

    // Compute slice of vector mask region.
    SmallVector<int64_t, 4> sliceMaskDimSizes;
    assert(sliceOffsets.size() == maskDimSizes.size());
    for (auto it : llvm::zip(maskDimSizes, sliceOffsets, sliceSizes)) {
      int64_t maskDimSize = std::get<0>(it);
      int64_t sliceOffset = std::get<1>(it);
      int64_t sliceSize = std::get<2>(it);
      int64_t sliceMaskDimSize = std::max(
          static_cast<int64_t>(0),
          std::min(sliceOffset + sliceSize, maskDimSize) - sliceOffset);
      sliceMaskDimSizes.push_back(sliceMaskDimSize);
    }
    // If any of 'sliceMaskDimSizes' are zero, then set all to zero (masked
    // region is a conjunction of mask dim intervals).
    if (llvm::is_contained(sliceMaskDimSizes, 0))
      sliceMaskDimSizes.assign(maskDimSizes.size(), 0);

    // Replace 'extractStridedSliceOp' with ConstantMaskOp with sliced mask
    // region.
    rewriter.replaceOpWithNewOp<ConstantMaskOp>(
        extractStridedSliceOp, extractStridedSliceOp.getResult().getType(),
        vector::getVectorSubscriptAttr(rewriter, sliceMaskDimSizes));
    return success();
  }
};

// Pattern to rewrite a ExtractStridedSliceOp(splat ConstantOp) -> ConstantOp.
class StridedSliceConstantFolder final
    : public OpRewritePattern<ExtractStridedSliceOp> {
public:
  using OpRewritePattern<ExtractStridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractStridedSliceOp extractStridedSliceOp,
                                PatternRewriter &rewriter) const override {
    // Return if 'extractStridedSliceOp' operand is not defined by a
    // ConstantOp.
    auto constantOp =
        extractStridedSliceOp.vector().getDefiningOp<arith::ConstantOp>();
    if (!constantOp)
      return failure();
    auto dense = constantOp.getValue().dyn_cast<SplatElementsAttr>();
    if (!dense)
      return failure();
    auto newAttr = DenseElementsAttr::get(extractStridedSliceOp.getType(),
                                          dense.getSplatValue<Attribute>());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(extractStridedSliceOp,
                                                   newAttr);
    return success();
  }
};

// Pattern to rewrite an ExtractStridedSliceOp(BroadcastOp) to
// BroadcastOp(ExtractStrideSliceOp).
class StridedSliceBroadcast final
    : public OpRewritePattern<ExtractStridedSliceOp> {
public:
  using OpRewritePattern<ExtractStridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto broadcast = op.vector().getDefiningOp<BroadcastOp>();
    if (!broadcast)
      return failure();
    auto srcVecType = broadcast.source().getType().dyn_cast<VectorType>();
    unsigned srcRrank = srcVecType ? srcVecType.getRank() : 0;
    auto dstVecType = op.getType().cast<VectorType>();
    unsigned dstRank = dstVecType.getRank();
    unsigned rankDiff = dstRank - srcRrank;
    // Check if the most inner dimensions of the source of the broadcast are the
    // same as the destination of the extract. If this is the case we can just
    // use a broadcast as the original dimensions are untouched.
    bool lowerDimMatch = true;
    for (unsigned i = 0; i < srcRrank; i++) {
      if (srcVecType.getDimSize(i) != dstVecType.getDimSize(i + rankDiff)) {
        lowerDimMatch = false;
        break;
      }
    }
    Value source = broadcast.source();
    if (!lowerDimMatch) {
      // The inner dimensions don't match, it means we need to extract from the
      // source of the orignal broadcast and then broadcast the extracted value.
      source = rewriter.create<ExtractStridedSliceOp>(
          op->getLoc(), source,
          getI64SubArray(op.offsets(), /* dropFront=*/rankDiff),
          getI64SubArray(op.sizes(), /* dropFront=*/rankDiff),
          getI64SubArray(op.strides(), /* dropFront=*/rankDiff));
    }
    rewriter.replaceOpWithNewOp<BroadcastOp>(op, op.getType(), source);
    return success();
  }
};

/// Pattern to rewrite an ExtractStridedSliceOp(SplatOp) to SplatOp.
class StridedSliceSplat final : public OpRewritePattern<ExtractStridedSliceOp> {
public:
  using OpRewritePattern<ExtractStridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto splat = op.vector().getDefiningOp<SplatOp>();
    if (!splat)
      return failure();
    rewriter.replaceOpWithNewOp<SplatOp>(op, op.getType(), splat.getInput());
    return success();
  }
};

} // namespace

void ExtractStridedSliceOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  // Pattern to rewrite a ExtractStridedSliceOp(ConstantMaskOp) ->
  // ConstantMaskOp and ExtractStridedSliceOp(ConstantOp) -> ConstantOp.
  results.add<StridedSliceConstantMaskFolder, StridedSliceConstantFolder,
              StridedSliceBroadcast, StridedSliceSplat>(context);
}

//===----------------------------------------------------------------------===//
// TransferReadOp
//===----------------------------------------------------------------------===//

/// 1. Builder that sets padding to zero and an empty mask (variant with attrs).
void TransferReadOp::build(OpBuilder &builder, OperationState &result,
                           VectorType vectorType, Value source,
                           ValueRange indices, AffineMapAttr permutationMapAttr,
                           /*optional*/ ArrayAttr inBoundsAttr) {
  Type elemType = source.getType().cast<ShapedType>().getElementType();
  Value padding = builder.create<arith::ConstantOp>(
      result.location, elemType, builder.getZeroAttr(elemType));
  build(builder, result, vectorType, source, indices, permutationMapAttr,
        padding, /*mask=*/Value(), inBoundsAttr);
}

/// 2. Builder that sets padding to zero an empty mask (variant without attrs).
void TransferReadOp::build(OpBuilder &builder, OperationState &result,
                           VectorType vectorType, Value source,
                           ValueRange indices, AffineMap permutationMap,
                           Optional<ArrayRef<bool>> inBounds) {
  auto permutationMapAttr = AffineMapAttr::get(permutationMap);
  auto inBoundsAttr = (inBounds && !inBounds.getValue().empty())
                          ? builder.getBoolArrayAttr(inBounds.getValue())
                          : ArrayAttr();
  build(builder, result, vectorType, source, indices, permutationMapAttr,
        inBoundsAttr);
}

/// 3. Builder that sets permutation map to 'getMinorIdentityMap'.
void TransferReadOp::build(OpBuilder &builder, OperationState &result,
                           VectorType vectorType, Value source,
                           ValueRange indices, Value padding,
                           Optional<ArrayRef<bool>> inBounds) {
  AffineMap permutationMap = getTransferMinorIdentityMap(
      source.getType().cast<ShapedType>(), vectorType);
  auto permutationMapAttr = AffineMapAttr::get(permutationMap);
  auto inBoundsAttr = (inBounds && !inBounds.getValue().empty())
                          ? builder.getBoolArrayAttr(inBounds.getValue())
                          : ArrayAttr();
  build(builder, result, vectorType, source, indices, permutationMapAttr,
        padding,
        /*mask=*/Value(), inBoundsAttr);
}

/// 4. Builder that sets padding to zero and permutation map to
/// 'getMinorIdentityMap'.
void TransferReadOp::build(OpBuilder &builder, OperationState &result,
                           VectorType vectorType, Value source,
                           ValueRange indices,
                           Optional<ArrayRef<bool>> inBounds) {
  Type elemType = source.getType().cast<ShapedType>().getElementType();
  Value padding = builder.create<arith::ConstantOp>(
      result.location, elemType, builder.getZeroAttr(elemType));
  build(builder, result, vectorType, source, indices, padding, inBounds);
}

template <typename EmitFun>
static LogicalResult verifyPermutationMap(AffineMap permutationMap,
                                          EmitFun emitOpError) {
  SmallVector<bool, 8> seen(permutationMap.getNumInputs(), false);
  for (auto expr : permutationMap.getResults()) {
    auto dim = expr.dyn_cast<AffineDimExpr>();
    auto zero = expr.dyn_cast<AffineConstantExpr>();
    if (zero) {
      if (zero.getValue() != 0) {
        return emitOpError(
            "requires a projected permutation_map (at most one dim or the zero "
            "constant can appear in each result)");
      }
      continue;
    }
    if (!dim) {
      return emitOpError("requires a projected permutation_map (at most one "
                         "dim or the zero constant can appear in each result)");
    }
    if (seen[dim.getPosition()]) {
      return emitOpError(
          "requires a permutation_map that is a permutation (found one dim "
          "used more than once)");
    }
    seen[dim.getPosition()] = true;
  }
  return success();
}

static LogicalResult
verifyTransferOp(VectorTransferOpInterface op, ShapedType shapedType,
                 VectorType vectorType, VectorType maskType,
                 AffineMap permutationMap, ArrayAttr inBounds) {
  if (op->hasAttr("masked")) {
    return op->emitOpError("masked attribute has been removed. "
                           "Use in_bounds instead.");
  }

  if (!shapedType.isa<MemRefType, RankedTensorType>())
    return op->emitOpError(
        "requires source to be a memref or ranked tensor type");

  auto elementType = shapedType.getElementType();
  DataLayout dataLayout = DataLayout::closest(op);
  if (auto vectorElementType = elementType.dyn_cast<VectorType>()) {
    // Memref or tensor has vector element type.
    unsigned sourceVecSize =
        dataLayout.getTypeSizeInBits(vectorElementType.getElementType()) *
        vectorElementType.getShape().back();
    unsigned resultVecSize =
        dataLayout.getTypeSizeInBits(vectorType.getElementType()) *
        vectorType.getShape().back();
    if (resultVecSize % sourceVecSize != 0)
      return op->emitOpError(
          "requires the bitwidth of the minor 1-D vector to be an integral "
          "multiple of the bitwidth of the minor 1-D vector of the source");

    unsigned sourceVecEltRank = vectorElementType.getRank();
    unsigned resultVecRank = vectorType.getRank();
    if (sourceVecEltRank > resultVecRank)
      return op->emitOpError(
          "requires source vector element and vector result ranks to match.");
    unsigned rankOffset = resultVecRank - sourceVecEltRank;
    // Check that permutation map results match 'rankOffset' of vector type.
    if (permutationMap.getNumResults() != rankOffset)
      return op->emitOpError("requires a permutation_map with result dims of "
                             "the same rank as the vector type");

    if (maskType)
      return op->emitOpError("does not support masks with vector element type");
  } else {
    // Memref or tensor has scalar element type.
    unsigned minorSize =
        vectorType.getRank() == 0 ? 1 : vectorType.getShape().back();
    unsigned resultVecSize =
        dataLayout.getTypeSizeInBits(vectorType.getElementType()) * minorSize;
    if (resultVecSize % dataLayout.getTypeSizeInBits(elementType) != 0)
      return op->emitOpError(
          "requires the bitwidth of the minor 1-D vector to be an integral "
          "multiple of the bitwidth of the source element type");

    // Check that permutation map results match rank of vector type.
    if (permutationMap.getNumResults() != vectorType.getRank())
      return op->emitOpError("requires a permutation_map with result dims of "
                             "the same rank as the vector type");

    VectorType expectedMaskType =
        vector::detail::transferMaskType(vectorType, permutationMap);
    if (maskType && expectedMaskType != maskType)
      return op->emitOpError("expects mask type consistent with permutation "
                             "map: ")
             << maskType;
  }

  if (permutationMap.getNumSymbols() != 0)
    return op->emitOpError("requires permutation_map without symbols");

  if (permutationMap.getNumInputs() != shapedType.getRank())
    return op->emitOpError("requires a permutation_map with input dims of the "
                           "same rank as the source type");

  if (inBounds) {
    if (permutationMap.getNumResults() != static_cast<int64_t>(inBounds.size()))
      return op->emitOpError("expects the optional in_bounds attr of same rank "
                             "as permutation_map results: ")
             << AffineMapAttr::get(permutationMap)
             << " vs inBounds of size: " << inBounds.size();
    for (unsigned int i = 0; i < permutationMap.getNumResults(); ++i)
      if (permutationMap.getResult(i).isa<AffineConstantExpr>() &&
          !inBounds.getValue()[i].cast<BoolAttr>().getValue())
        return op->emitOpError("requires broadcast dimensions to be in-bounds");
  }

  return success();
}

static void printTransferAttrs(OpAsmPrinter &p, VectorTransferOpInterface op) {
  SmallVector<StringRef, 3> elidedAttrs;
  elidedAttrs.push_back(TransferReadOp::getOperandSegmentSizeAttr());
  if (op.permutation_map().isMinorIdentity())
    elidedAttrs.push_back(op.getPermutationMapAttrName());
  bool elideInBounds = true;
  if (auto inBounds = op.in_bounds()) {
    for (auto attr : *inBounds) {
      if (attr.template cast<BoolAttr>().getValue()) {
        elideInBounds = false;
        break;
      }
    }
  }
  if (elideInBounds)
    elidedAttrs.push_back(op.getInBoundsAttrName());
  p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}

static void print(OpAsmPrinter &p, TransferReadOp op) {
  p << " " << op.source() << "[" << op.indices() << "], " << op.padding();
  if (op.mask())
    p << ", " << op.mask();
  printTransferAttrs(p, cast<VectorTransferOpInterface>(op.getOperation()));
  p << " : " << op.getShapedType() << ", " << op.getVectorType();
}

static ParseResult parseTransferReadOp(OpAsmParser &parser,
                                       OperationState &result) {
  auto &builder = parser.getBuilder();
  llvm::SMLoc typesLoc;
  OpAsmParser::OperandType sourceInfo;
  SmallVector<OpAsmParser::OperandType, 8> indexInfo;
  OpAsmParser::OperandType paddingInfo;
  SmallVector<Type, 2> types;
  OpAsmParser::OperandType maskInfo;
  // Parsing with support for paddingValue.
  if (parser.parseOperand(sourceInfo) ||
      parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseOperand(paddingInfo))
    return failure();
  ParseResult hasMask = parser.parseOptionalComma();
  if (hasMask.succeeded()) {
    parser.parseOperand(maskInfo);
  }
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");
  auto indexType = builder.getIndexType();
  auto shapedType = types[0].dyn_cast<ShapedType>();
  if (!shapedType || !shapedType.isa<MemRefType, RankedTensorType>())
    return parser.emitError(typesLoc, "requires memref or ranked tensor type");
  VectorType vectorType = types[1].dyn_cast<VectorType>();
  if (!vectorType)
    return parser.emitError(typesLoc, "requires vector type");
  auto permutationAttrName = TransferReadOp::getPermutationMapAttrName();
  Attribute mapAttr = result.attributes.get(permutationAttrName);
  if (!mapAttr) {
    auto permMap = getTransferMinorIdentityMap(shapedType, vectorType);
    // Update `mapAttr` that is used later to determine mask type.
    mapAttr = AffineMapAttr::get(permMap);
    result.attributes.set(permutationAttrName, mapAttr);
  }
  if (parser.resolveOperand(sourceInfo, shapedType, result.operands) ||
      parser.resolveOperands(indexInfo, indexType, result.operands) ||
      parser.resolveOperand(paddingInfo, shapedType.getElementType(),
                            result.operands))
    return failure();
  if (hasMask.succeeded()) {
    if (shapedType.getElementType().dyn_cast<VectorType>())
      return parser.emitError(
          maskInfo.location, "does not support masks with vector element type");
    auto map = mapAttr.dyn_cast<AffineMapAttr>().getValue();
    // Instead of adding the mask type as an op type, compute it based on the
    // vector type and the permutation map (to keep the type signature small).
    auto maskType = mlir::vector::detail::transferMaskType(vectorType, map);
    if (parser.resolveOperand(maskInfo, maskType, result.operands))
      return failure();
  }
  result.addAttribute(
      TransferReadOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({1, static_cast<int32_t>(indexInfo.size()), 1,
                                static_cast<int32_t>(hasMask.succeeded())}));
  return parser.addTypeToList(vectorType, result.types);
}

static LogicalResult verify(TransferReadOp op) {
  // Consistency of elemental types in source and vector.
  ShapedType shapedType = op.getShapedType();
  VectorType vectorType = op.getVectorType();
  VectorType maskType = op.getMaskType();
  auto paddingType = op.padding().getType();
  auto permutationMap = op.permutation_map();
  auto sourceElementType = shapedType.getElementType();

  if (static_cast<int64_t>(op.indices().size()) != shapedType.getRank())
    return op.emitOpError("requires ") << shapedType.getRank() << " indices";

  if (failed(
          verifyTransferOp(cast<VectorTransferOpInterface>(op.getOperation()),
                           shapedType, vectorType, maskType, permutationMap,
                           op.in_bounds() ? *op.in_bounds() : ArrayAttr())))
    return failure();

  if (auto sourceVectorElementType = sourceElementType.dyn_cast<VectorType>()) {
    // Source has vector element type.
    // Check that 'sourceVectorElementType' and 'paddingType' types match.
    if (sourceVectorElementType != paddingType)
      return op.emitOpError(
          "requires source element type and padding type to match.");

  } else {
    // Check that 'paddingType' is valid to store in a vector type.
    if (!VectorType::isValidElementType(paddingType))
      return op.emitOpError("requires valid padding vector elemental type");

    // Check that padding type and vector element types match.
    if (paddingType != sourceElementType)
      return op.emitOpError(
          "requires formal padding and source of the same elemental type");
  }

  return verifyPermutationMap(permutationMap,
                              [&op](Twine t) { return op.emitOpError(t); });
}

/// This is a common class used for patterns of the form
/// ```
///    someop(memrefcast) -> someop
/// ```
/// It folds the source of the memref.cast into the root operation directly.
static LogicalResult foldMemRefCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto castOp = operand.get().getDefiningOp<memref::CastOp>();
    if (castOp && memref::CastOp::canFoldIntoConsumerOp(castOp)) {
      operand.set(castOp.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

static LogicalResult foldTensorCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto castOp = operand.get().getDefiningOp<tensor::CastOp>();
    if (castOp && tensor::canFoldIntoConsumerOp(castOp)) {
      operand.set(castOp.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

template <typename TransferOp>
static bool isInBounds(TransferOp op, int64_t resultIdx, int64_t indicesIdx) {
  // TODO: support more aggressive createOrFold on:
  // `op.indices()[indicesIdx] + vectorType < dim(op.source(), indicesIdx)`
  if (op.getShapedType().isDynamicDim(indicesIdx))
    return false;
  Value index = op.indices()[indicesIdx];
  auto cstOp = index.getDefiningOp<arith::ConstantIndexOp>();
  if (!cstOp)
    return false;

  int64_t sourceSize = op.getShapedType().getDimSize(indicesIdx);
  int64_t vectorSize = op.getVectorType().getDimSize(resultIdx);

  return cstOp.value() + vectorSize <= sourceSize;
}

template <typename TransferOp>
static LogicalResult foldTransferInBoundsAttribute(TransferOp op) {
  // TODO: support 0-d corner case.
  // TODO: Be less conservative.
  if (op.getTransferRank() == 0)
    return failure();
  AffineMap permutationMap = op.permutation_map();
  bool changed = false;
  SmallVector<bool, 4> newInBounds;
  newInBounds.reserve(op.getTransferRank());
  for (unsigned i = 0; i < op.getTransferRank(); ++i) {
    // Already marked as in-bounds, nothing to see here.
    if (op.isDimInBounds(i)) {
      newInBounds.push_back(true);
      continue;
    }
    // Currently out-of-bounds, check whether we can statically determine it is
    // inBounds.
    auto dimExpr = permutationMap.getResult(i).dyn_cast<AffineDimExpr>();
    assert(dimExpr && "Broadcast dims must be in-bounds");
    auto inBounds =
        isInBounds(op, /*resultIdx=*/i, /*indicesIdx=*/dimExpr.getPosition());
    newInBounds.push_back(inBounds);
    // We commit the pattern if it is "more inbounds".
    changed |= inBounds;
  }
  if (!changed)
    return failure();
  // OpBuilder is only used as a helper to build an I64ArrayAttr.
  OpBuilder b(op.getContext());
  op->setAttr(TransferOp::getInBoundsAttrName(),
              b.getBoolArrayAttr(newInBounds));
  return success();
}

///  ```
///  %w0 = vector.transfer_write %v0, %arg0[%c1, %c0] {in_bounds = [true, true]}
///    : vector<1x4xf32>, tensor<4x4xf32>
///  %0 = vector.transfer_read %w0[%c1, %c0], %cf0 {in_bounds = [true, true]}
///    : tensor<4x4xf32>, vector<1x4xf32>
///  ```
///  -> Folds into
///  ```
///  %v0
///  ```
static Value foldRAW(TransferReadOp readOp) {
  if (!readOp.getShapedType().isa<RankedTensorType>())
    return {};
  auto defWrite = readOp.source().getDefiningOp<vector::TransferWriteOp>();
  while (defWrite) {
    if (checkSameValueRAW(defWrite, readOp))
      return defWrite.vector();
    if (!isDisjointTransferIndices(
            cast<VectorTransferOpInterface>(defWrite.getOperation()),
            cast<VectorTransferOpInterface>(readOp.getOperation())))
      break;
    defWrite = defWrite.source().getDefiningOp<vector::TransferWriteOp>();
  }
  return {};
}

OpFoldResult TransferReadOp::fold(ArrayRef<Attribute>) {
  if (Value vec = foldRAW(*this))
    return vec;
  /// transfer_read(memrefcast) -> transfer_read
  if (succeeded(foldTransferInBoundsAttribute(*this)))
    return getResult();
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  if (succeeded(foldTensorCast(*this)))
    return getResult();
  return OpFoldResult();
}

Optional<SmallVector<int64_t, 4>> TransferReadOp::getShapeForUnroll() {
  return llvm::to_vector<4>(getVectorType().getShape());
}

void TransferReadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (getShapedType().isa<MemRefType>())
    effects.emplace_back(MemoryEffects::Read::get(), source(),
                         SideEffects::DefaultResource::get());
}

namespace {
/// Fold transfer_reads of a tensor.extract_slice op. E.g.:
///
/// ```
/// %0 = tensor.extract_slice %t[%a, %b] [%c, %d] [1, 1]
///     : tensor<?x?xf32> to tensor<?x?xf32>
/// %1 = vector.transfer_read %0[%e, %f], %cst {in_bounds = [true, true]}
///     : tensor<?x?xf32>, vector<4x5xf32>
/// ```
/// is rewritten to:
/// ```
/// %p0 = arith.addi %a, %e : index
/// %p1 = arith.addi %b, %f : index
/// %1 = vector.transfer_read %t[%p0, %p1], %cst {in_bounds = [true, true]}
///     : tensor<?x?xf32>, vector<4x5xf32>
/// ```
struct FoldExtractSliceIntoTransferRead
    : public OpRewritePattern<TransferReadOp> {
public:
  using OpRewritePattern<TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferReadOp xferOp,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (xferOp.getTransferRank() == 0)
      return failure();
    if (xferOp.hasOutOfBoundsDim())
      return failure();
    if (!xferOp.permutation_map().isIdentity())
      return failure();
    if (xferOp.mask())
      return failure();
    auto extractOp = xferOp.source().getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractOp)
      return failure();
    if (!extractOp.hasUnitStride())
      return failure();

    // Bail on illegal rank-reduction: we need to check that the rank-reduced
    // dims are exactly the leading dims. I.e. the following is illegal:
    // ```
    //    %0 = tensor.extract_slice %t[0,0,0][2,1,4][1,1,1] :
    //      tensor<2x1x4xf32> to tensor<2x4xf32>
    //    %1 = vector.transfer_read %0[0,0], %cst :
    //      tensor<2x4xf32>, vector<2x4xf32>
    // ```
    //
    // Cannot fold into:
    // ```
    //    %0 = vector.transfer_read %t[0,0,0], %cst :
    //      tensor<2x1x4xf32>, vector<2x4xf32>
    // ```
    // For this, check the trailing `vectorRank` dims of the extract_slice
    // result tensor match the trailing dims of the inferred result tensor.
    int64_t rankReduced =
        extractOp.getSourceType().getRank() - extractOp.getType().getRank();
    int64_t vectorRank = xferOp.getVectorType().getRank();
    RankedTensorType inferredDestTensorType =
        tensor::ExtractSliceOp::inferResultType(
            extractOp.getSourceType(), extractOp.getMixedOffsets(),
            extractOp.getMixedSizes(), extractOp.getMixedStrides());
    auto actualDestTensorShape = extractOp.getType().getShape();
    if (rankReduced > 0 &&
        actualDestTensorShape.take_back(vectorRank) !=
            inferredDestTensorType.getShape().take_back(vectorRank))
      return failure();

    SmallVector<Value> newIndices;
    // In case this is a rank-reducing ExtractSliceOp, copy rank-reduced
    // indices first.
    for (int64_t i = 0; i < rankReduced; ++i) {
      OpFoldResult offset = extractOp.getMixedOffsets()[i];
      newIndices.push_back(getValueOrCreateConstantIndexOp(
          rewriter, extractOp.getLoc(), offset));
    }
    for (const auto &it : llvm::enumerate(xferOp.indices())) {
      OpFoldResult offset =
          extractOp.getMixedOffsets()[it.index() + rankReduced];
      newIndices.push_back(rewriter.create<arith::AddIOp>(
          xferOp->getLoc(), it.value(),
          getValueOrCreateConstantIndexOp(rewriter, extractOp.getLoc(),
                                          offset)));
    }
    SmallVector<bool> inBounds(xferOp.getTransferRank(), true);
    rewriter.replaceOpWithNewOp<TransferReadOp>(
        xferOp, xferOp.getVectorType(), extractOp.source(), newIndices,
        xferOp.padding(), ArrayRef<bool>{inBounds});

    return success();
  }
};
} // namespace

void TransferReadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<FoldExtractSliceIntoTransferRead>(context);
}

//===----------------------------------------------------------------------===//
// TransferWriteOp
//===----------------------------------------------------------------------===//

/// 1. Builder with type inference.
void TransferWriteOp::build(OpBuilder &builder, OperationState &result,
                            Value vector, Value dest, ValueRange indices,
                            AffineMapAttr permutationMapAttr,
                            /*optional*/ Value mask,
                            /*optional*/ ArrayAttr inBoundsAttr) {
  Type resultType = dest.getType().dyn_cast<RankedTensorType>();
  build(builder, result, resultType, vector, dest, indices, permutationMapAttr,
        mask, inBoundsAttr);
}

/// 2. Builder with type inference that sets an empty mask (variant with attrs).
void TransferWriteOp::build(OpBuilder &builder, OperationState &result,
                            Value vector, Value dest, ValueRange indices,
                            AffineMapAttr permutationMapAttr,
                            /*optional*/ ArrayAttr inBoundsAttr) {
  build(builder, result, vector, dest, indices, permutationMapAttr,
        /*mask=*/Value(), inBoundsAttr);
}

/// 3. Builder with type inference that sets an empty mask (variant without
/// attrs)
void TransferWriteOp::build(OpBuilder &builder, OperationState &result,
                            Value vector, Value dest, ValueRange indices,
                            AffineMap permutationMap,
                            Optional<ArrayRef<bool>> inBounds) {
  auto permutationMapAttr = AffineMapAttr::get(permutationMap);
  auto inBoundsAttr = (inBounds && !inBounds.getValue().empty())
                          ? builder.getBoolArrayAttr(inBounds.getValue())
                          : ArrayAttr();
  build(builder, result, vector, dest, indices, permutationMapAttr,
        /*mask=*/Value(), inBoundsAttr);
}

/// 4. Builder with type inference that sets an empty mask and sets permutation
///    map to 'getMinorIdentityMap'.
void TransferWriteOp::build(OpBuilder &builder, OperationState &result,
                            Value vector, Value dest, ValueRange indices,
                            Optional<ArrayRef<bool>> inBounds) {
  auto vectorType = vector.getType().cast<VectorType>();
  AffineMap permutationMap = getTransferMinorIdentityMap(
      dest.getType().cast<ShapedType>(), vectorType);
  build(builder, result, vector, dest, indices, permutationMap, inBounds);
}

static ParseResult parseTransferWriteOp(OpAsmParser &parser,
                                        OperationState &result) {
  auto &builder = parser.getBuilder();
  llvm::SMLoc typesLoc;
  OpAsmParser::OperandType vectorInfo, sourceInfo;
  SmallVector<OpAsmParser::OperandType, 8> indexInfo;
  SmallVector<Type, 2> types;
  OpAsmParser::OperandType maskInfo;
  if (parser.parseOperand(vectorInfo) || parser.parseComma() ||
      parser.parseOperand(sourceInfo) ||
      parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square))
    return failure();
  ParseResult hasMask = parser.parseOptionalComma();
  if (hasMask.succeeded() && parser.parseOperand(maskInfo))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");
  auto indexType = builder.getIndexType();
  VectorType vectorType = types[0].dyn_cast<VectorType>();
  if (!vectorType)
    return parser.emitError(typesLoc, "requires vector type");
  ShapedType shapedType = types[1].dyn_cast<ShapedType>();
  if (!shapedType || !shapedType.isa<MemRefType, RankedTensorType>())
    return parser.emitError(typesLoc, "requires memref or ranked tensor type");
  auto permutationAttrName = TransferWriteOp::getPermutationMapAttrName();
  auto attr = result.attributes.get(permutationAttrName);
  if (!attr) {
    auto permMap = getTransferMinorIdentityMap(shapedType, vectorType);
    result.attributes.set(permutationAttrName, AffineMapAttr::get(permMap));
  }
  if (parser.resolveOperand(vectorInfo, vectorType, result.operands) ||
      parser.resolveOperand(sourceInfo, shapedType, result.operands) ||
      parser.resolveOperands(indexInfo, indexType, result.operands))
    return failure();
  if (hasMask.succeeded()) {
    if (shapedType.getElementType().dyn_cast<VectorType>())
      return parser.emitError(
          maskInfo.location, "does not support masks with vector element type");
    auto maskType = VectorType::get(vectorType.getShape(), builder.getI1Type());
    if (parser.resolveOperand(maskInfo, maskType, result.operands))
      return failure();
  }
  result.addAttribute(
      TransferWriteOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({1, 1, static_cast<int32_t>(indexInfo.size()),
                                static_cast<int32_t>(hasMask.succeeded())}));
  return failure(shapedType.isa<RankedTensorType>() &&
                 parser.addTypeToList(shapedType, result.types));
}

static void print(OpAsmPrinter &p, TransferWriteOp op) {
  p << " " << op.vector() << ", " << op.source() << "[" << op.indices() << "]";
  if (op.mask())
    p << ", " << op.mask();
  printTransferAttrs(p, cast<VectorTransferOpInterface>(op.getOperation()));
  p << " : " << op.getVectorType() << ", " << op.getShapedType();
}

static LogicalResult verify(TransferWriteOp op) {
  // Consistency of elemental types in shape and vector.
  ShapedType shapedType = op.getShapedType();
  VectorType vectorType = op.getVectorType();
  VectorType maskType = op.getMaskType();
  auto permutationMap = op.permutation_map();

  if (llvm::size(op.indices()) != shapedType.getRank())
    return op.emitOpError("requires ") << shapedType.getRank() << " indices";

  // We do not allow broadcast dimensions on TransferWriteOps for the moment,
  // as the semantics is unclear. This can be revisited later if necessary.
  if (op.hasBroadcastDim())
    return op.emitOpError("should not have broadcast dimensions");

  if (failed(
          verifyTransferOp(cast<VectorTransferOpInterface>(op.getOperation()),
                           shapedType, vectorType, maskType, permutationMap,
                           op.in_bounds() ? *op.in_bounds() : ArrayAttr())))
    return failure();

  return verifyPermutationMap(permutationMap,
                              [&op](Twine t) { return op.emitOpError(t); });
}

/// Fold:
/// ```
///    %t1 = ...
///    %v = vector.transfer_read %t0[%c0...], {in_bounds = [true...]} :
///      tensor<static_sizesxf32>, vector<static_sizesxf32>
///    %t2 = vector.transfer_write %v, %t1[%c0...] {in_bounds = [true...]} :
///      vector<static_sizesxf32>, tensor<static_sizesxf32>
/// ```
///
/// into:
///
/// ```
///    %t0
/// ```
///
/// The producer of t1 may or may not be DCE'd depending on whether it is a
/// block argument or has side effects.
static LogicalResult foldReadInitWrite(TransferWriteOp write,
                                       ArrayRef<Attribute>,
                                       SmallVectorImpl<OpFoldResult> &results) {
  // TODO: support 0-d corner case.
  if (write.getTransferRank() == 0)
    return failure();
  auto rankedTensorType = write.source().getType().dyn_cast<RankedTensorType>();
  // If not operating on tensors, bail.
  if (!rankedTensorType)
    return failure();
  // If no read, bail.
  auto read = write.vector().getDefiningOp<vector::TransferReadOp>();
  if (!read)
    return failure();
  // TODO: support 0-d corner case.
  if (read.getTransferRank() == 0)
    return failure();
  // For now, only accept minor identity. Future: composition is minor identity.
  if (!read.permutation_map().isMinorIdentity() ||
      !write.permutation_map().isMinorIdentity())
    return failure();
  // Bail on mismatching ranks.
  if (read.getTransferRank() != write.getTransferRank())
    return failure();
  // Bail on potential out-of-bounds accesses.
  if (read.hasOutOfBoundsDim() || write.hasOutOfBoundsDim())
    return failure();
  // Tensor types must be the same.
  if (read.source().getType() != rankedTensorType)
    return failure();
  // Vector types must be the same.
  if (read.getVectorType() != write.getVectorType())
    return failure();
  // Vector and Tensor shapes must match.
  if (read.getVectorType().getShape() != rankedTensorType.getShape())
    return failure();
  // If any index is nonzero.
  auto isNotConstantZero = [](Value v) {
    auto cstOp = v.getDefiningOp<arith::ConstantIndexOp>();
    return !cstOp || cstOp.value() != 0;
  };
  if (llvm::any_of(read.indices(), isNotConstantZero) ||
      llvm::any_of(write.indices(), isNotConstantZero))
    return failure();
  // Success.
  results.push_back(read.source());
  return success();
}

static bool checkSameValueWAR(vector::TransferReadOp read,
                              vector::TransferWriteOp write) {
  return read.source() == write.source() && read.indices() == write.indices() &&
         read.permutation_map() == write.permutation_map() &&
         read.getVectorType() == write.getVectorType() && !read.mask() &&
         !write.mask();
}
/// Fold transfer_write write after read:
/// ```
///    %t0 = ...
///    %v = vector.transfer_read %t0[%c0...] :
///      tensor<static_sizesxf32>, vector<static_sizesxf32>
///    %t1 = vector.transfer_write %v, %t0[%c0...] :
///      vector<static_sizesxf32>, tensor<static_sizesxf32>
/// ```
///
/// into:
///
/// ```
///    %t0
/// ```
static LogicalResult foldWAR(TransferWriteOp write,
                             SmallVectorImpl<OpFoldResult> &results) {
  if (!write.source().getType().isa<RankedTensorType>())
    return failure();
  auto read = write.vector().getDefiningOp<vector::TransferReadOp>();
  if (!read)
    return failure();

  if (!checkSameValueWAR(read, write))
    return failure();
  results.push_back(read.source());
  return success();
}

LogicalResult TransferWriteOp::fold(ArrayRef<Attribute> operands,
                                    SmallVectorImpl<OpFoldResult> &results) {
  if (succeeded(foldReadInitWrite(*this, operands, results)))
    return success();
  if (succeeded(foldWAR(*this, results)))
    return success();
  if (succeeded(foldTransferInBoundsAttribute(*this)))
    return success();
  return foldMemRefCast(*this);
}

Optional<SmallVector<int64_t, 4>> TransferWriteOp::getShapeForUnroll() {
  return llvm::to_vector<4>(getVectorType().getShape());
}

void TransferWriteOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (getShapedType().isa<MemRefType>())
    effects.emplace_back(MemoryEffects::Write::get(), source(),
                         SideEffects::DefaultResource::get());
}

namespace {
/// Remove dead transfer write from the SSA chain so that it an be eliminated by
/// DCE
/// ```
///  %w0 = vector.transfer_write %v0, %arg0[%c1, %c0] {in_bounds = [true, true]}
///    : vector<1x4xf32>, tensor<4x4xf32>
///  %w1 = vector.transfer_write %v0, %w0[%c2, %c0] {in_bounds = [true, true]}
///    : vector<1x4xf32>, tensor<4x4xf32>
///  %w2 = vector.transfer_write %v1, %w1[%c1, %c0] {in_bounds = [true, true]}
///    : vector<1x4xf32>, tensor<4x4xf32>
/// ```
///
/// into:
///
/// ```
///  %w0 = vector.transfer_write %v0, %arg0[%c1, %c0] {in_bounds = [true, true]}
///    : vector<1x4xf32>, tensor<4x4xf32>
///  %w1 = vector.transfer_write %v0, %arg0[%c2, %c0] {in_bounds = [true, true]}
///    : vector<1x4xf32>, tensor<4x4xf32>
///  %w2 = vector.transfer_write %v1, %w1[%c1, %c0] {in_bounds = [true, true]}
///    : vector<1x4xf32>, tensor<4x4xf32>
/// ```
///
/// `%w0 = vector.transfer_write` op will be removed by DCE if it doesn't have
/// any other uses.
class FoldWaw final : public OpRewritePattern<TransferWriteOp> {
public:
  using OpRewritePattern<TransferWriteOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    if (!writeOp.getShapedType().isa<RankedTensorType>())
      return failure();
    vector::TransferWriteOp writeToModify = writeOp;

    auto defWrite = writeOp.source().getDefiningOp<vector::TransferWriteOp>();
    while (defWrite) {
      if (checkSameValueWAW(writeOp, defWrite)) {
        writeToModify.sourceMutable().assign(defWrite.source());
        return success();
      }
      if (!isDisjointTransferIndices(
              cast<VectorTransferOpInterface>(defWrite.getOperation()),
              cast<VectorTransferOpInterface>(writeOp.getOperation())))
        break;
      // If the previous write op doesn't have any other use we an safely look
      // at the previous store to see if it can be removed.
      if (!defWrite->hasOneUse())
        break;
      writeToModify = defWrite;
      defWrite = defWrite.source().getDefiningOp<vector::TransferWriteOp>();
    }
    return failure();
  }
};

/// Fold tensor.insert_slice into vector.transfer_write if the transfer_write
/// could directly write to the insert_slice's destination. E.g.:
///
/// ```
/// %0 = vector.transfer_write %v, %t1[%c0, %c0] {in_bounds = [true, true]}
///     : vector<4x5xf32>, tensor<4x5xf32>
/// %1 = tensor.insert_slice %0 into %t2[%a, %b] [4, 5] [1, 1]
///     : tensor<4x5xf32> into tensor<?x?xf32>
/// ```
/// is rewritten to:
/// ```
/// %1 = vector.transfer_write %v, %t2[%a, %b] {in_bounds = [true, true]}
///     : vector<4x5xf32>, tensor<?x?xf32>
/// ```
struct FoldInsertSliceIntoTransferWrite
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (!insertOp.hasUnitStride())
      return failure();

    auto xferOp = insertOp.source().getDefiningOp<TransferWriteOp>();
    if (!xferOp)
      return failure();
    // TODO: support 0-d corner case.
    if (xferOp.getTransferRank() == 0)
      return failure();

    if (xferOp.hasOutOfBoundsDim())
      return failure();
    if (xferOp.getVectorType().getRank() != xferOp.getShapedType().getRank())
      return failure();
    if (xferOp.mask())
      return failure();
    // Fold only if the TransferWriteOp completely overwrites the `source` with
    // a vector. I.e., the result of the TransferWriteOp is a new tensor whose
    // content is the data of the vector.
    if (!llvm::equal(xferOp.getVectorType().getShape(),
                     xferOp.getShapedType().getShape()))
      return failure();
    if (!xferOp.permutation_map().isIdentity())
      return failure();

    // Bail on illegal rank-reduction: we need to check that the rank-reduced
    // dims are exactly the leading dims. I.e. the following is illegal:
    // ```
    //    %0 = vector.transfer_write %v, %t[0,0], %cst :
    //      vector<2x4xf32>, tensor<2x4xf32>
    //    %1 = tensor.insert_slice %0 into %tt[0,0,0][2,1,4][1,1,1] :
    //      tensor<2x4xf32> into tensor<2x1x4xf32>
    // ```
    //
    // Cannot fold into:
    // ```
    //    %0 = vector.transfer_write %v, %t[0,0,0], %cst :
    //      vector<2x4xf32>, tensor<2x1x4xf32>
    // ```
    // For this, check the trailing `vectorRank` dims of the insert_slice result
    // tensor match the trailing dims of the inferred result tensor.
    int64_t rankReduced =
        insertOp.getType().getRank() - insertOp.getSourceType().getRank();
    int64_t vectorRank = xferOp.getVectorType().getRank();
    RankedTensorType inferredSourceTensorType =
        tensor::ExtractSliceOp::inferResultType(
            insertOp.getType(), insertOp.getMixedOffsets(),
            insertOp.getMixedSizes(), insertOp.getMixedStrides());
    auto actualSourceTensorShape = insertOp.getSourceType().getShape();
    if (rankReduced > 0 &&
        actualSourceTensorShape.take_back(vectorRank) !=
            inferredSourceTensorType.getShape().take_back(vectorRank))
      return failure();

    SmallVector<Value> indices = getValueOrCreateConstantIndexOp(
        rewriter, insertOp.getLoc(), insertOp.getMixedOffsets());
    SmallVector<bool> inBounds(xferOp.getTransferRank(), true);
    rewriter.replaceOpWithNewOp<TransferWriteOp>(insertOp, xferOp.vector(),
                                                 insertOp.dest(), indices,
                                                 ArrayRef<bool>{inBounds});
    return success();
  }
};
} // namespace

void TransferWriteOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<FoldWaw, FoldInsertSliceIntoTransferWrite>(context);
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyLoadStoreMemRefLayout(Operation *op,
                                                 MemRefType memRefTy) {
  if (!isLastMemrefDimUnitStride(memRefTy))
    return op->emitOpError("most minor memref dim must have unit stride");
  return success();
}

static LogicalResult verify(vector::LoadOp op) {
  VectorType resVecTy = op.getVectorType();
  MemRefType memRefTy = op.getMemRefType();

  if (failed(verifyLoadStoreMemRefLayout(op, memRefTy)))
    return failure();

  // Checks for vector memrefs.
  Type memElemTy = memRefTy.getElementType();
  if (auto memVecTy = memElemTy.dyn_cast<VectorType>()) {
    if (memVecTy != resVecTy)
      return op.emitOpError("base memref and result vector types should match");
    memElemTy = memVecTy.getElementType();
  }

  if (resVecTy.getElementType() != memElemTy)
    return op.emitOpError("base and result element types should match");
  if (llvm::size(op.indices()) != memRefTy.getRank())
    return op.emitOpError("requires ") << memRefTy.getRank() << " indices";
  return success();
}

OpFoldResult LoadOp::fold(ArrayRef<Attribute>) {
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return OpFoldResult();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(vector::StoreOp op) {
  VectorType valueVecTy = op.getVectorType();
  MemRefType memRefTy = op.getMemRefType();

  if (failed(verifyLoadStoreMemRefLayout(op, memRefTy)))
    return failure();

  // Checks for vector memrefs.
  Type memElemTy = memRefTy.getElementType();
  if (auto memVecTy = memElemTy.dyn_cast<VectorType>()) {
    if (memVecTy != valueVecTy)
      return op.emitOpError(
          "base memref and valueToStore vector types should match");
    memElemTy = memVecTy.getElementType();
  }

  if (valueVecTy.getElementType() != memElemTy)
    return op.emitOpError("base and valueToStore element type should match");
  if (llvm::size(op.indices()) != memRefTy.getRank())
    return op.emitOpError("requires ") << memRefTy.getRank() << " indices";
  return success();
}

LogicalResult StoreOp::fold(ArrayRef<Attribute> operands,
                            SmallVectorImpl<OpFoldResult> &results) {
  return foldMemRefCast(*this);
}

//===----------------------------------------------------------------------===//
// MaskedLoadOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(MaskedLoadOp op) {
  VectorType maskVType = op.getMaskVectorType();
  VectorType passVType = op.getPassThruVectorType();
  VectorType resVType = op.getVectorType();
  MemRefType memType = op.getMemRefType();

  if (resVType.getElementType() != memType.getElementType())
    return op.emitOpError("base and result element type should match");
  if (llvm::size(op.indices()) != memType.getRank())
    return op.emitOpError("requires ") << memType.getRank() << " indices";
  if (resVType.getDimSize(0) != maskVType.getDimSize(0))
    return op.emitOpError("expected result dim to match mask dim");
  if (resVType != passVType)
    return op.emitOpError("expected pass_thru of same type as result type");
  return success();
}

namespace {
class MaskedLoadFolder final : public OpRewritePattern<MaskedLoadOp> {
public:
  using OpRewritePattern<MaskedLoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MaskedLoadOp load,
                                PatternRewriter &rewriter) const override {
    switch (get1DMaskFormat(load.mask())) {
    case MaskFormat::AllTrue:
      rewriter.replaceOpWithNewOp<vector::LoadOp>(load, load.getType(),
                                                  load.base(), load.indices());
      return success();
    case MaskFormat::AllFalse:
      rewriter.replaceOp(load, load.pass_thru());
      return success();
    case MaskFormat::Unknown:
      return failure();
    }
    llvm_unreachable("Unexpected 1DMaskFormat on MaskedLoad");
  }
};
} // namespace

void MaskedLoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<MaskedLoadFolder>(context);
}

OpFoldResult MaskedLoadOp::fold(ArrayRef<Attribute>) {
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return OpFoldResult();
}

//===----------------------------------------------------------------------===//
// MaskedStoreOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(MaskedStoreOp op) {
  VectorType maskVType = op.getMaskVectorType();
  VectorType valueVType = op.getVectorType();
  MemRefType memType = op.getMemRefType();

  if (valueVType.getElementType() != memType.getElementType())
    return op.emitOpError("base and valueToStore element type should match");
  if (llvm::size(op.indices()) != memType.getRank())
    return op.emitOpError("requires ") << memType.getRank() << " indices";
  if (valueVType.getDimSize(0) != maskVType.getDimSize(0))
    return op.emitOpError("expected valueToStore dim to match mask dim");
  return success();
}

namespace {
class MaskedStoreFolder final : public OpRewritePattern<MaskedStoreOp> {
public:
  using OpRewritePattern<MaskedStoreOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MaskedStoreOp store,
                                PatternRewriter &rewriter) const override {
    switch (get1DMaskFormat(store.mask())) {
    case MaskFormat::AllTrue:
      rewriter.replaceOpWithNewOp<vector::StoreOp>(
          store, store.valueToStore(), store.base(), store.indices());
      return success();
    case MaskFormat::AllFalse:
      rewriter.eraseOp(store);
      return success();
    case MaskFormat::Unknown:
      return failure();
    }
    llvm_unreachable("Unexpected 1DMaskFormat on MaskedStore");
  }
};
} // namespace

void MaskedStoreOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<MaskedStoreFolder>(context);
}

LogicalResult MaskedStoreOp::fold(ArrayRef<Attribute> operands,
                                  SmallVectorImpl<OpFoldResult> &results) {
  return foldMemRefCast(*this);
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(GatherOp op) {
  VectorType indVType = op.getIndexVectorType();
  VectorType maskVType = op.getMaskVectorType();
  VectorType resVType = op.getVectorType();
  MemRefType memType = op.getMemRefType();

  if (resVType.getElementType() != memType.getElementType())
    return op.emitOpError("base and result element type should match");
  if (llvm::size(op.indices()) != memType.getRank())
    return op.emitOpError("requires ") << memType.getRank() << " indices";
  if (resVType.getDimSize(0) != indVType.getDimSize(0))
    return op.emitOpError("expected result dim to match indices dim");
  if (resVType.getDimSize(0) != maskVType.getDimSize(0))
    return op.emitOpError("expected result dim to match mask dim");
  if (resVType != op.getPassThruVectorType())
    return op.emitOpError("expected pass_thru of same type as result type");
  return success();
}

namespace {
class GatherFolder final : public OpRewritePattern<GatherOp> {
public:
  using OpRewritePattern<GatherOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GatherOp gather,
                                PatternRewriter &rewriter) const override {
    switch (get1DMaskFormat(gather.mask())) {
    case MaskFormat::AllTrue:
      return failure(); // no unmasked equivalent
    case MaskFormat::AllFalse:
      rewriter.replaceOp(gather, gather.pass_thru());
      return success();
    case MaskFormat::Unknown:
      return failure();
    }
    llvm_unreachable("Unexpected 1DMaskFormat on GatherFolder");
  }
};
} // namespace

void GatherOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<GatherFolder>(context);
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ScatterOp op) {
  VectorType indVType = op.getIndexVectorType();
  VectorType maskVType = op.getMaskVectorType();
  VectorType valueVType = op.getVectorType();
  MemRefType memType = op.getMemRefType();

  if (valueVType.getElementType() != memType.getElementType())
    return op.emitOpError("base and valueToStore element type should match");
  if (llvm::size(op.indices()) != memType.getRank())
    return op.emitOpError("requires ") << memType.getRank() << " indices";
  if (valueVType.getDimSize(0) != indVType.getDimSize(0))
    return op.emitOpError("expected valueToStore dim to match indices dim");
  if (valueVType.getDimSize(0) != maskVType.getDimSize(0))
    return op.emitOpError("expected valueToStore dim to match mask dim");
  return success();
}

namespace {
class ScatterFolder final : public OpRewritePattern<ScatterOp> {
public:
  using OpRewritePattern<ScatterOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ScatterOp scatter,
                                PatternRewriter &rewriter) const override {
    switch (get1DMaskFormat(scatter.mask())) {
    case MaskFormat::AllTrue:
      return failure(); // no unmasked equivalent
    case MaskFormat::AllFalse:
      rewriter.eraseOp(scatter);
      return success();
    case MaskFormat::Unknown:
      return failure();
    }
    llvm_unreachable("Unexpected 1DMaskFormat on ScatterFolder");
  }
};
} // namespace

void ScatterOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ScatterFolder>(context);
}

//===----------------------------------------------------------------------===//
// ExpandLoadOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ExpandLoadOp op) {
  VectorType maskVType = op.getMaskVectorType();
  VectorType passVType = op.getPassThruVectorType();
  VectorType resVType = op.getVectorType();
  MemRefType memType = op.getMemRefType();

  if (resVType.getElementType() != memType.getElementType())
    return op.emitOpError("base and result element type should match");
  if (llvm::size(op.indices()) != memType.getRank())
    return op.emitOpError("requires ") << memType.getRank() << " indices";
  if (resVType.getDimSize(0) != maskVType.getDimSize(0))
    return op.emitOpError("expected result dim to match mask dim");
  if (resVType != passVType)
    return op.emitOpError("expected pass_thru of same type as result type");
  return success();
}

namespace {
class ExpandLoadFolder final : public OpRewritePattern<ExpandLoadOp> {
public:
  using OpRewritePattern<ExpandLoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ExpandLoadOp expand,
                                PatternRewriter &rewriter) const override {
    switch (get1DMaskFormat(expand.mask())) {
    case MaskFormat::AllTrue:
      rewriter.replaceOpWithNewOp<vector::LoadOp>(
          expand, expand.getType(), expand.base(), expand.indices());
      return success();
    case MaskFormat::AllFalse:
      rewriter.replaceOp(expand, expand.pass_thru());
      return success();
    case MaskFormat::Unknown:
      return failure();
    }
    llvm_unreachable("Unexpected 1DMaskFormat on ExpandLoadFolder");
  }
};
} // namespace

void ExpandLoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<ExpandLoadFolder>(context);
}

//===----------------------------------------------------------------------===//
// CompressStoreOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(CompressStoreOp op) {
  VectorType maskVType = op.getMaskVectorType();
  VectorType valueVType = op.getVectorType();
  MemRefType memType = op.getMemRefType();

  if (valueVType.getElementType() != memType.getElementType())
    return op.emitOpError("base and valueToStore element type should match");
  if (llvm::size(op.indices()) != memType.getRank())
    return op.emitOpError("requires ") << memType.getRank() << " indices";
  if (valueVType.getDimSize(0) != maskVType.getDimSize(0))
    return op.emitOpError("expected valueToStore dim to match mask dim");
  return success();
}

namespace {
class CompressStoreFolder final : public OpRewritePattern<CompressStoreOp> {
public:
  using OpRewritePattern<CompressStoreOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CompressStoreOp compress,
                                PatternRewriter &rewriter) const override {
    switch (get1DMaskFormat(compress.mask())) {
    case MaskFormat::AllTrue:
      rewriter.replaceOpWithNewOp<vector::StoreOp>(
          compress, compress.valueToStore(), compress.base(),
          compress.indices());
      return success();
    case MaskFormat::AllFalse:
      rewriter.eraseOp(compress);
      return success();
    case MaskFormat::Unknown:
      return failure();
    }
    llvm_unreachable("Unexpected 1DMaskFormat on CompressStoreFolder");
  }
};
} // namespace

void CompressStoreOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<CompressStoreFolder>(context);
}

//===----------------------------------------------------------------------===//
// ShapeCastOp
//===----------------------------------------------------------------------===//

/// Returns true if each element of 'a' is equal to the product of a contiguous
/// sequence of the elements of 'b'. Returns false otherwise.
static bool isValidShapeCast(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  unsigned rankA = a.size();
  unsigned rankB = b.size();
  assert(rankA < rankB);

  unsigned i = 0;
  unsigned j = 0;
  while (i < rankA && j < rankB) {
    int64_t dimA = a[i];
    int64_t dimB = 1;
    while (dimB < dimA && j < rankB)
      dimB *= b[j++];
    if (dimA != dimB)
      break;
    ++i;

    // Handle the case when trailing dimensions are of size 1.
    // Include them into the contiguous sequence.
    auto isOne = [](int64_t v) { return v == 1; };
    if (i < rankA && llvm::all_of(a.slice(i), isOne))
      i = rankA;
    if (j < rankB && llvm::all_of(b.slice(j), isOne))
      j = rankB;
  }

  return i == rankA && j == rankB;
}

static LogicalResult verifyVectorShapeCast(Operation *op,
                                           VectorType sourceVectorType,
                                           VectorType resultVectorType) {
  // Check that element type is the same.
  if (sourceVectorType.getElementType() != resultVectorType.getElementType())
    return op->emitOpError("source/result vectors must have same element type");
  auto sourceShape = sourceVectorType.getShape();
  auto resultShape = resultVectorType.getShape();

  // Check that product of source dim sizes matches product of result dim sizes.
  int64_t sourceDimProduct = std::accumulate(
      sourceShape.begin(), sourceShape.end(), 1LL, std::multiplies<int64_t>{});
  int64_t resultDimProduct = std::accumulate(
      resultShape.begin(), resultShape.end(), 1LL, std::multiplies<int64_t>{});
  if (sourceDimProduct != resultDimProduct)
    return op->emitOpError("source/result number of elements must match");

  // Check that expanding/contracting rank cases.
  unsigned sourceRank = sourceVectorType.getRank();
  unsigned resultRank = resultVectorType.getRank();
  if (sourceRank < resultRank) {
    if (!isValidShapeCast(sourceShape, resultShape))
      return op->emitOpError("invalid shape cast");
  } else if (sourceRank > resultRank) {
    if (!isValidShapeCast(resultShape, sourceShape))
      return op->emitOpError("invalid shape cast");
  }
  return success();
}

static LogicalResult verify(ShapeCastOp op) {
  auto sourceVectorType = op.source().getType().dyn_cast_or_null<VectorType>();
  auto resultVectorType = op.result().getType().dyn_cast_or_null<VectorType>();

  // Check if source/result are of vector type.
  if (sourceVectorType && resultVectorType)
    return verifyVectorShapeCast(op, sourceVectorType, resultVectorType);

  return success();
}

OpFoldResult ShapeCastOp::fold(ArrayRef<Attribute> operands) {
  // Nop shape cast.
  if (source().getType() == result().getType())
    return source();

  // Canceling shape casts.
  if (auto otherOp = source().getDefiningOp<ShapeCastOp>()) {
    if (result().getType() == otherOp.source().getType())
      return otherOp.source();

    // Only allows valid transitive folding.
    VectorType srcType = otherOp.source().getType().cast<VectorType>();
    VectorType resultType = getResult().getType().cast<VectorType>();
    if (srcType.getRank() < resultType.getRank()) {
      if (!isValidShapeCast(srcType.getShape(), resultType.getShape()))
        return {};
    } else if (srcType.getRank() > resultType.getRank()) {
      if (!isValidShapeCast(resultType.getShape(), srcType.getShape()))
        return {};
    } else {
      return {};
    }

    setOperand(otherOp.source());
    return getResult();
  }
  return {};
}

namespace {
// Pattern to rewrite a ShapeCast(splat ConstantOp) -> ConstantOp.
class ShapeCastConstantFolder final : public OpRewritePattern<ShapeCastOp> {
public:
  using OpRewritePattern<ShapeCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ShapeCastOp shapeCastOp,
                                PatternRewriter &rewriter) const override {
    auto constantOp = shapeCastOp.source().getDefiningOp<arith::ConstantOp>();
    if (!constantOp)
      return failure();
    // Only handle splat for now.
    auto dense = constantOp.getValue().dyn_cast<SplatElementsAttr>();
    if (!dense)
      return failure();
    auto newAttr =
        DenseElementsAttr::get(shapeCastOp.getType().cast<VectorType>(),
                               dense.getSplatValue<Attribute>());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(shapeCastOp, newAttr);
    return success();
  }
};

} // namespace

void ShapeCastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  // Pattern to rewrite a ShapeCastOp(ConstantOp) -> ConstantOp.
  results.add<ShapeCastConstantFolder>(context);
}

//===----------------------------------------------------------------------===//
// VectorBitCastOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(BitCastOp op) {
  auto sourceVectorType = op.getSourceVectorType();
  auto resultVectorType = op.getResultVectorType();

  for (int64_t i = 0, e = sourceVectorType.getRank() - 1; i < e; i++) {
    if (sourceVectorType.getDimSize(i) != resultVectorType.getDimSize(i))
      return op.emitOpError("dimension size mismatch at: ") << i;
  }

  DataLayout dataLayout = DataLayout::closest(op);
  auto sourceElementBits =
      dataLayout.getTypeSizeInBits(sourceVectorType.getElementType());
  auto resultElementBits =
      dataLayout.getTypeSizeInBits(resultVectorType.getElementType());

  if (sourceVectorType.getRank() == 0) {
    if (sourceElementBits != resultElementBits)
      return op.emitOpError("source/result bitwidth of the 0-D vector element "
                            "types must be equal");
  } else if (sourceElementBits * sourceVectorType.getShape().back() !=
             resultElementBits * resultVectorType.getShape().back()) {
    return op.emitOpError(
        "source/result bitwidth of the minor 1-D vectors must be equal");
  }

  return success();
}

OpFoldResult BitCastOp::fold(ArrayRef<Attribute> operands) {
  // Nop cast.
  if (source().getType() == result().getType())
    return source();

  // Canceling bitcasts.
  if (auto otherOp = source().getDefiningOp<BitCastOp>())
    if (result().getType() == otherOp.source().getType())
      return otherOp.source();

  Attribute sourceConstant = operands.front();
  if (!sourceConstant)
    return {};

  Type srcElemType = getSourceVectorType().getElementType();
  Type dstElemType = getResultVectorType().getElementType();

  if (auto floatPack = sourceConstant.dyn_cast<DenseFPElementsAttr>()) {
    if (floatPack.isSplat()) {
      auto splat = floatPack.getSplatValue<FloatAttr>();

      // Casting fp16 into fp32.
      if (srcElemType.isF16() && dstElemType.isF32()) {
        uint32_t bits = static_cast<uint32_t>(
            splat.getValue().bitcastToAPInt().getZExtValue());
        // Duplicate the 16-bit pattern.
        bits = (bits << 16) | (bits & 0xffff);
        APInt intBits(32, bits);
        APFloat floatBits(llvm::APFloat::IEEEsingle(), intBits);
        return DenseElementsAttr::get(getResultVectorType(), floatBits);
      }
    }
  }

  return {};
}

//===----------------------------------------------------------------------===//
// TypeCastOp
//===----------------------------------------------------------------------===//

static SmallVector<int64_t, 8> extractShape(MemRefType memRefType) {
  auto vectorType = memRefType.getElementType().dyn_cast<VectorType>();
  SmallVector<int64_t, 8> res(memRefType.getShape().begin(),
                              memRefType.getShape().end());
  if (vectorType)
    res.append(vectorType.getShape().begin(), vectorType.getShape().end());
  return res;
}

/// Build the canonical memRefType with a single vector.
/// E.g. memref<4 x 5 x vector<6 x f32>> -> memref<vector<4 x 5 x 6 x f32>>.
void TypeCastOp::build(OpBuilder &builder, OperationState &result,
                       Value source) {
  result.addOperands(source);
  MemRefType memRefType = source.getType().cast<MemRefType>();
  VectorType vectorType =
      VectorType::get(extractShape(memRefType),
                      getElementTypeOrSelf(getElementTypeOrSelf(memRefType)));
  result.addTypes(MemRefType::get({}, vectorType, MemRefLayoutAttrInterface(),
                                  memRefType.getMemorySpace()));
}

static LogicalResult verify(TypeCastOp op) {
  MemRefType canonicalType = canonicalizeStridedLayout(op.getMemRefType());
  if (!canonicalType.getLayout().isIdentity())
    return op.emitOpError(
        "expects operand to be a memref with identity layout");
  if (!op.getResultMemRefType().getLayout().isIdentity())
    return op.emitOpError("expects result to be a memref with identity layout");
  if (op.getResultMemRefType().getMemorySpace() !=
      op.getMemRefType().getMemorySpace())
    return op.emitOpError("expects result in same memory space");

  auto sourceType = op.getMemRefType();
  auto resultType = op.getResultMemRefType();
  if (getElementTypeOrSelf(getElementTypeOrSelf(sourceType)) !=
      getElementTypeOrSelf(getElementTypeOrSelf(resultType)))
    return op.emitOpError(
               "expects result and operand with same underlying scalar type: ")
           << resultType;
  if (extractShape(sourceType) != extractShape(resultType))
    return op.emitOpError(
               "expects concatenated result and operand shapes to be equal: ")
           << resultType;
  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

void vector::TransposeOp::build(OpBuilder &builder, OperationState &result,
                                Value vector, ArrayRef<int64_t> transp) {
  VectorType vt = vector.getType().cast<VectorType>();
  SmallVector<int64_t, 4> transposedShape(vt.getRank());
  for (unsigned i = 0; i < transp.size(); ++i)
    transposedShape[i] = vt.getShape()[transp[i]];

  result.addOperands(vector);
  result.addTypes(VectorType::get(transposedShape, vt.getElementType()));
  result.addAttribute(getTranspAttrName(), builder.getI64ArrayAttr(transp));
}

// Eliminates transpose operations, which produce values identical to their
// input values. This happens when the dimensions of the input vector remain in
// their original order after the transpose operation.
OpFoldResult vector::TransposeOp::fold(ArrayRef<Attribute> operands) {
  SmallVector<int64_t, 4> transp;
  getTransp(transp);

  // Check if the permutation of the dimensions contains sequential values:
  // {0, 1, 2, ...}.
  for (int64_t i = 0, e = transp.size(); i < e; i++) {
    if (transp[i] != i)
      return {};
  }

  return vector();
}

static LogicalResult verify(vector::TransposeOp op) {
  VectorType vectorType = op.getVectorType();
  VectorType resultType = op.getResultType();
  int64_t rank = resultType.getRank();
  if (vectorType.getRank() != rank)
    return op.emitOpError("vector result rank mismatch: ") << rank;
  // Verify transposition array.
  auto transpAttr = op.transp().getValue();
  int64_t size = transpAttr.size();
  if (rank != size)
    return op.emitOpError("transposition length mismatch: ") << size;
  SmallVector<bool, 8> seen(rank, false);
  for (const auto &ta : llvm::enumerate(transpAttr)) {
    int64_t i = ta.value().cast<IntegerAttr>().getInt();
    if (i < 0 || i >= rank)
      return op.emitOpError("transposition index out of range: ") << i;
    if (seen[i])
      return op.emitOpError("duplicate position index: ") << i;
    seen[i] = true;
    if (resultType.getDimSize(ta.index()) != vectorType.getDimSize(i))
      return op.emitOpError("dimension size mismatch at: ") << i;
  }
  return success();
}

namespace {

// Rewrites two back-to-back TransposeOp operations into a single TransposeOp.
class TransposeFolder final : public OpRewritePattern<vector::TransposeOp> {
public:
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    // Wrapper around vector::TransposeOp::getTransp() for cleaner code.
    auto getPermutation = [](vector::TransposeOp transpose) {
      SmallVector<int64_t, 4> permutation;
      transpose.getTransp(permutation);
      return permutation;
    };

    // Composes two permutations: result[i] = permutation1[permutation2[i]].
    auto composePermutations = [](ArrayRef<int64_t> permutation1,
                                  ArrayRef<int64_t> permutation2) {
      SmallVector<int64_t, 4> result;
      for (auto index : permutation2)
        result.push_back(permutation1[index]);
      return result;
    };

    // Return if the input of 'transposeOp' is not defined by another transpose.
    vector::TransposeOp parentTransposeOp =
        transposeOp.vector().getDefiningOp<vector::TransposeOp>();
    if (!parentTransposeOp)
      return failure();

    SmallVector<int64_t, 4> permutation = composePermutations(
        getPermutation(parentTransposeOp), getPermutation(transposeOp));
    // Replace 'transposeOp' with a new transpose operation.
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(
        transposeOp, transposeOp.getResult().getType(),
        parentTransposeOp.vector(),
        vector::getVectorSubscriptAttr(rewriter, permutation));
    return success();
  }
};

} // namespace

void vector::TransposeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<TransposeFolder>(context);
}

void vector::TransposeOp::getTransp(SmallVectorImpl<int64_t> &results) {
  populateFromInt64AttrArray(transp(), results);
}

//===----------------------------------------------------------------------===//
// ConstantMaskOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ConstantMaskOp &op) {
  auto resultType = op.getResult().getType().cast<VectorType>();
  // Check the corner case of 0-D vectors first.
  if (resultType.getRank() == 0) {
    if (op.mask_dim_sizes().size() != 1)
      return op->emitError("array attr must have length 1 for 0-D vectors");
    auto dim = op.mask_dim_sizes()[0].cast<IntegerAttr>().getInt();
    if (dim != 0 && dim != 1)
      return op->emitError(
          "mask dim size must be either 0 or 1 for 0-D vectors");
    return success();
  }

  // Verify that array attr size matches the rank of the vector result.
  if (static_cast<int64_t>(op.mask_dim_sizes().size()) != resultType.getRank())
    return op.emitOpError(
        "must specify array attr of size equal vector result rank");
  // Verify that each array attr element is in bounds of corresponding vector
  // result dimension size.
  auto resultShape = resultType.getShape();
  SmallVector<int64_t, 4> maskDimSizes;
  for (const auto &it : llvm::enumerate(op.mask_dim_sizes())) {
    int64_t attrValue = it.value().cast<IntegerAttr>().getInt();
    if (attrValue < 0 || attrValue > resultShape[it.index()])
      return op.emitOpError(
          "array attr of size out of bounds of vector result dimension size");
    maskDimSizes.push_back(attrValue);
  }
  // Verify that if one mask dim size is zero, they all should be zero (because
  // the mask region is a conjunction of each mask dimension interval).
  bool anyZeros = llvm::is_contained(maskDimSizes, 0);
  bool allZeros = llvm::all_of(maskDimSizes, [](int64_t s) { return s == 0; });
  if (anyZeros && !allZeros)
    return op.emitOpError("expected all mask dim sizes to be zeros, "
                          "as a result of conjunction with zero mask dim");
  return success();
}

//===----------------------------------------------------------------------===//
// CreateMaskOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(CreateMaskOp op) {
  auto vectorType = op.getResult().getType().cast<VectorType>();
  // Verify that an operand was specified for each result vector each dimension.
  if (vectorType.getRank() == 0) {
    if (op->getNumOperands() != 1)
      return op.emitOpError(
          "must specify exactly one operand for 0-D create_mask");
  } else if (op.getNumOperands() !=
             op.getResult().getType().cast<VectorType>().getRank()) {
    return op.emitOpError(
        "must specify an operand for each result vector dimension");
  }
  return success();
}

namespace {

// Pattern to rewrite a CreateMaskOp with a ConstantMaskOp.
class CreateMaskFolder final : public OpRewritePattern<CreateMaskOp> {
public:
  using OpRewritePattern<CreateMaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CreateMaskOp createMaskOp,
                                PatternRewriter &rewriter) const override {
    // Return if any of 'createMaskOp' operands are not defined by a constant.
    auto isNotDefByConstant = [](Value operand) {
      return !isa_and_nonnull<arith::ConstantIndexOp>(operand.getDefiningOp());
    };
    if (llvm::any_of(createMaskOp.operands(), isNotDefByConstant))
      return failure();
    // Gather constant mask dimension sizes.
    SmallVector<int64_t, 4> maskDimSizes;
    for (auto operand : createMaskOp.operands()) {
      auto *defOp = operand.getDefiningOp();
      maskDimSizes.push_back(cast<arith::ConstantIndexOp>(defOp).value());
    }
    // Replace 'createMaskOp' with ConstantMaskOp.
    rewriter.replaceOpWithNewOp<ConstantMaskOp>(
        createMaskOp, createMaskOp.getResult().getType(),
        vector::getVectorSubscriptAttr(rewriter, maskDimSizes));
    return success();
  }
};

} // namespace

void CreateMaskOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<CreateMaskFolder>(context);
}

void mlir::vector::populateVectorToVectorCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<CreateMaskFolder, MaskedLoadFolder, MaskedStoreFolder, GatherFolder,
           ScatterFolder, ExpandLoadFolder, CompressStoreFolder,
           StridedSliceConstantMaskFolder, TransposeFolder>(
          patterns.getContext());
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Vector/VectorOps.cpp.inc"
