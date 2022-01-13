//===- Builders.cpp - Helpers for constructing MLIR Classes ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

Identifier Builder::getIdentifier(const Twine &str) {
  return Identifier::get(str, context);
}

//===----------------------------------------------------------------------===//
// Locations.
//===----------------------------------------------------------------------===//

Location Builder::getUnknownLoc() { return UnknownLoc::get(context); }

Location Builder::getFusedLoc(ArrayRef<Location> locs, Attribute metadata) {
  return FusedLoc::get(locs, metadata, context);
}

//===----------------------------------------------------------------------===//
// Types.
//===----------------------------------------------------------------------===//

FloatType Builder::getBF16Type() { return FloatType::getBF16(context); }

FloatType Builder::getF16Type() { return FloatType::getF16(context); }

FloatType Builder::getF32Type() { return FloatType::getF32(context); }

FloatType Builder::getF64Type() { return FloatType::getF64(context); }

FloatType Builder::getF80Type() { return FloatType::getF80(context); }

FloatType Builder::getF128Type() { return FloatType::getF128(context); }

IndexType Builder::getIndexType() { return IndexType::get(context); }

IntegerType Builder::getI1Type() { return IntegerType::get(context, 1); }

IntegerType Builder::getI8Type() { return IntegerType::get(context, 8); }

IntegerType Builder::getI32Type() { return IntegerType::get(context, 32); }

IntegerType Builder::getI64Type() { return IntegerType::get(context, 64); }

IntegerType Builder::getIntegerType(unsigned width) {
  return IntegerType::get(context, width);
}

IntegerType Builder::getIntegerType(unsigned width, bool isSigned) {
  return IntegerType::get(
      context, width, isSigned ? IntegerType::Signed : IntegerType::Unsigned);
}

FunctionType Builder::getFunctionType(TypeRange inputs, TypeRange results) {
  return FunctionType::get(context, inputs, results);
}

TupleType Builder::getTupleType(TypeRange elementTypes) {
  return TupleType::get(context, elementTypes);
}

NoneType Builder::getNoneType() { return NoneType::get(context); }

//===----------------------------------------------------------------------===//
// Attributes.
//===----------------------------------------------------------------------===//

NamedAttribute Builder::getNamedAttr(StringRef name, Attribute val) {
  return NamedAttribute(getIdentifier(name), val);
}

UnitAttr Builder::getUnitAttr() { return UnitAttr::get(context); }

BoolAttr Builder::getBoolAttr(bool value) {
  return BoolAttr::get(context, value);
}

DictionaryAttr Builder::getDictionaryAttr(ArrayRef<NamedAttribute> value) {
  return DictionaryAttr::get(context, value);
}

IntegerAttr Builder::getIndexAttr(int64_t value) {
  return IntegerAttr::get(getIndexType(), APInt(64, value));
}

IntegerAttr Builder::getI64IntegerAttr(int64_t value) {
  return IntegerAttr::get(getIntegerType(64), APInt(64, value));
}

DenseIntElementsAttr Builder::getBoolVectorAttr(ArrayRef<bool> values) {
  return DenseIntElementsAttr::get(
      VectorType::get(static_cast<int64_t>(values.size()), getI1Type()),
      values);
}

DenseIntElementsAttr Builder::getI32VectorAttr(ArrayRef<int32_t> values) {
  return DenseIntElementsAttr::get(
      VectorType::get(static_cast<int64_t>(values.size()), getIntegerType(32)),
      values);
}

DenseIntElementsAttr Builder::getI64VectorAttr(ArrayRef<int64_t> values) {
  return DenseIntElementsAttr::get(
      VectorType::get(static_cast<int64_t>(values.size()), getIntegerType(64)),
      values);
}

DenseIntElementsAttr Builder::getIndexVectorAttr(ArrayRef<int64_t> values) {
  return DenseIntElementsAttr::get(
      VectorType::get(static_cast<int64_t>(values.size()), getIndexType()),
      values);
}

DenseIntElementsAttr Builder::getI32TensorAttr(ArrayRef<int32_t> values) {
  return DenseIntElementsAttr::get(
      RankedTensorType::get(static_cast<int64_t>(values.size()),
                            getIntegerType(32)),
      values);
}

DenseIntElementsAttr Builder::getI64TensorAttr(ArrayRef<int64_t> values) {
  return DenseIntElementsAttr::get(
      RankedTensorType::get(static_cast<int64_t>(values.size()),
                            getIntegerType(64)),
      values);
}

DenseIntElementsAttr Builder::getIndexTensorAttr(ArrayRef<int64_t> values) {
  return DenseIntElementsAttr::get(
      RankedTensorType::get(static_cast<int64_t>(values.size()),
                            getIndexType()),
      values);
}

IntegerAttr Builder::getI32IntegerAttr(int32_t value) {
  return IntegerAttr::get(getIntegerType(32), APInt(32, value));
}

IntegerAttr Builder::getSI32IntegerAttr(int32_t value) {
  return IntegerAttr::get(getIntegerType(32, /*isSigned=*/true),
                          APInt(32, value, /*isSigned=*/true));
}

IntegerAttr Builder::getUI32IntegerAttr(uint32_t value) {
  return IntegerAttr::get(getIntegerType(32, /*isSigned=*/false),
                          APInt(32, (uint64_t)value, /*isSigned=*/false));
}

IntegerAttr Builder::getI16IntegerAttr(int16_t value) {
  return IntegerAttr::get(getIntegerType(16), APInt(16, value));
}

IntegerAttr Builder::getI8IntegerAttr(int8_t value) {
  return IntegerAttr::get(getIntegerType(8), APInt(8, value));
}

IntegerAttr Builder::getIntegerAttr(Type type, int64_t value) {
  if (type.isIndex())
    return IntegerAttr::get(type, APInt(64, value));
  return IntegerAttr::get(
      type, APInt(type.getIntOrFloatBitWidth(), value, type.isSignedInteger()));
}

IntegerAttr Builder::getIntegerAttr(Type type, const APInt &value) {
  return IntegerAttr::get(type, value);
}

FloatAttr Builder::getF64FloatAttr(double value) {
  return FloatAttr::get(getF64Type(), APFloat(value));
}

FloatAttr Builder::getF32FloatAttr(float value) {
  return FloatAttr::get(getF32Type(), APFloat(value));
}

FloatAttr Builder::getF16FloatAttr(float value) {
  return FloatAttr::get(getF16Type(), value);
}

FloatAttr Builder::getFloatAttr(Type type, double value) {
  return FloatAttr::get(type, value);
}

FloatAttr Builder::getFloatAttr(Type type, const APFloat &value) {
  return FloatAttr::get(type, value);
}

StringAttr Builder::getStringAttr(const Twine &bytes) {
  return StringAttr::get(context, bytes);
}

ArrayAttr Builder::getArrayAttr(ArrayRef<Attribute> value) {
  return ArrayAttr::get(context, value);
}

FlatSymbolRefAttr Builder::getSymbolRefAttr(Operation *value) {
  auto symName =
      value->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
  assert(symName && "value does not have a valid symbol name");
  return getSymbolRefAttr(symName.getValue());
}
FlatSymbolRefAttr Builder::getSymbolRefAttr(StringRef value) {
  return SymbolRefAttr::get(getContext(), value);
}
SymbolRefAttr
Builder::getSymbolRefAttr(StringRef value,
                          ArrayRef<FlatSymbolRefAttr> nestedReferences) {
  return SymbolRefAttr::get(getContext(), value, nestedReferences);
}

ArrayAttr Builder::getBoolArrayAttr(ArrayRef<bool> values) {
  auto attrs = llvm::to_vector<8>(llvm::map_range(
      values, [this](bool v) -> Attribute { return getBoolAttr(v); }));
  return getArrayAttr(attrs);
}

ArrayAttr Builder::getI32ArrayAttr(ArrayRef<int32_t> values) {
  auto attrs = llvm::to_vector<8>(llvm::map_range(
      values, [this](int32_t v) -> Attribute { return getI32IntegerAttr(v); }));
  return getArrayAttr(attrs);
}
ArrayAttr Builder::getI64ArrayAttr(ArrayRef<int64_t> values) {
  auto attrs = llvm::to_vector<8>(llvm::map_range(
      values, [this](int64_t v) -> Attribute { return getI64IntegerAttr(v); }));
  return getArrayAttr(attrs);
}

ArrayAttr Builder::getIndexArrayAttr(ArrayRef<int64_t> values) {
  auto attrs = llvm::to_vector<8>(
      llvm::map_range(values, [this](int64_t v) -> Attribute {
        return getIntegerAttr(IndexType::get(getContext()), v);
      }));
  return getArrayAttr(attrs);
}

ArrayAttr Builder::getF32ArrayAttr(ArrayRef<float> values) {
  auto attrs = llvm::to_vector<8>(llvm::map_range(
      values, [this](float v) -> Attribute { return getF32FloatAttr(v); }));
  return getArrayAttr(attrs);
}

ArrayAttr Builder::getF64ArrayAttr(ArrayRef<double> values) {
  auto attrs = llvm::to_vector<8>(llvm::map_range(
      values, [this](double v) -> Attribute { return getF64FloatAttr(v); }));
  return getArrayAttr(attrs);
}

ArrayAttr Builder::getStrArrayAttr(ArrayRef<StringRef> values) {
  auto attrs = llvm::to_vector<8>(llvm::map_range(
      values, [this](StringRef v) -> Attribute { return getStringAttr(v); }));
  return getArrayAttr(attrs);
}

ArrayAttr Builder::getTypeArrayAttr(TypeRange values) {
  auto attrs = llvm::to_vector<8>(llvm::map_range(
      values, [](Type v) -> Attribute { return TypeAttr::get(v); }));
  return getArrayAttr(attrs);
}

ArrayAttr Builder::getAffineMapArrayAttr(ArrayRef<AffineMap> values) {
  auto attrs = llvm::to_vector<8>(llvm::map_range(
      values, [](AffineMap v) -> Attribute { return AffineMapAttr::get(v); }));
  return getArrayAttr(attrs);
}

Attribute Builder::getZeroAttr(Type type) {
  if (type.isa<FloatType>())
    return getFloatAttr(type, 0.0);
  if (type.isa<IndexType>())
    return getIndexAttr(0);
  if (auto integerType = type.dyn_cast<IntegerType>())
    return getIntegerAttr(type, APInt(type.cast<IntegerType>().getWidth(), 0));
  if (type.isa<RankedTensorType, VectorType>()) {
    auto vtType = type.cast<ShapedType>();
    auto element = getZeroAttr(vtType.getElementType());
    if (!element)
      return {};
    return DenseElementsAttr::get(vtType, element);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Affine Expressions, Affine Maps, and Integer Sets.
//===----------------------------------------------------------------------===//

AffineExpr Builder::getAffineDimExpr(unsigned position) {
  return mlir::getAffineDimExpr(position, context);
}

AffineExpr Builder::getAffineSymbolExpr(unsigned position) {
  return mlir::getAffineSymbolExpr(position, context);
}

AffineExpr Builder::getAffineConstantExpr(int64_t constant) {
  return mlir::getAffineConstantExpr(constant, context);
}

AffineMap Builder::getEmptyAffineMap() { return AffineMap::get(context); }

AffineMap Builder::getConstantAffineMap(int64_t val) {
  return AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0,
                        getAffineConstantExpr(val));
}

AffineMap Builder::getDimIdentityMap() {
  return AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, getAffineDimExpr(0));
}

AffineMap Builder::getMultiDimIdentityMap(unsigned rank) {
  SmallVector<AffineExpr, 4> dimExprs;
  dimExprs.reserve(rank);
  for (unsigned i = 0; i < rank; ++i)
    dimExprs.push_back(getAffineDimExpr(i));
  return AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/0, dimExprs,
                        context);
}

AffineMap Builder::getSymbolIdentityMap() {
  return AffineMap::get(/*dimCount=*/0, /*symbolCount=*/1,
                        getAffineSymbolExpr(0));
}

AffineMap Builder::getSingleDimShiftAffineMap(int64_t shift) {
  // expr = d0 + shift.
  auto expr = getAffineDimExpr(0) + shift;
  return AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, expr);
}

AffineMap Builder::getShiftedAffineMap(AffineMap map, int64_t shift) {
  SmallVector<AffineExpr, 4> shiftedResults;
  shiftedResults.reserve(map.getNumResults());
  for (auto resultExpr : map.getResults())
    shiftedResults.push_back(resultExpr + shift);
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), shiftedResults,
                        context);
}

//===----------------------------------------------------------------------===//
// OpBuilder
//===----------------------------------------------------------------------===//

OpBuilder::Listener::~Listener() {}

/// Insert the given operation at the current insertion point and return it.
Operation *OpBuilder::insert(Operation *op) {
  if (block)
    block->getOperations().insert(insertPoint, op);

  if (listener)
    listener->notifyOperationInserted(op);
  return op;
}

/// Add new block with 'argTypes' arguments and set the insertion point to the
/// end of it. The block is inserted at the provided insertion point of
/// 'parent'.
Block *OpBuilder::createBlock(Region *parent, Region::iterator insertPt,
                              TypeRange argTypes, ArrayRef<Location> locs) {
  assert(parent && "expected valid parent region");
  if (insertPt == Region::iterator())
    insertPt = parent->end();

  Block *b = new Block();
  b->addArguments(argTypes, locs);
  parent->getBlocks().insert(insertPt, b);
  setInsertionPointToEnd(b);

  if (listener)
    listener->notifyBlockCreated(b);
  return b;
}

/// Add new block with 'argTypes' arguments and set the insertion point to the
/// end of it.  The block is placed before 'insertBefore'.
Block *OpBuilder::createBlock(Block *insertBefore, TypeRange argTypes,
                              ArrayRef<Location> locs) {
  assert(insertBefore && "expected valid insertion block");
  return createBlock(insertBefore->getParent(), Region::iterator(insertBefore),
                     argTypes, locs);
}

/// Create an operation given the fields represented as an OperationState.
Operation *OpBuilder::createOperation(const OperationState &state) {
  return insert(Operation::create(state));
}

/// Attempts to fold the given operation and places new results within
/// 'results'. Returns success if the operation was folded, failure otherwise.
/// Note: This function does not erase the operation on a successful fold.
LogicalResult OpBuilder::tryFold(Operation *op,
                                 SmallVectorImpl<Value> &results) {
  results.reserve(op->getNumResults());
  auto cleanupFailure = [&] {
    results.assign(op->result_begin(), op->result_end());
    return failure();
  };

  // If this operation is already a constant, there is nothing to do.
  if (matchPattern(op, m_Constant()))
    return cleanupFailure();

  // Check to see if any operands to the operation is constant and whether
  // the operation knows how to constant fold itself.
  SmallVector<Attribute, 4> constOperands(op->getNumOperands());
  for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
    matchPattern(op->getOperand(i), m_Constant(&constOperands[i]));

  // Try to fold the operation.
  SmallVector<OpFoldResult, 4> foldResults;
  if (failed(op->fold(constOperands, foldResults)) || foldResults.empty())
    return cleanupFailure();

  // A temporary builder used for creating constants during folding.
  OpBuilder cstBuilder(context);
  SmallVector<Operation *, 1> generatedConstants;

  // Populate the results with the folded results.
  Dialect *dialect = op->getDialect();
  for (auto &it : llvm::enumerate(foldResults)) {
    // Normal values get pushed back directly.
    if (auto value = it.value().dyn_cast<Value>()) {
      results.push_back(value);
      continue;
    }

    // Otherwise, try to materialize a constant operation.
    if (!dialect)
      return cleanupFailure();

    // Ask the dialect to materialize a constant operation for this value.
    Attribute attr = it.value().get<Attribute>();
    auto *constOp = dialect->materializeConstant(
        cstBuilder, attr, op->getResult(it.index()).getType(), op->getLoc());
    if (!constOp) {
      // Erase any generated constants.
      for (Operation *cst : generatedConstants)
        cst->erase();
      return cleanupFailure();
    }
    assert(matchPattern(constOp, m_Constant()));

    generatedConstants.push_back(constOp);
    results.push_back(constOp->getResult(0));
  }

  // If we were successful, insert any generated constants.
  for (Operation *cst : generatedConstants)
    insert(cst);

  return success();
}

Operation *OpBuilder::clone(Operation &op, BlockAndValueMapping &mapper) {
  Operation *newOp = op.clone(mapper);
  // The `insert` call below handles the notification for inserting `newOp`
  // itself. But if `newOp` has any regions, we need to notify the listener
  // about any ops that got inserted inside those regions as part of cloning.
  if (listener) {
    auto walkFn = [&](Operation *walkedOp) {
      listener->notifyOperationInserted(walkedOp);
    };
    for (Region &region : newOp->getRegions())
      region.walk(walkFn);
  }
  return insert(newOp);
}

Operation *OpBuilder::clone(Operation &op) {
  BlockAndValueMapping mapper;
  return clone(op, mapper);
}
