//===- Pattern.cpp - Conversion pattern to the LLVM dialect ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ConvertToLLVMPattern
//===----------------------------------------------------------------------===//

ConvertToLLVMPattern::ConvertToLLVMPattern(StringRef rootOpName,
                                           MLIRContext *context,
                                           LLVMTypeConverter &typeConverter,
                                           PatternBenefit benefit)
    : ConversionPattern(typeConverter, rootOpName, benefit, context) {}

LLVMTypeConverter *ConvertToLLVMPattern::getTypeConverter() const {
  return static_cast<LLVMTypeConverter *>(
      ConversionPattern::getTypeConverter());
}

LLVM::LLVMDialect &ConvertToLLVMPattern::getDialect() const {
  return *getTypeConverter()->getDialect();
}

Type ConvertToLLVMPattern::getIndexType() const {
  return getTypeConverter()->getIndexType();
}

Type ConvertToLLVMPattern::getIntPtrType(unsigned addressSpace) const {
  return IntegerType::get(&getTypeConverter()->getContext(),
                          getTypeConverter()->getPointerBitwidth(addressSpace));
}

Type ConvertToLLVMPattern::getVoidType() const {
  return LLVM::LLVMVoidType::get(&getTypeConverter()->getContext());
}

Type ConvertToLLVMPattern::getVoidPtrType() const {
  return LLVM::LLVMPointerType::get(
      IntegerType::get(&getTypeConverter()->getContext(), 8));
}

Value ConvertToLLVMPattern::createIndexAttrConstant(OpBuilder &builder,
                                                    Location loc,
                                                    Type resultType,
                                                    int64_t value) {
  return builder.create<LLVM::ConstantOp>(
      loc, resultType, builder.getIntegerAttr(builder.getIndexType(), value));
}

Value ConvertToLLVMPattern::createIndexConstant(
    ConversionPatternRewriter &builder, Location loc, uint64_t value) const {
  return createIndexAttrConstant(builder, loc, getIndexType(), value);
}

Value ConvertToLLVMPattern::getStridedElementPtr(
    Location loc, MemRefType type, Value memRefDesc, ValueRange indices,
    ConversionPatternRewriter &rewriter) const {

  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto successStrides = getStridesAndOffset(type, strides, offset);
  assert(succeeded(successStrides) && "unexpected non-strided memref");
  (void)successStrides;

  MemRefDescriptor memRefDescriptor(memRefDesc);
  Value base = memRefDescriptor.alignedPtr(rewriter, loc);

  Value index;
  if (offset != 0) // Skip if offset is zero.
    index = MemRefType::isDynamicStrideOrOffset(offset)
                ? memRefDescriptor.offset(rewriter, loc)
                : createIndexConstant(rewriter, loc, offset);

  for (int i = 0, e = indices.size(); i < e; ++i) {
    Value increment = indices[i];
    if (strides[i] != 1) { // Skip if stride is 1.
      Value stride = MemRefType::isDynamicStrideOrOffset(strides[i])
                         ? memRefDescriptor.stride(rewriter, loc, i)
                         : createIndexConstant(rewriter, loc, strides[i]);
      increment = rewriter.create<LLVM::MulOp>(loc, increment, stride);
    }
    index =
        index ? rewriter.create<LLVM::AddOp>(loc, index, increment) : increment;
  }

  Type elementPtrType = memRefDescriptor.getElementPtrType();
  return index ? rewriter.create<LLVM::GEPOp>(loc, elementPtrType, base, index)
               : base;
}

// Check if the MemRefType `type` is supported by the lowering. We currently
// only support memrefs with identity maps.
bool ConvertToLLVMPattern::isConvertibleAndHasIdentityMaps(
    MemRefType type) const {
  if (!typeConverter->convertType(type.getElementType()))
    return false;
  return type.getAffineMaps().empty() ||
         llvm::all_of(type.getAffineMaps(),
                      [](AffineMap map) { return map.isIdentity(); });
}

Type ConvertToLLVMPattern::getElementPtrType(MemRefType type) const {
  auto elementType = type.getElementType();
  auto structElementType = typeConverter->convertType(elementType);
  return LLVM::LLVMPointerType::get(structElementType,
                                    type.getMemorySpaceAsInt());
}

void ConvertToLLVMPattern::getMemRefDescriptorSizes(
    Location loc, MemRefType memRefType, ValueRange dynamicSizes,
    ConversionPatternRewriter &rewriter, SmallVectorImpl<Value> &sizes,
    SmallVectorImpl<Value> &strides, Value &sizeBytes) const {
  assert(isConvertibleAndHasIdentityMaps(memRefType) &&
         "layout maps must have been normalized away");
  assert(count(memRefType.getShape(), ShapedType::kDynamicSize) ==
             static_cast<ssize_t>(dynamicSizes.size()) &&
         "dynamicSizes size doesn't match dynamic sizes count in memref shape");

  sizes.reserve(memRefType.getRank());
  unsigned dynamicIndex = 0;
  for (int64_t size : memRefType.getShape()) {
    sizes.push_back(size == ShapedType::kDynamicSize
                        ? dynamicSizes[dynamicIndex++]
                        : createIndexConstant(rewriter, loc, size));
  }

  // Strides: iterate sizes in reverse order and multiply.
  int64_t stride = 1;
  Value runningStride = createIndexConstant(rewriter, loc, 1);
  strides.resize(memRefType.getRank());
  for (auto i = memRefType.getRank(); i-- > 0;) {
    strides[i] = runningStride;

    int64_t size = memRefType.getShape()[i];
    if (size == 0)
      continue;
    bool useSizeAsStride = stride == 1;
    if (size == ShapedType::kDynamicSize)
      stride = ShapedType::kDynamicSize;
    if (stride != ShapedType::kDynamicSize)
      stride *= size;

    if (useSizeAsStride)
      runningStride = sizes[i];
    else if (stride == ShapedType::kDynamicSize)
      runningStride =
          rewriter.create<LLVM::MulOp>(loc, runningStride, sizes[i]);
    else
      runningStride = createIndexConstant(rewriter, loc, stride);
  }

  // Buffer size in bytes.
  Type elementPtrType = getElementPtrType(memRefType);
  Value nullPtr = rewriter.create<LLVM::NullOp>(loc, elementPtrType);
  Value gepPtr = rewriter.create<LLVM::GEPOp>(
      loc, elementPtrType, ArrayRef<Value>{nullPtr, runningStride});
  sizeBytes = rewriter.create<LLVM::PtrToIntOp>(loc, getIndexType(), gepPtr);
}

Value ConvertToLLVMPattern::getSizeInBytes(
    Location loc, Type type, ConversionPatternRewriter &rewriter) const {
  // Compute the size of an individual element. This emits the MLIR equivalent
  // of the following sizeof(...) implementation in LLVM IR:
  //   %0 = getelementptr %elementType* null, %indexType 1
  //   %1 = ptrtoint %elementType* %0 to %indexType
  // which is a common pattern of getting the size of a type in bytes.
  auto convertedPtrType =
      LLVM::LLVMPointerType::get(typeConverter->convertType(type));
  auto nullPtr = rewriter.create<LLVM::NullOp>(loc, convertedPtrType);
  auto gep = rewriter.create<LLVM::GEPOp>(
      loc, convertedPtrType,
      ArrayRef<Value>{nullPtr, createIndexConstant(rewriter, loc, 1)});
  return rewriter.create<LLVM::PtrToIntOp>(loc, getIndexType(), gep);
}

Value ConvertToLLVMPattern::getNumElements(
    Location loc, ArrayRef<Value> shape,
    ConversionPatternRewriter &rewriter) const {
  // Compute the total number of memref elements.
  Value numElements =
      shape.empty() ? createIndexConstant(rewriter, loc, 1) : shape.front();
  for (unsigned i = 1, e = shape.size(); i < e; ++i)
    numElements = rewriter.create<LLVM::MulOp>(loc, numElements, shape[i]);
  return numElements;
}

/// Creates and populates the memref descriptor struct given all its fields.
MemRefDescriptor ConvertToLLVMPattern::createMemRefDescriptor(
    Location loc, MemRefType memRefType, Value allocatedPtr, Value alignedPtr,
    ArrayRef<Value> sizes, ArrayRef<Value> strides,
    ConversionPatternRewriter &rewriter) const {
  auto structType = typeConverter->convertType(memRefType);
  auto memRefDescriptor = MemRefDescriptor::undef(rewriter, loc, structType);

  // Field 1: Allocated pointer, used for malloc/free.
  memRefDescriptor.setAllocatedPtr(rewriter, loc, allocatedPtr);

  // Field 2: Actual aligned pointer to payload.
  memRefDescriptor.setAlignedPtr(rewriter, loc, alignedPtr);

  // Field 3: Offset in aligned pointer.
  memRefDescriptor.setOffset(rewriter, loc,
                             createIndexConstant(rewriter, loc, 0));

  // Fields 4: Sizes.
  for (auto en : llvm::enumerate(sizes))
    memRefDescriptor.setSize(rewriter, loc, en.index(), en.value());

  // Field 5: Strides.
  for (auto en : llvm::enumerate(strides))
    memRefDescriptor.setStride(rewriter, loc, en.index(), en.value());

  return memRefDescriptor;
}

LogicalResult ConvertToLLVMPattern::copyUnrankedDescriptors(
    OpBuilder &builder, Location loc, TypeRange origTypes,
    SmallVectorImpl<Value> &operands, bool toDynamic) const {
  assert(origTypes.size() == operands.size() &&
         "expected as may original types as operands");

  // Find operands of unranked memref type and store them.
  SmallVector<UnrankedMemRefDescriptor, 4> unrankedMemrefs;
  for (unsigned i = 0, e = operands.size(); i < e; ++i)
    if (origTypes[i].isa<UnrankedMemRefType>())
      unrankedMemrefs.emplace_back(operands[i]);

  if (unrankedMemrefs.empty())
    return success();

  // Compute allocation sizes.
  SmallVector<Value, 4> sizes;
  UnrankedMemRefDescriptor::computeSizes(builder, loc, *getTypeConverter(),
                                         unrankedMemrefs, sizes);

  // Get frequently used types.
  MLIRContext *context = builder.getContext();
  Type voidPtrType = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  auto i1Type = IntegerType::get(context, 1);
  Type indexType = getTypeConverter()->getIndexType();

  // Find the malloc and free, or declare them if necessary.
  auto module = builder.getInsertionPoint()->getParentOfType<ModuleOp>();
  LLVM::LLVMFuncOp freeFunc, mallocFunc;
  if (toDynamic)
    mallocFunc = LLVM::lookupOrCreateMallocFn(module, indexType);
  if (!toDynamic)
    freeFunc = LLVM::lookupOrCreateFreeFn(module);

  // Initialize shared constants.
  Value zero =
      builder.create<LLVM::ConstantOp>(loc, i1Type, builder.getBoolAttr(false));

  unsigned unrankedMemrefPos = 0;
  for (unsigned i = 0, e = operands.size(); i < e; ++i) {
    Type type = origTypes[i];
    if (!type.isa<UnrankedMemRefType>())
      continue;
    Value allocationSize = sizes[unrankedMemrefPos++];
    UnrankedMemRefDescriptor desc(operands[i]);

    // Allocate memory, copy, and free the source if necessary.
    Value memory =
        toDynamic
            ? builder.create<LLVM::CallOp>(loc, mallocFunc, allocationSize)
                  .getResult(0)
            : builder.create<LLVM::AllocaOp>(loc, voidPtrType, allocationSize,
                                             /*alignment=*/0);
    Value source = desc.memRefDescPtr(builder, loc);
    builder.create<LLVM::MemcpyOp>(loc, memory, source, allocationSize, zero);
    if (!toDynamic)
      builder.create<LLVM::CallOp>(loc, freeFunc, source);

    // Create a new descriptor. The same descriptor can be returned multiple
    // times, attempting to modify its pointer can lead to memory leaks
    // (allocated twice and overwritten) or double frees (the caller does not
    // know if the descriptor points to the same memory).
    Type descriptorType = getTypeConverter()->convertType(type);
    if (!descriptorType)
      return failure();
    auto updatedDesc =
        UnrankedMemRefDescriptor::undef(builder, loc, descriptorType);
    Value rank = desc.rank(builder, loc);
    updatedDesc.setRank(builder, loc, rank);
    updatedDesc.setMemRefDescPtr(builder, loc, memory);

    operands[i] = updatedDesc;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Detail methods
//===----------------------------------------------------------------------===//

/// Replaces the given operation "op" with a new operation of type "targetOp"
/// and given operands.
LogicalResult LLVM::detail::oneToOneRewrite(
    Operation *op, StringRef targetOp, ValueRange operands,
    LLVMTypeConverter &typeConverter, ConversionPatternRewriter &rewriter) {
  unsigned numResults = op->getNumResults();

  Type packedType;
  if (numResults != 0) {
    packedType = typeConverter.packFunctionResults(op->getResultTypes());
    if (!packedType)
      return failure();
  }

  // Create the operation through state since we don't know its C++ type.
  OperationState state(op->getLoc(), targetOp);
  state.addTypes(packedType);
  state.addOperands(operands);
  state.addAttributes(op->getAttrs());
  Operation *newOp = rewriter.createOperation(state);

  // If the operation produced 0 or 1 result, return them immediately.
  if (numResults == 0)
    return rewriter.eraseOp(op), success();
  if (numResults == 1)
    return rewriter.replaceOp(op, newOp->getResult(0)), success();

  // Otherwise, it had been converted to an operation producing a structure.
  // Extract individual results from the structure and return them as list.
  SmallVector<Value, 4> results;
  results.reserve(numResults);
  for (unsigned i = 0; i < numResults; ++i) {
    auto type = typeConverter.convertType(op->getResult(i).getType());
    results.push_back(rewriter.create<LLVM::ExtractValueOp>(
        op->getLoc(), type, newOp->getResult(0), rewriter.getI64ArrayAttr(i)));
  }
  rewriter.replaceOp(op, results);
  return success();
}
