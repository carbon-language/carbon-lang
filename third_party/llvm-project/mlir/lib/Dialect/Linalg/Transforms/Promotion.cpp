//===- Promotion.cpp - Implementation of linalg Promotion -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the linalg dialect Promotion pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::scf;

using llvm::MapVector;

#define DEBUG_TYPE "linalg-promotion"

/// Alloc a new buffer of `size` * `width` i8; where `width` is given by the
/// data `layout` for `elementType`.
/// Use AllocOp or AllocaOp depending on `options`.
/// Take an optional alignment.
static Value allocBuffer(ImplicitLocOpBuilder &b,
                         const LinalgPromotionOptions &options,
                         Type elementType, Value allocSize, DataLayout &layout,
                         Optional<unsigned> alignment = None) {
  auto width = layout.getTypeSize(elementType);

  IntegerAttr alignmentAttr;
  if (alignment.hasValue())
    alignmentAttr = b.getI64IntegerAttr(alignment.getValue());

  // Static buffer.
  if (auto cst = allocSize.getDefiningOp<ConstantIndexOp>()) {
    auto staticBufferType =
        MemRefType::get(width * cst.getValue(), b.getIntegerType(8));
    if (options.useAlloca) {
      return b.createOrFold<memref::AllocaOp>(staticBufferType, ValueRange{},
                                              alignmentAttr);
    }
    return b.createOrFold<memref::AllocOp>(staticBufferType, ValueRange{},
                                           alignmentAttr);
  }

  // Fallback dynamic buffer.
  auto dynamicBufferType = MemRefType::get(-1, b.getIntegerType(8));
  Value mul =
      b.createOrFold<MulIOp>(b.create<ConstantIndexOp>(width), allocSize);
  if (options.useAlloca)
    return b.create<memref::AllocaOp>(dynamicBufferType, mul, alignmentAttr);
  return b.create<memref::AllocOp>(dynamicBufferType, mul, alignmentAttr);
}

/// Default allocation callback function. This allocates a promoted buffer when
/// no call back to do so is provided. The default is to allocate a
/// memref<..xi8> and return a view to get a memref type of shape
/// boundingSubViewSize.
static Optional<Value>
defaultAllocBufferCallBack(const LinalgPromotionOptions &options,
                           OpBuilder &builder, memref::SubViewOp subView,
                           ArrayRef<Value> boundingSubViewSize,
                           Optional<unsigned> alignment, DataLayout &layout) {
  ShapedType viewType = subView.getType();
  ImplicitLocOpBuilder b(subView.getLoc(), builder);
  auto zero = b.createOrFold<ConstantIndexOp>(0);
  auto one = b.createOrFold<ConstantIndexOp>(1);

  Value allocSize = one;
  for (auto size : llvm::enumerate(boundingSubViewSize))
    allocSize = b.createOrFold<MulIOp>(allocSize, size.value());
  Value buffer = allocBuffer(b, options, viewType.getElementType(), allocSize,
                             layout, alignment);
  SmallVector<int64_t, 4> dynSizes(boundingSubViewSize.size(),
                                   ShapedType::kDynamicSize);
  Value view = b.createOrFold<memref::ViewOp>(
      MemRefType::get(dynSizes, viewType.getElementType()), buffer, zero,
      boundingSubViewSize);
  return view;
}

/// Default implementation of deallocation of the buffer use for promotion. It
/// expects to get the same value that the default allocation method returned,
/// i.e. result of a ViewOp.
static LogicalResult
defaultDeallocBufferCallBack(const LinalgPromotionOptions &options,
                             OpBuilder &b, Value fullLocalView) {
  if (!options.useAlloca) {
    auto viewOp = cast<memref::ViewOp>(fullLocalView.getDefiningOp());
    b.create<memref::DeallocOp>(viewOp.source().getLoc(), viewOp.source());
  }
  return success();
}

namespace {

/// Helper struct that captures the information required to apply the
/// transformation on each op. This bridges the abstraction gap with the
/// user-facing API which exposes positional arguments to control which operands
/// are promoted.
struct LinalgOpInstancePromotionOptions {
  LinalgOpInstancePromotionOptions(LinalgOp op,
                                   const LinalgPromotionOptions &options);
  /// SubViews to promote.
  MapVector<int64_t, Value> subViews;
  /// True if the full view should be used for the promoted buffer.
  DenseMap<Value, bool> useFullTileBuffers;

  /// Callback functions for allocation and deallocation of promoted buffers, as
  /// well as to copy the data into and out of these buffers.
  AllocBufferCallbackFn allocationFn;
  DeallocBufferCallbackFn deallocationFn;
  CopyCallbackFn copyInFn;
  CopyCallbackFn copyOutFn;

  /// Allow the use of dynamically-sized buffers.
  bool dynamicBuffers;

  /// Alignment of promoted buffer.
  Optional<unsigned> alignment;
};
} // namespace

LinalgOpInstancePromotionOptions::LinalgOpInstancePromotionOptions(
    LinalgOp linalgOp, const LinalgPromotionOptions &options)
    : subViews(), dynamicBuffers(options.dynamicBuffers),
      alignment(options.alignment) {
  assert(linalgOp.hasBufferSemantics() && "revisit usage of shaped operand");
  auto vUseFullTileBuffers =
      options.useFullTileBuffers.getValueOr(llvm::SmallBitVector());
  vUseFullTileBuffers.resize(linalgOp.getNumInputsAndOutputs(),
                             options.useFullTileBuffersDefault);

  for (OpOperand *opOperand : linalgOp.getInputAndOutputOperands()) {
    int64_t operandNumber = opOperand->getOperandNumber();
    if (options.operandsToPromote &&
        !options.operandsToPromote->count(operandNumber))
      continue;
    Operation *op = opOperand->get().getDefiningOp();
    if (auto sv = dyn_cast_or_null<memref::SubViewOp>(op)) {
      subViews[operandNumber] = sv;
      useFullTileBuffers[sv] = vUseFullTileBuffers[operandNumber];
    }
  }

  if (options.allocationFn) {
    allocationFn = *options.allocationFn;
  } else {
    allocationFn = [&](OpBuilder &b, memref::SubViewOp subViewOp,
                       ArrayRef<Value> boundingSubViewSize,
                       DataLayout &layout) -> Optional<Value> {
      return defaultAllocBufferCallBack(options, b, subViewOp,
                                        boundingSubViewSize, alignment, layout);
    };
  }

  if (options.deallocationFn) {
    deallocationFn = *options.deallocationFn;
  } else {
    deallocationFn = [&](OpBuilder &b, Value buffer) {
      return defaultDeallocBufferCallBack(options, b, buffer);
    };
  }

  // Save the loc because `linalgOp` goes out of scope.
  Location loc = linalgOp.getLoc();
  auto defaultCopyCallBack = [loc](OpBuilder &b, Value src,
                                   Value dst) -> LogicalResult {
    b.create<linalg::CopyOp>(loc, src, dst);
    return success();
  };
  copyInFn = (options.copyInFn ? *(options.copyInFn) : defaultCopyCallBack);
  copyOutFn = (options.copyOutFn ? *(options.copyOutFn) : defaultCopyCallBack);
}

// Performs promotion of a `subView` into a local buffer of the size of the
// *ranges* of the `subView`. This produces a buffer whose size may be bigger
// than the actual size of the `subView` at the boundaries.
// This is related to the full/partial tile problem.
// Returns a PromotionInfo containing a `buffer`, `fullLocalView` and
// `partialLocalView` such that:
//   * `buffer` is always the size of the full tile.
//   * `fullLocalView` is a dense contiguous view into that buffer.
//   * `partialLocalView` is a dense non-contiguous slice of `fullLocalView`
//     that corresponds to the size of `subView` and accounting for boundary
//     effects.
// The point of the full tile buffer is that constant static tile sizes are
// folded and result in a buffer type with statically known size and alignment
// properties.
// To account for general boundary effects, padding must be performed on the
// boundary tiles. For now this is done with an unconditional `fill` op followed
// by a partial `copy` op.
Optional<PromotionInfo> mlir::linalg::promoteSubviewAsNewBuffer(
    OpBuilder &b, Location loc, memref::SubViewOp subView,
    AllocBufferCallbackFn allocationFn, DataLayout &layout) {
  auto viewType = subView.getType();
  auto rank = viewType.getRank();
  SmallVector<Value, 4> fullSizes;
  SmallVector<OpFoldResult> partialSizes;
  fullSizes.reserve(rank);
  partialSizes.reserve(rank);
  for (auto en : llvm::enumerate(subView.getOrCreateRanges(b, loc))) {
    auto rangeValue = en.value();
    // Try to extract a tight constant.
    LLVM_DEBUG(llvm::dbgs() << "Extract tightest: " << rangeValue.size << "\n");
    IntegerAttr sizeAttr = getSmallestBoundingIndex(rangeValue.size);
    Value size =
        (!sizeAttr) ? rangeValue.size : b.create<ConstantOp>(loc, sizeAttr);
    LLVM_DEBUG(llvm::dbgs() << "Extracted tightest: " << size << "\n");
    fullSizes.push_back(size);
    partialSizes.push_back(
        b.createOrFold<memref::DimOp>(loc, subView, en.index()));
  }
  SmallVector<int64_t, 4> dynSizes(fullSizes.size(), -1);
  // If a callback is not specified, then use the default implementation for
  // allocating the promoted buffer.
  Optional<Value> fullLocalView = allocationFn(b, subView, fullSizes, layout);
  if (!fullLocalView)
    return {};
  SmallVector<OpFoldResult, 4> zeros(fullSizes.size(), b.getIndexAttr(0));
  SmallVector<OpFoldResult, 4> ones(fullSizes.size(), b.getIndexAttr(1));
  auto partialLocalView = b.createOrFold<memref::SubViewOp>(
      loc, *fullLocalView, zeros, partialSizes, ones);
  return PromotionInfo{*fullLocalView, partialLocalView};
}

static Optional<MapVector<int64_t, PromotionInfo>>
promoteSubViews(ImplicitLocOpBuilder &b,
                LinalgOpInstancePromotionOptions options, DataLayout &layout) {
  if (options.subViews.empty())
    return {};

  MapVector<int64_t, PromotionInfo> promotionInfoMap;

  for (auto v : options.subViews) {
    memref::SubViewOp subView =
        cast<memref::SubViewOp>(v.second.getDefiningOp());
    Optional<PromotionInfo> promotionInfo = promoteSubviewAsNewBuffer(
        b, b.getLoc(), subView, options.allocationFn, layout);
    if (!promotionInfo)
      return {};
    promotionInfoMap[v.first] = *promotionInfo;

    // Only fill the buffer if the full local view is used
    if (!options.useFullTileBuffers[v.second])
      continue;
    Type subviewEltType = subView.getType().getElementType();
    Value fillVal =
        llvm::TypeSwitch<Type, Value>(subviewEltType)
            .Case([&](FloatType t) {
              return b.create<ConstantOp>(FloatAttr::get(t, 0.0));
            })
            .Case([&](IntegerType t) {
              return b.create<ConstantOp>(IntegerAttr::get(t, 0));
            })
            .Case([&](ComplexType t) {
              Value tmp;
              if (auto et = t.getElementType().dyn_cast<FloatType>())
                tmp = b.create<ConstantOp>(FloatAttr::get(et, 0.0));
              else if (auto et = t.getElementType().cast<IntegerType>())
                tmp = b.create<ConstantOp>(IntegerAttr::get(et, 0));
              return b.create<complex::CreateOp>(t, tmp, tmp);
            })
            .Default([](auto) { return Value(); });
    if (!fillVal)
      return {};
    b.create<linalg::FillOp>(fillVal, promotionInfo->fullLocalView);
  }

  // Copy data into the promoted buffers. Use callback if provided.
  for (auto v : options.subViews) {
    auto info = promotionInfoMap.find(v.first);
    if (info == promotionInfoMap.end())
      continue;
    if (failed(options.copyInFn(
            b, cast<memref::SubViewOp>(v.second.getDefiningOp()),
            info->second.partialLocalView)))
      return {};
  }
  return promotionInfoMap;
}

static Optional<LinalgOp>
promoteSubViews(ImplicitLocOpBuilder &b, LinalgOp op,
                LinalgOpInstancePromotionOptions options, DataLayout &layout) {
  assert(op.hasBufferSemantics() && "expected linalg op with buffer semantics");

  if (auto convOp = dyn_cast<linalg::ConvOp>(op.getOperation())) {
    // TODO: add a level of indirection to linalg.generic.
    if (convOp.padding())
      return {};
  }

  // 1. Promote the specified views and use them in the new op.
  auto promotedBuffersAndViews = promoteSubViews(b, options, layout);
  if (!promotedBuffersAndViews ||
      promotedBuffersAndViews->size() != options.subViews.size())
    return {};

  // 2. Append all other operands as they appear, this enforces that such
  // operands are not views. This is to support cases such as FillOp taking
  // extra scalars etc.  Keep a reference to output buffers;
  SmallVector<Value, 8> opViews;
  opViews.reserve(op.getNumInputsAndOutputs());
  SmallVector<std::pair<Value, Value>, 8> writebackViews;
  writebackViews.reserve(promotedBuffersAndViews->size());
  for (OpOperand *opOperand : op.getInputAndOutputOperands()) {
    int64_t operandNumber = opOperand->getOperandNumber();
    if (options.subViews.count(operandNumber) != 0) {
      if (options.useFullTileBuffers[opOperand->get()])
        opViews.push_back(
            (*promotedBuffersAndViews)[operandNumber].fullLocalView);
      else
        opViews.push_back(
            (*promotedBuffersAndViews)[operandNumber].partialLocalView);
      if (operandNumber >= op.getNumInputs())
        writebackViews.emplace_back(std::make_pair(
            opOperand->get(),
            (*promotedBuffersAndViews)[operandNumber].partialLocalView));
    } else {
      opViews.push_back(opOperand->get());
    }
  }
  op->setOperands(0, opViews.size(), opViews);

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(op);
  // 3. Emit write-back for the promoted output views: copy the partial view.
  for (auto viewAndPartialLocalView : writebackViews) {
    if (failed(options.copyOutFn(b, viewAndPartialLocalView.second,
                                 viewAndPartialLocalView.first)))
      return {};
  }

  // 4. Dealloc all local buffers.
  for (const auto &pi : *promotedBuffersAndViews)
    (void)options.deallocationFn(b, pi.second.fullLocalView);
  return op;
}

LogicalResult
mlir::linalg::promoteSubviewsPrecondition(Operation *op,
                                          LinalgPromotionOptions options) {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  // Transformation applies to buffers only.
  if (!linalgOp || !linalgOp.hasBufferSemantics())
    return failure();
  // Check that at least one of the requested operands is indeed a subview.
  for (OpOperand *opOperand : linalgOp.getInputAndOutputOperands()) {
    auto sv =
        isa_and_nonnull<memref::SubViewOp>(opOperand->get().getDefiningOp());
    if (sv) {
      if (!options.operandsToPromote.hasValue() ||
          options.operandsToPromote->count(opOperand->getOperandNumber()))
        return success();
    }
  }
  // TODO: Check all subviews requested are bound by a static constant.
  // TODO: Check that the total footprint fits within a given size.
  return failure();
}

Optional<LinalgOp>
mlir::linalg::promoteSubViews(OpBuilder &builder, LinalgOp linalgOp,
                              LinalgPromotionOptions options) {
  LinalgOpInstancePromotionOptions linalgOptions(linalgOp, options);
  auto layout = DataLayout::closest(linalgOp);
  ImplicitLocOpBuilder b(linalgOp.getLoc(), builder);
  return ::promoteSubViews(b, linalgOp, linalgOptions, layout);
}

namespace {
struct LinalgPromotionPass : public LinalgPromotionBase<LinalgPromotionPass> {
  LinalgPromotionPass() = default;
  LinalgPromotionPass(bool dynamicBuffers, bool useAlloca) {
    this->dynamicBuffers = dynamicBuffers;
    this->useAlloca = useAlloca;
  }

  void runOnFunction() override {
    getFunction().walk([&](LinalgOp op) {
      auto options = LinalgPromotionOptions()
                         .setDynamicBuffers(dynamicBuffers)
                         .setUseAlloca(useAlloca);
      if (failed(promoteSubviewsPrecondition(op, options)))
        return;
      LLVM_DEBUG(llvm::dbgs() << "Promote: " << *(op.getOperation()) << "\n");
      ImplicitLocOpBuilder b(op.getLoc(), op);
      promoteSubViews(b, op, options);
    });
  }
};
} // namespace

// TODO: support more transformation options in the pass.
std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgPromotionPass(bool dynamicBuffers, bool useAlloca) {
  return std::make_unique<LinalgPromotionPass>(dynamicBuffers, useAlloca);
}
std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgPromotionPass() {
  return std::make_unique<LinalgPromotionPass>();
}
