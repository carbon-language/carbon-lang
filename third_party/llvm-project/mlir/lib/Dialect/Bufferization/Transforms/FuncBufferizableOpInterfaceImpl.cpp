//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace bufferization {
namespace func_ext {

void FuncAnalysisState::startFunctionAnalysis(FuncOp funcOp) {
  analyzedFuncOps[funcOp] = FuncOpAnalysisState::InProgress;
  auto createdEquiv = equivalentFuncArgs.try_emplace(funcOp, IndexMapping());
  auto createdAliasingOperands =
      aliasingFuncArgs.try_emplace(funcOp, IndexToIndexListMapping());
  auto createdAliasingResults =
      aliasingReturnVals.try_emplace(funcOp, IndexToIndexListMapping());
  auto createdRead = readBbArgs.try_emplace(funcOp, BbArgIndexSet());
  auto createdWritten = writtenBbArgs.try_emplace(funcOp, BbArgIndexSet());
  (void)createdEquiv;
  (void)createdAliasingOperands;
  (void)createdAliasingResults;
  (void)createdRead;
  (void)createdWritten;
#ifndef NDEBUG
  assert(createdEquiv.second && "equivalence info exists already");
  assert(createdAliasingOperands.second && "aliasing info exists already");
  assert(createdAliasingResults.second && "aliasing info exists already");
  assert(createdRead.second && "bbarg access info exists already");
  assert(createdWritten.second && "bbarg access info exists already");
#endif // NDEBUG
}

/// Return the unique ReturnOp that terminates `funcOp`.
/// Return nullptr if there is no such unique ReturnOp.
static func::ReturnOp getAssumedUniqueReturnOp(FuncOp funcOp) {
  func::ReturnOp returnOp;
  for (Block &b : funcOp.getBody()) {
    if (auto candidateOp = dyn_cast<func::ReturnOp>(b.getTerminator())) {
      if (returnOp)
        return nullptr;
      returnOp = candidateOp;
    }
  }
  return returnOp;
}

/// Return the index-th bufferized function argument type. This assumes that the
/// specified argument is a tensor. If the tensor is ranked, a layout map may be
/// specified by the user. If no layout map is specified, the default layout map
/// (as per `options.functionBoundaryTypeConversion`) is used.
static BaseMemRefType
getBufferizedFunctionArgType(FuncOp funcOp, int64_t index,
                             const BufferizationOptions &options) {
  auto tensorType =
      funcOp.getFunctionType().getInput(index).dyn_cast<TensorType>();
  assert(tensorType && "expected TensorType");

  BaseMemRefType memrefType;
  if (options.functionBoundaryTypeConversion ==
      BufferizationOptions::LayoutMapOption::IdentityLayoutMap) {
    memrefType = getMemRefTypeWithStaticIdentityLayout(tensorType);
  } else {
    // Note: Layout maps on function parameters cannot be inferred. The best we
    // can do at the moment is "fully dynamic".
    memrefType = getMemRefTypeWithFullyDynamicLayout(tensorType);
  }

  auto layoutAttr = funcOp.getArgAttrOfType<AffineMapAttr>(
      index, BufferizationDialect::kBufferLayoutAttrName);
  if (!layoutAttr)
    return memrefType;

  auto rankedMemrefType = memrefType.dyn_cast<MemRefType>();
  assert(rankedMemrefType && "buffer layout not supported on unranked tensors");
  return MemRefType::get(
      rankedMemrefType.getShape(), rankedMemrefType.getElementType(),
      layoutAttr.getValue(), rankedMemrefType.getMemorySpaceAsInt());
}

/// Return the FuncOp called by `callOp`.
static FuncOp getCalledFunction(CallOpInterface callOp) {
  SymbolRefAttr sym = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

/// Get FuncAnalysisState.
static const FuncAnalysisState &
getFuncAnalysisState(const AnalysisState &state) {
  Optional<const FuncAnalysisState *> maybeState =
      state.getDialectState<FuncAnalysisState>(
          func::FuncDialect::getDialectNamespace());
  assert(maybeState.hasValue() && "FuncAnalysisState does not exist");
  return **maybeState;
}

/// Return the state (phase) of analysis of the FuncOp.
static FuncOpAnalysisState getFuncOpAnalysisState(const AnalysisState &state,
                                                  FuncOp funcOp) {
  const FuncAnalysisState &funcState = getFuncAnalysisState(state);
  auto it = funcState.analyzedFuncOps.find(funcOp);
  if (it == funcState.analyzedFuncOps.end())
    return FuncOpAnalysisState::NotAnalyzed;
  return it->second;
}

/// Return the index of the bbArg in the given FuncOp that is equivalent to the
/// specified return value (if any).
static Optional<int64_t> getEquivalentFuncArgIdx(FuncOp funcOp,
                                                 const FuncAnalysisState &state,
                                                 int64_t returnValIdx) {
  auto funcOpIt = state.equivalentFuncArgs.find(funcOp);
  if (funcOpIt == state.equivalentFuncArgs.end())
    // No equivalence info stores for funcOp.
    return None;

  auto retValIt = funcOpIt->getSecond().find(returnValIdx);
  if (retValIt == funcOpIt->getSecond().end())
    // Return value has no equivalent bbArg.
    return None;

  return retValIt->getSecond();
}

struct CallOpInterface
    : public BufferizableOpInterface::ExternalModel<CallOpInterface,
                                                    func::CallOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    func::CallOp callOp = cast<func::CallOp>(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");

    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    if (getFuncOpAnalysisState(state, funcOp) != FuncOpAnalysisState::Analyzed)
      // FuncOp not analyzed yet. Assume that OpOperand is read.
      return true;

    return funcState.readBbArgs.lookup(funcOp).contains(
        opOperand.getOperandNumber());
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    func::CallOp callOp = cast<func::CallOp>(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");

    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    if (getFuncOpAnalysisState(state, funcOp) != FuncOpAnalysisState::Analyzed)
      // FuncOp not analyzed yet. Assume that OpOperand is written.
      return true;

    return funcState.writtenBbArgs.lookup(funcOp).contains(
        opOperand.getOperandNumber());
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    func::CallOp callOp = cast<func::CallOp>(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");
    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    if (getFuncOpAnalysisState(state, funcOp) !=
        FuncOpAnalysisState::Analyzed) {
      // FuncOp not analyzed yet. Any OpResult may be aliasing.
      SmallVector<OpResult> result;
      for (OpResult opResult : op->getOpResults())
        if (opResult.getType().isa<TensorType>())
          result.push_back(opResult);
      return result;
    }

    // Get aliasing results from state.
    auto aliasingReturnVals =
        funcState.aliasingReturnVals.lookup(funcOp).lookup(
            opOperand.getOperandNumber());
    SmallVector<OpResult> result;
    for (int64_t resultIdx : aliasingReturnVals)
      result.push_back(callOp->getOpResult(resultIdx));
    return result;
  }

  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const AnalysisState &state) const {
    func::CallOp callOp = cast<func::CallOp>(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");
    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    if (getFuncOpAnalysisState(state, funcOp) !=
        FuncOpAnalysisState::Analyzed) {
      // FuncOp not analyzed yet. Any OpOperand may be aliasing.
      SmallVector<OpOperand *> result;
      for (OpOperand &opOperand : op->getOpOperands())
        if (opOperand.get().getType().isa<TensorType>())
          result.push_back(&opOperand);
      return result;
    }

    // Get aliasing bbArgs from state.
    auto aliasingFuncArgs = funcState.aliasingFuncArgs.lookup(funcOp).lookup(
        opResult.getResultNumber());
    SmallVector<OpOperand *> result;
    for (int64_t bbArgIdx : aliasingFuncArgs)
      result.push_back(&callOp->getOpOperand(bbArgIdx));
    return result;
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    func::CallOp callOp = cast<func::CallOp>(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");
    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    if (getFuncOpAnalysisState(state, funcOp) !=
        FuncOpAnalysisState::Analyzed) {
      // Function not analyzed yet. The conservative answer is "None".
      return BufferRelation::None;
    }

    Optional<int64_t> maybeEquiv =
        getEquivalentFuncArgIdx(funcOp, funcState, opResult.getResultNumber());
    if (maybeEquiv.hasValue()) {
#ifndef NDEBUG
      SmallVector<OpOperand *> aliasingOpOperands =
          getAliasingOpOperand(op, opResult, state);
      assert(aliasingOpOperands.size() == 1 &&
             "expected exactly 1 aliasing OpOperand");
      assert(aliasingOpOperands.front()->getOperandNumber() ==
                 maybeEquiv.getValue() &&
             "inconsistent analysis state");
#endif
      return BufferRelation::Equivalent;
    }
    return BufferRelation::None;
  }

  /// All function arguments are writable. It is the responsibility of the
  /// CallOp to insert buffer copies where necessary.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
    func::CallOp callOp = cast<func::CallOp>(op);
    unsigned numResults = callOp.getNumResults();
    unsigned numOperands = callOp->getNumOperands();
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");
    FunctionType funcType = funcOp.getFunctionType();

    // Result types of the bufferized CallOp.
    SmallVector<Type> resultTypes;
    // Replacement values for the existing CallOp. These are usually the results
    // of the bufferized CallOp, unless a tensor result folds onto an operand.
    SmallVector<Value> replacementValues(numResults, Value());
    // For non-tensor results: A mapping from return val indices of the old
    // CallOp to return val indices of the bufferized CallOp.
    SmallVector<Optional<unsigned>> retValMapping(numResults, None);
    // Operands of the bufferized CallOp.
    SmallVector<Value> newOperands(numOperands, Value());

    // 1. Compute the result types of the new CallOp.
    for (const auto &it : llvm::enumerate(callOp.getResultTypes())) {
      unsigned returnValIdx = it.index();
      Type returnType = it.value();
      if (!returnType.isa<TensorType>()) {
        // Non-tensor values are returned.
        retValMapping[returnValIdx] = resultTypes.size();
        resultTypes.push_back(returnType);
        continue;
      }

      // Returning a memref.
      retValMapping[returnValIdx] = resultTypes.size();
      resultTypes.push_back(funcType.getResult(resultTypes.size()));
    }

    // 2. Rewrite tensor operands as memrefs based on `bufferizedFuncType`.
    for (OpOperand &opOperand : callOp->getOpOperands()) {
      unsigned idx = opOperand.getOperandNumber();
      Value tensorOperand = opOperand.get();

      // Non-tensor operands are just copied.
      if (!tensorOperand.getType().isa<TensorType>()) {
        newOperands[idx] = tensorOperand;
        continue;
      }

      // Retrieve buffers for tensor operands.
      Value buffer = newOperands[idx];
      if (!buffer) {
        FailureOr<Value> bufferOrFailure = state.getBuffer(rewriter, opOperand);
        if (failed(bufferOrFailure))
          return failure();
        buffer = *bufferOrFailure;
      }

      // Caller / callee type mismatch is handled with a CastOp.
      auto memRefType = funcType.getInput(idx);
      // Since we don't yet have a clear layout story, to_memref may
      // conservatively turn tensors into more dynamic memref than necessary.
      // If the memref type of the callee fails, introduce an extra memref.cast
      // that will either canonicalize away or fail compilation until we can do
      // something better.
      if (buffer.getType() != memRefType) {
        assert(
            memref::CastOp::areCastCompatible(buffer.getType(), memRefType) &&
            "CallOp::bufferize: cast incompatible");
        Value castBuffer = rewriter.create<memref::CastOp>(callOp.getLoc(),
                                                           memRefType, buffer);
        buffer = castBuffer;
      }
      newOperands[idx] = buffer;
    }

    // 3. Create the new CallOp.
    Operation *newCallOp = rewriter.create<func::CallOp>(
        callOp.getLoc(), funcOp.getSymName(), resultTypes, newOperands);
    newCallOp->setAttrs(callOp->getAttrs());
    // Get replacement values.
    for (unsigned i = 0; i < replacementValues.size(); ++i) {
      if (replacementValues[i])
        continue;
      replacementValues[i] = newCallOp->getResult(*retValMapping[i]);
    }

    // 4. Replace the old op with the new op.
    replaceOpWithBufferizedValues(rewriter, callOp, replacementValues);

    return success();
  }
};

struct ReturnOpInterface
    : public BufferizableOpInterface::ExternalModel<ReturnOpInterface,
                                                    func::ReturnOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
#ifndef NDEBUG
    auto returnOp = cast<func::ReturnOp>(op);
    assert(isa<FuncOp>(returnOp->getParentOp()) &&
           "only support FuncOp parent for ReturnOp");
#endif // NDEBUG

    // ReturnOps are bufferized as part of FuncOps.
    return success();
  }
};

struct FuncOpInterface
    : public BufferizableOpInterface::ExternalModel<FuncOpInterface, FuncOp> {
  /// Rewrite function bbArgs and return values into buffer form. This function
  /// bufferizes the function signature and the ReturnOp. When the entire
  /// function body has been bufferized, function return types can be switched
  /// to more concise memref types as part of `foldMemRefCasts`.
  ///
  /// All function bbArgs are writable unless they are explicitly marked as
  /// read-only. Callers must insert copies when needed.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
    auto funcOp = cast<FuncOp>(op);
    FunctionType funcType = funcOp.getFunctionType();
    const OneShotBufferizationOptions &options =
        static_cast<const OneShotBufferizationOptions &>(state.getOptions());

    // Construct the bufferized function type.
    SmallVector<Type> argTypes;
    for (const auto &it : llvm::enumerate(funcType.getInputs())) {
      Type argType = it.value();
      if (auto tensorType = argType.dyn_cast<TensorType>()) {
        argTypes.push_back(
            getBufferizedFunctionArgType(funcOp, it.index(), options));
        continue;
      }
      argTypes.push_back(argType);
    }

    // Bodiless functions are assumed opaque and we cannot know the
    // bufferization contract they want to enforce. As a consequence, only
    // support functions that don't return any tensors atm.
    if (funcOp.getBody().empty()) {
      SmallVector<Type> retTypes;
      for (Type resultType : funcType.getResults()) {
        if (resultType.isa<TensorType>())
          return funcOp->emitError() << "cannot bufferize bodiless function "
                                     << "that returns a tensor";
        retTypes.push_back(resultType);
      }
      funcOp.setType(FunctionType::get(op->getContext(), argTypes, retTypes));
      return success();
    }

    // TODO: Support functions with multiple returns.
    func::ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
    assert(returnOp && "expected func with single return op");
    Location loc = returnOp.getLoc();

    // 1. Rewrite the bbArgs. Turn every tensor bbArg into a memref bbArg.
    Block &frontBlock = funcOp.getBody().front();
    for (BlockArgument &bbArg : frontBlock.getArguments()) {
      auto tensorType = bbArg.getType().dyn_cast<TensorType>();
      // Non-tensor types stay the same.
      if (!tensorType)
        continue;

      // Collect all uses of the bbArg.
      SmallVector<OpOperand *> bbArgUses;
      for (OpOperand &use : bbArg.getUses())
        bbArgUses.push_back(&use);

      // Change the bbArg type to memref.
      Type memrefType =
          getBufferizedFunctionArgType(funcOp, bbArg.getArgNumber(), options);
      bbArg.setType(memrefType);

      // Replace all uses of the original tensor bbArg.
      rewriter.setInsertionPointToStart(&frontBlock);
      if (!bbArgUses.empty()) {
        // Insert to_tensor because the remaining function body has not been
        // bufferized yet.
        Value toTensorOp =
            rewriter.create<bufferization::ToTensorOp>(funcOp.getLoc(), bbArg);
        for (OpOperand *use : bbArgUses)
          use->set(toTensorOp);
      }
    }

    // 2. For each result, keep track of which inplace argument it reuses.
    SmallVector<Value> returnValues;
    for (OpOperand &returnOperand : returnOp->getOpOperands()) {
      Value returnVal = returnOperand.get();
      auto tensorType = returnVal.getType().dyn_cast<TensorType>();
      rewriter.setInsertionPoint(returnOp);

      // If not a tensor type just forward it.
      if (!tensorType) {
        returnValues.push_back(returnVal);
        continue;
      }

      BaseMemRefType resultType;
      if (options.functionBoundaryTypeConversion ==
          BufferizationOptions::LayoutMapOption::IdentityLayoutMap) {
        resultType = getMemRefTypeWithStaticIdentityLayout(tensorType);
      } else {
        // Note: If `InferLayoutMap`, cast are later folded away.
        resultType = getMemRefTypeWithFullyDynamicLayout(tensorType);
      }
      Value toMemrefOp = rewriter.create<bufferization::ToMemrefOp>(
          loc, resultType, returnVal);
      returnValues.push_back(toMemrefOp);
    }

    // 3. Rewrite the terminator without the in-place bufferizable values.
    returnOp.operandsMutable().assign(returnValues);

    // 4. Rewrite the FuncOp type to buffer form.
    funcOp.setType(FunctionType::get(op->getContext(), argTypes,
                                     ValueRange(returnValues).getTypes()));

    return success();
  }

  /// Return `true` if the given function argument is writable.
  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    auto funcOp = cast<FuncOp>(op);
    BlockArgument bbArg = value.dyn_cast<BlockArgument>();
    assert(bbArg && "expected BlockArgument");

    // "bufferization.writable" overrides other writability decisions. This is
    // currently used for testing only.
    if (BoolAttr writable = funcOp.getArgAttrOfType<BoolAttr>(
            bbArg.getArgNumber(), BufferizationDialect::kWritableAttrName))
      return writable.getValue();

    // All function arguments are writable by default.
    return true;
  }
};

} // namespace func_ext
} // namespace bufferization
} // namespace mlir

void mlir::bufferization::func_ext::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, func::FuncDialect *dialect) {
    func::CallOp::attachInterface<func_ext::CallOpInterface>(*ctx);
    func::FuncOp::attachInterface<func_ext::FuncOpInterface>(*ctx);
    func::ReturnOp::attachInterface<func_ext::ReturnOpInterface>(*ctx);
  });
}
