//===- DropEquivalentBufferResults.cpp - Calling convention conversion ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass drops return values from functions if they are equivalent to one of
// their arguments. E.g.:
//
// ```
// func.func @foo(%m : memref<?xf32>) -> (memref<?xf32>) {
//   return %m : memref<?xf32>
// }
// ```
//
// This functions is rewritten to:
//
// ```
// func.func @foo(%m : memref<?xf32>) {
//   return
// }
// ```
//
// All call sites are updated accordingly. If a function returns a cast of a
// function argument, it is also considered equivalent. A cast is inserted at
// the call site in that case.

#include "PassDetail.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

/// Return the unique ReturnOp that terminates `funcOp`.
/// Return nullptr if there is no such unique ReturnOp.
static func::ReturnOp getAssumedUniqueReturnOp(func::FuncOp funcOp) {
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

/// Return the func::FuncOp called by `callOp`.
static func::FuncOp getCalledFunction(CallOpInterface callOp) {
  SymbolRefAttr sym = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

LogicalResult
mlir::bufferization::dropEquivalentBufferResults(ModuleOp module) {
  IRRewriter rewriter(module.getContext());

  for (auto funcOp : module.getOps<func::FuncOp>()) {
    if (funcOp.isExternal())
      continue;
    func::ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
    // TODO: Support functions with multiple blocks.
    if (!returnOp)
      continue;

    // Compute erased results.
    SmallVector<Value> newReturnValues;
    BitVector erasedResultIndices(funcOp.getFunctionType().getNumResults());
    DenseMap<int64_t, int64_t> resultToArgs;
    for (const auto &it : llvm::enumerate(returnOp.operands())) {
      bool erased = false;
      for (BlockArgument bbArg : funcOp.getArguments()) {
        Value val = it.value();
        while (auto castOp = val.getDefiningOp<memref::CastOp>())
          val = castOp.source();

        if (val == bbArg) {
          resultToArgs[it.index()] = bbArg.getArgNumber();
          erased = true;
          break;
        }
      }

      if (erased) {
        erasedResultIndices.set(it.index());
      } else {
        newReturnValues.push_back(it.value());
      }
    }

    // Update function.
    funcOp.eraseResults(erasedResultIndices);
    returnOp.operandsMutable().assign(newReturnValues);

    // Update function calls.
    module.walk([&](func::CallOp callOp) {
      if (getCalledFunction(callOp) != funcOp)
        return WalkResult::skip();

      rewriter.setInsertionPoint(callOp);
      auto newCallOp = rewriter.create<func::CallOp>(callOp.getLoc(), funcOp,
                                                     callOp.operands());
      SmallVector<Value> newResults;
      int64_t nextResult = 0;
      for (int64_t i = 0; i < callOp.getNumResults(); ++i) {
        if (!resultToArgs.count(i)) {
          // This result was not erased.
          newResults.push_back(newCallOp.getResult(nextResult++));
          continue;
        }

        // This result was erased.
        Value replacement = callOp.getOperand(resultToArgs[i]);
        Type expectedType = callOp.getResult(i).getType();
        if (replacement.getType() != expectedType) {
          // A cast must be inserted at the call site.
          replacement = rewriter.create<memref::CastOp>(
              callOp.getLoc(), expectedType, replacement);
        }
        newResults.push_back(replacement);
      }
      rewriter.replaceOp(callOp, newResults);
      return WalkResult::advance();
    });
  }

  return success();
}

namespace {
struct DropEquivalentBufferResultsPass
    : DropEquivalentBufferResultsBase<DropEquivalentBufferResultsPass> {
  void runOnOperation() override {
    if (failed(bufferization::dropEquivalentBufferResults(getOperation())))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::bufferization::createDropEquivalentBufferResultsPass() {
  return std::make_unique<DropEquivalentBufferResultsPass>();
}
