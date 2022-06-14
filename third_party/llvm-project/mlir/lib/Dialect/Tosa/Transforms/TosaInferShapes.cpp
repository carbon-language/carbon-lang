//===- TosaInferShapes.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Propogate shapes forward along TOSA operations to resolve dynamic shape
// operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/PassDetail.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::tosa;

namespace {

void propagateShapesInRegion(Region &region);

void propagateShapesToTosaIf(
    Operation &op, DenseMap<Value, ShapedTypeComponents> &shapesStorage) {
  IfOp ifOp = dyn_cast<IfOp>(op);
  if (!ifOp)
    return;

  for (auto &region : op.getRegions()) {
    Block &frontBlock = region.front();
    if (frontBlock.getNumArguments() + 1 != ifOp.getNumOperands())
      return;

    for (unsigned int i = 1, s = op.getNumOperands(); i < s; i++) {
      auto inferredTy = shapesStorage[op.getOperand(i)];
      auto blockArg = frontBlock.getArgument(i - 1);
      auto oldType = blockArg.getType().cast<ShapedType>();

      if (inferredTy.hasRank()) {
        Type newType = oldType.clone(inferredTy.getDims());
        blockArg.setType(newType);
      }
    }

    for (int i = 0, e = frontBlock.getNumArguments(); i < e; i++) {
      ValueKnowledge operandKnowledge = ValueKnowledge::getKnowledgeFromType(
          ifOp.getOperand(i + 1).getType());
      ValueKnowledge blockKnowledge = ValueKnowledge::getKnowledgeFromType(
          frontBlock.getArgument(i).getType());
      ValueKnowledge joinedKnowledge =
          ValueKnowledge::join(operandKnowledge, blockKnowledge);
      if (!joinedKnowledge)
        continue;
      frontBlock.getArgument(i).setType(joinedKnowledge.getType());
    }

    propagateShapesInRegion(region);
  }
}

void propagateShapesToTosaWhile(
    Operation &op, DenseMap<Value, ShapedTypeComponents> &shapesStorage) {
  WhileOp whileOp = dyn_cast<WhileOp>(op);
  if (!whileOp)
    return;

  // Determine what the expected argument types are to the cond/body blocks.
  // The expected arguments should be compatible with ever iteration of the
  // loop body / condition for tosa.while.
  llvm::SmallVector<Type> argTypes;
  for (auto operand : op.getOperands()) {
    auto operandTy = operand.getType().cast<ShapedType>();
    auto shapedTypeComponent = shapesStorage[operand];
    if (shapedTypeComponent.hasRank()) {
      auto newTy = operandTy.clone(shapedTypeComponent.getDims());
      argTypes.push_back(newTy);
    } else {
      argTypes.push_back(operand.getType());
    }
  }

  // Save out the type information so we can restore at the end.
  llvm::DenseMap<Value, Type> originalTypeMap;
  for (auto &block : op.getRegion(1)) {
    for (auto arg : block.getArguments())
      originalTypeMap[arg] = arg.getType();
    for (auto &op : block)
      for (auto result : op.getResults())
        originalTypeMap[result] = result.getType();
  }

  bool hasNewTypes = true;
  while (hasNewTypes) {

    // Set types on the block args.
    Region &bodyRegion = op.getRegion(1);
    Block &block = bodyRegion.front();
    for (int i = 0, s = argTypes.size(); i < s; i++) {
      block.getArgument(i).setType(argTypes[i]);
    }

    // Propagate to the end.
    propagateShapesInRegion(bodyRegion);

    // Find all the tosa yield types and verify there is atleast one.
    llvm::SmallVector<YieldOp> yieldOps;
    for (auto &block : bodyRegion)
      if (auto yieldOp = dyn_cast<YieldOp>(block.getTerminator()))
        yieldOps.push_back(yieldOp);

    if (yieldOps.empty())
      return;

    // Using the new tosa.yield operand types, infer the new subtypes.
    llvm::SmallVector<ValueKnowledge> yieldTypeInfo;
    for (auto ty : argTypes) {
      yieldTypeInfo.push_back(ValueKnowledge::getKnowledgeFromType(ty));
    }

    for (auto yieldOp : yieldOps) {
      for (const auto &it : llvm::enumerate(yieldOp.getOperands())) {
        auto newKnowledge =
            ValueKnowledge::getKnowledgeFromType(it.value().getType());
        yieldTypeInfo[it.index()] =
            ValueKnowledge::meet(yieldTypeInfo[it.index()], newKnowledge);
      }
    }

    // This should never happen.
    if (yieldTypeInfo.size() != argTypes.size()) {
      op.emitWarning("has a tosa.yield with the incorrect number of operands");
      return;
    }

    // Determine the new block args and see if any changed.
    hasNewTypes = false;
    for (int i = 0, s = yieldTypeInfo.size(); i < s; i++) {
      Type newType = yieldTypeInfo[i].getType();
      hasNewTypes |= (newType != argTypes[i]);
      argTypes[i] = newType;
    }

    // The types inferred in the block assume the operand types specified for
    // this iteration. We need to restore the original types to ensure that
    // future iterations only use the already specified types, not possible
    // types from previous iterations.
    for (auto &block : bodyRegion) {
      for (auto arg : block.getArguments())
        arg.setType(originalTypeMap[arg]);
      for (auto &op : block)
        for (auto result : op.getResults())
          result.setType(originalTypeMap[result]);
    }
  }

  // We now set the block arguments according to the most recent shape
  // inference results. This gives us the block arg types for the next
  // iteration.
  for (auto &region : op.getRegions()) {
    for (unsigned int i = 0, s = argTypes.size(); i < s; i++) {
      region.front().getArgument(i).setType(argTypes[i]);
    }

    propagateShapesInRegion(region);
  }
}

void propagateShapesInRegion(Region &region) {
  DenseMap<Value, ShapedTypeComponents> shapesStorage;
  auto setShapes = [&](Value val, Type t) {
    if (auto st = t.dyn_cast<ShapedType>())
      shapesStorage[val] = st;
    else
      shapesStorage[val] = t;
  };
  auto operandShape = [&](Value val) -> ShapeAdaptor {
    // Query the WIP mapping rather than the type if set.
    auto it = shapesStorage.find(val);
    if (it == shapesStorage.end())
      return nullptr;
    return it->second;
  };

  for (auto &block : region) {
    for (Operation &op : block) {
      if (op.getDialect()->getNamespace() != TosaDialect::getDialectNamespace())
        continue;

      propagateShapesToTosaIf(op, shapesStorage);
      propagateShapesToTosaWhile(op, shapesStorage);

      InferShapedTypeOpInterface shapeInterface =
          dyn_cast<InferShapedTypeOpInterface>(op);
      if (!shapeInterface)
        continue;

      SmallVector<ShapedTypeComponents> returnedShapes;

      ValueShapeRange range(op.getOperands(), operandShape);
      if (shapeInterface
              .inferReturnTypeComponents(op.getContext(), op.getLoc(), range,
                                         op.getAttrDictionary(),
                                         op.getRegions(), returnedShapes)
              .succeeded()) {
        for (auto it : llvm::zip(op.getResults(), returnedShapes)) {
          Value result = std::get<0>(it);
          ShapedTypeComponents predictedShape = std::get<1>(it);

          // Check whether this use case is replaceable. We define an op as
          // being replaceable if it is used by a ReturnOp or a TosaOp.
          bool replaceable = true;
          for (auto *user : result.getUsers()) {
            if (isa<func::ReturnOp>(user))
              continue;
            if (user->getDialect()->getNamespace() ==
                TosaDialect::getDialectNamespace())
              continue;

            replaceable = false;
          }

          // Determine the knowledge based on the output type.
          // TODO: should also query WIP type probably
          Type resultTy = result.getType();
          auto currentKnowledge =
              ValueKnowledge::getKnowledgeFromType(resultTy);

          // Compute the knowledge based on the inferred type.
          auto inferredKnowledge = ValueKnowledge::getPessimisticValueState();
          inferredKnowledge.dtype =
              resultTy.cast<ShapedType>().getElementType();
          inferredKnowledge.hasRank = predictedShape.hasRank();
          if (predictedShape.hasRank()) {
            for (auto dim : predictedShape.getDims()) {
              inferredKnowledge.sizes.push_back(dim);
            }
          }

          if (!replaceable)
            continue;

          // Compute the new type based on the joined version.
          auto newKnowledge =
              ValueKnowledge::join(currentKnowledge, inferredKnowledge);
          if (!newKnowledge)
            continue;
          setShapes(result, newKnowledge.getType());
        }
      }
    }
  }

  // Actually update types with updated shape knowledge.
  for (auto it : shapesStorage) {
    auto result = it.second;
    if (result.hasRank()) {
      Type t = it.first.getType().cast<ShapedType>().clone(result.getDims());
      it.first.setType(t);
    }
  }
}

/// Pass that performs shape propagation across TOSA operations. This includes
/// migrating to within the regions of if/while operations.
struct TosaInferShapes : public TosaInferShapesBase<TosaInferShapes> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    IRRewriter rewriter(func.getContext());

    propagateShapesInRegion(func.getBody());

    // Insert UnrealizedConversionCasts to guarantee ReturnOp agress with
    // the FuncOp type.
    func.walk([&](func::ReturnOp op) {
      func::FuncOp parent = dyn_cast<func::FuncOp>(op->getParentOp());
      if (!parent)
        return;

      rewriter.setInsertionPoint(op);
      FunctionType funcTy = func.getFunctionType();
      auto resultTys = funcTy.getResults();

      bool castAdded = false;
      SmallVector<Value> castedValues;
      for (auto it : llvm::zip(op->getOperands(), resultTys)) {
        auto operand = std::get<0>(it);
        auto currentTy = operand.getType();
        auto castTy = std::get<1>(it);
        if (currentTy == castTy) {
          castedValues.push_back(operand);
          continue;
        }

        castedValues.push_back(
            rewriter.create<tensor::CastOp>(op.getLoc(), castTy, operand)
                .getResult());

        castAdded = true;
      }

      if (castAdded) {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, castedValues);
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaInferShapesPass() {
  return std::make_unique<TosaInferShapes>();
}
