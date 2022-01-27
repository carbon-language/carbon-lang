//===- LoopPipelining.cpp - Code to perform loop software pipelining-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop software pipelining
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SCF/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::scf;

namespace {

/// Helper to keep internal information during pipelining transformation.
struct LoopPipelinerInternal {
  /// Coarse liverange information for ops used across stages.
  struct LiverangeInfo {
    unsigned lastUseStage = 0;
    unsigned defStage = 0;
  };

protected:
  ForOp forOp;
  unsigned maxStage = 0;
  DenseMap<Operation *, unsigned> stages;
  std::vector<Operation *> opOrder;
  int64_t ub;
  int64_t lb;
  int64_t step;

  // When peeling the kernel we generate several version of each value for
  // different stage of the prologue. This map tracks the mapping between
  // original Values in the loop and the different versions
  // peeled from the loop.
  DenseMap<Value, llvm::SmallVector<Value>> valueMapping;

  /// Assign a value to `valueMapping`, this means `val` represents the version
  /// `idx` of `key` in the epilogue.
  void setValueMapping(Value key, Value el, int64_t idx);

public:
  /// Initalize the information for the given `op`, return true if it
  /// satisfies the pre-condition to apply pipelining.
  bool initializeLoopInfo(ForOp op, const PipeliningOption &options);
  /// Emits the prologue, this creates `maxStage - 1` part which will contain
  /// operations from stages [0; i], where i is the part index.
  void emitPrologue(PatternRewriter &rewriter);
  /// Gather liverange information for Values that are used in a different stage
  /// than its definition.
  llvm::MapVector<Value, LiverangeInfo> analyzeCrossStageValues();
  scf::ForOp createKernelLoop(
      const llvm::MapVector<Value, LiverangeInfo> &crossStageValues,
      PatternRewriter &rewriter,
      llvm::DenseMap<std::pair<Value, unsigned>, unsigned> &loopArgMap);
  /// Emits the pipelined kernel. This clones loop operations following user
  /// order and remaps operands defined in a different stage as their use.
  void createKernel(
      scf::ForOp newForOp,
      const llvm::MapVector<Value, LiverangeInfo> &crossStageValues,
      const llvm::DenseMap<std::pair<Value, unsigned>, unsigned> &loopArgMap,
      PatternRewriter &rewriter);
  /// Emits the epilogue, this creates `maxStage - 1` part which will contain
  /// operations from stages [i; maxStage], where i is the part index.
  llvm::SmallVector<Value> emitEpilogue(PatternRewriter &rewriter);
};

bool LoopPipelinerInternal::initializeLoopInfo(
    ForOp op, const PipeliningOption &options) {
  forOp = op;
  auto upperBoundCst =
      forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto lowerBoundCst =
      forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto stepCst = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!upperBoundCst || !lowerBoundCst || !stepCst)
    return false;
  ub = upperBoundCst.value();
  lb = lowerBoundCst.value();
  step = stepCst.value();
  int64_t numIteration = ceilDiv(ub - lb, step);
  std::vector<std::pair<Operation *, unsigned>> schedule;
  options.getScheduleFn(forOp, schedule);
  if (schedule.empty())
    return false;

  opOrder.reserve(schedule.size());
  for (auto &opSchedule : schedule) {
    maxStage = std::max(maxStage, opSchedule.second);
    stages[opSchedule.first] = opSchedule.second;
    opOrder.push_back(opSchedule.first);
  }
  if (numIteration <= maxStage)
    return false;

  // All operations need to have a stage.
  if (forOp
          .walk([this](Operation *op) {
            if (op != forOp.getOperation() && !isa<scf::YieldOp>(op) &&
                stages.find(op) == stages.end())
              return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted())
    return false;

  // Only support loop carried dependency with a distance of 1. This means the
  // source of all the scf.yield operands needs to be defined by operations in
  // the loop.
  if (llvm::any_of(forOp.getBody()->getTerminator()->getOperands(),
                   [this](Value operand) {
                     Operation *def = operand.getDefiningOp();
                     return !def || stages.find(def) == stages.end();
                   }))
    return false;
  return true;
}

void LoopPipelinerInternal::emitPrologue(PatternRewriter &rewriter) {
  // Initialize the iteration argument to the loop initiale values.
  for (BlockArgument &arg : forOp.getRegionIterArgs()) {
    OpOperand &operand = forOp.getOpOperandForRegionIterArg(arg);
    setValueMapping(arg, operand.get(), 0);
  }
  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  for (int64_t i = 0; i < maxStage; i++) {
    // special handling for induction variable as the increment is implicit.
    Value iv = rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(), lb + i);
    setValueMapping(forOp.getInductionVar(), iv, i);
    for (Operation *op : opOrder) {
      if (stages[op] > i)
        continue;
      Operation *newOp = rewriter.clone(*op);
      for (unsigned opIdx = 0; opIdx < op->getNumOperands(); opIdx++) {
        auto it = valueMapping.find(op->getOperand(opIdx));
        if (it != valueMapping.end())
          newOp->setOperand(opIdx, it->second[i - stages[op]]);
      }
      for (unsigned destId : llvm::seq(unsigned(0), op->getNumResults())) {
        setValueMapping(op->getResult(destId), newOp->getResult(destId),
                        i - stages[op]);
        // If the value is a loop carried dependency update the loop argument
        // mapping.
        for (OpOperand &operand : yield->getOpOperands()) {
          if (operand.get() != op->getResult(destId))
            continue;
          setValueMapping(forOp.getRegionIterArgs()[operand.getOperandNumber()],
                          newOp->getResult(destId), i - stages[op] + 1);
        }
      }
    }
  }
}

llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo>
LoopPipelinerInternal::analyzeCrossStageValues() {
  llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo> crossStageValues;
  for (Operation *op : opOrder) {
    unsigned stage = stages[op];
    for (OpOperand &operand : op->getOpOperands()) {
      Operation *def = operand.get().getDefiningOp();
      if (!def)
        continue;
      auto defStage = stages.find(def);
      if (defStage == stages.end() || defStage->second == stage)
        continue;
      assert(stage > defStage->second);
      LiverangeInfo &info = crossStageValues[operand.get()];
      info.defStage = defStage->second;
      info.lastUseStage = std::max(info.lastUseStage, stage);
    }
  }
  return crossStageValues;
}

scf::ForOp LoopPipelinerInternal::createKernelLoop(
    const llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo>
        &crossStageValues,
    PatternRewriter &rewriter,
    llvm::DenseMap<std::pair<Value, unsigned>, unsigned> &loopArgMap) {
  // Creates the list of initial values associated to values used across
  // stages. The initial values come from the prologue created above.
  // Keep track of the kernel argument associated to each version of the
  // values passed to the kernel.
  llvm::SmallVector<Value> newLoopArg;
  // For existing loop argument initialize them with the right version from the
  // prologue.
  for (const auto &retVal :
       llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
    Operation *def = retVal.value().getDefiningOp();
    assert(def && "Only support loop carried dependencies of distance 1");
    unsigned defStage = stages[def];
    Value valueVersion = valueMapping[forOp.getRegionIterArgs()[retVal.index()]]
                                     [maxStage - defStage];
    assert(valueVersion);
    newLoopArg.push_back(valueVersion);
  }
  for (auto escape : crossStageValues) {
    LiverangeInfo &info = escape.second;
    Value value = escape.first;
    for (unsigned stageIdx = 0; stageIdx < info.lastUseStage - info.defStage;
         stageIdx++) {
      Value valueVersion =
          valueMapping[value][maxStage - info.lastUseStage + stageIdx];
      assert(valueVersion);
      newLoopArg.push_back(valueVersion);
      loopArgMap[std::make_pair(value, info.lastUseStage - info.defStage -
                                           stageIdx)] = newLoopArg.size() - 1;
    }
  }

  // Create the new kernel loop. Since we need to peel `numStages - 1`
  // iteration we change the upper bound to remove those iterations.
  Value newUb = rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(),
                                                        ub - maxStage * step);
  auto newForOp =
      rewriter.create<scf::ForOp>(forOp.getLoc(), forOp.getLowerBound(), newUb,
                                  forOp.getStep(), newLoopArg);
  return newForOp;
}

void LoopPipelinerInternal::createKernel(
    scf::ForOp newForOp,
    const llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo>
        &crossStageValues,
    const llvm::DenseMap<std::pair<Value, unsigned>, unsigned> &loopArgMap,
    PatternRewriter &rewriter) {
  valueMapping.clear();

  // Create the kernel, we clone instruction based on the order given by
  // user and remap operands coming from a previous stages.
  rewriter.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->begin());
  BlockAndValueMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs())) {
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  }
  for (Operation *op : opOrder) {
    int64_t useStage = stages[op];
    auto *newOp = rewriter.clone(*op, mapping);
    for (OpOperand &operand : op->getOpOperands()) {
      // Special case for the induction variable uses. We replace it with a
      // version incremented based on the stage where it is used.
      if (operand.get() == forOp.getInductionVar()) {
        rewriter.setInsertionPoint(newOp);
        Value offset = rewriter.create<arith::ConstantIndexOp>(
            forOp.getLoc(), (maxStage - stages[op]) * step);
        Value iv = rewriter.create<arith::AddIOp>(
            forOp.getLoc(), newForOp.getInductionVar(), offset);
        newOp->setOperand(operand.getOperandNumber(), iv);
        rewriter.setInsertionPointAfter(newOp);
        continue;
      }
      auto arg = operand.get().dyn_cast<BlockArgument>();
      if (arg && arg.getOwner() == forOp.getBody()) {
        // If the value is a loop carried value coming from stage N + 1 remap,
        // it will become a direct use.
        Value ret = forOp.getBody()->getTerminator()->getOperand(
            arg.getArgNumber() - 1);
        Operation *dep = ret.getDefiningOp();
        if (!dep)
          continue;
        auto stageDep = stages.find(dep);
        if (stageDep == stages.end() || stageDep->second == useStage)
          continue;
        assert(stageDep->second == useStage + 1);
        newOp->setOperand(operand.getOperandNumber(),
                          mapping.lookupOrDefault(ret));
        continue;
      }
      // For operands defined in a previous stage we need to remap it to use
      // the correct region argument. We look for the right version of the
      // Value based on the stage where it is used.
      Operation *def = operand.get().getDefiningOp();
      if (!def)
        continue;
      auto stageDef = stages.find(def);
      if (stageDef == stages.end() || stageDef->second == useStage)
        continue;
      auto remap = loopArgMap.find(
          std::make_pair(operand.get(), useStage - stageDef->second));
      assert(remap != loopArgMap.end());
      newOp->setOperand(operand.getOperandNumber(),
                        newForOp.getRegionIterArgs()[remap->second]);
    }
  }

  // Collect the Values that need to be returned by the forOp. For each
  // value we need to have `LastUseStage - DefStage` number of versions
  // returned.
  // We create a mapping between original values and the associated loop
  // returned values that will be needed by the epilogue.
  llvm::SmallVector<Value> yieldOperands;
  for (Value retVal : forOp.getBody()->getTerminator()->getOperands()) {
    yieldOperands.push_back(mapping.lookupOrDefault(retVal));
  }
  for (auto &it : crossStageValues) {
    int64_t version = maxStage - it.second.lastUseStage + 1;
    unsigned numVersionReturned = it.second.lastUseStage - it.second.defStage;
    // add the original verstion to yield ops.
    // If there is a liverange spanning across more than 2 stages we need to add
    // extra arg.
    for (unsigned i = 1; i < numVersionReturned; i++) {
      setValueMapping(it.first, newForOp->getResult(yieldOperands.size()),
                      version++);
      yieldOperands.push_back(
          newForOp.getBody()->getArguments()[yieldOperands.size() + 1 +
                                             newForOp.getNumInductionVars()]);
    }
    setValueMapping(it.first, newForOp->getResult(yieldOperands.size()),
                    version++);
    yieldOperands.push_back(mapping.lookupOrDefault(it.first));
  }
  // Map the yield operand to the forOp returned value.
  for (const auto &retVal :
       llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
    Operation *def = retVal.value().getDefiningOp();
    assert(def && "Only support loop carried dependencies of distance 1");
    unsigned defStage = stages[def];
    setValueMapping(forOp.getRegionIterArgs()[retVal.index()],
                    newForOp->getResult(retVal.index()),
                    maxStage - defStage + 1);
  }
  rewriter.create<scf::YieldOp>(forOp.getLoc(), yieldOperands);
}

llvm::SmallVector<Value>
LoopPipelinerInternal::emitEpilogue(PatternRewriter &rewriter) {
  llvm::SmallVector<Value> returnValues(forOp->getNumResults());
  // Emit different versions of the induction variable. They will be
  // removed by dead code if not used.
  for (int64_t i = 0; i < maxStage; i++) {
    Value newlastIter = rewriter.create<arith::ConstantIndexOp>(
        forOp.getLoc(), lb + step * ((((ub - 1) - lb) / step) - i));
    setValueMapping(forOp.getInductionVar(), newlastIter, maxStage - i);
  }
  // Emit `maxStage - 1` epilogue part that includes operations fro stages
  // [i; maxStage].
  for (int64_t i = 1; i <= maxStage; i++) {
    for (Operation *op : opOrder) {
      if (stages[op] < i)
        continue;
      Operation *newOp = rewriter.clone(*op);
      for (unsigned opIdx = 0; opIdx < op->getNumOperands(); opIdx++) {
        auto it = valueMapping.find(op->getOperand(opIdx));
        if (it != valueMapping.end()) {
          Value v = it->second[maxStage - stages[op] + i];
          assert(v);
          newOp->setOperand(opIdx, v);
        }
      }
      for (unsigned destId : llvm::seq(unsigned(0), op->getNumResults())) {
        setValueMapping(op->getResult(destId), newOp->getResult(destId),
                        maxStage - stages[op] + i);
        // If the value is a loop carried dependency update the loop argument
        // mapping and keep track of the last version to replace the original
        // forOp uses.
        for (OpOperand &operand :
             forOp.getBody()->getTerminator()->getOpOperands()) {
          if (operand.get() != op->getResult(destId))
            continue;
          unsigned version = maxStage - stages[op] + i + 1;
          // If the version is greater than maxStage it means it maps to the
          // original forOp returned value.
          if (version > maxStage) {
            returnValues[operand.getOperandNumber()] = newOp->getResult(destId);
            continue;
          }
          setValueMapping(forOp.getRegionIterArgs()[operand.getOperandNumber()],
                          newOp->getResult(destId), version);
        }
      }
    }
  }
  return returnValues;
}

void LoopPipelinerInternal::setValueMapping(Value key, Value el, int64_t idx) {
  auto it = valueMapping.find(key);
  // If the value is not in the map yet add a vector big enough to store all
  // versions.
  if (it == valueMapping.end())
    it =
        valueMapping
            .insert(std::make_pair(key, llvm::SmallVector<Value>(maxStage + 1)))
            .first;
  it->second[idx] = el;
}

/// Generate a pipelined version of the scf.for loop based on the schedule given
/// as option. This applies the mechanical transformation of changing the loop
/// and generating the prologue/epilogue for the pipelining and doesn't make any
/// decision regarding the schedule.
/// Based on the option the loop is split into several stages.
/// The transformation assumes that the scheduling given by user is valid.
/// For example if we break a loop into 3 stages named S0, S1, S2 we would
/// generate the following code with the number in parenthesis the iteration
/// index:
/// S0(0)                        // Prologue
/// S0(1) S1(0)                  // Prologue
/// scf.for %I = %C0 to %N - 2 {
///  S0(I+2) S1(I+1) S2(I)       // Pipelined kernel
/// }
/// S1(N) S2(N-1)                // Epilogue
/// S2(N)                        // Epilogue
struct ForLoopPipelining : public OpRewritePattern<ForOp> {
  ForLoopPipelining(const PipeliningOption &options, MLIRContext *context)
      : OpRewritePattern<ForOp>(context), options(options) {}
  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override {

    LoopPipelinerInternal pipeliner;
    if (!pipeliner.initializeLoopInfo(forOp, options))
      return failure();

    // 1. Emit prologue.
    pipeliner.emitPrologue(rewriter);

    // 2. Track values used across stages. When a value cross stages it will
    // need to be passed as loop iteration arguments.
    // We first collect the values that are used in a different stage than where
    // they are defined.
    llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo>
        crossStageValues = pipeliner.analyzeCrossStageValues();

    // Mapping between original loop values used cross stage and the block
    // arguments associated after pipelining. A Value may map to several
    // arguments if its liverange spans across more than 2 stages.
    llvm::DenseMap<std::pair<Value, unsigned>, unsigned> loopArgMap;
    // 3. Create the new kernel loop and return the block arguments mapping.
    ForOp newForOp =
        pipeliner.createKernelLoop(crossStageValues, rewriter, loopArgMap);
    // Create the kernel block, order ops based on user choice and remap
    // operands.
    pipeliner.createKernel(newForOp, crossStageValues, loopArgMap, rewriter);

    // 4. Emit the epilogue after the new forOp.
    rewriter.setInsertionPointAfter(newForOp);
    llvm::SmallVector<Value> returnValues = pipeliner.emitEpilogue(rewriter);

    // 5. Erase the original loop and replace the uses with the epilogue output.
    if (forOp->getNumResults() > 0)
      rewriter.replaceOp(forOp, returnValues);
    else
      rewriter.eraseOp(forOp);

    return success();
  }

protected:
  PipeliningOption options;
};

} // namespace

void mlir::scf::populateSCFLoopPipeliningPatterns(
    RewritePatternSet &patterns, const PipeliningOption &options) {
  patterns.add<ForLoopPipelining>(options, patterns.getContext());
}
