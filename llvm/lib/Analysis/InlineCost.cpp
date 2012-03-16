//===- InlineCost.cpp - Cost analysis for inliner -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements inline cost analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/InlineCost.h"
#include "llvm/Support/CallSite.h"
#include "llvm/CallingConv.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace llvm;

unsigned InlineCostAnalyzer::FunctionInfo::countCodeReductionForConstant(
    const CodeMetrics &Metrics, Value *V) {
  unsigned Reduction = 0;
  SmallVector<Value *, 4> Worklist;
  Worklist.push_back(V);
  do {
    Value *V = Worklist.pop_back_val();
    for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E;++UI){
      User *U = *UI;
      if (isa<BranchInst>(U) || isa<SwitchInst>(U)) {
        // We will be able to eliminate all but one of the successors.
        const TerminatorInst &TI = cast<TerminatorInst>(*U);
        const unsigned NumSucc = TI.getNumSuccessors();
        unsigned Instrs = 0;
        for (unsigned I = 0; I != NumSucc; ++I)
          Instrs += Metrics.NumBBInsts.lookup(TI.getSuccessor(I));
        // We don't know which blocks will be eliminated, so use the average size.
        Reduction += InlineConstants::InstrCost*Instrs*(NumSucc-1)/NumSucc;
        continue;
      }

      // Figure out if this instruction will be removed due to simple constant
      // propagation.
      Instruction &Inst = cast<Instruction>(*U);

      // We can't constant propagate instructions which have effects or
      // read memory.
      //
      // FIXME: It would be nice to capture the fact that a load from a
      // pointer-to-constant-global is actually a *really* good thing to zap.
      // Unfortunately, we don't know the pointer that may get propagated here,
      // so we can't make this decision.
      if (Inst.mayReadFromMemory() || Inst.mayHaveSideEffects() ||
          isa<AllocaInst>(Inst))
        continue;

      bool AllOperandsConstant = true;
      for (unsigned i = 0, e = Inst.getNumOperands(); i != e; ++i)
        if (!isa<Constant>(Inst.getOperand(i)) && Inst.getOperand(i) != V) {
          AllOperandsConstant = false;
          break;
        }
      if (!AllOperandsConstant)
        continue;

      // We will get to remove this instruction...
      Reduction += InlineConstants::InstrCost;

      // And any other instructions that use it which become constants
      // themselves.
      Worklist.push_back(&Inst);
    }
  } while (!Worklist.empty());
  return Reduction;
}

static unsigned countCodeReductionForAllocaICmp(const CodeMetrics &Metrics,
                                                ICmpInst *ICI) {
  unsigned Reduction = 0;

  // Bail if this is comparing against a non-constant; there is nothing we can
  // do there.
  if (!isa<Constant>(ICI->getOperand(1)))
    return Reduction;

  // An icmp pred (alloca, C) becomes true if the predicate is true when
  // equal and false otherwise.
  bool Result = ICI->isTrueWhenEqual();

  SmallVector<Instruction *, 4> Worklist;
  Worklist.push_back(ICI);
  do {
    Instruction *U = Worklist.pop_back_val();
    Reduction += InlineConstants::InstrCost;
    for (Value::use_iterator UI = U->use_begin(), UE = U->use_end();
         UI != UE; ++UI) {
      Instruction *I = dyn_cast<Instruction>(*UI);
      if (!I || I->mayHaveSideEffects()) continue;
      if (I->getNumOperands() == 1)
        Worklist.push_back(I);
      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
        // If BO produces the same value as U, then the other operand is
        // irrelevant and we can put it into the Worklist to continue
        // deleting dead instructions. If BO produces the same value as the
        // other operand, we can delete BO but that's it.
        if (Result == true) {
          if (BO->getOpcode() == Instruction::Or)
            Worklist.push_back(I);
          if (BO->getOpcode() == Instruction::And)
            Reduction += InlineConstants::InstrCost;
        } else {
          if (BO->getOpcode() == Instruction::Or ||
              BO->getOpcode() == Instruction::Xor)
            Reduction += InlineConstants::InstrCost;
          if (BO->getOpcode() == Instruction::And)
            Worklist.push_back(I);
        }
      }
      if (BranchInst *BI = dyn_cast<BranchInst>(I)) {
        BasicBlock *BB = BI->getSuccessor(Result ? 0 : 1);
        if (BB->getSinglePredecessor())
          Reduction
            += InlineConstants::InstrCost * Metrics.NumBBInsts.lookup(BB);
      }
    }
  } while (!Worklist.empty());

  return Reduction;
}

/// \brief Compute the reduction possible for a given instruction if we are able
/// to SROA an alloca.
///
/// The reduction for this instruction is added to the SROAReduction output
/// parameter. Returns false if this instruction is expected to defeat SROA in
/// general.
static bool countCodeReductionForSROAInst(Instruction *I,
                                          SmallVectorImpl<Value *> &Worklist,
                                          unsigned &SROAReduction) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    if (!LI->isSimple())
      return false;
    SROAReduction += InlineConstants::InstrCost;
    return true;
  }

  if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    if (!SI->isSimple())
      return false;
    SROAReduction += InlineConstants::InstrCost;
    return true;
  }

  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
    // If the GEP has variable indices, we won't be able to do much with it.
    if (!GEP->hasAllConstantIndices())
      return false;
    // A non-zero GEP will likely become a mask operation after SROA.
    if (GEP->hasAllZeroIndices())
      SROAReduction += InlineConstants::InstrCost;
    Worklist.push_back(GEP);
    return true;
  }

  if (BitCastInst *BCI = dyn_cast<BitCastInst>(I)) {
    // Track pointer through bitcasts.
    Worklist.push_back(BCI);
    SROAReduction += InlineConstants::InstrCost;
    return true;
  }

  // We just look for non-constant operands to ICmp instructions as those will
  // defeat SROA. The actual reduction for these happens even without SROA.
  if (ICmpInst *ICI = dyn_cast<ICmpInst>(I))
    return isa<Constant>(ICI->getOperand(1));

  if (SelectInst *SI = dyn_cast<SelectInst>(I)) {
    // SROA can handle a select of alloca iff all uses of the alloca are
    // loads, and dereferenceable. We assume it's dereferenceable since
    // we're told the input is an alloca.
    for (Value::use_iterator UI = SI->use_begin(), UE = SI->use_end();
         UI != UE; ++UI) {
      LoadInst *LI = dyn_cast<LoadInst>(*UI);
      if (LI == 0 || !LI->isSimple())
        return false;
    }
    // We don't know whether we'll be deleting the rest of the chain of
    // instructions from the SelectInst on, because we don't know whether
    // the other side of the select is also an alloca or not.
    return true;
  }

  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    default:
      return false;
    case Intrinsic::memset:
    case Intrinsic::memcpy:
    case Intrinsic::memmove:
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
      // SROA can usually chew through these intrinsics.
      SROAReduction += InlineConstants::InstrCost;
      return true;
    }
  }

  // If there is some other strange instruction, we're not going to be
  // able to do much if we inline this.
  return false;
}

unsigned InlineCostAnalyzer::FunctionInfo::countCodeReductionForAlloca(
    const CodeMetrics &Metrics, Value *V) {
  if (!V->getType()->isPointerTy()) return 0;  // Not a pointer
  unsigned Reduction = 0;
  unsigned SROAReduction = 0;
  bool CanSROAAlloca = true;

  SmallVector<Value *, 4> Worklist;
  Worklist.push_back(V);
  do {
    Value *V = Worklist.pop_back_val();
    for (Value::use_iterator UI = V->use_begin(), E = V->use_end();
         UI != E; ++UI){
      Instruction *I = cast<Instruction>(*UI);

      if (ICmpInst *ICI = dyn_cast<ICmpInst>(I))
        Reduction += countCodeReductionForAllocaICmp(Metrics, ICI);

      if (CanSROAAlloca)
        CanSROAAlloca = countCodeReductionForSROAInst(I, Worklist,
                                                      SROAReduction);
    }
  } while (!Worklist.empty());

  return Reduction + (CanSROAAlloca ? SROAReduction : 0);
}

void InlineCostAnalyzer::FunctionInfo::countCodeReductionForPointerPair(
    const CodeMetrics &Metrics, DenseMap<Value *, unsigned> &PointerArgs,
    Value *V, unsigned ArgIdx) {
  SmallVector<Value *, 4> Worklist;
  Worklist.push_back(V);
  do {
    Value *V = Worklist.pop_back_val();
    for (Value::use_iterator UI = V->use_begin(), E = V->use_end();
         UI != E; ++UI){
      Instruction *I = cast<Instruction>(*UI);

      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
        // If the GEP has variable indices, we won't be able to do much with it.
        if (!GEP->hasAllConstantIndices())
          continue;
        // Unless the GEP is in-bounds, some comparisons will be non-constant.
        // Fortunately, the real-world cases where this occurs uses in-bounds
        // GEPs, and so we restrict the optimization to them here.
        if (!GEP->isInBounds())
          continue;

        // Constant indices just change the constant offset. Add the resulting
        // value both to our worklist for this argument, and to the set of
        // viable paired values with future arguments.
        PointerArgs[GEP] = ArgIdx;
        Worklist.push_back(GEP);
        continue;
      }

      // Track pointer through casts. Even when the result is not a pointer, it
      // remains a constant relative to constants derived from other constant
      // pointers.
      if (CastInst *CI = dyn_cast<CastInst>(I)) {
        PointerArgs[CI] = ArgIdx;
        Worklist.push_back(CI);
        continue;
      }

      // There are two instructions which produce a strict constant value when
      // applied to two related pointer values. Ignore everything else.
      if (!isa<ICmpInst>(I) && I->getOpcode() != Instruction::Sub)
        continue;
      assert(I->getNumOperands() == 2);

      // Ensure that the two operands are in our set of potentially paired
      // pointers (or are derived from them).
      Value *OtherArg = I->getOperand(0);
      if (OtherArg == V)
        OtherArg = I->getOperand(1);
      DenseMap<Value *, unsigned>::const_iterator ArgIt
        = PointerArgs.find(OtherArg);
      if (ArgIt == PointerArgs.end())
        continue;
      std::pair<unsigned, unsigned> ArgPair(ArgIt->second, ArgIdx);
      if (ArgPair.first > ArgPair.second)
        std::swap(ArgPair.first, ArgPair.second);

      PointerArgPairWeights[ArgPair]
        += countCodeReductionForConstant(Metrics, I);
    }
  } while (!Worklist.empty());
}

/// analyzeFunction - Fill in the current structure with information gleaned
/// from the specified function.
void InlineCostAnalyzer::FunctionInfo::analyzeFunction(Function *F,
                                                       const TargetData *TD) {
  Metrics.analyzeFunction(F, TD);

  // A function with exactly one return has it removed during the inlining
  // process (see InlineFunction), so don't count it.
  // FIXME: This knowledge should really be encoded outside of FunctionInfo.
  if (Metrics.NumRets==1)
    --Metrics.NumInsts;

  ArgumentWeights.reserve(F->arg_size());
  DenseMap<Value *, unsigned> PointerArgs;
  unsigned ArgIdx = 0;
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
       ++I, ++ArgIdx) {
    // Count how much code can be eliminated if one of the arguments is
    // a constant or an alloca.
    ArgumentWeights.push_back(ArgInfo(countCodeReductionForConstant(Metrics, I),
                                      countCodeReductionForAlloca(Metrics, I)));

    // If the argument is a pointer, also check for pairs of pointers where
    // knowing a fixed offset between them allows simplification. This pattern
    // arises mostly due to STL algorithm patterns where pointers are used as
    // random access iterators.
    if (!I->getType()->isPointerTy())
      continue;
    PointerArgs[I] = ArgIdx;
    countCodeReductionForPointerPair(Metrics, PointerArgs, I, ArgIdx);
  }
}

/// NeverInline - returns true if the function should never be inlined into
/// any caller
bool InlineCostAnalyzer::FunctionInfo::NeverInline() {
  return (Metrics.exposesReturnsTwice || Metrics.isRecursive ||
          Metrics.containsIndirectBr);
}

// ConstantFunctionBonus - Figure out how much of a bonus we can get for
// possibly devirtualizing a function. We'll subtract the size of the function
// we may wish to inline from the indirect call bonus providing a limit on
// growth. Leave an upper limit of 0 for the bonus - we don't want to penalize
// inlining because we decide we don't want to give a bonus for
// devirtualizing.
int InlineCostAnalyzer::ConstantFunctionBonus(CallSite CS, Constant *C) {

  // This could just be NULL.
  if (!C) return 0;

  Function *F = dyn_cast<Function>(C);
  if (!F) return 0;

  int Bonus = InlineConstants::IndirectCallBonus + getInlineSize(CS, F);
  return (Bonus > 0) ? 0 : Bonus;
}

// CountBonusForConstant - Figure out an approximation for how much per-call
// performance boost we can expect if the specified value is constant.
int InlineCostAnalyzer::CountBonusForConstant(Value *V, Constant *C) {
  unsigned Bonus = 0;
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E;++UI){
    User *U = *UI;
    if (CallInst *CI = dyn_cast<CallInst>(U)) {
      // Turning an indirect call into a direct call is a BIG win
      if (CI->getCalledValue() == V)
        Bonus += ConstantFunctionBonus(CallSite(CI), C);
    } else if (InvokeInst *II = dyn_cast<InvokeInst>(U)) {
      // Turning an indirect call into a direct call is a BIG win
      if (II->getCalledValue() == V)
        Bonus += ConstantFunctionBonus(CallSite(II), C);
    }
    // FIXME: Eliminating conditional branches and switches should
    // also yield a per-call performance boost.
    else {
      // Figure out the bonuses that wll accrue due to simple constant
      // propagation.
      Instruction &Inst = cast<Instruction>(*U);

      // We can't constant propagate instructions which have effects or
      // read memory.
      //
      // FIXME: It would be nice to capture the fact that a load from a
      // pointer-to-constant-global is actually a *really* good thing to zap.
      // Unfortunately, we don't know the pointer that may get propagated here,
      // so we can't make this decision.
      if (Inst.mayReadFromMemory() || Inst.mayHaveSideEffects() ||
          isa<AllocaInst>(Inst))
        continue;

      bool AllOperandsConstant = true;
      for (unsigned i = 0, e = Inst.getNumOperands(); i != e; ++i)
        if (!isa<Constant>(Inst.getOperand(i)) && Inst.getOperand(i) != V) {
          AllOperandsConstant = false;
          break;
        }

      if (AllOperandsConstant)
        Bonus += CountBonusForConstant(&Inst);
    }
  }

  return Bonus;
}

int InlineCostAnalyzer::getInlineSize(CallSite CS, Function *Callee) {
  // Get information about the callee.
  FunctionInfo *CalleeFI = &CachedFunctionInfo[Callee];

  // If we haven't calculated this information yet, do so now.
  if (CalleeFI->Metrics.NumBlocks == 0)
    CalleeFI->analyzeFunction(Callee, TD);

  // InlineCost - This value measures how good of an inline candidate this call
  // site is to inline.  A lower inline cost make is more likely for the call to
  // be inlined.  This value may go negative.
  //
  int InlineCost = 0;

  // Compute any size reductions we can expect due to arguments being passed into
  // the function.
  //
  unsigned ArgNo = 0;
  CallSite::arg_iterator I = CS.arg_begin();
  for (Function::arg_iterator FI = Callee->arg_begin(), FE = Callee->arg_end();
       FI != FE; ++I, ++FI, ++ArgNo) {

    // If an alloca is passed in, inlining this function is likely to allow
    // significant future optimization possibilities (like scalar promotion, and
    // scalarization), so encourage the inlining of the function.
    //
    if (isa<AllocaInst>(I))
      InlineCost -= CalleeFI->ArgumentWeights[ArgNo].AllocaWeight;

    // If this is a constant being passed into the function, use the argument
    // weights calculated for the callee to determine how much will be folded
    // away with this information.
    else if (isa<Constant>(I))
      InlineCost -= CalleeFI->ArgumentWeights[ArgNo].ConstantWeight;
  }

  const DenseMap<std::pair<unsigned, unsigned>, unsigned> &ArgPairWeights
    = CalleeFI->PointerArgPairWeights;
  for (DenseMap<std::pair<unsigned, unsigned>, unsigned>::const_iterator I
         = ArgPairWeights.begin(), E = ArgPairWeights.end();
       I != E; ++I)
    if (CS.getArgument(I->first.first)->stripInBoundsConstantOffsets() ==
        CS.getArgument(I->first.second)->stripInBoundsConstantOffsets())
      InlineCost -= I->second;

  // Each argument passed in has a cost at both the caller and the callee
  // sides.  Measurements show that each argument costs about the same as an
  // instruction.
  InlineCost -= (CS.arg_size() * InlineConstants::InstrCost);

  // Now that we have considered all of the factors that make the call site more
  // likely to be inlined, look at factors that make us not want to inline it.

  // Calls usually take a long time, so they make the inlining gain smaller.
  InlineCost += CalleeFI->Metrics.NumCalls * InlineConstants::CallPenalty;

  // Look at the size of the callee. Each instruction counts as 5.
  InlineCost += CalleeFI->Metrics.NumInsts * InlineConstants::InstrCost;

  return InlineCost;
}

int InlineCostAnalyzer::getInlineBonuses(CallSite CS, Function *Callee) {
  // Get information about the callee.
  FunctionInfo *CalleeFI = &CachedFunctionInfo[Callee];

  // If we haven't calculated this information yet, do so now.
  if (CalleeFI->Metrics.NumBlocks == 0)
    CalleeFI->analyzeFunction(Callee, TD);

  bool isDirectCall = CS.getCalledFunction() == Callee;
  Instruction *TheCall = CS.getInstruction();
  int Bonus = 0;

  // If there is only one call of the function, and it has internal linkage,
  // make it almost guaranteed to be inlined.
  //
  if (Callee->hasLocalLinkage() && Callee->hasOneUse() && isDirectCall)
    Bonus += InlineConstants::LastCallToStaticBonus;

  // If the instruction after the call, or if the normal destination of the
  // invoke is an unreachable instruction, the function is noreturn.  As such,
  // there is little point in inlining this.
  if (InvokeInst *II = dyn_cast<InvokeInst>(TheCall)) {
    if (isa<UnreachableInst>(II->getNormalDest()->begin()))
      Bonus += InlineConstants::NoreturnPenalty;
  } else if (isa<UnreachableInst>(++BasicBlock::iterator(TheCall)))
    Bonus += InlineConstants::NoreturnPenalty;

  // If this function uses the coldcc calling convention, prefer not to inline
  // it.
  if (Callee->getCallingConv() == CallingConv::Cold)
    Bonus += InlineConstants::ColdccPenalty;

  // Add to the inline quality for properties that make the call valuable to
  // inline.  This includes factors that indicate that the result of inlining
  // the function will be optimizable.  Currently this just looks at arguments
  // passed into the function.
  //
  CallSite::arg_iterator I = CS.arg_begin();
  for (Function::arg_iterator FI = Callee->arg_begin(), FE = Callee->arg_end();
       FI != FE; ++I, ++FI)
    // Compute any constant bonus due to inlining we want to give here.
    if (isa<Constant>(I))
      Bonus += CountBonusForConstant(FI, cast<Constant>(I));

  return Bonus;
}

// getInlineCost - The heuristic used to determine if we should inline the
// function call or not.
//
InlineCost InlineCostAnalyzer::getInlineCost(CallSite CS) {
  return getInlineCost(CS, CS.getCalledFunction());
}

InlineCost InlineCostAnalyzer::getInlineCost(CallSite CS, Function *Callee) {
  Instruction *TheCall = CS.getInstruction();
  Function *Caller = TheCall->getParent()->getParent();

  // Don't inline functions which can be redefined at link-time to mean
  // something else.  Don't inline functions marked noinline or call sites
  // marked noinline.
  if (Callee->mayBeOverridden() || Callee->hasFnAttr(Attribute::NoInline) ||
      CS.isNoInline())
    return llvm::InlineCost::getNever();

  // Get information about the callee.
  FunctionInfo *CalleeFI = &CachedFunctionInfo[Callee];

  // If we haven't calculated this information yet, do so now.
  if (CalleeFI->Metrics.NumBlocks == 0)
    CalleeFI->analyzeFunction(Callee, TD);

  // If we should never inline this, return a huge cost.
  if (CalleeFI->NeverInline())
    return InlineCost::getNever();

  // FIXME: It would be nice to kill off CalleeFI->NeverInline. Then we
  // could move this up and avoid computing the FunctionInfo for
  // things we are going to just return always inline for. This
  // requires handling setjmp somewhere else, however.
  if (!Callee->isDeclaration() && Callee->hasFnAttr(Attribute::AlwaysInline))
    return InlineCost::getAlways();

  if (CalleeFI->Metrics.usesDynamicAlloca) {
    // Get information about the caller.
    FunctionInfo &CallerFI = CachedFunctionInfo[Caller];

    // If we haven't calculated this information yet, do so now.
    if (CallerFI.Metrics.NumBlocks == 0) {
      CallerFI.analyzeFunction(Caller, TD);

      // Recompute the CalleeFI pointer, getting Caller could have invalidated
      // it.
      CalleeFI = &CachedFunctionInfo[Callee];
    }

    // Don't inline a callee with dynamic alloca into a caller without them.
    // Functions containing dynamic alloca's are inefficient in various ways;
    // don't create more inefficiency.
    if (!CallerFI.Metrics.usesDynamicAlloca)
      return InlineCost::getNever();
  }

  // InlineCost - This value measures how good of an inline candidate this call
  // site is to inline.  A lower inline cost make is more likely for the call to
  // be inlined.  This value may go negative due to the fact that bonuses
  // are negative numbers.
  //
  int InlineCost = getInlineSize(CS, Callee) + getInlineBonuses(CS, Callee);
  return llvm::InlineCost::get(InlineCost);
}

// getInlineFudgeFactor - Return a > 1.0 factor if the inliner should use a
// higher threshold to determine if the function call should be inlined.
float InlineCostAnalyzer::getInlineFudgeFactor(CallSite CS) {
  Function *Callee = CS.getCalledFunction();

  // Get information about the callee.
  FunctionInfo &CalleeFI = CachedFunctionInfo[Callee];

  // If we haven't calculated this information yet, do so now.
  if (CalleeFI.Metrics.NumBlocks == 0)
    CalleeFI.analyzeFunction(Callee, TD);

  float Factor = 1.0f;
  // Single BB functions are often written to be inlined.
  if (CalleeFI.Metrics.NumBlocks == 1)
    Factor += 0.5f;

  // Be more aggressive if the function contains a good chunk (if it mades up
  // at least 10% of the instructions) of vector instructions.
  if (CalleeFI.Metrics.NumVectorInsts > CalleeFI.Metrics.NumInsts/2)
    Factor += 2.0f;
  else if (CalleeFI.Metrics.NumVectorInsts > CalleeFI.Metrics.NumInsts/10)
    Factor += 1.5f;
  return Factor;
}

/// growCachedCostInfo - update the cached cost info for Caller after Callee has
/// been inlined.
void
InlineCostAnalyzer::growCachedCostInfo(Function *Caller, Function *Callee) {
  CodeMetrics &CallerMetrics = CachedFunctionInfo[Caller].Metrics;

  // For small functions we prefer to recalculate the cost for better accuracy.
  if (CallerMetrics.NumBlocks < 10 && CallerMetrics.NumInsts < 1000) {
    resetCachedCostInfo(Caller);
    return;
  }

  // For large functions, we can save a lot of computation time by skipping
  // recalculations.
  if (CallerMetrics.NumCalls > 0)
    --CallerMetrics.NumCalls;

  if (Callee == 0) return;

  CodeMetrics &CalleeMetrics = CachedFunctionInfo[Callee].Metrics;

  // If we don't have metrics for the callee, don't recalculate them just to
  // update an approximation in the caller.  Instead, just recalculate the
  // caller info from scratch.
  if (CalleeMetrics.NumBlocks == 0) {
    resetCachedCostInfo(Caller);
    return;
  }

  // Since CalleeMetrics were already calculated, we know that the CallerMetrics
  // reference isn't invalidated: both were in the DenseMap.
  CallerMetrics.usesDynamicAlloca |= CalleeMetrics.usesDynamicAlloca;

  // FIXME: If any of these three are true for the callee, the callee was
  // not inlined into the caller, so I think they're redundant here.
  CallerMetrics.exposesReturnsTwice |= CalleeMetrics.exposesReturnsTwice;
  CallerMetrics.isRecursive |= CalleeMetrics.isRecursive;
  CallerMetrics.containsIndirectBr |= CalleeMetrics.containsIndirectBr;

  CallerMetrics.NumInsts += CalleeMetrics.NumInsts;
  CallerMetrics.NumBlocks += CalleeMetrics.NumBlocks;
  CallerMetrics.NumCalls += CalleeMetrics.NumCalls;
  CallerMetrics.NumVectorInsts += CalleeMetrics.NumVectorInsts;
  CallerMetrics.NumRets += CalleeMetrics.NumRets;

  // analyzeBasicBlock counts each function argument as an inst.
  if (CallerMetrics.NumInsts >= Callee->arg_size())
    CallerMetrics.NumInsts -= Callee->arg_size();
  else
    CallerMetrics.NumInsts = 0;

  // We are not updating the argument weights. We have already determined that
  // Caller is a fairly large function, so we accept the loss of precision.
}

/// clear - empty the cache of inline costs
void InlineCostAnalyzer::clear() {
  CachedFunctionInfo.clear();
}
