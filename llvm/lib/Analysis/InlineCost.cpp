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

/// callIsSmall - If a call is likely to lower to a single target instruction,
/// or is otherwise deemed small return true.
/// TODO: Perhaps calls like memcpy, strcpy, etc?
bool llvm::callIsSmall(const Function *F) {
  if (!F) return false;

  if (F->hasLocalLinkage()) return false;

  if (!F->hasName()) return false;

  StringRef Name = F->getName();

  // These will all likely lower to a single selection DAG node.
  if (Name == "copysign" || Name == "copysignf" || Name == "copysignl" ||
      Name == "fabs" || Name == "fabsf" || Name == "fabsl" ||
      Name == "sin" || Name == "sinf" || Name == "sinl" ||
      Name == "cos" || Name == "cosf" || Name == "cosl" ||
      Name == "sqrt" || Name == "sqrtf" || Name == "sqrtl" )
    return true;

  // These are all likely to be optimized into something smaller.
  if (Name == "pow" || Name == "powf" || Name == "powl" ||
      Name == "exp2" || Name == "exp2l" || Name == "exp2f" ||
      Name == "floor" || Name == "floorf" || Name == "ceil" ||
      Name == "round" || Name == "ffs" || Name == "ffsl" ||
      Name == "abs" || Name == "labs" || Name == "llabs")
    return true;

  return false;
}

/// analyzeBasicBlock - Fill in the current structure with information gleaned
/// from the specified block.
void CodeMetrics::analyzeBasicBlock(const BasicBlock *BB,
                                    const TargetData *TD) {
  ++NumBlocks;
  unsigned NumInstsBeforeThisBB = NumInsts;
  for (BasicBlock::const_iterator II = BB->begin(), E = BB->end();
       II != E; ++II) {
    if (isa<PHINode>(II)) continue;           // PHI nodes don't count.

    // Special handling for calls.
    if (isa<CallInst>(II) || isa<InvokeInst>(II)) {
      if (isa<DbgInfoIntrinsic>(II))
        continue;  // Debug intrinsics don't count as size.

      ImmutableCallSite CS(cast<Instruction>(II));

      if (const Function *F = CS.getCalledFunction()) {
        // If a function is both internal and has a single use, then it is
        // extremely likely to get inlined in the future (it was probably
        // exposed by an interleaved devirtualization pass).
        if (F->hasInternalLinkage() && F->hasOneUse())
          ++NumInlineCandidates;

        // If this call is to function itself, then the function is recursive.
        // Inlining it into other functions is a bad idea, because this is
        // basically just a form of loop peeling, and our metrics aren't useful
        // for that case.
        if (F == BB->getParent())
          isRecursive = true;
      }

      if (!isa<IntrinsicInst>(II) && !callIsSmall(CS.getCalledFunction())) {
        // Each argument to a call takes on average one instruction to set up.
        NumInsts += CS.arg_size();

        // We don't want inline asm to count as a call - that would prevent loop
        // unrolling. The argument setup cost is still real, though.
        if (!isa<InlineAsm>(CS.getCalledValue()))
          ++NumCalls;
      }
    }

    if (const AllocaInst *AI = dyn_cast<AllocaInst>(II)) {
      if (!AI->isStaticAlloca())
        this->usesDynamicAlloca = true;
    }

    if (isa<ExtractElementInst>(II) || II->getType()->isVectorTy())
      ++NumVectorInsts;

    if (const CastInst *CI = dyn_cast<CastInst>(II)) {
      // Noop casts, including ptr <-> int,  don't count.
      if (CI->isLosslessCast() || isa<IntToPtrInst>(CI) ||
          isa<PtrToIntInst>(CI))
        continue;
      // trunc to a native type is free (assuming the target has compare and
      // shift-right of the same width).
      if (isa<TruncInst>(CI) && TD &&
          TD->isLegalInteger(TD->getTypeSizeInBits(CI->getType())))
        continue;
      // Result of a cmp instruction is often extended (to be used by other
      // cmp instructions, logical or return instructions). These are usually
      // nop on most sane targets.
      if (isa<CmpInst>(CI->getOperand(0)))
        continue;
    } else if (const GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(II)){
      // If a GEP has all constant indices, it will probably be folded with
      // a load/store.
      if (GEPI->hasAllConstantIndices())
        continue;
    }

    ++NumInsts;
  }

  if (isa<ReturnInst>(BB->getTerminator()))
    ++NumRets;

  // We never want to inline functions that contain an indirectbr.  This is
  // incorrect because all the blockaddress's (in static global initializers
  // for example) would be referring to the original function, and this indirect
  // jump would jump from the inlined copy of the function into the original
  // function which is extremely undefined behavior.
  if (isa<IndirectBrInst>(BB->getTerminator()))
    containsIndirectBr = true;

  // Remember NumInsts for this BB.
  NumBBInsts[BB] = NumInsts - NumInstsBeforeThisBB;
}

// CountCodeReductionForConstant - Figure out an approximation for how many
// instructions will be constant folded if the specified value is constant.
//
unsigned CodeMetrics::CountCodeReductionForConstant(Value *V) {
  unsigned Reduction = 0;
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E;++UI){
    User *U = *UI;
    if (isa<BranchInst>(U) || isa<SwitchInst>(U)) {
      // We will be able to eliminate all but one of the successors.
      const TerminatorInst &TI = cast<TerminatorInst>(*U);
      const unsigned NumSucc = TI.getNumSuccessors();
      unsigned Instrs = 0;
      for (unsigned I = 0; I != NumSucc; ++I)
        Instrs += NumBBInsts[TI.getSuccessor(I)];
      // We don't know which blocks will be eliminated, so use the average size.
      Reduction += InlineConstants::InstrCost*Instrs*(NumSucc-1)/NumSucc;
    } else {
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

      if (AllOperandsConstant) {
        // We will get to remove this instruction...
        Reduction += InlineConstants::InstrCost;

        // And any other instructions that use it which become constants
        // themselves.
        Reduction += CountCodeReductionForConstant(&Inst);
      }
    }
  }
  return Reduction;
}

// CountCodeReductionForAlloca - Figure out an approximation of how much smaller
// the function will be if it is inlined into a context where an argument
// becomes an alloca.
//
unsigned CodeMetrics::CountCodeReductionForAlloca(Value *V) {
  if (!V->getType()->isPointerTy()) return 0;  // Not a pointer
  unsigned Reduction = 0;
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E;++UI){
    Instruction *I = cast<Instruction>(*UI);
    if (isa<LoadInst>(I) || isa<StoreInst>(I))
      Reduction += InlineConstants::InstrCost;
    else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
      // If the GEP has variable indices, we won't be able to do much with it.
      if (GEP->hasAllConstantIndices())
        Reduction += CountCodeReductionForAlloca(GEP);
    } else if (BitCastInst *BCI = dyn_cast<BitCastInst>(I)) {
      // Track pointer through bitcasts.
      Reduction += CountCodeReductionForAlloca(BCI);
    } else {
      // If there is some other strange instruction, we're not going to be able
      // to do much if we inline this.
      return 0;
    }
  }

  return Reduction;
}

/// analyzeFunction - Fill in the current structure with information gleaned
/// from the specified function.
void CodeMetrics::analyzeFunction(Function *F, const TargetData *TD) {
  // If this function contains a call that "returns twice" (e.g., setjmp or
  // _setjmp), never inline it. This is a hack because we depend on the user
  // marking their local variables as volatile if they are live across a setjmp
  // call, and they probably won't do this in callers.
  callsSetJmp = F->callsFunctionThatReturnsTwice();

  // Look at the size of the callee.
  for (Function::const_iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
    analyzeBasicBlock(&*BB, TD);
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

  // Check out all of the arguments to the function, figuring out how much
  // code can be eliminated if one of the arguments is a constant.
  ArgumentWeights.reserve(F->arg_size());
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E; ++I)
    ArgumentWeights.push_back(ArgInfo(Metrics.CountCodeReductionForConstant(I),
                                      Metrics.CountCodeReductionForAlloca(I)));
}

/// NeverInline - returns true if the function should never be inlined into
/// any caller
bool InlineCostAnalyzer::FunctionInfo::NeverInline() {
  return (Metrics.callsSetJmp || Metrics.isRecursive ||
          Metrics.containsIndirectBr);
}
// getSpecializationBonus - The heuristic used to determine the per-call
// performance boost for using a specialization of Callee with argument
// specializedArgNo replaced by a constant.
int InlineCostAnalyzer::getSpecializationBonus(Function *Callee,
         SmallVectorImpl<unsigned> &SpecializedArgNos)
{
  if (Callee->mayBeOverridden())
    return 0;

  int Bonus = 0;
  // If this function uses the coldcc calling convention, prefer not to
  // specialize it.
  if (Callee->getCallingConv() == CallingConv::Cold)
    Bonus -= InlineConstants::ColdccPenalty;

  // Get information about the callee.
  FunctionInfo *CalleeFI = &CachedFunctionInfo[Callee];

  // If we haven't calculated this information yet, do so now.
  if (CalleeFI->Metrics.NumBlocks == 0)
    CalleeFI->analyzeFunction(Callee, TD);

  unsigned ArgNo = 0;
  unsigned i = 0;
  for (Function::arg_iterator I = Callee->arg_begin(), E = Callee->arg_end();
       I != E; ++I, ++ArgNo)
    if (ArgNo == SpecializedArgNos[i]) {
      ++i;
      Bonus += CountBonusForConstant(I);
    }

  // Calls usually take a long time, so they make the specialization gain
  // smaller.
  Bonus -= CalleeFI->Metrics.NumCalls * InlineConstants::CallPenalty;

  return Bonus;
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

  // Each argument passed in has a cost at both the caller and the callee
  // sides.  Measurements show that each argument costs about the same as an
  // instruction.
  InlineCost -= (CS.arg_size() * InlineConstants::InstrCost);

  // Now that we have considered all of the factors that make the call site more
  // likely to be inlined, look at factors that make us not want to inline it.

  // Calls usually take a long time, so they make the inlining gain smaller.
  InlineCost += CalleeFI->Metrics.NumCalls * InlineConstants::CallPenalty;

  // Look at the size of the callee. Each instruction counts as 5.
  InlineCost += CalleeFI->Metrics.NumInsts*InlineConstants::InstrCost;

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
InlineCost InlineCostAnalyzer::getInlineCost(CallSite CS,
                               SmallPtrSet<const Function*, 16> &NeverInline) {
  return getInlineCost(CS, CS.getCalledFunction(), NeverInline);
}

InlineCost InlineCostAnalyzer::getInlineCost(CallSite CS,
                               Function *Callee,
                               SmallPtrSet<const Function*, 16> &NeverInline) {
  Instruction *TheCall = CS.getInstruction();
  Function *Caller = TheCall->getParent()->getParent();

  // Don't inline functions which can be redefined at link-time to mean
  // something else.  Don't inline functions marked noinline or call sites
  // marked noinline.
  if (Callee->mayBeOverridden() ||
      Callee->hasFnAttr(Attribute::NoInline) || NeverInline.count(Callee) ||
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

// getSpecializationCost - The heuristic used to determine the code-size
// impact of creating a specialized version of Callee with argument
// SpecializedArgNo replaced by a constant.
InlineCost InlineCostAnalyzer::getSpecializationCost(Function *Callee,
                               SmallVectorImpl<unsigned> &SpecializedArgNos)
{
  // Don't specialize functions which can be redefined at link-time to mean
  // something else.
  if (Callee->mayBeOverridden())
    return llvm::InlineCost::getNever();

  // Get information about the callee.
  FunctionInfo *CalleeFI = &CachedFunctionInfo[Callee];

  // If we haven't calculated this information yet, do so now.
  if (CalleeFI->Metrics.NumBlocks == 0)
    CalleeFI->analyzeFunction(Callee, TD);

  int Cost = 0;

  // Look at the original size of the callee.  Each instruction counts as 5.
  Cost += CalleeFI->Metrics.NumInsts * InlineConstants::InstrCost;

  // Offset that with the amount of code that can be constant-folded
  // away with the given arguments replaced by constants.
  for (SmallVectorImpl<unsigned>::iterator an = SpecializedArgNos.begin(),
       ae = SpecializedArgNos.end(); an != ae; ++an)
    Cost -= CalleeFI->ArgumentWeights[*an].ConstantWeight;

  return llvm::InlineCost::get(Cost);
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
  CallerMetrics.callsSetJmp |= CalleeMetrics.callsSetJmp;
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
