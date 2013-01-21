//===- CodeMetrics.cpp - Code cost measurements ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements code cost measurement utilities.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CallSite.h"

using namespace llvm;

/// callIsSmall - If a call is likely to lower to a single target instruction,
/// or is otherwise deemed small return true.
/// TODO: Perhaps calls like memcpy, strcpy, etc?
bool llvm::callIsSmall(ImmutableCallSite CS) {
  if (isa<IntrinsicInst>(CS.getInstruction()))
    return true;

  const Function *F = CS.getCalledFunction();
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
                                    const TargetTransformInfo &TTI) {
  ++NumBlocks;
  unsigned NumInstsBeforeThisBB = NumInsts;
  for (BasicBlock::const_iterator II = BB->begin(), E = BB->end();
       II != E; ++II) {
    if (TargetTransformInfo::TCC_Free == TTI.getUserCost(&*II))
      continue;

    // Special handling for calls.
    if (isa<CallInst>(II) || isa<InvokeInst>(II)) {
      ImmutableCallSite CS(cast<Instruction>(II));

      if (const Function *F = CS.getCalledFunction()) {
        // If a function is both internal and has a single use, then it is
        // extremely likely to get inlined in the future (it was probably
        // exposed by an interleaved devirtualization pass).
        if (!CS.isNoInline() && F->hasInternalLinkage() && F->hasOneUse())
          ++NumInlineCandidates;

        // If this call is to function itself, then the function is recursive.
        // Inlining it into other functions is a bad idea, because this is
        // basically just a form of loop peeling, and our metrics aren't useful
        // for that case.
        if (F == BB->getParent())
          isRecursive = true;
      }

      if (!callIsSmall(CS)) {
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

    if (const CallInst *CI = dyn_cast<CallInst>(II))
      if (CI->hasFnAttr(Attribute::NoDuplicate))
        notDuplicatable = true;

    if (const InvokeInst *InvI = dyn_cast<InvokeInst>(II))
      if (InvI->hasFnAttr(Attribute::NoDuplicate))
        notDuplicatable = true;

    ++NumInsts;
  }

  if (isa<ReturnInst>(BB->getTerminator()))
    ++NumRets;

  // We never want to inline functions that contain an indirectbr.  This is
  // incorrect because all the blockaddress's (in static global initializers
  // for example) would be referring to the original function, and this indirect
  // jump would jump from the inlined copy of the function into the original
  // function which is extremely undefined behavior.
  // FIXME: This logic isn't really right; we can safely inline functions
  // with indirectbr's as long as no other function or global references the
  // blockaddress of a block within the current function.  And as a QOI issue,
  // if someone is using a blockaddress without an indirectbr, and that
  // reference somehow ends up in another function or global, we probably
  // don't want to inline this function.
  notDuplicatable |= isa<IndirectBrInst>(BB->getTerminator());

  // Remember NumInsts for this BB.
  NumBBInsts[BB] = NumInsts - NumInstsBeforeThisBB;
}
