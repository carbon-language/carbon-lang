//===-- DwarfEHPrepare - Prepare exception handling for code generation ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass mulches exception handling code into a form adapted to code
// generation. Required if using dwarf exception handling.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dwarfehprepare"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
using namespace llvm;

STATISTIC(NumLandingPadsSplit,     "Number of landing pads split");
STATISTIC(NumUnwindsLowered,       "Number of unwind instructions lowered");
STATISTIC(NumResumesLowered,       "Number of eh.resume calls lowered");
STATISTIC(NumExceptionValuesMoved, "Number of eh.exception calls moved");

namespace {
  class DwarfEHPrepare : public FunctionPass {
    const TargetMachine *TM;
    const TargetLowering *TLI;

    // The eh.exception intrinsic.
    Function *ExceptionValueIntrinsic;

    // The eh.selector intrinsic.
    Function *SelectorIntrinsic;

    // _Unwind_Resume_or_Rethrow or _Unwind_SjLj_Resume call.
    Constant *URoR;

    // The EH language-specific catch-all type.
    GlobalVariable *EHCatchAllValue;

    // _Unwind_Resume or the target equivalent.
    Constant *RewindFunction;

    // We both use and preserve dominator info.
    DominatorTree *DT;

    // The function we are running on.
    Function *F;

    // The landing pads for this function.
    typedef SmallPtrSet<BasicBlock*, 8> BBSet;
    BBSet LandingPads;

    bool NormalizeLandingPads();
    bool LowerUnwindsAndResumes();
    bool MoveExceptionValueCalls();

    Instruction *CreateExceptionValueCall(BasicBlock *BB);

    /// CleanupSelectors - Any remaining eh.selector intrinsic calls which still
    /// use the "llvm.eh.catch.all.value" call need to convert to using its
    /// initializer instead.
    bool CleanupSelectors(SmallPtrSet<IntrinsicInst*, 32> &Sels);

    bool HasCatchAllInSelector(IntrinsicInst *);

    /// FindAllCleanupSelectors - Find all eh.selector calls that are clean-ups.
    void FindAllCleanupSelectors(SmallPtrSet<IntrinsicInst*, 32> &Sels,
                                 SmallPtrSet<IntrinsicInst*, 32> &CatchAllSels);

    /// FindAllURoRInvokes - Find all URoR invokes in the function.
    void FindAllURoRInvokes(SmallPtrSet<InvokeInst*, 32> &URoRInvokes);

    /// HandleURoRInvokes - Handle invokes of "_Unwind_Resume_or_Rethrow" or
    /// "_Unwind_SjLj_Resume" calls. The "unwind" part of these invokes jump to
    /// a landing pad within the current function. This is a candidate to merge
    /// the selector associated with the URoR invoke with the one from the
    /// URoR's landing pad.
    bool HandleURoRInvokes();

    /// FindSelectorAndURoR - Find the eh.selector call and URoR call associated
    /// with the eh.exception call. This recursively looks past instructions
    /// which don't change the EH pointer value, like casts or PHI nodes.
    bool FindSelectorAndURoR(Instruction *Inst, bool &URoRInvoke,
                             SmallPtrSet<IntrinsicInst*, 8> &SelCalls,
                             SmallPtrSet<PHINode*, 32> &SeenPHIs);
      
  public:
    static char ID; // Pass identification, replacement for typeid.
    DwarfEHPrepare(const TargetMachine *tm) :
      FunctionPass(ID), TM(tm), TLI(TM->getTargetLowering()),
      ExceptionValueIntrinsic(0), SelectorIntrinsic(0),
      URoR(0), EHCatchAllValue(0), RewindFunction(0) {
        initializeDominatorTreePass(*PassRegistry::getPassRegistry());
      }

    virtual bool runOnFunction(Function &Fn);

    // getAnalysisUsage - We need the dominator tree for handling URoR.
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DominatorTree>();
      AU.addPreserved<DominatorTree>();
    }

    const char *getPassName() const {
      return "Exception handling preparation";
    }

  };
} // end anonymous namespace

char DwarfEHPrepare::ID = 0;

FunctionPass *llvm::createDwarfEHPass(const TargetMachine *tm) {
  return new DwarfEHPrepare(tm);
}

/// HasCatchAllInSelector - Return true if the intrinsic instruction has a
/// catch-all.
bool DwarfEHPrepare::HasCatchAllInSelector(IntrinsicInst *II) {
  if (!EHCatchAllValue) return false;

  unsigned ArgIdx = II->getNumArgOperands() - 1;
  GlobalVariable *GV = dyn_cast<GlobalVariable>(II->getArgOperand(ArgIdx));
  return GV == EHCatchAllValue;
}

/// FindAllCleanupSelectors - Find all eh.selector calls that are clean-ups.
void DwarfEHPrepare::
FindAllCleanupSelectors(SmallPtrSet<IntrinsicInst*, 32> &Sels,
                        SmallPtrSet<IntrinsicInst*, 32> &CatchAllSels) {
  for (Value::use_iterator
         I = SelectorIntrinsic->use_begin(),
         E = SelectorIntrinsic->use_end(); I != E; ++I) {
    IntrinsicInst *II = cast<IntrinsicInst>(*I);

    if (II->getParent()->getParent() != F)
      continue;

    if (!HasCatchAllInSelector(II))
      Sels.insert(II);
    else
      CatchAllSels.insert(II);
  }
}

/// FindAllURoRInvokes - Find all URoR invokes in the function.
void DwarfEHPrepare::
FindAllURoRInvokes(SmallPtrSet<InvokeInst*, 32> &URoRInvokes) {
  for (Value::use_iterator
         I = URoR->use_begin(),
         E = URoR->use_end(); I != E; ++I) {
    if (InvokeInst *II = dyn_cast<InvokeInst>(*I))
      URoRInvokes.insert(II);
  }
}

/// CleanupSelectors - Any remaining eh.selector intrinsic calls which still use
/// the "llvm.eh.catch.all.value" call need to convert to using its
/// initializer instead.
bool DwarfEHPrepare::CleanupSelectors(SmallPtrSet<IntrinsicInst*, 32> &Sels) {
  if (!EHCatchAllValue) return false;

  if (!SelectorIntrinsic) {
    SelectorIntrinsic =
      Intrinsic::getDeclaration(F->getParent(), Intrinsic::eh_selector);
    if (!SelectorIntrinsic) return false;
  }

  bool Changed = false;
  for (SmallPtrSet<IntrinsicInst*, 32>::iterator
         I = Sels.begin(), E = Sels.end(); I != E; ++I) {
    IntrinsicInst *Sel = *I;

    // Index of the "llvm.eh.catch.all.value" variable.
    unsigned OpIdx = Sel->getNumArgOperands() - 1;
    GlobalVariable *GV = dyn_cast<GlobalVariable>(Sel->getArgOperand(OpIdx));
    if (GV != EHCatchAllValue) continue;
    Sel->setArgOperand(OpIdx, EHCatchAllValue->getInitializer());
    Changed = true;
  }

  return Changed;
}

/// FindSelectorAndURoR - Find the eh.selector call associated with the
/// eh.exception call. And indicate if there is a URoR "invoke" associated with
/// the eh.exception call. This recursively looks past instructions which don't
/// change the EH pointer value, like casts or PHI nodes.
bool
DwarfEHPrepare::FindSelectorAndURoR(Instruction *Inst, bool &URoRInvoke,
                                    SmallPtrSet<IntrinsicInst*, 8> &SelCalls,
                                    SmallPtrSet<PHINode*, 32> &SeenPHIs) {
  bool Changed = false;

  for (Value::use_iterator
         I = Inst->use_begin(), E = Inst->use_end(); I != E; ++I) {
    Instruction *II = dyn_cast<Instruction>(*I);
    if (!II || II->getParent()->getParent() != F) continue;
    
    if (IntrinsicInst *Sel = dyn_cast<IntrinsicInst>(II)) {
      if (Sel->getIntrinsicID() == Intrinsic::eh_selector)
        SelCalls.insert(Sel);
    } else if (InvokeInst *Invoke = dyn_cast<InvokeInst>(II)) {
      if (Invoke->getCalledFunction() == URoR)
        URoRInvoke = true;
    } else if (CastInst *CI = dyn_cast<CastInst>(II)) {
      Changed |= FindSelectorAndURoR(CI, URoRInvoke, SelCalls, SeenPHIs);
    } else if (PHINode *PN = dyn_cast<PHINode>(II)) {
      if (SeenPHIs.insert(PN))
        // Don't process a PHI node more than once.
        Changed |= FindSelectorAndURoR(PN, URoRInvoke, SelCalls, SeenPHIs);
    }
  }

  return Changed;
}

/// HandleURoRInvokes - Handle invokes of "_Unwind_Resume_or_Rethrow" or
/// "_Unwind_SjLj_Resume" calls. The "unwind" part of these invokes jump to a
/// landing pad within the current function. This is a candidate to merge the
/// selector associated with the URoR invoke with the one from the URoR's
/// landing pad.
bool DwarfEHPrepare::HandleURoRInvokes() {
  if (!EHCatchAllValue) {
    EHCatchAllValue =
      F->getParent()->getNamedGlobal("llvm.eh.catch.all.value");
    if (!EHCatchAllValue) return false;
  }

  if (!SelectorIntrinsic) {
    SelectorIntrinsic =
      Intrinsic::getDeclaration(F->getParent(), Intrinsic::eh_selector);
    if (!SelectorIntrinsic) return false;
  }

  SmallPtrSet<IntrinsicInst*, 32> Sels;
  SmallPtrSet<IntrinsicInst*, 32> CatchAllSels;
  FindAllCleanupSelectors(Sels, CatchAllSels);

  if (!URoR) {
    URoR = F->getParent()->getFunction("_Unwind_Resume_or_Rethrow");
    if (!URoR) return CleanupSelectors(CatchAllSels);
  }

  SmallPtrSet<InvokeInst*, 32> URoRInvokes;
  FindAllURoRInvokes(URoRInvokes);

  SmallPtrSet<IntrinsicInst*, 32> SelsToConvert;

  for (SmallPtrSet<IntrinsicInst*, 32>::iterator
         SI = Sels.begin(), SE = Sels.end(); SI != SE; ++SI) {
    const BasicBlock *SelBB = (*SI)->getParent();
    for (SmallPtrSet<InvokeInst*, 32>::iterator
           UI = URoRInvokes.begin(), UE = URoRInvokes.end(); UI != UE; ++UI) {
      const BasicBlock *URoRBB = (*UI)->getParent();
      if (DT->dominates(SelBB, URoRBB)) {
        SelsToConvert.insert(*SI);
        break;
      }
    }
  }

  bool Changed = false;

  if (Sels.size() != SelsToConvert.size()) {
    // If we haven't been able to convert all of the clean-up selectors, then
    // loop through the slow way to see if they still need to be converted.
    if (!ExceptionValueIntrinsic) {
      ExceptionValueIntrinsic =
        Intrinsic::getDeclaration(F->getParent(), Intrinsic::eh_exception);
      if (!ExceptionValueIntrinsic)
        return CleanupSelectors(CatchAllSels);
    }

    for (Value::use_iterator
           I = ExceptionValueIntrinsic->use_begin(),
           E = ExceptionValueIntrinsic->use_end(); I != E; ++I) {
      IntrinsicInst *EHPtr = dyn_cast<IntrinsicInst>(*I);
      if (!EHPtr || EHPtr->getParent()->getParent() != F) continue;

      bool URoRInvoke = false;
      SmallPtrSet<IntrinsicInst*, 8> SelCalls;
      SmallPtrSet<PHINode*, 32> SeenPHIs;
      Changed |= FindSelectorAndURoR(EHPtr, URoRInvoke, SelCalls, SeenPHIs);

      if (URoRInvoke) {
        // This EH pointer is being used by an invoke of an URoR instruction and
        // an eh.selector intrinsic call. If the eh.selector is a 'clean-up', we
        // need to convert it to a 'catch-all'.
        for (SmallPtrSet<IntrinsicInst*, 8>::iterator
               SI = SelCalls.begin(), SE = SelCalls.end(); SI != SE; ++SI)
          if (!HasCatchAllInSelector(*SI))
              SelsToConvert.insert(*SI);
      }
    }
  }

  if (!SelsToConvert.empty()) {
    // Convert all clean-up eh.selectors, which are associated with "invokes" of
    // URoR calls, into catch-all eh.selectors.
    Changed = true;

    for (SmallPtrSet<IntrinsicInst*, 8>::iterator
           SI = SelsToConvert.begin(), SE = SelsToConvert.end();
         SI != SE; ++SI) {
      IntrinsicInst *II = *SI;

      // Use the exception object pointer and the personality function
      // from the original selector.
      CallSite CS(II);
      IntrinsicInst::op_iterator I = CS.arg_begin();
      IntrinsicInst::op_iterator E = CS.arg_end();
      IntrinsicInst::op_iterator B = prior(E);

      // Exclude last argument if it is an integer.
      if (isa<ConstantInt>(B)) E = B;

      // Add exception object pointer (front).
      // Add personality function (next).
      // Add in any filter IDs (rest).
      SmallVector<Value*, 8> Args(I, E);

      Args.push_back(EHCatchAllValue->getInitializer()); // Catch-all indicator.

      CallInst *NewSelector =
        CallInst::Create(SelectorIntrinsic, Args, "eh.sel.catch.all", II);

      NewSelector->setTailCall(II->isTailCall());
      NewSelector->setAttributes(II->getAttributes());
      NewSelector->setCallingConv(II->getCallingConv());

      II->replaceAllUsesWith(NewSelector);
      II->eraseFromParent();
    }
  }

  Changed |= CleanupSelectors(CatchAllSels);
  return Changed;
}

/// NormalizeLandingPads - Normalize and discover landing pads, noting them
/// in the LandingPads set.  A landing pad is normal if the only CFG edges
/// that end at it are unwind edges from invoke instructions. If we inlined
/// through an invoke we could have a normal branch from the previous
/// unwind block through to the landing pad for the original invoke.
/// Abnormal landing pads are fixed up by redirecting all unwind edges to
/// a new basic block which falls through to the original.
bool DwarfEHPrepare::NormalizeLandingPads() {
  bool Changed = false;

  const MCAsmInfo *MAI = TM->getMCAsmInfo();
  bool usingSjLjEH = MAI->getExceptionHandlingType() == ExceptionHandling::SjLj;

  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    TerminatorInst *TI = I->getTerminator();
    if (!isa<InvokeInst>(TI))
      continue;
    BasicBlock *LPad = TI->getSuccessor(1);
    // Skip landing pads that have already been normalized.
    if (LandingPads.count(LPad))
      continue;

    // Check that only invoke unwind edges end at the landing pad.
    bool OnlyUnwoundTo = true;
    bool SwitchOK = usingSjLjEH;
    for (pred_iterator PI = pred_begin(LPad), PE = pred_end(LPad);
         PI != PE; ++PI) {
      TerminatorInst *PT = (*PI)->getTerminator();
      // The SjLj dispatch block uses a switch instruction. This is effectively
      // an unwind edge, so we can disregard it here. There will only ever
      // be one dispatch, however, so if there are multiple switches, one
      // of them truly is a normal edge, not an unwind edge.
      if (SwitchOK && isa<SwitchInst>(PT)) {
        SwitchOK = false;
        continue;
      }
      if (!isa<InvokeInst>(PT) || LPad == PT->getSuccessor(0)) {
        OnlyUnwoundTo = false;
        break;
      }
    }

    if (OnlyUnwoundTo) {
      // Only unwind edges lead to the landing pad.  Remember the landing pad.
      LandingPads.insert(LPad);
      continue;
    }

    // At least one normal edge ends at the landing pad.  Redirect the unwind
    // edges to a new basic block which falls through into this one.

    // Create the new basic block.
    BasicBlock *NewBB = BasicBlock::Create(F->getContext(),
                                           LPad->getName() + "_unwind_edge");

    // Insert it into the function right before the original landing pad.
    LPad->getParent()->getBasicBlockList().insert(LPad, NewBB);

    // Redirect unwind edges from the original landing pad to NewBB.
    for (pred_iterator PI = pred_begin(LPad), PE = pred_end(LPad); PI != PE; ) {
      TerminatorInst *PT = (*PI++)->getTerminator();
      if (isa<InvokeInst>(PT) && PT->getSuccessor(1) == LPad)
        // Unwind to the new block.
        PT->setSuccessor(1, NewBB);
    }

    // If there are any PHI nodes in LPad, we need to update them so that they
    // merge incoming values from NewBB instead.
    for (BasicBlock::iterator II = LPad->begin(); isa<PHINode>(II); ++II) {
      PHINode *PN = cast<PHINode>(II);
      pred_iterator PB = pred_begin(NewBB), PE = pred_end(NewBB);

      // Check to see if all of the values coming in via unwind edges are the
      // same.  If so, we don't need to create a new PHI node.
      Value *InVal = PN->getIncomingValueForBlock(*PB);
      for (pred_iterator PI = PB; PI != PE; ++PI) {
        if (PI != PB && InVal != PN->getIncomingValueForBlock(*PI)) {
          InVal = 0;
          break;
        }
      }

      if (InVal == 0) {
        // Different unwind edges have different values.  Create a new PHI node
        // in NewBB.
        PHINode *NewPN = PHINode::Create(PN->getType(),
                                         PN->getNumIncomingValues(),
                                         PN->getName()+".unwind", NewBB);
        // Add an entry for each unwind edge, using the value from the old PHI.
        for (pred_iterator PI = PB; PI != PE; ++PI)
          NewPN->addIncoming(PN->getIncomingValueForBlock(*PI), *PI);

        // Now use this new PHI as the common incoming value for NewBB in PN.
        InVal = NewPN;
      }

      // Revector exactly one entry in the PHI node to come from NewBB
      // and delete all other entries that come from unwind edges.  If
      // there are both normal and unwind edges from the same predecessor,
      // this leaves an entry for the normal edge.
      for (pred_iterator PI = PB; PI != PE; ++PI)
        PN->removeIncomingValue(*PI);
      PN->addIncoming(InVal, NewBB);
    }

    // Add a fallthrough from NewBB to the original landing pad.
    BranchInst::Create(LPad, NewBB);

    // Now update DominatorTree analysis information.
    DT->splitBlock(NewBB);

    // Remember the newly constructed landing pad.  The original landing pad
    // LPad is no longer a landing pad now that all unwind edges have been
    // revectored to NewBB.
    LandingPads.insert(NewBB);
    ++NumLandingPadsSplit;
    Changed = true;
  }

  return Changed;
}

/// LowerUnwinds - Turn unwind instructions into calls to _Unwind_Resume,
/// rethrowing any previously caught exception.  This will crash horribly
/// at runtime if there is no such exception: using unwind to throw a new
/// exception is currently not supported.
bool DwarfEHPrepare::LowerUnwindsAndResumes() {
  SmallVector<Instruction*, 16> ResumeInsts;

  for (Function::iterator fi = F->begin(), fe = F->end(); fi != fe; ++fi) {
    for (BasicBlock::iterator bi = fi->begin(), be = fi->end(); bi != be; ++bi){
      if (isa<UnwindInst>(bi))
        ResumeInsts.push_back(bi);
      else if (CallInst *call = dyn_cast<CallInst>(bi))
        if (Function *fn = dyn_cast<Function>(call->getCalledValue()))
          if (fn->getName() == "llvm.eh.resume")
            ResumeInsts.push_back(bi);
    }
  }

  if (ResumeInsts.empty()) return false;

  // Find the rewind function if we didn't already.
  if (!RewindFunction) {
    LLVMContext &Ctx = ResumeInsts[0]->getContext();
    FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx),
                                          Type::getInt8PtrTy(Ctx), false);
    const char *RewindName = TLI->getLibcallName(RTLIB::UNWIND_RESUME);
    RewindFunction = F->getParent()->getOrInsertFunction(RewindName, FTy);
  }

  bool Changed = false;

  for (SmallVectorImpl<Instruction*>::iterator
         I = ResumeInsts.begin(), E = ResumeInsts.end(); I != E; ++I) {
    Instruction *RI = *I;

    // Replace the resuming instruction with a call to _Unwind_Resume (or the
    // appropriate target equivalent).

    llvm::Value *ExnValue;
    if (isa<UnwindInst>(RI))
      ExnValue = CreateExceptionValueCall(RI->getParent());
    else
      ExnValue = cast<CallInst>(RI)->getArgOperand(0);

    // Create the call...
    CallInst *CI = CallInst::Create(RewindFunction, ExnValue, "", RI);
    CI->setCallingConv(TLI->getLibcallCallingConv(RTLIB::UNWIND_RESUME));

    // ...followed by an UnreachableInst, if it was an unwind.
    // Calls to llvm.eh.resume are typically already followed by this.
    if (isa<UnwindInst>(RI))
      new UnreachableInst(RI->getContext(), RI);

    if (isa<UnwindInst>(RI))
      ++NumUnwindsLowered;
    else
      ++NumResumesLowered;

    // Nuke the resume instruction.
    RI->eraseFromParent();

    Changed = true;
  }

  return Changed;
}

/// MoveExceptionValueCalls - Ensure that eh.exception is only ever called from
/// landing pads by replacing calls outside of landing pads with direct use of
/// a register holding the appropriate value; this requires adding calls inside
/// all landing pads to initialize the register.  Also, move eh.exception calls
/// inside landing pads to the start of the landing pad (optional, but may make
/// things simpler for later passes).
bool DwarfEHPrepare::MoveExceptionValueCalls() {
  // If the eh.exception intrinsic is not declared in the module then there is
  // nothing to do.  Speed up compilation by checking for this common case.
  if (!ExceptionValueIntrinsic &&
      !F->getParent()->getFunction(Intrinsic::getName(Intrinsic::eh_exception)))
    return false;

  bool Changed = false;

  // Move calls to eh.exception that are inside a landing pad to the start of
  // the landing pad.
  for (BBSet::const_iterator LI = LandingPads.begin(), LE = LandingPads.end();
       LI != LE; ++LI) {
    BasicBlock *LP = *LI;
    for (BasicBlock::iterator II = LP->getFirstNonPHIOrDbg(), IE = LP->end();
         II != IE;)
      if (EHExceptionInst *EI = dyn_cast<EHExceptionInst>(II++)) {
        // Found a call to eh.exception.
        if (!EI->use_empty()) {
          // If there is already a call to eh.exception at the start of the
          // landing pad, then get hold of it; otherwise create such a call.
          Value *CallAtStart = CreateExceptionValueCall(LP);

          // If the call was at the start of a landing pad then leave it alone.
          if (EI == CallAtStart)
            continue;
          EI->replaceAllUsesWith(CallAtStart);
        }
        EI->eraseFromParent();
        ++NumExceptionValuesMoved;
        Changed = true;
      }
  }

  // Look for calls to eh.exception that are not in a landing pad.  If one is
  // found, then a register that holds the exception value will be created in
  // each landing pad, and the SSAUpdater will be used to compute the values
  // returned by eh.exception calls outside of landing pads.
  SSAUpdater SSA;

  // Remember where we found the eh.exception call, to avoid rescanning earlier
  // basic blocks which we already know contain no eh.exception calls.
  bool FoundCallOutsideLandingPad = false;
  Function::iterator BB = F->begin();
  for (Function::iterator BE = F->end(); BB != BE; ++BB) {
    // Skip over landing pads.
    if (LandingPads.count(BB))
      continue;

    for (BasicBlock::iterator II = BB->getFirstNonPHIOrDbg(), IE = BB->end();
         II != IE; ++II)
      if (isa<EHExceptionInst>(II)) {
        SSA.Initialize(II->getType(), II->getName());
        FoundCallOutsideLandingPad = true;
        break;
      }

    if (FoundCallOutsideLandingPad)
      break;
  }

  // If all calls to eh.exception are in landing pads then we are done.
  if (!FoundCallOutsideLandingPad)
    return Changed;

  // Add a call to eh.exception at the start of each landing pad, and tell the
  // SSAUpdater that this is the value produced by the landing pad.
  for (BBSet::iterator LI = LandingPads.begin(), LE = LandingPads.end();
       LI != LE; ++LI)
    SSA.AddAvailableValue(*LI, CreateExceptionValueCall(*LI));

  // Now turn all calls to eh.exception that are not in a landing pad into a use
  // of the appropriate register.
  for (Function::iterator BE = F->end(); BB != BE; ++BB) {
    // Skip over landing pads.
    if (LandingPads.count(BB))
      continue;

    for (BasicBlock::iterator II = BB->getFirstNonPHIOrDbg(), IE = BB->end();
         II != IE;)
      if (EHExceptionInst *EI = dyn_cast<EHExceptionInst>(II++)) {
        // Found a call to eh.exception, replace it with the value from any
        // upstream landing pad(s).
        EI->replaceAllUsesWith(SSA.GetValueAtEndOfBlock(BB));
        EI->eraseFromParent();
        ++NumExceptionValuesMoved;
      }
  }

  return true;
}

/// CreateExceptionValueCall - Insert a call to the eh.exception intrinsic at
/// the start of the basic block (unless there already is one, in which case
/// the existing call is returned).
Instruction *DwarfEHPrepare::CreateExceptionValueCall(BasicBlock *BB) {
  Instruction *Start = BB->getFirstNonPHIOrDbg();
  // Is this a call to eh.exception?
  if (IntrinsicInst *CI = dyn_cast<IntrinsicInst>(Start))
    if (CI->getIntrinsicID() == Intrinsic::eh_exception)
      // Reuse the existing call.
      return Start;

  // Find the eh.exception intrinsic if we didn't already.
  if (!ExceptionValueIntrinsic)
    ExceptionValueIntrinsic = Intrinsic::getDeclaration(F->getParent(),
                                                       Intrinsic::eh_exception);

  // Create the call.
  return CallInst::Create(ExceptionValueIntrinsic, "eh.value.call", Start);
}

bool DwarfEHPrepare::runOnFunction(Function &Fn) {
  bool Changed = false;

  // Initialize internal state.
  DT = &getAnalysis<DominatorTree>();
  F = &Fn;

  // Ensure that only unwind edges end at landing pads (a landing pad is a
  // basic block where an invoke unwind edge ends).
  Changed |= NormalizeLandingPads();

  // Turn unwind instructions and eh.resume calls into libcalls.
  Changed |= LowerUnwindsAndResumes();

  // TODO: Move eh.selector calls to landing pads and combine them.

  // Move eh.exception calls to landing pads.
  Changed |= MoveExceptionValueCalls();

  Changed |= HandleURoRInvokes();

  LandingPads.clear();

  return Changed;
}
