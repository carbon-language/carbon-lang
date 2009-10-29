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
// generation.  Required if using dwarf exception handling.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dwarfehprepare"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
using namespace llvm;

STATISTIC(NumExceptionValuesMoved, "Number of eh.exception calls moved");
STATISTIC(NumLonelyLandingPads,    "Number of landing pads with no selector");
STATISTIC(NumLonelySelectors,      "Number of lonely selectors lowered");
STATISTIC(NumLandingPadsSplit,     "Number of landing pads split");
STATISTIC(NumSelectorsAdjusted,    "Number of selector results adjusted");
STATISTIC(NumSelectorsSimplified,  "Number of selectors truncated");
STATISTIC(NumStackTempsIntroduced, "Number of stack temporaries introduced");
STATISTIC(NumUnwindsLowered,       "Number of unwind instructions lowered");

namespace {
  class DwarfEHPrepare : public FunctionPass {
    const TargetLowering *TLI;

    // The eh.exception intrinsic.
    Function *ExceptionIntrinsic;

    // The eh.selector intrinsic.
    Function *SelectorIntrinsic;

    // The eh.typeid.for intrinsic.
    Function *TypeIdIntrinsic;

    // _Unwind_Resume or the target equivalent.
    Constant *RewindFunction;

    // _Unwind_RaiseException.
    Constant *UnwindFunction;

    // Dominator info is used when turning stack temporaries into registers.
    DominatorTree *DT;
    DominanceFrontier *DF;

    // The function we are running on.
    Function *F;

    // The current context.
    LLVMContext *Context;

    // The personality and catch-all value for this function.
    Constant *Personality;
    Constant *CatchAll;

    // The landing pads for this function.
    typedef SmallPtrSet<BasicBlock*, 8> BBSet;
    BBSet LandingPads;

    // Stack temporary used to hold eh.exception values.
    AllocaInst *ExceptionValueVar;

    bool NormalizeLandingPads();
    bool LowerUnwinds();
    bool MoveSelectorCalls();
    bool RectifySelectorCalls();
    bool MoveExceptionValueCalls();
    bool AddMissingSelectors();
    bool FinishStackTemporaries();
    bool PromoteStackTemporaries();

    Instruction *CreateExceptionValueCall(BasicBlock *BB);
    Instruction *CreateValueLoad(BasicBlock *BB);

    /// CreateReadOfExceptionValue - Return the result of the eh.exception
    /// intrinsic by calling the intrinsic if in a landing pad, or loading
    /// it from the exception value variable otherwise.
    Instruction *CreateReadOfExceptionValue(BasicBlock *BB) {
      return LandingPads.count(BB) ?
        CreateExceptionValueCall(BB) : CreateValueLoad(BB);
    }

  public:
    static char ID; // Pass identification, replacement for typeid.
    DwarfEHPrepare(const TargetLowering *tli) :
      FunctionPass(&ID), TLI(tli), ExceptionIntrinsic(0),
      SelectorIntrinsic(0), TypeIdIntrinsic(0), RewindFunction(0),
      UnwindFunction(0) {}

    virtual bool runOnFunction(Function &Fn);

    const char *getPassName() const {
      return "Exception handling preparation";
    }

  };
} // end anonymous namespace

char DwarfEHPrepare::ID = 0;

FunctionPass *llvm::createDwarfEHPass(const TargetLowering *tli) {
  return new DwarfEHPrepare(tli);
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
    for (pred_iterator PI = pred_begin(LPad), PE = pred_end(LPad);
         PI != PE; ++PI) {
      TerminatorInst *PT = (*PI)->getTerminator();
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
    BasicBlock *NewBB = BasicBlock::Create(*Context,
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
        PHINode *NewPN = PHINode::Create(PN->getType(), PN->getName()+".unwind",
                                         NewBB);
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

    // Now update DominatorTree and DominanceFrontier analysis information.
    if (DT)
      DT->splitBlock(NewBB);
    if (DF)
      DF->splitBlock(NewBB);

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
bool DwarfEHPrepare::LowerUnwinds() {
  SmallVector<TerminatorInst*, 16> UnwindInsts;

  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    TerminatorInst *TI = I->getTerminator();
    if (isa<UnwindInst>(TI))
      UnwindInsts.push_back(TI);
  }

  if (UnwindInsts.empty()) return false;

  // Find the rewind function if we didn't already.
  if (!RewindFunction) {
    std::vector<const Type*>
      Params(1, Type::getInt8PtrTy(*Context));
    FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Context),
                                          Params, false);
    const char *RewindName = TLI->getLibcallName(RTLIB::UNWIND_RESUME);
    RewindFunction = F->getParent()->getOrInsertFunction(RewindName, FTy);
  }

  for (SmallVectorImpl<TerminatorInst*>::iterator
         I = UnwindInsts.begin(), E = UnwindInsts.end(); I != E; ++I) {
    TerminatorInst *TI = *I;

    // Replace the unwind instruction with a call to _Unwind_Resume (or the
    // appropriate target equivalent) followed by an UnreachableInst.

    // Create the call...
    CallInst *CI = CallInst::Create(RewindFunction,
                                    CreateReadOfExceptionValue(TI->getParent()),
                                    "", TI);
    CI->setCallingConv(TLI->getLibcallCallingConv(RTLIB::UNWIND_RESUME));
    // ...followed by an UnreachableInst.
    new UnreachableInst(*Context, TI);

    // Nuke the unwind instruction.
    TI->eraseFromParent();
    ++NumUnwindsLowered;
  }

  return true;
}

/// MoveSelectorCalls - Make sure that every call to eh.selector occurs in its
/// own landing pad, the landing pad corresponding to the exception object.
bool DwarfEHPrepare::MoveSelectorCalls() {
  // If the eh.selector intrinsic is not declared in the module then there is
  // nothing to do.  Speed up compilation by checking for this common case.
  if (!F->getParent()->getFunction(Intrinsic::getName(Intrinsic::eh_selector)))
    return false;

  // TODO: There is a lot of room for optimization here.

  bool Changed = false;
  BasicBlock *UnrBB = 0;

  for (Function::iterator BB = F->begin(); BB != F->end(); ++BB) {
    // If this basic block is not a landing pad then synthesize a landing pad
    // for every selector in it.
    bool SynthesizeLandingPad = !LandingPads.count(BB);

    for (BasicBlock::iterator II = BB->begin(), IE = BB->end(); II != IE; ++II) {
      EHSelectorInst *SI = dyn_cast<EHSelectorInst>(II);
      // Only interested in eh.selector calls.
      if (!SI)
        continue;

      // Note the personality and catch-all for later use.
      Personality = cast<Constant>(SI->getOperand(2));
      CatchAll = cast<Constant>(SI->getOperand(SI->getNumOperands() - 1)
                                ->stripPointerCasts());

      // The exception object.
      Value *Exception = SI->getOperand(1);

      if (!SynthesizeLandingPad) {
        // Did the exception come from unwinding to this landing pad or another?
        // If it comes from a different landing pad then we need to synthesize a
        // new landing pad for the selector.
        EHExceptionInst *EI = dyn_cast<EHExceptionInst>(Exception);
        SynthesizeLandingPad = !EI || EI->getParent() != BB;
      }

      if (!SynthesizeLandingPad) {
        // This is the first selector in this landing pad, and it is the landing
        // pad corresponding to the exception object.  No need to do anything to
        // this selector, but any subsequent selectors in this landing pad will
        // need their own invoke in order to make them independent of this one.
        SynthesizeLandingPad = true;
        continue;
      }

      // Rethrow the exception and catch it again, generating a landing pad for
      // this selector to live in.

      // Find _Unwind_RaiseException if we didn't already.
      if (!UnwindFunction) {
        std::vector<const Type*> ArgTys(1, Type::getInt8PtrTy(*Context));
        const FunctionType *FTy =
          FunctionType::get(Type::getInt32Ty(*Context), ArgTys, true);

        const char *Name = "_Unwind_RaiseException";
        UnwindFunction = F->getParent()->getOrInsertFunction(Name, FTy);
      }

      // Create a basic block containing only an unreachable instruction if we
      // didn't already.
      if (!UnrBB) {
        UnrBB = BasicBlock::Create(*Context, "unreachable", F);
        new UnreachableInst(*Context, UnrBB);
      }

      // Split the basic block before the selector.
      BasicBlock *NewBB = SplitBlock(BB, SI, this);

      // Replace the terminator with an invoke of _Unwind_RaiseException.
      BB->getTerminator()->eraseFromParent();
      InvokeInst::Create(UnwindFunction, UnrBB, NewBB, &Exception,
                         1 + &Exception, "", BB);

      // The split off basic block is now a landing pad.
      LandingPads.insert(NewBB);

      // Replace the exception argument in the selector call with a call to
      // eh.exception.  This is not really necessary but it makes things more
      // regular.
      Exception = CreateExceptionValueCall(NewBB);
      SI->setOperand(1, Exception);

      ++NumLonelySelectors;
      Changed = true;

      // All instructions still in the original basic block have been scanned.
      // Move on to the next basic block.
      break;
    }
  }

  return Changed;
}

/// RectifySelectorCalls - Remove useless catch-all clauses from the ends of
/// selectors, or correct the selector result for the presence of the catch-all
/// if it is really needed.
bool DwarfEHPrepare::RectifySelectorCalls() {
  // If the eh.selector intrinsic is not declared in the module then there is
  // nothing to do.  Speed up compilation by checking for this common case.
  if (!F->getParent()->getFunction(Intrinsic::getName(Intrinsic::eh_selector)))
    return false;

  bool Changed = false;

  for (BBSet::iterator I = LandingPads.begin(), E = LandingPads.end(); I != E;
       ++I)
    for (BasicBlock::iterator II = (*I)->begin(), IE = (*I)->end(); II != IE; )
      if (EHSelectorInst *SI = dyn_cast<EHSelectorInst>(II++)) {
        // Found a call to eh.selector.  Check whether it has a catch-all in the
        // middle.
        unsigned LastIndex = 0;
        for (unsigned i = 3, e = SI->getNumOperands() - 1; i < e; ++i) {
          Value *V = SI->getOperand(i);
          if (V->stripPointerCasts() == CatchAll) {
            // A catch-all.  The catch-all at the end was not needed.
            LastIndex = i;
            break;
          } else if (ConstantInt *FilterLength = dyn_cast<ConstantInt>(V)) {
            // A cleanup or a filter.
            unsigned Length = FilterLength->getZExtValue();
            if (Length == 0)
              // A cleanup - skip it.
              continue;
            if (Length == 1) {
              // A catch-all filter.  Drop everything that follows.
              LastIndex = i;
              break;
            }
            // A filter, skip over the typeinfos.
            i += Length - 1;
          }
        }

        if (LastIndex) {
          // Drop the pointless catch-all from the end.  In fact drop everything
          // after LastIndex as an optimization.
          SmallVector<Value*, 16> Args;
          Args.reserve(LastIndex);
          for (unsigned i = 1; i <= LastIndex; ++i)
            Args.push_back(SI->getOperand(i));
          CallInst *CI = CallInst::Create(SI->getOperand(0), Args.begin(),
                                          Args.end(), "", SI);
          CI->takeName(SI);
          SI->replaceAllUsesWith(CI);
          SI->eraseFromParent();
          ++NumSelectorsSimplified;
        } else if (!isa<ConstantInt>(CatchAll) && // Not a cleanup.
                   !SI->use_empty()) {
          // Correct the selector value to return zero if the catch-all matches.
          Constant *Zero = ConstantInt::getNullValue(Type::getInt32Ty(*Context));

          // Create the new selector value, with placeholders instead of the
          // real operands and make everyone use it.  The reason for this round
          // about approach is that the computation of the new value makes use
          // of the old value, so we can't just compute it then do RAUW.
          SelectInst *S = SelectInst::Create(ConstantInt::getFalse(*Context),
                                             Zero, Zero, "", II);
          SI->replaceAllUsesWith(S);

          // Now calculate the operands of the select.
          IRBuilder<> Builder(*I, S);

          // Find the eh.typeid.for intrinsic if we didn't already.
          if (!TypeIdIntrinsic)
            TypeIdIntrinsic = Intrinsic::getDeclaration(F->getParent(),
                                                      Intrinsic::eh_typeid_for);

          // Obtain the id of the catch-all.
          Value *CatchAllId = Builder.CreateCall(TypeIdIntrinsic,
              ConstantExpr::getBitCast(CatchAll, Type::getInt8PtrTy(*Context)));

          // Compare it with the original selector result.  If it matched then
          // the selector result is zero, otherwise it is the original selector.
          Value *MatchesCatchAll = Builder.CreateICmpEQ(SI, CatchAllId);
          S->setOperand(0, MatchesCatchAll);
          S->setOperand(2, SI);
          ++NumSelectorsAdjusted;
        }

        Changed = true;
        break;
      }

  return Changed;
}

/// Make sure every landing pad has a selector in it.
bool DwarfEHPrepare::AddMissingSelectors() {
  if (!Personality)
    // We only know how to codegen invokes if there is a personality.
    // FIXME: This results in wrong code.
    return false;

  bool Changed = false;

  for (BBSet::iterator I = LandingPads.begin(), E = LandingPads.end(); I != E;
       ++I) {
    bool FoundSelector = false;

    // Check whether the landing pad already contains a call to eh.selector.
    for (BasicBlock::iterator II = (*I)->begin(), IE = (*I)->end(); II != IE;
         ++II)
      if (isa<EHSelectorInst>(II)) {
        FoundSelector = true;
        break;
      }

    if (FoundSelector)
      continue;

    // Find the eh.selector intrinsic if we didn't already.
    if (!SelectorIntrinsic)
      SelectorIntrinsic = Intrinsic::getDeclaration(F->getParent(),
                                                    Intrinsic::eh_selector);

    // Get the exception object.
    Instruction *Exception = CreateExceptionValueCall(*I);

    Value *Args[3] = { Exception, Personality, CatchAll };
    CallInst *Selector = CallInst::Create(SelectorIntrinsic, Args, Args + 3);
    Selector->insertAfter(Exception);

    ++NumLonelyLandingPads;
    Changed = true;
  }

  return Changed;
}

/// MoveExceptionValueCalls - Ensure that eh.exception is only ever called from
/// landing pads by replacing calls outside of landing pads with loads from a
/// stack temporary.  Move eh.exception calls inside landing pads to the start
/// of the landing pad (optional, but may make things simpler for later passes).
bool DwarfEHPrepare::MoveExceptionValueCalls() {
  // If the eh.exception intrinsic is not declared in the module then there is
  // nothing to do.  Speed up compilation by checking for this common case.
  if (!ExceptionIntrinsic &&
      !F->getParent()->getFunction(Intrinsic::getName(Intrinsic::eh_exception)))
    return false;

  bool Changed = false;

  for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E;)
      if (IntrinsicInst *CI = dyn_cast<IntrinsicInst>(II++))
        if (CI->getIntrinsicID() == Intrinsic::eh_exception) {
          if (!CI->use_empty()) {
            Value *ExceptionValue = CreateReadOfExceptionValue(BB);
            if (CI == ExceptionValue) {
              // The call was at the start of a landing pad - leave it alone.
              assert(LandingPads.count(BB) &&
                     "Created eh.exception call outside landing pad!");
              continue;
            }
            CI->replaceAllUsesWith(ExceptionValue);
          }
          CI->eraseFromParent();
          ++NumExceptionValuesMoved;
          Changed = true;
        }
  }

  return Changed;
}

/// FinishStackTemporaries - If we introduced a stack variable to hold the
/// exception value then initialize it in each landing pad.
bool DwarfEHPrepare::FinishStackTemporaries() {
  if (!ExceptionValueVar)
    // Nothing to do.
    return false;

  bool Changed = false;

  // Make sure that there is a store of the exception value at the start of
  // each landing pad.
  for (BBSet::iterator LI = LandingPads.begin(), LE = LandingPads.end();
       LI != LE; ++LI) {
    Instruction *ExceptionValue = CreateReadOfExceptionValue(*LI);
    Instruction *Store = new StoreInst(ExceptionValue, ExceptionValueVar);
    Store->insertAfter(ExceptionValue);
    Changed = true;
  }

  return Changed;
}

/// PromoteStackTemporaries - Turn any stack temporaries we introduced into
/// registers if possible.
bool DwarfEHPrepare::PromoteStackTemporaries() {
  if (ExceptionValueVar && DT && DF && isAllocaPromotable(ExceptionValueVar)) {
    // Turn the exception temporary into registers and phi nodes if possible.
    std::vector<AllocaInst*> Allocas(1, ExceptionValueVar);
    PromoteMemToReg(Allocas, *DT, *DF, *Context);
    return true;
  }
  return false;
}

/// CreateExceptionValueCall - Insert a call to the eh.exception intrinsic at
/// the start of the basic block (unless there already is one, in which case
/// the existing call is returned).
Instruction *DwarfEHPrepare::CreateExceptionValueCall(BasicBlock *BB) {
  Instruction *Start = BB->getFirstNonPHI();
  // Is this a call to eh.exception?
  if (IntrinsicInst *CI = dyn_cast<IntrinsicInst>(Start))
    if (CI->getIntrinsicID() == Intrinsic::eh_exception)
      // Reuse the existing call.
      return Start;

  // Find the eh.exception intrinsic if we didn't already.
  if (!ExceptionIntrinsic)
    ExceptionIntrinsic = Intrinsic::getDeclaration(F->getParent(),
                                                       Intrinsic::eh_exception);

  // Create the call.
  return CallInst::Create(ExceptionIntrinsic, "eh.value.call", Start);
}

/// CreateValueLoad - Insert a load of the exception value stack variable
/// (creating it if necessary) at the start of the basic block (unless
/// there already is a load, in which case the existing load is returned).
Instruction *DwarfEHPrepare::CreateValueLoad(BasicBlock *BB) {
  Instruction *Start = BB->getFirstNonPHI();
  // Is this a load of the exception temporary?
  if (ExceptionValueVar)
    if (LoadInst* LI = dyn_cast<LoadInst>(Start))
      if (LI->getPointerOperand() == ExceptionValueVar)
        // Reuse the existing load.
        return Start;

  // Create the temporary if we didn't already.
  if (!ExceptionValueVar) {
    ExceptionValueVar = new AllocaInst(PointerType::getUnqual(
           Type::getInt8Ty(*Context)), "eh.value", F->begin()->begin());
    ++NumStackTempsIntroduced;
  }

  // Load the value.
  return new LoadInst(ExceptionValueVar, "eh.value.load", Start);
}

bool DwarfEHPrepare::runOnFunction(Function &Fn) {
  bool Changed = false;

  // Initialize internal state.
  DT = getAnalysisIfAvailable<DominatorTree>();
  DF = getAnalysisIfAvailable<DominanceFrontier>();
  ExceptionValueVar = 0;
  Personality = 0;
  CatchAll = 0;
  Context = &Fn.getContext();
  F = &Fn;

  // Ensure that only unwind edges end at landing pads (a landing pad is a
  // basic block where an invoke unwind edge ends).
  Changed |= NormalizeLandingPads();

  // Turn unwind instructions into libcalls.
  Changed |= LowerUnwinds();

  // Make sure that every call to eh.selector occurs in its own landing pad.
  Changed |= MoveSelectorCalls();

  // Remove useless catch-all clauses from the ends of selectors, or correct the
  // selector result for the presence of the catch-all if it is really needed.
  Changed |= RectifySelectorCalls();

  // Make sure every landing pad has a selector in it.
  Changed |= AddMissingSelectors();

  // Move eh.exception calls to landing pads.
  Changed |= MoveExceptionValueCalls();

  // Initialize any stack temporaries we introduced.
  Changed |= FinishStackTemporaries();

  // Turn any stack temporaries into registers if possible.
//TODO  if (!CompileFast)
//TODO    Changed |= PromoteStackTemporaries();

  LandingPads.clear();

  return Changed;
}
