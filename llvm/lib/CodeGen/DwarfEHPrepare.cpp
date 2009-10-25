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
#include "llvm/Target/TargetLowering.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
using namespace llvm;

STATISTIC(NumLandingPadsSplit,     "Number of landing pads split");
STATISTIC(NumUnwindsLowered,       "Number of unwind instructions lowered");
STATISTIC(NumExceptionValuesMoved, "Number of eh.exception calls moved");
STATISTIC(NumStackTempsIntroduced, "Number of stack temporaries introduced");

namespace {
  class DwarfEHPrepare : public FunctionPass {
    const TargetLowering *TLI;
    bool CompileFast;

    // The eh.exception intrinsic.
    Function *ExceptionValueIntrinsic;

    // _Unwind_Resume or the target equivalent.
    Constant *RewindFunction;

    // Dominator info is used when turning stack temporaries into registers.
    DominatorTree *DT;
    DominanceFrontier *DF;

    // The function we are running on.
    Function *F;

    // The landing pads for this function.
    typedef SmallPtrSet<BasicBlock*, 8> BBSet;
    BBSet LandingPads;

    // Stack temporary used to hold eh.exception values.
    AllocaInst *ExceptionValueVar;

    bool NormalizeLandingPads();
    bool LowerUnwinds();
    bool MoveExceptionValueCalls();
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
    DwarfEHPrepare(const TargetLowering *tli, bool fast) :
      FunctionPass(&ID), TLI(tli), CompileFast(fast),
      ExceptionValueIntrinsic(0), RewindFunction(0) {}

    virtual bool runOnFunction(Function &Fn);

    // getAnalysisUsage - We need dominance frontiers for memory promotion.
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      if (!CompileFast)
        AU.addRequired<DominatorTree>();
      AU.addPreserved<DominatorTree>();
      if (!CompileFast)
        AU.addRequired<DominanceFrontier>();
      AU.addPreserved<DominanceFrontier>();
    }

    const char *getPassName() const {
      return "Exception handling preparation";
    }

  };
} // end anonymous namespace

char DwarfEHPrepare::ID = 0;

FunctionPass *llvm::createDwarfEHPass(const TargetLowering *tli, bool fast) {
  return new DwarfEHPrepare(tli, fast);
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
    LLVMContext &Ctx = UnwindInsts[0]->getContext();
    std::vector<const Type*>
      Params(1, Type::getInt8PtrTy(Ctx));
    FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx),
                                          Params, false);
    const char *RewindName = TLI->getLibcallName(RTLIB::UNWIND_RESUME);
    RewindFunction = F->getParent()->getOrInsertFunction(RewindName, FTy);
  }

  bool Changed = false;

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
    new UnreachableInst(TI->getContext(), TI);

    // Nuke the unwind instruction.
    TI->eraseFromParent();
    ++NumUnwindsLowered;
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
  if (!ExceptionValueIntrinsic &&
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
    PromoteMemToReg(Allocas, *DT, *DF, ExceptionValueVar->getContext());
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
  if (!ExceptionValueIntrinsic)
    ExceptionValueIntrinsic = Intrinsic::getDeclaration(F->getParent(),
                                                       Intrinsic::eh_exception);

  // Create the call.
  return CallInst::Create(ExceptionValueIntrinsic, "eh.value.call", Start);
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
           Type::getInt8Ty(BB->getContext())), "eh.value", F->begin()->begin());
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
  F = &Fn;

  // Ensure that only unwind edges end at landing pads (a landing pad is a
  // basic block where an invoke unwind edge ends).
  Changed |= NormalizeLandingPads();

  // Turn unwind instructions into libcalls.
  Changed |= LowerUnwinds();

  // TODO: Move eh.selector calls to landing pads and combine them.

  // Move eh.exception calls to landing pads.
  Changed |= MoveExceptionValueCalls();

  // Initialize any stack temporaries we introduced.
  Changed |= FinishStackTemporaries();

  // Turn any stack temporaries into registers if possible.
  if (!CompileFast)
    Changed |= PromoteStackTemporaries();

  LandingPads.clear();

  return Changed;
}
