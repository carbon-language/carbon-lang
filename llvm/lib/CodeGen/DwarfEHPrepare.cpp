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

STATISTIC(NumResumesLowered, "Number of resume calls lowered");

namespace {
  class DwarfEHPrepare : public FunctionPass {
    const TargetMachine *TM;
    const TargetLowering *TLI;

    // RewindFunction - _Unwind_Resume or the target equivalent.
    Constant *RewindFunction;

    bool InsertUnwindResumeCalls(Function &Fn);

  public:
    static char ID; // Pass identification, replacement for typeid.
    DwarfEHPrepare(const TargetMachine *tm) :
      FunctionPass(ID), TM(tm), TLI(TM->getTargetLowering()),
      RewindFunction(0) {
        initializeDominatorTreePass(*PassRegistry::getPassRegistry());
      }

    virtual bool runOnFunction(Function &Fn);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const { }

    const char *getPassName() const {
      return "Exception handling preparation";
    }
  };
} // end anonymous namespace

char DwarfEHPrepare::ID = 0;

FunctionPass *llvm::createDwarfEHPass(const TargetMachine *tm) {
  return new DwarfEHPrepare(tm);
}

/// InsertUnwindResumeCalls - Convert the ResumeInsts that are still present
/// into calls to the appropriate _Unwind_Resume function.
bool DwarfEHPrepare::InsertUnwindResumeCalls(Function &Fn) {
  bool UsesNewEH = false;
  SmallVector<ResumeInst*, 16> Resumes;
  for (Function::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I) {
    TerminatorInst *TI = I->getTerminator();
    if (ResumeInst *RI = dyn_cast<ResumeInst>(TI))
      Resumes.push_back(RI);
    else if (InvokeInst *II = dyn_cast<InvokeInst>(TI))
      UsesNewEH = II->getUnwindDest()->isLandingPad();
  }

  if (Resumes.empty())
    return UsesNewEH;

  // Find the rewind function if we didn't already.
  if (!RewindFunction) {
    LLVMContext &Ctx = Resumes[0]->getContext();
    FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx),
                                          Type::getInt8PtrTy(Ctx), false);
    const char *RewindName = TLI->getLibcallName(RTLIB::UNWIND_RESUME);
    RewindFunction = Fn.getParent()->getOrInsertFunction(RewindName, FTy);
  }

  // Create the basic block where the _Unwind_Resume call will live.
  LLVMContext &Ctx = Fn.getContext();
  BasicBlock *UnwindBB = BasicBlock::Create(Ctx, "unwind_resume", &Fn);
  PHINode *PN = PHINode::Create(Type::getInt8PtrTy(Ctx), Resumes.size(),
                                "exn.obj", UnwindBB);

  // Extract the exception object from the ResumeInst and add it to the PHI node
  // that feeds the _Unwind_Resume call.
  for (SmallVectorImpl<ResumeInst*>::iterator
         I = Resumes.begin(), E = Resumes.end(); I != E; ++I) {
    ResumeInst *RI = *I;
    BranchInst::Create(UnwindBB, RI->getParent());
    ExtractValueInst *ExnObj = ExtractValueInst::Create(RI->getOperand(0),
                                                        0, "exn.obj", RI);
    PN->addIncoming(ExnObj, RI->getParent());
    RI->eraseFromParent();
    ++NumResumesLowered;
  }

  // Call the function.
  CallInst *CI = CallInst::Create(RewindFunction, PN, "", UnwindBB);
  CI->setCallingConv(TLI->getLibcallCallingConv(RTLIB::UNWIND_RESUME));

  // We never expect _Unwind_Resume to return.
  new UnreachableInst(Ctx, UnwindBB);
  return true;
}

bool DwarfEHPrepare::runOnFunction(Function &Fn) {
  bool Changed = InsertUnwindResumeCalls(Fn);
  return Changed;
}
