//===---- MipsOptimizeMathLibCalls.cpp - Optimize math lib calls.      ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass does an IR transformation which enables the backend to emit native
// math instructions.
//
//===----------------------------------------------------------------------===//

#include "MipsTargetMachine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

static cl::opt<bool> DisableOpt("disable-mips-math-optimization",
                                cl::init(false),
                                cl::desc("MIPS: Disable math lib call "
                                         "optimization."), cl::Hidden);

namespace {
  class MipsOptimizeMathLibCalls : public FunctionPass {
  public:
    static char ID;

    MipsOptimizeMathLibCalls(MipsTargetMachine &TM_) :
      FunctionPass(ID), TM(TM_) {}

    virtual const char *getPassName() const {
      return "MIPS: Optimize calls to math library functions.";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const;

    virtual bool runOnFunction(Function &F);

  private:
    /// Optimize calls to sqrt.
    bool optimizeSQRT(CallInst *Call, Function *CalledFunc,
                      BasicBlock &CurrBB,
                      Function::iterator &BB);

    const TargetMachine &TM;
  };

  char MipsOptimizeMathLibCalls::ID = 0;
}

FunctionPass *llvm::createMipsOptimizeMathLibCalls(MipsTargetMachine &TM) {
  return new MipsOptimizeMathLibCalls(TM);
}

void MipsOptimizeMathLibCalls::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetLibraryInfo>();
  FunctionPass::getAnalysisUsage(AU);
}

bool MipsOptimizeMathLibCalls::runOnFunction(Function &F) {
  if (DisableOpt)
    return false;

  const MipsSubtarget &Subtarget = TM.getSubtarget<MipsSubtarget>();

  if (Subtarget.inMips16Mode())
    return false;

  bool Changed = false;
  Function::iterator CurrBB;
  const TargetLibraryInfo *LibInfo = &getAnalysis<TargetLibraryInfo>();

  for (Function::iterator BB = F.begin(), BE = F.end(); BB != BE;) {
    CurrBB = BB++;

    for (BasicBlock::iterator II = CurrBB->begin(), IE = CurrBB->end();
         II != IE; ++II) {
      CallInst *Call = dyn_cast<CallInst>(&*II);
      Function *CalledFunc;

      if (!Call || !(CalledFunc = Call->getCalledFunction()))
        continue;

      LibFunc::Func LibFunc;
      Attribute A = CalledFunc->getAttributes()
        .getAttribute(AttributeSet::FunctionIndex, "use-soft-float");

      // Skip if function has "use-soft-float" attribute.
      if ((A.isStringAttribute() && (A.getValueAsString() == "true")) ||
          TM.Options.UseSoftFloat)
        continue;

      // Skip if function either has local linkage or is not a known library
      // function.
      if (CalledFunc->hasLocalLinkage() || !CalledFunc->hasName() ||
          !LibInfo->getLibFunc(CalledFunc->getName(), LibFunc))
        continue;

      switch (LibFunc) {
      case LibFunc::sqrtf:
      case LibFunc::sqrt:
        if (optimizeSQRT(Call, CalledFunc, *CurrBB, BB))
          break;
        continue;
      default:
        continue;
      }

      Changed = true;
      break;
    }
  }

  return Changed;
}

bool MipsOptimizeMathLibCalls::optimizeSQRT(CallInst *Call,
                                            Function *CalledFunc,
                                            BasicBlock &CurrBB,
                                            Function::iterator &BB) {
  // There is no need to change the IR, since backend will emit sqrt
  // instruction if the call has already been marked read-only.
  if (Call->onlyReadsMemory())
    return false;

  // Do the following transformation:
  //
  // (before)
  // dst = sqrt(src)
  //
  // (after)
  // v0 = sqrt_noreadmem(src) # native sqrt instruction.
  // if (v0 is a NaN)
  //   v1 = sqrt(src)         # library call.
  // dst = phi(v0, v1)
  //

  // Move all instructions following Call to newly created block JoinBB.
  // Create phi and replace all uses.
  BasicBlock *JoinBB = llvm::SplitBlock(&CurrBB, Call->getNextNode(), this);
  IRBuilder<> Builder(JoinBB, JoinBB->begin());
  PHINode *Phi = Builder.CreatePHI(Call->getType(), 2);
  Call->replaceAllUsesWith(Phi);

  // Create basic block LibCallBB and insert a call to library function sqrt.
  BasicBlock *LibCallBB = BasicBlock::Create(CurrBB.getContext(), "call.sqrt",
                                             CurrBB.getParent(), JoinBB);
  Builder.SetInsertPoint(LibCallBB);
  Instruction *LibCall = Call->clone();
  Builder.Insert(LibCall);
  Builder.CreateBr(JoinBB);

  // Add attribute "readnone" so that backend can use a native sqrt instruction
  // for this call. Insert a FP compare instruction and a conditional branch
  // at the end of CurrBB.
  Call->addAttribute(AttributeSet::FunctionIndex, Attribute::ReadNone);
  CurrBB.getTerminator()->eraseFromParent();
  Builder.SetInsertPoint(&CurrBB);
  Value *FCmp = Builder.CreateFCmpOEQ(Call, Call);
  Builder.CreateCondBr(FCmp, JoinBB, LibCallBB);

  // Add phi operands.
  Phi->addIncoming(Call, &CurrBB);
  Phi->addIncoming(LibCall, LibCallBB);

  BB = JoinBB;
  return true;
}
