#define DEBUG_TYPE "lower-expect-intrinsic"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/LLVMContext.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Metadata.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include <vector>

using namespace llvm;

STATISTIC(IfHandled, "Number of 'expect' intrinsic intructions handled");

static cl::opt<uint32_t>
LikelyBranchWeight("likely-branch-weight", cl::Hidden, cl::init(64),
                   cl::desc("Weight of the branch likely to be taken (default = 64)"));
static cl::opt<uint32_t>
UnlikelyBranchWeight("unlikely-branch-weight", cl::Hidden, cl::init(4),
                   cl::desc("Weight of the branch unlikely to be taken (default = 4)"));

namespace {

  class LowerExpectIntrinsic : public FunctionPass {

    bool HandleSwitchExpect(SwitchInst *SI);

    bool HandleIfExpect(BranchInst *BI);

  public:
    static char ID;
    LowerExpectIntrinsic() : FunctionPass(ID) {
      initializeLowerExpectIntrinsicPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F);
  };
}


bool LowerExpectIntrinsic::HandleSwitchExpect(SwitchInst *SI) {
  CallInst *CI = dyn_cast<CallInst>(SI->getCondition());
  if (!CI)
    return false;

  Function *Fn = CI->getCalledFunction();
  if (!Fn || Fn->getIntrinsicID() != Intrinsic::expect)
    return false;

  Value *ArgValue = CI->getArgOperand(0);
  ConstantInt *ExpectedValue = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  if (!ExpectedValue)
    return false;

  LLVMContext &Context = CI->getContext();
  const Type *Int32Ty = Type::getInt32Ty(Context);

  unsigned caseNo = SI->findCaseValue(ExpectedValue);
  std::vector<Value *> Vec;
  unsigned n = SI->getNumCases();
  Vec.resize(n + 1); // +1 for MDString

  Vec[0] = MDString::get(Context, "branch_weights");
  for (unsigned i = 0; i < n; ++i) {
    Vec[i + 1] = ConstantInt::get(Int32Ty, i == caseNo ? LikelyBranchWeight : UnlikelyBranchWeight);
  }

  MDNode *WeightsNode = llvm::MDNode::get(Context, Vec);
  SI->setMetadata(LLVMContext::MD_prof, WeightsNode);

  SI->setCondition(ArgValue);
  return true;
}


bool LowerExpectIntrinsic::HandleIfExpect(BranchInst *BI) {
  if (BI->isUnconditional())
    return false;

  // Handle non-optimized IR code like:
  //   %expval = call i64 @llvm.expect.i64.i64(i64 %conv1, i64 1)
  //   %tobool = icmp ne i64 %expval, 0
  //   br i1 %tobool, label %if.then, label %if.end

  ICmpInst *CmpI = dyn_cast<ICmpInst>(BI->getCondition());
  if (!CmpI || CmpI->getPredicate() != CmpInst::ICMP_NE)
    return false;

  CallInst *CI = dyn_cast<CallInst>(CmpI->getOperand(0));
  if (!CI)
    return false;

  Function *Fn = CI->getCalledFunction();
  if (!Fn || Fn->getIntrinsicID() != Intrinsic::expect)
    return false;

  Value *ArgValue = CI->getArgOperand(0);
  ConstantInt *ExpectedValue = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  if (!ExpectedValue)
    return false;

  LLVMContext &Context = CI->getContext();
  const Type *Int32Ty = Type::getInt32Ty(Context);
  bool Likely = ExpectedValue->isOne();

  // If expect value is equal to 1 it means that we are more likely to take
  // branch 0, in other case more likely is branch 1.
  Value *Ops[] = {
    MDString::get(Context, "branch_weights"),
    ConstantInt::get(Int32Ty, Likely ? LikelyBranchWeight : UnlikelyBranchWeight),
    ConstantInt::get(Int32Ty, Likely ? UnlikelyBranchWeight : LikelyBranchWeight)
  };

  MDNode *WeightsNode = MDNode::get(Context, Ops);
  BI->setMetadata(LLVMContext::MD_prof, WeightsNode);

  CmpI->setOperand(0, ArgValue);
  return true;
}


bool LowerExpectIntrinsic::runOnFunction(Function &F) {
  for (Function::iterator I = F.begin(), E = F.end(); I != E;) {
    BasicBlock *BB = I++;

    // Create "block_weights" metadata.
    if (BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator())) {
      if (HandleIfExpect(BI))
        IfHandled++;
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(BB->getTerminator())) {
      if (HandleSwitchExpect(SI))
        IfHandled++;
    }

    // remove llvm.expect intrinsics.
    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end();
         BI != BE; ) {
      CallInst *CI = dyn_cast<CallInst>(BI++);
      if (!CI)
        continue;

      Function *Fn = CI->getCalledFunction();
      if (Fn && Fn->getIntrinsicID() == Intrinsic::expect) {
        Value *Exp = CI->getArgOperand(0);
        CI->replaceAllUsesWith(Exp);
        CI->eraseFromParent();
      }
    }
  }

  return false;
}


char LowerExpectIntrinsic::ID = 0;
INITIALIZE_PASS(LowerExpectIntrinsic, "lower-expect", "Lower 'expect' "
                "Intrinsics", false, false)

FunctionPass *llvm::createLowerExpectIntrinsicPass() {
  return new LowerExpectIntrinsic();
}
