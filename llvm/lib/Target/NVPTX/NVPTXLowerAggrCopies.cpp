//===- NVPTXLowerAggrCopies.cpp - ------------------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Lower aggregate copies, memset, memcpy, memmov intrinsics into loops when
// the size is large or is not a compile-time constant.
//
//===----------------------------------------------------------------------===//

#include "NVPTXLowerAggrCopies.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "nvptx"

using namespace llvm;

namespace {

// actual analysis class, which is a functionpass
struct NVPTXLowerAggrCopies : public FunctionPass {
  static char ID;

  NVPTXLowerAggrCopies() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<MachineFunctionAnalysis>();
    AU.addPreserved<StackProtector>();
  }

  bool runOnFunction(Function &F) override;

  static const unsigned MaxAggrCopySize = 128;

  const char *getPassName() const override {
    return "Lower aggregate copies/intrinsics into loops";
  }
};

char NVPTXLowerAggrCopies::ID = 0;

// Lower memcpy to loop.
void convertMemCpyToLoop(Instruction *splitAt, Value *srcAddr, Value *dstAddr,
                         Value *len, bool srcVolatile, bool dstVolatile,
                         LLVMContext &Context, Function &F) {
  Type *indType = len->getType();

  BasicBlock *origBB = splitAt->getParent();
  BasicBlock *newBB = splitAt->getParent()->splitBasicBlock(splitAt, "split");
  BasicBlock *loopBB = BasicBlock::Create(Context, "loadstoreloop", &F, newBB);

  origBB->getTerminator()->setSuccessor(0, loopBB);
  IRBuilder<> builder(origBB, origBB->getTerminator());

  // srcAddr and dstAddr are expected to be pointer types,
  // so no check is made here.
  unsigned srcAS = cast<PointerType>(srcAddr->getType())->getAddressSpace();
  unsigned dstAS = cast<PointerType>(dstAddr->getType())->getAddressSpace();

  // Cast pointers to (char *)
  srcAddr = builder.CreateBitCast(srcAddr, Type::getInt8PtrTy(Context, srcAS));
  dstAddr = builder.CreateBitCast(dstAddr, Type::getInt8PtrTy(Context, dstAS));

  IRBuilder<> loop(loopBB);
  // The loop index (ind) is a phi node.
  PHINode *ind = loop.CreatePHI(indType, 0);
  // Incoming value for ind is 0
  ind->addIncoming(ConstantInt::get(indType, 0), origBB);

  // load from srcAddr+ind
  // TODO: we can leverage the align parameter of llvm.memcpy for more efficient
  // word-sized loads and stores.
  Value *val = loop.CreateLoad(loop.CreateGEP(loop.getInt8Ty(), srcAddr, ind),
                               srcVolatile);
  // store at dstAddr+ind
  loop.CreateStore(val, loop.CreateGEP(loop.getInt8Ty(), dstAddr, ind),
                   dstVolatile);

  // The value for ind coming from backedge is (ind + 1)
  Value *newind = loop.CreateAdd(ind, ConstantInt::get(indType, 1));
  ind->addIncoming(newind, loopBB);

  loop.CreateCondBr(loop.CreateICmpULT(newind, len), loopBB, newBB);
}

// Lower memmove to IR. memmove is required to correctly copy overlapping memory
// regions; therefore, it has to check the relative positions of the source and
// destination pointers and choose the copy direction accordingly.
//
// The code below is an IR rendition of this C function:
//
// void* memmove(void* dst, const void* src, size_t n) {
//   unsigned char* d = dst;
//   const unsigned char* s = src;
//   if (s < d) {
//     // copy backwards
//     while (n--) {
//       d[n] = s[n];
//     }
//   } else {
//     // copy forward
//     for (size_t i = 0; i < n; ++i) {
//       d[i] = s[i];
//     }
//   }
//   return dst;
// }
void convertMemMoveToLoop(Instruction *splitAt, Value *srcAddr, Value *dstAddr,
                          Value *len, bool srcVolatile, bool dstVolatile,
                          LLVMContext &Context, Function &F) {
  Type *TypeOfLen = len->getType();
  BasicBlock *OrigBB = splitAt->getParent();

  // Create the a comparison of src and dst, based on which we jump to either
  // the forward-copy part of the function (if src >= dst) or the backwards-copy
  // part (if src < dst).
  // SplitBlockAndInsertIfThenElse conveniently creates the basic if-then-else
  // structure. Its block terminators (unconditional branches) are replaced by
  // the appropriate conditional branches when the loop is built.
  ICmpInst *PtrCompare = new ICmpInst(splitAt, ICmpInst::ICMP_ULT, srcAddr,
                                      dstAddr, "compare_src_dst");
  TerminatorInst *ThenTerm, *ElseTerm;
  SplitBlockAndInsertIfThenElse(PtrCompare, splitAt, &ThenTerm, &ElseTerm);

  // Each part of the function consists of two blocks:
  //   copy_backwards:        used to skip the loop when n == 0
  //   copy_backwards_loop:   the actual backwards loop BB
  //   copy_forward:          used to skip the loop when n == 0
  //   copy_forward_loop:     the actual forward loop BB
  BasicBlock *CopyBackwardsBB = ThenTerm->getParent();
  CopyBackwardsBB->setName("copy_backwards");
  BasicBlock *CopyForwardBB = ElseTerm->getParent();
  CopyForwardBB->setName("copy_forward");
  BasicBlock *ExitBB = splitAt->getParent();
  ExitBB->setName("memmove_done");

  // Initial comparison of n == 0 that lets us skip the loops altogether. Shared
  // between both backwards and forward copy clauses.
  ICmpInst *CompareN =
      new ICmpInst(OrigBB->getTerminator(), ICmpInst::ICMP_EQ, len,
                   ConstantInt::get(TypeOfLen, 0), "compare_n_to_0");

  // Copying backwards.
  BasicBlock *LoopBB =
      BasicBlock::Create(Context, "copy_backwards_loop", &F, CopyForwardBB);
  IRBuilder<> LoopBuilder(LoopBB);
  PHINode *LoopPhi = LoopBuilder.CreatePHI(TypeOfLen, 0);
  Value *IndexPtr = LoopBuilder.CreateSub(
      LoopPhi, ConstantInt::get(TypeOfLen, 1), "index_ptr");
  Value *Element = LoopBuilder.CreateLoad(
      LoopBuilder.CreateInBoundsGEP(srcAddr, IndexPtr), "element");
  LoopBuilder.CreateStore(Element,
                          LoopBuilder.CreateInBoundsGEP(dstAddr, IndexPtr));
  LoopBuilder.CreateCondBr(
      LoopBuilder.CreateICmpEQ(IndexPtr, ConstantInt::get(TypeOfLen, 0)),
      ExitBB, LoopBB);
  LoopPhi->addIncoming(IndexPtr, LoopBB);
  LoopPhi->addIncoming(len, CopyBackwardsBB);
  BranchInst::Create(ExitBB, LoopBB, CompareN, ThenTerm);
  ThenTerm->eraseFromParent();

  // Copying forward.
  BasicBlock *FwdLoopBB =
      BasicBlock::Create(Context, "copy_forward_loop", &F, ExitBB);
  IRBuilder<> FwdLoopBuilder(FwdLoopBB);
  PHINode *FwdCopyPhi = FwdLoopBuilder.CreatePHI(TypeOfLen, 0, "index_ptr");
  Value *FwdElement = FwdLoopBuilder.CreateLoad(
      FwdLoopBuilder.CreateInBoundsGEP(srcAddr, FwdCopyPhi), "element");
  FwdLoopBuilder.CreateStore(
      FwdElement, FwdLoopBuilder.CreateInBoundsGEP(dstAddr, FwdCopyPhi));
  Value *FwdIndexPtr = FwdLoopBuilder.CreateAdd(
      FwdCopyPhi, ConstantInt::get(TypeOfLen, 1), "index_increment");
  FwdLoopBuilder.CreateCondBr(FwdLoopBuilder.CreateICmpEQ(FwdIndexPtr, len),
                              ExitBB, FwdLoopBB);
  FwdCopyPhi->addIncoming(FwdIndexPtr, FwdLoopBB);
  FwdCopyPhi->addIncoming(ConstantInt::get(TypeOfLen, 0), CopyForwardBB);

  BranchInst::Create(ExitBB, FwdLoopBB, CompareN, ElseTerm);
  ElseTerm->eraseFromParent();
}

// Lower memset to loop.
void convertMemSetToLoop(Instruction *splitAt, Value *dstAddr, Value *len,
                         Value *val, LLVMContext &Context, Function &F) {
  BasicBlock *origBB = splitAt->getParent();
  BasicBlock *newBB = splitAt->getParent()->splitBasicBlock(splitAt, "split");
  BasicBlock *loopBB = BasicBlock::Create(Context, "loadstoreloop", &F, newBB);

  origBB->getTerminator()->setSuccessor(0, loopBB);
  IRBuilder<> builder(origBB, origBB->getTerminator());

  unsigned dstAS = cast<PointerType>(dstAddr->getType())->getAddressSpace();

  // Cast pointer to the type of value getting stored
  dstAddr =
      builder.CreateBitCast(dstAddr, PointerType::get(val->getType(), dstAS));

  IRBuilder<> loop(loopBB);
  PHINode *ind = loop.CreatePHI(len->getType(), 0);
  ind->addIncoming(ConstantInt::get(len->getType(), 0), origBB);

  loop.CreateStore(val, loop.CreateGEP(val->getType(), dstAddr, ind), false);

  Value *newind = loop.CreateAdd(ind, ConstantInt::get(len->getType(), 1));
  ind->addIncoming(newind, loopBB);

  loop.CreateCondBr(loop.CreateICmpULT(newind, len), loopBB, newBB);
}

bool NVPTXLowerAggrCopies::runOnFunction(Function &F) {
  SmallVector<LoadInst *, 4> aggrLoads;
  SmallVector<MemIntrinsic *, 4> MemCalls;

  const DataLayout &DL = F.getParent()->getDataLayout();
  LLVMContext &Context = F.getParent()->getContext();

  // Collect all aggregate loads and mem* calls.
  for (Function::iterator BI = F.begin(), BE = F.end(); BI != BE; ++BI) {
    for (BasicBlock::iterator II = BI->begin(), IE = BI->end(); II != IE;
         ++II) {
      if (LoadInst *load = dyn_cast<LoadInst>(II)) {
        if (!load->hasOneUse())
          continue;

        if (DL.getTypeStoreSize(load->getType()) < MaxAggrCopySize)
          continue;

        User *use = load->user_back();
        if (StoreInst *store = dyn_cast<StoreInst>(use)) {
          if (store->getOperand(0) != load)
            continue;
          aggrLoads.push_back(load);
        }
      } else if (MemIntrinsic *IntrCall = dyn_cast<MemIntrinsic>(II)) {
        // Convert intrinsic calls with variable size or with constant size
        // larger than the MaxAggrCopySize threshold.
        if (ConstantInt *LenCI = dyn_cast<ConstantInt>(IntrCall->getLength())) {
          if (LenCI->getZExtValue() >= MaxAggrCopySize) {
            MemCalls.push_back(IntrCall);
          }
        } else {
          MemCalls.push_back(IntrCall);
        }
      }
    }
  }

  if (aggrLoads.size() == 0 && MemCalls.size() == 0) {
    return false;
  }

  //
  // Do the transformation of an aggr load/copy/set to a loop
  //
  for (LoadInst *load : aggrLoads) {
    StoreInst *store = dyn_cast<StoreInst>(*load->user_begin());
    Value *srcAddr = load->getOperand(0);
    Value *dstAddr = store->getOperand(1);
    unsigned numLoads = DL.getTypeStoreSize(load->getType());
    Value *len = ConstantInt::get(Type::getInt32Ty(Context), numLoads);

    convertMemCpyToLoop(store, srcAddr, dstAddr, len, load->isVolatile(),
                        store->isVolatile(), Context, F);

    store->eraseFromParent();
    load->eraseFromParent();
  }

  // Transform mem* intrinsic calls.
  for (MemIntrinsic *MemCall : MemCalls) {
    if (MemCpyInst *Memcpy = dyn_cast<MemCpyInst>(MemCall)) {
      convertMemCpyToLoop(/* splitAt */ Memcpy,
                          /* srcAddr */ Memcpy->getRawSource(),
                          /* dstAddr */ Memcpy->getRawDest(),
                          /* len */ Memcpy->getLength(),
                          /* srcVolatile */ Memcpy->isVolatile(),
                          /* dstVolatile */ Memcpy->isVolatile(),
                          /* Context */ Context,
                          /* Function F */ F);
    } else if (MemMoveInst *Memmove = dyn_cast<MemMoveInst>(MemCall)) {
      convertMemMoveToLoop(/* splitAt */ Memmove,
                           /* srcAddr */ Memmove->getRawSource(),
                           /* dstAddr */ Memmove->getRawDest(),
                           /* len */ Memmove->getLength(),
                           /* srcVolatile */ Memmove->isVolatile(),
                           /* dstVolatile */ Memmove->isVolatile(),
                           /* Context */ Context,
                           /* Function F */ F);

    } else if (MemSetInst *Memset = dyn_cast<MemSetInst>(MemCall)) {
      convertMemSetToLoop(/* splitAt */ Memset,
                          /* dstAddr */ Memset->getRawDest(),
                          /* len */ Memset->getLength(),
                          /* val */ Memset->getValue(),
                          /* Context */ Context,
                          /* F */ F);
    }
    MemCall->eraseFromParent();
  }

  return true;
}

} // namespace

namespace llvm {
void initializeNVPTXLowerAggrCopiesPass(PassRegistry &);
}

INITIALIZE_PASS(NVPTXLowerAggrCopies, "nvptx-lower-aggr-copies",
                "Lower aggregate copies, and llvm.mem* intrinsics into loops",
                false, false)

FunctionPass *llvm::createLowerAggrCopies() {
  return new NVPTXLowerAggrCopies();
}
