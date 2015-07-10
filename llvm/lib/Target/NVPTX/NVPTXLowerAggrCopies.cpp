//===- NVPTXLowerAggrCopies.cpp - ------------------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
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
} // namespace

char NVPTXLowerAggrCopies::ID = 0;

// Lower MemTransferInst or load-store pair to loop
static void convertTransferToLoop(
    Instruction *splitAt, Value *srcAddr, Value *dstAddr, Value *len,
    bool srcVolatile, bool dstVolatile, LLVMContext &Context, Function &F) {
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

// Lower MemSetInst to loop
static void convertMemSetToLoop(Instruction *splitAt, Value *dstAddr,
                                Value *len, Value *val, LLVMContext &Context,
                                Function &F) {
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
  SmallVector<MemTransferInst *, 4> aggrMemcpys;
  SmallVector<MemSetInst *, 4> aggrMemsets;

  const DataLayout &DL = F.getParent()->getDataLayout();
  LLVMContext &Context = F.getParent()->getContext();

  //
  // Collect all the aggrLoads, aggrMemcpys and addrMemsets.
  //
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
      } else if (MemTransferInst *intr = dyn_cast<MemTransferInst>(II)) {
        Value *len = intr->getLength();
        // If the number of elements being copied is greater
        // than MaxAggrCopySize, lower it to a loop
        if (ConstantInt *len_int = dyn_cast<ConstantInt>(len)) {
          if (len_int->getZExtValue() >= MaxAggrCopySize) {
            aggrMemcpys.push_back(intr);
          }
        } else {
          // turn variable length memcpy/memmov into loop
          aggrMemcpys.push_back(intr);
        }
      } else if (MemSetInst *memsetintr = dyn_cast<MemSetInst>(II)) {
        Value *len = memsetintr->getLength();
        if (ConstantInt *len_int = dyn_cast<ConstantInt>(len)) {
          if (len_int->getZExtValue() >= MaxAggrCopySize) {
            aggrMemsets.push_back(memsetintr);
          }
        } else {
          // turn variable length memset into loop
          aggrMemsets.push_back(memsetintr);
        }
      }
    }
  }
  if ((aggrLoads.size() == 0) && (aggrMemcpys.size() == 0) &&
      (aggrMemsets.size() == 0))
    return false;

  //
  // Do the transformation of an aggr load/copy/set to a loop
  //
  for (LoadInst *load : aggrLoads) {
    StoreInst *store = dyn_cast<StoreInst>(*load->user_begin());
    Value *srcAddr = load->getOperand(0);
    Value *dstAddr = store->getOperand(1);
    unsigned numLoads = DL.getTypeStoreSize(load->getType());
    Value *len = ConstantInt::get(Type::getInt32Ty(Context), numLoads);

    convertTransferToLoop(store, srcAddr, dstAddr, len, load->isVolatile(),
                          store->isVolatile(), Context, F);

    store->eraseFromParent();
    load->eraseFromParent();
  }

  for (MemTransferInst *cpy : aggrMemcpys) {
    convertTransferToLoop(/* splitAt */ cpy,
                          /* srcAddr */ cpy->getSource(),
                          /* dstAddr */ cpy->getDest(),
                          /* len */ cpy->getLength(),
                          /* srcVolatile */ cpy->isVolatile(),
                          /* dstVolatile */ cpy->isVolatile(),
                          /* Context */ Context,
                          /* Function F */ F);
    cpy->eraseFromParent();
  }

  for (MemSetInst *memsetinst : aggrMemsets) {
    Value *len = memsetinst->getLength();
    Value *val = memsetinst->getValue();
    convertMemSetToLoop(memsetinst, memsetinst->getDest(), len, val, Context,
                        F);
    memsetinst->eraseFromParent();
  }

  return true;
}

FunctionPass *llvm::createLowerAggrCopies() {
  return new NVPTXLowerAggrCopies();
}
