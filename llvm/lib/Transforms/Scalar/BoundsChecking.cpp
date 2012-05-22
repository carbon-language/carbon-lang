//===- BoundsChecking.cpp - Instrumentation for run-time bounds checking --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that instruments the code to perform run-time
// bounds checking on loads, stores, and other memory intrinsics.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "bounds-checking"
#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/TargetFolder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Operator.h"
#include "llvm/Pass.h"
using namespace llvm;

STATISTIC(ChecksAdded, "Bounds checks added");
STATISTIC(ChecksSkipped, "Bounds checks skipped");
STATISTIC(ChecksUnable, "Bounds checks unable to add");

typedef IRBuilder<true, TargetFolder> BuilderTy;

namespace {
  enum ConstTriState {
    NotConst, Const, Dunno
  };

  struct BoundsChecking : public FunctionPass {
    const TargetData *TD;
    BuilderTy *Builder;
    Function *Fn;
    BasicBlock *TrapBB;
    unsigned Penalty;
    static char ID;

    BoundsChecking(unsigned _Penalty = 5) : FunctionPass(ID), Penalty(_Penalty){
      initializeBoundsCheckingPass(*PassRegistry::getPassRegistry());
    }

    BasicBlock *getTrapBB();
    ConstTriState computeAllocSize(Value *Alloc, uint64_t &Size, Value* &SizeValue);
    bool instrument(Value *Ptr, Value *Val);

    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();
    }
 };
}

char BoundsChecking::ID = 0;
INITIALIZE_PASS(BoundsChecking, "bounds-checking", "Run-time bounds checking",
                false, false)


/// getTrapBB - create a basic block that traps. All overflowing conditions
/// branch to this block. There's only one trap block per function.
BasicBlock *BoundsChecking::getTrapBB() {
  if (TrapBB)
    return TrapBB;

  BasicBlock::iterator PrevInsertPoint = Builder->GetInsertPoint();
  TrapBB = BasicBlock::Create(Fn->getContext(), "trap", Fn);
  Builder->SetInsertPoint(TrapBB);

  llvm::Value *F = Intrinsic::getDeclaration(Fn->getParent(), Intrinsic::trap);
  CallInst *TrapCall = Builder->CreateCall(F);
  TrapCall->setDoesNotReturn();
  TrapCall->setDoesNotThrow();
  Builder->CreateUnreachable();

  Builder->SetInsertPoint(PrevInsertPoint);
  return TrapBB;
}


/// computeAllocSize - compute the object size allocated by an allocation
/// site. Returns NotConst if the size is not constant (in SizeValue), Const if
/// the size is constant (in Size), and Dunno if the size could not be
/// determined within the given maximum Penalty that the computation would
/// incurr at run-time.
ConstTriState BoundsChecking::computeAllocSize(Value *Alloc, uint64_t &Size,
                                     Value* &SizeValue) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Alloc)) {
    if (GV->hasDefinitiveInitializer()) {
      Constant *C = GV->getInitializer();
      Size = TD->getTypeAllocSize(C->getType());
      return Const;
    }
    return Dunno;

  } else if (AllocaInst *AI = dyn_cast<AllocaInst>(Alloc)) {
    if (!AI->getAllocatedType()->isSized())
      return Dunno;

    Size = TD->getTypeAllocSize(AI->getAllocatedType());
    if (!AI->isArrayAllocation())
      return Const; // we are done

    Value *ArraySize = AI->getArraySize();
    if (const ConstantInt *C = dyn_cast<ConstantInt>(ArraySize)) {
      Size *= C->getZExtValue();
      return Const;
    }

    if (Penalty < 2)
      return Dunno;

    SizeValue = ConstantInt::get(ArraySize->getType(), Size);
    SizeValue = Builder->CreateMul(SizeValue, ArraySize);
    return NotConst;

  } else if (CallInst *MI = extractMallocCall(Alloc)) {
    SizeValue = MI->getArgOperand(0);
    if (ConstantInt *CI = dyn_cast<ConstantInt>(SizeValue)) {
      Size = CI->getZExtValue();
      return Const;
    }
    return Penalty >= 2 ? NotConst : Dunno;

  } else if (CallInst *MI = extractCallocCall(Alloc)) {
    Value *Arg1 = MI->getArgOperand(0);
    Value *Arg2 = MI->getArgOperand(1);
    if (ConstantInt *CI1 = dyn_cast<ConstantInt>(Arg1)) {
      if (ConstantInt *CI2 = dyn_cast<ConstantInt>(Arg2)) {
        Size = (CI1->getValue() * CI2->getValue()).getZExtValue();
        return Const;
      }
    }

    if (Penalty < 2)
      return Dunno;

    SizeValue = Builder->CreateMul(Arg1, Arg2);
    return NotConst;
  }

  DEBUG(dbgs() << "computeAllocSize failed:\n" << *Alloc);
  return Dunno;
}


bool BoundsChecking::instrument(Value *Ptr, Value *InstVal) {
  uint64_t NeededSize = TD->getTypeStoreSize(InstVal->getType());
  DEBUG(dbgs() << "Instrument " << *Ptr << " for " << Twine(NeededSize)
              << " bytes\n");

  Type *SizeTy = Type::getInt64Ty(Fn->getContext());

  // Get to the real allocated thing and offset as fast as possible.
  Ptr = Ptr->stripPointerCasts();
  GEPOperator *GEP;

  if ((GEP = dyn_cast<GEPOperator>(Ptr))) {
    // check if we will be able to get the offset
    if (!GEP->hasAllConstantIndices() && Penalty < 2) {
      ++ChecksUnable;
      return false;
    }
    Ptr = GEP->getPointerOperand()->stripPointerCasts();
  }

  uint64_t Size = 0;
  Value *SizeValue = 0;
  ConstTriState ConstAlloc = computeAllocSize(Ptr, Size, SizeValue);
  if (ConstAlloc == Dunno) {
    ++ChecksUnable;
    return false;
  }
  assert(ConstAlloc == Const || SizeValue);

  uint64_t Offset = 0;
  Value *OffsetValue = 0;

  if (GEP) {
    if (GEP->hasAllConstantIndices()) {
      SmallVector<Value*, 8> Ops(GEP->idx_begin(), GEP->idx_end());
      assert(GEP->getPointerOperandType()->isPointerTy());
      Offset = TD->getIndexedOffset(GEP->getPointerOperandType(), Ops);
    } else {
      OffsetValue = EmitGEPOffset(Builder, *TD, GEP);
    }
  }

  if (!OffsetValue && ConstAlloc == Const) {
    if (Size < Offset || (Size - Offset) < NeededSize) {
      // Out of bounds
      Builder->CreateBr(getTrapBB());
      ++ChecksAdded;
      return true;
    }
    // in bounds
    ++ChecksSkipped;
    return false;
  }

  if (OffsetValue)
    OffsetValue = Builder->CreateZExt(OffsetValue, SizeTy);
  else
    OffsetValue = ConstantInt::get(SizeTy, Offset);

  if (SizeValue)
    SizeValue = Builder->CreateZExt(SizeValue, SizeTy);
  else
    SizeValue = ConstantInt::get(SizeTy, Size);

  Value *NeededSizeVal = ConstantInt::get(SizeTy, NeededSize);
  Value *ObjSize = Builder->CreateSub(SizeValue, OffsetValue);
  Value *Cmp1 = Builder->CreateICmpULT(SizeValue, OffsetValue);
  Value *Cmp2 = Builder->CreateICmpULT(ObjSize, NeededSizeVal);
  Value *Or = Builder->CreateOr(Cmp1, Cmp2);

  // FIXME: add unlikely branch taken metadata?
  Instruction *Inst = Builder->GetInsertPoint();
  BasicBlock *OldBB = Inst->getParent();
  BasicBlock *Cont = OldBB->splitBasicBlock(Inst);
  OldBB->getTerminator()->eraseFromParent();
  BranchInst::Create(getTrapBB(), Cont, Or, OldBB);
  ++ChecksAdded;
  return true;
}

bool BoundsChecking::runOnFunction(Function &F) {
  TD = &getAnalysis<TargetData>();

  TrapBB = 0;
  Fn = &F;
  BuilderTy TheBuilder(F.getContext(), TargetFolder(TD));
  Builder = &TheBuilder;

  // check HANDLE_MEMORY_INST in include/llvm/Instruction.def for memory
  // touching instructions
  std::vector<Instruction*> WorkList;
  for (inst_iterator i = inst_begin(F), e = inst_end(F); i != e; ++i) {
    Instruction *I = &*i;
    if (isa<LoadInst>(I) || isa<StoreInst>(I) || isa<AtomicCmpXchgInst>(I) ||
        isa<AtomicRMWInst>(I))
        WorkList.push_back(I);
  }

  bool MadeChange = false;
  while (!WorkList.empty()) {
    Instruction *I = WorkList.back();
    WorkList.pop_back();

    Builder->SetInsertPoint(I);
    if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      MadeChange |= instrument(LI->getPointerOperand(), LI);
    } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      MadeChange |= instrument(SI->getPointerOperand(), SI->getValueOperand());
    } else if (AtomicCmpXchgInst *AI = dyn_cast<AtomicCmpXchgInst>(I)) {
      MadeChange |= instrument(AI->getPointerOperand(),AI->getCompareOperand());
    } else if (AtomicRMWInst *AI = dyn_cast<AtomicRMWInst>(I)) {
      MadeChange |= instrument(AI->getPointerOperand(), AI->getValOperand());
    } else {
      llvm_unreachable("unknown Instruction type");
    }
  }
  return MadeChange;
}

FunctionPass *llvm::createBoundsCheckingPass(unsigned Penalty) {
  return new BoundsChecking(Penalty);
}
