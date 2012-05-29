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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetFolder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Metadata.h"
#include "llvm/Operator.h"
#include "llvm/Pass.h"
using namespace llvm;

STATISTIC(ChecksAdded, "Bounds checks added");
STATISTIC(ChecksSkipped, "Bounds checks skipped");
STATISTIC(ChecksUnable, "Bounds checks unable to add");
STATISTIC(ChecksUnableInterproc, "Bounds checks unable to add (interprocedural)");
STATISTIC(ChecksUnableLoad, "Bounds checks unable to add (LoadInst)");

typedef IRBuilder<true, TargetFolder> BuilderTy;

namespace {
  enum ConstTriState {
    NotConst, Const, Dunno
  };

  struct BoundsChecking : public FunctionPass {
    static char ID;

    BoundsChecking(unsigned _Penalty = 5) : FunctionPass(ID), Penalty(_Penalty){
      initializeBoundsCheckingPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();
      AU.addRequired<LoopInfo>();
      AU.addRequired<ScalarEvolution>();
    }

  private:
    const TargetData *TD;
    LoopInfo *LI;
    ScalarEvolution *SE;
    BuilderTy *Builder;
    Function *Fn;
    BasicBlock *TrapBB;
    unsigned Penalty;

    BasicBlock *getTrapBB();
    void emitBranchToTrap(Value *Cmp = 0);
    ConstTriState computeAllocSize(Value *Alloc, uint64_t &Size,
                                   Value* &SizeValue);
    bool instrument(Value *Ptr, Value *Val);
 };
}

char BoundsChecking::ID = 0;
INITIALIZE_PASS_BEGIN(BoundsChecking, "bounds-checking",
                      "Run-time bounds checking", false, false)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_END(BoundsChecking, "bounds-checking",
                      "Run-time bounds checking", false, false)


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


/// emitBranchToTrap - emit a branch instruction to a trap block.
/// If Cmp is non-null, perform a jump only if its value evaluates to true.
void BoundsChecking::emitBranchToTrap(Value *Cmp) {
  Instruction *Inst = Builder->GetInsertPoint();
  BasicBlock *OldBB = Inst->getParent();
  BasicBlock *Cont = OldBB->splitBasicBlock(Inst);
  OldBB->getTerminator()->eraseFromParent();

  if (Cmp)
    BranchInst::Create(getTrapBB(), Cont, Cmp, OldBB);
  else
    BranchInst::Create(getTrapBB(), OldBB);
}


/// computeAllocSize - compute the object size allocated by an allocation
/// site. Returns NotConst if the size is not constant (in SizeValue), Const if
/// the size is constant (in Size), and Dunno if the size could not be
/// determined within the given maximum Penalty that the computation would
/// incurr at run-time.
ConstTriState BoundsChecking::computeAllocSize(Value *Alloc, uint64_t &Size,
                                     Value* &SizeValue) {
  IntegerType *RetTy = TD->getIntPtrType(Fn->getContext());

  // global variable with definitive size
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Alloc)) {
    if (GV->hasDefinitiveInitializer()) {
      Constant *C = GV->getInitializer();
      Size = TD->getTypeAllocSize(C->getType());
      return Const;
    }
    return Dunno;

  // stack allocation
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

  // function arguments
  } else if (Argument *A = dyn_cast<Argument>(Alloc)) {
    if (!A->hasByValAttr()) {
      ++ChecksUnableInterproc;
      return Dunno;
    }

    PointerType *PT = cast<PointerType>(A->getType());
    Size = TD->getTypeAllocSize(PT->getElementType());
    return Const;

  // ptr = select(ptr1, ptr2)
  } else if (SelectInst *SI = dyn_cast<SelectInst>(Alloc)) {
    uint64_t SizeFalse;
    Value *SizeValueFalse;
    ConstTriState TrueConst = computeAllocSize(SI->getTrueValue(), Size,
                                               SizeValue);
    ConstTriState FalseConst = computeAllocSize(SI->getFalseValue(), SizeFalse,
                                                SizeValueFalse);

    if (TrueConst == Const && FalseConst == Const && Size == SizeFalse)
      return Const;

    if (Penalty < 2 || (TrueConst == Dunno && FalseConst == Dunno))
      return Dunno;

    // if one of the branches is Dunno, assume it is ok and check just the other
    APInt MaxSize = APInt::getMaxValue(TD->getTypeSizeInBits(RetTy));

    if (TrueConst == Const)
      SizeValue = ConstantInt::get(RetTy, Size);
    else if (TrueConst == Dunno)
      SizeValue = ConstantInt::get(RetTy, MaxSize);

    if (FalseConst == Const)
      SizeValueFalse = ConstantInt::get(RetTy, SizeFalse);
    else if (FalseConst == Dunno)
      SizeValueFalse = ConstantInt::get(RetTy, MaxSize);

    SizeValue = Builder->CreateSelect(SI->getCondition(), SizeValue,
                                      SizeValueFalse);
    return NotConst;

  // call allocation function
  } else if (CallInst *CI = dyn_cast<CallInst>(Alloc)) {
    SmallVector<unsigned, 4> Args;

    if (MDNode *MD = CI->getMetadata("alloc_size")) {
      for (unsigned i = 0, e = MD->getNumOperands(); i != e; ++i)
        Args.push_back(cast<ConstantInt>(MD->getOperand(i))->getZExtValue());

    } else if (Function *Callee = CI->getCalledFunction()) {
      FunctionType *FTy = Callee->getFunctionType();

      // alloc(size)
      if (FTy->getNumParams() == 1 && FTy->getParamType(0)->isIntegerTy()) {
        if ((Callee->getName() == "malloc" ||
             Callee->getName() == "valloc" ||
             Callee->getName() == "_Znwj"  || // operator new(unsigned int)
             Callee->getName() == "_Znwm"  || // operator new(unsigned long)
             Callee->getName() == "_Znaj"  || // operator new[](unsigned int)
             Callee->getName() == "_Znam")) {
          Args.push_back(0);
        }
      } else if (FTy->getNumParams() == 2) {
        // alloc(_, x)
        if (FTy->getParamType(1)->isIntegerTy() &&
            ((Callee->getName() == "realloc" ||
              Callee->getName() == "reallocf"))) {
          Args.push_back(1);

        // alloc(x, y)
        } else if (FTy->getParamType(0)->isIntegerTy() &&
                   FTy->getParamType(1)->isIntegerTy() &&
                   Callee->getName() == "calloc") {
          Args.push_back(0);
          Args.push_back(1);
        }
      }
    }

    if (Args.empty())
      return Dunno;

    // check if all arguments are constant. if so, the object size is also const
    bool AllConst = true;
    for (SmallVectorImpl<unsigned>::iterator I = Args.begin(), E = Args.end();
         I != E; ++I) {
      if (!isa<ConstantInt>(CI->getArgOperand(*I))) {
        AllConst = false;
        break;
      }
    }

    if (AllConst) {
      Size = 1;
      for (SmallVectorImpl<unsigned>::iterator I = Args.begin(), E = Args.end();
           I != E; ++I) {
        ConstantInt *Arg = cast<ConstantInt>(CI->getArgOperand(*I));
        Size *= (size_t)Arg->getZExtValue();
      }
      return Const;
    }

    if (Penalty < 2)
      return Dunno;

    // not all arguments are constant, so create a sequence of multiplications
    bool First = true;
    for (SmallVectorImpl<unsigned>::iterator I = Args.begin(), E = Args.end();
         I != E; ++I) {
      Value *Arg = CI->getArgOperand(*I);
      if (First) {
        SizeValue = Arg;
        First = false;
        continue;
      }
      SizeValue = Builder->CreateMul(SizeValue, Arg);
    }
    return NotConst;

    // TODO: handle more standard functions:
    // - strdup / strndup
    // - strcpy / strncpy
    // - memcpy / memmove
    // - strcat / strncat

  } else if (isa<LoadInst>(Alloc)) {
    ++ChecksUnableLoad;
    return Dunno;
  }

  return Dunno;
}


/// instrument - adds run-time bounds checks to memory accessing instructions.
/// Ptr is the pointer that will be read/written, and InstVal is either the
/// result from the load or the value being stored. It is used to determine the
/// size of memory block that is touched.
/// Returns true if any change was made to the IR, false otherwise.
bool BoundsChecking::instrument(Value *Ptr, Value *InstVal) {
  uint64_t NeededSize = TD->getTypeStoreSize(InstVal->getType());
  DEBUG(dbgs() << "Instrument " << *Ptr << " for " << Twine(NeededSize)
              << " bytes\n");

  Type *SizeTy = TD->getIntPtrType(Fn->getContext());

  // Get to the real allocated thing and offset as fast as possible.
  Ptr = Ptr->stripPointerCasts();

  // try to hoist the check if the instruction is inside a loop
  Value *LoopOffset = 0;
  if (Loop *L = LI->getLoopFor(Builder->GetInsertPoint()->getParent())) {
    const SCEV *PtrSCEV  = SE->getSCEVAtScope(Ptr, L->getParentLoop());
    const SCEV *BaseSCEV = SE->getPointerBase(PtrSCEV);

    if (const SCEVUnknown *PointerBase = dyn_cast<SCEVUnknown>(BaseSCEV)) {
      Ptr = PointerBase->getValue()->stripPointerCasts();
      Instruction *InsertPoint = L->getLoopPreheader()->getFirstInsertionPt();
      Builder->SetInsertPoint(InsertPoint);

      SCEVExpander Expander(*SE, "bounds-checking");
      const SCEV *OffsetSCEV = SE->getMinusSCEV(PtrSCEV, PointerBase);
      LoopOffset = Expander.expandCodeFor(OffsetSCEV, SizeTy, InsertPoint);
    }
  }

  GEPOperator *GEP = dyn_cast<GEPOperator>(Ptr);
  if (GEP) {
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
    DEBUG(dbgs() << "computeAllocSize failed:\n" << *Ptr << "\n");
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

  if (!LoopOffset && !OffsetValue && ConstAlloc == Const) {
    if (Size < Offset || (Size - Offset) < NeededSize) {
      // Out of bounds
      emitBranchToTrap();
      ++ChecksAdded;
      return true;
    }
    // in bounds
    ++ChecksSkipped;
    return false;
  }

  if (!OffsetValue)
    OffsetValue = ConstantInt::get(SizeTy, Offset);

  if (SizeValue)
    SizeValue = Builder->CreateZExt(SizeValue, SizeTy);
  else
    SizeValue = ConstantInt::get(SizeTy, Size);

  // add the loop offset if the check was hoisted
  if (LoopOffset)
    OffsetValue = Builder->CreateAdd(OffsetValue, LoopOffset);

  Value *NeededSizeVal = ConstantInt::get(SizeTy, NeededSize);
  Value *ObjSize = Builder->CreateSub(SizeValue, OffsetValue);
  Value *Cmp1 = Builder->CreateICmpULT(SizeValue, OffsetValue);
  Value *Cmp2 = Builder->CreateICmpULT(ObjSize, NeededSizeVal);
  Value *Or = Builder->CreateOr(Cmp1, Cmp2);
  emitBranchToTrap(Or);

  ++ChecksAdded;
  return true;
}

bool BoundsChecking::runOnFunction(Function &F) {
  TD = &getAnalysis<TargetData>();
  LI = &getAnalysis<LoopInfo>();
  SE = &getAnalysis<ScalarEvolution>();

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
  for (std::vector<Instruction*>::iterator i = WorkList.begin(),
       e = WorkList.end(); i != e; ++i) {
    Instruction *I = *i;

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
