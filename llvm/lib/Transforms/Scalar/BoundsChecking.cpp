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
#include "llvm/ADT/DenseMap.h"
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
  // FIXME: can use unions here to save space
  struct CacheData {
    APInt Offset;
    Value *OffsetValue;
    APInt Size;
    Value *SizeValue;
    bool ReturnVal;
  };
  typedef DenseMap<Value*, CacheData> CacheMapTy;

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
    CacheMapTy CacheMap;

    BasicBlock *getTrapBB();
    void emitBranchToTrap(Value *Cmp = 0);
    bool computeAllocSize(Value *Ptr, APInt &Offset, Value* &OffsetValue,
                          APInt &Size, Value* &SizeValue);
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


#define GET_VALUE(Val, Int) \
  if (!Val) \
    Val = ConstantInt::get(IntTy, Int)

#define RETURN(Val) \
  do { ReturnVal = Val; goto cache_and_return; } while (0)

/// computeAllocSize - compute the object size and the offset within the object
/// pointed by Ptr. OffsetValue/SizeValue will be null if they are constant, and
/// therefore the result is given in Offset/Size variables instead.
/// Returns true if the offset and size could be computed within the given
/// maximum run-time penalty.
bool BoundsChecking::computeAllocSize(Value *Ptr, APInt &Offset,
                                      Value* &OffsetValue, APInt &Size,
                                      Value* &SizeValue) {
  Ptr = Ptr->stripPointerCasts();

  // lookup to see if we've seen the Ptr before
  CacheMapTy::iterator CacheIt = CacheMap.find(Ptr);
  if (CacheIt != CacheMap.end()) {
    CacheData &Cache = CacheIt->second;
    Offset = Cache.Offset;
    OffsetValue = Cache.OffsetValue;
    Size = Cache.Size;
    SizeValue = Cache.SizeValue;
    return Cache.ReturnVal;
  }

  IntegerType *IntTy = TD->getIntPtrType(Fn->getContext());
  unsigned IntTyBits = IntTy->getBitWidth();
  bool ReturnVal;

  // always generate code immediately before the instruction being processed, so
  // that the generated code dominates the same BBs
  Instruction *PrevInsertPoint = Builder->GetInsertPoint();
  if (Instruction *I = dyn_cast<Instruction>(Ptr))
    Builder->SetInsertPoint(I);

  // initalize with "don't know" state: offset=0 and size=uintmax
  Offset = 0;
  Size = APInt::getMaxValue(TD->getTypeSizeInBits(IntTy));
  OffsetValue = SizeValue = 0;

  if (GEPOperator *GEP = dyn_cast<GEPOperator>(Ptr)) {
    APInt PtrOffset(IntTyBits, 0);
    Value *PtrOffsetValue = 0;
    if (!computeAllocSize(GEP->getPointerOperand(), PtrOffset, PtrOffsetValue,
                          Size, SizeValue))
      RETURN(false);

    if (GEP->hasAllConstantIndices()) {
      SmallVector<Value*, 8> Ops(GEP->idx_begin(), GEP->idx_end());
      Offset = TD->getIndexedOffset(GEP->getPointerOperandType(), Ops);
      // if PtrOffset is constant, return immediately
      if (!PtrOffsetValue) {
        Offset += PtrOffset;
        RETURN(true);
      }
      OffsetValue = ConstantInt::get(IntTy, Offset);
    } else {
      OffsetValue = EmitGEPOffset(Builder, *TD, GEP);
    }

    GET_VALUE(PtrOffsetValue, PtrOffset);
    OffsetValue = Builder->CreateAdd(PtrOffsetValue, OffsetValue);
    RETURN(true);

  // global variable with definitive size
  } else if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Ptr)) {
    if (GV->hasDefinitiveInitializer()) {
      Constant *C = GV->getInitializer();
      Size = TD->getTypeAllocSize(C->getType());
      RETURN(true);
    }
    RETURN(false);

  // stack allocation
  } else if (AllocaInst *AI = dyn_cast<AllocaInst>(Ptr)) {
    if (!AI->getAllocatedType()->isSized())
      RETURN(false);

    Size = TD->getTypeAllocSize(AI->getAllocatedType());
    if (!AI->isArrayAllocation())
      RETURN(true); // we are done

    Value *ArraySize = AI->getArraySize();
    if (const ConstantInt *C = dyn_cast<ConstantInt>(ArraySize)) {
      Size *= C->getValue();
      RETURN(true);
    }

    if (Penalty < 2)
      RETURN(false);

    // VLA: compute size dynamically
    SizeValue = ConstantInt::get(ArraySize->getType(), Size);
    SizeValue = Builder->CreateMul(SizeValue, ArraySize);
    RETURN(true);

  // function arguments
  } else if (Argument *A = dyn_cast<Argument>(Ptr)) {
    // right now we only support byval arguments, so that no interprocedural
    // analysis is necessary
    if (!A->hasByValAttr()) {
      ++ChecksUnableInterproc;
      RETURN(false);
    }

    PointerType *PT = cast<PointerType>(A->getType());
    Size = TD->getTypeAllocSize(PT->getElementType());
    RETURN(true);

  // ptr = select(ptr1, ptr2)
  } else if (SelectInst *SI = dyn_cast<SelectInst>(Ptr)) {
    APInt OffsetTrue(IntTyBits, 0), OffsetFalse(IntTyBits, 0);
    APInt SizeTrue(IntTyBits, 0), SizeFalse(IntTyBits, 0);
    Value *OffsetValueTrue = 0, *OffsetValueFalse = 0;
    Value *SizeValueTrue = 0, *SizeValueFalse = 0;

    bool TrueAlloc = computeAllocSize(SI->getTrueValue(), OffsetTrue,
                                      OffsetValueTrue, SizeTrue, SizeValueTrue);
    bool FalseAlloc = computeAllocSize(SI->getFalseValue(), OffsetFalse,
                                       OffsetValueFalse, SizeFalse,
                                       SizeValueFalse);
    if (!TrueAlloc && !FalseAlloc)
      RETURN(false);

    // fold constant sizes & offsets if they are equal
    if (!OffsetValueTrue && !OffsetValueFalse && OffsetTrue == OffsetFalse)
      Offset = OffsetTrue;
    else if (Penalty > 1) {
      GET_VALUE(OffsetValueTrue, OffsetTrue);
      GET_VALUE(OffsetValueFalse, OffsetFalse);
      OffsetValue = Builder->CreateSelect(SI->getCondition(), OffsetValueTrue,
                                          OffsetValueFalse);
    } else
      RETURN(false);

    if (!SizeValueTrue && !SizeValueFalse && SizeTrue == SizeFalse)
      Size = SizeTrue;
    else if (Penalty > 1) {
      GET_VALUE(SizeValueTrue, SizeTrue);
      GET_VALUE(SizeValueFalse, SizeFalse);
      SizeValue = Builder->CreateSelect(SI->getCondition(), SizeValueTrue,
                                        SizeValueFalse);
    } else
      RETURN(false);
    RETURN(true);

  // call allocation function
  } else if (CallInst *CI = dyn_cast<CallInst>(Ptr)) {
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
      } else if (FTy->getNumParams() == 3) {
        // alloc(_, _, x)
        if (FTy->getParamType(2)->isIntegerTy() &&
            Callee->getName() == "posix_memalign") {
          Args.push_back(2);
        }
      }
    }

    if (Args.empty())
      RETURN(false);

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
        Size *= Arg->getValue().zextOrSelf(IntTyBits);
      }
      RETURN(true);
    }

    if (Penalty < 2)
      RETURN(false);

    // not all arguments are constant, so create a sequence of multiplications
    for (SmallVectorImpl<unsigned>::iterator I = Args.begin(), E = Args.end();
         I != E; ++I) {
      Value *Arg = Builder->CreateZExt(CI->getArgOperand(*I), IntTy);
      if (!SizeValue) {
        SizeValue = Arg;
        continue;
      }
      SizeValue = Builder->CreateMul(SizeValue, Arg);
    }
    RETURN(true);

    // TODO: handle more standard functions:
    // - strdup / strndup
    // - strcpy / strncpy
    // - memcpy / memmove
    // - strcat / strncat

  } else if (PHINode *PHI = dyn_cast<PHINode>(Ptr)) {
    // create 2 PHIs: one for offset and another for size
    PHINode *OffsetPHI = Builder->CreatePHI(IntTy, PHI->getNumIncomingValues());
    PHINode *SizePHI   = Builder->CreatePHI(IntTy, PHI->getNumIncomingValues());

    // insert right away in the cache to handle recursive PHIs
    CacheData CacheEntry;
    CacheEntry.Offset = CacheEntry.Size = 0;
    CacheEntry.OffsetValue = OffsetPHI;
    CacheEntry.SizeValue = SizePHI;
    CacheEntry.ReturnVal = true;
    CacheMap[Ptr] = CacheEntry;

    // compute offset/size for each PHI incoming pointer
    bool someOk = false;
    for (unsigned i = 0, e = PHI->getNumIncomingValues(); i != e; ++i) {
      Builder->SetInsertPoint(PHI->getIncomingBlock(i)->getFirstInsertionPt());

      APInt PhiOffset(IntTyBits, 0), PhiSize(IntTyBits, 0);
      Value *PhiOffsetValue = 0, *PhiSizeValue = 0;
      someOk |= computeAllocSize(PHI->getIncomingValue(i), PhiOffset,
                                 PhiOffsetValue, PhiSize, PhiSizeValue);

      GET_VALUE(PhiOffsetValue, PhiOffset);
      GET_VALUE(PhiSizeValue, PhiSize);

      OffsetPHI->addIncoming(PhiOffsetValue, PHI->getIncomingBlock(i));
      SizePHI->addIncoming(PhiSizeValue, PHI->getIncomingBlock(i));
    }

    // fail here if we couldn't compute the size/offset in any incoming edge
    if (!someOk)
      RETURN(false);

    OffsetValue = OffsetPHI;
    SizeValue = SizePHI;
    RETURN(true);    

  } else if (isa<UndefValue>(Ptr)) {
    Size = 0;
    RETURN(true);

  } else if (isa<LoadInst>(Ptr)) {
    ++ChecksUnableLoad;
    RETURN(false);
  }

  RETURN(false);

cache_and_return:
  // cache the result and return
  CacheData CacheEntry;
  CacheEntry.Offset = Offset;
  CacheEntry.OffsetValue = OffsetValue;
  CacheEntry.Size = Size;
  CacheEntry.SizeValue = SizeValue;
  CacheEntry.ReturnVal = ReturnVal;
  CacheMap[Ptr] = CacheEntry;

  Builder->SetInsertPoint(PrevInsertPoint);
  return ReturnVal;
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

  IntegerType *IntTy = TD->getIntPtrType(Fn->getContext());
  unsigned IntTyBits = IntTy->getBitWidth();

  APInt Offset(IntTyBits, 0), Size(IntTyBits, 0);
  Value *OffsetValue = 0, *SizeValue = 0;

  if (!computeAllocSize(Ptr, Offset, OffsetValue, Size, SizeValue)) {
    DEBUG(dbgs() << "computeAllocSize failed:\n" << *Ptr << "\n");
    ++ChecksUnable;
    return false;
  }

  // three checks are required to ensure safety:
  // . Offset >= 0  (since the offset is given from the base ptr)
  // . Size >= Offset  (unsigned)
  // . Size - Offset >= NeededSize  (unsigned)
  if (!OffsetValue && !SizeValue) {
    if (Offset.slt(0) || Size.ult(Offset) || (Size - Offset).ult(NeededSize)) {
      // Out of bounds
      emitBranchToTrap();
      ++ChecksAdded;
      return true;
    }
    // in bounds
    ++ChecksSkipped;
    return false;
  }

  // emit check for offset < 0
  Value *CmpOffset = 0;
  if (OffsetValue)
    CmpOffset = Builder->CreateICmpSLT(OffsetValue, ConstantInt::get(IntTy, 0));
  else if (Offset.slt(0)) {
    // offset proved to be negative
    emitBranchToTrap();
    ++ChecksAdded;
    return true;
  }

  // we couldn't determine statically if the memory access is safe; emit a
  // run-time check
  GET_VALUE(OffsetValue, Offset);
  GET_VALUE(SizeValue, Size);

  Value *NeededSizeVal = ConstantInt::get(IntTy, NeededSize);
  // FIXME: add NSW/NUW here?  -- we dont care if the subtraction overflows
  Value *ObjSize = Builder->CreateSub(SizeValue, OffsetValue);
  Value *Cmp1 = Builder->CreateICmpULT(SizeValue, OffsetValue);
  Value *Cmp2 = Builder->CreateICmpULT(ObjSize, NeededSizeVal);
  Value *Or = Builder->CreateOr(Cmp1, Cmp2);
  if (CmpOffset)
    Or = Builder->CreateOr(CmpOffset, Or);
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
