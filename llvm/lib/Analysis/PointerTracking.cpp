//===- PointerTracking.cpp - Pointer Bounds Tracking ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements tracking of pointer bounds.
//
//===----------------------------------------------------------------------===//
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PointerTracking.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Value.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetData.h"

namespace llvm {
char PointerTracking::ID=0;
PointerTracking::PointerTracking() : FunctionPass(&ID) {}

bool PointerTracking::runOnFunction(Function &F) {
  predCache.clear();
  assert(analyzing.empty());
  FF = &F;
  TD = getAnalysisIfAvailable<TargetData>();
  SE = &getAnalysis<ScalarEvolution>();
  LI = &getAnalysis<LoopInfo>();
  DT = &getAnalysis<DominatorTree>();
  return false;
}

void PointerTracking::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequiredTransitive<DominatorTree>();
  AU.addRequiredTransitive<LoopInfo>();
  AU.addRequiredTransitive<ScalarEvolution>();
  AU.setPreservesAll();
}

bool PointerTracking::doInitialization(Module &M) {
  const Type *PTy = PointerType::getUnqual(Type::getInt8Ty(M.getContext()));

  // Find calloc(i64, i64) or calloc(i32, i32).
  callocFunc = M.getFunction("calloc");
  if (callocFunc) {
    const FunctionType *Ty = callocFunc->getFunctionType();

    std::vector<const Type*> args, args2;
    args.push_back(Type::getInt64Ty(M.getContext()));
    args.push_back(Type::getInt64Ty(M.getContext()));
    args2.push_back(Type::getInt32Ty(M.getContext()));
    args2.push_back(Type::getInt32Ty(M.getContext()));
    const FunctionType *Calloc1Type =
      FunctionType::get(PTy, args, false);
    const FunctionType *Calloc2Type =
      FunctionType::get(PTy, args2, false);
    if (Ty != Calloc1Type && Ty != Calloc2Type)
      callocFunc = 0; // Give up
  }

  // Find realloc(i8*, i64) or realloc(i8*, i32).
  reallocFunc = M.getFunction("realloc");
  if (reallocFunc) {
    const FunctionType *Ty = reallocFunc->getFunctionType();
    std::vector<const Type*> args, args2;
    args.push_back(PTy);
    args.push_back(Type::getInt64Ty(M.getContext()));
    args2.push_back(PTy);
    args2.push_back(Type::getInt32Ty(M.getContext()));

    const FunctionType *Realloc1Type =
      FunctionType::get(PTy, args, false);
    const FunctionType *Realloc2Type =
      FunctionType::get(PTy, args2, false);
    if (Ty != Realloc1Type && Ty != Realloc2Type)
      reallocFunc = 0; // Give up
  }
  return false;
}

// Calculates the number of elements allocated for pointer P,
// the type of the element is stored in Ty.
const SCEV *PointerTracking::computeAllocationCount(Value *P,
                                                    const Type *&Ty) const {
  Value *V = P->stripPointerCasts();
  if (AllocationInst *AI = dyn_cast<AllocationInst>(V)) {
    Value *arraySize = AI->getArraySize();
    Ty = AI->getAllocatedType();
    // arraySize elements of type Ty.
    return SE->getSCEV(arraySize);
  }

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
    if (GV->hasDefinitiveInitializer()) {
      Constant *C = GV->getInitializer();
      if (const ArrayType *ATy = dyn_cast<ArrayType>(C->getType())) {
        Ty = ATy->getElementType();
        return SE->getConstant(Type::getInt32Ty(P->getContext()),
                               ATy->getNumElements());
      }
    }
    Ty = GV->getType();
    return SE->getConstant(Type::getInt32Ty(P->getContext()), 1);
    //TODO: implement more tracking for globals
  }

  if (CallInst *CI = dyn_cast<CallInst>(V)) {
    CallSite CS(CI);
    Function *F = dyn_cast<Function>(CS.getCalledValue()->stripPointerCasts());
    const Loop *L = LI->getLoopFor(CI->getParent());
    if (F == callocFunc) {
      Ty = Type::getInt8Ty(P->getContext());
      // calloc allocates arg0*arg1 bytes.
      return SE->getSCEVAtScope(SE->getMulExpr(SE->getSCEV(CS.getArgument(0)),
                                               SE->getSCEV(CS.getArgument(1))),
                                L);
    } else if (F == reallocFunc) {
      Ty = Type::getInt8Ty(P->getContext());
      // realloc allocates arg1 bytes.
      return SE->getSCEVAtScope(CS.getArgument(1), L);
    }
  }

  return SE->getCouldNotCompute();
}

// Calculates the number of elements of type Ty allocated for P.
const SCEV *PointerTracking::computeAllocationCountForType(Value *P,
                                                           const Type *Ty)
  const {
    const Type *elementTy;
    const SCEV *Count = computeAllocationCount(P, elementTy);
    if (isa<SCEVCouldNotCompute>(Count))
      return Count;
    if (elementTy == Ty)
      return Count;

    if (!TD) // need TargetData from this point forward
      return SE->getCouldNotCompute();

    uint64_t elementSize = TD->getTypeAllocSize(elementTy);
    uint64_t wantSize = TD->getTypeAllocSize(Ty);
    if (elementSize == wantSize)
      return Count;
    if (elementSize % wantSize) //fractional counts not possible
      return SE->getCouldNotCompute();
    return SE->getMulExpr(Count, SE->getConstant(Count->getType(),
                                                 elementSize/wantSize));
}

const SCEV *PointerTracking::getAllocationElementCount(Value *V) const {
  // We only deal with pointers.
  const PointerType *PTy = cast<PointerType>(V->getType());
  return computeAllocationCountForType(V, PTy->getElementType());
}

const SCEV *PointerTracking::getAllocationSizeInBytes(Value *V) const {
  return computeAllocationCountForType(V, Type::getInt8Ty(V->getContext()));
}

// Helper for isLoopGuardedBy that checks the swapped and inverted predicate too
enum SolverResult PointerTracking::isLoopGuardedBy(const Loop *L,
                                                   Predicate Pred,
                                                   const SCEV *A,
                                                   const SCEV *B) const {
  if (SE->isLoopGuardedByCond(L, Pred, A, B))
    return AlwaysTrue;
  Pred = ICmpInst::getSwappedPredicate(Pred);
  if (SE->isLoopGuardedByCond(L, Pred, B, A))
    return AlwaysTrue;

  Pred = ICmpInst::getInversePredicate(Pred);
  if (SE->isLoopGuardedByCond(L, Pred, B, A))
    return AlwaysFalse;
  Pred = ICmpInst::getSwappedPredicate(Pred);
  if (SE->isLoopGuardedByCond(L, Pred, A, B))
    return AlwaysTrue;
  return Unknown;
}

enum SolverResult PointerTracking::checkLimits(const SCEV *Offset,
                                               const SCEV *Limit,
                                               BasicBlock *BB)
{
  //FIXME: merge implementation
  return Unknown;
}

void PointerTracking::getPointerOffset(Value *Pointer, Value *&Base,
                                       const SCEV *&Limit,
                                       const SCEV *&Offset) const
{
    Pointer = Pointer->stripPointerCasts();
    Base = Pointer->getUnderlyingObject();
    Limit = getAllocationSizeInBytes(Base);
    if (isa<SCEVCouldNotCompute>(Limit)) {
      Base = 0;
      Offset = Limit;
      return;
    }

    Offset = SE->getMinusSCEV(SE->getSCEV(Pointer), SE->getSCEV(Base));
    if (isa<SCEVCouldNotCompute>(Offset)) {
      Base = 0;
      Limit = Offset;
    }
}

void PointerTracking::print(raw_ostream &OS, const Module* M) const {
  // Calling some PT methods may cause caches to be updated, however
  // this should be safe for the same reason its safe for SCEV.
  PointerTracking &PT = *const_cast<PointerTracking*>(this);
  for (inst_iterator I=inst_begin(*FF), E=inst_end(*FF); I != E; ++I) {
    if (!isa<PointerType>(I->getType()))
      continue;
    Value *Base;
    const SCEV *Limit, *Offset;
    getPointerOffset(&*I, Base, Limit, Offset);
    if (!Base)
      continue;

    if (Base == &*I) {
      const SCEV *S = getAllocationElementCount(Base);
      OS << *Base << " ==> " << *S << " elements, ";
      OS << *Limit << " bytes allocated\n";
      continue;
    }
    OS << &*I << " -- base: " << *Base;
    OS << " offset: " << *Offset;

    enum SolverResult res = PT.checkLimits(Offset, Limit, I->getParent());
    switch (res) {
    case AlwaysTrue:
      OS << " always safe\n";
      break;
    case AlwaysFalse:
      OS << " always unsafe\n";
      break;
    case Unknown:
      OS << " <<unknown>>\n";
      break;
    }
  }
}

void PointerTracking::print(std::ostream &o, const Module* M) const {
  raw_os_ostream OS(o);
  print(OS, M);
}

static RegisterPass<PointerTracking> X("pointertracking",
                                       "Track pointer bounds", false, true);
}
