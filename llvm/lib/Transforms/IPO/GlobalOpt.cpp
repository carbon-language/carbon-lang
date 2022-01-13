//===- GlobalOpt.cpp - Optimize Global Variables --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass transforms simple global variables that never have their address
// taken.  If obviously true, it marks read/write globals as constant, deletes
// variables only stored to, etc.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/GlobalOpt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/CtorUtils.h"
#include "llvm/Transforms/Utils/Evaluator.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "globalopt"

STATISTIC(NumMarked    , "Number of globals marked constant");
STATISTIC(NumUnnamed   , "Number of globals marked unnamed_addr");
STATISTIC(NumSRA       , "Number of aggregate globals broken into scalars");
STATISTIC(NumSubstitute,"Number of globals with initializers stored into them");
STATISTIC(NumDeleted   , "Number of globals deleted");
STATISTIC(NumGlobUses  , "Number of global uses devirtualized");
STATISTIC(NumLocalized , "Number of globals localized");
STATISTIC(NumShrunkToBool  , "Number of global vars shrunk to booleans");
STATISTIC(NumFastCallFns   , "Number of functions converted to fastcc");
STATISTIC(NumCtorsEvaluated, "Number of static ctors evaluated");
STATISTIC(NumNestRemoved   , "Number of nest attributes removed");
STATISTIC(NumAliasesResolved, "Number of global aliases resolved");
STATISTIC(NumAliasesRemoved, "Number of global aliases eliminated");
STATISTIC(NumCXXDtorsRemoved, "Number of global C++ destructors removed");
STATISTIC(NumInternalFunc, "Number of internal functions");
STATISTIC(NumColdCC, "Number of functions marked coldcc");

static cl::opt<bool>
    EnableColdCCStressTest("enable-coldcc-stress-test",
                           cl::desc("Enable stress test of coldcc by adding "
                                    "calling conv to all internal functions."),
                           cl::init(false), cl::Hidden);

static cl::opt<int> ColdCCRelFreq(
    "coldcc-rel-freq", cl::Hidden, cl::init(2), cl::ZeroOrMore,
    cl::desc(
        "Maximum block frequency, expressed as a percentage of caller's "
        "entry frequency, for a call site to be considered cold for enabling"
        "coldcc"));

/// Is this global variable possibly used by a leak checker as a root?  If so,
/// we might not really want to eliminate the stores to it.
static bool isLeakCheckerRoot(GlobalVariable *GV) {
  // A global variable is a root if it is a pointer, or could plausibly contain
  // a pointer.  There are two challenges; one is that we could have a struct
  // the has an inner member which is a pointer.  We recurse through the type to
  // detect these (up to a point).  The other is that we may actually be a union
  // of a pointer and another type, and so our LLVM type is an integer which
  // gets converted into a pointer, or our type is an [i8 x #] with a pointer
  // potentially contained here.

  if (GV->hasPrivateLinkage())
    return false;

  SmallVector<Type *, 4> Types;
  Types.push_back(GV->getValueType());

  unsigned Limit = 20;
  do {
    Type *Ty = Types.pop_back_val();
    switch (Ty->getTypeID()) {
      default: break;
      case Type::PointerTyID:
        return true;
      case Type::FixedVectorTyID:
      case Type::ScalableVectorTyID:
        if (cast<VectorType>(Ty)->getElementType()->isPointerTy())
          return true;
        break;
      case Type::ArrayTyID:
        Types.push_back(cast<ArrayType>(Ty)->getElementType());
        break;
      case Type::StructTyID: {
        StructType *STy = cast<StructType>(Ty);
        if (STy->isOpaque()) return true;
        for (StructType::element_iterator I = STy->element_begin(),
                 E = STy->element_end(); I != E; ++I) {
          Type *InnerTy = *I;
          if (isa<PointerType>(InnerTy)) return true;
          if (isa<StructType>(InnerTy) || isa<ArrayType>(InnerTy) ||
              isa<VectorType>(InnerTy))
            Types.push_back(InnerTy);
        }
        break;
      }
    }
    if (--Limit == 0) return true;
  } while (!Types.empty());
  return false;
}

/// Given a value that is stored to a global but never read, determine whether
/// it's safe to remove the store and the chain of computation that feeds the
/// store.
static bool IsSafeComputationToRemove(
    Value *V, function_ref<TargetLibraryInfo &(Function &)> GetTLI) {
  do {
    if (isa<Constant>(V))
      return true;
    if (!V->hasOneUse())
      return false;
    if (isa<LoadInst>(V) || isa<InvokeInst>(V) || isa<Argument>(V) ||
        isa<GlobalValue>(V))
      return false;
    if (isAllocationFn(V, GetTLI))
      return true;

    Instruction *I = cast<Instruction>(V);
    if (I->mayHaveSideEffects())
      return false;
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
      if (!GEP->hasAllConstantIndices())
        return false;
    } else if (I->getNumOperands() != 1) {
      return false;
    }

    V = I->getOperand(0);
  } while (true);
}

/// This GV is a pointer root.  Loop over all users of the global and clean up
/// any that obviously don't assign the global a value that isn't dynamically
/// allocated.
static bool
CleanupPointerRootUsers(GlobalVariable *GV,
                        function_ref<TargetLibraryInfo &(Function &)> GetTLI) {
  // A brief explanation of leak checkers.  The goal is to find bugs where
  // pointers are forgotten, causing an accumulating growth in memory
  // usage over time.  The common strategy for leak checkers is to explicitly
  // allow the memory pointed to by globals at exit.  This is popular because it
  // also solves another problem where the main thread of a C++ program may shut
  // down before other threads that are still expecting to use those globals. To
  // handle that case, we expect the program may create a singleton and never
  // destroy it.

  bool Changed = false;

  // If Dead[n].first is the only use of a malloc result, we can delete its
  // chain of computation and the store to the global in Dead[n].second.
  SmallVector<std::pair<Instruction *, Instruction *>, 32> Dead;

  // Constants can't be pointers to dynamically allocated memory.
  for (User *U : llvm::make_early_inc_range(GV->users())) {
    if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
      Value *V = SI->getValueOperand();
      if (isa<Constant>(V)) {
        Changed = true;
        SI->eraseFromParent();
      } else if (Instruction *I = dyn_cast<Instruction>(V)) {
        if (I->hasOneUse())
          Dead.push_back(std::make_pair(I, SI));
      }
    } else if (MemSetInst *MSI = dyn_cast<MemSetInst>(U)) {
      if (isa<Constant>(MSI->getValue())) {
        Changed = true;
        MSI->eraseFromParent();
      } else if (Instruction *I = dyn_cast<Instruction>(MSI->getValue())) {
        if (I->hasOneUse())
          Dead.push_back(std::make_pair(I, MSI));
      }
    } else if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(U)) {
      GlobalVariable *MemSrc = dyn_cast<GlobalVariable>(MTI->getSource());
      if (MemSrc && MemSrc->isConstant()) {
        Changed = true;
        MTI->eraseFromParent();
      } else if (Instruction *I = dyn_cast<Instruction>(MemSrc)) {
        if (I->hasOneUse())
          Dead.push_back(std::make_pair(I, MTI));
      }
    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(U)) {
      if (CE->use_empty()) {
        CE->destroyConstant();
        Changed = true;
      }
    } else if (Constant *C = dyn_cast<Constant>(U)) {
      if (isSafeToDestroyConstant(C)) {
        C->destroyConstant();
        // This could have invalidated UI, start over from scratch.
        Dead.clear();
        CleanupPointerRootUsers(GV, GetTLI);
        return true;
      }
    }
  }

  for (int i = 0, e = Dead.size(); i != e; ++i) {
    if (IsSafeComputationToRemove(Dead[i].first, GetTLI)) {
      Dead[i].second->eraseFromParent();
      Instruction *I = Dead[i].first;
      do {
        if (isAllocationFn(I, GetTLI))
          break;
        Instruction *J = dyn_cast<Instruction>(I->getOperand(0));
        if (!J)
          break;
        I->eraseFromParent();
        I = J;
      } while (true);
      I->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

/// We just marked GV constant.  Loop over all users of the global, cleaning up
/// the obvious ones.  This is largely just a quick scan over the use list to
/// clean up the easy and obvious cruft.  This returns true if it made a change.
static bool CleanupConstantGlobalUsers(GlobalVariable *GV,
                                       const DataLayout &DL) {
  Constant *Init = GV->getInitializer();
  SmallVector<User *, 8> WorkList(GV->users());
  SmallPtrSet<User *, 8> Visited;
  bool Changed = false;

  SmallVector<WeakTrackingVH> MaybeDeadInsts;
  auto EraseFromParent = [&](Instruction *I) {
    for (Value *Op : I->operands())
      if (auto *OpI = dyn_cast<Instruction>(Op))
        MaybeDeadInsts.push_back(OpI);
    I->eraseFromParent();
    Changed = true;
  };
  while (!WorkList.empty()) {
    User *U = WorkList.pop_back_val();
    if (!Visited.insert(U).second)
      continue;

    if (auto *BO = dyn_cast<BitCastOperator>(U))
      append_range(WorkList, BO->users());
    if (auto *ASC = dyn_cast<AddrSpaceCastOperator>(U))
      append_range(WorkList, ASC->users());
    else if (auto *GEP = dyn_cast<GEPOperator>(U))
      append_range(WorkList, GEP->users());
    else if (auto *LI = dyn_cast<LoadInst>(U)) {
      // A load from a uniform value is always the same, regardless of any
      // applied offset.
      Type *Ty = LI->getType();
      if (Constant *Res = ConstantFoldLoadFromUniformValue(Init, Ty)) {
        LI->replaceAllUsesWith(Res);
        EraseFromParent(LI);
        continue;
      }

      Value *PtrOp = LI->getPointerOperand();
      APInt Offset(DL.getIndexTypeSizeInBits(PtrOp->getType()), 0);
      PtrOp = PtrOp->stripAndAccumulateConstantOffsets(
          DL, Offset, /* AllowNonInbounds */ true);
      if (PtrOp == GV) {
        if (auto *Value = ConstantFoldLoadFromConst(Init, Ty, Offset, DL)) {
          LI->replaceAllUsesWith(Value);
          EraseFromParent(LI);
        }
      }
    } else if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
      // Store must be unreachable or storing Init into the global.
      EraseFromParent(SI);
    } else if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(U)) { // memset/cpy/mv
      if (getUnderlyingObject(MI->getRawDest()) == GV)
        EraseFromParent(MI);
    }
  }

  Changed |=
      RecursivelyDeleteTriviallyDeadInstructionsPermissive(MaybeDeadInsts);
  GV->removeDeadConstantUsers();
  return Changed;
}

static bool isSafeSROAElementUse(Value *V);

/// Return true if the specified GEP is a safe user of a derived
/// expression from a global that we want to SROA.
static bool isSafeSROAGEP(User *U) {
  // Check to see if this ConstantExpr GEP is SRA'able.  In particular, we
  // don't like < 3 operand CE's, and we don't like non-constant integer
  // indices.  This enforces that all uses are 'gep GV, 0, C, ...' for some
  // value of C.
  if (U->getNumOperands() < 3 || !isa<Constant>(U->getOperand(1)) ||
      !cast<Constant>(U->getOperand(1))->isNullValue())
    return false;

  gep_type_iterator GEPI = gep_type_begin(U), E = gep_type_end(U);
  ++GEPI; // Skip over the pointer index.

  // For all other level we require that the indices are constant and inrange.
  // In particular, consider: A[0][i].  We cannot know that the user isn't doing
  // invalid things like allowing i to index an out-of-range subscript that
  // accesses A[1]. This can also happen between different members of a struct
  // in llvm IR.
  for (; GEPI != E; ++GEPI) {
    if (GEPI.isStruct())
      continue;

    ConstantInt *IdxVal = dyn_cast<ConstantInt>(GEPI.getOperand());
    if (!IdxVal || (GEPI.isBoundedSequential() &&
                    IdxVal->getZExtValue() >= GEPI.getSequentialNumElements()))
      return false;
  }

  return llvm::all_of(U->users(), isSafeSROAElementUse);
}

/// Return true if the specified instruction is a safe user of a derived
/// expression from a global that we want to SROA.
static bool isSafeSROAElementUse(Value *V) {
  // We might have a dead and dangling constant hanging off of here.
  if (Constant *C = dyn_cast<Constant>(V))
    return isSafeToDestroyConstant(C);

  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return false;

  // Loads are ok.
  if (isa<LoadInst>(I)) return true;

  // Stores *to* the pointer are ok.
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->getOperand(0) != V;

  // Otherwise, it must be a GEP. Check it and its users are safe to SRA.
  return isa<GetElementPtrInst>(I) && isSafeSROAGEP(I);
}

/// Look at all uses of the global and decide whether it is safe for us to
/// perform this transformation.
static bool GlobalUsersSafeToSRA(GlobalValue *GV) {
  for (User *U : GV->users()) {
    // The user of the global must be a GEP Inst or a ConstantExpr GEP.
    if (!isa<GetElementPtrInst>(U) &&
        (!isa<ConstantExpr>(U) ||
        cast<ConstantExpr>(U)->getOpcode() != Instruction::GetElementPtr))
      return false;

    // Check the gep and it's users are safe to SRA
    if (!isSafeSROAGEP(U))
      return false;
  }

  return true;
}

static bool IsSRASequential(Type *T) {
  return isa<ArrayType>(T) || isa<VectorType>(T);
}
static uint64_t GetSRASequentialNumElements(Type *T) {
  if (ArrayType *AT = dyn_cast<ArrayType>(T))
    return AT->getNumElements();
  return cast<FixedVectorType>(T)->getNumElements();
}
static Type *GetSRASequentialElementType(Type *T) {
  if (ArrayType *AT = dyn_cast<ArrayType>(T))
    return AT->getElementType();
  return cast<VectorType>(T)->getElementType();
}
static bool CanDoGlobalSRA(GlobalVariable *GV) {
  Constant *Init = GV->getInitializer();

  if (isa<StructType>(Init->getType())) {
    // nothing to check
  } else if (IsSRASequential(Init->getType())) {
    if (GetSRASequentialNumElements(Init->getType()) > 16 &&
        GV->hasNUsesOrMore(16))
      return false; // It's not worth it.
  } else
    return false;

  return GlobalUsersSafeToSRA(GV);
}

/// Copy over the debug info for a variable to its SRA replacements.
static void transferSRADebugInfo(GlobalVariable *GV, GlobalVariable *NGV,
                                 uint64_t FragmentOffsetInBits,
                                 uint64_t FragmentSizeInBits,
                                 uint64_t VarSize) {
  SmallVector<DIGlobalVariableExpression *, 1> GVs;
  GV->getDebugInfo(GVs);
  for (auto *GVE : GVs) {
    DIVariable *Var = GVE->getVariable();
    DIExpression *Expr = GVE->getExpression();
    // If the FragmentSize is smaller than the variable,
    // emit a fragment expression.
    if (FragmentSizeInBits < VarSize) {
      if (auto E = DIExpression::createFragmentExpression(
              Expr, FragmentOffsetInBits, FragmentSizeInBits))
        Expr = *E;
      else
        return;
    }
    auto *NGVE = DIGlobalVariableExpression::get(GVE->getContext(), Var, Expr);
    NGV->addDebugInfo(NGVE);
  }
}

/// Perform scalar replacement of aggregates on the specified global variable.
/// This opens the door for other optimizations by exposing the behavior of the
/// program in a more fine-grained way.  We have determined that this
/// transformation is safe already.  We return the first global variable we
/// insert so that the caller can reprocess it.
static GlobalVariable *SRAGlobal(GlobalVariable *GV, const DataLayout &DL) {
  // Make sure this global only has simple uses that we can SRA.
  if (!CanDoGlobalSRA(GV))
    return nullptr;

  assert(GV->hasLocalLinkage());
  Constant *Init = GV->getInitializer();
  Type *Ty = Init->getType();
  uint64_t VarSize = DL.getTypeSizeInBits(Ty);

  std::map<unsigned, GlobalVariable *> NewGlobals;

  // Get the alignment of the global, either explicit or target-specific.
  Align StartAlignment =
      DL.getValueOrABITypeAlignment(GV->getAlign(), GV->getType());

  // Loop over all users and create replacement variables for used aggregate
  // elements.
  for (User *GEP : GV->users()) {
    assert(((isa<ConstantExpr>(GEP) && cast<ConstantExpr>(GEP)->getOpcode() ==
                                           Instruction::GetElementPtr) ||
            isa<GetElementPtrInst>(GEP)) &&
           "NonGEP CE's are not SRAable!");

    // Ignore the 1th operand, which has to be zero or else the program is quite
    // broken (undefined).  Get the 2nd operand, which is the structure or array
    // index.
    unsigned ElementIdx = cast<ConstantInt>(GEP->getOperand(2))->getZExtValue();
    if (NewGlobals.count(ElementIdx) == 1)
      continue; // we`ve already created replacement variable
    assert(NewGlobals.count(ElementIdx) == 0);

    Type *ElTy = nullptr;
    if (StructType *STy = dyn_cast<StructType>(Ty))
      ElTy = STy->getElementType(ElementIdx);
    else
      ElTy = GetSRASequentialElementType(Ty);
    assert(ElTy);

    Constant *In = Init->getAggregateElement(ElementIdx);
    assert(In && "Couldn't get element of initializer?");

    GlobalVariable *NGV = new GlobalVariable(
        ElTy, false, GlobalVariable::InternalLinkage, In,
        GV->getName() + "." + Twine(ElementIdx), GV->getThreadLocalMode(),
        GV->getType()->getAddressSpace());
    NGV->copyAttributesFrom(GV);
    NewGlobals.insert(std::make_pair(ElementIdx, NGV));

    if (StructType *STy = dyn_cast<StructType>(Ty)) {
      const StructLayout &Layout = *DL.getStructLayout(STy);

      // Calculate the known alignment of the field.  If the original aggregate
      // had 256 byte alignment for example, something might depend on that:
      // propagate info to each field.
      uint64_t FieldOffset = Layout.getElementOffset(ElementIdx);
      Align NewAlign = commonAlignment(StartAlignment, FieldOffset);
      if (NewAlign > DL.getABITypeAlign(STy->getElementType(ElementIdx)))
        NGV->setAlignment(NewAlign);

      // Copy over the debug info for the variable.
      uint64_t Size = DL.getTypeAllocSizeInBits(NGV->getValueType());
      uint64_t FragmentOffsetInBits = Layout.getElementOffsetInBits(ElementIdx);
      transferSRADebugInfo(GV, NGV, FragmentOffsetInBits, Size, VarSize);
    } else {
      uint64_t EltSize = DL.getTypeAllocSize(ElTy);
      Align EltAlign = DL.getABITypeAlign(ElTy);
      uint64_t FragmentSizeInBits = DL.getTypeAllocSizeInBits(ElTy);

      // Calculate the known alignment of the field.  If the original aggregate
      // had 256 byte alignment for example, something might depend on that:
      // propagate info to each field.
      Align NewAlign = commonAlignment(StartAlignment, EltSize * ElementIdx);
      if (NewAlign > EltAlign)
        NGV->setAlignment(NewAlign);
      transferSRADebugInfo(GV, NGV, FragmentSizeInBits * ElementIdx,
                           FragmentSizeInBits, VarSize);
    }
  }

  if (NewGlobals.empty())
    return nullptr;

  Module::GlobalListType &Globals = GV->getParent()->getGlobalList();
  for (auto NewGlobalVar : NewGlobals)
    Globals.push_back(NewGlobalVar.second);

  LLVM_DEBUG(dbgs() << "PERFORMING GLOBAL SRA ON: " << *GV << "\n");

  Constant *NullInt =Constant::getNullValue(Type::getInt32Ty(GV->getContext()));

  // Loop over all of the uses of the global, replacing the constantexpr geps,
  // with smaller constantexpr geps or direct references.
  while (!GV->use_empty()) {
    User *GEP = GV->user_back();
    assert(((isa<ConstantExpr>(GEP) &&
             cast<ConstantExpr>(GEP)->getOpcode()==Instruction::GetElementPtr)||
            isa<GetElementPtrInst>(GEP)) && "NonGEP CE's are not SRAable!");

    // Ignore the 1th operand, which has to be zero or else the program is quite
    // broken (undefined).  Get the 2nd operand, which is the structure or array
    // index.
    unsigned ElementIdx = cast<ConstantInt>(GEP->getOperand(2))->getZExtValue();
    assert(NewGlobals.count(ElementIdx) == 1);

    Value *NewPtr = NewGlobals[ElementIdx];
    Type *NewTy = NewGlobals[ElementIdx]->getValueType();

    // Form a shorter GEP if needed.
    if (GEP->getNumOperands() > 3) {
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(GEP)) {
        SmallVector<Constant*, 8> Idxs;
        Idxs.push_back(NullInt);
        for (unsigned i = 3, e = CE->getNumOperands(); i != e; ++i)
          Idxs.push_back(CE->getOperand(i));
        NewPtr =
            ConstantExpr::getGetElementPtr(NewTy, cast<Constant>(NewPtr), Idxs);
      } else {
        GetElementPtrInst *GEPI = cast<GetElementPtrInst>(GEP);
        SmallVector<Value*, 8> Idxs;
        Idxs.push_back(NullInt);
        for (unsigned i = 3, e = GEPI->getNumOperands(); i != e; ++i)
          Idxs.push_back(GEPI->getOperand(i));
        NewPtr = GetElementPtrInst::Create(
            NewTy, NewPtr, Idxs, GEPI->getName() + "." + Twine(ElementIdx),
            GEPI);
      }
    }
    GEP->replaceAllUsesWith(NewPtr);

    // We changed the pointer of any memory access user. Recalculate alignments.
    for (User *U : NewPtr->users()) {
      if (auto *Load = dyn_cast<LoadInst>(U)) {
        Align PrefAlign = DL.getPrefTypeAlign(Load->getType());
        Align NewAlign = getOrEnforceKnownAlignment(Load->getPointerOperand(),
                                                    PrefAlign, DL, Load);
        Load->setAlignment(NewAlign);
      }
      if (auto *Store = dyn_cast<StoreInst>(U)) {
        Align PrefAlign =
            DL.getPrefTypeAlign(Store->getValueOperand()->getType());
        Align NewAlign = getOrEnforceKnownAlignment(Store->getPointerOperand(),
                                                    PrefAlign, DL, Store);
        Store->setAlignment(NewAlign);
      }
    }

    if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(GEP))
      GEPI->eraseFromParent();
    else
      cast<ConstantExpr>(GEP)->destroyConstant();
  }

  // Delete the old global, now that it is dead.
  Globals.erase(GV);
  ++NumSRA;

  assert(NewGlobals.size() > 0);
  return NewGlobals.begin()->second;
}

/// Return true if all users of the specified value will trap if the value is
/// dynamically null.  PHIs keeps track of any phi nodes we've seen to avoid
/// reprocessing them.
static bool AllUsesOfValueWillTrapIfNull(const Value *V,
                                        SmallPtrSetImpl<const PHINode*> &PHIs) {
  for (const User *U : V->users()) {
    if (const Instruction *I = dyn_cast<Instruction>(U)) {
      // If null pointer is considered valid, then all uses are non-trapping.
      // Non address-space 0 globals have already been pruned by the caller.
      if (NullPointerIsDefined(I->getFunction()))
        return false;
    }
    if (isa<LoadInst>(U)) {
      // Will trap.
    } else if (const StoreInst *SI = dyn_cast<StoreInst>(U)) {
      if (SI->getOperand(0) == V) {
        //cerr << "NONTRAPPING USE: " << *U;
        return false;  // Storing the value.
      }
    } else if (const CallInst *CI = dyn_cast<CallInst>(U)) {
      if (CI->getCalledOperand() != V) {
        //cerr << "NONTRAPPING USE: " << *U;
        return false;  // Not calling the ptr
      }
    } else if (const InvokeInst *II = dyn_cast<InvokeInst>(U)) {
      if (II->getCalledOperand() != V) {
        //cerr << "NONTRAPPING USE: " << *U;
        return false;  // Not calling the ptr
      }
    } else if (const BitCastInst *CI = dyn_cast<BitCastInst>(U)) {
      if (!AllUsesOfValueWillTrapIfNull(CI, PHIs)) return false;
    } else if (const GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(U)) {
      if (!AllUsesOfValueWillTrapIfNull(GEPI, PHIs)) return false;
    } else if (const PHINode *PN = dyn_cast<PHINode>(U)) {
      // If we've already seen this phi node, ignore it, it has already been
      // checked.
      if (PHIs.insert(PN).second && !AllUsesOfValueWillTrapIfNull(PN, PHIs))
        return false;
    } else if (isa<ICmpInst>(U) &&
               !ICmpInst::isSigned(cast<ICmpInst>(U)->getPredicate()) &&
               isa<LoadInst>(U->getOperand(0)) &&
               isa<ConstantPointerNull>(U->getOperand(1))) {
      assert(isa<GlobalValue>(cast<LoadInst>(U->getOperand(0))
                                  ->getPointerOperand()
                                  ->stripPointerCasts()) &&
             "Should be GlobalVariable");
      // This and only this kind of non-signed ICmpInst is to be replaced with
      // the comparing of the value of the created global init bool later in
      // optimizeGlobalAddressOfMalloc for the global variable.
    } else {
      //cerr << "NONTRAPPING USE: " << *U;
      return false;
    }
  }
  return true;
}

/// Return true if all uses of any loads from GV will trap if the loaded value
/// is null.  Note that this also permits comparisons of the loaded value
/// against null, as a special case.
static bool allUsesOfLoadedValueWillTrapIfNull(const GlobalVariable *GV) {
  SmallVector<const Value *, 4> Worklist;
  Worklist.push_back(GV);
  while (!Worklist.empty()) {
    const Value *P = Worklist.pop_back_val();
    for (auto *U : P->users()) {
      if (auto *LI = dyn_cast<LoadInst>(U)) {
        SmallPtrSet<const PHINode *, 8> PHIs;
        if (!AllUsesOfValueWillTrapIfNull(LI, PHIs))
          return false;
      } else if (auto *SI = dyn_cast<StoreInst>(U)) {
        // Ignore stores to the global.
        if (SI->getPointerOperand() != P)
          return false;
      } else if (auto *CE = dyn_cast<ConstantExpr>(U)) {
        if (CE->stripPointerCasts() != GV)
          return false;
        // Check further the ConstantExpr.
        Worklist.push_back(CE);
      } else {
        // We don't know or understand this user, bail out.
        return false;
      }
    }
  }

  return true;
}

/// Get all the loads/store uses for global variable \p GV.
static void allUsesOfLoadAndStores(GlobalVariable *GV,
                                   SmallVector<Value *, 4> &Uses) {
  SmallVector<Value *, 4> Worklist;
  Worklist.push_back(GV);
  while (!Worklist.empty()) {
    auto *P = Worklist.pop_back_val();
    for (auto *U : P->users()) {
      if (auto *CE = dyn_cast<ConstantExpr>(U)) {
        Worklist.push_back(CE);
        continue;
      }

      assert((isa<LoadInst>(U) || isa<StoreInst>(U)) &&
             "Expect only load or store instructions");
      Uses.push_back(U);
    }
  }
}

static bool OptimizeAwayTrappingUsesOfValue(Value *V, Constant *NewV) {
  bool Changed = false;
  for (auto UI = V->user_begin(), E = V->user_end(); UI != E; ) {
    Instruction *I = cast<Instruction>(*UI++);
    // Uses are non-trapping if null pointer is considered valid.
    // Non address-space 0 globals are already pruned by the caller.
    if (NullPointerIsDefined(I->getFunction()))
      return false;
    if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      LI->setOperand(0, NewV);
      Changed = true;
    } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      if (SI->getOperand(1) == V) {
        SI->setOperand(1, NewV);
        Changed = true;
      }
    } else if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
      CallBase *CB = cast<CallBase>(I);
      if (CB->getCalledOperand() == V) {
        // Calling through the pointer!  Turn into a direct call, but be careful
        // that the pointer is not also being passed as an argument.
        CB->setCalledOperand(NewV);
        Changed = true;
        bool PassedAsArg = false;
        for (unsigned i = 0, e = CB->arg_size(); i != e; ++i)
          if (CB->getArgOperand(i) == V) {
            PassedAsArg = true;
            CB->setArgOperand(i, NewV);
          }

        if (PassedAsArg) {
          // Being passed as an argument also.  Be careful to not invalidate UI!
          UI = V->user_begin();
        }
      }
    } else if (CastInst *CI = dyn_cast<CastInst>(I)) {
      Changed |= OptimizeAwayTrappingUsesOfValue(CI,
                                ConstantExpr::getCast(CI->getOpcode(),
                                                      NewV, CI->getType()));
      if (CI->use_empty()) {
        Changed = true;
        CI->eraseFromParent();
      }
    } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(I)) {
      // Should handle GEP here.
      SmallVector<Constant*, 8> Idxs;
      Idxs.reserve(GEPI->getNumOperands()-1);
      for (User::op_iterator i = GEPI->op_begin() + 1, e = GEPI->op_end();
           i != e; ++i)
        if (Constant *C = dyn_cast<Constant>(*i))
          Idxs.push_back(C);
        else
          break;
      if (Idxs.size() == GEPI->getNumOperands()-1)
        Changed |= OptimizeAwayTrappingUsesOfValue(
            GEPI, ConstantExpr::getGetElementPtr(GEPI->getSourceElementType(),
                                                 NewV, Idxs));
      if (GEPI->use_empty()) {
        Changed = true;
        GEPI->eraseFromParent();
      }
    }
  }

  return Changed;
}

/// The specified global has only one non-null value stored into it.  If there
/// are uses of the loaded value that would trap if the loaded value is
/// dynamically null, then we know that they cannot be reachable with a null
/// optimize away the load.
static bool OptimizeAwayTrappingUsesOfLoads(
    GlobalVariable *GV, Constant *LV, const DataLayout &DL,
    function_ref<TargetLibraryInfo &(Function &)> GetTLI) {
  bool Changed = false;

  // Keep track of whether we are able to remove all the uses of the global
  // other than the store that defines it.
  bool AllNonStoreUsesGone = true;

  // Replace all uses of loads with uses of uses of the stored value.
  for (User *GlobalUser : llvm::make_early_inc_range(GV->users())) {
    if (LoadInst *LI = dyn_cast<LoadInst>(GlobalUser)) {
      Changed |= OptimizeAwayTrappingUsesOfValue(LI, LV);
      // If we were able to delete all uses of the loads
      if (LI->use_empty()) {
        LI->eraseFromParent();
        Changed = true;
      } else {
        AllNonStoreUsesGone = false;
      }
    } else if (isa<StoreInst>(GlobalUser)) {
      // Ignore the store that stores "LV" to the global.
      assert(GlobalUser->getOperand(1) == GV &&
             "Must be storing *to* the global");
    } else {
      AllNonStoreUsesGone = false;

      // If we get here we could have other crazy uses that are transitively
      // loaded.
      assert((isa<PHINode>(GlobalUser) || isa<SelectInst>(GlobalUser) ||
              isa<ConstantExpr>(GlobalUser) || isa<CmpInst>(GlobalUser) ||
              isa<BitCastInst>(GlobalUser) ||
              isa<GetElementPtrInst>(GlobalUser)) &&
             "Only expect load and stores!");
    }
  }

  if (Changed) {
    LLVM_DEBUG(dbgs() << "OPTIMIZED LOADS FROM STORED ONCE POINTER: " << *GV
                      << "\n");
    ++NumGlobUses;
  }

  // If we nuked all of the loads, then none of the stores are needed either,
  // nor is the global.
  if (AllNonStoreUsesGone) {
    if (isLeakCheckerRoot(GV)) {
      Changed |= CleanupPointerRootUsers(GV, GetTLI);
    } else {
      Changed = true;
      CleanupConstantGlobalUsers(GV, DL);
    }
    if (GV->use_empty()) {
      LLVM_DEBUG(dbgs() << "  *** GLOBAL NOW DEAD!\n");
      Changed = true;
      GV->eraseFromParent();
      ++NumDeleted;
    }
  }
  return Changed;
}

/// Walk the use list of V, constant folding all of the instructions that are
/// foldable.
static void ConstantPropUsersOf(Value *V, const DataLayout &DL,
                                TargetLibraryInfo *TLI) {
  for (Value::user_iterator UI = V->user_begin(), E = V->user_end(); UI != E; )
    if (Instruction *I = dyn_cast<Instruction>(*UI++))
      if (Constant *NewC = ConstantFoldInstruction(I, DL, TLI)) {
        I->replaceAllUsesWith(NewC);

        // Advance UI to the next non-I use to avoid invalidating it!
        // Instructions could multiply use V.
        while (UI != E && *UI == I)
          ++UI;
        if (isInstructionTriviallyDead(I, TLI))
          I->eraseFromParent();
      }
}

/// This function takes the specified global variable, and transforms the
/// program as if it always contained the result of the specified malloc.
/// Because it is always the result of the specified malloc, there is no reason
/// to actually DO the malloc.  Instead, turn the malloc into a global, and any
/// loads of GV as uses of the new global.
static GlobalVariable *
OptimizeGlobalAddressOfMalloc(GlobalVariable *GV, CallInst *CI, Type *AllocTy,
                              ConstantInt *NElements, const DataLayout &DL,
                              TargetLibraryInfo *TLI) {
  LLVM_DEBUG(errs() << "PROMOTING GLOBAL: " << *GV << "  CALL = " << *CI
                    << '\n');

  Type *GlobalType;
  if (NElements->getZExtValue() == 1)
    GlobalType = AllocTy;
  else
    // If we have an array allocation, the global variable is of an array.
    GlobalType = ArrayType::get(AllocTy, NElements->getZExtValue());

  // Create the new global variable.  The contents of the malloc'd memory is
  // undefined, so initialize with an undef value.
  GlobalVariable *NewGV = new GlobalVariable(
      *GV->getParent(), GlobalType, false, GlobalValue::InternalLinkage,
      UndefValue::get(GlobalType), GV->getName() + ".body", nullptr,
      GV->getThreadLocalMode());

  // If there are bitcast users of the malloc (which is typical, usually we have
  // a malloc + bitcast) then replace them with uses of the new global.  Update
  // other users to use the global as well.
  BitCastInst *TheBC = nullptr;
  while (!CI->use_empty()) {
    Instruction *User = cast<Instruction>(CI->user_back());
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(User)) {
      if (BCI->getType() == NewGV->getType()) {
        BCI->replaceAllUsesWith(NewGV);
        BCI->eraseFromParent();
      } else {
        BCI->setOperand(0, NewGV);
      }
    } else {
      if (!TheBC)
        TheBC = new BitCastInst(NewGV, CI->getType(), "newgv", CI);
      User->replaceUsesOfWith(CI, TheBC);
    }
  }

  SmallPtrSet<Constant *, 1> RepValues;
  RepValues.insert(NewGV);

  // If there is a comparison against null, we will insert a global bool to
  // keep track of whether the global was initialized yet or not.
  GlobalVariable *InitBool =
    new GlobalVariable(Type::getInt1Ty(GV->getContext()), false,
                       GlobalValue::InternalLinkage,
                       ConstantInt::getFalse(GV->getContext()),
                       GV->getName()+".init", GV->getThreadLocalMode());
  bool InitBoolUsed = false;

  // Loop over all instruction uses of GV, processing them in turn.
  SmallVector<Value *, 4> Guses;
  allUsesOfLoadAndStores(GV, Guses);
  for (auto *U : Guses) {
    if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
      // The global is initialized when the store to it occurs. If the stored
      // value is null value, the global bool is set to false, otherwise true.
      new StoreInst(ConstantInt::getBool(
                        GV->getContext(),
                        !isa<ConstantPointerNull>(SI->getValueOperand())),
                    InitBool, false, Align(1), SI->getOrdering(),
                    SI->getSyncScopeID(), SI);
      SI->eraseFromParent();
      continue;
    }

    LoadInst *LI = cast<LoadInst>(U);
    while (!LI->use_empty()) {
      Use &LoadUse = *LI->use_begin();
      ICmpInst *ICI = dyn_cast<ICmpInst>(LoadUse.getUser());
      if (!ICI) {
        auto *CE = ConstantExpr::getBitCast(NewGV, LI->getType());
        RepValues.insert(CE);
        LoadUse.set(CE);
        continue;
      }

      // Replace the cmp X, 0 with a use of the bool value.
      Value *LV = new LoadInst(InitBool->getValueType(), InitBool,
                               InitBool->getName() + ".val", false, Align(1),
                               LI->getOrdering(), LI->getSyncScopeID(), LI);
      InitBoolUsed = true;
      switch (ICI->getPredicate()) {
      default: llvm_unreachable("Unknown ICmp Predicate!");
      case ICmpInst::ICMP_ULT: // X < null -> always false
        LV = ConstantInt::getFalse(GV->getContext());
        break;
      case ICmpInst::ICMP_UGE: // X >= null -> always true
        LV = ConstantInt::getTrue(GV->getContext());
        break;
      case ICmpInst::ICMP_ULE:
      case ICmpInst::ICMP_EQ:
        LV = BinaryOperator::CreateNot(LV, "notinit", ICI);
        break;
      case ICmpInst::ICMP_NE:
      case ICmpInst::ICMP_UGT:
        break;  // no change.
      }
      ICI->replaceAllUsesWith(LV);
      ICI->eraseFromParent();
    }
    LI->eraseFromParent();
  }

  // If the initialization boolean was used, insert it, otherwise delete it.
  if (!InitBoolUsed) {
    while (!InitBool->use_empty())  // Delete initializations
      cast<StoreInst>(InitBool->user_back())->eraseFromParent();
    delete InitBool;
  } else
    GV->getParent()->getGlobalList().insert(GV->getIterator(), InitBool);

  // Now the GV is dead, nuke it and the malloc..
  GV->eraseFromParent();
  CI->eraseFromParent();

  // To further other optimizations, loop over all users of NewGV and try to
  // constant prop them.  This will promote GEP instructions with constant
  // indices into GEP constant-exprs, which will allow global-opt to hack on it.
  for (auto *CE : RepValues)
    ConstantPropUsersOf(CE, DL, TLI);

  return NewGV;
}

/// Scan the use-list of GV checking to make sure that there are no complex uses
/// of GV.  We permit simple things like dereferencing the pointer, but not
/// storing through the address, unless it is to the specified global.
static bool
valueIsOnlyUsedLocallyOrStoredToOneGlobal(const CallInst *CI,
                                          const GlobalVariable *GV) {
  SmallPtrSet<const Value *, 4> Visited;
  SmallVector<const Value *, 4> Worklist;
  Worklist.push_back(CI);

  while (!Worklist.empty()) {
    const Value *V = Worklist.pop_back_val();
    if (!Visited.insert(V).second)
      continue;

    for (const Use &VUse : V->uses()) {
      const User *U = VUse.getUser();
      if (isa<LoadInst>(U) || isa<CmpInst>(U))
        continue; // Fine, ignore.

      if (auto *SI = dyn_cast<StoreInst>(U)) {
        if (SI->getValueOperand() == V &&
            SI->getPointerOperand()->stripPointerCasts() != GV)
          return false; // Storing the pointer not into GV... bad.
        continue; // Otherwise, storing through it, or storing into GV... fine.
      }

      if (auto *BCI = dyn_cast<BitCastInst>(U)) {
        Worklist.push_back(BCI);
        continue;
      }

      if (auto *GEPI = dyn_cast<GetElementPtrInst>(U)) {
        Worklist.push_back(GEPI);
        continue;
      }

      return false;
    }
  }

  return true;
}

/// getMallocType - Returns the PointerType resulting from the malloc call.
/// The PointerType depends on the number of bitcast uses of the malloc call:
///   0: PointerType is the calls' return type.
///   1: PointerType is the bitcast's result type.
///  >1: Unique PointerType cannot be determined, return NULL.
static PointerType *getMallocType(const CallInst *CI,
                                  const TargetLibraryInfo *TLI) {
  assert(isMallocLikeFn(CI, TLI) && "getMallocType and not malloc call");

  PointerType *MallocType = nullptr;
  unsigned NumOfBitCastUses = 0;

  // Determine if CallInst has a bitcast use.
  for (const User *U : CI->users())
    if (const BitCastInst *BCI = dyn_cast<BitCastInst>(U)) {
      MallocType = cast<PointerType>(BCI->getDestTy());
      NumOfBitCastUses++;
    }

  // Malloc call has 1 bitcast use, so type is the bitcast's destination type.
  if (NumOfBitCastUses == 1)
    return MallocType;

  // Malloc call was not bitcast, so type is the malloc function's return type.
  if (NumOfBitCastUses == 0)
    return cast<PointerType>(CI->getType());

  // Type could not be determined.
  return nullptr;
}

/// getMallocAllocatedType - Returns the Type allocated by malloc call.
/// The Type depends on the number of bitcast uses of the malloc call:
///   0: PointerType is the malloc calls' return type.
///   1: PointerType is the bitcast's result type.
///  >1: Unique PointerType cannot be determined, return NULL.
static Type *getMallocAllocatedType(const CallInst *CI,
                                    const TargetLibraryInfo *TLI) {
  PointerType *PT = getMallocType(CI, TLI);
  return PT ? PT->getElementType() : nullptr;
}

static Value *computeArraySize(const CallInst *CI, const DataLayout &DL,
                               const TargetLibraryInfo *TLI,
                               bool LookThroughSExt = false) {
  if (!CI)
    return nullptr;

  // The size of the malloc's result type must be known to determine array size.
  Type *T = getMallocAllocatedType(CI, TLI);
  if (!T || !T->isSized())
    return nullptr;

  unsigned ElementSize = DL.getTypeAllocSize(T);
  if (StructType *ST = dyn_cast<StructType>(T))
    ElementSize = DL.getStructLayout(ST)->getSizeInBytes();

  // If malloc call's arg can be determined to be a multiple of ElementSize,
  // return the multiple.  Otherwise, return NULL.
  Value *MallocArg = CI->getArgOperand(0);
  Value *Multiple = nullptr;
  if (ComputeMultiple(MallocArg, ElementSize, Multiple, LookThroughSExt))
    return Multiple;

  return nullptr;
}

/// getMallocArraySize - Returns the array size of a malloc call.  If the
/// argument passed to malloc is a multiple of the size of the malloced type,
/// then return that multiple.  For non-array mallocs, the multiple is
/// constant 1.  Otherwise, return NULL for mallocs whose array size cannot be
/// determined.
static Value *getMallocArraySize(CallInst *CI, const DataLayout &DL,
                                 const TargetLibraryInfo *TLI,
                                 bool LookThroughSExt) {
  assert(isMallocLikeFn(CI, TLI) && "getMallocArraySize and not malloc call");
  return computeArraySize(CI, DL, TLI, LookThroughSExt);
}


/// This function is called when we see a pointer global variable with a single
/// value stored it that is a malloc or cast of malloc.
static bool tryToOptimizeStoreOfMallocToGlobal(GlobalVariable *GV, CallInst *CI,
                                               Type *AllocTy,
                                               AtomicOrdering Ordering,
                                               const DataLayout &DL,
                                               TargetLibraryInfo *TLI) {
  // If this is a malloc of an abstract type, don't touch it.
  if (!AllocTy->isSized())
    return false;

  // We can't optimize this global unless all uses of it are *known* to be
  // of the malloc value, not of the null initializer value (consider a use
  // that compares the global's value against zero to see if the malloc has
  // been reached).  To do this, we check to see if all uses of the global
  // would trap if the global were null: this proves that they must all
  // happen after the malloc.
  if (!allUsesOfLoadedValueWillTrapIfNull(GV))
    return false;

  // We can't optimize this if the malloc itself is used in a complex way,
  // for example, being stored into multiple globals.  This allows the
  // malloc to be stored into the specified global, loaded, gep, icmp'd.
  // These are all things we could transform to using the global for.
  if (!valueIsOnlyUsedLocallyOrStoredToOneGlobal(CI, GV))
    return false;

  // If we have a global that is only initialized with a fixed size malloc,
  // transform the program to use global memory instead of malloc'd memory.
  // This eliminates dynamic allocation, avoids an indirection accessing the
  // data, and exposes the resultant global to further GlobalOpt.
  // We cannot optimize the malloc if we cannot determine malloc array size.
  Value *NElems = getMallocArraySize(CI, DL, TLI, true);
  if (!NElems)
    return false;

  if (ConstantInt *NElements = dyn_cast<ConstantInt>(NElems))
    // Restrict this transformation to only working on small allocations
    // (2048 bytes currently), as we don't want to introduce a 16M global or
    // something.
    if (NElements->getZExtValue() * DL.getTypeAllocSize(AllocTy) < 2048) {
      OptimizeGlobalAddressOfMalloc(GV, CI, AllocTy, NElements, DL, TLI);
      return true;
    }

  return false;
}

// Try to optimize globals based on the knowledge that only one value (besides
// its initializer) is ever stored to the global.
static bool
optimizeOnceStoredGlobal(GlobalVariable *GV, Value *StoredOnceVal,
                         AtomicOrdering Ordering, const DataLayout &DL,
                         function_ref<TargetLibraryInfo &(Function &)> GetTLI) {
  // Ignore no-op GEPs and bitcasts.
  StoredOnceVal = StoredOnceVal->stripPointerCasts();

  // If we are dealing with a pointer global that is initialized to null and
  // only has one (non-null) value stored into it, then we can optimize any
  // users of the loaded value (often calls and loads) that would trap if the
  // value was null.
  if (GV->getInitializer()->getType()->isPointerTy() &&
      GV->getInitializer()->isNullValue() &&
      StoredOnceVal->getType()->isPointerTy() &&
      !NullPointerIsDefined(
          nullptr /* F */,
          GV->getInitializer()->getType()->getPointerAddressSpace())) {
    if (Constant *SOVC = dyn_cast<Constant>(StoredOnceVal)) {
      if (GV->getInitializer()->getType() != SOVC->getType())
        SOVC = ConstantExpr::getBitCast(SOVC, GV->getInitializer()->getType());

      // Optimize away any trapping uses of the loaded value.
      if (OptimizeAwayTrappingUsesOfLoads(GV, SOVC, DL, GetTLI))
        return true;
    } else if (isMallocLikeFn(StoredOnceVal, GetTLI)) {
      if (auto *CI = dyn_cast<CallInst>(StoredOnceVal)) {
        auto *TLI = &GetTLI(*CI->getFunction());
        Type *MallocType = getMallocAllocatedType(CI, TLI);
        if (MallocType && tryToOptimizeStoreOfMallocToGlobal(GV, CI, MallocType,
                                                             Ordering, DL, TLI))
          return true;
      }
    }
  }

  return false;
}

/// At this point, we have learned that the only two values ever stored into GV
/// are its initializer and OtherVal.  See if we can shrink the global into a
/// boolean and select between the two values whenever it is used.  This exposes
/// the values to other scalar optimizations.
static bool TryToShrinkGlobalToBoolean(GlobalVariable *GV, Constant *OtherVal) {
  Type *GVElType = GV->getValueType();

  // If GVElType is already i1, it is already shrunk.  If the type of the GV is
  // an FP value, pointer or vector, don't do this optimization because a select
  // between them is very expensive and unlikely to lead to later
  // simplification.  In these cases, we typically end up with "cond ? v1 : v2"
  // where v1 and v2 both require constant pool loads, a big loss.
  if (GVElType == Type::getInt1Ty(GV->getContext()) ||
      GVElType->isFloatingPointTy() ||
      GVElType->isPointerTy() || GVElType->isVectorTy())
    return false;

  // Walk the use list of the global seeing if all the uses are load or store.
  // If there is anything else, bail out.
  for (User *U : GV->users())
    if (!isa<LoadInst>(U) && !isa<StoreInst>(U))
      return false;

  LLVM_DEBUG(dbgs() << "   *** SHRINKING TO BOOL: " << *GV << "\n");

  // Create the new global, initializing it to false.
  GlobalVariable *NewGV = new GlobalVariable(Type::getInt1Ty(GV->getContext()),
                                             false,
                                             GlobalValue::InternalLinkage,
                                        ConstantInt::getFalse(GV->getContext()),
                                             GV->getName()+".b",
                                             GV->getThreadLocalMode(),
                                             GV->getType()->getAddressSpace());
  NewGV->copyAttributesFrom(GV);
  GV->getParent()->getGlobalList().insert(GV->getIterator(), NewGV);

  Constant *InitVal = GV->getInitializer();
  assert(InitVal->getType() != Type::getInt1Ty(GV->getContext()) &&
         "No reason to shrink to bool!");

  SmallVector<DIGlobalVariableExpression *, 1> GVs;
  GV->getDebugInfo(GVs);

  // If initialized to zero and storing one into the global, we can use a cast
  // instead of a select to synthesize the desired value.
  bool IsOneZero = false;
  bool EmitOneOrZero = true;
  auto *CI = dyn_cast<ConstantInt>(OtherVal);
  if (CI && CI->getValue().getActiveBits() <= 64) {
    IsOneZero = InitVal->isNullValue() && CI->isOne();

    auto *CIInit = dyn_cast<ConstantInt>(GV->getInitializer());
    if (CIInit && CIInit->getValue().getActiveBits() <= 64) {
      uint64_t ValInit = CIInit->getZExtValue();
      uint64_t ValOther = CI->getZExtValue();
      uint64_t ValMinus = ValOther - ValInit;

      for(auto *GVe : GVs){
        DIGlobalVariable *DGV = GVe->getVariable();
        DIExpression *E = GVe->getExpression();
        const DataLayout &DL = GV->getParent()->getDataLayout();
        unsigned SizeInOctets =
            DL.getTypeAllocSizeInBits(NewGV->getValueType()) / 8;

        // It is expected that the address of global optimized variable is on
        // top of the stack. After optimization, value of that variable will
        // be ether 0 for initial value or 1 for other value. The following
        // expression should return constant integer value depending on the
        // value at global object address:
        // val * (ValOther - ValInit) + ValInit:
        // DW_OP_deref DW_OP_constu <ValMinus>
        // DW_OP_mul DW_OP_constu <ValInit> DW_OP_plus DW_OP_stack_value
        SmallVector<uint64_t, 12> Ops = {
            dwarf::DW_OP_deref_size, SizeInOctets,
            dwarf::DW_OP_constu, ValMinus,
            dwarf::DW_OP_mul, dwarf::DW_OP_constu, ValInit,
            dwarf::DW_OP_plus};
        bool WithStackValue = true;
        E = DIExpression::prependOpcodes(E, Ops, WithStackValue);
        DIGlobalVariableExpression *DGVE =
          DIGlobalVariableExpression::get(NewGV->getContext(), DGV, E);
        NewGV->addDebugInfo(DGVE);
     }
     EmitOneOrZero = false;
    }
  }

  if (EmitOneOrZero) {
     // FIXME: This will only emit address for debugger on which will
     // be written only 0 or 1.
     for(auto *GV : GVs)
       NewGV->addDebugInfo(GV);
   }

  while (!GV->use_empty()) {
    Instruction *UI = cast<Instruction>(GV->user_back());
    if (StoreInst *SI = dyn_cast<StoreInst>(UI)) {
      // Change the store into a boolean store.
      bool StoringOther = SI->getOperand(0) == OtherVal;
      // Only do this if we weren't storing a loaded value.
      Value *StoreVal;
      if (StoringOther || SI->getOperand(0) == InitVal) {
        StoreVal = ConstantInt::get(Type::getInt1Ty(GV->getContext()),
                                    StoringOther);
      } else {
        // Otherwise, we are storing a previously loaded copy.  To do this,
        // change the copy from copying the original value to just copying the
        // bool.
        Instruction *StoredVal = cast<Instruction>(SI->getOperand(0));

        // If we've already replaced the input, StoredVal will be a cast or
        // select instruction.  If not, it will be a load of the original
        // global.
        if (LoadInst *LI = dyn_cast<LoadInst>(StoredVal)) {
          assert(LI->getOperand(0) == GV && "Not a copy!");
          // Insert a new load, to preserve the saved value.
          StoreVal = new LoadInst(NewGV->getValueType(), NewGV,
                                  LI->getName() + ".b", false, Align(1),
                                  LI->getOrdering(), LI->getSyncScopeID(), LI);
        } else {
          assert((isa<CastInst>(StoredVal) || isa<SelectInst>(StoredVal)) &&
                 "This is not a form that we understand!");
          StoreVal = StoredVal->getOperand(0);
          assert(isa<LoadInst>(StoreVal) && "Not a load of NewGV!");
        }
      }
      StoreInst *NSI =
          new StoreInst(StoreVal, NewGV, false, Align(1), SI->getOrdering(),
                        SI->getSyncScopeID(), SI);
      NSI->setDebugLoc(SI->getDebugLoc());
    } else {
      // Change the load into a load of bool then a select.
      LoadInst *LI = cast<LoadInst>(UI);
      LoadInst *NLI = new LoadInst(NewGV->getValueType(), NewGV,
                                   LI->getName() + ".b", false, Align(1),
                                   LI->getOrdering(), LI->getSyncScopeID(), LI);
      Instruction *NSI;
      if (IsOneZero)
        NSI = new ZExtInst(NLI, LI->getType(), "", LI);
      else
        NSI = SelectInst::Create(NLI, OtherVal, InitVal, "", LI);
      NSI->takeName(LI);
      // Since LI is split into two instructions, NLI and NSI both inherit the
      // same DebugLoc
      NLI->setDebugLoc(LI->getDebugLoc());
      NSI->setDebugLoc(LI->getDebugLoc());
      LI->replaceAllUsesWith(NSI);
    }
    UI->eraseFromParent();
  }

  // Retain the name of the old global variable. People who are debugging their
  // programs may expect these variables to be named the same.
  NewGV->takeName(GV);
  GV->eraseFromParent();
  return true;
}

static bool deleteIfDead(
    GlobalValue &GV, SmallPtrSetImpl<const Comdat *> &NotDiscardableComdats) {
  GV.removeDeadConstantUsers();

  if (!GV.isDiscardableIfUnused() && !GV.isDeclaration())
    return false;

  if (const Comdat *C = GV.getComdat())
    if (!GV.hasLocalLinkage() && NotDiscardableComdats.count(C))
      return false;

  bool Dead;
  if (auto *F = dyn_cast<Function>(&GV))
    Dead = (F->isDeclaration() && F->use_empty()) || F->isDefTriviallyDead();
  else
    Dead = GV.use_empty();
  if (!Dead)
    return false;

  LLVM_DEBUG(dbgs() << "GLOBAL DEAD: " << GV << "\n");
  GV.eraseFromParent();
  ++NumDeleted;
  return true;
}

static bool isPointerValueDeadOnEntryToFunction(
    const Function *F, GlobalValue *GV,
    function_ref<DominatorTree &(Function &)> LookupDomTree) {
  // Find all uses of GV. We expect them all to be in F, and if we can't
  // identify any of the uses we bail out.
  //
  // On each of these uses, identify if the memory that GV points to is
  // used/required/live at the start of the function. If it is not, for example
  // if the first thing the function does is store to the GV, the GV can
  // possibly be demoted.
  //
  // We don't do an exhaustive search for memory operations - simply look
  // through bitcasts as they're quite common and benign.
  const DataLayout &DL = GV->getParent()->getDataLayout();
  SmallVector<LoadInst *, 4> Loads;
  SmallVector<StoreInst *, 4> Stores;
  for (auto *U : GV->users()) {
    if (Operator::getOpcode(U) == Instruction::BitCast) {
      for (auto *UU : U->users()) {
        if (auto *LI = dyn_cast<LoadInst>(UU))
          Loads.push_back(LI);
        else if (auto *SI = dyn_cast<StoreInst>(UU))
          Stores.push_back(SI);
        else
          return false;
      }
      continue;
    }

    Instruction *I = dyn_cast<Instruction>(U);
    if (!I)
      return false;
    assert(I->getParent()->getParent() == F);

    if (auto *LI = dyn_cast<LoadInst>(I))
      Loads.push_back(LI);
    else if (auto *SI = dyn_cast<StoreInst>(I))
      Stores.push_back(SI);
    else
      return false;
  }

  // We have identified all uses of GV into loads and stores. Now check if all
  // of them are known not to depend on the value of the global at the function
  // entry point. We do this by ensuring that every load is dominated by at
  // least one store.
  auto &DT = LookupDomTree(*const_cast<Function *>(F));

  // The below check is quadratic. Check we're not going to do too many tests.
  // FIXME: Even though this will always have worst-case quadratic time, we
  // could put effort into minimizing the average time by putting stores that
  // have been shown to dominate at least one load at the beginning of the
  // Stores array, making subsequent dominance checks more likely to succeed
  // early.
  //
  // The threshold here is fairly large because global->local demotion is a
  // very powerful optimization should it fire.
  const unsigned Threshold = 100;
  if (Loads.size() * Stores.size() > Threshold)
    return false;

  for (auto *L : Loads) {
    auto *LTy = L->getType();
    if (none_of(Stores, [&](const StoreInst *S) {
          auto *STy = S->getValueOperand()->getType();
          // The load is only dominated by the store if DomTree says so
          // and the number of bits loaded in L is less than or equal to
          // the number of bits stored in S.
          return DT.dominates(S, L) &&
                 DL.getTypeStoreSize(LTy).getFixedSize() <=
                     DL.getTypeStoreSize(STy).getFixedSize();
        }))
      return false;
  }
  // All loads have known dependences inside F, so the global can be localized.
  return true;
}

/// C may have non-instruction users. Can all of those users be turned into
/// instructions?
static bool allNonInstructionUsersCanBeMadeInstructions(Constant *C) {
  // We don't do this exhaustively. The most common pattern that we really need
  // to care about is a constant GEP or constant bitcast - so just looking
  // through one single ConstantExpr.
  //
  // The set of constants that this function returns true for must be able to be
  // handled by makeAllConstantUsesInstructions.
  for (auto *U : C->users()) {
    if (isa<Instruction>(U))
      continue;
    if (!isa<ConstantExpr>(U))
      // Non instruction, non-constantexpr user; cannot convert this.
      return false;
    for (auto *UU : U->users())
      if (!isa<Instruction>(UU))
        // A constantexpr used by another constant. We don't try and recurse any
        // further but just bail out at this point.
        return false;
  }

  return true;
}

/// C may have non-instruction users, and
/// allNonInstructionUsersCanBeMadeInstructions has returned true. Convert the
/// non-instruction users to instructions.
static void makeAllConstantUsesInstructions(Constant *C) {
  SmallVector<ConstantExpr*,4> Users;
  for (auto *U : C->users()) {
    if (isa<ConstantExpr>(U))
      Users.push_back(cast<ConstantExpr>(U));
    else
      // We should never get here; allNonInstructionUsersCanBeMadeInstructions
      // should not have returned true for C.
      assert(
          isa<Instruction>(U) &&
          "Can't transform non-constantexpr non-instruction to instruction!");
  }

  SmallVector<Value*,4> UUsers;
  for (auto *U : Users) {
    UUsers.clear();
    append_range(UUsers, U->users());
    for (auto *UU : UUsers) {
      Instruction *UI = cast<Instruction>(UU);
      Instruction *NewU = U->getAsInstruction(UI);
      UI->replaceUsesOfWith(U, NewU);
    }
    // We've replaced all the uses, so destroy the constant. (destroyConstant
    // will update value handles and metadata.)
    U->destroyConstant();
  }
}

/// Analyze the specified global variable and optimize
/// it if possible.  If we make a change, return true.
static bool
processInternalGlobal(GlobalVariable *GV, const GlobalStatus &GS,
                      function_ref<TargetTransformInfo &(Function &)> GetTTI,
                      function_ref<TargetLibraryInfo &(Function &)> GetTLI,
                      function_ref<DominatorTree &(Function &)> LookupDomTree) {
  auto &DL = GV->getParent()->getDataLayout();
  // If this is a first class global and has only one accessing function and
  // this function is non-recursive, we replace the global with a local alloca
  // in this function.
  //
  // NOTE: It doesn't make sense to promote non-single-value types since we
  // are just replacing static memory to stack memory.
  //
  // If the global is in different address space, don't bring it to stack.
  if (!GS.HasMultipleAccessingFunctions &&
      GS.AccessingFunction &&
      GV->getValueType()->isSingleValueType() &&
      GV->getType()->getAddressSpace() == 0 &&
      !GV->isExternallyInitialized() &&
      allNonInstructionUsersCanBeMadeInstructions(GV) &&
      GS.AccessingFunction->doesNotRecurse() &&
      isPointerValueDeadOnEntryToFunction(GS.AccessingFunction, GV,
                                          LookupDomTree)) {
    const DataLayout &DL = GV->getParent()->getDataLayout();

    LLVM_DEBUG(dbgs() << "LOCALIZING GLOBAL: " << *GV << "\n");
    Instruction &FirstI = const_cast<Instruction&>(*GS.AccessingFunction
                                                   ->getEntryBlock().begin());
    Type *ElemTy = GV->getValueType();
    // FIXME: Pass Global's alignment when globals have alignment
    AllocaInst *Alloca = new AllocaInst(ElemTy, DL.getAllocaAddrSpace(), nullptr,
                                        GV->getName(), &FirstI);
    if (!isa<UndefValue>(GV->getInitializer()))
      new StoreInst(GV->getInitializer(), Alloca, &FirstI);

    makeAllConstantUsesInstructions(GV);

    GV->replaceAllUsesWith(Alloca);
    GV->eraseFromParent();
    ++NumLocalized;
    return true;
  }

  bool Changed = false;

  // If the global is never loaded (but may be stored to), it is dead.
  // Delete it now.
  if (!GS.IsLoaded) {
    LLVM_DEBUG(dbgs() << "GLOBAL NEVER LOADED: " << *GV << "\n");

    if (isLeakCheckerRoot(GV)) {
      // Delete any constant stores to the global.
      Changed = CleanupPointerRootUsers(GV, GetTLI);
    } else {
      // Delete any stores we can find to the global.  We may not be able to
      // make it completely dead though.
      Changed = CleanupConstantGlobalUsers(GV, DL);
    }

    // If the global is dead now, delete it.
    if (GV->use_empty()) {
      GV->eraseFromParent();
      ++NumDeleted;
      Changed = true;
    }
    return Changed;

  }
  if (GS.StoredType <= GlobalStatus::InitializerStored) {
    LLVM_DEBUG(dbgs() << "MARKING CONSTANT: " << *GV << "\n");

    // Don't actually mark a global constant if it's atomic because atomic loads
    // are implemented by a trivial cmpxchg in some edge-cases and that usually
    // requires write access to the variable even if it's not actually changed.
    if (GS.Ordering == AtomicOrdering::NotAtomic) {
      assert(!GV->isConstant() && "Expected a non-constant global");
      GV->setConstant(true);
      Changed = true;
    }

    // Clean up any obviously simplifiable users now.
    Changed |= CleanupConstantGlobalUsers(GV, DL);

    // If the global is dead now, just nuke it.
    if (GV->use_empty()) {
      LLVM_DEBUG(dbgs() << "   *** Marking constant allowed us to simplify "
                        << "all users and delete global!\n");
      GV->eraseFromParent();
      ++NumDeleted;
      return true;
    }

    // Fall through to the next check; see if we can optimize further.
    ++NumMarked;
  }
  if (!GV->getInitializer()->getType()->isSingleValueType()) {
    const DataLayout &DL = GV->getParent()->getDataLayout();
    if (SRAGlobal(GV, DL))
      return true;
  }
  Value *StoredOnceValue = GS.getStoredOnceValue();
  if (GS.StoredType == GlobalStatus::StoredOnce && StoredOnceValue) {
    // Avoid speculating constant expressions that might trap (div/rem).
    auto *SOVConstant = dyn_cast<Constant>(StoredOnceValue);
    if (SOVConstant && SOVConstant->canTrap())
      return Changed;

    Function &StoreFn =
        const_cast<Function &>(*GS.StoredOnceStore->getFunction());
    bool CanHaveNonUndefGlobalInitializer =
        GetTTI(StoreFn).canHaveNonUndefGlobalInitializerInAddressSpace(
            GV->getType()->getAddressSpace());
    // If the initial value for the global was an undef value, and if only
    // one other value was stored into it, we can just change the
    // initializer to be the stored value, then delete all stores to the
    // global.  This allows us to mark it constant.
    // This is restricted to address spaces that allow globals to have
    // initializers. NVPTX, for example, does not support initializers for
    // shared memory (AS 3).
    if (SOVConstant && isa<UndefValue>(GV->getInitializer()) &&
        DL.getTypeAllocSize(SOVConstant->getType()) ==
            DL.getTypeAllocSize(GV->getValueType()) &&
        CanHaveNonUndefGlobalInitializer) {
      if (SOVConstant->getType() == GV->getValueType()) {
        // Change the initializer in place.
        GV->setInitializer(SOVConstant);
      } else {
        // Create a new global with adjusted type.
        auto *NGV = new GlobalVariable(
            *GV->getParent(), SOVConstant->getType(), GV->isConstant(),
            GV->getLinkage(), SOVConstant, "", GV, GV->getThreadLocalMode(),
            GV->getAddressSpace());
        NGV->takeName(GV);
        NGV->copyAttributesFrom(GV);
        GV->replaceAllUsesWith(ConstantExpr::getBitCast(NGV, GV->getType()));
        GV->eraseFromParent();
        GV = NGV;
      }

      // Clean up any obviously simplifiable users now.
      CleanupConstantGlobalUsers(GV, DL);

      if (GV->use_empty()) {
        LLVM_DEBUG(dbgs() << "   *** Substituting initializer allowed us to "
                          << "simplify all users and delete global!\n");
        GV->eraseFromParent();
        ++NumDeleted;
      }
      ++NumSubstitute;
      return true;
    }

    // Try to optimize globals based on the knowledge that only one value
    // (besides its initializer) is ever stored to the global.
    if (optimizeOnceStoredGlobal(GV, StoredOnceValue, GS.Ordering, DL, GetTLI))
      return true;

    // Otherwise, if the global was not a boolean, we can shrink it to be a
    // boolean. Skip this optimization for AS that doesn't allow an initializer.
    if (SOVConstant && GS.Ordering == AtomicOrdering::NotAtomic &&
        (!isa<UndefValue>(GV->getInitializer()) ||
         CanHaveNonUndefGlobalInitializer)) {
      if (TryToShrinkGlobalToBoolean(GV, SOVConstant)) {
        ++NumShrunkToBool;
        return true;
      }
    }
  }

  return Changed;
}

/// Analyze the specified global variable and optimize it if possible.  If we
/// make a change, return true.
static bool
processGlobal(GlobalValue &GV,
              function_ref<TargetTransformInfo &(Function &)> GetTTI,
              function_ref<TargetLibraryInfo &(Function &)> GetTLI,
              function_ref<DominatorTree &(Function &)> LookupDomTree) {
  if (GV.getName().startswith("llvm."))
    return false;

  GlobalStatus GS;

  if (GlobalStatus::analyzeGlobal(&GV, GS))
    return false;

  bool Changed = false;
  if (!GS.IsCompared && !GV.hasGlobalUnnamedAddr()) {
    auto NewUnnamedAddr = GV.hasLocalLinkage() ? GlobalValue::UnnamedAddr::Global
                                               : GlobalValue::UnnamedAddr::Local;
    if (NewUnnamedAddr != GV.getUnnamedAddr()) {
      GV.setUnnamedAddr(NewUnnamedAddr);
      NumUnnamed++;
      Changed = true;
    }
  }

  // Do more involved optimizations if the global is internal.
  if (!GV.hasLocalLinkage())
    return Changed;

  auto *GVar = dyn_cast<GlobalVariable>(&GV);
  if (!GVar)
    return Changed;

  if (GVar->isConstant() || !GVar->hasInitializer())
    return Changed;

  return processInternalGlobal(GVar, GS, GetTTI, GetTLI, LookupDomTree) ||
         Changed;
}

/// Walk all of the direct calls of the specified function, changing them to
/// FastCC.
static void ChangeCalleesToFastCall(Function *F) {
  for (User *U : F->users()) {
    if (isa<BlockAddress>(U))
      continue;
    cast<CallBase>(U)->setCallingConv(CallingConv::Fast);
  }
}

static AttributeList StripAttr(LLVMContext &C, AttributeList Attrs,
                               Attribute::AttrKind A) {
  unsigned AttrIndex;
  if (Attrs.hasAttrSomewhere(A, &AttrIndex))
    return Attrs.removeAttributeAtIndex(C, AttrIndex, A);
  return Attrs;
}

static void RemoveAttribute(Function *F, Attribute::AttrKind A) {
  F->setAttributes(StripAttr(F->getContext(), F->getAttributes(), A));
  for (User *U : F->users()) {
    if (isa<BlockAddress>(U))
      continue;
    CallBase *CB = cast<CallBase>(U);
    CB->setAttributes(StripAttr(F->getContext(), CB->getAttributes(), A));
  }
}

/// Return true if this is a calling convention that we'd like to change.  The
/// idea here is that we don't want to mess with the convention if the user
/// explicitly requested something with performance implications like coldcc,
/// GHC, or anyregcc.
static bool hasChangeableCC(Function *F) {
  CallingConv::ID CC = F->getCallingConv();

  // FIXME: Is it worth transforming x86_stdcallcc and x86_fastcallcc?
  if (CC != CallingConv::C && CC != CallingConv::X86_ThisCall)
    return false;

  // FIXME: Change CC for the whole chain of musttail calls when possible.
  //
  // Can't change CC of the function that either has musttail calls, or is a
  // musttail callee itself
  for (User *U : F->users()) {
    if (isa<BlockAddress>(U))
      continue;
    CallInst* CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;

    if (CI->isMustTailCall())
      return false;
  }

  for (BasicBlock &BB : *F)
    if (BB.getTerminatingMustTailCall())
      return false;

  return true;
}

/// Return true if the block containing the call site has a BlockFrequency of
/// less than ColdCCRelFreq% of the entry block.
static bool isColdCallSite(CallBase &CB, BlockFrequencyInfo &CallerBFI) {
  const BranchProbability ColdProb(ColdCCRelFreq, 100);
  auto *CallSiteBB = CB.getParent();
  auto CallSiteFreq = CallerBFI.getBlockFreq(CallSiteBB);
  auto CallerEntryFreq =
      CallerBFI.getBlockFreq(&(CB.getCaller()->getEntryBlock()));
  return CallSiteFreq < CallerEntryFreq * ColdProb;
}

// This function checks if the input function F is cold at all call sites. It
// also looks each call site's containing function, returning false if the
// caller function contains other non cold calls. The input vector AllCallsCold
// contains a list of functions that only have call sites in cold blocks.
static bool
isValidCandidateForColdCC(Function &F,
                          function_ref<BlockFrequencyInfo &(Function &)> GetBFI,
                          const std::vector<Function *> &AllCallsCold) {

  if (F.user_empty())
    return false;

  for (User *U : F.users()) {
    if (isa<BlockAddress>(U))
      continue;

    CallBase &CB = cast<CallBase>(*U);
    Function *CallerFunc = CB.getParent()->getParent();
    BlockFrequencyInfo &CallerBFI = GetBFI(*CallerFunc);
    if (!isColdCallSite(CB, CallerBFI))
      return false;
    if (!llvm::is_contained(AllCallsCold, CallerFunc))
      return false;
  }
  return true;
}

static void changeCallSitesToColdCC(Function *F) {
  for (User *U : F->users()) {
    if (isa<BlockAddress>(U))
      continue;
    cast<CallBase>(U)->setCallingConv(CallingConv::Cold);
  }
}

// This function iterates over all the call instructions in the input Function
// and checks that all call sites are in cold blocks and are allowed to use the
// coldcc calling convention.
static bool
hasOnlyColdCalls(Function &F,
                 function_ref<BlockFrequencyInfo &(Function &)> GetBFI) {
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (CallInst *CI = dyn_cast<CallInst>(&I)) {
        // Skip over isline asm instructions since they aren't function calls.
        if (CI->isInlineAsm())
          continue;
        Function *CalledFn = CI->getCalledFunction();
        if (!CalledFn)
          return false;
        if (!CalledFn->hasLocalLinkage())
          return false;
        // Skip over instrinsics since they won't remain as function calls.
        if (CalledFn->getIntrinsicID() != Intrinsic::not_intrinsic)
          continue;
        // Check if it's valid to use coldcc calling convention.
        if (!hasChangeableCC(CalledFn) || CalledFn->isVarArg() ||
            CalledFn->hasAddressTaken())
          return false;
        BlockFrequencyInfo &CallerBFI = GetBFI(F);
        if (!isColdCallSite(*CI, CallerBFI))
          return false;
      }
    }
  }
  return true;
}

static bool hasMustTailCallers(Function *F) {
  for (User *U : F->users()) {
    CallBase *CB = dyn_cast<CallBase>(U);
    if (!CB) {
      assert(isa<BlockAddress>(U) &&
             "Expected either CallBase or BlockAddress");
      continue;
    }
    if (CB->isMustTailCall())
      return true;
  }
  return false;
}

static bool hasInvokeCallers(Function *F) {
  for (User *U : F->users())
    if (isa<InvokeInst>(U))
      return true;
  return false;
}

static void RemovePreallocated(Function *F) {
  RemoveAttribute(F, Attribute::Preallocated);

  auto *M = F->getParent();

  IRBuilder<> Builder(M->getContext());

  // Cannot modify users() while iterating over it, so make a copy.
  SmallVector<User *, 4> PreallocatedCalls(F->users());
  for (User *U : PreallocatedCalls) {
    CallBase *CB = dyn_cast<CallBase>(U);
    if (!CB)
      continue;

    assert(
        !CB->isMustTailCall() &&
        "Shouldn't call RemotePreallocated() on a musttail preallocated call");
    // Create copy of call without "preallocated" operand bundle.
    SmallVector<OperandBundleDef, 1> OpBundles;
    CB->getOperandBundlesAsDefs(OpBundles);
    CallBase *PreallocatedSetup = nullptr;
    for (auto *It = OpBundles.begin(); It != OpBundles.end(); ++It) {
      if (It->getTag() == "preallocated") {
        PreallocatedSetup = cast<CallBase>(*It->input_begin());
        OpBundles.erase(It);
        break;
      }
    }
    assert(PreallocatedSetup && "Did not find preallocated bundle");
    uint64_t ArgCount =
        cast<ConstantInt>(PreallocatedSetup->getArgOperand(0))->getZExtValue();

    assert((isa<CallInst>(CB) || isa<InvokeInst>(CB)) &&
           "Unknown indirect call type");
    CallBase *NewCB = CallBase::Create(CB, OpBundles, CB);
    CB->replaceAllUsesWith(NewCB);
    NewCB->takeName(CB);
    CB->eraseFromParent();

    Builder.SetInsertPoint(PreallocatedSetup);
    auto *StackSave =
        Builder.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::stacksave));

    Builder.SetInsertPoint(NewCB->getNextNonDebugInstruction());
    Builder.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::stackrestore),
                       StackSave);

    // Replace @llvm.call.preallocated.arg() with alloca.
    // Cannot modify users() while iterating over it, so make a copy.
    // @llvm.call.preallocated.arg() can be called with the same index multiple
    // times. So for each @llvm.call.preallocated.arg(), we see if we have
    // already created a Value* for the index, and if not, create an alloca and
    // bitcast right after the @llvm.call.preallocated.setup() so that it
    // dominates all uses.
    SmallVector<Value *, 2> ArgAllocas(ArgCount);
    SmallVector<User *, 2> PreallocatedArgs(PreallocatedSetup->users());
    for (auto *User : PreallocatedArgs) {
      auto *UseCall = cast<CallBase>(User);
      assert(UseCall->getCalledFunction()->getIntrinsicID() ==
                 Intrinsic::call_preallocated_arg &&
             "preallocated token use was not a llvm.call.preallocated.arg");
      uint64_t AllocArgIndex =
          cast<ConstantInt>(UseCall->getArgOperand(1))->getZExtValue();
      Value *AllocaReplacement = ArgAllocas[AllocArgIndex];
      if (!AllocaReplacement) {
        auto AddressSpace = UseCall->getType()->getPointerAddressSpace();
        auto *ArgType =
            UseCall->getFnAttr(Attribute::Preallocated).getValueAsType();
        auto *InsertBefore = PreallocatedSetup->getNextNonDebugInstruction();
        Builder.SetInsertPoint(InsertBefore);
        auto *Alloca =
            Builder.CreateAlloca(ArgType, AddressSpace, nullptr, "paarg");
        auto *BitCast = Builder.CreateBitCast(
            Alloca, Type::getInt8PtrTy(M->getContext()), UseCall->getName());
        ArgAllocas[AllocArgIndex] = BitCast;
        AllocaReplacement = BitCast;
      }

      UseCall->replaceAllUsesWith(AllocaReplacement);
      UseCall->eraseFromParent();
    }
    // Remove @llvm.call.preallocated.setup().
    cast<Instruction>(PreallocatedSetup)->eraseFromParent();
  }
}

static bool
OptimizeFunctions(Module &M,
                  function_ref<TargetLibraryInfo &(Function &)> GetTLI,
                  function_ref<TargetTransformInfo &(Function &)> GetTTI,
                  function_ref<BlockFrequencyInfo &(Function &)> GetBFI,
                  function_ref<DominatorTree &(Function &)> LookupDomTree,
                  SmallPtrSetImpl<const Comdat *> &NotDiscardableComdats) {

  bool Changed = false;

  std::vector<Function *> AllCallsCold;
  for (Function &F : llvm::make_early_inc_range(M))
    if (hasOnlyColdCalls(F, GetBFI))
      AllCallsCold.push_back(&F);

  // Optimize functions.
  for (Function &F : llvm::make_early_inc_range(M)) {
    // Don't perform global opt pass on naked functions; we don't want fast
    // calling conventions for naked functions.
    if (F.hasFnAttribute(Attribute::Naked))
      continue;

    // Functions without names cannot be referenced outside this module.
    if (!F.hasName() && !F.isDeclaration() && !F.hasLocalLinkage())
      F.setLinkage(GlobalValue::InternalLinkage);

    if (deleteIfDead(F, NotDiscardableComdats)) {
      Changed = true;
      continue;
    }

    // LLVM's definition of dominance allows instructions that are cyclic
    // in unreachable blocks, e.g.:
    // %pat = select i1 %condition, @global, i16* %pat
    // because any instruction dominates an instruction in a block that's
    // not reachable from entry.
    // So, remove unreachable blocks from the function, because a) there's
    // no point in analyzing them and b) GlobalOpt should otherwise grow
    // some more complicated logic to break these cycles.
    // Removing unreachable blocks might invalidate the dominator so we
    // recalculate it.
    if (!F.isDeclaration()) {
      if (removeUnreachableBlocks(F)) {
        auto &DT = LookupDomTree(F);
        DT.recalculate(F);
        Changed = true;
      }
    }

    Changed |= processGlobal(F, GetTTI, GetTLI, LookupDomTree);

    if (!F.hasLocalLinkage())
      continue;

    // If we have an inalloca parameter that we can safely remove the
    // inalloca attribute from, do so. This unlocks optimizations that
    // wouldn't be safe in the presence of inalloca.
    // FIXME: We should also hoist alloca affected by this to the entry
    // block if possible.
    if (F.getAttributes().hasAttrSomewhere(Attribute::InAlloca) &&
        !F.hasAddressTaken() && !hasMustTailCallers(&F)) {
      RemoveAttribute(&F, Attribute::InAlloca);
      Changed = true;
    }

    // FIXME: handle invokes
    // FIXME: handle musttail
    if (F.getAttributes().hasAttrSomewhere(Attribute::Preallocated)) {
      if (!F.hasAddressTaken() && !hasMustTailCallers(&F) &&
          !hasInvokeCallers(&F)) {
        RemovePreallocated(&F);
        Changed = true;
      }
      continue;
    }

    if (hasChangeableCC(&F) && !F.isVarArg() && !F.hasAddressTaken()) {
      NumInternalFunc++;
      TargetTransformInfo &TTI = GetTTI(F);
      // Change the calling convention to coldcc if either stress testing is
      // enabled or the target would like to use coldcc on functions which are
      // cold at all call sites and the callers contain no other non coldcc
      // calls.
      if (EnableColdCCStressTest ||
          (TTI.useColdCCForColdCall(F) &&
           isValidCandidateForColdCC(F, GetBFI, AllCallsCold))) {
        F.setCallingConv(CallingConv::Cold);
        changeCallSitesToColdCC(&F);
        Changed = true;
        NumColdCC++;
      }
    }

    if (hasChangeableCC(&F) && !F.isVarArg() && !F.hasAddressTaken()) {
      // If this function has a calling convention worth changing, is not a
      // varargs function, and is only called directly, promote it to use the
      // Fast calling convention.
      F.setCallingConv(CallingConv::Fast);
      ChangeCalleesToFastCall(&F);
      ++NumFastCallFns;
      Changed = true;
    }

    if (F.getAttributes().hasAttrSomewhere(Attribute::Nest) &&
        !F.hasAddressTaken()) {
      // The function is not used by a trampoline intrinsic, so it is safe
      // to remove the 'nest' attribute.
      RemoveAttribute(&F, Attribute::Nest);
      ++NumNestRemoved;
      Changed = true;
    }
  }
  return Changed;
}

static bool
OptimizeGlobalVars(Module &M,
                   function_ref<TargetTransformInfo &(Function &)> GetTTI,
                   function_ref<TargetLibraryInfo &(Function &)> GetTLI,
                   function_ref<DominatorTree &(Function &)> LookupDomTree,
                   SmallPtrSetImpl<const Comdat *> &NotDiscardableComdats) {
  bool Changed = false;

  for (GlobalVariable &GV : llvm::make_early_inc_range(M.globals())) {
    // Global variables without names cannot be referenced outside this module.
    if (!GV.hasName() && !GV.isDeclaration() && !GV.hasLocalLinkage())
      GV.setLinkage(GlobalValue::InternalLinkage);
    // Simplify the initializer.
    if (GV.hasInitializer())
      if (auto *C = dyn_cast<Constant>(GV.getInitializer())) {
        auto &DL = M.getDataLayout();
        // TLI is not used in the case of a Constant, so use default nullptr
        // for that optional parameter, since we don't have a Function to
        // provide GetTLI anyway.
        Constant *New = ConstantFoldConstant(C, DL, /*TLI*/ nullptr);
        if (New != C)
          GV.setInitializer(New);
      }

    if (deleteIfDead(GV, NotDiscardableComdats)) {
      Changed = true;
      continue;
    }

    Changed |= processGlobal(GV, GetTTI, GetTLI, LookupDomTree);
  }
  return Changed;
}

/// Evaluate static constructors in the function, if we can.  Return true if we
/// can, false otherwise.
static bool EvaluateStaticConstructor(Function *F, const DataLayout &DL,
                                      TargetLibraryInfo *TLI) {
  // Call the function.
  Evaluator Eval(DL, TLI);
  Constant *RetValDummy;
  bool EvalSuccess = Eval.EvaluateFunction(F, RetValDummy,
                                           SmallVector<Constant*, 0>());

  if (EvalSuccess) {
    ++NumCtorsEvaluated;

    // We succeeded at evaluation: commit the result.
    auto NewInitializers = Eval.getMutatedInitializers();
    LLVM_DEBUG(dbgs() << "FULLY EVALUATED GLOBAL CTOR FUNCTION '"
                      << F->getName() << "' to " << NewInitializers.size()
                      << " stores.\n");
    for (const auto &Pair : NewInitializers)
      Pair.first->setInitializer(Pair.second);
    for (GlobalVariable *GV : Eval.getInvariants())
      GV->setConstant(true);
  }

  return EvalSuccess;
}

static int compareNames(Constant *const *A, Constant *const *B) {
  Value *AStripped = (*A)->stripPointerCasts();
  Value *BStripped = (*B)->stripPointerCasts();
  return AStripped->getName().compare(BStripped->getName());
}

static void setUsedInitializer(GlobalVariable &V,
                               const SmallPtrSetImpl<GlobalValue *> &Init) {
  if (Init.empty()) {
    V.eraseFromParent();
    return;
  }

  // Type of pointer to the array of pointers.
  PointerType *Int8PtrTy = Type::getInt8PtrTy(V.getContext(), 0);

  SmallVector<Constant *, 8> UsedArray;
  for (GlobalValue *GV : Init) {
    Constant *Cast
      = ConstantExpr::getPointerBitCastOrAddrSpaceCast(GV, Int8PtrTy);
    UsedArray.push_back(Cast);
  }
  // Sort to get deterministic order.
  array_pod_sort(UsedArray.begin(), UsedArray.end(), compareNames);
  ArrayType *ATy = ArrayType::get(Int8PtrTy, UsedArray.size());

  Module *M = V.getParent();
  V.removeFromParent();
  GlobalVariable *NV =
      new GlobalVariable(*M, ATy, false, GlobalValue::AppendingLinkage,
                         ConstantArray::get(ATy, UsedArray), "");
  NV->takeName(&V);
  NV->setSection("llvm.metadata");
  delete &V;
}

namespace {

/// An easy to access representation of llvm.used and llvm.compiler.used.
class LLVMUsed {
  SmallPtrSet<GlobalValue *, 4> Used;
  SmallPtrSet<GlobalValue *, 4> CompilerUsed;
  GlobalVariable *UsedV;
  GlobalVariable *CompilerUsedV;

public:
  LLVMUsed(Module &M) {
    SmallVector<GlobalValue *, 4> Vec;
    UsedV = collectUsedGlobalVariables(M, Vec, false);
    Used = {Vec.begin(), Vec.end()};
    Vec.clear();
    CompilerUsedV = collectUsedGlobalVariables(M, Vec, true);
    CompilerUsed = {Vec.begin(), Vec.end()};
  }

  using iterator = SmallPtrSet<GlobalValue *, 4>::iterator;
  using used_iterator_range = iterator_range<iterator>;

  iterator usedBegin() { return Used.begin(); }
  iterator usedEnd() { return Used.end(); }

  used_iterator_range used() {
    return used_iterator_range(usedBegin(), usedEnd());
  }

  iterator compilerUsedBegin() { return CompilerUsed.begin(); }
  iterator compilerUsedEnd() { return CompilerUsed.end(); }

  used_iterator_range compilerUsed() {
    return used_iterator_range(compilerUsedBegin(), compilerUsedEnd());
  }

  bool usedCount(GlobalValue *GV) const { return Used.count(GV); }

  bool compilerUsedCount(GlobalValue *GV) const {
    return CompilerUsed.count(GV);
  }

  bool usedErase(GlobalValue *GV) { return Used.erase(GV); }
  bool compilerUsedErase(GlobalValue *GV) { return CompilerUsed.erase(GV); }
  bool usedInsert(GlobalValue *GV) { return Used.insert(GV).second; }

  bool compilerUsedInsert(GlobalValue *GV) {
    return CompilerUsed.insert(GV).second;
  }

  void syncVariablesAndSets() {
    if (UsedV)
      setUsedInitializer(*UsedV, Used);
    if (CompilerUsedV)
      setUsedInitializer(*CompilerUsedV, CompilerUsed);
  }
};

} // end anonymous namespace

static bool hasUseOtherThanLLVMUsed(GlobalAlias &GA, const LLVMUsed &U) {
  if (GA.use_empty()) // No use at all.
    return false;

  assert((!U.usedCount(&GA) || !U.compilerUsedCount(&GA)) &&
         "We should have removed the duplicated "
         "element from llvm.compiler.used");
  if (!GA.hasOneUse())
    // Strictly more than one use. So at least one is not in llvm.used and
    // llvm.compiler.used.
    return true;

  // Exactly one use. Check if it is in llvm.used or llvm.compiler.used.
  return !U.usedCount(&GA) && !U.compilerUsedCount(&GA);
}

static bool hasMoreThanOneUseOtherThanLLVMUsed(GlobalValue &V,
                                               const LLVMUsed &U) {
  unsigned N = 2;
  assert((!U.usedCount(&V) || !U.compilerUsedCount(&V)) &&
         "We should have removed the duplicated "
         "element from llvm.compiler.used");
  if (U.usedCount(&V) || U.compilerUsedCount(&V))
    ++N;
  return V.hasNUsesOrMore(N);
}

static bool mayHaveOtherReferences(GlobalAlias &GA, const LLVMUsed &U) {
  if (!GA.hasLocalLinkage())
    return true;

  return U.usedCount(&GA) || U.compilerUsedCount(&GA);
}

static bool hasUsesToReplace(GlobalAlias &GA, const LLVMUsed &U,
                             bool &RenameTarget) {
  RenameTarget = false;
  bool Ret = false;
  if (hasUseOtherThanLLVMUsed(GA, U))
    Ret = true;

  // If the alias is externally visible, we may still be able to simplify it.
  if (!mayHaveOtherReferences(GA, U))
    return Ret;

  // If the aliasee has internal linkage, give it the name and linkage
  // of the alias, and delete the alias.  This turns:
  //   define internal ... @f(...)
  //   @a = alias ... @f
  // into:
  //   define ... @a(...)
  Constant *Aliasee = GA.getAliasee();
  GlobalValue *Target = cast<GlobalValue>(Aliasee->stripPointerCasts());
  if (!Target->hasLocalLinkage())
    return Ret;

  // Do not perform the transform if multiple aliases potentially target the
  // aliasee. This check also ensures that it is safe to replace the section
  // and other attributes of the aliasee with those of the alias.
  if (hasMoreThanOneUseOtherThanLLVMUsed(*Target, U))
    return Ret;

  RenameTarget = true;
  return true;
}

static bool
OptimizeGlobalAliases(Module &M,
                      SmallPtrSetImpl<const Comdat *> &NotDiscardableComdats) {
  bool Changed = false;
  LLVMUsed Used(M);

  for (GlobalValue *GV : Used.used())
    Used.compilerUsedErase(GV);

  for (GlobalAlias &J : llvm::make_early_inc_range(M.aliases())) {
    // Aliases without names cannot be referenced outside this module.
    if (!J.hasName() && !J.isDeclaration() && !J.hasLocalLinkage())
      J.setLinkage(GlobalValue::InternalLinkage);

    if (deleteIfDead(J, NotDiscardableComdats)) {
      Changed = true;
      continue;
    }

    // If the alias can change at link time, nothing can be done - bail out.
    if (J.isInterposable())
      continue;

    Constant *Aliasee = J.getAliasee();
    GlobalValue *Target = dyn_cast<GlobalValue>(Aliasee->stripPointerCasts());
    // We can't trivially replace the alias with the aliasee if the aliasee is
    // non-trivial in some way. We also can't replace the alias with the aliasee
    // if the aliasee is interposable because aliases point to the local
    // definition.
    // TODO: Try to handle non-zero GEPs of local aliasees.
    if (!Target || Target->isInterposable())
      continue;
    Target->removeDeadConstantUsers();

    // Make all users of the alias use the aliasee instead.
    bool RenameTarget;
    if (!hasUsesToReplace(J, Used, RenameTarget))
      continue;

    J.replaceAllUsesWith(ConstantExpr::getBitCast(Aliasee, J.getType()));
    ++NumAliasesResolved;
    Changed = true;

    if (RenameTarget) {
      // Give the aliasee the name, linkage and other attributes of the alias.
      Target->takeName(&J);
      Target->setLinkage(J.getLinkage());
      Target->setDSOLocal(J.isDSOLocal());
      Target->setVisibility(J.getVisibility());
      Target->setDLLStorageClass(J.getDLLStorageClass());

      if (Used.usedErase(&J))
        Used.usedInsert(Target);

      if (Used.compilerUsedErase(&J))
        Used.compilerUsedInsert(Target);
    } else if (mayHaveOtherReferences(J, Used))
      continue;

    // Delete the alias.
    M.getAliasList().erase(&J);
    ++NumAliasesRemoved;
    Changed = true;
  }

  Used.syncVariablesAndSets();

  return Changed;
}

static Function *
FindCXAAtExit(Module &M, function_ref<TargetLibraryInfo &(Function &)> GetTLI) {
  // Hack to get a default TLI before we have actual Function.
  auto FuncIter = M.begin();
  if (FuncIter == M.end())
    return nullptr;
  auto *TLI = &GetTLI(*FuncIter);

  LibFunc F = LibFunc_cxa_atexit;
  if (!TLI->has(F))
    return nullptr;

  Function *Fn = M.getFunction(TLI->getName(F));
  if (!Fn)
    return nullptr;

  // Now get the actual TLI for Fn.
  TLI = &GetTLI(*Fn);

  // Make sure that the function has the correct prototype.
  if (!TLI->getLibFunc(*Fn, F) || F != LibFunc_cxa_atexit)
    return nullptr;

  return Fn;
}

/// Returns whether the given function is an empty C++ destructor and can
/// therefore be eliminated.
/// Note that we assume that other optimization passes have already simplified
/// the code so we simply check for 'ret'.
static bool cxxDtorIsEmpty(const Function &Fn) {
  // FIXME: We could eliminate C++ destructors if they're readonly/readnone and
  // nounwind, but that doesn't seem worth doing.
  if (Fn.isDeclaration())
    return false;

  for (auto &I : Fn.getEntryBlock()) {
    if (I.isDebugOrPseudoInst())
      continue;
    if (isa<ReturnInst>(I))
      return true;
    break;
  }
  return false;
}

static bool OptimizeEmptyGlobalCXXDtors(Function *CXAAtExitFn) {
  /// Itanium C++ ABI p3.3.5:
  ///
  ///   After constructing a global (or local static) object, that will require
  ///   destruction on exit, a termination function is registered as follows:
  ///
  ///   extern "C" int __cxa_atexit ( void (*f)(void *), void *p, void *d );
  ///
  ///   This registration, e.g. __cxa_atexit(f,p,d), is intended to cause the
  ///   call f(p) when DSO d is unloaded, before all such termination calls
  ///   registered before this one. It returns zero if registration is
  ///   successful, nonzero on failure.

  // This pass will look for calls to __cxa_atexit where the function is trivial
  // and remove them.
  bool Changed = false;

  for (User *U : llvm::make_early_inc_range(CXAAtExitFn->users())) {
    // We're only interested in calls. Theoretically, we could handle invoke
    // instructions as well, but neither llvm-gcc nor clang generate invokes
    // to __cxa_atexit.
    CallInst *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;

    Function *DtorFn =
      dyn_cast<Function>(CI->getArgOperand(0)->stripPointerCasts());
    if (!DtorFn || !cxxDtorIsEmpty(*DtorFn))
      continue;

    // Just remove the call.
    CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
    CI->eraseFromParent();

    ++NumCXXDtorsRemoved;

    Changed |= true;
  }

  return Changed;
}

static bool optimizeGlobalsInModule(
    Module &M, const DataLayout &DL,
    function_ref<TargetLibraryInfo &(Function &)> GetTLI,
    function_ref<TargetTransformInfo &(Function &)> GetTTI,
    function_ref<BlockFrequencyInfo &(Function &)> GetBFI,
    function_ref<DominatorTree &(Function &)> LookupDomTree) {
  SmallPtrSet<const Comdat *, 8> NotDiscardableComdats;
  bool Changed = false;
  bool LocalChange = true;
  while (LocalChange) {
    LocalChange = false;

    NotDiscardableComdats.clear();
    for (const GlobalVariable &GV : M.globals())
      if (const Comdat *C = GV.getComdat())
        if (!GV.isDiscardableIfUnused() || !GV.use_empty())
          NotDiscardableComdats.insert(C);
    for (Function &F : M)
      if (const Comdat *C = F.getComdat())
        if (!F.isDefTriviallyDead())
          NotDiscardableComdats.insert(C);
    for (GlobalAlias &GA : M.aliases())
      if (const Comdat *C = GA.getComdat())
        if (!GA.isDiscardableIfUnused() || !GA.use_empty())
          NotDiscardableComdats.insert(C);

    // Delete functions that are trivially dead, ccc -> fastcc
    LocalChange |= OptimizeFunctions(M, GetTLI, GetTTI, GetBFI, LookupDomTree,
                                     NotDiscardableComdats);

    // Optimize global_ctors list.
    LocalChange |= optimizeGlobalCtorsList(M, [&](Function *F) {
      return EvaluateStaticConstructor(F, DL, &GetTLI(*F));
    });

    // Optimize non-address-taken globals.
    LocalChange |= OptimizeGlobalVars(M, GetTTI, GetTLI, LookupDomTree,
                                      NotDiscardableComdats);

    // Resolve aliases, when possible.
    LocalChange |= OptimizeGlobalAliases(M, NotDiscardableComdats);

    // Try to remove trivial global destructors if they are not removed
    // already.
    Function *CXAAtExitFn = FindCXAAtExit(M, GetTLI);
    if (CXAAtExitFn)
      LocalChange |= OptimizeEmptyGlobalCXXDtors(CXAAtExitFn);

    Changed |= LocalChange;
  }

  // TODO: Move all global ctors functions to the end of the module for code
  // layout.

  return Changed;
}

PreservedAnalyses GlobalOptPass::run(Module &M, ModuleAnalysisManager &AM) {
    auto &DL = M.getDataLayout();
    auto &FAM =
        AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    auto LookupDomTree = [&FAM](Function &F) -> DominatorTree &{
      return FAM.getResult<DominatorTreeAnalysis>(F);
    };
    auto GetTLI = [&FAM](Function &F) -> TargetLibraryInfo & {
      return FAM.getResult<TargetLibraryAnalysis>(F);
    };
    auto GetTTI = [&FAM](Function &F) -> TargetTransformInfo & {
      return FAM.getResult<TargetIRAnalysis>(F);
    };

    auto GetBFI = [&FAM](Function &F) -> BlockFrequencyInfo & {
      return FAM.getResult<BlockFrequencyAnalysis>(F);
    };

    if (!optimizeGlobalsInModule(M, DL, GetTLI, GetTTI, GetBFI, LookupDomTree))
      return PreservedAnalyses::all();
    return PreservedAnalyses::none();
}

namespace {

struct GlobalOptLegacyPass : public ModulePass {
  static char ID; // Pass identification, replacement for typeid

  GlobalOptLegacyPass() : ModulePass(ID) {
    initializeGlobalOptLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;

    auto &DL = M.getDataLayout();
    auto LookupDomTree = [this](Function &F) -> DominatorTree & {
      return this->getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
    };
    auto GetTLI = [this](Function &F) -> TargetLibraryInfo & {
      return this->getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    };
    auto GetTTI = [this](Function &F) -> TargetTransformInfo & {
      return this->getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    };

    auto GetBFI = [this](Function &F) -> BlockFrequencyInfo & {
      return this->getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
    };

    return optimizeGlobalsInModule(M, DL, GetTLI, GetTTI, GetBFI,
                                   LookupDomTree);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
  }
};

} // end anonymous namespace

char GlobalOptLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(GlobalOptLegacyPass, "globalopt",
                      "Global Variable Optimizer", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(BlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(GlobalOptLegacyPass, "globalopt",
                    "Global Variable Optimizer", false, false)

ModulePass *llvm::createGlobalOptimizerPass() {
  return new GlobalOptLegacyPass();
}
