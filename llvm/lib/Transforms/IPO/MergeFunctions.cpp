//===- MergeFunctions.cpp - Merge identical functions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass looks for equivalent functions that are mergable and folds them.
//
// A hash is computed from the function, based on its type and number of
// basic blocks.
//
// Once all hashes are computed, we perform an expensive equality comparison
// on each function pair. This takes n^2/2 comparisons per bucket, so it's
// important that the hash function be high quality. The equality comparison
// iterates through each instruction in each basic block.
//
// When a match is found the functions are folded. If both functions are
// overridable, we move the functionality into a new internal function and
// leave two overridable thunks to it.
//
//===----------------------------------------------------------------------===//
//
// Future work:
//
// * virtual functions.
//
// Many functions have their address taken by the virtual function table for
// the object they belong to. However, as long as it's only used for a lookup
// and call, this is irrelevant, and we'd like to fold such functions.
//
// * switch from n^2 pair-wise comparisons to an n-way comparison for each
// bucket.
//
// * be smarter about bitcasts.
//
// In order to fold functions, we will sometimes add either bitcast instructions
// or bitcast constant expressions. Unfortunately, this can confound further
// analysis since the two functions differ where one has a bitcast and the
// other doesn't. We should learn to look through bitcasts.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mergefunc"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Constants.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetData.h"
#include <vector>
using namespace llvm;

STATISTIC(NumFunctionsMerged, "Number of functions merged");
STATISTIC(NumThunksWritten, "Number of thunks generated");
STATISTIC(NumAliasesWritten, "Number of aliases generated");
STATISTIC(NumDoubleWeak, "Number of new functions created");

/// ProfileFunction - Creates a hash-code for the function which is the same
/// for any two functions that will compare equal, without looking at the
/// instructions inside the function.
static unsigned ProfileFunction(const Function *F) {
  const FunctionType *FTy = F->getFunctionType();

  FoldingSetNodeID ID;
  ID.AddInteger(F->size());
  ID.AddInteger(F->getCallingConv());
  ID.AddBoolean(F->hasGC());
  ID.AddBoolean(FTy->isVarArg());
  ID.AddInteger(FTy->getReturnType()->getTypeID());
  for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i)
    ID.AddInteger(FTy->getParamType(i)->getTypeID());
  return ID.ComputeHash();
}

namespace {

class ComparableFunction {
public:
  static const ComparableFunction EmptyKey;
  static const ComparableFunction TombstoneKey;

  ComparableFunction(Function *Func, TargetData *TD)
    : Func(Func), Hash(ProfileFunction(Func)), TD(TD) {}

  Function *getFunc() const { return Func; }
  unsigned getHash() const { return Hash; }
  TargetData *getTD() const { return TD; }

  // Drops AssertingVH reference to the function. Outside of debug mode, this
  // does nothing.
  void release() {
    assert(Func &&
           "Attempted to release function twice, or release empty/tombstone!");
    Func = NULL;
  }

  bool &getOrInsertCachedComparison(const ComparableFunction &Other,
                                    bool &inserted) const {
    typedef DenseMap<Function *, bool>::iterator iterator;
    std::pair<iterator, bool> p =
        CompareResultCache.insert(std::make_pair(Other.getFunc(), false));
    inserted = p.second;
    return p.first->second;
  }

private:
  explicit ComparableFunction(unsigned Hash)
    : Func(NULL), Hash(Hash), TD(NULL) {}

  // DenseMap::grow() triggers a recomparison of all keys in the map, which is
  // wildly expensive. This cache tries to preserve known results.
  mutable DenseMap<Function *, bool> CompareResultCache;

  AssertingVH<Function> Func;
  unsigned Hash;
  TargetData *TD;
};

const ComparableFunction ComparableFunction::EmptyKey = ComparableFunction(0);
const ComparableFunction ComparableFunction::TombstoneKey =
    ComparableFunction(1);

}

namespace llvm {
  template <>
  struct DenseMapInfo<ComparableFunction> {
    static ComparableFunction getEmptyKey() {
      return ComparableFunction::EmptyKey;
    }
    static ComparableFunction getTombstoneKey() {
      return ComparableFunction::TombstoneKey;
    }
    static unsigned getHashValue(const ComparableFunction &CF) {
      return CF.getHash();
    }
    static bool isEqual(const ComparableFunction &LHS,
                        const ComparableFunction &RHS);
  };
}

namespace {

/// MergeFunctions finds functions which will generate identical machine code,
/// by considering all pointer types to be equivalent. Once identified,
/// MergeFunctions will fold them by replacing a call to one to a call to a
/// bitcast of the other.
///
class MergeFunctions : public ModulePass {
public:
  static char ID;
  MergeFunctions()
    : ModulePass(ID), HasGlobalAliases(false) {
    initializeMergeFunctionsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M);

private:
  typedef DenseSet<ComparableFunction> FnSetType;

  /// A work queue of functions that may have been modified and should be
  /// analyzed again.
  std::vector<WeakVH> Deferred;

  /// Insert a ComparableFunction into the FnSet, or merge it away if it's
  /// equal to one that's already present.
  bool Insert(ComparableFunction &NewF);

  /// Remove a Function from the FnSet and queue it up for a second sweep of
  /// analysis.
  void Remove(Function *F);

  /// Find the functions that use this Value and remove them from FnSet and
  /// queue the functions.
  void RemoveUsers(Value *V);

  /// Replace all direct calls of Old with calls of New. Will bitcast New if
  /// necessary to make types match.
  void replaceDirectCallers(Function *Old, Function *New);

  /// MergeTwoFunctions - Merge two equivalent functions. Upon completion, G
  /// may be deleted, or may be converted into a thunk. In either case, it
  /// should never be visited again.
  void MergeTwoFunctions(Function *F, Function *G);

  /// WriteThunkOrAlias - Replace G with a thunk or an alias to F. Deletes G.
  void WriteThunkOrAlias(Function *F, Function *G);

  /// WriteThunk - Replace G with a simple tail call to bitcast(F). Also
  /// replace direct uses of G with bitcast(F). Deletes G.
  void WriteThunk(Function *F, Function *G);

  /// WriteAlias - Replace G with an alias to F. Deletes G.
  void WriteAlias(Function *F, Function *G);

  /// The set of all distinct functions. Use the Insert and Remove methods to
  /// modify it.
  FnSetType FnSet;

  /// TargetData for more accurate GEP comparisons. May be NULL.
  TargetData *TD;

  /// Whether or not the target supports global aliases.
  bool HasGlobalAliases;
};

}  // end anonymous namespace

char MergeFunctions::ID = 0;
INITIALIZE_PASS(MergeFunctions, "mergefunc", "Merge Functions", false, false)

ModulePass *llvm::createMergeFunctionsPass() {
  return new MergeFunctions();
}

namespace {
/// FunctionComparator - Compares two functions to determine whether or not
/// they will generate machine code with the same behaviour. TargetData is
/// used if available. The comparator always fails conservatively (erring on the
/// side of claiming that two functions are different).
class FunctionComparator {
public:
  FunctionComparator(const TargetData *TD, const Function *F1,
                     const Function *F2)
    : F1(F1), F2(F2), TD(TD), IDMap1Count(0), IDMap2Count(0) {}

  /// Compare - test whether the two functions have equivalent behaviour.
  bool Compare();

private:
  /// Compare - test whether two basic blocks have equivalent behaviour.
  bool Compare(const BasicBlock *BB1, const BasicBlock *BB2);

  /// Enumerate - Assign or look up previously assigned numbers for the two
  /// values, and return whether the numbers are equal. Numbers are assigned in
  /// the order visited.
  bool Enumerate(const Value *V1, const Value *V2);

  /// isEquivalentOperation - Compare two Instructions for equivalence, similar
  /// to Instruction::isSameOperationAs but with modifications to the type
  /// comparison.
  bool isEquivalentOperation(const Instruction *I1,
                             const Instruction *I2) const;

  /// isEquivalentGEP - Compare two GEPs for equivalent pointer arithmetic.
  bool isEquivalentGEP(const GEPOperator *GEP1, const GEPOperator *GEP2);
  bool isEquivalentGEP(const GetElementPtrInst *GEP1,
                       const GetElementPtrInst *GEP2) {
    return isEquivalentGEP(cast<GEPOperator>(GEP1), cast<GEPOperator>(GEP2));
  }

  /// isEquivalentType - Compare two Types, treating all pointer types as equal.
  bool isEquivalentType(const Type *Ty1, const Type *Ty2) const;

  // The two functions undergoing comparison.
  const Function *F1, *F2;

  const TargetData *TD;

  typedef DenseMap<const Value *, unsigned long> IDMap;
  IDMap Map1, Map2;
  unsigned long IDMap1Count, IDMap2Count;
};
}

/// isEquivalentType - any two pointers in the same address space are
/// equivalent. Otherwise, standard type equivalence rules apply.
bool FunctionComparator::isEquivalentType(const Type *Ty1,
                                          const Type *Ty2) const {
  if (Ty1 == Ty2)
    return true;
  if (Ty1->getTypeID() != Ty2->getTypeID())
    return false;

  switch(Ty1->getTypeID()) {
  default:
    llvm_unreachable("Unknown type!");
    // Fall through in Release mode.
  case Type::IntegerTyID:
  case Type::OpaqueTyID:
    // Ty1 == Ty2 would have returned true earlier.
    return false;

  case Type::VoidTyID:
  case Type::FloatTyID:
  case Type::DoubleTyID:
  case Type::X86_FP80TyID:
  case Type::FP128TyID:
  case Type::PPC_FP128TyID:
  case Type::LabelTyID:
  case Type::MetadataTyID:
    return true;

  case Type::PointerTyID: {
    const PointerType *PTy1 = cast<PointerType>(Ty1);
    const PointerType *PTy2 = cast<PointerType>(Ty2);
    return PTy1->getAddressSpace() == PTy2->getAddressSpace();
  }

  case Type::StructTyID: {
    const StructType *STy1 = cast<StructType>(Ty1);
    const StructType *STy2 = cast<StructType>(Ty2);
    if (STy1->getNumElements() != STy2->getNumElements())
      return false;

    if (STy1->isPacked() != STy2->isPacked())
      return false;

    for (unsigned i = 0, e = STy1->getNumElements(); i != e; ++i) {
      if (!isEquivalentType(STy1->getElementType(i), STy2->getElementType(i)))
        return false;
    }
    return true;
  }

  case Type::FunctionTyID: {
    const FunctionType *FTy1 = cast<FunctionType>(Ty1);
    const FunctionType *FTy2 = cast<FunctionType>(Ty2);
    if (FTy1->getNumParams() != FTy2->getNumParams() ||
        FTy1->isVarArg() != FTy2->isVarArg())
      return false;

    if (!isEquivalentType(FTy1->getReturnType(), FTy2->getReturnType()))
      return false;

    for (unsigned i = 0, e = FTy1->getNumParams(); i != e; ++i) {
      if (!isEquivalentType(FTy1->getParamType(i), FTy2->getParamType(i)))
        return false;
    }
    return true;
  }

  case Type::ArrayTyID: {
    const ArrayType *ATy1 = cast<ArrayType>(Ty1);
    const ArrayType *ATy2 = cast<ArrayType>(Ty2);
    return ATy1->getNumElements() == ATy2->getNumElements() &&
           isEquivalentType(ATy1->getElementType(), ATy2->getElementType());
  }

  case Type::VectorTyID: {
    const VectorType *VTy1 = cast<VectorType>(Ty1);
    const VectorType *VTy2 = cast<VectorType>(Ty2);
    return VTy1->getNumElements() == VTy2->getNumElements() &&
           isEquivalentType(VTy1->getElementType(), VTy2->getElementType());
  }
  }
}

/// isEquivalentOperation - determine whether the two operations are the same
/// except that pointer-to-A and pointer-to-B are equivalent. This should be
/// kept in sync with Instruction::isSameOperationAs.
bool FunctionComparator::isEquivalentOperation(const Instruction *I1,
                                               const Instruction *I2) const {
  if (I1->getOpcode() != I2->getOpcode() ||
      I1->getNumOperands() != I2->getNumOperands() ||
      !isEquivalentType(I1->getType(), I2->getType()) ||
      !I1->hasSameSubclassOptionalData(I2))
    return false;

  // We have two instructions of identical opcode and #operands.  Check to see
  // if all operands are the same type
  for (unsigned i = 0, e = I1->getNumOperands(); i != e; ++i)
    if (!isEquivalentType(I1->getOperand(i)->getType(),
                          I2->getOperand(i)->getType()))
      return false;

  // Check special state that is a part of some instructions.
  if (const LoadInst *LI = dyn_cast<LoadInst>(I1))
    return LI->isVolatile() == cast<LoadInst>(I2)->isVolatile() &&
           LI->getAlignment() == cast<LoadInst>(I2)->getAlignment();
  if (const StoreInst *SI = dyn_cast<StoreInst>(I1))
    return SI->isVolatile() == cast<StoreInst>(I2)->isVolatile() &&
           SI->getAlignment() == cast<StoreInst>(I2)->getAlignment();
  if (const CmpInst *CI = dyn_cast<CmpInst>(I1))
    return CI->getPredicate() == cast<CmpInst>(I2)->getPredicate();
  if (const CallInst *CI = dyn_cast<CallInst>(I1))
    return CI->isTailCall() == cast<CallInst>(I2)->isTailCall() &&
           CI->getCallingConv() == cast<CallInst>(I2)->getCallingConv() &&
           CI->getAttributes().getRawPointer() ==
             cast<CallInst>(I2)->getAttributes().getRawPointer();
  if (const InvokeInst *CI = dyn_cast<InvokeInst>(I1))
    return CI->getCallingConv() == cast<InvokeInst>(I2)->getCallingConv() &&
           CI->getAttributes().getRawPointer() ==
             cast<InvokeInst>(I2)->getAttributes().getRawPointer();
  if (const InsertValueInst *IVI = dyn_cast<InsertValueInst>(I1)) {
    if (IVI->getNumIndices() != cast<InsertValueInst>(I2)->getNumIndices())
      return false;
    for (unsigned i = 0, e = IVI->getNumIndices(); i != e; ++i)
      if (IVI->idx_begin()[i] != cast<InsertValueInst>(I2)->idx_begin()[i])
        return false;
    return true;
  }
  if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(I1)) {
    if (EVI->getNumIndices() != cast<ExtractValueInst>(I2)->getNumIndices())
      return false;
    for (unsigned i = 0, e = EVI->getNumIndices(); i != e; ++i)
      if (EVI->idx_begin()[i] != cast<ExtractValueInst>(I2)->idx_begin()[i])
        return false;
    return true;
  }

  return true;
}

/// isEquivalentGEP - determine whether two GEP operations perform the same
/// underlying arithmetic.
bool FunctionComparator::isEquivalentGEP(const GEPOperator *GEP1,
                                         const GEPOperator *GEP2) {
  // When we have target data, we can reduce the GEP down to the value in bytes
  // added to the address.
  if (TD && GEP1->hasAllConstantIndices() && GEP2->hasAllConstantIndices()) {
    SmallVector<Value *, 8> Indices1(GEP1->idx_begin(), GEP1->idx_end());
    SmallVector<Value *, 8> Indices2(GEP2->idx_begin(), GEP2->idx_end());
    uint64_t Offset1 = TD->getIndexedOffset(GEP1->getPointerOperandType(),
                                            Indices1.data(), Indices1.size());
    uint64_t Offset2 = TD->getIndexedOffset(GEP2->getPointerOperandType(),
                                            Indices2.data(), Indices2.size());
    return Offset1 == Offset2;
  }

  if (GEP1->getPointerOperand()->getType() !=
      GEP2->getPointerOperand()->getType())
    return false;

  if (GEP1->getNumOperands() != GEP2->getNumOperands())
    return false;

  for (unsigned i = 0, e = GEP1->getNumOperands(); i != e; ++i) {
    if (!Enumerate(GEP1->getOperand(i), GEP2->getOperand(i)))
      return false;
  }

  return true;
}

/// Enumerate - Compare two values used by the two functions under pair-wise
/// comparison. If this is the first time the values are seen, they're added to
/// the mapping so that we will detect mismatches on next use.
bool FunctionComparator::Enumerate(const Value *V1, const Value *V2) {
  // Check for function @f1 referring to itself and function @f2 referring to
  // itself, or referring to each other, or both referring to either of them.
  // They're all equivalent if the two functions are otherwise equivalent.
  if (V1 == F1 && V2 == F2)
    return true;
  if (V1 == F2 && V2 == F1)
    return true;

  // TODO: constant expressions with GEP or references to F1 or F2.
  if (isa<Constant>(V1))
    return V1 == V2;

  if (isa<InlineAsm>(V1) && isa<InlineAsm>(V2)) {
    const InlineAsm *IA1 = cast<InlineAsm>(V1);
    const InlineAsm *IA2 = cast<InlineAsm>(V2);
    return IA1->getAsmString() == IA2->getAsmString() &&
           IA1->getConstraintString() == IA2->getConstraintString();
  }

  unsigned long &ID1 = Map1[V1];
  if (!ID1)
    ID1 = ++IDMap1Count;

  unsigned long &ID2 = Map2[V2];
  if (!ID2)
    ID2 = ++IDMap2Count;

  return ID1 == ID2;
}

/// Compare - test whether two basic blocks have equivalent behaviour.
bool FunctionComparator::Compare(const BasicBlock *BB1, const BasicBlock *BB2) {
  BasicBlock::const_iterator F1I = BB1->begin(), F1E = BB1->end();
  BasicBlock::const_iterator F2I = BB2->begin(), F2E = BB2->end();

  do {
    if (!Enumerate(F1I, F2I))
      return false;

    if (const GetElementPtrInst *GEP1 = dyn_cast<GetElementPtrInst>(F1I)) {
      const GetElementPtrInst *GEP2 = dyn_cast<GetElementPtrInst>(F2I);
      if (!GEP2)
        return false;

      if (!Enumerate(GEP1->getPointerOperand(), GEP2->getPointerOperand()))
        return false;

      if (!isEquivalentGEP(GEP1, GEP2))
        return false;
    } else {
      if (!isEquivalentOperation(F1I, F2I))
        return false;

      assert(F1I->getNumOperands() == F2I->getNumOperands());
      for (unsigned i = 0, e = F1I->getNumOperands(); i != e; ++i) {
        Value *OpF1 = F1I->getOperand(i);
        Value *OpF2 = F2I->getOperand(i);

        if (!Enumerate(OpF1, OpF2))
          return false;

        if (OpF1->getValueID() != OpF2->getValueID() ||
            !isEquivalentType(OpF1->getType(), OpF2->getType()))
          return false;
      }
    }

    ++F1I, ++F2I;
  } while (F1I != F1E && F2I != F2E);

  return F1I == F1E && F2I == F2E;
}

/// Compare - test whether the two functions have equivalent behaviour.
bool FunctionComparator::Compare() {
  // We need to recheck everything, but check the things that weren't included
  // in the hash first.

  if (F1->getAttributes() != F2->getAttributes())
    return false;

  if (F1->hasGC() != F2->hasGC())
    return false;

  if (F1->hasGC() && F1->getGC() != F2->getGC())
    return false;

  if (F1->hasSection() != F2->hasSection())
    return false;

  if (F1->hasSection() && F1->getSection() != F2->getSection())
    return false;

  if (F1->isVarArg() != F2->isVarArg())
    return false;

  // TODO: if it's internal and only used in direct calls, we could handle this
  // case too.
  if (F1->getCallingConv() != F2->getCallingConv())
    return false;

  if (!isEquivalentType(F1->getFunctionType(), F2->getFunctionType()))
    return false;

  assert(F1->arg_size() == F2->arg_size() &&
         "Identically typed functions have different numbers of args!");

  // Visit the arguments so that they get enumerated in the order they're
  // passed in.
  for (Function::const_arg_iterator f1i = F1->arg_begin(),
         f2i = F2->arg_begin(), f1e = F1->arg_end(); f1i != f1e; ++f1i, ++f2i) {
    if (!Enumerate(f1i, f2i))
      llvm_unreachable("Arguments repeat!");
  }

  // We do a CFG-ordered walk since the actual ordering of the blocks in the
  // linked list is immaterial. Our walk starts at the entry block for both
  // functions, then takes each block from each terminator in order. As an
  // artifact, this also means that unreachable blocks are ignored.
  SmallVector<const BasicBlock *, 8> F1BBs, F2BBs;
  SmallSet<const BasicBlock *, 128> VisitedBBs; // in terms of F1.

  F1BBs.push_back(&F1->getEntryBlock());
  F2BBs.push_back(&F2->getEntryBlock());

  VisitedBBs.insert(F1BBs[0]);
  while (!F1BBs.empty()) {
    const BasicBlock *F1BB = F1BBs.pop_back_val();
    const BasicBlock *F2BB = F2BBs.pop_back_val();

    if (!Enumerate(F1BB, F2BB) || !Compare(F1BB, F2BB))
      return false;

    const TerminatorInst *F1TI = F1BB->getTerminator();
    const TerminatorInst *F2TI = F2BB->getTerminator();

    assert(F1TI->getNumSuccessors() == F2TI->getNumSuccessors());
    for (unsigned i = 0, e = F1TI->getNumSuccessors(); i != e; ++i) {
      if (!VisitedBBs.insert(F1TI->getSuccessor(i)))
        continue;

      F1BBs.push_back(F1TI->getSuccessor(i));
      F2BBs.push_back(F2TI->getSuccessor(i));
    }
  }
  return true;
}

/// Replace direct callers of Old with New.
void MergeFunctions::replaceDirectCallers(Function *Old, Function *New) {
  Constant *BitcastNew = ConstantExpr::getBitCast(New, Old->getType());
  for (Value::use_iterator UI = Old->use_begin(), UE = Old->use_end();
       UI != UE;) {
    Value::use_iterator TheIter = UI;
    ++UI;
    CallSite CS(*TheIter);
    if (CS && CS.isCallee(TheIter)) {
      Remove(CS.getInstruction()->getParent()->getParent());
      TheIter.getUse().set(BitcastNew);
    }
  }
}

void MergeFunctions::WriteThunkOrAlias(Function *F, Function *G) {
  if (HasGlobalAliases && G->hasUnnamedAddr()) {
    if (G->hasExternalLinkage() || G->hasLocalLinkage() ||
        G->hasWeakLinkage()) {
      WriteAlias(F, G);
      return;
    }
  }

  WriteThunk(F, G);
}

/// WriteThunk - Replace G with a simple tail call to bitcast(F). Also replace
/// direct uses of G with bitcast(F). Deletes G.
void MergeFunctions::WriteThunk(Function *F, Function *G) {
  if (!G->mayBeOverridden()) {
    // Redirect direct callers of G to F.
    replaceDirectCallers(G, F);
  }

  // If G was internal then we may have replaced all uses of G with F. If so,
  // stop here and delete G. There's no need for a thunk.
  if (G->hasLocalLinkage() && G->use_empty()) {
    G->eraseFromParent();
    return;
  }

  Function *NewG = Function::Create(G->getFunctionType(), G->getLinkage(), "",
                                    G->getParent());
  BasicBlock *BB = BasicBlock::Create(F->getContext(), "", NewG);
  IRBuilder<false> Builder(BB);

  SmallVector<Value *, 16> Args;
  unsigned i = 0;
  const FunctionType *FFTy = F->getFunctionType();
  for (Function::arg_iterator AI = NewG->arg_begin(), AE = NewG->arg_end();
       AI != AE; ++AI) {
    Args.push_back(Builder.CreateBitCast(AI, FFTy->getParamType(i)));
    ++i;
  }

  CallInst *CI = Builder.CreateCall(F, Args.begin(), Args.end());
  CI->setTailCall();
  CI->setCallingConv(F->getCallingConv());
  if (NewG->getReturnType()->isVoidTy()) {
    Builder.CreateRetVoid();
  } else {
    Builder.CreateRet(Builder.CreateBitCast(CI, NewG->getReturnType()));
  }

  NewG->copyAttributesFrom(G);
  NewG->takeName(G);
  RemoveUsers(G);
  G->replaceAllUsesWith(NewG);
  G->eraseFromParent();

  DEBUG(dbgs() << "WriteThunk: " << NewG->getName() << '\n');
  ++NumThunksWritten;
}

/// WriteAlias - Replace G with an alias to F and delete G.
void MergeFunctions::WriteAlias(Function *F, Function *G) {
  Constant *BitcastF = ConstantExpr::getBitCast(F, G->getType());
  GlobalAlias *GA = new GlobalAlias(G->getType(), G->getLinkage(), "",
                                    BitcastF, G->getParent());
  F->setAlignment(std::max(F->getAlignment(), G->getAlignment()));
  GA->takeName(G);
  GA->setVisibility(G->getVisibility());
  RemoveUsers(G);
  G->replaceAllUsesWith(GA);
  G->eraseFromParent();

  DEBUG(dbgs() << "WriteAlias: " << GA->getName() << '\n');
  ++NumAliasesWritten;
}

/// MergeTwoFunctions - Merge two equivalent functions. Upon completion,
/// Function G is deleted.
void MergeFunctions::MergeTwoFunctions(Function *F, Function *G) {
  if (F->mayBeOverridden()) {
    assert(G->mayBeOverridden());

    if (HasGlobalAliases) {
      // Make them both thunks to the same internal function.
      Function *H = Function::Create(F->getFunctionType(), F->getLinkage(), "",
                                     F->getParent());
      H->copyAttributesFrom(F);
      H->takeName(F);
      RemoveUsers(F);
      F->replaceAllUsesWith(H);

      unsigned MaxAlignment = std::max(G->getAlignment(), H->getAlignment());

      WriteAlias(F, G);
      WriteAlias(F, H);

      F->setAlignment(MaxAlignment);
      F->setLinkage(GlobalValue::PrivateLinkage);
    } else {
      // We can't merge them. Instead, pick one and update all direct callers
      // to call it and hope that we improve the instruction cache hit rate.
      replaceDirectCallers(G, F);
    }

    ++NumDoubleWeak;
  } else {
    WriteThunkOrAlias(F, G);
  }

  ++NumFunctionsMerged;
}

// Insert - Insert a ComparableFunction into the FnSet, or merge it away if
// equal to one that's already inserted.
bool MergeFunctions::Insert(ComparableFunction &NewF) {
  std::pair<FnSetType::iterator, bool> Result = FnSet.insert(NewF);
  if (Result.second)
    return false;

  const ComparableFunction &OldF = *Result.first;

  // Never thunk a strong function to a weak function.
  assert(!OldF.getFunc()->mayBeOverridden() ||
         NewF.getFunc()->mayBeOverridden());

  DEBUG(dbgs() << "  " << OldF.getFunc()->getName() << " == "
               << NewF.getFunc()->getName() << '\n');

  Function *DeleteF = NewF.getFunc();
  NewF.release();
  MergeTwoFunctions(OldF.getFunc(), DeleteF);
  return true;
}

// Remove - Remove a function from FnSet. If it was already in FnSet, add it to
// Deferred so that we'll look at it in the next round.
void MergeFunctions::Remove(Function *F) {
  ComparableFunction CF = ComparableFunction(F, TD);
  if (FnSet.erase(CF)) {
    Deferred.push_back(F);
  }
}

// RemoveUsers - For each instruction used by the value, Remove() the function
// that contains the instruction. This should happen right before a call to RAUW.
void MergeFunctions::RemoveUsers(Value *V) {
  std::vector<Value *> Worklist;
  Worklist.push_back(V);
  while (!Worklist.empty()) {
    Value *V = Worklist.back();
    Worklist.pop_back();

    for (Value::use_iterator UI = V->use_begin(), UE = V->use_end();
         UI != UE; ++UI) {
      Use &U = UI.getUse();
      if (Instruction *I = dyn_cast<Instruction>(U.getUser())) {
        Remove(I->getParent()->getParent());
      } else if (isa<GlobalValue>(U.getUser())) {
        // do nothing
      } else if (Constant *C = dyn_cast<Constant>(U.getUser())) {
        for (Value::use_iterator CUI = C->use_begin(), CUE = C->use_end();
             CUI != CUE; ++CUI)
          Worklist.push_back(*CUI);
      }
    }
  }
}

bool MergeFunctions::runOnModule(Module &M) {
  bool Changed = false;
  TD = getAnalysisIfAvailable<TargetData>();

  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    Deferred.push_back(WeakVH(I));
  }

  do {
    std::vector<WeakVH> Worklist;
    Deferred.swap(Worklist);

    DEBUG(dbgs() << "size of module: " << M.size() << '\n');
    DEBUG(dbgs() << "size of worklist: " << Worklist.size() << '\n');

    // Insert only strong functions and merge them. Strong function merging
    // always deletes one of them.
    for (std::vector<WeakVH>::iterator I = Worklist.begin(),
           E = Worklist.end(); I != E; ++I) {
      if (!*I) continue;
      Function *F = cast<Function>(*I);
      if (!F->isDeclaration() && !F->hasAvailableExternallyLinkage() &&
          !F->mayBeOverridden()) {
        ComparableFunction CF = ComparableFunction(F, TD);
        Changed |= Insert(CF);
      }
    }

    // Insert only weak functions and merge them. By doing these second we
    // create thunks to the strong function when possible. When two weak
    // functions are identical, we create a new strong function with two weak
    // weak thunks to it which are identical but not mergable.
    for (std::vector<WeakVH>::iterator I = Worklist.begin(),
           E = Worklist.end(); I != E; ++I) {
      if (!*I) continue;
      Function *F = cast<Function>(*I);
      if (!F->isDeclaration() && !F->hasAvailableExternallyLinkage() &&
          F->mayBeOverridden()) {
        ComparableFunction CF = ComparableFunction(F, TD);
        Changed |= Insert(CF);
      }
    }
    DEBUG(dbgs() << "size of FnSet: " << FnSet.size() << '\n');
  } while (!Deferred.empty());

  FnSet.clear();

  return Changed;
}

bool DenseMapInfo<ComparableFunction>::isEqual(const ComparableFunction &LHS,
                                               const ComparableFunction &RHS) {
  if (LHS.getFunc() == RHS.getFunc() &&
      LHS.getHash() == RHS.getHash())
    return true;
  if (!LHS.getFunc() || !RHS.getFunc())
    return false;
  assert(LHS.getTD() == RHS.getTD() &&
         "Comparing functions for different targets");

  bool inserted;
  bool &result1 = LHS.getOrInsertCachedComparison(RHS, inserted);
  if (!inserted)
    return result1;
  bool &result2 = RHS.getOrInsertCachedComparison(LHS, inserted);
  if (!inserted)
    return result1 = result2;

  return result1 = result2 = FunctionComparator(LHS.getTD(), LHS.getFunc(),
                                                RHS.getFunc()).Compare();
}
