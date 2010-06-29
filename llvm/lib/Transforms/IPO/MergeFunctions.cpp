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
// and call, this is irrelevant, and we'd like to fold such implementations.
//
// * use SCC to cut down on pair-wise comparisons and solve larger cycles.
//
// The current implementation loops over a pair-wise comparison of all
// functions in the program where the two functions in the pair are treated as
// assumed to be equal until proven otherwise. We could both use fewer
// comparisons and optimize more complex cases if we used strongly connected
// components of the call graph.
//
// * be smarter about bitcast.
//
// In order to fold functions, we will sometimes add either bitcast instructions
// or bitcast constant expressions. Unfortunately, this can confound further
// analysis since the two functions differ where one has a bitcast and the
// other doesn't. We should learn to peer through bitcasts without imposing bad
// performance properties.
//
// * don't emit aliases for Mach-O.
//
// Mach-O doesn't support aliases which means that we must avoid introducing
// them in the bitcode on architectures which don't support them, such as
// Mac OSX. There's a few approaches to this problem;
//   a) teach codegen to lower global aliases to thunks on platforms which don't
//      support them.
//   b) always emit thunks, and create a separate thunk-to-alias pass which
//      runs on ELF systems. This has the added benefit of transforming other
//      thunks such as those produced by a C++ frontend into aliases when legal
//      to do so.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mergefunc"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Constants.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetData.h"
#include <map>
#include <vector>
using namespace llvm;

STATISTIC(NumFunctionsMerged, "Number of functions merged");

namespace {
  class MergeFunctions : public ModulePass {
  public:
    static char ID; // Pass identification, replacement for typeid
    MergeFunctions() : ModulePass(&ID) {}

    bool runOnModule(Module &M);

  private:
    bool isEquivalentGEP(const GetElementPtrInst *GEP1,
                         const GetElementPtrInst *GEP2);

    bool equals(const BasicBlock *BB1, const BasicBlock *BB2);
    bool equals(const Function *F, const Function *G);

    bool compare(const Value *V1, const Value *V2);

    const Function *LHS, *RHS;
    typedef DenseMap<const Value *, unsigned long> IDMap;
    IDMap Map;
    DenseMap<const Function *, IDMap> Domains;
    DenseMap<const Function *, unsigned long> DomainCount;
    TargetData *TD;
  };
}

char MergeFunctions::ID = 0;
static RegisterPass<MergeFunctions> X("mergefunc", "Merge Functions");

ModulePass *llvm::createMergeFunctionsPass() {
  return new MergeFunctions();
}

// ===----------------------------------------------------------------------===
// Comparison of functions
// ===----------------------------------------------------------------------===

static unsigned long hash(const Function *F) {
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

/// isEquivalentType - any two pointers are equivalent. Otherwise, standard
/// type equivalence rules apply.
static bool isEquivalentType(const Type *Ty1, const Type *Ty2) {
  if (Ty1 == Ty2)
    return true;
  if (Ty1->getTypeID() != Ty2->getTypeID())
    return false;

  switch(Ty1->getTypeID()) {
  default:
    llvm_unreachable("Unknown type!");
    // Fall through in Release-Asserts mode.
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

  case Type::UnionTyID: {
    const UnionType *UTy1 = cast<UnionType>(Ty1);
    const UnionType *UTy2 = cast<UnionType>(Ty2);

    // TODO: we could be fancy with union(A, union(A, B)) === union(A, B), etc.
    if (UTy1->getNumElements() != UTy2->getNumElements())
      return false;

    for (unsigned i = 0, e = UTy1->getNumElements(); i != e; ++i) {
      if (!isEquivalentType(UTy1->getElementType(i), UTy2->getElementType(i)))
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

  case Type::ArrayTyID:
  case Type::VectorTyID: {
    const SequentialType *STy1 = cast<SequentialType>(Ty1);
    const SequentialType *STy2 = cast<SequentialType>(Ty2);
    return isEquivalentType(STy1->getElementType(), STy2->getElementType());
  }
  }
}

/// isEquivalentOperation - determine whether the two operations are the same
/// except that pointer-to-A and pointer-to-B are equivalent. This should be
/// kept in sync with Instruction::isSameOperationAs.
static bool
isEquivalentOperation(const Instruction *I1, const Instruction *I2) {
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

bool MergeFunctions::isEquivalentGEP(const GetElementPtrInst *GEP1,
                                     const GetElementPtrInst *GEP2) {
  if (TD && GEP1->hasAllConstantIndices() && GEP2->hasAllConstantIndices()) {
    SmallVector<Value *, 8> Indices1, Indices2;
    for (GetElementPtrInst::const_op_iterator I = GEP1->idx_begin(),
           E = GEP1->idx_end(); I != E; ++I) {
      Indices1.push_back(*I);
    }
    for (GetElementPtrInst::const_op_iterator I = GEP2->idx_begin(),
           E = GEP2->idx_end(); I != E; ++I) {
      Indices2.push_back(*I);
    }
    uint64_t Offset1 = TD->getIndexedOffset(GEP1->getPointerOperandType(),
                                            Indices1.data(), Indices1.size());
    uint64_t Offset2 = TD->getIndexedOffset(GEP2->getPointerOperandType(),
                                            Indices2.data(), Indices2.size());
    return Offset1 == Offset2;
  }

  // Equivalent types aren't enough.
  if (GEP1->getPointerOperand()->getType() !=
      GEP2->getPointerOperand()->getType())
    return false;

  if (GEP1->getNumOperands() != GEP2->getNumOperands())
    return false;

  for (unsigned i = 0, e = GEP1->getNumOperands(); i != e; ++i) {
    if (!compare(GEP1->getOperand(i), GEP2->getOperand(i)))
      return false;
  }

  return true;
}

bool MergeFunctions::compare(const Value *V1, const Value *V2) {
  if (V1 == LHS || V1 == RHS)
    if (V2 == LHS || V2 == RHS)
      return true;

  // TODO: constant expressions in terms of LHS and RHS
  if (isa<Constant>(V1))
    return V1 == V2;

  if (isa<InlineAsm>(V1) && isa<InlineAsm>(V2)) {
    const InlineAsm *IA1 = cast<InlineAsm>(V1);
    const InlineAsm *IA2 = cast<InlineAsm>(V2);
    return IA1->getAsmString() == IA2->getAsmString() &&
           IA1->getConstraintString() == IA2->getConstraintString();
  }

  // We enumerate constants globally and arguments, basic blocks or
  // instructions within the function they belong to.
  const Function *Domain1 = NULL;
  if (const Argument *A = dyn_cast<Argument>(V1)) {
    Domain1 = A->getParent();
  } else if (const BasicBlock *BB = dyn_cast<BasicBlock>(V1)) {
    Domain1 = BB->getParent();
  } else if (const Instruction *I = dyn_cast<Instruction>(V1)) {
    Domain1 = I->getParent()->getParent();
  }

  const Function *Domain2 = NULL;
  if (const Argument *A = dyn_cast<Argument>(V2)) {
    Domain2 = A->getParent();
  } else if (const BasicBlock *BB = dyn_cast<BasicBlock>(V2)) {
    Domain2 = BB->getParent();
  } else if (const Instruction *I = dyn_cast<Instruction>(V2)) {
    Domain2 = I->getParent()->getParent();
  }

  if (Domain1 != Domain2)
    if (Domain1 != LHS && Domain1 != RHS)
      if (Domain2 != LHS && Domain2 != RHS)
        return false;

  IDMap &Map1 = Domains[Domain1];
  unsigned long &ID1 = Map1[V1];
  if (!ID1)
    ID1 = ++DomainCount[Domain1];

  IDMap &Map2 = Domains[Domain2];
  unsigned long &ID2 = Map2[V2];
  if (!ID2)
    ID2 = ++DomainCount[Domain2];

  return ID1 == ID2;
}

bool MergeFunctions::equals(const BasicBlock *BB1, const BasicBlock *BB2) {
  BasicBlock::const_iterator FI = BB1->begin(), FE = BB1->end();
  BasicBlock::const_iterator GI = BB2->begin(), GE = BB2->end();

  do {
    if (!compare(FI, GI))
      return false;

    if (isa<GetElementPtrInst>(FI) && isa<GetElementPtrInst>(GI)) {
      const GetElementPtrInst *GEP1 = cast<GetElementPtrInst>(FI);
      const GetElementPtrInst *GEP2 = cast<GetElementPtrInst>(GI);

      if (!compare(GEP1->getPointerOperand(), GEP2->getPointerOperand()))
        return false;

      if (!isEquivalentGEP(GEP1, GEP2))
        return false;
    } else {
      if (!isEquivalentOperation(FI, GI))
        return false;

      for (unsigned i = 0, e = FI->getNumOperands(); i != e; ++i) {
        Value *OpF = FI->getOperand(i);
        Value *OpG = GI->getOperand(i);

        if (!compare(OpF, OpG))
          return false;

        if (OpF->getValueID() != OpG->getValueID() ||
            !isEquivalentType(OpF->getType(), OpG->getType()))
          return false;
      }
    }

    ++FI, ++GI;
  } while (FI != FE && GI != GE);

  return FI == FE && GI == GE;
}

bool MergeFunctions::equals(const Function *F, const Function *G) {
  // We need to recheck everything, but check the things that weren't included
  // in the hash first.

  if (F->getAttributes() != G->getAttributes())
    return false;

  if (F->hasGC() != G->hasGC())
    return false;

  if (F->hasGC() && F->getGC() != G->getGC())
    return false;

  if (F->hasSection() != G->hasSection())
    return false;

  if (F->hasSection() && F->getSection() != G->getSection())
    return false;

  if (F->isVarArg() != G->isVarArg())
    return false;

  // TODO: if it's internal and only used in direct calls, we could handle this
  // case too.
  if (F->getCallingConv() != G->getCallingConv())
    return false;

  if (!isEquivalentType(F->getFunctionType(), G->getFunctionType()))
    return false;

  assert(F->arg_size() == G->arg_size() &&
         "Identical functions have a different number of args.");

  LHS = F;
  RHS = G;

  // Visit the arguments so that they get enumerated in the order they're
  // passed in.
  for (Function::const_arg_iterator fi = F->arg_begin(), gi = G->arg_begin(),
         fe = F->arg_end(); fi != fe; ++fi, ++gi) {
    if (!compare(fi, gi))
      llvm_unreachable("Arguments repeat");
  }

  SmallVector<const BasicBlock *, 8> FBBs, GBBs;
  SmallSet<const BasicBlock *, 128> VisitedBBs; // in terms of F.
  FBBs.push_back(&F->getEntryBlock());
  GBBs.push_back(&G->getEntryBlock());
  VisitedBBs.insert(FBBs[0]);
  while (!FBBs.empty()) {
    const BasicBlock *FBB = FBBs.pop_back_val();
    const BasicBlock *GBB = GBBs.pop_back_val();
    if (!compare(FBB, GBB) || !equals(FBB, GBB)) {
      Domains.clear();
      DomainCount.clear();
      return false;
    }
    const TerminatorInst *FTI = FBB->getTerminator();
    const TerminatorInst *GTI = GBB->getTerminator();
    assert(FTI->getNumSuccessors() == GTI->getNumSuccessors());
    for (unsigned i = 0, e = FTI->getNumSuccessors(); i != e; ++i) {
      if (!VisitedBBs.insert(FTI->getSuccessor(i)))
        continue;
      FBBs.push_back(FTI->getSuccessor(i));
      GBBs.push_back(GTI->getSuccessor(i));
    }
  }

  Domains.clear();
  DomainCount.clear();
  return true;
}

// ===----------------------------------------------------------------------===
// Folding of functions
// ===----------------------------------------------------------------------===

// Cases:
// * F is external strong, G is external strong:
//   turn G into a thunk to F    (1)
// * F is external strong, G is external weak:
//   turn G into a thunk to F    (1)
// * F is external weak, G is external weak:
//   unfoldable
// * F is external strong, G is internal:
//   address of G taken:
//     turn G into a thunk to F  (1)
//   address of G not taken:
//     make G an alias to F      (2)
// * F is internal, G is external weak
//   address of F is taken:
//     turn G into a thunk to F  (1)
//   address of F is not taken:
//     make G an alias of F      (2)
// * F is internal, G is internal:
//   address of F and G are taken:
//     turn G into a thunk to F  (1)
//   address of G is not taken:
//     make G an alias to F      (2)
//
// alias requires linkage == (external,local,weak) fallback to creating a thunk
// external means 'externally visible' linkage != (internal,private)
// internal means linkage == (internal,private)
// weak means linkage mayBeOverridable
// being external implies that the address is taken
//
// 1. turn G into a thunk to F
// 2. make G an alias to F

enum LinkageCategory {
  ExternalStrong,
  ExternalWeak,
  Internal
};

static LinkageCategory categorize(const Function *F) {
  switch (F->getLinkage()) {
  case GlobalValue::InternalLinkage:
  case GlobalValue::PrivateLinkage:
  case GlobalValue::LinkerPrivateLinkage:
    return Internal;

  case GlobalValue::WeakAnyLinkage:
  case GlobalValue::WeakODRLinkage:
  case GlobalValue::ExternalWeakLinkage:
    return ExternalWeak;

  case GlobalValue::ExternalLinkage:
  case GlobalValue::AvailableExternallyLinkage:
  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::LinkOnceODRLinkage:
  case GlobalValue::AppendingLinkage:
  case GlobalValue::DLLImportLinkage:
  case GlobalValue::DLLExportLinkage:
  case GlobalValue::CommonLinkage:
    return ExternalStrong;
  }

  llvm_unreachable("Unknown LinkageType.");
  return ExternalWeak;
}

static void ThunkGToF(Function *F, Function *G) {
  if (!G->mayBeOverridden()) {
    // Redirect direct callers of G to F.
    Constant *BitcastF = ConstantExpr::getBitCast(F, G->getType());
    for (Value::use_iterator UI = G->use_begin(), UE = G->use_end();
         UI != UE;) {
      Value::use_iterator TheIter = UI;
      ++UI;
      CallSite CS(*TheIter);
      if (CS && CS.isCallee(TheIter))
        TheIter.getUse().set(BitcastF);
    }
  }

  Function *NewG = Function::Create(G->getFunctionType(), G->getLinkage(), "",
                                    G->getParent());
  BasicBlock *BB = BasicBlock::Create(F->getContext(), "", NewG);

  SmallVector<Value *, 16> Args;
  unsigned i = 0;
  const FunctionType *FFTy = F->getFunctionType();
  for (Function::arg_iterator AI = NewG->arg_begin(), AE = NewG->arg_end();
       AI != AE; ++AI) {
    if (FFTy->getParamType(i) == AI->getType()) {
      Args.push_back(AI);
    } else {
      Args.push_back(new BitCastInst(AI, FFTy->getParamType(i), "", BB));
    }
    ++i;
  }

  CallInst *CI = CallInst::Create(F, Args.begin(), Args.end(), "", BB);
  CI->setTailCall();
  CI->setCallingConv(F->getCallingConv());
  if (NewG->getReturnType()->isVoidTy()) {
    ReturnInst::Create(F->getContext(), BB);
  } else if (CI->getType() != NewG->getReturnType()) {
    Value *BCI = new BitCastInst(CI, NewG->getReturnType(), "", BB);
    ReturnInst::Create(F->getContext(), BCI, BB);
  } else {
    ReturnInst::Create(F->getContext(), CI, BB);
  }

  NewG->copyAttributesFrom(G);
  NewG->takeName(G);
  G->replaceAllUsesWith(NewG);
  G->eraseFromParent();
}

static void AliasGToF(Function *F, Function *G) {
  if (!G->hasExternalLinkage() && !G->hasLocalLinkage() && !G->hasWeakLinkage())
    return ThunkGToF(F, G);

  GlobalAlias *GA = new GlobalAlias(
    G->getType(), G->getLinkage(), "",
    ConstantExpr::getBitCast(F, G->getType()), G->getParent());
  F->setAlignment(std::max(F->getAlignment(), G->getAlignment()));
  GA->takeName(G);
  GA->setVisibility(G->getVisibility());
  G->replaceAllUsesWith(GA);
  G->eraseFromParent();
}

static bool fold(std::vector<Function *> &FnVec, unsigned i, unsigned j) {
  Function *F = FnVec[i];
  Function *G = FnVec[j];

  LinkageCategory catF = categorize(F);
  LinkageCategory catG = categorize(G);

  if (catF == ExternalWeak || (catF == Internal && catG == ExternalStrong)) {
    std::swap(FnVec[i], FnVec[j]);
    std::swap(F, G);
    std::swap(catF, catG);
  }

  switch (catF) {
  case ExternalStrong:
    switch (catG) {
    case ExternalStrong:
    case ExternalWeak:
      ThunkGToF(F, G);
      break;
    case Internal:
      if (G->hasAddressTaken())
        ThunkGToF(F, G);
      else
        AliasGToF(F, G);
      break;
    }
    break;

  case ExternalWeak: {
    assert(catG == ExternalWeak);

    // Make them both thunks to the same internal function.
    F->setAlignment(std::max(F->getAlignment(), G->getAlignment()));
    Function *H = Function::Create(F->getFunctionType(), F->getLinkage(), "",
                                   F->getParent());
    H->copyAttributesFrom(F);
    H->takeName(F);
    F->replaceAllUsesWith(H);

    ThunkGToF(F, G);
    ThunkGToF(F, H);

    F->setLinkage(GlobalValue::InternalLinkage);
  } break;

  case Internal:
    switch (catG) {
    case ExternalStrong:
      llvm_unreachable(0);
      // fall-through
    case ExternalWeak:
      if (F->hasAddressTaken())
        ThunkGToF(F, G);
      else
        AliasGToF(F, G);
      break;
    case Internal: {
      bool addrTakenF = F->hasAddressTaken();
      bool addrTakenG = G->hasAddressTaken();
      if (!addrTakenF && addrTakenG) {
        std::swap(FnVec[i], FnVec[j]);
        std::swap(F, G);
        std::swap(addrTakenF, addrTakenG);
      }

      if (addrTakenF && addrTakenG) {
        ThunkGToF(F, G);
      } else {
        assert(!addrTakenG);
        AliasGToF(F, G);
      }
    } break;
  } break;
  }

  ++NumFunctionsMerged;
  return true;
}

// ===----------------------------------------------------------------------===
// Pass definition
// ===----------------------------------------------------------------------===

bool MergeFunctions::runOnModule(Module &M) {
  bool Changed = false;

  std::map<unsigned long, std::vector<Function *> > FnMap;

  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    if (F->isDeclaration())
      continue;

    FnMap[hash(F)].push_back(F);
  }

  TD = getAnalysisIfAvailable<TargetData>();

  bool LocalChanged;
  do {
    LocalChanged = false;
    DEBUG(dbgs() << "size: " << FnMap.size() << "\n");
    for (std::map<unsigned long, std::vector<Function *> >::iterator
           I = FnMap.begin(), E = FnMap.end(); I != E; ++I) {
      std::vector<Function *> &FnVec = I->second;
      DEBUG(dbgs() << "hash (" << I->first << "): " << FnVec.size() << "\n");

      for (int i = 0, e = FnVec.size(); i != e; ++i) {
        for (int j = i + 1; j != e; ++j) {
          bool isEqual = equals(FnVec[i], FnVec[j]);

          DEBUG(dbgs() << "  " << FnVec[i]->getName()
                << (isEqual ? " == " : " != ")
                << FnVec[j]->getName() << "\n");

          if (isEqual) {
            if (fold(FnVec, i, j)) {
              LocalChanged = true;
              FnVec.erase(FnVec.begin() + j);
              --j, --e;
            }
          }
        }
      }

    }
    Changed |= LocalChanged;
  } while (LocalChanged);

  return Changed;
}
