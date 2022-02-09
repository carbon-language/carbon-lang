//===- Scalarizer.cpp - Scalarize vector operations -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass converts vector operations into scalar operations, in order
// to expose optimization opportunities on the individual scalar operations.
// It is mainly intended for targets that do not have vector units, but it
// may also be useful for revectorizing code to different vector widths.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/Scalarizer.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cassert>
#include <cstdint>
#include <iterator>
#include <map>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "scalarizer"

static cl::opt<bool> ScalarizeVariableInsertExtract(
    "scalarize-variable-insert-extract", cl::init(true), cl::Hidden,
    cl::desc("Allow the scalarizer pass to scalarize "
             "insertelement/extractelement with variable index"));

// This is disabled by default because having separate loads and stores
// makes it more likely that the -combiner-alias-analysis limits will be
// reached.
static cl::opt<bool>
    ScalarizeLoadStore("scalarize-load-store", cl::init(false), cl::Hidden,
                       cl::desc("Allow the scalarizer pass to scalarize loads and store"));

namespace {

BasicBlock::iterator skipPastPhiNodesAndDbg(BasicBlock::iterator Itr) {
  BasicBlock *BB = Itr->getParent();
  if (isa<PHINode>(Itr))
    Itr = BB->getFirstInsertionPt();
  if (Itr != BB->end())
    Itr = skipDebugIntrinsics(Itr);
  return Itr;
}

// Used to store the scattered form of a vector.
using ValueVector = SmallVector<Value *, 8>;

// Used to map a vector Value to its scattered form.  We use std::map
// because we want iterators to persist across insertion and because the
// values are relatively large.
using ScatterMap = std::map<Value *, ValueVector>;

// Lists Instructions that have been replaced with scalar implementations,
// along with a pointer to their scattered forms.
using GatherList = SmallVector<std::pair<Instruction *, ValueVector *>, 16>;

// Provides a very limited vector-like interface for lazily accessing one
// component of a scattered vector or vector pointer.
class Scatterer {
public:
  Scatterer() = default;

  // Scatter V into Size components.  If new instructions are needed,
  // insert them before BBI in BB.  If Cache is nonnull, use it to cache
  // the results.
  Scatterer(BasicBlock *bb, BasicBlock::iterator bbi, Value *v,
            ValueVector *cachePtr = nullptr);

  // Return component I, creating a new Value for it if necessary.
  Value *operator[](unsigned I);

  // Return the number of components.
  unsigned size() const { return Size; }

private:
  BasicBlock *BB;
  BasicBlock::iterator BBI;
  Value *V;
  ValueVector *CachePtr;
  PointerType *PtrTy;
  ValueVector Tmp;
  unsigned Size;
};

// FCmpSpliiter(FCI)(Builder, X, Y, Name) uses Builder to create an FCmp
// called Name that compares X and Y in the same way as FCI.
struct FCmpSplitter {
  FCmpSplitter(FCmpInst &fci) : FCI(fci) {}

  Value *operator()(IRBuilder<> &Builder, Value *Op0, Value *Op1,
                    const Twine &Name) const {
    return Builder.CreateFCmp(FCI.getPredicate(), Op0, Op1, Name);
  }

  FCmpInst &FCI;
};

// ICmpSpliiter(ICI)(Builder, X, Y, Name) uses Builder to create an ICmp
// called Name that compares X and Y in the same way as ICI.
struct ICmpSplitter {
  ICmpSplitter(ICmpInst &ici) : ICI(ici) {}

  Value *operator()(IRBuilder<> &Builder, Value *Op0, Value *Op1,
                    const Twine &Name) const {
    return Builder.CreateICmp(ICI.getPredicate(), Op0, Op1, Name);
  }

  ICmpInst &ICI;
};

// UnarySpliiter(UO)(Builder, X, Name) uses Builder to create
// a unary operator like UO called Name with operand X.
struct UnarySplitter {
  UnarySplitter(UnaryOperator &uo) : UO(uo) {}

  Value *operator()(IRBuilder<> &Builder, Value *Op, const Twine &Name) const {
    return Builder.CreateUnOp(UO.getOpcode(), Op, Name);
  }

  UnaryOperator &UO;
};

// BinarySpliiter(BO)(Builder, X, Y, Name) uses Builder to create
// a binary operator like BO called Name with operands X and Y.
struct BinarySplitter {
  BinarySplitter(BinaryOperator &bo) : BO(bo) {}

  Value *operator()(IRBuilder<> &Builder, Value *Op0, Value *Op1,
                    const Twine &Name) const {
    return Builder.CreateBinOp(BO.getOpcode(), Op0, Op1, Name);
  }

  BinaryOperator &BO;
};

// Information about a load or store that we're scalarizing.
struct VectorLayout {
  VectorLayout() = default;

  // Return the alignment of element I.
  Align getElemAlign(unsigned I) {
    return commonAlignment(VecAlign, I * ElemSize);
  }

  // The type of the vector.
  VectorType *VecTy = nullptr;

  // The type of each element.
  Type *ElemTy = nullptr;

  // The alignment of the vector.
  Align VecAlign;

  // The size of each element.
  uint64_t ElemSize = 0;
};

class ScalarizerVisitor : public InstVisitor<ScalarizerVisitor, bool> {
public:
  ScalarizerVisitor(unsigned ParallelLoopAccessMDKind, DominatorTree *DT)
    : ParallelLoopAccessMDKind(ParallelLoopAccessMDKind), DT(DT) {
  }

  bool visit(Function &F);

  // InstVisitor methods.  They return true if the instruction was scalarized,
  // false if nothing changed.
  bool visitInstruction(Instruction &I) { return false; }
  bool visitSelectInst(SelectInst &SI);
  bool visitICmpInst(ICmpInst &ICI);
  bool visitFCmpInst(FCmpInst &FCI);
  bool visitUnaryOperator(UnaryOperator &UO);
  bool visitBinaryOperator(BinaryOperator &BO);
  bool visitGetElementPtrInst(GetElementPtrInst &GEPI);
  bool visitCastInst(CastInst &CI);
  bool visitBitCastInst(BitCastInst &BCI);
  bool visitInsertElementInst(InsertElementInst &IEI);
  bool visitExtractElementInst(ExtractElementInst &EEI);
  bool visitShuffleVectorInst(ShuffleVectorInst &SVI);
  bool visitPHINode(PHINode &PHI);
  bool visitLoadInst(LoadInst &LI);
  bool visitStoreInst(StoreInst &SI);
  bool visitCallInst(CallInst &ICI);

private:
  Scatterer scatter(Instruction *Point, Value *V);
  void gather(Instruction *Op, const ValueVector &CV);
  bool canTransferMetadata(unsigned Kind);
  void transferMetadataAndIRFlags(Instruction *Op, const ValueVector &CV);
  Optional<VectorLayout> getVectorLayout(Type *Ty, Align Alignment,
                                         const DataLayout &DL);
  bool finish();

  template<typename T> bool splitUnary(Instruction &, const T &);
  template<typename T> bool splitBinary(Instruction &, const T &);

  bool splitCall(CallInst &CI);

  ScatterMap Scattered;
  GatherList Gathered;

  SmallVector<WeakTrackingVH, 32> PotentiallyDeadInstrs;

  unsigned ParallelLoopAccessMDKind;

  DominatorTree *DT;
};

class ScalarizerLegacyPass : public FunctionPass {
public:
  static char ID;

  ScalarizerLegacyPass() : FunctionPass(ID) {
    initializeScalarizerLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage& AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }
};

} // end anonymous namespace

char ScalarizerLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(ScalarizerLegacyPass, "scalarizer",
                      "Scalarize vector operations", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(ScalarizerLegacyPass, "scalarizer",
                    "Scalarize vector operations", false, false)

Scatterer::Scatterer(BasicBlock *bb, BasicBlock::iterator bbi, Value *v,
                     ValueVector *cachePtr)
  : BB(bb), BBI(bbi), V(v), CachePtr(cachePtr) {
  Type *Ty = V->getType();
  PtrTy = dyn_cast<PointerType>(Ty);
  if (PtrTy)
    Ty = PtrTy->getElementType();
  Size = cast<FixedVectorType>(Ty)->getNumElements();
  if (!CachePtr)
    Tmp.resize(Size, nullptr);
  else if (CachePtr->empty())
    CachePtr->resize(Size, nullptr);
  else
    assert(Size == CachePtr->size() && "Inconsistent vector sizes");
}

// Return component I, creating a new Value for it if necessary.
Value *Scatterer::operator[](unsigned I) {
  ValueVector &CV = (CachePtr ? *CachePtr : Tmp);
  // Try to reuse a previous value.
  if (CV[I])
    return CV[I];
  IRBuilder<> Builder(BB, BBI);
  if (PtrTy) {
    Type *ElTy = cast<VectorType>(PtrTy->getElementType())->getElementType();
    if (!CV[0]) {
      Type *NewPtrTy = PointerType::get(ElTy, PtrTy->getAddressSpace());
      CV[0] = Builder.CreateBitCast(V, NewPtrTy, V->getName() + ".i0");
    }
    if (I != 0)
      CV[I] = Builder.CreateConstGEP1_32(ElTy, CV[0], I,
                                         V->getName() + ".i" + Twine(I));
  } else {
    // Search through a chain of InsertElementInsts looking for element I.
    // Record other elements in the cache.  The new V is still suitable
    // for all uncached indices.
    while (true) {
      InsertElementInst *Insert = dyn_cast<InsertElementInst>(V);
      if (!Insert)
        break;
      ConstantInt *Idx = dyn_cast<ConstantInt>(Insert->getOperand(2));
      if (!Idx)
        break;
      unsigned J = Idx->getZExtValue();
      V = Insert->getOperand(0);
      if (I == J) {
        CV[J] = Insert->getOperand(1);
        return CV[J];
      } else if (!CV[J]) {
        // Only cache the first entry we find for each index we're not actively
        // searching for. This prevents us from going too far up the chain and
        // caching incorrect entries.
        CV[J] = Insert->getOperand(1);
      }
    }
    CV[I] = Builder.CreateExtractElement(V, Builder.getInt32(I),
                                         V->getName() + ".i" + Twine(I));
  }
  return CV[I];
}

bool ScalarizerLegacyPass::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  Module &M = *F.getParent();
  unsigned ParallelLoopAccessMDKind =
      M.getContext().getMDKindID("llvm.mem.parallel_loop_access");
  DominatorTree *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  ScalarizerVisitor Impl(ParallelLoopAccessMDKind, DT);
  return Impl.visit(F);
}

FunctionPass *llvm::createScalarizerPass() {
  return new ScalarizerLegacyPass();
}

bool ScalarizerVisitor::visit(Function &F) {
  assert(Gathered.empty() && Scattered.empty());

  // To ensure we replace gathered components correctly we need to do an ordered
  // traversal of the basic blocks in the function.
  ReversePostOrderTraversal<BasicBlock *> RPOT(&F.getEntryBlock());
  for (BasicBlock *BB : RPOT) {
    for (BasicBlock::iterator II = BB->begin(), IE = BB->end(); II != IE;) {
      Instruction *I = &*II;
      bool Done = InstVisitor::visit(I);
      ++II;
      if (Done && I->getType()->isVoidTy())
        I->eraseFromParent();
    }
  }
  return finish();
}

// Return a scattered form of V that can be accessed by Point.  V must be a
// vector or a pointer to a vector.
Scatterer ScalarizerVisitor::scatter(Instruction *Point, Value *V) {
  if (Argument *VArg = dyn_cast<Argument>(V)) {
    // Put the scattered form of arguments in the entry block,
    // so that it can be used everywhere.
    Function *F = VArg->getParent();
    BasicBlock *BB = &F->getEntryBlock();
    return Scatterer(BB, BB->begin(), V, &Scattered[V]);
  }
  if (Instruction *VOp = dyn_cast<Instruction>(V)) {
    // When scalarizing PHI nodes we might try to examine/rewrite InsertElement
    // nodes in predecessors. If those predecessors are unreachable from entry,
    // then the IR in those blocks could have unexpected properties resulting in
    // infinite loops in Scatterer::operator[]. By simply treating values
    // originating from instructions in unreachable blocks as undef we do not
    // need to analyse them further.
    if (!DT->isReachableFromEntry(VOp->getParent()))
      return Scatterer(Point->getParent(), Point->getIterator(),
                       UndefValue::get(V->getType()));
    // Put the scattered form of an instruction directly after the
    // instruction, skipping over PHI nodes and debug intrinsics.
    BasicBlock *BB = VOp->getParent();
    return Scatterer(
        BB, skipPastPhiNodesAndDbg(std::next(BasicBlock::iterator(VOp))), V,
        &Scattered[V]);
  }
  // In the fallback case, just put the scattered before Point and
  // keep the result local to Point.
  return Scatterer(Point->getParent(), Point->getIterator(), V);
}

// Replace Op with the gathered form of the components in CV.  Defer the
// deletion of Op and creation of the gathered form to the end of the pass,
// so that we can avoid creating the gathered form if all uses of Op are
// replaced with uses of CV.
void ScalarizerVisitor::gather(Instruction *Op, const ValueVector &CV) {
  transferMetadataAndIRFlags(Op, CV);

  // If we already have a scattered form of Op (created from ExtractElements
  // of Op itself), replace them with the new form.
  ValueVector &SV = Scattered[Op];
  if (!SV.empty()) {
    for (unsigned I = 0, E = SV.size(); I != E; ++I) {
      Value *V = SV[I];
      if (V == nullptr || SV[I] == CV[I])
        continue;

      Instruction *Old = cast<Instruction>(V);
      if (isa<Instruction>(CV[I]))
        CV[I]->takeName(Old);
      Old->replaceAllUsesWith(CV[I]);
      PotentiallyDeadInstrs.emplace_back(Old);
    }
  }
  SV = CV;
  Gathered.push_back(GatherList::value_type(Op, &SV));
}

// Return true if it is safe to transfer the given metadata tag from
// vector to scalar instructions.
bool ScalarizerVisitor::canTransferMetadata(unsigned Tag) {
  return (Tag == LLVMContext::MD_tbaa
          || Tag == LLVMContext::MD_fpmath
          || Tag == LLVMContext::MD_tbaa_struct
          || Tag == LLVMContext::MD_invariant_load
          || Tag == LLVMContext::MD_alias_scope
          || Tag == LLVMContext::MD_noalias
          || Tag == ParallelLoopAccessMDKind
          || Tag == LLVMContext::MD_access_group);
}

// Transfer metadata from Op to the instructions in CV if it is known
// to be safe to do so.
void ScalarizerVisitor::transferMetadataAndIRFlags(Instruction *Op,
                                                   const ValueVector &CV) {
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  Op->getAllMetadataOtherThanDebugLoc(MDs);
  for (unsigned I = 0, E = CV.size(); I != E; ++I) {
    if (Instruction *New = dyn_cast<Instruction>(CV[I])) {
      for (const auto &MD : MDs)
        if (canTransferMetadata(MD.first))
          New->setMetadata(MD.first, MD.second);
      New->copyIRFlags(Op);
      if (Op->getDebugLoc() && !New->getDebugLoc())
        New->setDebugLoc(Op->getDebugLoc());
    }
  }
}

// Try to fill in Layout from Ty, returning true on success.  Alignment is
// the alignment of the vector, or None if the ABI default should be used.
Optional<VectorLayout>
ScalarizerVisitor::getVectorLayout(Type *Ty, Align Alignment,
                                   const DataLayout &DL) {
  VectorLayout Layout;
  // Make sure we're dealing with a vector.
  Layout.VecTy = dyn_cast<VectorType>(Ty);
  if (!Layout.VecTy)
    return None;
  // Check that we're dealing with full-byte elements.
  Layout.ElemTy = Layout.VecTy->getElementType();
  if (!DL.typeSizeEqualsStoreSize(Layout.ElemTy))
    return None;
  Layout.VecAlign = Alignment;
  Layout.ElemSize = DL.getTypeStoreSize(Layout.ElemTy);
  return Layout;
}

// Scalarize one-operand instruction I, using Split(Builder, X, Name)
// to create an instruction like I with operand X and name Name.
template<typename Splitter>
bool ScalarizerVisitor::splitUnary(Instruction &I, const Splitter &Split) {
  VectorType *VT = dyn_cast<VectorType>(I.getType());
  if (!VT)
    return false;

  unsigned NumElems = cast<FixedVectorType>(VT)->getNumElements();
  IRBuilder<> Builder(&I);
  Scatterer Op = scatter(&I, I.getOperand(0));
  assert(Op.size() == NumElems && "Mismatched unary operation");
  ValueVector Res;
  Res.resize(NumElems);
  for (unsigned Elem = 0; Elem < NumElems; ++Elem)
    Res[Elem] = Split(Builder, Op[Elem], I.getName() + ".i" + Twine(Elem));
  gather(&I, Res);
  return true;
}

// Scalarize two-operand instruction I, using Split(Builder, X, Y, Name)
// to create an instruction like I with operands X and Y and name Name.
template<typename Splitter>
bool ScalarizerVisitor::splitBinary(Instruction &I, const Splitter &Split) {
  VectorType *VT = dyn_cast<VectorType>(I.getType());
  if (!VT)
    return false;

  unsigned NumElems = cast<FixedVectorType>(VT)->getNumElements();
  IRBuilder<> Builder(&I);
  Scatterer VOp0 = scatter(&I, I.getOperand(0));
  Scatterer VOp1 = scatter(&I, I.getOperand(1));
  assert(VOp0.size() == NumElems && "Mismatched binary operation");
  assert(VOp1.size() == NumElems && "Mismatched binary operation");
  ValueVector Res;
  Res.resize(NumElems);
  for (unsigned Elem = 0; Elem < NumElems; ++Elem) {
    Value *Op0 = VOp0[Elem];
    Value *Op1 = VOp1[Elem];
    Res[Elem] = Split(Builder, Op0, Op1, I.getName() + ".i" + Twine(Elem));
  }
  gather(&I, Res);
  return true;
}

static bool isTriviallyScalariable(Intrinsic::ID ID) {
  return isTriviallyVectorizable(ID);
}

// All of the current scalarizable intrinsics only have one mangled type.
static Function *getScalarIntrinsicDeclaration(Module *M,
                                               Intrinsic::ID ID,
                                               ArrayRef<Type*> Tys) {
  return Intrinsic::getDeclaration(M, ID, Tys);
}

/// If a call to a vector typed intrinsic function, split into a scalar call per
/// element if possible for the intrinsic.
bool ScalarizerVisitor::splitCall(CallInst &CI) {
  VectorType *VT = dyn_cast<VectorType>(CI.getType());
  if (!VT)
    return false;

  Function *F = CI.getCalledFunction();
  if (!F)
    return false;

  Intrinsic::ID ID = F->getIntrinsicID();
  if (ID == Intrinsic::not_intrinsic || !isTriviallyScalariable(ID))
    return false;

  unsigned NumElems = cast<FixedVectorType>(VT)->getNumElements();
  unsigned NumArgs = CI.arg_size();

  ValueVector ScalarOperands(NumArgs);
  SmallVector<Scatterer, 8> Scattered(NumArgs);

  Scattered.resize(NumArgs);

  SmallVector<llvm::Type *, 3> Tys;
  Tys.push_back(VT->getScalarType());

  // Assumes that any vector type has the same number of elements as the return
  // vector type, which is true for all current intrinsics.
  for (unsigned I = 0; I != NumArgs; ++I) {
    Value *OpI = CI.getOperand(I);
    if (OpI->getType()->isVectorTy()) {
      Scattered[I] = scatter(&CI, OpI);
      assert(Scattered[I].size() == NumElems && "mismatched call operands");
    } else {
      ScalarOperands[I] = OpI;
      if (hasVectorInstrinsicOverloadedScalarOpd(ID, I))
        Tys.push_back(OpI->getType());
    }
  }

  ValueVector Res(NumElems);
  ValueVector ScalarCallOps(NumArgs);

  Function *NewIntrin = getScalarIntrinsicDeclaration(F->getParent(), ID, Tys);
  IRBuilder<> Builder(&CI);

  // Perform actual scalarization, taking care to preserve any scalar operands.
  for (unsigned Elem = 0; Elem < NumElems; ++Elem) {
    ScalarCallOps.clear();

    for (unsigned J = 0; J != NumArgs; ++J) {
      if (hasVectorInstrinsicScalarOpd(ID, J))
        ScalarCallOps.push_back(ScalarOperands[J]);
      else
        ScalarCallOps.push_back(Scattered[J][Elem]);
    }

    Res[Elem] = Builder.CreateCall(NewIntrin, ScalarCallOps,
                                   CI.getName() + ".i" + Twine(Elem));
  }

  gather(&CI, Res);
  return true;
}

bool ScalarizerVisitor::visitSelectInst(SelectInst &SI) {
  VectorType *VT = dyn_cast<VectorType>(SI.getType());
  if (!VT)
    return false;

  unsigned NumElems = cast<FixedVectorType>(VT)->getNumElements();
  IRBuilder<> Builder(&SI);
  Scatterer VOp1 = scatter(&SI, SI.getOperand(1));
  Scatterer VOp2 = scatter(&SI, SI.getOperand(2));
  assert(VOp1.size() == NumElems && "Mismatched select");
  assert(VOp2.size() == NumElems && "Mismatched select");
  ValueVector Res;
  Res.resize(NumElems);

  if (SI.getOperand(0)->getType()->isVectorTy()) {
    Scatterer VOp0 = scatter(&SI, SI.getOperand(0));
    assert(VOp0.size() == NumElems && "Mismatched select");
    for (unsigned I = 0; I < NumElems; ++I) {
      Value *Op0 = VOp0[I];
      Value *Op1 = VOp1[I];
      Value *Op2 = VOp2[I];
      Res[I] = Builder.CreateSelect(Op0, Op1, Op2,
                                    SI.getName() + ".i" + Twine(I));
    }
  } else {
    Value *Op0 = SI.getOperand(0);
    for (unsigned I = 0; I < NumElems; ++I) {
      Value *Op1 = VOp1[I];
      Value *Op2 = VOp2[I];
      Res[I] = Builder.CreateSelect(Op0, Op1, Op2,
                                    SI.getName() + ".i" + Twine(I));
    }
  }
  gather(&SI, Res);
  return true;
}

bool ScalarizerVisitor::visitICmpInst(ICmpInst &ICI) {
  return splitBinary(ICI, ICmpSplitter(ICI));
}

bool ScalarizerVisitor::visitFCmpInst(FCmpInst &FCI) {
  return splitBinary(FCI, FCmpSplitter(FCI));
}

bool ScalarizerVisitor::visitUnaryOperator(UnaryOperator &UO) {
  return splitUnary(UO, UnarySplitter(UO));
}

bool ScalarizerVisitor::visitBinaryOperator(BinaryOperator &BO) {
  return splitBinary(BO, BinarySplitter(BO));
}

bool ScalarizerVisitor::visitGetElementPtrInst(GetElementPtrInst &GEPI) {
  VectorType *VT = dyn_cast<VectorType>(GEPI.getType());
  if (!VT)
    return false;

  IRBuilder<> Builder(&GEPI);
  unsigned NumElems = cast<FixedVectorType>(VT)->getNumElements();
  unsigned NumIndices = GEPI.getNumIndices();

  // The base pointer might be scalar even if it's a vector GEP. In those cases,
  // splat the pointer into a vector value, and scatter that vector.
  Value *Op0 = GEPI.getOperand(0);
  if (!Op0->getType()->isVectorTy())
    Op0 = Builder.CreateVectorSplat(NumElems, Op0);
  Scatterer Base = scatter(&GEPI, Op0);

  SmallVector<Scatterer, 8> Ops;
  Ops.resize(NumIndices);
  for (unsigned I = 0; I < NumIndices; ++I) {
    Value *Op = GEPI.getOperand(I + 1);

    // The indices might be scalars even if it's a vector GEP. In those cases,
    // splat the scalar into a vector value, and scatter that vector.
    if (!Op->getType()->isVectorTy())
      Op = Builder.CreateVectorSplat(NumElems, Op);

    Ops[I] = scatter(&GEPI, Op);
  }

  ValueVector Res;
  Res.resize(NumElems);
  for (unsigned I = 0; I < NumElems; ++I) {
    SmallVector<Value *, 8> Indices;
    Indices.resize(NumIndices);
    for (unsigned J = 0; J < NumIndices; ++J)
      Indices[J] = Ops[J][I];
    Res[I] = Builder.CreateGEP(GEPI.getSourceElementType(), Base[I], Indices,
                               GEPI.getName() + ".i" + Twine(I));
    if (GEPI.isInBounds())
      if (GetElementPtrInst *NewGEPI = dyn_cast<GetElementPtrInst>(Res[I]))
        NewGEPI->setIsInBounds();
  }
  gather(&GEPI, Res);
  return true;
}

bool ScalarizerVisitor::visitCastInst(CastInst &CI) {
  VectorType *VT = dyn_cast<VectorType>(CI.getDestTy());
  if (!VT)
    return false;

  unsigned NumElems = cast<FixedVectorType>(VT)->getNumElements();
  IRBuilder<> Builder(&CI);
  Scatterer Op0 = scatter(&CI, CI.getOperand(0));
  assert(Op0.size() == NumElems && "Mismatched cast");
  ValueVector Res;
  Res.resize(NumElems);
  for (unsigned I = 0; I < NumElems; ++I)
    Res[I] = Builder.CreateCast(CI.getOpcode(), Op0[I], VT->getElementType(),
                                CI.getName() + ".i" + Twine(I));
  gather(&CI, Res);
  return true;
}

bool ScalarizerVisitor::visitBitCastInst(BitCastInst &BCI) {
  VectorType *DstVT = dyn_cast<VectorType>(BCI.getDestTy());
  VectorType *SrcVT = dyn_cast<VectorType>(BCI.getSrcTy());
  if (!DstVT || !SrcVT)
    return false;

  unsigned DstNumElems = cast<FixedVectorType>(DstVT)->getNumElements();
  unsigned SrcNumElems = cast<FixedVectorType>(SrcVT)->getNumElements();
  IRBuilder<> Builder(&BCI);
  Scatterer Op0 = scatter(&BCI, BCI.getOperand(0));
  ValueVector Res;
  Res.resize(DstNumElems);

  if (DstNumElems == SrcNumElems) {
    for (unsigned I = 0; I < DstNumElems; ++I)
      Res[I] = Builder.CreateBitCast(Op0[I], DstVT->getElementType(),
                                     BCI.getName() + ".i" + Twine(I));
  } else if (DstNumElems > SrcNumElems) {
    // <M x t1> -> <N*M x t2>.  Convert each t1 to <N x t2> and copy the
    // individual elements to the destination.
    unsigned FanOut = DstNumElems / SrcNumElems;
    auto *MidTy = FixedVectorType::get(DstVT->getElementType(), FanOut);
    unsigned ResI = 0;
    for (unsigned Op0I = 0; Op0I < SrcNumElems; ++Op0I) {
      Value *V = Op0[Op0I];
      Instruction *VI;
      // Look through any existing bitcasts before converting to <N x t2>.
      // In the best case, the resulting conversion might be a no-op.
      while ((VI = dyn_cast<Instruction>(V)) &&
             VI->getOpcode() == Instruction::BitCast)
        V = VI->getOperand(0);
      V = Builder.CreateBitCast(V, MidTy, V->getName() + ".cast");
      Scatterer Mid = scatter(&BCI, V);
      for (unsigned MidI = 0; MidI < FanOut; ++MidI)
        Res[ResI++] = Mid[MidI];
    }
  } else {
    // <N*M x t1> -> <M x t2>.  Convert each group of <N x t1> into a t2.
    unsigned FanIn = SrcNumElems / DstNumElems;
    auto *MidTy = FixedVectorType::get(SrcVT->getElementType(), FanIn);
    unsigned Op0I = 0;
    for (unsigned ResI = 0; ResI < DstNumElems; ++ResI) {
      Value *V = PoisonValue::get(MidTy);
      for (unsigned MidI = 0; MidI < FanIn; ++MidI)
        V = Builder.CreateInsertElement(V, Op0[Op0I++], Builder.getInt32(MidI),
                                        BCI.getName() + ".i" + Twine(ResI)
                                        + ".upto" + Twine(MidI));
      Res[ResI] = Builder.CreateBitCast(V, DstVT->getElementType(),
                                        BCI.getName() + ".i" + Twine(ResI));
    }
  }
  gather(&BCI, Res);
  return true;
}

bool ScalarizerVisitor::visitInsertElementInst(InsertElementInst &IEI) {
  VectorType *VT = dyn_cast<VectorType>(IEI.getType());
  if (!VT)
    return false;

  unsigned NumElems = cast<FixedVectorType>(VT)->getNumElements();
  IRBuilder<> Builder(&IEI);
  Scatterer Op0 = scatter(&IEI, IEI.getOperand(0));
  Value *NewElt = IEI.getOperand(1);
  Value *InsIdx = IEI.getOperand(2);

  ValueVector Res;
  Res.resize(NumElems);

  if (auto *CI = dyn_cast<ConstantInt>(InsIdx)) {
    for (unsigned I = 0; I < NumElems; ++I)
      Res[I] = CI->getValue().getZExtValue() == I ? NewElt : Op0[I];
  } else {
    if (!ScalarizeVariableInsertExtract)
      return false;

    for (unsigned I = 0; I < NumElems; ++I) {
      Value *ShouldReplace =
          Builder.CreateICmpEQ(InsIdx, ConstantInt::get(InsIdx->getType(), I),
                               InsIdx->getName() + ".is." + Twine(I));
      Value *OldElt = Op0[I];
      Res[I] = Builder.CreateSelect(ShouldReplace, NewElt, OldElt,
                                    IEI.getName() + ".i" + Twine(I));
    }
  }

  gather(&IEI, Res);
  return true;
}

bool ScalarizerVisitor::visitExtractElementInst(ExtractElementInst &EEI) {
  VectorType *VT = dyn_cast<VectorType>(EEI.getOperand(0)->getType());
  if (!VT)
    return false;

  unsigned NumSrcElems = cast<FixedVectorType>(VT)->getNumElements();
  IRBuilder<> Builder(&EEI);
  Scatterer Op0 = scatter(&EEI, EEI.getOperand(0));
  Value *ExtIdx = EEI.getOperand(1);

  if (auto *CI = dyn_cast<ConstantInt>(ExtIdx)) {
    Value *Res = Op0[CI->getValue().getZExtValue()];
    gather(&EEI, {Res});
    return true;
  }

  if (!ScalarizeVariableInsertExtract)
    return false;

  Value *Res = UndefValue::get(VT->getElementType());
  for (unsigned I = 0; I < NumSrcElems; ++I) {
    Value *ShouldExtract =
        Builder.CreateICmpEQ(ExtIdx, ConstantInt::get(ExtIdx->getType(), I),
                             ExtIdx->getName() + ".is." + Twine(I));
    Value *Elt = Op0[I];
    Res = Builder.CreateSelect(ShouldExtract, Elt, Res,
                               EEI.getName() + ".upto" + Twine(I));
  }
  gather(&EEI, {Res});
  return true;
}

bool ScalarizerVisitor::visitShuffleVectorInst(ShuffleVectorInst &SVI) {
  VectorType *VT = dyn_cast<VectorType>(SVI.getType());
  if (!VT)
    return false;

  unsigned NumElems = cast<FixedVectorType>(VT)->getNumElements();
  Scatterer Op0 = scatter(&SVI, SVI.getOperand(0));
  Scatterer Op1 = scatter(&SVI, SVI.getOperand(1));
  ValueVector Res;
  Res.resize(NumElems);

  for (unsigned I = 0; I < NumElems; ++I) {
    int Selector = SVI.getMaskValue(I);
    if (Selector < 0)
      Res[I] = UndefValue::get(VT->getElementType());
    else if (unsigned(Selector) < Op0.size())
      Res[I] = Op0[Selector];
    else
      Res[I] = Op1[Selector - Op0.size()];
  }
  gather(&SVI, Res);
  return true;
}

bool ScalarizerVisitor::visitPHINode(PHINode &PHI) {
  VectorType *VT = dyn_cast<VectorType>(PHI.getType());
  if (!VT)
    return false;

  unsigned NumElems = cast<FixedVectorType>(VT)->getNumElements();
  IRBuilder<> Builder(&PHI);
  ValueVector Res;
  Res.resize(NumElems);

  unsigned NumOps = PHI.getNumOperands();
  for (unsigned I = 0; I < NumElems; ++I)
    Res[I] = Builder.CreatePHI(VT->getElementType(), NumOps,
                               PHI.getName() + ".i" + Twine(I));

  for (unsigned I = 0; I < NumOps; ++I) {
    Scatterer Op = scatter(&PHI, PHI.getIncomingValue(I));
    BasicBlock *IncomingBlock = PHI.getIncomingBlock(I);
    for (unsigned J = 0; J < NumElems; ++J)
      cast<PHINode>(Res[J])->addIncoming(Op[J], IncomingBlock);
  }
  gather(&PHI, Res);
  return true;
}

bool ScalarizerVisitor::visitLoadInst(LoadInst &LI) {
  if (!ScalarizeLoadStore)
    return false;
  if (!LI.isSimple())
    return false;

  Optional<VectorLayout> Layout = getVectorLayout(
      LI.getType(), LI.getAlign(), LI.getModule()->getDataLayout());
  if (!Layout)
    return false;

  unsigned NumElems = cast<FixedVectorType>(Layout->VecTy)->getNumElements();
  IRBuilder<> Builder(&LI);
  Scatterer Ptr = scatter(&LI, LI.getPointerOperand());
  ValueVector Res;
  Res.resize(NumElems);

  for (unsigned I = 0; I < NumElems; ++I)
    Res[I] = Builder.CreateAlignedLoad(Layout->VecTy->getElementType(), Ptr[I],
                                       Align(Layout->getElemAlign(I)),
                                       LI.getName() + ".i" + Twine(I));
  gather(&LI, Res);
  return true;
}

bool ScalarizerVisitor::visitStoreInst(StoreInst &SI) {
  if (!ScalarizeLoadStore)
    return false;
  if (!SI.isSimple())
    return false;

  Value *FullValue = SI.getValueOperand();
  Optional<VectorLayout> Layout = getVectorLayout(
      FullValue->getType(), SI.getAlign(), SI.getModule()->getDataLayout());
  if (!Layout)
    return false;

  unsigned NumElems = cast<FixedVectorType>(Layout->VecTy)->getNumElements();
  IRBuilder<> Builder(&SI);
  Scatterer VPtr = scatter(&SI, SI.getPointerOperand());
  Scatterer VVal = scatter(&SI, FullValue);

  ValueVector Stores;
  Stores.resize(NumElems);
  for (unsigned I = 0; I < NumElems; ++I) {
    Value *Val = VVal[I];
    Value *Ptr = VPtr[I];
    Stores[I] = Builder.CreateAlignedStore(Val, Ptr, Layout->getElemAlign(I));
  }
  transferMetadataAndIRFlags(&SI, Stores);
  return true;
}

bool ScalarizerVisitor::visitCallInst(CallInst &CI) {
  return splitCall(CI);
}

// Delete the instructions that we scalarized.  If a full vector result
// is still needed, recreate it using InsertElements.
bool ScalarizerVisitor::finish() {
  // The presence of data in Gathered or Scattered indicates changes
  // made to the Function.
  if (Gathered.empty() && Scattered.empty())
    return false;
  for (const auto &GMI : Gathered) {
    Instruction *Op = GMI.first;
    ValueVector &CV = *GMI.second;
    if (!Op->use_empty()) {
      // The value is still needed, so recreate it using a series of
      // InsertElements.
      Value *Res = PoisonValue::get(Op->getType());
      if (auto *Ty = dyn_cast<VectorType>(Op->getType())) {
        BasicBlock *BB = Op->getParent();
        unsigned Count = cast<FixedVectorType>(Ty)->getNumElements();
        IRBuilder<> Builder(Op);
        if (isa<PHINode>(Op))
          Builder.SetInsertPoint(BB, BB->getFirstInsertionPt());
        for (unsigned I = 0; I < Count; ++I)
          Res = Builder.CreateInsertElement(Res, CV[I], Builder.getInt32(I),
                                            Op->getName() + ".upto" + Twine(I));
        Res->takeName(Op);
      } else {
        assert(CV.size() == 1 && Op->getType() == CV[0]->getType());
        Res = CV[0];
        if (Op == Res)
          continue;
      }
      Op->replaceAllUsesWith(Res);
    }
    PotentiallyDeadInstrs.emplace_back(Op);
  }
  Gathered.clear();
  Scattered.clear();

  RecursivelyDeleteTriviallyDeadInstructionsPermissive(PotentiallyDeadInstrs);

  return true;
}

PreservedAnalyses ScalarizerPass::run(Function &F, FunctionAnalysisManager &AM) {
  Module &M = *F.getParent();
  unsigned ParallelLoopAccessMDKind =
      M.getContext().getMDKindID("llvm.mem.parallel_loop_access");
  DominatorTree *DT = &AM.getResult<DominatorTreeAnalysis>(F);
  ScalarizerVisitor Impl(ParallelLoopAccessMDKind, DT);
  bool Changed = Impl.visit(F);
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  return Changed ? PA : PreservedAnalyses::all();
}
