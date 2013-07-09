//===- SLPVectorizer.cpp - A bottom up SLP Vectorizer ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass implements the Bottom Up SLP vectorizer. It detects consecutive
// stores that can be put together into vector-stores. Next, it attempts to
// construct vectorizable tree using the use-def chains. If a profitable tree
// was found, the SLP vectorizer performs vectorization on the tree.
//
// The pass is inspired by the work described in the paper:
//  "Loop-Aware SLP in GCC" by Ira Rosen, Dorit Nuzman, Ayal Zaks.
//
//===----------------------------------------------------------------------===//
#define SV_NAME "slp-vectorizer"
#define DEBUG_TYPE "SLP"

#include "llvm/Transforms/Vectorize.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>

using namespace llvm;

static cl::opt<int>
    SLPCostThreshold("slp-threshold", cl::init(0), cl::Hidden,
                     cl::desc("Only vectorize if you gain more than this "
                              "number "));
namespace {

static const unsigned MinVecRegSize = 128;

static const unsigned RecursionMaxDepth = 12;

/// RAII pattern to save the insertion point of the IR builder.
class BuilderLocGuard {
public:
  BuilderLocGuard(IRBuilder<> &B) : Builder(B), Loc(B.GetInsertPoint()) {}
  ~BuilderLocGuard() { if (Loc) Builder.SetInsertPoint(Loc); }

private:
  // Prevent copying.
  BuilderLocGuard(const BuilderLocGuard &);
  BuilderLocGuard &operator=(const BuilderLocGuard &);
  IRBuilder<> &Builder;
  AssertingVH<Instruction> Loc;
};

/// A helper class for numbering instructions in multible blocks.
/// Numbers starts at zero for each basic block.
struct BlockNumbering {

  BlockNumbering(BasicBlock *Bb) : BB(Bb), Valid(false) {}

  BlockNumbering() : BB(0), Valid(false) {}

  void numberInstructions() {
    unsigned Loc = 0;
    InstrIdx.clear();
    InstrVec.clear();
    // Number the instructions in the block.
    for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
      InstrIdx[it] = Loc++;
      InstrVec.push_back(it);
      assert(InstrVec[InstrIdx[it]] == it && "Invalid allocation");
    }
    Valid = true;
  }

  int getIndex(Instruction *I) {
    assert(I->getParent() == BB && "Invalid instruction");
    if (!Valid)
      numberInstructions();
    assert(InstrIdx.count(I) && "Unknown instruction");
    return InstrIdx[I];
  }

  Instruction *getInstruction(unsigned loc) {
    if (!Valid)
      numberInstructions();
    assert(InstrVec.size() > loc && "Invalid Index");
    return InstrVec[loc];
  }

  void forget() { Valid = false; }

private:
  /// The block we are numbering.
  BasicBlock *BB;
  /// Is the block numbered.
  bool Valid;
  /// Maps instructions to numbers and back.
  SmallDenseMap<Instruction *, int> InstrIdx;
  /// Maps integers to Instructions.
  std::vector<Instruction *> InstrVec;
};

/// \returns the parent basic block if all of the instructions in \p VL
/// are in the same block or null otherwise.
static BasicBlock *getSameBlock(ArrayRef<Value *> VL) {
  Instruction *I0 = dyn_cast<Instruction>(VL[0]);
  if (!I0)
    return 0;
  BasicBlock *BB = I0->getParent();
  for (int i = 1, e = VL.size(); i < e; i++) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    if (!I)
      return 0;

    if (BB != I->getParent())
      return 0;
  }
  return BB;
}

/// \returns True if all of the values in \p VL are constants.
static bool allConstant(ArrayRef<Value *> VL) {
  for (unsigned i = 0, e = VL.size(); i < e; ++i)
    if (!isa<Constant>(VL[i]))
      return false;
  return true;
}

/// \returns True if all of the values in \p VL are identical.
static bool isSplat(ArrayRef<Value *> VL) {
  for (unsigned i = 1, e = VL.size(); i < e; ++i)
    if (VL[i] != VL[0])
      return false;
  return true;
}

/// \returns The opcode if all of the Instructions in \p VL have the same
/// opcode, or zero.
static unsigned getSameOpcode(ArrayRef<Value *> VL) {
  Instruction *I0 = dyn_cast<Instruction>(VL[0]);
  if (!I0)
    return 0;
  unsigned Opcode = I0->getOpcode();
  for (int i = 1, e = VL.size(); i < e; i++) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    if (!I || Opcode != I->getOpcode())
      return 0;
  }
  return Opcode;
}

/// \returns The type that all of the values in \p VL have or null if there
/// are different types.
static Type* getSameType(ArrayRef<Value *> VL) {
  Type *Ty = VL[0]->getType();
  for (int i = 1, e = VL.size(); i < e; i++)
    if (VL[0]->getType() != Ty)
      return 0;

  return Ty;
}

/// \returns True if the ExtractElement instructions in VL can be vectorized
/// to use the original vector.
static bool CanReuseExtract(ArrayRef<Value *> VL) {
  assert(Instruction::ExtractElement == getSameOpcode(VL) && "Invalid opcode");
  // Check if all of the extracts come from the same vector and from the
  // correct offset.
  Value *VL0 = VL[0];
  ExtractElementInst *E0 = cast<ExtractElementInst>(VL0);
  Value *Vec = E0->getOperand(0);

  // We have to extract from the same vector type.
  unsigned NElts = Vec->getType()->getVectorNumElements();

  if (NElts != VL.size())
    return false;

  // Check that all of the indices extract from the correct offset.
  ConstantInt *CI = dyn_cast<ConstantInt>(E0->getOperand(1));
  if (!CI || CI->getZExtValue())
    return false;

  for (unsigned i = 1, e = VL.size(); i < e; ++i) {
    ExtractElementInst *E = cast<ExtractElementInst>(VL[i]);
    ConstantInt *CI = dyn_cast<ConstantInt>(E->getOperand(1));

    if (!CI || CI->getZExtValue() != i || E->getOperand(0) != Vec)
      return false;
  }

  return true;
}

/// Bottom Up SLP Vectorizer.
class BoUpSLP {
public:
  typedef SmallVector<Value *, 8> ValueList;
  typedef SmallVector<Instruction *, 16> InstrList;
  typedef SmallPtrSet<Value *, 16> ValueSet;
  typedef SmallVector<StoreInst *, 8> StoreList;

  BoUpSLP(Function *Func, ScalarEvolution *Se, DataLayout *Dl,
          TargetTransformInfo *Tti, AliasAnalysis *Aa, LoopInfo *Li,
          DominatorTree *Dt) :
    F(Func), SE(Se), DL(Dl), TTI(Tti), AA(Aa), LI(Li), DT(Dt),
    Builder(Se->getContext()) {
      // Setup the block numbering utility for all of the blocks in the
      // function.
      for (Function::iterator it = F->begin(), e = F->end(); it != e; ++it) {
        BasicBlock *BB = it;
        BlocksNumbers[BB] = BlockNumbering(BB);
      }
    }

  /// \brief Vectorize the tree that starts with the elements in \p VL.
  void vectorizeTree();

  /// \returns the vectorization cost of the subtree that starts at \p VL.
  /// A negative number means that this is profitable.
  int getTreeCost();

  /// Construct a vectorizable tree that starts at \p Roots.
  void buildTree(ArrayRef<Value *> Roots);

  /// Clear the internal data structures that are created by 'buildTree'.
  void deleteTree() {
    VectorizableTree.clear();
    ScalarToTreeEntry.clear();
    MustGather.clear();
    MemBarrierIgnoreList.clear();
  }

  /// \returns the scalarization cost for this list of values. Assuming that
  /// this subtree gets vectorized, we may need to extract the values from the
  /// roots. This method calculates the cost of extracting the values.
  int getGatherCost(ArrayRef<Value *> VL);

  /// \returns true if the memory operations A and B are consecutive.
  bool isConsecutiveAccess(Value *A, Value *B);

  /// \brief Perform LICM and CSE on the newly generated gather sequences.
  void optimizeGatherSequence();
private:
  struct TreeEntry;

  /// \returns the cost of the vectorizable entry.
  int getEntryCost(TreeEntry *E);

  /// This is the recursive part of buildTree.
  void buildTree_rec(ArrayRef<Value *> Roots, unsigned Depth);

  /// Vectorizer a single entry in the tree.
  Value *vectorizeTree(TreeEntry *E);

  /// Vectorizer a single entry in the tree, starting in \p VL.
  Value *vectorizeTree(ArrayRef<Value *> VL);

  /// \brief Take the pointer operand from the Load/Store instruction.
  /// \returns NULL if this is not a valid Load/Store instruction.
  static Value *getPointerOperand(Value *I);

  /// \brief Take the address space operand from the Load/Store instruction.
  /// \returns -1 if this is not a valid Load/Store instruction.
  static unsigned getAddressSpaceOperand(Value *I);

  /// \returns the scalarization cost for this type. Scalarization in this
  /// context means the creation of vectors from a group of scalars.
  int getGatherCost(Type *Ty);

  /// \returns the AA location that is being access by the instruction.
  AliasAnalysis::Location getLocation(Instruction *I);

  /// \brief Checks if it is possible to sink an instruction from
  /// \p Src to \p Dst.
  /// \returns the pointer to the barrier instruction if we can't sink.
  Value *getSinkBarrier(Instruction *Src, Instruction *Dst);

  /// \returns the index of the last instrucion in the BB from \p VL.
  int getLastIndex(ArrayRef<Value *> VL);

  /// \returns the Instrucion in the bundle \p VL.
  Instruction *getLastInstruction(ArrayRef<Value *> VL);

  /// \returns the Instruction at index \p Index which is in Block \p BB.
  Instruction *getInstructionForIndex(unsigned Index, BasicBlock *BB);

  /// \returns the index of the first User of \p VL.
  int getFirstUserIndex(ArrayRef<Value *> VL);

  /// \returns a vector from a collection of scalars in \p VL.
  Value *Gather(ArrayRef<Value *> VL, VectorType *Ty);

  struct TreeEntry {
    TreeEntry() : Scalars(), VectorizedValue(0), LastScalarIndex(0),
    NeedToGather(0) {}

    /// \returns true if the scalars in VL are equal to this entry.
    bool isSame(ArrayRef<Value *> VL) {
      assert(VL.size() == Scalars.size() && "Invalid size");
      for (int i = 0, e = VL.size(); i != e; ++i)
        if (VL[i] != Scalars[i])
          return false;
      return true;
    }

    /// A vector of scalars.
    ValueList Scalars;

    /// The Scalars are vectorized into this value. It is initialized to Null.
    Value *VectorizedValue;

    /// The index in the basic block of the last scalar.
    int LastScalarIndex;

    /// Do we need to gather this sequence ?
    bool NeedToGather;
  };

  /// Create a new VectorizableTree entry.
  TreeEntry *newTreeEntry(ArrayRef<Value *> VL, bool Vectorized) {
    VectorizableTree.push_back(TreeEntry());
    int idx = VectorizableTree.size() - 1;
    TreeEntry *Last = &VectorizableTree[idx];
    Last->Scalars.insert(Last->Scalars.begin(), VL.begin(), VL.end());
    Last->NeedToGather = !Vectorized;
    if (Vectorized) {
      Last->LastScalarIndex = getLastIndex(VL);
      for (int i = 0, e = VL.size(); i != e; ++i) {
        assert(!ScalarToTreeEntry.count(VL[i]) && "Scalar already in tree!");
        ScalarToTreeEntry[VL[i]] = idx;
      }
    } else {
      Last->LastScalarIndex = 0;
      MustGather.insert(VL.begin(), VL.end());
    }
    return Last;
  }

  /// -- Vectorization State --
  /// Holds all of the tree entries.
  std::vector<TreeEntry> VectorizableTree;

  /// Maps a specific scalar to its tree entry.
  SmallDenseMap<Value*, int> ScalarToTreeEntry;

  /// A list of scalars that we found that we need to keep as scalars.
  ValueSet MustGather;

  /// A list of instructions to ignore while sinking
  /// memory instructions. This map must be reset between runs of getCost.
  ValueSet MemBarrierIgnoreList;

  /// Holds all of the instructions that we gathered.
  SetVector<Instruction *> GatherSeq;

  /// Numbers instructions in different blocks.
  std::map<BasicBlock *, BlockNumbering> BlocksNumbers;

  // Analysis and block reference.
  Function *F;
  ScalarEvolution *SE;
  DataLayout *DL;
  TargetTransformInfo *TTI;
  AliasAnalysis *AA;
  LoopInfo *LI;
  DominatorTree *DT;
  /// Instruction builder to construct the vectorized tree.
  IRBuilder<> Builder;
};

void BoUpSLP::buildTree(ArrayRef<Value *> Roots) {
  deleteTree();
  buildTree_rec(Roots, 0);
}


void BoUpSLP::buildTree_rec(ArrayRef<Value *> VL, unsigned Depth) {
  bool SameTy = getSameType(VL); (void)SameTy;
  assert(SameTy && "Invalid types!");

  if (Depth == RecursionMaxDepth) {
    DEBUG(dbgs() << "SLP: Gathering due to max recursion depth.\n");
    newTreeEntry(VL, false);
    return;
  }

  // Don't handle vectors.
  if (VL[0]->getType()->isVectorTy()) {
    DEBUG(dbgs() << "SLP: Gathering due to vector type.\n");
    newTreeEntry(VL, false);
    return;
  }

  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    if (SI->getValueOperand()->getType()->isVectorTy()) {
      DEBUG(dbgs() << "SLP: Gathering due to store vector type.\n");
      newTreeEntry(VL, false);
      return;
    }

  // If all of the operands are identical or constant we have a simple solution.
  if (allConstant(VL) || isSplat(VL) || !getSameBlock(VL) ||
      !getSameOpcode(VL)) {
    DEBUG(dbgs() << "SLP: Gathering due to C,S,B,O. \n");
    newTreeEntry(VL, false);
    return;
  }

  // We now know that this is a vector of instructions of the same type from
  // the same block.

  // Check if this is a duplicate of another entry.
  if (ScalarToTreeEntry.count(VL[0])) {
    int Idx = ScalarToTreeEntry[VL[0]];
    TreeEntry *E = &VectorizableTree[Idx];
    for (unsigned i = 0, e = VL.size(); i != e; ++i) {
      DEBUG(dbgs() << "SLP: \tChecking bundle: " << *VL[i] << ".\n");
      if (E->Scalars[i] != VL[i]) {
        DEBUG(dbgs() << "SLP: Gathering due to partial overlap.\n");
        newTreeEntry(VL, false);
        return;
      }
    }
    DEBUG(dbgs() << "SLP: Perfect diamond merge at " << *VL[0] << ".\n");
    return;
  }

  // Check that none of the instructions in the bundle are already in the tree.
  for (unsigned i = 0, e = VL.size(); i != e; ++i) {
    if (ScalarToTreeEntry.count(VL[i])) {
      DEBUG(dbgs() << "SLP: The instruction (" << *VL[i] <<
            ") is already in tree.\n");
      newTreeEntry(VL, false);
      return;
    }
  }

  // If any of the scalars appears in the table OR it is marked as a value that
  // needs to stat scalar then we need to gather the scalars.
  for (unsigned i = 0, e = VL.size(); i != e; ++i) {
    if (ScalarToTreeEntry.count(VL[i]) || MustGather.count(VL[i])) {
      DEBUG(dbgs() << "SLP: Gathering due to gathered scalar. \n");
      newTreeEntry(VL, false);
      return;
    }
  }

  // Check that all of the users of the scalars that we want to vectorize are
  // schedulable.
  Instruction *VL0 = cast<Instruction>(VL[0]);
  int MyLastIndex = getLastIndex(VL);
  BasicBlock *BB = cast<Instruction>(VL0)->getParent();

  for (unsigned i = 0, e = VL.size(); i != e; ++i) {
    Instruction *Scalar = cast<Instruction>(VL[i]);
    DEBUG(dbgs() << "SLP: Checking users of  " << *Scalar << ". \n");
    for (Value::use_iterator U = Scalar->use_begin(), UE = Scalar->use_end();
         U != UE; ++U) {
      DEBUG(dbgs() << "SLP: \tUser " << **U << ". \n");
      Instruction *User = dyn_cast<Instruction>(*U);
      if (!User) {
        DEBUG(dbgs() << "SLP: Gathering due unknown user. \n");
        newTreeEntry(VL, false);
        return;
      }

      // We don't care if the user is in a different basic block.
      BasicBlock *UserBlock = User->getParent();
      if (UserBlock != BB) {
        DEBUG(dbgs() << "SLP: User from a different basic block "
              << *User << ". \n");
        continue;
      }

      // If this is a PHINode within this basic block then we can place the
      // extract wherever we want.
      if (isa<PHINode>(*User)) {
        DEBUG(dbgs() << "SLP: \tWe can schedule PHIs:" << *User << ". \n");
        continue;
      }

      // Check if this is a safe in-tree user.
      if (ScalarToTreeEntry.count(User)) {
        int Idx = ScalarToTreeEntry[User];
        int VecLocation = VectorizableTree[Idx].LastScalarIndex;
        if (VecLocation <= MyLastIndex) {
          DEBUG(dbgs() << "SLP: Gathering due to unschedulable vector. \n");
          newTreeEntry(VL, false);
          return;
        }
        DEBUG(dbgs() << "SLP: In-tree user (" << *User << ") at #" <<
              VecLocation << " vector value (" << *Scalar << ") at #"
              << MyLastIndex << ".\n");
        continue;
      }

      // Make sure that we can schedule this unknown user.
      BlockNumbering &BN = BlocksNumbers[BB];
      int UserIndex = BN.getIndex(User);
      if (UserIndex < MyLastIndex) {

        DEBUG(dbgs() << "SLP: Can't schedule extractelement for "
              << *User << ". \n");
        newTreeEntry(VL, false);
        return;
      }
    }
  }

  // Check that every instructions appears once in this bundle.
  for (unsigned i = 0, e = VL.size(); i < e; ++i)
    for (unsigned j = i+1; j < e; ++j)
      if (VL[i] == VL[j]) {
        DEBUG(dbgs() << "SLP: Scalar used twice in bundle.\n");
        newTreeEntry(VL, false);
        return;
      }

  // Check that instructions in this bundle don't reference other instructions.
  // The runtime of this check is O(N * N-1 * uses(N)) and a typical N is 4.
  for (unsigned i = 0, e = VL.size(); i < e; ++i) {
    for (Value::use_iterator U = VL[i]->use_begin(), UE = VL[i]->use_end();
         U != UE; ++U) {
      for (unsigned j = 0; j < e; ++j) {
        if (i != j && *U == VL[j]) {
          DEBUG(dbgs() << "SLP: Intra-bundle dependencies!" << **U << ". \n");
          newTreeEntry(VL, false);
          return;
        }
      }
    }
  }

  DEBUG(dbgs() << "SLP: We are able to schedule this bundle.\n");

  unsigned Opcode = getSameOpcode(VL);

  // Check if it is safe to sink the loads or the stores.
  if (Opcode == Instruction::Load || Opcode == Instruction::Store) {
    Instruction *Last = getLastInstruction(VL);

    for (unsigned i = 0, e = VL.size(); i < e; ++i) {
      if (VL[i] == Last)
        continue;
      Value *Barrier = getSinkBarrier(cast<Instruction>(VL[i]), Last);
      if (Barrier) {
        DEBUG(dbgs() << "SLP: Can't sink " << *VL[i] << "\n down to " << *Last
              << "\n because of " << *Barrier << ".  Gathering.\n");
        newTreeEntry(VL, false);
        return;
      }
    }
  }

  switch (Opcode) {
    case Instruction::PHI: {
      PHINode *PH = dyn_cast<PHINode>(VL0);
      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of PHINodes.\n");

      for (unsigned i = 0, e = PH->getNumIncomingValues(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (unsigned j = 0; j < VL.size(); ++j)
          Operands.push_back(cast<PHINode>(VL[j])->getIncomingValue(i));

        buildTree_rec(Operands, Depth + 1);
      }
      return;
    }
    case Instruction::ExtractElement: {
      bool Reuse = CanReuseExtract(VL);
      if (Reuse) {
        DEBUG(dbgs() << "SLP: Reusing extract sequence.\n");
      }
      newTreeEntry(VL, Reuse);
      return;
    }
    case Instruction::Load: {
      // Check if the loads are consecutive or of we need to swizzle them.
      for (unsigned i = 0, e = VL.size() - 1; i < e; ++i)
        if (!isConsecutiveAccess(VL[i], VL[i + 1])) {
          newTreeEntry(VL, false);
          DEBUG(dbgs() << "SLP: Need to swizzle loads.\n");
          return;
        }

      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of loads.\n");
      return;
    }
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::SIToFP:
    case Instruction::UIToFP:
    case Instruction::Trunc:
    case Instruction::FPTrunc:
    case Instruction::BitCast: {
      Type *SrcTy = VL0->getOperand(0)->getType();
      for (unsigned i = 0; i < VL.size(); ++i) {
        Type *Ty = cast<Instruction>(VL[i])->getOperand(0)->getType();
        if (Ty != SrcTy || Ty->isAggregateType() || Ty->isVectorTy()) {
          newTreeEntry(VL, false);
          DEBUG(dbgs() << "SLP: Gathering casts with different src types.\n");
          return;
        }
      }
      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of casts.\n");

      for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (unsigned j = 0; j < VL.size(); ++j)
          Operands.push_back(cast<Instruction>(VL[j])->getOperand(i));

        buildTree_rec(Operands, Depth+1);
      }
      return;
    }
    case Instruction::ICmp:
    case Instruction::FCmp: {
      // Check that all of the compares have the same predicate.
      CmpInst::Predicate P0 = dyn_cast<CmpInst>(VL0)->getPredicate();
      for (unsigned i = 1, e = VL.size(); i < e; ++i) {
        CmpInst *Cmp = cast<CmpInst>(VL[i]);
        if (Cmp->getPredicate() != P0) {
          newTreeEntry(VL, false);
          DEBUG(dbgs() << "SLP: Gathering cmp with different predicate.\n");
          return;
        }
      }

      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of compares.\n");

      for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (unsigned j = 0; j < VL.size(); ++j)
          Operands.push_back(cast<Instruction>(VL[j])->getOperand(i));

        buildTree_rec(Operands, Depth+1);
      }
      return;
    }
    case Instruction::Select:
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of bin op.\n");

      for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (unsigned j = 0; j < VL.size(); ++j)
          Operands.push_back(cast<Instruction>(VL[j])->getOperand(i));

        buildTree_rec(Operands, Depth+1);
      }
      return;
    }
    case Instruction::Store: {
      // Check if the stores are consecutive or of we need to swizzle them.
      for (unsigned i = 0, e = VL.size() - 1; i < e; ++i)
        if (!isConsecutiveAccess(VL[i], VL[i + 1])) {
          newTreeEntry(VL, false);
          DEBUG(dbgs() << "SLP: Non consecutive store.\n");
          return;
        }

      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of stores.\n");

      ValueList Operands;
      for (unsigned j = 0; j < VL.size(); ++j)
        Operands.push_back(cast<Instruction>(VL[j])->getOperand(0));

      // We can ignore these values because we are sinking them down.
      MemBarrierIgnoreList.insert(VL.begin(), VL.end());
      buildTree_rec(Operands, Depth + 1);
      return;
    }
    default:
      newTreeEntry(VL, false);
      DEBUG(dbgs() << "SLP: Gathering unknown instruction.\n");
      return;
  }
}

int BoUpSLP::getEntryCost(TreeEntry *E) {
  ArrayRef<Value*> VL = E->Scalars;

  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, VL.size());

  if (E->NeedToGather) {
    if (allConstant(VL))
      return 0;
    if (isSplat(VL)) {
      return TTI->getShuffleCost(TargetTransformInfo::SK_Broadcast, VecTy, 0);
    }
    return getGatherCost(E->Scalars);
  }

  assert(getSameOpcode(VL) && getSameType(VL) && getSameBlock(VL) &&
         "Invalid VL");
  Instruction *VL0 = cast<Instruction>(VL[0]);
  unsigned Opcode = VL0->getOpcode();
  switch (Opcode) {
    case Instruction::PHI: {
      return 0;
    }
    case Instruction::ExtractElement: {
      if (CanReuseExtract(VL))
        return 0;
      return getGatherCost(VecTy);
    }
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::SIToFP:
    case Instruction::UIToFP:
    case Instruction::Trunc:
    case Instruction::FPTrunc:
    case Instruction::BitCast: {
      Type *SrcTy = VL0->getOperand(0)->getType();

      // Calculate the cost of this instruction.
      int ScalarCost = VL.size() * TTI->getCastInstrCost(VL0->getOpcode(),
                                                         VL0->getType(), SrcTy);

      VectorType *SrcVecTy = VectorType::get(SrcTy, VL.size());
      int VecCost = TTI->getCastInstrCost(VL0->getOpcode(), VecTy, SrcVecTy);
      return VecCost - ScalarCost;
    }
    case Instruction::FCmp:
    case Instruction::ICmp:
    case Instruction::Select:
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      // Calculate the cost of this instruction.
      int ScalarCost = 0;
      int VecCost = 0;
      if (Opcode == Instruction::FCmp || Opcode == Instruction::ICmp ||
          Opcode == Instruction::Select) {
        VectorType *MaskTy = VectorType::get(Builder.getInt1Ty(), VL.size());
        ScalarCost = VecTy->getNumElements() *
        TTI->getCmpSelInstrCost(Opcode, ScalarTy, Builder.getInt1Ty());
        VecCost = TTI->getCmpSelInstrCost(Opcode, VecTy, MaskTy);
      } else {
        ScalarCost = VecTy->getNumElements() *
        TTI->getArithmeticInstrCost(Opcode, ScalarTy);
        VecCost = TTI->getArithmeticInstrCost(Opcode, VecTy);
      }
      return VecCost - ScalarCost;
    }
    case Instruction::Load: {
      // Cost of wide load - cost of scalar loads.
      int ScalarLdCost = VecTy->getNumElements() *
      TTI->getMemoryOpCost(Instruction::Load, ScalarTy, 1, 0);
      int VecLdCost = TTI->getMemoryOpCost(Instruction::Load, ScalarTy, 1, 0);
      return VecLdCost - ScalarLdCost;
    }
    case Instruction::Store: {
      // We know that we can merge the stores. Calculate the cost.
      int ScalarStCost = VecTy->getNumElements() *
      TTI->getMemoryOpCost(Instruction::Store, ScalarTy, 1, 0);
      int VecStCost = TTI->getMemoryOpCost(Instruction::Store, ScalarTy, 1, 0);
      return VecStCost - ScalarStCost;
    }
    default:
      llvm_unreachable("Unknown instruction");
  }
}

int BoUpSLP::getTreeCost() {
  int Cost = 0;
  DEBUG(dbgs() << "SLP: Calculating cost for tree of size " <<
        VectorizableTree.size() << ".\n");

  for (unsigned i = 0, e = VectorizableTree.size(); i != e; ++i) {
    int C = getEntryCost(&VectorizableTree[i]);
    DEBUG(dbgs() << "SLP: Adding cost " << C << " for bundle that starts with "
          << *VectorizableTree[i].Scalars[0] << " .\n");
    Cost += C;
  }
  DEBUG(dbgs() << "SLP: Total Cost " << Cost << ".\n");
  return  Cost;
}

int BoUpSLP::getGatherCost(Type *Ty) {
  int Cost = 0;
  for (unsigned i = 0, e = cast<VectorType>(Ty)->getNumElements(); i < e; ++i)
    Cost += TTI->getVectorInstrCost(Instruction::InsertElement, Ty, i);
  return Cost;
}

int BoUpSLP::getGatherCost(ArrayRef<Value *> VL) {
  // Find the type of the operands in VL.
  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, VL.size());
  // Find the cost of inserting/extracting values from the vector.
  return getGatherCost(VecTy);
}

AliasAnalysis::Location BoUpSLP::getLocation(Instruction *I) {
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return AA->getLocation(SI);
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return AA->getLocation(LI);
  return AliasAnalysis::Location();
}

Value *BoUpSLP::getPointerOperand(Value *I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return LI->getPointerOperand();
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->getPointerOperand();
  return 0;
}

unsigned BoUpSLP::getAddressSpaceOperand(Value *I) {
  if (LoadInst *L = dyn_cast<LoadInst>(I))
    return L->getPointerAddressSpace();
  if (StoreInst *S = dyn_cast<StoreInst>(I))
    return S->getPointerAddressSpace();
  return -1;
}

bool BoUpSLP::isConsecutiveAccess(Value *A, Value *B) {
  Value *PtrA = getPointerOperand(A);
  Value *PtrB = getPointerOperand(B);
  unsigned ASA = getAddressSpaceOperand(A);
  unsigned ASB = getAddressSpaceOperand(B);

  // Check that the address spaces match and that the pointers are valid.
  if (!PtrA || !PtrB || (ASA != ASB))
    return false;

  // Check that A and B are of the same type.
  if (PtrA->getType() != PtrB->getType())
    return false;

  // Calculate the distance.
  const SCEV *PtrSCEVA = SE->getSCEV(PtrA);
  const SCEV *PtrSCEVB = SE->getSCEV(PtrB);
  const SCEV *OffsetSCEV = SE->getMinusSCEV(PtrSCEVA, PtrSCEVB);
  const SCEVConstant *ConstOffSCEV = dyn_cast<SCEVConstant>(OffsetSCEV);

  // Non constant distance.
  if (!ConstOffSCEV)
    return false;

  int64_t Offset = ConstOffSCEV->getValue()->getSExtValue();
  Type *Ty = cast<PointerType>(PtrA->getType())->getElementType();
  // The Instructions are connsecutive if the size of the first load/store is
  // the same as the offset.
  int64_t Sz = DL->getTypeStoreSize(Ty);
  return ((-Offset) == Sz);
}

Value *BoUpSLP::getSinkBarrier(Instruction *Src, Instruction *Dst) {
  assert(Src->getParent() == Dst->getParent() && "Not the same BB");
  BasicBlock::iterator I = Src, E = Dst;
  /// Scan all of the instruction from SRC to DST and check if
  /// the source may alias.
  for (++I; I != E; ++I) {
    // Ignore store instructions that are marked as 'ignore'.
    if (MemBarrierIgnoreList.count(I))
      continue;
    if (Src->mayWriteToMemory()) /* Write */ {
      if (!I->mayReadOrWriteMemory())
        continue;
    } else /* Read */ {
      if (!I->mayWriteToMemory())
        continue;
    }
    AliasAnalysis::Location A = getLocation(&*I);
    AliasAnalysis::Location B = getLocation(Src);

    if (!A.Ptr || !B.Ptr || AA->alias(A, B))
      return I;
  }
  return 0;
}

int BoUpSLP::getLastIndex(ArrayRef<Value *> VL) {
  BasicBlock *BB = cast<Instruction>(VL[0])->getParent();
  assert(BB == getSameBlock(VL) && BlocksNumbers.count(BB) && "Invalid block");
  BlockNumbering &BN = BlocksNumbers[BB];

  int MaxIdx = BN.getIndex(BB->getFirstNonPHI());
  for (unsigned i = 0, e = VL.size(); i < e; ++i)
    MaxIdx = std::max(MaxIdx, BN.getIndex(cast<Instruction>(VL[i])));
  return MaxIdx;
}

Instruction *BoUpSLP::getLastInstruction(ArrayRef<Value *> VL) {
  BasicBlock *BB = cast<Instruction>(VL[0])->getParent();
  assert(BB == getSameBlock(VL) && BlocksNumbers.count(BB) && "Invalid block");
  BlockNumbering &BN = BlocksNumbers[BB];

  int MaxIdx = BN.getIndex(cast<Instruction>(VL[0]));
  for (unsigned i = 1, e = VL.size(); i < e; ++i)
    MaxIdx = std::max(MaxIdx, BN.getIndex(cast<Instruction>(VL[i])));
  Instruction *I = BN.getInstruction(MaxIdx);
  assert(I && "bad location");
  return I;
}

Instruction *BoUpSLP::getInstructionForIndex(unsigned Index, BasicBlock *BB) {
  BlockNumbering &BN = BlocksNumbers[BB];
  return BN.getInstruction(Index);
}

int BoUpSLP::getFirstUserIndex(ArrayRef<Value *> VL) {
  BasicBlock *BB = getSameBlock(VL);
  assert(BB && "All instructions must come from the same block");
  BlockNumbering &BN = BlocksNumbers[BB];

  // Find the first user of the values.
  int FirstUser = BN.getIndex(BB->getTerminator());
  for (unsigned i = 0, e = VL.size(); i < e; ++i) {
    for (Value::use_iterator U = VL[i]->use_begin(), UE = VL[i]->use_end();
         U != UE; ++U) {
      Instruction *Instr = dyn_cast<Instruction>(*U);

      if (!Instr || Instr->getParent() != BB)
        continue;

      FirstUser = std::min(FirstUser, BN.getIndex(Instr));
    }
  }
  return FirstUser;
}

Value *BoUpSLP::Gather(ArrayRef<Value *> VL, VectorType *Ty) {
  Value *Vec = UndefValue::get(Ty);
  // Generate the 'InsertElement' instruction.
  for (unsigned i = 0; i < Ty->getNumElements(); ++i) {
    Vec = Builder.CreateInsertElement(Vec, VL[i], Builder.getInt32(i));
    if (Instruction *I = dyn_cast<Instruction>(Vec))
      GatherSeq.insert(I);
  }

  return Vec;
}

Value *BoUpSLP::vectorizeTree(ArrayRef<Value *> VL) {
  if (ScalarToTreeEntry.count(VL[0])) {
    int Idx = ScalarToTreeEntry[VL[0]];
    TreeEntry *E = &VectorizableTree[Idx];
    if (E->isSame(VL))
      return vectorizeTree(E);
  }

  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, VL.size());

  return Gather(VL, VecTy);
}

Value *BoUpSLP::vectorizeTree(TreeEntry *E) {
  BuilderLocGuard Guard(Builder);

  if (E->VectorizedValue) {
    DEBUG(dbgs() << "SLP: Diamond merged for " << *E->Scalars[0] << ".\n");
    return E->VectorizedValue;
  }

  Type *ScalarTy = E->Scalars[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(E->Scalars[0]))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, E->Scalars.size());

  if (E->NeedToGather) {
    return Gather(E->Scalars, VecTy);
  }

  Instruction *VL0 = cast<Instruction>(E->Scalars[0]);
  unsigned Opcode = VL0->getOpcode();
  assert(Opcode == getSameOpcode(E->Scalars) && "Invalid opcode");

  switch (Opcode) {
    case Instruction::PHI: {
      PHINode *PH = dyn_cast<PHINode>(VL0);
      Builder.SetInsertPoint(PH->getParent()->getFirstInsertionPt());
      PHINode *NewPhi = Builder.CreatePHI(VecTy, PH->getNumIncomingValues());
      E->VectorizedValue = NewPhi;

      for (unsigned i = 0, e = PH->getNumIncomingValues(); i < e; ++i) {
        ValueList Operands;
        BasicBlock *IBB = PH->getIncomingBlock(i);

        // Prepare the operand vector.
        for (unsigned j = 0; j < E->Scalars.size(); ++j)
          Operands.push_back(cast<PHINode>(E->Scalars[j])->
                             getIncomingValueForBlock(IBB));

        Builder.SetInsertPoint(IBB->getTerminator());
        Value *Vec = vectorizeTree(Operands);
        NewPhi->addIncoming(Vec, IBB);
      }

      assert(NewPhi->getNumIncomingValues() == PH->getNumIncomingValues() &&
             "Invalid number of incoming values");
      return NewPhi;
    }

    case Instruction::ExtractElement: {
      if (CanReuseExtract(E->Scalars)) {
        Value *V = VL0->getOperand(0);
        E->VectorizedValue = V;
        return V;
      }
      return Gather(E->Scalars, VecTy);
    }
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::SIToFP:
    case Instruction::UIToFP:
    case Instruction::Trunc:
    case Instruction::FPTrunc:
    case Instruction::BitCast: {
      ValueList INVL;
      for (int i = 0, e = E->Scalars.size(); i < e; ++i)
        INVL.push_back(cast<Instruction>(E->Scalars[i])->getOperand(0));

      Builder.SetInsertPoint(getLastInstruction(E->Scalars));
      Value *InVec = vectorizeTree(INVL);
      CastInst *CI = dyn_cast<CastInst>(VL0);
      Value *V = Builder.CreateCast(CI->getOpcode(), InVec, VecTy);
      E->VectorizedValue = V;
      return V;
    }
    case Instruction::FCmp:
    case Instruction::ICmp: {
      ValueList LHSV, RHSV;
      for (int i = 0, e = E->Scalars.size(); i < e; ++i) {
        LHSV.push_back(cast<Instruction>(E->Scalars[i])->getOperand(0));
        RHSV.push_back(cast<Instruction>(E->Scalars[i])->getOperand(1));
      }

      Builder.SetInsertPoint(getLastInstruction(E->Scalars));
      Value *L = vectorizeTree(LHSV);
      Value *R = vectorizeTree(RHSV);
      Value *V;

      CmpInst::Predicate P0 = dyn_cast<CmpInst>(VL0)->getPredicate();
      if (Opcode == Instruction::FCmp)
        V = Builder.CreateFCmp(P0, L, R);
      else
        V = Builder.CreateICmp(P0, L, R);

      E->VectorizedValue = V;
      return V;
    }
    case Instruction::Select: {
      ValueList TrueVec, FalseVec, CondVec;
      for (int i = 0, e = E->Scalars.size(); i < e; ++i) {
        CondVec.push_back(cast<Instruction>(E->Scalars[i])->getOperand(0));
        TrueVec.push_back(cast<Instruction>(E->Scalars[i])->getOperand(1));
        FalseVec.push_back(cast<Instruction>(E->Scalars[i])->getOperand(2));
      }

      Builder.SetInsertPoint(getLastInstruction(E->Scalars));
      Value *Cond = vectorizeTree(CondVec);
      Value *True = vectorizeTree(TrueVec);
      Value *False = vectorizeTree(FalseVec);
      Value *V = Builder.CreateSelect(Cond, True, False);
      E->VectorizedValue = V;
      return V;
    }
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      ValueList LHSVL, RHSVL;
      for (int i = 0, e = E->Scalars.size(); i < e; ++i) {
        LHSVL.push_back(cast<Instruction>(E->Scalars[i])->getOperand(0));
        RHSVL.push_back(cast<Instruction>(E->Scalars[i])->getOperand(1));
      }

      Builder.SetInsertPoint(getLastInstruction(E->Scalars));
      Value *LHS = vectorizeTree(LHSVL);
      Value *RHS = vectorizeTree(RHSVL);

      if (LHS == RHS && isa<Instruction>(LHS)) {
        assert((VL0->getOperand(0) == VL0->getOperand(1)) && "Invalid order");
      }

      BinaryOperator *BinOp = cast<BinaryOperator>(VL0);
      Value *V = Builder.CreateBinOp(BinOp->getOpcode(), LHS, RHS);
      E->VectorizedValue = V;
      return V;
    }
    case Instruction::Load: {
      // Loads are inserted at the head of the tree because we don't want to
      // sink them all the way down past store instructions.
      Builder.SetInsertPoint(getLastInstruction(E->Scalars));
      LoadInst *LI = cast<LoadInst>(VL0);
      Value *VecPtr =
      Builder.CreateBitCast(LI->getPointerOperand(), VecTy->getPointerTo());
      unsigned Alignment = LI->getAlignment();
      LI = Builder.CreateLoad(VecPtr);
      LI->setAlignment(Alignment);
      E->VectorizedValue = LI;
      return LI;
    }
    case Instruction::Store: {
      StoreInst *SI = cast<StoreInst>(VL0);
      unsigned Alignment = SI->getAlignment();

      ValueList ValueOp;
      for (int i = 0, e = E->Scalars.size(); i < e; ++i)
        ValueOp.push_back(cast<StoreInst>(E->Scalars[i])->getValueOperand());

      Builder.SetInsertPoint(getLastInstruction(E->Scalars));
      Value *VecValue = vectorizeTree(ValueOp);
      Value *VecPtr =
      Builder.CreateBitCast(SI->getPointerOperand(), VecTy->getPointerTo());
      StoreInst *S = Builder.CreateStore(VecValue, VecPtr);
      S->setAlignment(Alignment);
      E->VectorizedValue = S;
      return S;
    }
    default:
    llvm_unreachable("unknown inst");
  }
  return 0;
}

void BoUpSLP::vectorizeTree() {
  Builder.SetInsertPoint(F->getEntryBlock().begin());
  vectorizeTree(&VectorizableTree[0]);

  // For each vectorized value:
  for (int EIdx = 0, EE = VectorizableTree.size(); EIdx < EE; ++EIdx) {
    TreeEntry *Entry = &VectorizableTree[EIdx];

    // For each lane:
    for (int Lane = 0, LE = Entry->Scalars.size(); Lane != LE; ++Lane) {
      Value *Scalar = Entry->Scalars[Lane];

      // No need to handle users of gathered values.
      if (Entry->NeedToGather)
        continue;

      Value *Vec = Entry->VectorizedValue;
      assert(Vec && "Can't find vectorizable value");

      SmallVector<User*, 16> Users(Scalar->use_begin(), Scalar->use_end());

      for (SmallVector<User*, 16>::iterator User = Users.begin(),
           UE = Users.end(); User != UE; ++User) {
        DEBUG(dbgs() << "SLP: \tupdating user  " << **User << ".\n");

        bool Gathered = MustGather.count(*User);

        // Skip in-tree scalars that become vectors.
        if (ScalarToTreeEntry.count(*User) && !Gathered) {
          DEBUG(dbgs() << "SLP: \tUser will be removed soon:" <<
                **User << ".\n");
          int Idx = ScalarToTreeEntry[*User]; (void) Idx;
          assert(!VectorizableTree[Idx].NeedToGather && "bad state ?");
          continue;
        }

        if (!isa<Instruction>(*User))
          continue;

        // Generate extracts for out-of-tree users.
        // Find the insertion point for the extractelement lane.
        Instruction *Loc = 0;
        if (PHINode *PN = dyn_cast<PHINode>(Vec)) {
          Loc = PN->getParent()->getFirstInsertionPt();
        } else if (Instruction *Iv = dyn_cast<Instruction>(Vec)){
          Loc = ++((BasicBlock::iterator)*Iv);
        } else {
          Loc = F->getEntryBlock().begin();
        }

        Builder.SetInsertPoint(Loc);
        Value *Ex = Builder.CreateExtractElement(Vec, Builder.getInt32(Lane));
        (*User)->replaceUsesOfWith(Scalar, Ex);
        DEBUG(dbgs() << "SLP: \tupdated user:" << **User << ".\n");
      }

      Type *Ty = Scalar->getType();
      if (!Ty->isVoidTy()) {
        for (Value::use_iterator User = Scalar->use_begin(), UE = Scalar->use_end();
             User != UE; ++User) {
          DEBUG(dbgs() << "SLP: \tvalidating user:" << **User << ".\n");
          assert(!MustGather.count(*User) &&
                 "Replacing gathered value with undef");
          assert(ScalarToTreeEntry.count(*User) &&
                 "Replacing out-of-tree value with undef");
        }
        Value *Undef = UndefValue::get(Ty);
        Scalar->replaceAllUsesWith(Undef);
      }
      DEBUG(dbgs() << "SLP: \tErasing scalar:" << *Scalar << ".\n");
      cast<Instruction>(Scalar)->eraseFromParent();
    }
  }

  for (Function::iterator it = F->begin(), e = F->end(); it != e; ++it) {
    BlocksNumbers[it].forget();
  }
  Builder.ClearInsertionPoint();
}

void BoUpSLP::optimizeGatherSequence() {
  DEBUG(dbgs() << "SLP: Optimizing " << GatherSeq.size()
        << " gather sequences instructions.\n");
  // LICM InsertElementInst sequences.
  for (SetVector<Instruction *>::iterator it = GatherSeq.begin(),
       e = GatherSeq.end(); it != e; ++it) {
    InsertElementInst *Insert = dyn_cast<InsertElementInst>(*it);

    if (!Insert)
      continue;

    // Check if this block is inside a loop.
    Loop *L = LI->getLoopFor(Insert->getParent());
    if (!L)
      continue;

    // Check if it has a preheader.
    BasicBlock *PreHeader = L->getLoopPreheader();
    if (!PreHeader)
      continue;

    // If the vector or the element that we insert into it are
    // instructions that are defined in this basic block then we can't
    // hoist this instruction.
    Instruction *CurrVec = dyn_cast<Instruction>(Insert->getOperand(0));
    Instruction *NewElem = dyn_cast<Instruction>(Insert->getOperand(1));
    if (CurrVec && L->contains(CurrVec))
      continue;
    if (NewElem && L->contains(NewElem))
      continue;

    // We can hoist this instruction. Move it to the pre-header.
    Insert->moveBefore(PreHeader->getTerminator());
  }

  // Perform O(N^2) search over the gather sequences and merge identical
  // instructions. TODO: We can further optimize this scan if we split the
  // instructions into different buckets based on the insert lane.
  SmallPtrSet<Instruction*, 16> Visited;
  SmallVector<Instruction*, 16> ToRemove;
  ReversePostOrderTraversal<Function*> RPOT(F);
  for (ReversePostOrderTraversal<Function*>::rpo_iterator I = RPOT.begin(),
       E = RPOT.end(); I != E; ++I) {
    BasicBlock *BB = *I;
    // For all instructions in the function:
    for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
      InsertElementInst *Insert = dyn_cast<InsertElementInst>(it);
      if (!Insert || !GatherSeq.count(Insert))
        continue;

      // Check if we can replace this instruction with any of the
      // visited instructions.
      for (SmallPtrSet<Instruction*, 16>::iterator v = Visited.begin(),
           ve = Visited.end(); v != ve; ++v) {
        if (Insert->isIdenticalTo(*v) &&
            DT->dominates((*v)->getParent(), Insert->getParent())) {
          Insert->replaceAllUsesWith(*v);
          ToRemove.push_back(Insert);
          Insert = 0;
          break;
        }
      }
      if (Insert)
        Visited.insert(Insert);
    }
  }

  // Erase all of the instructions that we RAUWed.
  for (SmallVectorImpl<Instruction *>::iterator v = ToRemove.begin(),
       ve = ToRemove.end(); v != ve; ++v) {
    assert((*v)->getNumUses() == 0 && "Can't remove instructions with uses");
    (*v)->eraseFromParent();
  }
}

/// The SLPVectorizer Pass.
struct SLPVectorizer : public FunctionPass {
  typedef SmallVector<StoreInst *, 8> StoreList;
  typedef MapVector<Value *, StoreList> StoreListMap;

  /// Pass identification, replacement for typeid
  static char ID;

  explicit SLPVectorizer() : FunctionPass(ID) {
    initializeSLPVectorizerPass(*PassRegistry::getPassRegistry());
  }

  ScalarEvolution *SE;
  DataLayout *DL;
  TargetTransformInfo *TTI;
  AliasAnalysis *AA;
  LoopInfo *LI;
  DominatorTree *DT;

  virtual bool runOnFunction(Function &F) {
    SE = &getAnalysis<ScalarEvolution>();
    DL = getAnalysisIfAvailable<DataLayout>();
    TTI = &getAnalysis<TargetTransformInfo>();
    AA = &getAnalysis<AliasAnalysis>();
    LI = &getAnalysis<LoopInfo>();
    DT = &getAnalysis<DominatorTree>();

    StoreRefs.clear();
    bool Changed = false;

    // Must have DataLayout. We can't require it because some tests run w/o
    // triple.
    if (!DL)
      return false;

    DEBUG(dbgs() << "SLP: Analyzing blocks in " << F.getName() << ".\n");

    // Use the bollom up slp vectorizer to construct chains that start with
    // he store instructions.
    BoUpSLP R(&F, SE, DL, TTI, AA, LI, DT);

    // Scan the blocks in the function in post order.
    for (po_iterator<BasicBlock*> it = po_begin(&F.getEntryBlock()),
         e = po_end(&F.getEntryBlock()); it != e; ++it) {
      BasicBlock *BB = *it;

      // Vectorize trees that end at reductions.
      Changed |= vectorizeChainsInBlock(BB, R);

      // Vectorize trees that end at stores.
      if (unsigned count = collectStores(BB, R)) {
        (void)count;
        DEBUG(dbgs() << "SLP: Found " << count << " stores to vectorize.\n");
        Changed |= vectorizeStoreChains(R);
      }
    }

    if (Changed) {
      R.optimizeGatherSequence();
      DEBUG(dbgs() << "SLP: vectorized \"" << F.getName() << "\"\n");
      DEBUG(verifyFunction(F));
    }
    return Changed;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    FunctionPass::getAnalysisUsage(AU);
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<AliasAnalysis>();
    AU.addRequired<TargetTransformInfo>();
    AU.addRequired<LoopInfo>();
    AU.addRequired<DominatorTree>();
    AU.addPreserved<LoopInfo>();
    AU.addPreserved<DominatorTree>();
    AU.setPreservesCFG();
  }

private:

  /// \brief Collect memory references and sort them according to their base
  /// object. We sort the stores to their base objects to reduce the cost of the
  /// quadratic search on the stores. TODO: We can further reduce this cost
  /// if we flush the chain creation every time we run into a memory barrier.
  unsigned collectStores(BasicBlock *BB, BoUpSLP &R);

  /// \brief Try to vectorize a chain that starts at two arithmetic instrs.
  bool tryToVectorizePair(Value *A, Value *B, BoUpSLP &R);

  /// \brief Try to vectorize a list of operands. If \p NeedExtracts is true
  /// then we calculate the cost of extracting the scalars from the vector.
  /// \returns true if a value was vectorized.
  bool tryToVectorizeList(ArrayRef<Value *> VL, BoUpSLP &R, bool NeedExtracts);

  /// \brief Try to vectorize a chain that may start at the operands of \V;
  bool tryToVectorize(BinaryOperator *V, BoUpSLP &R);

  /// \brief Vectorize the stores that were collected in StoreRefs.
  bool vectorizeStoreChains(BoUpSLP &R);

  /// \brief Scan the basic block and look for patterns that are likely to start
  /// a vectorization chain.
  bool vectorizeChainsInBlock(BasicBlock *BB, BoUpSLP &R);

  bool vectorizeStoreChain(ArrayRef<Value *> Chain, int CostThreshold,
                           BoUpSLP &R);

  bool vectorizeStores(ArrayRef<StoreInst *> Stores, int costThreshold,
                       BoUpSLP &R);
private:
  StoreListMap StoreRefs;
};

bool SLPVectorizer::vectorizeStoreChain(ArrayRef<Value *> Chain,
                                          int CostThreshold, BoUpSLP &R) {
  unsigned ChainLen = Chain.size();
  DEBUG(dbgs() << "SLP: Analyzing a store chain of length " << ChainLen
        << "\n");
  Type *StoreTy = cast<StoreInst>(Chain[0])->getValueOperand()->getType();
  unsigned Sz = DL->getTypeSizeInBits(StoreTy);
  unsigned VF = MinVecRegSize / Sz;

  if (!isPowerOf2_32(Sz) || VF < 2)
    return false;

  bool Changed = false;
  // Look for profitable vectorizable trees at all offsets, starting at zero.
  for (unsigned i = 0, e = ChainLen; i < e; ++i) {
    if (i + VF > e)
      break;
    DEBUG(dbgs() << "SLP: Analyzing " << VF << " stores at offset " << i
          << "\n");
    ArrayRef<Value *> Operands = Chain.slice(i, VF);

    R.buildTree(Operands);

    int Cost = R.getTreeCost();

    DEBUG(dbgs() << "SLP: Found cost=" << Cost << " for VF=" << VF << "\n");
    if (Cost < CostThreshold) {
      DEBUG(dbgs() << "SLP: Decided to vectorize cost=" << Cost << "\n");
      R.vectorizeTree();

      // Move to the next bundle.
      i += VF - 1;
      Changed = true;
    }
  }

  if (Changed || ChainLen > VF)
    return Changed;

  // Handle short chains. This helps us catch types such as <3 x float> that
  // are smaller than vector size.
  R.buildTree(Chain);

  int Cost = R.getTreeCost();

  if (Cost < CostThreshold) {
    DEBUG(dbgs() << "SLP: Found store chain cost = " << Cost
          << " for size = " << ChainLen << "\n");
    R.vectorizeTree();
    return true;
  }

  return false;
}

bool SLPVectorizer::vectorizeStores(ArrayRef<StoreInst *> Stores,
                                      int costThreshold, BoUpSLP &R) {
  SetVector<Value *> Heads, Tails;
  SmallDenseMap<Value *, Value *> ConsecutiveChain;

  // We may run into multiple chains that merge into a single chain. We mark the
  // stores that we vectorized so that we don't visit the same store twice.
  BoUpSLP::ValueSet VectorizedStores;
  bool Changed = false;

  // Do a quadratic search on all of the given stores and find
  // all of the pairs of loads that follow each other.
  for (unsigned i = 0, e = Stores.size(); i < e; ++i)
    for (unsigned j = 0; j < e; ++j) {
      if (i == j)
        continue;

      if (R.isConsecutiveAccess(Stores[i], Stores[j])) {
        Tails.insert(Stores[j]);
        Heads.insert(Stores[i]);
        ConsecutiveChain[Stores[i]] = Stores[j];
      }
    }

  // For stores that start but don't end a link in the chain:
  for (SetVector<Value *>::iterator it = Heads.begin(), e = Heads.end();
       it != e; ++it) {
    if (Tails.count(*it))
      continue;

    // We found a store instr that starts a chain. Now follow the chain and try
    // to vectorize it.
    BoUpSLP::ValueList Operands;
    Value *I = *it;
    // Collect the chain into a list.
    while (Tails.count(I) || Heads.count(I)) {
      if (VectorizedStores.count(I))
        break;
      Operands.push_back(I);
      // Move to the next value in the chain.
      I = ConsecutiveChain[I];
    }

    bool Vectorized = vectorizeStoreChain(Operands, costThreshold, R);

    // Mark the vectorized stores so that we don't vectorize them again.
    if (Vectorized)
      VectorizedStores.insert(Operands.begin(), Operands.end());
    Changed |= Vectorized;
  }

  return Changed;
}


unsigned SLPVectorizer::collectStores(BasicBlock *BB, BoUpSLP &R) {
  unsigned count = 0;
  StoreRefs.clear();
  for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
    StoreInst *SI = dyn_cast<StoreInst>(it);
    if (!SI)
      continue;

    // Check that the pointer points to scalars.
    Type *Ty = SI->getValueOperand()->getType();
    if (Ty->isAggregateType() || Ty->isVectorTy())
      return 0;

    // Find the base of the GEP.
    Value *Ptr = SI->getPointerOperand();
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr))
      Ptr = GEP->getPointerOperand();

    // Save the store locations.
    StoreRefs[Ptr].push_back(SI);
    count++;
  }
  return count;
}

bool SLPVectorizer::tryToVectorizePair(Value *A, Value *B, BoUpSLP &R) {
  if (!A || !B)
    return false;
  Value *VL[] = { A, B };
  return tryToVectorizeList(VL, R, true);
}

bool SLPVectorizer::tryToVectorizeList(ArrayRef<Value *> VL, BoUpSLP &R,
                                       bool NeedExtracts) {
  if (VL.size() < 2)
    return false;

  DEBUG(dbgs() << "SLP: Vectorizing a list of length = " << VL.size() << ".\n");

  // Check that all of the parts are scalar instructions of the same type.
  Instruction *I0 = dyn_cast<Instruction>(VL[0]);
  if (!I0)
    return 0;

  unsigned Opcode0 = I0->getOpcode();

  for (int i = 0, e = VL.size(); i < e; ++i) {
    Type *Ty = VL[i]->getType();
    if (Ty->isAggregateType() || Ty->isVectorTy())
      return 0;
    Instruction *Inst = dyn_cast<Instruction>(VL[i]);
    if (!Inst || Inst->getOpcode() != Opcode0)
      return 0;
  }

  R.buildTree(VL);
  int Cost = R.getTreeCost();

  int ExtrCost = NeedExtracts ? R.getGatherCost(VL) : 0;
  DEBUG(dbgs() << "SLP: Cost of pair:" << Cost
               << " Cost of extract:" << ExtrCost << ".\n");
  if ((Cost + ExtrCost) >= -SLPCostThreshold)
    return false;
  DEBUG(dbgs() << "SLP: Vectorizing pair.\n");
  R.vectorizeTree();
  return true;
}

bool SLPVectorizer::tryToVectorize(BinaryOperator *V, BoUpSLP &R) {
  if (!V)
    return false;

  // Try to vectorize V.
  if (tryToVectorizePair(V->getOperand(0), V->getOperand(1), R))
    return true;

  BinaryOperator *A = dyn_cast<BinaryOperator>(V->getOperand(0));
  BinaryOperator *B = dyn_cast<BinaryOperator>(V->getOperand(1));
  // Try to skip B.
  if (B && B->hasOneUse()) {
    BinaryOperator *B0 = dyn_cast<BinaryOperator>(B->getOperand(0));
    BinaryOperator *B1 = dyn_cast<BinaryOperator>(B->getOperand(1));
    if (tryToVectorizePair(A, B0, R)) {
      B->moveBefore(V);
      return true;
    }
    if (tryToVectorizePair(A, B1, R)) {
      B->moveBefore(V);
      return true;
    }
  }

  // Try to skip A.
  if (A && A->hasOneUse()) {
    BinaryOperator *A0 = dyn_cast<BinaryOperator>(A->getOperand(0));
    BinaryOperator *A1 = dyn_cast<BinaryOperator>(A->getOperand(1));
    if (tryToVectorizePair(A0, B, R)) {
      A->moveBefore(V);
      return true;
    }
    if (tryToVectorizePair(A1, B, R)) {
      A->moveBefore(V);
      return true;
    }
  }
  return 0;
}

bool SLPVectorizer::vectorizeChainsInBlock(BasicBlock *BB, BoUpSLP &R) {
  bool Changed = false;
  for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
    if (isa<DbgInfoIntrinsic>(it))
      continue;

    // Try to vectorize reductions that use PHINodes.
    if (PHINode *P = dyn_cast<PHINode>(it)) {
      // Check that the PHI is a reduction PHI.
      if (P->getNumIncomingValues() != 2)
        return Changed;
      Value *Rdx =
          (P->getIncomingBlock(0) == BB
               ? (P->getIncomingValue(0))
               : (P->getIncomingBlock(1) == BB ? P->getIncomingValue(1) : 0));
      // Check if this is a Binary Operator.
      BinaryOperator *BI = dyn_cast_or_null<BinaryOperator>(Rdx);
      if (!BI)
        continue;

      Value *Inst = BI->getOperand(0);
      if (Inst == P)
        Inst = BI->getOperand(1);

      Changed |= tryToVectorize(dyn_cast<BinaryOperator>(Inst), R);
      continue;
    }

    // Try to vectorize trees that start at compare instructions.
    if (CmpInst *CI = dyn_cast<CmpInst>(it)) {
      if (tryToVectorizePair(CI->getOperand(0), CI->getOperand(1), R)) {
        Changed |= true;
        continue;
      }
      for (int i = 0; i < 2; ++i)
        if (BinaryOperator *BI = dyn_cast<BinaryOperator>(CI->getOperand(i)))
          Changed |=
              tryToVectorizePair(BI->getOperand(0), BI->getOperand(1), R);
      continue;
    }
  }

  // Scan the PHINodes in our successors in search for pairing hints.
  for (succ_iterator it = succ_begin(BB), e = succ_end(BB); it != e; ++it) {
    BasicBlock *Succ = *it;
    SmallVector<Value *, 4> Incoming;

    // Collect the incoming values from the PHIs.
    for (BasicBlock::iterator instr = Succ->begin(), ie = Succ->end();
         instr != ie; ++instr) {
      PHINode *P = dyn_cast<PHINode>(instr);

      if (!P)
        break;

      Value *V = P->getIncomingValueForBlock(BB);
      if (Instruction *I = dyn_cast<Instruction>(V))
        if (I->getParent() == BB)
          Incoming.push_back(I);
    }

    if (Incoming.size() > 1)
      Changed |= tryToVectorizeList(Incoming, R, true);
  }

  return Changed;
}

bool SLPVectorizer::vectorizeStoreChains(BoUpSLP &R) {
  bool Changed = false;
  // Attempt to sort and vectorize each of the store-groups.
  for (StoreListMap::iterator it = StoreRefs.begin(), e = StoreRefs.end();
       it != e; ++it) {
    if (it->second.size() < 2)
      continue;

    DEBUG(dbgs() << "SLP: Analyzing a store chain of length "
                 << it->second.size() << ".\n");

    Changed |= vectorizeStores(it->second, -SLPCostThreshold, R);
  }
  return Changed;
}

} // end anonymous namespace

char SLPVectorizer::ID = 0;
static const char lv_name[] = "SLP Vectorizer";
INITIALIZE_PASS_BEGIN(SLPVectorizer, SV_NAME, lv_name, false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_AG_DEPENDENCY(TargetTransformInfo)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_END(SLPVectorizer, SV_NAME, lv_name, false, false)

namespace llvm {
Pass *createSLPVectorizerPass() { return new SLPVectorizer(); }
}
