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
                     cl::desc("Only vectorize trees if the gain is above this "
                              "number. (gain = -cost of vectorization)"));
namespace {

static const unsigned MinVecRegSize = 128;

static const unsigned RecursionMaxDepth = 12;

/// RAII pattern to save the insertion point of the IR builder.
class BuilderLocGuard {
public:
  BuilderLocGuard(IRBuilder<> &B) : Builder(B), Loc(B.GetInsertPoint()) {}
  ~BuilderLocGuard() { Builder.SetInsertPoint(Loc); }

private:
  // Prevent copying.
  BuilderLocGuard(const BuilderLocGuard &);
  BuilderLocGuard &operator=(const BuilderLocGuard &);
  IRBuilder<> &Builder;
  BasicBlock::iterator Loc;
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

class FuncSLP {
  typedef SmallVector<Value *, 8> ValueList;
  typedef SmallVector<Instruction *, 16> InstrList;
  typedef SmallPtrSet<Value *, 16> ValueSet;
  typedef SmallVector<StoreInst *, 8> StoreList;

public:
  static const int MAX_COST = INT_MIN;

  FuncSLP(Function *Func, ScalarEvolution *Se, DataLayout *Dl,
          TargetTransformInfo *Tti, AliasAnalysis *Aa, LoopInfo *Li, 
          DominatorTree *Dt) :
    F(Func), SE(Se), DL(Dl), TTI(Tti), AA(Aa), LI(Li), DT(Dt),
    Builder(Se->getContext()) {
    for (Function::iterator it = F->begin(), e = F->end(); it != e; ++it) {
      BasicBlock *BB = it;
      BlocksNumbers[BB] = BlockNumbering(BB);
    }
  }

  /// \brief Take the pointer operand from the Load/Store instruction.
  /// \returns NULL if this is not a valid Load/Store instruction.
  static Value *getPointerOperand(Value *I);

  /// \brief Take the address space operand from the Load/Store instruction.
  /// \returns -1 if this is not a valid Load/Store instruction.
  static unsigned getAddressSpaceOperand(Value *I);

  /// \returns true if the memory operations A and B are consecutive.
  bool isConsecutiveAccess(Value *A, Value *B);

  /// \brief Vectorize the tree that starts with the elements in \p VL.
  /// \returns the vectorized value.
  Value *vectorizeTree(ArrayRef<Value *> VL);

  /// \returns the vectorization cost of the subtree that starts at \p VL.
  /// A negative number means that this is profitable.
  int getTreeCost(ArrayRef<Value *> VL);

  /// \returns the scalarization cost for this list of values. Assuming that
  /// this subtree gets vectorized, we may need to extract the values from the
  /// roots. This method calculates the cost of extracting the values.
  int getGatherCost(ArrayRef<Value *> VL);

  /// \brief Attempts to order and vectorize a sequence of stores. This
  /// function does a quadratic scan of the given stores.
  /// \returns true if the basic block was modified.
  bool vectorizeStores(ArrayRef<StoreInst *> Stores, int costThreshold);

  /// \brief Vectorize a group of scalars into a vector tree.
  /// \returns the vectorized value.
  Value *vectorizeArith(ArrayRef<Value *> Operands);

  /// \brief This method contains the recursive part of getTreeCost.
  int getTreeCost_rec(ArrayRef<Value *> VL, unsigned Depth);

  /// \brief This recursive method looks for vectorization hazards such as
  /// values that are used by multiple users and checks that values are used
  /// by only one vector lane. It updates the variables LaneMap, MultiUserVals.
  void getTreeUses_rec(ArrayRef<Value *> VL, unsigned Depth);

  /// \brief This method contains the recursive part of vectorizeTree.
  Value *vectorizeTree_rec(ArrayRef<Value *> VL);

  ///  \brief Vectorize a sorted sequence of stores.
  bool vectorizeStoreChain(ArrayRef<Value *> Chain, int CostThreshold);

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

  /// \brief Perform LICM and CSE on the newly generated gather sequences.
  void optimizeGatherSequence();

  bool needToGatherAny(ArrayRef<Value *> VL) {
    for (int i = 0, e = VL.size(); i < e; ++i)
      if (MustGather.count(VL[i]))
        return true;
    return false;
  }

  /// -- Vectorization State --

  /// Maps values in the tree to the vector lanes that uses them. This map must
  /// be reset between runs of getCost.
  std::map<Value *, int> LaneMap;
  /// A list of instructions to ignore while sinking
  /// memory instructions. This map must be reset between runs of getCost.
  ValueSet MemBarrierIgnoreList;

  /// Maps between the first scalar to the vector. This map must be reset
  /// between runs.
  DenseMap<Value *, Value *> VectorizedValues;

  /// Contains values that must be gathered because they are used
  /// by multiple lanes, or by users outside the tree.
  /// NOTICE: The vectorization methods also use this set.
  ValueSet MustGather;

  /// Contains a list of values that are used outside the current tree. This
  /// set must be reset between runs.
  SetVector<Value *> MultiUserVals;

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

int FuncSLP::getGatherCost(Type *Ty) {
  int Cost = 0;
  for (unsigned i = 0, e = cast<VectorType>(Ty)->getNumElements(); i < e; ++i)
    Cost += TTI->getVectorInstrCost(Instruction::InsertElement, Ty, i);
  return Cost;
}

int FuncSLP::getGatherCost(ArrayRef<Value *> VL) {
  // Find the type of the operands in VL.
  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, VL.size());
  // Find the cost of inserting/extracting values from the vector.
  return getGatherCost(VecTy);
}

AliasAnalysis::Location FuncSLP::getLocation(Instruction *I) {
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return AA->getLocation(SI);
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return AA->getLocation(LI);
  return AliasAnalysis::Location();
}

Value *FuncSLP::getPointerOperand(Value *I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return LI->getPointerOperand();
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->getPointerOperand();
  return 0;
}

unsigned FuncSLP::getAddressSpaceOperand(Value *I) {
  if (LoadInst *L = dyn_cast<LoadInst>(I))
    return L->getPointerAddressSpace();
  if (StoreInst *S = dyn_cast<StoreInst>(I))
    return S->getPointerAddressSpace();
  return -1;
}

bool FuncSLP::isConsecutiveAccess(Value *A, Value *B) {
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

Value *FuncSLP::getSinkBarrier(Instruction *Src, Instruction *Dst) {
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

static BasicBlock *getSameBlock(ArrayRef<Value *> VL) {
  BasicBlock *BB = 0;
  for (int i = 0, e = VL.size(); i < e; i++) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    if (!I)
      return 0;

    if (!BB) {
      BB = I->getParent();
      continue;
    }

    if (BB != I->getParent())
      return 0;
  }
  return BB;
}

static bool allConstant(ArrayRef<Value *> VL) {
  for (unsigned i = 0, e = VL.size(); i < e; ++i)
    if (!isa<Constant>(VL[i]))
      return false;
  return true;
}

static bool isSplat(ArrayRef<Value *> VL) {
  for (unsigned i = 1, e = VL.size(); i < e; ++i)
    if (VL[i] != VL[0])
      return false;
  return true;
}

static unsigned getSameOpcode(ArrayRef<Value *> VL) {
  unsigned Opcode = 0;
  for (int i = 0, e = VL.size(); i < e; i++) {
    if (Instruction *I = dyn_cast<Instruction>(VL[i])) {
      if (!Opcode) {
        Opcode = I->getOpcode();
        continue;
      }
      if (Opcode != I->getOpcode())
        return 0;
    }
  }
  return Opcode;
}

static bool CanReuseExtract(ArrayRef<Value *> VL, unsigned VF,
                            VectorType *VecTy) {
  assert(Instruction::ExtractElement == getSameOpcode(VL) && "Invalid opcode");
  // Check if all of the extracts come from the same vector and from the
  // correct offset.
  Value *VL0 = VL[0];
  ExtractElementInst *E0 = cast<ExtractElementInst>(VL0);
  Value *Vec = E0->getOperand(0);

  // We have to extract from the same vector type.
  if (Vec->getType() != VecTy)
    return false;

  // Check that all of the indices extract from the correct offset.
  ConstantInt *CI = dyn_cast<ConstantInt>(E0->getOperand(1));
  if (!CI || CI->getZExtValue())
    return false;

  for (unsigned i = 1, e = VF; i < e; ++i) {
    ExtractElementInst *E = cast<ExtractElementInst>(VL[i]);
    ConstantInt *CI = dyn_cast<ConstantInt>(E->getOperand(1));

    if (!CI || CI->getZExtValue() != i || E->getOperand(0) != Vec)
      return false;
  }

  return true;
}

void FuncSLP::getTreeUses_rec(ArrayRef<Value *> VL, unsigned Depth) {
  if (Depth == RecursionMaxDepth)
    return MustGather.insert(VL.begin(), VL.end());

  // Don't handle vectors.
  if (VL[0]->getType()->isVectorTy())
    return;

  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    if (SI->getValueOperand()->getType()->isVectorTy())
      return;

  // If all of the operands are identical or constant we have a simple solution.
  if (allConstant(VL) || isSplat(VL) || !getSameBlock(VL))
    return MustGather.insert(VL.begin(), VL.end());

  // Stop the scan at unknown IR.
  Instruction *VL0 = dyn_cast<Instruction>(VL[0]);
  assert(VL0 && "Invalid instruction");

  // Mark instructions with multiple users.
  for (unsigned i = 0, e = VL.size(); i < e; ++i) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    // Remember to check if all of the users of this instruction are vectorized
    // within our tree. At depth zero we have no local users, only external
    // users that we don't care about.
    if (Depth && I && I->getNumUses() > 1) {
      DEBUG(dbgs() << "SLP: Adding to MultiUserVals "
                      "because it has multiple users:" << *I << " \n");
      MultiUserVals.insert(I);
    }
  }

  // Check that the instruction is only used within one lane.
  for (int i = 0, e = VL.size(); i < e; ++i) {
    if (LaneMap.count(VL[i]) && LaneMap[VL[i]] != i) {
      DEBUG(dbgs() << "SLP: Value used by multiple lanes:" << *VL[i] << "\n");
      return MustGather.insert(VL.begin(), VL.end());
    }
    // Make this instruction as 'seen' and remember the lane.
    LaneMap[VL[i]] = i;
  }

  unsigned Opcode = getSameOpcode(VL);
  if (!Opcode)
    return MustGather.insert(VL.begin(), VL.end());

  switch (Opcode) {
  case Instruction::ExtractElement: {
    VectorType *VecTy = VectorType::get(VL[0]->getType(), VL.size());
    // No need to follow ExtractElements that are going to be optimized away.
    if (CanReuseExtract(VL, VL.size(), VecTy))
      return;
    // Fall through.
  }
  case Instruction::Load:
    return;
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
  case Instruction::BitCast:
  case Instruction::Select:
  case Instruction::ICmp:
  case Instruction::FCmp:
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
    for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
      ValueList Operands;
      // Prepare the operand vector.
      for (unsigned j = 0; j < VL.size(); ++j)
        Operands.push_back(cast<Instruction>(VL[j])->getOperand(i));

      getTreeUses_rec(Operands, Depth + 1);
    }
    return;
  }
  case Instruction::Store: {
    ValueList Operands;
    for (unsigned j = 0; j < VL.size(); ++j)
      Operands.push_back(cast<Instruction>(VL[j])->getOperand(0));
    getTreeUses_rec(Operands, Depth + 1);
    return;
  }
  default:
    return MustGather.insert(VL.begin(), VL.end());
  }
}

int FuncSLP::getLastIndex(ArrayRef<Value *> VL) {
  BasicBlock *BB = cast<Instruction>(VL[0])->getParent();
  assert(BB == getSameBlock(VL) && BlocksNumbers.count(BB) && "Invalid block");
  BlockNumbering &BN = BlocksNumbers[BB];

  int MaxIdx = BN.getIndex(BB->getFirstNonPHI());
  for (unsigned i = 0, e = VL.size(); i < e; ++i)
    MaxIdx = std::max(MaxIdx, BN.getIndex(cast<Instruction>(VL[i])));
  return MaxIdx;
}

Instruction *FuncSLP::getLastInstruction(ArrayRef<Value *> VL) {
  BasicBlock *BB = cast<Instruction>(VL[0])->getParent();
  assert(BB == getSameBlock(VL) && BlocksNumbers.count(BB) && "Invalid block");
  BlockNumbering &BN = BlocksNumbers[BB];

  int MaxIdx = BN.getIndex(cast<Instruction>(VL[0]));
  for (unsigned i = 1, e = VL.size(); i < e; ++i)
    MaxIdx = std::max(MaxIdx, BN.getIndex(cast<Instruction>(VL[i])));
  return BN.getInstruction(MaxIdx);
}

Instruction *FuncSLP::getInstructionForIndex(unsigned Index, BasicBlock *BB) {
  BlockNumbering &BN = BlocksNumbers[BB];
  return BN.getInstruction(Index);
}

int FuncSLP::getFirstUserIndex(ArrayRef<Value *> VL) {
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

int FuncSLP::getTreeCost_rec(ArrayRef<Value *> VL, unsigned Depth) {
  Type *ScalarTy = VL[0]->getType();

  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();

  /// Don't mess with vectors.
  if (ScalarTy->isVectorTy())
    return FuncSLP::MAX_COST;

  if (allConstant(VL))
    return 0;

  VectorType *VecTy = VectorType::get(ScalarTy, VL.size());

  if (isSplat(VL))
    return TTI->getShuffleCost(TargetTransformInfo::SK_Broadcast, VecTy, 0);

  int GatherCost = getGatherCost(VecTy);
  if (Depth == RecursionMaxDepth || needToGatherAny(VL))
    return GatherCost;

  BasicBlock *BB = getSameBlock(VL);
  unsigned Opcode = getSameOpcode(VL);
  assert(Opcode && BB && "Invalid Instruction Value");

  // Check if it is safe to sink the loads or the stores.
  if (Opcode == Instruction::Load || Opcode == Instruction::Store) {
    int MaxIdx = getLastIndex(VL);
    Instruction *Last = getInstructionForIndex(MaxIdx, BB);

    for (unsigned i = 0, e = VL.size(); i < e; ++i) {
      if (VL[i] == Last)
        continue;
      Value *Barrier = getSinkBarrier(cast<Instruction>(VL[i]), Last);
      if (Barrier) {
        DEBUG(dbgs() << "SLP: Can't sink " << *VL[i] << "\n down to " << *Last
                     << "\n because of " << *Barrier << "\n");
        return MAX_COST;
      }
    }
  }

  Instruction *VL0 = cast<Instruction>(VL[0]);
  switch (Opcode) {
  case Instruction::ExtractElement: {
    if (CanReuseExtract(VL, VL.size(), VecTy))
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
    ValueList Operands;
    Type *SrcTy = VL0->getOperand(0)->getType();
    // Prepare the operand vector.
    for (unsigned j = 0; j < VL.size(); ++j) {
      Operands.push_back(cast<Instruction>(VL[j])->getOperand(0));
      // Check that the casted type is the same for all users.
      if (cast<Instruction>(VL[j])->getOperand(0)->getType() != SrcTy)
        return getGatherCost(VecTy);
    }

    int Cost = getTreeCost_rec(Operands, Depth + 1);
    if (Cost == FuncSLP::MAX_COST)
      return Cost;

    // Calculate the cost of this instruction.
    int ScalarCost = VL.size() * TTI->getCastInstrCost(VL0->getOpcode(),
                                                       VL0->getType(), SrcTy);

    VectorType *SrcVecTy = VectorType::get(SrcTy, VL.size());
    int VecCost = TTI->getCastInstrCost(VL0->getOpcode(), VecTy, SrcVecTy);
    Cost += (VecCost - ScalarCost);

    if (Cost > GatherCost) {
      MustGather.insert(VL.begin(), VL.end());
      return GatherCost;
    }

    return Cost;
  }
  case Instruction::FCmp:
  case Instruction::ICmp: {
    // Check that all of the compares have the same predicate.
    CmpInst::Predicate P0 = dyn_cast<CmpInst>(VL0)->getPredicate();
    for (unsigned i = 1, e = VL.size(); i < e; ++i) {
      CmpInst *Cmp = cast<CmpInst>(VL[i]);
      if (Cmp->getPredicate() != P0)
        return getGatherCost(VecTy);
    }
    // Fall through.
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
    int TotalCost = 0;
    // Calculate the cost of all of the operands.
    for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
      ValueList Operands;
      // Prepare the operand vector.
      for (unsigned j = 0; j < VL.size(); ++j)
        Operands.push_back(cast<Instruction>(VL[j])->getOperand(i));

      int Cost = getTreeCost_rec(Operands, Depth + 1);
      if (Cost == MAX_COST)
        return MAX_COST;
      TotalCost += TotalCost;
    }

    // Calculate the cost of this instruction.
    int ScalarCost = 0;
    int VecCost = 0;
    if (Opcode == Instruction::FCmp || Opcode == Instruction::ICmp ||
        Opcode == Instruction::Select) {
      VectorType *MaskTy = VectorType::get(Builder.getInt1Ty(), VL.size());
      ScalarCost =
          VecTy->getNumElements() *
          TTI->getCmpSelInstrCost(Opcode, ScalarTy, Builder.getInt1Ty());
      VecCost = TTI->getCmpSelInstrCost(Opcode, VecTy, MaskTy);
    } else {
      ScalarCost = VecTy->getNumElements() *
                   TTI->getArithmeticInstrCost(Opcode, ScalarTy);
      VecCost = TTI->getArithmeticInstrCost(Opcode, VecTy);
    }
    TotalCost += (VecCost - ScalarCost);

    if (TotalCost > GatherCost) {
      MustGather.insert(VL.begin(), VL.end());
      return GatherCost;
    }

    return TotalCost;
  }
  case Instruction::Load: {
    // If we are scalarize the loads, add the cost of forming the vector.
    for (unsigned i = 0, e = VL.size() - 1; i < e; ++i)
      if (!isConsecutiveAccess(VL[i], VL[i + 1]))
        return getGatherCost(VecTy);

    // Cost of wide load - cost of scalar loads.
    int ScalarLdCost = VecTy->getNumElements() *
                       TTI->getMemoryOpCost(Instruction::Load, ScalarTy, 1, 0);
    int VecLdCost = TTI->getMemoryOpCost(Instruction::Load, ScalarTy, 1, 0);
    int TotalCost = VecLdCost - ScalarLdCost;

    if (TotalCost > GatherCost) {
      MustGather.insert(VL.begin(), VL.end());
      return GatherCost;
    }

    return TotalCost;
  }
  case Instruction::Store: {
    // We know that we can merge the stores. Calculate the cost.
    int ScalarStCost = VecTy->getNumElements() *
                       TTI->getMemoryOpCost(Instruction::Store, ScalarTy, 1, 0);
    int VecStCost = TTI->getMemoryOpCost(Instruction::Store, ScalarTy, 1, 0);
    int StoreCost = VecStCost - ScalarStCost;

    ValueList Operands;
    for (unsigned j = 0; j < VL.size(); ++j) {
      Operands.push_back(cast<Instruction>(VL[j])->getOperand(0));
      MemBarrierIgnoreList.insert(VL[j]);
    }

    int Cost = getTreeCost_rec(Operands, Depth + 1);
    if (Cost == MAX_COST)
      return MAX_COST;

    int TotalCost = StoreCost + Cost;
    return TotalCost;
  }
  default:
    // Unable to vectorize unknown instructions.
    return getGatherCost(VecTy);
  }
}

int FuncSLP::getTreeCost(ArrayRef<Value *> VL) {
  // Get rid of the list of stores that were removed, and from the
  // lists of instructions with multiple users.
  MemBarrierIgnoreList.clear();
  LaneMap.clear();
  MultiUserVals.clear();
  MustGather.clear();

  if (!getSameBlock(VL))
    return MAX_COST;

  // Find the location of the last root.
  int LastRootIndex = getLastIndex(VL);
  int FirstUserIndex = getFirstUserIndex(VL);

  // Don't vectorize if there are users of the tree roots inside the tree
  // itself.
  if (LastRootIndex > FirstUserIndex)
    return MAX_COST;

  // Scan the tree and find which value is used by which lane, and which values
  // must be scalarized.
  getTreeUses_rec(VL, 0);

  // Check that instructions with multiple users can be vectorized. Mark unsafe
  // instructions.
  for (SetVector<Value *>::iterator it = MultiUserVals.begin(),
                                    e = MultiUserVals.end();
       it != e; ++it) {
    // Check that all of the users of this instr are within the tree.
    for (Value::use_iterator I = (*it)->use_begin(), E = (*it)->use_end();
         I != E; ++I) {
      if (LaneMap.find(*I) == LaneMap.end()) {
        DEBUG(dbgs() << "SLP: Adding to MustExtract "
                        "because of an out of tree usage.\n");
        MustGather.insert(*it);
        continue;
      }
    }
  }

  // Now calculate the cost of vectorizing the tree.
  return getTreeCost_rec(VL, 0);
}
bool FuncSLP::vectorizeStoreChain(ArrayRef<Value *> Chain, int CostThreshold) {
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

    int Cost = getTreeCost(Operands);
    if (Cost == FuncSLP::MAX_COST)
      continue;
    DEBUG(dbgs() << "SLP: Found cost=" << Cost << " for VF=" << VF << "\n");
    if (Cost < CostThreshold) {
      DEBUG(dbgs() << "SLP: Decided to vectorize cost=" << Cost << "\n");
      vectorizeTree(Operands);

      // Remove the scalar stores.
      for (int j = 0, e = VF; j < e; ++j)
        cast<Instruction>(Operands[j])->eraseFromParent();

      // Move to the next bundle.
      i += VF - 1;
      Changed = true;
    }
  }

  if (Changed || ChainLen > VF)
    return Changed;

  // Handle short chains. This helps us catch types such as <3 x float> that
  // are smaller than vector size.
  int Cost = getTreeCost(Chain);
  if (Cost == FuncSLP::MAX_COST)
    return false;
  if (Cost < CostThreshold) {
    DEBUG(dbgs() << "SLP: Found store chain cost = " << Cost
                 << " for size = " << ChainLen << "\n");
    vectorizeTree(Chain);

    // Remove all of the scalar stores.
    for (int i = 0, e = Chain.size(); i < e; ++i)
      cast<Instruction>(Chain[i])->eraseFromParent();

    return true;
  }

  return false;
}

bool FuncSLP::vectorizeStores(ArrayRef<StoreInst *> Stores, int costThreshold) {
  SetVector<Value *> Heads, Tails;
  SmallDenseMap<Value *, Value *> ConsecutiveChain;

  // We may run into multiple chains that merge into a single chain. We mark the
  // stores that we vectorized so that we don't visit the same store twice.
  ValueSet VectorizedStores;
  bool Changed = false;

  // Do a quadratic search on all of the given stores and find
  // all of the pairs of loads that follow each other.
  for (unsigned i = 0, e = Stores.size(); i < e; ++i)
    for (unsigned j = 0; j < e; ++j) {
      if (i == j)
        continue;

      if (isConsecutiveAccess(Stores[i], Stores[j])) {
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
    ValueList Operands;
    Value *I = *it;
    // Collect the chain into a list.
    while (Tails.count(I) || Heads.count(I)) {
      if (VectorizedStores.count(I))
        break;
      Operands.push_back(I);
      // Move to the next value in the chain.
      I = ConsecutiveChain[I];
    }

    bool Vectorized = vectorizeStoreChain(Operands, costThreshold);

    // Mark the vectorized stores so that we don't vectorize them again.
    if (Vectorized)
      VectorizedStores.insert(Operands.begin(), Operands.end());
    Changed |= Vectorized;
  }

  return Changed;
}

Value *FuncSLP::Gather(ArrayRef<Value *> VL, VectorType *Ty) {
  Value *Vec = UndefValue::get(Ty);
  // Generate the 'InsertElement' instruction.
  for (unsigned i = 0; i < Ty->getNumElements(); ++i) {
    Vec = Builder.CreateInsertElement(Vec, VL[i], Builder.getInt32(i));
    if (Instruction *I = dyn_cast<Instruction>(Vec))
      GatherSeq.insert(I);
  }

  return Vec;
}

Value *FuncSLP::vectorizeTree_rec(ArrayRef<Value *> VL) {
  BuilderLocGuard Guard(Builder);

  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, VL.size());

  if (needToGatherAny(VL))
    return Gather(VL, VecTy);

  if (VectorizedValues.count(VL[0])) {
    DEBUG(dbgs() << "SLP: Diamond merged at depth.\n");
    return VectorizedValues[VL[0]];
  }

  Instruction *VL0 = cast<Instruction>(VL[0]);
  unsigned Opcode = VL0->getOpcode();
  assert(Opcode == getSameOpcode(VL) && "Invalid opcode");

  switch (Opcode) {
  case Instruction::ExtractElement: {
    if (CanReuseExtract(VL, VL.size(), VecTy))
      return VL0->getOperand(0);
    return Gather(VL, VecTy);
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
    for (int i = 0, e = VL.size(); i < e; ++i)
      INVL.push_back(cast<Instruction>(VL[i])->getOperand(0));

    Builder.SetInsertPoint(getLastInstruction(VL));
    Value *InVec = vectorizeTree_rec(INVL);
    CastInst *CI = dyn_cast<CastInst>(VL0);
    Value *V = Builder.CreateCast(CI->getOpcode(), InVec, VecTy);
    VectorizedValues[VL0] = V;
    return V;
  }
  case Instruction::FCmp:
  case Instruction::ICmp: {
    // Check that all of the compares have the same predicate.
    CmpInst::Predicate P0 = dyn_cast<CmpInst>(VL0)->getPredicate();
    for (unsigned i = 1, e = VL.size(); i < e; ++i) {
      CmpInst *Cmp = cast<CmpInst>(VL[i]);
      if (Cmp->getPredicate() != P0)
        return Gather(VL, VecTy);
    }

    ValueList LHSV, RHSV;
    for (int i = 0, e = VL.size(); i < e; ++i) {
      LHSV.push_back(cast<Instruction>(VL[i])->getOperand(0));
      RHSV.push_back(cast<Instruction>(VL[i])->getOperand(1));
    }

    Builder.SetInsertPoint(getLastInstruction(VL));
    Value *L = vectorizeTree_rec(LHSV);
    Value *R = vectorizeTree_rec(RHSV);
    Value *V;

    if (Opcode == Instruction::FCmp)
      V = Builder.CreateFCmp(P0, L, R);
    else
      V = Builder.CreateICmp(P0, L, R);

    VectorizedValues[VL0] = V;
    return V;
  }
  case Instruction::Select: {
    ValueList TrueVec, FalseVec, CondVec;
    for (int i = 0, e = VL.size(); i < e; ++i) {
      CondVec.push_back(cast<Instruction>(VL[i])->getOperand(0));
      TrueVec.push_back(cast<Instruction>(VL[i])->getOperand(1));
      FalseVec.push_back(cast<Instruction>(VL[i])->getOperand(2));
    }

    Builder.SetInsertPoint(getLastInstruction(VL));
    Value *True = vectorizeTree_rec(TrueVec);
    Value *False = vectorizeTree_rec(FalseVec);
    Value *Cond = vectorizeTree_rec(CondVec);
    Value *V = Builder.CreateSelect(Cond, True, False);
    VectorizedValues[VL0] = V;
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
    for (int i = 0, e = VL.size(); i < e; ++i) {
      LHSVL.push_back(cast<Instruction>(VL[i])->getOperand(0));
      RHSVL.push_back(cast<Instruction>(VL[i])->getOperand(1));
    }

    Builder.SetInsertPoint(getLastInstruction(VL));
    Value *LHS = vectorizeTree_rec(LHSVL);
    Value *RHS = vectorizeTree_rec(RHSVL);

    if (LHS == RHS) {
      assert((VL0->getOperand(0) == VL0->getOperand(1)) && "Invalid order");
    }

    BinaryOperator *BinOp = cast<BinaryOperator>(VL0);
    Value *V = Builder.CreateBinOp(BinOp->getOpcode(), LHS, RHS);
    VectorizedValues[VL0] = V;
    return V;
  }
  case Instruction::Load: {
    // Check if all of the loads are consecutive.
    for (unsigned i = 1, e = VL.size(); i < e; ++i)
      if (!isConsecutiveAccess(VL[i - 1], VL[i]))
        return Gather(VL, VecTy);

    // Loads are inserted at the head of the tree because we don't want to
    // sink them all the way down past store instructions.
    Builder.SetInsertPoint(getLastInstruction(VL));
    LoadInst *LI = cast<LoadInst>(VL0);
    Value *VecPtr =
        Builder.CreateBitCast(LI->getPointerOperand(), VecTy->getPointerTo());
    unsigned Alignment = LI->getAlignment();
    LI = Builder.CreateLoad(VecPtr);
    LI->setAlignment(Alignment);

    VectorizedValues[VL0] = LI;
    return LI;
  }
  case Instruction::Store: {
    StoreInst *SI = cast<StoreInst>(VL0);
    unsigned Alignment = SI->getAlignment();

    ValueList ValueOp;
    for (int i = 0, e = VL.size(); i < e; ++i)
      ValueOp.push_back(cast<StoreInst>(VL[i])->getValueOperand());

    Value *VecValue = vectorizeTree_rec(ValueOp);

    Builder.SetInsertPoint(getLastInstruction(VL));
    Value *VecPtr =
        Builder.CreateBitCast(SI->getPointerOperand(), VecTy->getPointerTo());
    Builder.CreateStore(VecValue, VecPtr)->setAlignment(Alignment);
    return 0;
  }
  default:
    return Gather(VL, VecTy);
  }
}

Value *FuncSLP::vectorizeTree(ArrayRef<Value *> VL) {
  Builder.SetInsertPoint(getLastInstruction(VL));
  Value *V = vectorizeTree_rec(VL);

  // We moved some instructions around. We have to number them again
  // before we can do any analysis.
  for (Function::iterator it = F->begin(), e = F->end(); it != e; ++it)
    BlocksNumbers[it].forget();
  // Clear the state.
  MustGather.clear();
  VectorizedValues.clear();
  MemBarrierIgnoreList.clear();
  return V;
}

Value *FuncSLP::vectorizeArith(ArrayRef<Value *> Operands) {
  Value *Vec = vectorizeTree(Operands);
  // After vectorizing the operands we need to generate extractelement
  // instructions and replace all of the uses of the scalar values with
  // the values that we extracted from the vectorized tree.
  for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
    Value *S = Builder.CreateExtractElement(Vec, Builder.getInt32(i));
    Operands[i]->replaceAllUsesWith(S);
  }

  return Vec;
}

void FuncSLP::optimizeGatherSequence() {
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
      return;

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
          break;
        }
      }
      Visited.insert(Insert);
    }
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
    FuncSLP R(&F, SE, DL, TTI, AA, LI, DT);

    for (Function::iterator it = F.begin(), e = F.end(); it != e; ++it) {
      BasicBlock *BB = it;

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
  }

private:

  /// \brief Collect memory references and sort them according to their base
  /// object. We sort the stores to their base objects to reduce the cost of the
  /// quadratic search on the stores. TODO: We can further reduce this cost
  /// if we flush the chain creation every time we run into a memory barrier.
  unsigned collectStores(BasicBlock *BB, FuncSLP &R);

  /// \brief Try to vectorize a chain that starts at two arithmetic instrs.
  bool tryToVectorizePair(Value *A, Value *B, FuncSLP &R);

  /// \brief Try to vectorize a list of operands. If \p NeedExtracts is true
  /// then we calculate the cost of extracting the scalars from the vector.
  /// \returns true if a value was vectorized.
  bool tryToVectorizeList(ArrayRef<Value *> VL, FuncSLP &R, bool NeedExtracts);

  /// \brief Try to vectorize a chain that may start at the operands of \V;
  bool tryToVectorize(BinaryOperator *V, FuncSLP &R);

  /// \brief Vectorize the stores that were collected in StoreRefs.
  bool vectorizeStoreChains(FuncSLP &R);

  /// \brief Scan the basic block and look for patterns that are likely to start
  /// a vectorization chain.
  bool vectorizeChainsInBlock(BasicBlock *BB, FuncSLP &R);

private:
  StoreListMap StoreRefs;
};

unsigned SLPVectorizer::collectStores(BasicBlock *BB, FuncSLP &R) {
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

bool SLPVectorizer::tryToVectorizePair(Value *A, Value *B, FuncSLP &R) {
  if (!A || !B)
    return false;
  Value *VL[] = { A, B };
  return tryToVectorizeList(VL, R, true);
}

bool SLPVectorizer::tryToVectorizeList(ArrayRef<Value *> VL, FuncSLP &R,
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

  int Cost = R.getTreeCost(VL);
  if (Cost == FuncSLP::MAX_COST)
    return false;

  int ExtrCost = NeedExtracts ? R.getGatherCost(VL) : 0;
  DEBUG(dbgs() << "SLP: Cost of pair:" << Cost
               << " Cost of extract:" << ExtrCost << ".\n");
  if ((Cost + ExtrCost) >= -SLPCostThreshold)
    return false;
  DEBUG(dbgs() << "SLP: Vectorizing pair.\n");
  R.vectorizeArith(VL);
  return true;
}

bool SLPVectorizer::tryToVectorize(BinaryOperator *V, FuncSLP &R) {
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

bool SLPVectorizer::vectorizeChainsInBlock(BasicBlock *BB, FuncSLP &R) {
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

bool SLPVectorizer::vectorizeStoreChains(FuncSLP &R) {
  bool Changed = false;
  // Attempt to sort and vectorize each of the store-groups.
  for (StoreListMap::iterator it = StoreRefs.begin(), e = StoreRefs.end();
       it != e; ++it) {
    if (it->second.size() < 2)
      continue;

    DEBUG(dbgs() << "SLP: Analyzing a store chain of length "
                 << it->second.size() << ".\n");

    Changed |= R.vectorizeStores(it->second, -SLPCostThreshold);
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
