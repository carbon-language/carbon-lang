//===- VecUtils.cpp --- Vectorization Utilities ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "SLP"

#include "VecUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <map>

using namespace llvm;

static const unsigned MinVecRegSize = 128;

static const unsigned RecursionMaxDepth = 6;

namespace llvm {

BoUpSLP::BoUpSLP(BasicBlock *Bb, ScalarEvolution *S, DataLayout *Dl,
                 TargetTransformInfo *Tti, AliasAnalysis *Aa, Loop *Lp) :
  BB(Bb), SE(S), DL(Dl), TTI(Tti), AA(Aa), L(Lp)  {
  numberInstructions();
}

void BoUpSLP::numberInstructions() {
  int Loc = 0;
  InstrIdx.clear();
  InstrVec.clear();
  // Number the instructions in the block.
  for (BasicBlock::iterator it=BB->begin(), e=BB->end(); it != e; ++it) {
    InstrIdx[it] = Loc++;
    InstrVec.push_back(it);
    assert(InstrVec[InstrIdx[it]] == it && "Invalid allocation");
  }
}

Value *BoUpSLP::getPointerOperand(Value *I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) return LI->getPointerOperand();
  if (StoreInst *SI = dyn_cast<StoreInst>(I)) return SI->getPointerOperand();
  return 0;
}

unsigned BoUpSLP::getAddressSpaceOperand(Value *I) {
  if (LoadInst *L=dyn_cast<LoadInst>(I)) return L->getPointerAddressSpace();
  if (StoreInst *S=dyn_cast<StoreInst>(I)) return S->getPointerAddressSpace();
  return -1;
}

bool BoUpSLP::isConsecutiveAccess(Value *A, Value *B) {
  Value *PtrA = getPointerOperand(A);
  Value *PtrB = getPointerOperand(B);
  unsigned ASA = getAddressSpaceOperand(A);
  unsigned ASB = getAddressSpaceOperand(B);

  // Check that the address spaces match and that the pointers are valid.
  if (!PtrA || !PtrB || (ASA != ASB)) return false;

  // Check that A and B are of the same type.
  if (PtrA->getType() != PtrB->getType()) return false;

  // Calculate the distance.
  const SCEV *PtrSCEVA = SE->getSCEV(PtrA);
  const SCEV *PtrSCEVB = SE->getSCEV(PtrB);
  const SCEV *OffsetSCEV = SE->getMinusSCEV(PtrSCEVA, PtrSCEVB);
  const SCEVConstant *ConstOffSCEV = dyn_cast<SCEVConstant>(OffsetSCEV);

  // Non constant distance.
  if (!ConstOffSCEV) return false;

  int64_t Offset = ConstOffSCEV->getValue()->getSExtValue();
  Type *Ty = cast<PointerType>(PtrA->getType())->getElementType();
  // The Instructions are connsecutive if the size of the first load/store is
  // the same as the offset.
  int64_t Sz = DL->getTypeStoreSize(Ty);
  return ((-Offset) == Sz);
}

bool BoUpSLP::vectorizeStoreChain(ValueList &Chain, int CostThreshold) {
  Type *StoreTy = cast<StoreInst>(Chain[0])->getValueOperand()->getType();
  unsigned Sz = DL->getTypeSizeInBits(StoreTy);
  unsigned VF = MinVecRegSize / Sz;

  if (!isPowerOf2_32(Sz) || VF < 2) return false;

  bool Changed = false;
  // Look for profitable vectorizable trees at all offsets, starting at zero.
  for (unsigned i = 0, e = Chain.size(); i < e; ++i) {
    if (i + VF > e) return Changed;
    DEBUG(dbgs()<<"SLP: Analyzing " << VF << " stores at offset "<< i << "\n");
    ValueList Operands(&Chain[i], &Chain[i] + VF);

    int Cost = getTreeCost(Operands);
    DEBUG(dbgs() << "SLP: Found cost=" << Cost << " for VF=" << VF << "\n");
    if (Cost < CostThreshold) {
      DEBUG(dbgs() << "SLP: Decided to vectorize cost=" << Cost << "\n");
      vectorizeTree(Operands, VF);
      i += VF - 1;
      Changed = true;
    }
  }

  return Changed;
}

bool BoUpSLP::vectorizeStores(StoreList &Stores, int costThreshold) {
  ValueSet Heads, Tails;
  SmallDenseMap<Value*, Value*> ConsecutiveChain;

  // We may run into multiple chains that merge into a single chain. We mark the
  // stores that we vectorized so that we don't visit the same store twice.
  ValueSet VectorizedStores;
  bool Changed = false;

  // Do a quadratic search on all of the given stores and find
  // all of the pairs of loads that follow each other.
  for (unsigned i = 0, e = Stores.size(); i < e; ++i)
    for (unsigned j = 0; j < e; ++j) {
      if (i == j) continue;
      if (isConsecutiveAccess(Stores[i], Stores[j])) {
        Tails.insert(Stores[j]);
        Heads.insert(Stores[i]);
        ConsecutiveChain[Stores[i]] = Stores[j];
      }
    }

  // For stores that start but don't end a link in the chain:
  for (ValueSet::iterator it = Heads.begin(), e = Heads.end();it != e; ++it) {
    if (Tails.count(*it)) continue;

    // We found a store instr that starts a chain. Now follow the chain and try
    // to vectorize it.
    ValueList Operands;
    Value *I = *it;
    // Collect the chain into a list.
    while (Tails.count(I) || Heads.count(I)) {
      if (VectorizedStores.count(I)) break;
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

int BoUpSLP::getScalarizationCost(ValueList &VL) {
  // Find the type of the operands in VL.
  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, VL.size());
  // Find the cost of inserting/extracting values from the vector.
  return getScalarizationCost(VecTy);
}

int BoUpSLP::getScalarizationCost(Type *Ty) {
  int Cost = 0;
  for (unsigned i = 0, e = cast<VectorType>(Ty)->getNumElements(); i < e; ++i)
    Cost += TTI->getVectorInstrCost(Instruction::InsertElement, Ty, i);
  return Cost;
}

AliasAnalysis::Location BoUpSLP::getLocation(Instruction *I) {
  if (StoreInst *SI = dyn_cast<StoreInst>(I)) return AA->getLocation(SI);
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) return AA->getLocation(LI);
  return AliasAnalysis::Location();
}

Value *BoUpSLP::isUnsafeToSink(Instruction *Src, Instruction *Dst) {
  assert(Src->getParent() == Dst->getParent() && "Not the same BB");
  BasicBlock::iterator I = Src, E = Dst;
  /// Scan all of the instruction from SRC to DST and check if
  /// the source may alias.
  for (++I; I != E; ++I) {
    // Ignore store instructions that are marked as 'ignore'.
    if (MemBarrierIgnoreList.count(I)) continue;
    if (Src->mayWriteToMemory()) /* Write */ {
      if (!I->mayReadOrWriteMemory()) continue;
    } else /* Read */ {
      if (!I->mayWriteToMemory()) continue;
    }
    AliasAnalysis::Location A = getLocation(&*I);
    AliasAnalysis::Location B = getLocation(Src);

    if (!A.Ptr || !B.Ptr || AA->alias(A, B))
      return I;
  }
  return 0;
}

void BoUpSLP::vectorizeArith(ValueList &Operands) {
  Value *Vec = vectorizeTree(Operands, Operands.size());
  BasicBlock::iterator Loc = cast<Instruction>(Vec);
  IRBuilder<> Builder(++Loc);
  // After vectorizing the operands we need to generate extractelement
  // instructions and replace all of the uses of the scalar values with
  // the values that we extracted from the vectorized tree.
  for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
    Value *S = Builder.CreateExtractElement(Vec, Builder.getInt32(i));
    Operands[i]->replaceAllUsesWith(S);
  }
}

int BoUpSLP::getTreeCost(ValueList &VL) {
  // Get rid of the list of stores that were removed, and from the
  // lists of instructions with multiple users.
  MemBarrierIgnoreList.clear();
  LaneMap.clear();
  MultiUserVals.clear();
  MustScalarize.clear();

  // Scan the tree and find which value is used by which lane, and which values
  // must be scalarized.
  getTreeUses_rec(VL, 0);

  // Check that instructions with multiple users can be vectorized. Mark unsafe
  // instructions.
  for (ValueSet::iterator it = MultiUserVals.begin(),
       e = MultiUserVals.end(); it != e; ++it) {
    // Check that all of the users of this instr are within the tree
    // and that they are all from the same lane.
    int Lane = -1;
    for (Value::use_iterator I = (*it)->use_begin(), E = (*it)->use_end();
         I != E; ++I) {
      if (LaneMap.find(*I) == LaneMap.end()) {
        MustScalarize.insert(*it);
        DEBUG(dbgs()<<"SLP: Adding " << **it <<
              " to MustScalarize because of an out of tree usage.\n");
        break;
      }
      if (Lane == -1) Lane = LaneMap[*I];
      if (Lane != LaneMap[*I]) {
        MustScalarize.insert(*it);
        DEBUG(dbgs()<<"Adding " << **it <<
              " to MustScalarize because multiple lane use it: "
              << Lane << " and " << LaneMap[*I] << ".\n");
        break;
      }
    }
  }

  // Now calculate the cost of vectorizing the tree.
  return getTreeCost_rec(VL, 0);
}

void BoUpSLP::getTreeUses_rec(ValueList &VL, unsigned Depth) {
  if (Depth == RecursionMaxDepth) return;

  // Don't handle vectors.
  if (VL[0]->getType()->isVectorTy()) return;
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    if (SI->getValueOperand()->getType()->isVectorTy()) return;

  // Check if all of the operands are constants.
  bool AllConst = true;
  bool AllSameScalar = true;
  for (unsigned i = 0, e = VL.size(); i < e; ++i) {
    AllConst &= isa<Constant>(VL[i]);
    AllSameScalar &= (VL[0] == VL[i]);
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    // If one of the instructions is out of this BB, we need to scalarize all.
    if (I && I->getParent() != BB) return;
  }

  // If all of the operands are identical or constant we have a simple solution.
  if (AllConst || AllSameScalar) return;

  // Scalarize unknown structures.
  Instruction *VL0 = dyn_cast<Instruction>(VL[0]);
  if (!VL0) return;

  unsigned Opcode = VL0->getOpcode();
  for (unsigned i = 0, e = VL.size(); i < e; ++i) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    // If not all of the instructions are identical then we have to scalarize.
    if (!I || Opcode != I->getOpcode()) return;
  }

  // Mark instructions with multiple users.
  for (unsigned i = 0, e = VL.size(); i < e; ++i) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    // Remember to check if all of the users of this instr are vectorized
    // within our tree.
    if (I && I->getNumUses() > 1) MultiUserVals.insert(I);
  }

  for (int i = 0, e = VL.size(); i < e; ++i) {
    // Check that the instruction is only used within
    // one lane.
    if (LaneMap.count(VL[i]) && LaneMap[VL[i]] != i) return;
    // Make this instruction as 'seen' and remember the lane.
    LaneMap[VL[i]] = i;
  }

  switch (Opcode) {
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

        getTreeUses_rec(Operands, Depth+1);
      }
    }
    case Instruction::Store: {
      ValueList Operands;
      for (unsigned j = 0; j < VL.size(); ++j)
        Operands.push_back(cast<Instruction>(VL[j])->getOperand(0));
      getTreeUses_rec(Operands, Depth+1);
      return;
    }
    default:
    return;
  }
}

int BoUpSLP::getTreeCost_rec(ValueList &VL, unsigned Depth) {
  Type *ScalarTy = VL[0]->getType();

  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();

  /// Don't mess with vectors.
  if (ScalarTy->isVectorTy()) return max_cost;
  VectorType *VecTy = VectorType::get(ScalarTy, VL.size());

  if (Depth == RecursionMaxDepth) return getScalarizationCost(VecTy);

  // Check if all of the operands are constants.
  bool AllConst = true;
  bool AllSameScalar = true;
  bool MustScalarizeFlag = false;
  for (unsigned i = 0, e = VL.size(); i < e; ++i) {
    AllConst &= isa<Constant>(VL[i]);
    AllSameScalar &= (VL[0] == VL[i]);
    // Must have a single use.
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    MustScalarizeFlag |= MustScalarize.count(VL[i]);
    // This instruction is outside the basic block.
    if (I && I->getParent() != BB)
      return getScalarizationCost(VecTy);
  }

  // Is this a simple vector constant.
  if (AllConst) return 0;

  // If all of the operands are identical we can broadcast them.
  Instruction *VL0 = dyn_cast<Instruction>(VL[0]);
  if (AllSameScalar) {
    // If we are in a loop, and this is not an instruction (e.g. constant or
    // argument) or the instruction is defined outside the loop then assume
    // that the cost is zero.
    if (L && (!VL0 || !L->contains(VL0)))
      return 0;

    // We need to broadcast the scalar.
    return TTI->getShuffleCost(TargetTransformInfo::SK_Broadcast, VecTy, 0);
  }

  // If this is not a constant, or a scalar from outside the loop then we
  // need to scalarize it.
  if (MustScalarizeFlag)
    return getScalarizationCost(VecTy);

  if (!VL0) return getScalarizationCost(VecTy);
  assert(VL0->getParent() == BB && "Wrong BB");

  unsigned Opcode = VL0->getOpcode();
  for (unsigned i = 0, e = VL.size(); i < e; ++i) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    // If not all of the instructions are identical then we have to scalarize.
    if (!I || Opcode != I->getOpcode()) return getScalarizationCost(VecTy);
  }

  // Check if it is safe to sink the loads or the stores.
  if (Opcode == Instruction::Load || Opcode == Instruction::Store) {
    int MaxIdx = InstrIdx[VL0];
    for (unsigned i = 1, e = VL.size(); i < e; ++i )
      MaxIdx = std::max(MaxIdx, InstrIdx[VL[i]]);

    Instruction *Last = InstrVec[MaxIdx];
    for (unsigned i = 0, e = VL.size(); i < e; ++i ) {
      if (VL[i] == Last) continue;
      Value *Barrier = isUnsafeToSink(cast<Instruction>(VL[i]), Last);
      if (Barrier) {
        DEBUG(dbgs() << "SLP: Can't sink " << *VL[i] << "\n down to " <<
              *Last << "\n because of " << *Barrier << "\n");
        return max_cost;
      }
    }
  }

  switch (Opcode) {
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
    int Cost = 0;
    // Calculate the cost of all of the operands.
    for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
      ValueList Operands;
      // Prepare the operand vector.
      for (unsigned j = 0; j < VL.size(); ++j)
        Operands.push_back(cast<Instruction>(VL[j])->getOperand(i));

      Cost += getTreeCost_rec(Operands, Depth+1);
      if (Cost >= max_cost) return max_cost;
    }

    // Calculate the cost of this instruction.
    int ScalarCost = VecTy->getNumElements() *
      TTI->getArithmeticInstrCost(Opcode, ScalarTy);

    int VecCost = TTI->getArithmeticInstrCost(Opcode, VecTy);
    Cost += (VecCost - ScalarCost);
    return Cost;
  }
  case Instruction::Load: {
    // If we are scalarize the loads, add the cost of forming the vector.
    for (unsigned i = 0, e = VL.size()-1; i < e; ++i)
      if (!isConsecutiveAccess(VL[i], VL[i+1]))
        return getScalarizationCost(VecTy);

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
    int VecStCost = TTI->getMemoryOpCost(Instruction::Store, ScalarTy, 1,0);
    int StoreCost = VecStCost - ScalarStCost;

    ValueList Operands;
    for (unsigned j = 0; j < VL.size(); ++j) {
      Operands.push_back(cast<Instruction>(VL[j])->getOperand(0));
      MemBarrierIgnoreList.insert(VL[j]);
    }

    int TotalCost = StoreCost + getTreeCost_rec(Operands, Depth + 1);
    return TotalCost;
  }
  default:
    // Unable to vectorize unknown instructions.
    return getScalarizationCost(VecTy);
  }
}

Instruction *BoUpSLP::GetLastInstr(ValueList &VL, unsigned VF) {
  int MaxIdx = InstrIdx[BB->getFirstNonPHI()];
  for (unsigned i = 0; i < VF; ++i )
    MaxIdx = std::max(MaxIdx, InstrIdx[VL[i]]);
  return InstrVec[MaxIdx + 1];
}

Value *BoUpSLP::Scalarize(ValueList &VL, VectorType *Ty) {
  IRBuilder<> Builder(GetLastInstr(VL, Ty->getNumElements()));
  Value *Vec = UndefValue::get(Ty);
  for (unsigned i=0; i < Ty->getNumElements(); ++i) {
    // Generate the 'InsertElement' instruction.
    Vec = Builder.CreateInsertElement(Vec, VL[i], Builder.getInt32(i));
    // Remember that this instruction is used as part of a 'gather' sequence.
    // The caller of the bottom-up slp vectorizer can try to hoist the sequence
    // if the users are outside of the basic block.
    GatherInstructions.push_back(Vec);
  }

  return Vec;
}

Value *BoUpSLP::vectorizeTree(ValueList &VL, int VF) {
  Value *V = vectorizeTree_rec(VL, VF);
  // We moved some instructions around. We have to number them again
  // before we can do any analysis.
  numberInstructions();
  MustScalarize.clear();
  return V;
}

Value *BoUpSLP::vectorizeTree_rec(ValueList &VL, int VF) {
  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, VF);

  // Check if all of the operands are constants or identical.
  bool AllConst = true;
  bool AllSameScalar = true;
  for (unsigned i = 0, e = VF; i < e; ++i) {
    AllConst &= !!dyn_cast<Constant>(VL[i]);
    AllSameScalar &= (VL[0] == VL[i]);
    // The instruction must be in the same BB, and it must be vectorizable.
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    if (MustScalarize.count(VL[i]) || (I && I->getParent() != BB))
      return Scalarize(VL, VecTy);
  }

  // Check that this is a simple vector constant.
  if (AllConst || AllSameScalar) return Scalarize(VL, VecTy);

  // Scalarize unknown structures.
  Instruction *VL0 = dyn_cast<Instruction>(VL[0]);
  if (!VL0) return Scalarize(VL, VecTy);

  if (VectorizedValues.count(VL0)) return VectorizedValues[VL0];

  unsigned Opcode = VL0->getOpcode();
  for (unsigned i = 0, e = VF; i < e; ++i) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    // If not all of the instructions are identical then we have to scalarize.
    if (!I || Opcode != I->getOpcode()) return Scalarize(VL, VecTy);
  }

  switch (Opcode) {
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
    for (int i = 0; i < VF; ++i) {
      RHSVL.push_back(cast<Instruction>(VL[i])->getOperand(0));
      LHSVL.push_back(cast<Instruction>(VL[i])->getOperand(1));
    }

    Value *RHS = vectorizeTree_rec(RHSVL, VF);
    Value *LHS = vectorizeTree_rec(LHSVL, VF);
    IRBuilder<> Builder(GetLastInstr(VL, VF));
    BinaryOperator *BinOp = dyn_cast<BinaryOperator>(VL0);
    Value *V = Builder.CreateBinOp(BinOp->getOpcode(), RHS,LHS);
    VectorizedValues[VL0] = V;
    return V;
  }
  case Instruction::Load: {
    LoadInst *LI = dyn_cast<LoadInst>(VL0);
    unsigned Alignment = LI->getAlignment();

    // Check if all of the loads are consecutive.
    for (unsigned i = 1, e = VF; i < e; ++i)
      if (!isConsecutiveAccess(VL[i-1], VL[i]))
        return Scalarize(VL, VecTy);

    IRBuilder<> Builder(GetLastInstr(VL, VF));
    Value *VecPtr = Builder.CreateBitCast(LI->getPointerOperand(),
                                          VecTy->getPointerTo());
    LI = Builder.CreateLoad(VecPtr);
    LI->setAlignment(Alignment);
    VectorizedValues[VL0] = LI;
    return LI;
  }
  case Instruction::Store: {
    StoreInst *SI = dyn_cast<StoreInst>(VL0);
    unsigned Alignment = SI->getAlignment();

    ValueList ValueOp;
    for (int i = 0; i < VF; ++i)
      ValueOp.push_back(cast<StoreInst>(VL[i])->getValueOperand());

    Value *VecValue = vectorizeTree_rec(ValueOp, VF);

    IRBuilder<> Builder(GetLastInstr(VL, VF));
    Value *VecPtr = Builder.CreateBitCast(SI->getPointerOperand(),
                                          VecTy->getPointerTo());
    Builder.CreateStore(VecValue, VecPtr)->setAlignment(Alignment);

    for (int i = 0; i < VF; ++i)
      cast<Instruction>(VL[i])->eraseFromParent();
    return 0;
  }
  default:
    Value *S = Scalarize(VL, VecTy);
    VectorizedValues[VL0] = S;
    return S;
  }
}

} // end of namespace
