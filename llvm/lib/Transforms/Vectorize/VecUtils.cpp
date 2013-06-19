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
  Builder(S->getContext()), BB(Bb), SE(S), DL(Dl), TTI(Tti), AA(Aa), L(Lp) {
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

bool BoUpSLP::vectorizeStoreChain(ArrayRef<Value *> Chain, int CostThreshold) {
  Type *StoreTy = cast<StoreInst>(Chain[0])->getValueOperand()->getType();
  unsigned Sz = DL->getTypeSizeInBits(StoreTy);
  unsigned VF = MinVecRegSize / Sz;

  if (!isPowerOf2_32(Sz) || VF < 2) return false;

  bool Changed = false;
  // Look for profitable vectorizable trees at all offsets, starting at zero.
  for (unsigned i = 0, e = Chain.size(); i < e; ++i) {
    if (i + VF > e) return Changed;
    DEBUG(dbgs()<<"SLP: Analyzing " << VF << " stores at offset "<< i << "\n");
    ArrayRef<Value *> Operands = Chain.slice(i, VF);

    int Cost = getTreeCost(Operands);
    DEBUG(dbgs() << "SLP: Found cost=" << Cost << " for VF=" << VF << "\n");
    if (Cost < CostThreshold) {
      DEBUG(dbgs() << "SLP: Decided to vectorize cost=" << Cost << "\n");
      Builder.SetInsertPoint(getInsertionPoint(getLastIndex(Operands,VF)));
      vectorizeTree(Operands, VF);
      i += VF - 1;
      Changed = true;
    }
  }

  return Changed;
}

bool BoUpSLP::vectorizeStores(ArrayRef<StoreInst *> Stores, int costThreshold) {
  SetVector<Value*> Heads, Tails;
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
  for (SetVector<Value*>::iterator it = Heads.begin(), e = Heads.end();
       it != e; ++it) {
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

int BoUpSLP::getScalarizationCost(ArrayRef<Value *> VL) {
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

Value *BoUpSLP::vectorizeArith(ArrayRef<Value *> Operands) {
  int LastIdx = getLastIndex(Operands, Operands.size());
  Instruction *Loc = getInsertionPoint(LastIdx);
  Builder.SetInsertPoint(Loc);

  assert(getFirstUserIndex(Operands, Operands.size()) > LastIdx  &&
         "Vectorizing with in-tree users");

  Value *Vec = vectorizeTree(Operands, Operands.size());
  // After vectorizing the operands we need to generate extractelement
  // instructions and replace all of the uses of the scalar values with
  // the values that we extracted from the vectorized tree.
  for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
    Value *S = Builder.CreateExtractElement(Vec, Builder.getInt32(i));
    Operands[i]->replaceAllUsesWith(S);
  }

  return Vec;
}

int BoUpSLP::getTreeCost(ArrayRef<Value *> VL) {
  // Get rid of the list of stores that were removed, and from the
  // lists of instructions with multiple users.
  MemBarrierIgnoreList.clear();
  LaneMap.clear();
  MultiUserVals.clear();
  MustScalarize.clear();
  MustExtract.clear();

  // Find the location of the last root.
  int LastRootIndex = getLastIndex(VL, VL.size());
  int FirstUserIndex = getFirstUserIndex(VL, VL.size());

  // Don't vectorize if there are users of the tree roots inside the tree
  // itself.
  if (LastRootIndex > FirstUserIndex)
    return max_cost;

  // Scan the tree and find which value is used by which lane, and which values
  // must be scalarized.
  getTreeUses_rec(VL, 0);

  // Check that instructions with multiple users can be vectorized. Mark unsafe
  // instructions.
  for (SetVector<Value*>::iterator it = MultiUserVals.begin(),
       e = MultiUserVals.end(); it != e; ++it) {
    // Check that all of the users of this instr are within the tree
    // and that they are all from the same lane.
    int Lane = -1;
    for (Value::use_iterator I = (*it)->use_begin(), E = (*it)->use_end();
         I != E; ++I) {
      if (LaneMap.find(*I) == LaneMap.end()) {
        DEBUG(dbgs()<<"SLP: Instr " << **it << " has multiple users.\n");

        // We don't have an ordering problem if the user is not in this basic
        // block.
        Instruction *Inst = cast<Instruction>(*I);
        if (Inst->getParent() != BB) {
          MustExtract.insert(*it);
          continue;
        }

        // We don't have an ordering problem if the user is after the last root.
        int Idx = InstrIdx[Inst];
        if (Idx < LastRootIndex) {
          MustScalarize.insert(*it);
          DEBUG(dbgs()<<"SLP: Adding to MustScalarize "
                "because of an unsafe out of tree usage.\n");
          break;
        }


        DEBUG(dbgs()<<"SLP: Adding to MustExtract "
              "because of a safe out of tree usage.\n");
        MustExtract.insert(*it);
        continue;
      }
      if (Lane == -1) Lane = LaneMap[*I];
      if (Lane != LaneMap[*I]) {
        MustScalarize.insert(*it);
        DEBUG(dbgs()<<"SLP: Adding " << **it <<
              " to MustScalarize because multiple lane use it: "
              << Lane << " and " << LaneMap[*I] << ".\n");
        break;
      }
    }
  }

  // Now calculate the cost of vectorizing the tree.
  return getTreeCost_rec(VL, 0);
}

void BoUpSLP::getTreeUses_rec(ArrayRef<Value *> VL, unsigned Depth) {
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

  for (int i = 0, e = VL.size(); i < e; ++i) {
    // Check that the instruction is only used within
    // one lane.
    if (LaneMap.count(VL[i]) && LaneMap[VL[i]] != i) return;
    // Make this instruction as 'seen' and remember the lane.
    LaneMap[VL[i]] = i;
  }

  // Mark instructions with multiple users.
  for (unsigned i = 0, e = VL.size(); i < e; ++i) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    // Remember to check if all of the users of this instr are vectorized
    // within our tree. At depth zero we have no local users, only external
    // users that we don't care about.
    if (Depth && I && I->getNumUses() > 1) {
      DEBUG(dbgs()<<"SLP: Adding to MultiUserVals "
            "because it has multiple users:" << *I << " \n");
      MultiUserVals.insert(I);
    }
  }

  switch (Opcode) {
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

        getTreeUses_rec(Operands, Depth+1);
      }
      return;
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

int BoUpSLP::getTreeCost_rec(ArrayRef<Value *> VL, unsigned Depth) {
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
    int MaxIdx = getLastIndex(VL, VL.size());
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

  // Calculate the extract cost.
  unsigned ExternalUserExtractCost = 0;
  for (unsigned i = 0, e = VL.size(); i < e; ++i)
    if (MustExtract.count(VL[i]))
      ExternalUserExtractCost +=
        TTI->getVectorInstrCost(Instruction::ExtractElement, VecTy, i);

  switch (Opcode) {
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
    int Cost = ExternalUserExtractCost;
    ValueList Operands;
    Type *SrcTy = VL0->getOperand(0)->getType();
    // Prepare the operand vector.
    for (unsigned j = 0; j < VL.size(); ++j) {
      Operands.push_back(cast<Instruction>(VL[j])->getOperand(0));
      // Check that the casted type is the same for all users.
      if (cast<Instruction>(VL[j])->getOperand(0)->getType() != SrcTy)
        return getScalarizationCost(VecTy);
    }

    Cost += getTreeCost_rec(Operands, Depth+1);
    if (Cost >= max_cost) return max_cost;

    // Calculate the cost of this instruction.
    int ScalarCost = VL.size() * TTI->getCastInstrCost(VL0->getOpcode(),
                                                       VL0->getType(), SrcTy);

    VectorType *SrcVecTy = VectorType::get(SrcTy, VL.size());
    int VecCost = TTI->getCastInstrCost(VL0->getOpcode(), VecTy, SrcVecTy);
    Cost += (VecCost - ScalarCost);
    return Cost;
  }
  case Instruction::FCmp:
  case Instruction::ICmp: {
    // Check that all of the compares have the same predicate.
    CmpInst::Predicate P0 = dyn_cast<CmpInst>(VL0)->getPredicate();
    for (unsigned i = 1, e = VL.size(); i < e; ++i) {
      CmpInst *Cmp = cast<CmpInst>(VL[i]);
      if (Cmp->getPredicate() != P0)
        return getScalarizationCost(VecTy);
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
    int Cost = ExternalUserExtractCost;
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
    return VecLdCost - ScalarLdCost + ExternalUserExtractCost;
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
    return TotalCost + ExternalUserExtractCost;
  }
  default:
    // Unable to vectorize unknown instructions.
    return getScalarizationCost(VecTy);
  }
}

int BoUpSLP::getLastIndex(ArrayRef<Value *> VL, unsigned VF) {
  int MaxIdx = InstrIdx[BB->getFirstNonPHI()];
  for (unsigned i = 0; i < VF; ++i )
    MaxIdx = std::max(MaxIdx, InstrIdx[VL[i]]);
  return MaxIdx;
}

int BoUpSLP::getFirstUserIndex(ArrayRef<Value *> VL, unsigned VF) {
  // Find the first user of the values.
  int FirstUser = InstrVec.size();
  for (unsigned i = 0; i < VF; ++i) {
    for (Value::use_iterator U = VL[i]->use_begin(), UE = VL[i]->use_end();
         U != UE; ++U) {
      Instruction *Instr = dyn_cast<Instruction>(*U);
      if (!Instr || Instr->getParent() != BB)
        continue;

      FirstUser = std::min(FirstUser, InstrIdx[Instr]);
    }
  }
  return FirstUser;
}

int BoUpSLP::getLastIndex(Instruction *I, Instruction *J) {
  assert(I->getParent() == BB && "Invalid parent for instruction I");
  assert(J->getParent() == BB && "Invalid parent for instruction J");
  return std::max(InstrIdx[I],InstrIdx[J]);
}

Instruction *BoUpSLP::getInsertionPoint(unsigned Index) {
  return InstrVec[Index + 1];
}

Value *BoUpSLP::Scalarize(ArrayRef<Value *> VL, VectorType *Ty) {
  Value *Vec = UndefValue::get(Ty);
  for (unsigned i=0; i < Ty->getNumElements(); ++i) {
    // Generate the 'InsertElement' instruction.
    Vec = Builder.CreateInsertElement(Vec, VL[i], Builder.getInt32(i));
    // Remember that this instruction is used as part of a 'gather' sequence.
    // The caller of the bottom-up slp vectorizer can try to hoist the sequence
    // if the users are outside of the basic block.
    GatherInstructions.push_back(Vec);
  }

  for (unsigned i = 0; i < Ty->getNumElements(); ++i)
    VectorizedValues[VL[i]] = Vec;

  return Vec;
}

Value *BoUpSLP::vectorizeTree(ArrayRef<Value *> VL, int VF) {
  Value *V = vectorizeTree_rec(VL, VF);

  int LastInstrIdx = getLastIndex(VL, VL.size());
  for (SetVector<Value*>::iterator it = MustExtract.begin(),
       e = MustExtract.end(); it != e; ++it) {
    Instruction *I = cast<Instruction>(*it);

    // This is a scalarized value, so we can use the original value.
    // No need to extract from the vector.
    if (!LaneMap.count(I))
      continue;

    Value *Vec = VectorizedValues[I];
    // We decided not to vectorize I because one of its users was not
    // vectorizerd. This is okay.
    if (!Vec)
      continue;

    Value *Idx = Builder.getInt32(LaneMap[I]);
    Value *Extract = Builder.CreateExtractElement(Vec, Idx);
    bool Replaced = false;
    for (Value::use_iterator U = I->use_begin(), UE = I->use_end(); U != UE;
         ++U) {
      Instruction *UI = cast<Instruction>(*U);
      if (UI->getParent() != I->getParent() || InstrIdx[UI] > LastInstrIdx)
        UI->replaceUsesOfWith(I ,Extract);
      Replaced = true;
    }
    assert(Replaced && "Must replace at least one outside user");
    (void)Replaced;
  }

  // We moved some instructions around. We have to number them again
  // before we can do any analysis.
  numberInstructions();
  MustScalarize.clear();
  MustExtract.clear();
  VectorizedValues.clear();
  return V;
}

Value *BoUpSLP::vectorizeTree_rec(ArrayRef<Value *> VL, int VF) {
  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, VF);

  // Check if all of the operands are constants or identical.
  bool AllConst = true;
  bool AllSameScalar = true;
  for (unsigned i = 0, e = VF; i < e; ++i) {
    AllConst &= isa<Constant>(VL[i]);
    AllSameScalar &= (VL[0] == VL[i]);
    // The instruction must be in the same BB, and it must be vectorizable.
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    if (MustScalarize.count(VL[i]) || (I && I->getParent() != BB))
      return Scalarize(VL, VecTy);
  }

  // Check that this is a simple vector constant.
  if (AllConst || AllSameScalar)
    return Scalarize(VL, VecTy);

  // Scalarize unknown structures.
  Instruction *VL0 = dyn_cast<Instruction>(VL[0]);
  if (!VL0)
    return Scalarize(VL, VecTy);

  if (VectorizedValues.count(VL0)) {
    Value * Vec = VectorizedValues[VL0];
    for (int i = 0; i < VF; ++i)
      VectorizedValues[VL[i]] = Vec;
    return Vec;
  }

  unsigned Opcode = VL0->getOpcode();
  for (unsigned i = 0, e = VF; i < e; ++i) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    // If not all of the instructions are identical then we have to scalarize.
    if (!I || Opcode != I->getOpcode())
      return Scalarize(VL, VecTy);
  }

  switch (Opcode) {
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
    for (int i = 0; i < VF; ++i)
      INVL.push_back(cast<Instruction>(VL[i])->getOperand(0));
    Value *InVec = vectorizeTree_rec(INVL, VF);
    CastInst *CI = dyn_cast<CastInst>(VL0);
    Value *V = Builder.CreateCast(CI->getOpcode(), InVec, VecTy);

    for (int i = 0; i < VF; ++i)
      VectorizedValues[VL[i]] = V;

    return V;
  }
  case Instruction::FCmp:
  case Instruction::ICmp: {
    // Check that all of the compares have the same predicate.
    CmpInst::Predicate P0 = dyn_cast<CmpInst>(VL0)->getPredicate();
    for (unsigned i = 1, e = VF; i < e; ++i) {
      CmpInst *Cmp = cast<CmpInst>(VL[i]);
      if (Cmp->getPredicate() != P0)
        return Scalarize(VL, VecTy);
    }

    ValueList LHSV, RHSV;
    for (int i = 0; i < VF; ++i) {
      LHSV.push_back(cast<Instruction>(VL[i])->getOperand(0));
      RHSV.push_back(cast<Instruction>(VL[i])->getOperand(1));
    }

    Value *L = vectorizeTree_rec(LHSV, VF);
    Value *R = vectorizeTree_rec(RHSV, VF);
    Value *V;
    if (VL0->getOpcode() == Instruction::FCmp)
      V = Builder.CreateFCmp(P0, L, R);
    else
      V = Builder.CreateICmp(P0, L, R);

    for (int i = 0; i < VF; ++i)
      VectorizedValues[VL[i]] = V;

    return V;

  }
  case Instruction::Select: {
    ValueList TrueVec, FalseVec, CondVec;
    for (int i = 0; i < VF; ++i) {
      CondVec.push_back(cast<Instruction>(VL[i])->getOperand(0));
      TrueVec.push_back(cast<Instruction>(VL[i])->getOperand(1));
      FalseVec.push_back(cast<Instruction>(VL[i])->getOperand(2));
    }

    Value *True = vectorizeTree_rec(TrueVec, VF);
    Value *False = vectorizeTree_rec(FalseVec, VF);
    Value *Cond = vectorizeTree_rec(CondVec, VF);
    Value *V = Builder.CreateSelect(Cond, True, False);

    for (int i = 0; i < VF; ++i)
      VectorizedValues[VL[i]] = V;

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
    for (int i = 0; i < VF; ++i) {
      LHSVL.push_back(cast<Instruction>(VL[i])->getOperand(0));
      RHSVL.push_back(cast<Instruction>(VL[i])->getOperand(1));
    }

    Value *LHS = vectorizeTree_rec(LHSVL, VF);
    Value *RHS = vectorizeTree_rec(RHSVL, VF);
    BinaryOperator *BinOp = cast<BinaryOperator>(VL0);
    Value *V = Builder.CreateBinOp(BinOp->getOpcode(), LHS,RHS);

    for (int i = 0; i < VF; ++i)
      VectorizedValues[VL[i]] = V;

    return V;
  }
  case Instruction::Load: {
    LoadInst *LI = cast<LoadInst>(VL0);
    unsigned Alignment = LI->getAlignment();

    // Check if all of the loads are consecutive.
    for (unsigned i = 1, e = VF; i < e; ++i)
      if (!isConsecutiveAccess(VL[i-1], VL[i]))
        return Scalarize(VL, VecTy);

    // Loads are inserted at the head of the tree because we don't want to sink
    // them all the way down past store instructions.
    Instruction *Loc = getInsertionPoint(getLastIndex(VL, VL.size()));
    IRBuilder<> LoadBuilder(Loc);
    Value *VecPtr = LoadBuilder.CreateBitCast(LI->getPointerOperand(),
                                              VecTy->getPointerTo());
    LI = LoadBuilder.CreateLoad(VecPtr);
    LI->setAlignment(Alignment);

    for (int i = 0; i < VF; ++i)
      VectorizedValues[VL[i]] = LI;

    return LI;
  }
  case Instruction::Store: {
    StoreInst *SI = cast<StoreInst>(VL0);
    unsigned Alignment = SI->getAlignment();

    ValueList ValueOp;
    for (int i = 0; i < VF; ++i)
      ValueOp.push_back(cast<StoreInst>(VL[i])->getValueOperand());

    Value *VecValue = vectorizeTree_rec(ValueOp, VF);
    Value *VecPtr = Builder.CreateBitCast(SI->getPointerOperand(),
                                          VecTy->getPointerTo());
    Builder.CreateStore(VecValue, VecPtr)->setAlignment(Alignment);

    for (int i = 0; i < VF; ++i)
      cast<Instruction>(VL[i])->eraseFromParent();
    return 0;
  }
  default:
    return Scalarize(VL, VecTy);
  }
}

} // end of namespace
