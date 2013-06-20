//===- VecUtils.h - Vectorization Utilities -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of classes and functions manipulate vectors and chains of
// vectors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VECUTILS_H
#define LLVM_TRANSFORMS_VECTORIZE_VECUTILS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/IRBuilder.h"
#include <vector>

namespace llvm {

class BasicBlock;
class Instruction;
class Type;
class VectorType;
class StoreInst;
class Value;
class ScalarEvolution;
class DataLayout;
class TargetTransformInfo;
class AliasAnalysis;
class Loop;

/// Bottom Up SLP vectorization utility class.
struct BoUpSLP {
  typedef SmallVector<Value *, 8> ValueList;
  typedef SmallVector<Instruction *, 16> InstrList;
  typedef SmallPtrSet<Value *, 16> ValueSet;
  typedef SmallVector<StoreInst *, 8> StoreList;
  static const int max_cost = 1 << 20;

  // \brief C'tor.
  BoUpSLP(BasicBlock *Bb, ScalarEvolution *Se, DataLayout *Dl,
          TargetTransformInfo *Tti, AliasAnalysis *Aa, Loop *Lp);

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
  Value *vectorizeTree(ArrayRef<Value *> VL, int VF);

  /// \returns the vectorization cost of the subtree that starts at \p VL.
  /// A negative number means that this is profitable.
  int getTreeCost(ArrayRef<Value *> VL);

  /// \returns the scalarization cost for this list of values. Assuming that
  /// this subtree gets vectorized, we may need to extract the values from the
  /// roots. This method calculates the cost of extracting the values.
  int getScalarizationCost(ArrayRef<Value *> VL);

  /// \brief Attempts to order and vectorize a sequence of stores. This
  /// function does a quadratic scan of the given stores.
  /// \returns true if the basic block was modified.
  bool vectorizeStores(ArrayRef<StoreInst *> Stores, int costThreshold);

  /// \brief Vectorize a group of scalars into a vector tree.
  /// \returns the vectorized value.
  Value *vectorizeArith(ArrayRef<Value *> Operands);

  /// \returns the list of new instructions that were added in order to collect
  /// scalars into vectors. This list can be used to further optimize the gather
  /// sequences.
  InstrList &getGatherSeqInstructions() { return GatherInstructions; }

private:
  /// \brief This method contains the recursive part of getTreeCost.
  int getTreeCost_rec(ArrayRef<Value *> VL, unsigned Depth);

  /// \brief This recursive method looks for vectorization hazards such as
  /// values that are used by multiple users and checks that values are used
  /// by only one vector lane. It updates the variables LaneMap, MultiUserVals.
  void getTreeUses_rec(ArrayRef<Value *> VL, unsigned Depth);

  /// \brief This method contains the recursive part of vectorizeTree.
  Value *vectorizeTree_rec(ArrayRef<Value *> VL, int VF);

  /// \brief Number all of the instructions in the block.
  void numberInstructions();

  ///  \brief Vectorize a sorted sequence of stores.
  bool vectorizeStoreChain(ArrayRef<Value *> Chain, int CostThreshold);

  /// \returns the scalarization cost for this type. Scalarization in this
  /// context means the creation of vectors from a group of scalars.
  int getScalarizationCost(Type *Ty);

  /// \returns the AA location that is being access by the instruction.
  AliasAnalysis::Location getLocation(Instruction *I);

  /// \brief Checks if it is possible to sink an instruction from
  /// \p Src to \p Dst.
  /// \returns the pointer to the barrier instruction if we can't sink.
  Value *isUnsafeToSink(Instruction *Src, Instruction *Dst);

  /// \returns the index of the last instrucion in the BB from \p VL.
  /// Only consider the first \p VF elements.
  int getLastIndex(ArrayRef<Value *> VL, unsigned VF);

  /// \returns the index of the first User of \p VL.
  /// Only consider the first \p VF elements.
  int getFirstUserIndex(ArrayRef<Value *> VL, unsigned VF);

  /// \returns the instruction \p I or \p J that appears last in the BB .
  int getLastIndex(Instruction *I, Instruction *J);

  /// \returns the insertion point for \p Index.
  Instruction *getInsertionPoint(unsigned Index);

  /// \returns a vector from a collection of scalars in \p VL.
  Value *Scalarize(ArrayRef<Value *> VL, VectorType *Ty);

private:
  /// Maps instructions to numbers and back.
  SmallDenseMap<Value *, int> InstrIdx;
  /// Maps integers to Instructions.
  std::vector<Instruction *> InstrVec;

  // -- containers that are used during getTreeCost -- //

  /// Contains values that must be scalarized because they are used
  /// by multiple lanes, or by users outside the tree.
  /// NOTICE: The vectorization methods also use this set.
  ValueSet MustScalarize;

  /// Contains values that have users outside of the vectorized graph.
  /// We need to generate extract instructions for these values.
  /// NOTICE: The vectorization methods also use this set.
  SetVector<Value *> MustExtract;

  /// Contains a list of values that are used outside the current tree. This
  /// set must be reset between runs.
  SetVector<Value *> MultiUserVals;
  /// Maps values in the tree to the vector lanes that uses them. This map must
  /// be reset between runs of getCost.
  std::map<Value *, int> LaneMap;
  /// A list of instructions to ignore while sinking
  /// memory instructions. This map must be reset between runs of getCost.
  ValueSet MemBarrierIgnoreList;

  // -- Containers that are used during vectorizeTree -- //

  /// Maps between the first scalar to the vector. This map must be reset
  /// between runs.
  DenseMap<Value *, Value *> VectorizedValues;

  // -- Containers that are used after vectorization by the caller -- //

  /// A list of instructions that are used when gathering scalars into vectors.
  /// In many cases these instructions can be hoisted outside of the BB.
  /// Iterating over this list is faster than calling LICM.
  /// Notice: We insert NULL ptrs to separate between the different gather
  /// sequences.
  InstrList GatherInstructions;

  /// Instruction builder to construct the vectorized tree.
  IRBuilder<> Builder;

  // Analysis and block reference.
  BasicBlock *BB;
  ScalarEvolution *SE;
  DataLayout *DL;
  TargetTransformInfo *TTI;
  AliasAnalysis *AA;
  Loop *L;
};

} // end of namespace

#endif // LLVM_TRANSFORMS_VECTORIZE_VECUTILS_H
