//===- VecUtils.cpp - Vectorization Utilities -----------------------------===//
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

#ifndef LLVM_TRANSFORMS_VECTORIZE_AOSVECTORIZER_H
#define  LLVM_TRANSFORMS_VECTORIZE_AOSVECTORIZER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include <vector>

using namespace llvm;

namespace llvm {

class BasicBlock; class Instruction; class Type;
class VectorType; class StoreInst; class Value;
class ScalarEvolution; class DataLayout;
class TargetTransformInfo; class AliasAnalysis;

/// Bottom Up SLP vectorization utility class.
struct BoUpSLP  {
  typedef SmallVector<Value*, 8> ValueList;
  typedef SmallPtrSet<Value*, 16> ValueSet;
  typedef SmallVector<StoreInst*, 8> StoreList;
  static const int max_cost = 1<<20;

  // \brief C'tor.
  BoUpSLP(BasicBlock *Bb, ScalarEvolution *Se, DataLayout *Dl,
         TargetTransformInfo *Tti, AliasAnalysis *Aa);

  /// \returns true if the memory operations A and B are consecutive.
  bool isConsecutiveAccess(Value *A, Value *B);

  /// \brief Vectorize the tree that starts with the elements in \p VL.
  /// \returns the vectorized value.
  Value *vectorizeTree(ValueList &VL, int VF);

  /// \returns the vectorization cost of the subtree that starts at \p VL.
  /// A negative number means that this is profitable.
  int getTreeRollCost(ValueList &VL, unsigned Depth);

  /// \brief Take the pointer operand from the Load/Store instruction.
  /// \returns NULL if this is not a valid Load/Store instruction.
  static Value *getPointerOperand(Value *I);

  /// \brief Take the address space operand from the Load/Store instruction.
  /// \returns -1 if this is not a valid Load/Store instruction.
  static unsigned getAddressSpaceOperand(Value *I);

  /// \brief Attempts to order and vectorize a sequence of stores. This
  /// function does a quadratic scan of the given stores.
  /// \returns true if the basic block was modified.
  bool vectorizeStores(StoreList &Stores, int costThreshold);

  /// \brief Number all of the instructions in the block.
  void numberInstructions();

private:
  /// \returns the scalarization cost for this type. Scalarization in this
  /// context means the creation of vectors from a group of scalars.
  int getScalarizationCost(Type *Ty);

  /// \returns the AA location that is being access by the instruction.
  AliasAnalysis::Location getLocation(Instruction *I);

  /// \brief Checks if it is possible to sink an instruction from
  /// \p Src to \p Dst.
  /// \returns the pointer to the barrier instruction if we can't sink.
  Value *isUnsafeToSink(Instruction *Src, Instruction *Dst);

  /// \returns the instruction that appears last in the BB from \p VL.
  /// Only consider the first \p VF elements.
  Instruction *GetLastInstr(ValueList &VL, unsigned VF);

  /// \returns a vector from a collection of scalars in \p VL.
  Value *Scalarize(ValueList &VL, VectorType *Ty);

  // Maps instructions to numbers and back.
  SmallDenseMap<Value*, int> InstrIdx;
  std::vector<Instruction*> InstrVec;
  // A list of instructions to ignore while sinking
  // memory instructions.
  SmallSet<Value*, 8> MemBarrierIgnoreList;
  // Analysis and block reference.
  BasicBlock *BB;
  ScalarEvolution *SE;
  DataLayout *DL;
  TargetTransformInfo *TTI;
  AliasAnalysis *AA;
};

} // end of namespace
# endif  //LLVM_TRANSFORMS_VECTORIZE_AOSVECTORIZER_H

