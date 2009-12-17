//===- llvm/Analysis/IVUsers.h - Induction Variable Users -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements bookkeeping for "interesting" users of expressions
// computed from induction variables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_IVUSERS_H
#define LLVM_ANALYSIS_IVUSERS_H

#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/ADT/SmallVector.h"
#include <map>

namespace llvm {

class DominatorTree;
class Instruction;
class Value;
struct IVUsersOfOneStride;

/// IVStrideUse - Keep track of one use of a strided induction variable, where
/// the stride is stored externally.  The Offset member keeps track of the
/// offset from the IV, User is the actual user of the operand, and
/// 'OperandValToReplace' is the operand of the User that is the use.
class IVStrideUse : public CallbackVH, public ilist_node<IVStrideUse> {
public:
  IVStrideUse(IVUsersOfOneStride *parent,
              const SCEV *offset,
              Instruction* U, Value *O)
    : CallbackVH(U), Parent(parent), Offset(offset),
      OperandValToReplace(O),
      IsUseOfPostIncrementedValue(false) {
  }

  /// getUser - Return the user instruction for this use.
  Instruction *getUser() const {
    return cast<Instruction>(getValPtr());
  }

  /// setUser - Assign a new user instruction for this use.
  void setUser(Instruction *NewUser) {
    setValPtr(NewUser);
  }

  /// getParent - Return a pointer to the IVUsersOfOneStride that owns
  /// this IVStrideUse.
  IVUsersOfOneStride *getParent() const { return Parent; }

  /// getOffset - Return the offset to add to a theoeretical induction
  /// variable that starts at zero and counts up by the stride to compute
  /// the value for the use. This always has the same type as the stride.
  const SCEV *getOffset() const { return Offset; }

  /// setOffset - Assign a new offset to this use.
  void setOffset(const SCEV *Val) {
    Offset = Val;
  }

  /// getOperandValToReplace - Return the Value of the operand in the user
  /// instruction that this IVStrideUse is representing.
  Value *getOperandValToReplace() const {
    return OperandValToReplace;
  }

  /// setOperandValToReplace - Assign a new Value as the operand value
  /// to replace.
  void setOperandValToReplace(Value *Op) {
    OperandValToReplace = Op;
  }

  /// isUseOfPostIncrementedValue - True if this should use the
  /// post-incremented version of this IV, not the preincremented version.
  /// This can only be set in special cases, such as the terminating setcc
  /// instruction for a loop or uses dominated by the loop.
  bool isUseOfPostIncrementedValue() const {
    return IsUseOfPostIncrementedValue;
  }

  /// setIsUseOfPostIncrmentedValue - set the flag that indicates whether
  /// this is a post-increment use.
  void setIsUseOfPostIncrementedValue(bool Val) {
    IsUseOfPostIncrementedValue = Val;
  }

private:
  /// Parent - a pointer to the IVUsersOfOneStride that owns this IVStrideUse.
  IVUsersOfOneStride *Parent;

  /// Offset - The offset to add to the base induction expression.
  const SCEV *Offset;

  /// OperandValToReplace - The Value of the operand in the user instruction
  /// that this IVStrideUse is representing.
  WeakVH OperandValToReplace;

  /// IsUseOfPostIncrementedValue - True if this should use the
  /// post-incremented version of this IV, not the preincremented version.
  bool IsUseOfPostIncrementedValue;

  /// Deleted - Implementation of CallbackVH virtual function to
  /// recieve notification when the User is deleted.
  virtual void deleted();
};

template<> struct ilist_traits<IVStrideUse>
  : public ilist_default_traits<IVStrideUse> {
  // createSentinel is used to get hold of a node that marks the end of
  // the list...
  // The sentinel is relative to this instance, so we use a non-static
  // method.
  IVStrideUse *createSentinel() const {
    // since i(p)lists always publicly derive from the corresponding
    // traits, placing a data member in this class will augment i(p)list.
    // But since the NodeTy is expected to publicly derive from
    // ilist_node<NodeTy>, there is a legal viable downcast from it
    // to NodeTy. We use this trick to superpose i(p)list with a "ghostly"
    // NodeTy, which becomes the sentinel. Dereferencing the sentinel is
    // forbidden (save the ilist_node<NodeTy>) so no one will ever notice
    // the superposition.
    return static_cast<IVStrideUse*>(&Sentinel);
  }
  static void destroySentinel(IVStrideUse*) {}

  IVStrideUse *provideInitialHead() const { return createSentinel(); }
  IVStrideUse *ensureHead(IVStrideUse*) const { return createSentinel(); }
  static void noteHead(IVStrideUse*, IVStrideUse*) {}

private:
  mutable ilist_node<IVStrideUse> Sentinel;
};

/// IVUsersOfOneStride - This structure keeps track of all instructions that
/// have an operand that is based on the trip count multiplied by some stride.
struct IVUsersOfOneStride : public ilist_node<IVUsersOfOneStride> {
private:
  IVUsersOfOneStride(const IVUsersOfOneStride &I); // do not implement
  void operator=(const IVUsersOfOneStride &I);     // do not implement

public:
  IVUsersOfOneStride() : Stride(0) {}

  explicit IVUsersOfOneStride(const SCEV *stride) : Stride(stride) {}

  /// Stride - The stride for all the contained IVStrideUses. This is
  /// a constant for affine strides.
  const SCEV *Stride;

  /// Users - Keep track of all of the users of this stride as well as the
  /// initial value and the operand that uses the IV.
  ilist<IVStrideUse> Users;

  void addUser(const SCEV *Offset, Instruction *User, Value *Operand) {
    Users.push_back(new IVStrideUse(this, Offset, User, Operand));
  }

  void removeUser(IVStrideUse *User) {
    Users.erase(User);
  }
};

class IVUsers : public LoopPass {
  friend class IVStrideUserVH;
  Loop *L;
  LoopInfo *LI;
  DominatorTree *DT;
  ScalarEvolution *SE;
  SmallPtrSet<Instruction*,16> Processed;

public:
  /// IVUses - A list of all tracked IV uses of induction variable expressions
  /// we are interested in.
  ilist<IVUsersOfOneStride> IVUses;

  /// IVUsesByStride - A mapping from the strides in StrideOrder to the
  /// uses in IVUses.
  std::map<const SCEV *, IVUsersOfOneStride*> IVUsesByStride;

  /// StrideOrder - An ordering of the keys in IVUsesByStride that is stable:
  /// We use this to iterate over the IVUsesByStride collection without being
  /// dependent on random ordering of pointers in the process.
  SmallVector<const SCEV *, 16> StrideOrder;

private:
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

  virtual bool runOnLoop(Loop *L, LPPassManager &LPM);

  virtual void releaseMemory();

public:
  static char ID; // Pass ID, replacement for typeid
  IVUsers();

  /// AddUsersIfInteresting - Inspect the specified Instruction.  If it is a
  /// reducible SCEV, recursively add its users to the IVUsesByStride set and
  /// return true.  Otherwise, return false.
  bool AddUsersIfInteresting(Instruction *I);

  void AddUser(const SCEV *Stride, const SCEV *Offset,
               Instruction *User, Value *Operand);

  /// getReplacementExpr - Return a SCEV expression which computes the
  /// value of the OperandValToReplace of the given IVStrideUse.
  const SCEV *getReplacementExpr(const IVStrideUse &U) const;

  void print(raw_ostream &OS, const Module* = 0) const;

  /// dump - This method is used for debugging.
  void dump() const;
};

Pass *createIVUsersPass();

}

#endif
