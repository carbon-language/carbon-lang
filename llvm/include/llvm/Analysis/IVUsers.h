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
#include "llvm/Support/ValueHandle.h"

namespace llvm {

class DominatorTree;
class Instruction;
class Value;
class IVUsers;
class ScalarEvolution;
class SCEV;

/// IVStrideUse - Keep track of one use of a strided induction variable.
/// The Expr member keeps track of the expression, User is the actual user
/// instruction of the operand, and 'OperandValToReplace' is the operand of
/// the User that is the use.
class IVStrideUse : public CallbackVH, public ilist_node<IVStrideUse> {
public:
  IVStrideUse(IVUsers *P, const SCEV *S, const SCEV *Off,
              Instruction* U, Value *O)
    : CallbackVH(U), Parent(P), Stride(S), Offset(Off),
      OperandValToReplace(O), IsUseOfPostIncrementedValue(false) {
  }

  /// getUser - Return the user instruction for this use.
  Instruction *getUser() const {
    return cast<Instruction>(getValPtr());
  }

  /// setUser - Assign a new user instruction for this use.
  void setUser(Instruction *NewUser) {
    setValPtr(NewUser);
  }

  /// getParent - Return a pointer to the IVUsers that owns
  /// this IVStrideUse.
  IVUsers *getParent() const { return Parent; }

  /// getStride - Return the expression for the stride for the use.
  const SCEV *getStride() const { return Stride; }

  /// setStride - Assign a new stride to this use.
  void setStride(const SCEV *Val) {
    Stride = Val;
  }

  /// getOffset - Return the offset to add to a theoretical induction
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
  /// Parent - a pointer to the IVUsers that owns this IVStrideUse.
  IVUsers *Parent;

  /// Stride - The stride for this use.
  const SCEV *Stride;

  /// Offset - The offset to add to the base induction expression.
  const SCEV *Offset;

  /// OperandValToReplace - The Value of the operand in the user instruction
  /// that this IVStrideUse is representing.
  WeakVH OperandValToReplace;

  /// IsUseOfPostIncrementedValue - True if this should use the
  /// post-incremented version of this IV, not the preincremented version.
  bool IsUseOfPostIncrementedValue;

  /// Deleted - Implementation of CallbackVH virtual function to
  /// receive notification when the User is deleted.
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

class IVUsers : public LoopPass {
  friend class IVStrideUse;
  Loop *L;
  LoopInfo *LI;
  DominatorTree *DT;
  ScalarEvolution *SE;
  SmallPtrSet<Instruction*,16> Processed;

  /// IVUses - A list of all tracked IV uses of induction variable expressions
  /// we are interested in.
  ilist<IVStrideUse> IVUses;

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

  IVStrideUse &AddUser(const SCEV *Stride, const SCEV *Offset,
                       Instruction *User, Value *Operand);

  /// getReplacementExpr - Return a SCEV expression which computes the
  /// value of the OperandValToReplace of the given IVStrideUse.
  const SCEV *getReplacementExpr(const IVStrideUse &U) const;

  /// getCanonicalExpr - Return a SCEV expression which computes the
  /// value of the SCEV of the given IVStrideUse, ignoring the 
  /// isUseOfPostIncrementedValue flag.
  const SCEV *getCanonicalExpr(const IVStrideUse &U) const;

  typedef ilist<IVStrideUse>::iterator iterator;
  typedef ilist<IVStrideUse>::const_iterator const_iterator;
  iterator begin() { return IVUses.begin(); }
  iterator end()   { return IVUses.end(); }
  const_iterator begin() const { return IVUses.begin(); }
  const_iterator end() const   { return IVUses.end(); }
  bool empty() const { return IVUses.empty(); }

  void print(raw_ostream &OS, const Module* = 0) const;

  /// dump - This method is used for debugging.
  void dump() const;
};

Pass *createIVUsersPass();

}

#endif
