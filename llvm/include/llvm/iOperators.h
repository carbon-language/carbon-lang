//===-- llvm/iOperators.h - Binary Operator node definitions ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the Binary Operator classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IOPERATORS_H
#define LLVM_IOPERATORS_H

#include "llvm/InstrTypes.h"

namespace llvm {

/// SetCondInst class - Represent a setCC operator, where CC is eq, ne, lt, gt,
/// le, or ge.
///
class SetCondInst : public BinaryOperator {
  BinaryOps OpType;
public:
  SetCondInst(BinaryOps Opcode, Value *LHS, Value *RHS,
	      const std::string &Name = "", Instruction *InsertBefore = 0);
  SetCondInst(BinaryOps Opcode, Value *LHS, Value *RHS,
	      const std::string &Name, BasicBlock *InsertAtEnd);

  /// getInverseCondition - Return the inverse of the current condition opcode.
  /// For example seteq -> setne, setgt -> setle, setlt -> setge, etc...
  ///
  BinaryOps getInverseCondition() const {
    return getInverseCondition(getOpcode());
  }

  /// getInverseCondition - Static version that you can use without an
  /// instruction available.
  ///
  static BinaryOps getInverseCondition(BinaryOps Opcode);

  /// getSwappedCondition - Return the condition opcode that would be the result
  /// of exchanging the two operands of the setcc instruction without changing
  /// the result produced.  Thus, seteq->seteq, setle->setge, setlt->setgt, etc.
  ///
  BinaryOps getSwappedCondition() const {
    return getSwappedCondition(getOpcode());
  }

  /// getSwappedCondition - Static version that you can use without an
  /// instruction available.
  ///
  static BinaryOps getSwappedCondition(BinaryOps Opcode);


  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SetCondInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == SetEQ || I->getOpcode() == SetNE ||
           I->getOpcode() == SetLE || I->getOpcode() == SetGE ||
           I->getOpcode() == SetLT || I->getOpcode() == SetGT;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

} // End llvm namespace

#endif
