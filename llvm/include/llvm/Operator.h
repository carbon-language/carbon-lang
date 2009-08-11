//===-- llvm/Operator.h - Operator utility subclass -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines various classes for working with Instructions and
// ConstantExprs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPERATOR_H
#define LLVM_OPERATOR_H

#include "llvm/Instruction.h"
#include "llvm/Constants.h"

namespace llvm {

class GetElementPtrInst;

/// Operator - This is a utility class that provides an abstraction for the
/// common functionality between Instructions and ConstantExprs.
///
class Operator : public User {
private:
  // Do not implement any of these. The Operator class is intended to be used
  // as a utility, and is never itself instantiated.
  void *operator new(size_t, unsigned);
  void *operator new(size_t s);
  Operator();
  ~Operator();

public:
  /// getOpcode - Return the opcode for this Instruction or ConstantExpr.
  ///
  unsigned getOpcode() const {
    if (const Instruction *I = dyn_cast<Instruction>(this))
      return I->getOpcode();
    return cast<ConstantExpr>(this)->getOpcode();
  }

  /// getOpcode - If V is an Instruction or ConstantExpr, return its
  /// opcode. Otherwise return UserOp1.
  ///
  static unsigned getOpcode(const Value *V) {
    if (const Instruction *I = dyn_cast<Instruction>(V))
      return I->getOpcode();
    if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
      return CE->getOpcode();
    return Instruction::UserOp1;
  }

  static inline bool classof(const Operator *) { return true; }
  static inline bool classof(const Instruction *I) { return true; }
  static inline bool classof(const ConstantExpr *I) { return true; }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) || isa<ConstantExpr>(V);
  }
};

/// OverflowingBinaryOperator - Utility class for integer arithmetic operators
/// which may exhibit overflow - Add, Sub, and Mul. It does not include SDiv,
/// despite that operator having the potential for overflow.
///
class OverflowingBinaryOperator : public Operator {
public:
  /// hasNoUnsignedOverflow - Test whether this operation is known to never
  /// undergo unsigned overflow.
  bool hasNoUnsignedOverflow() const {
    return SubclassOptionalData & (1 << 0);
  }
  void setHasNoUnsignedOverflow(bool B) {
    SubclassOptionalData = (SubclassOptionalData & ~(1 << 0)) | (B << 0);
  }

  /// hasNoSignedOverflow - Test whether this operation is known to never
  /// undergo signed overflow.
  bool hasNoSignedOverflow() const {
    return SubclassOptionalData & (1 << 1);
  }
  void setHasNoSignedOverflow(bool B) {
    SubclassOptionalData = (SubclassOptionalData & ~(1 << 1)) | (B << 1);
  }

  static inline bool classof(const OverflowingBinaryOperator *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Add ||
           I->getOpcode() == Instruction::Sub ||
           I->getOpcode() == Instruction::Mul;
  }
  static inline bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Instruction::Add ||
           CE->getOpcode() == Instruction::Sub ||
           CE->getOpcode() == Instruction::Mul;
  }
  static inline bool classof(const Value *V) {
    return (isa<Instruction>(V) && classof(cast<Instruction>(V))) ||
           (isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V)));
  }
};

/// AddOperator - Utility class for integer addition operators.
///
class AddOperator : public OverflowingBinaryOperator {
public:
  static inline bool classof(const AddOperator *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Add;
  }
  static inline bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Instruction::Add;
  }
  static inline bool classof(const Value *V) {
    return (isa<Instruction>(V) && classof(cast<Instruction>(V))) ||
           (isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V)));
  }
};

/// SubOperator - Utility class for integer subtraction operators.
///
class SubOperator : public OverflowingBinaryOperator {
public:
  static inline bool classof(const SubOperator *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Sub;
  }
  static inline bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Instruction::Sub;
  }
  static inline bool classof(const Value *V) {
    return (isa<Instruction>(V) && classof(cast<Instruction>(V))) ||
           (isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V)));
  }
};

/// MulOperator - Utility class for integer multiplication operators.
///
class MulOperator : public OverflowingBinaryOperator {
public:
  static inline bool classof(const MulOperator *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Mul;
  }
  static inline bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Instruction::Mul;
  }
  static inline bool classof(const Value *V) {
    return (isa<Instruction>(V) && classof(cast<Instruction>(V))) ||
           (isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V)));
  }
};

/// SDivOperator - An Operator with opcode Instruction::SDiv.
///
class SDivOperator : public Operator {
public:
  /// isExact - Test whether this division is known to be exact, with
  /// zero remainder.
  bool isExact() const {
    return SubclassOptionalData & (1 << 0);
  }
  void setIsExact(bool B) {
    SubclassOptionalData = (SubclassOptionalData & ~(1 << 0)) | (B << 0);
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SDivOperator *) { return true; }
  static inline bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Instruction::SDiv;
  }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::SDiv;
  }
  static inline bool classof(const Value *V) {
    return (isa<Instruction>(V) && classof(cast<Instruction>(V))) ||
           (isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V)));
  }
};

class GEPOperator : public Operator {
public:
  /// isInBounds - Test whether this is an inbounds GEP, as defined
  /// by LangRef.html.
  bool isInBounds() const {
    return SubclassOptionalData & (1 << 0);
  }
  void setIsInBounds(bool B) {
    SubclassOptionalData = (SubclassOptionalData & ~(1 << 0)) | (B << 0);
  }

  inline op_iterator       idx_begin()       { return op_begin()+1; }
  inline const_op_iterator idx_begin() const { return op_begin()+1; }
  inline op_iterator       idx_end()         { return op_end(); }
  inline const_op_iterator idx_end()   const { return op_end(); }

  Value *getPointerOperand() {
    return getOperand(0);
  }
  const Value *getPointerOperand() const {
    return getOperand(0);
  }
  static unsigned getPointerOperandIndex() {
    return 0U;                      // get index for modifying correct operand
  }

  /// getPointerOperandType - Method to return the pointer operand as a
  /// PointerType.
  const PointerType *getPointerOperandType() const {
    return reinterpret_cast<const PointerType*>(getPointerOperand()->getType());
  }

  unsigned getNumIndices() const {  // Note: always non-negative
    return getNumOperands() - 1;
  }

  bool hasIndices() const {
    return getNumOperands() > 1;
  }

  /// hasAllZeroIndices - Return true if all of the indices of this GEP are
  /// zeros.  If so, the result pointer and the first operand have the same
  /// value, just potentially different types.
  bool hasAllZeroIndices() const {
    for (const_op_iterator I = idx_begin(), E = idx_end(); I != E; ++I) {
      if (Constant *C = dyn_cast<Constant>(I))
        if (C->isNullValue())
          continue;
      return false;
    }
    return true;
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const GEPOperator *) { return true; }
  static inline bool classof(const GetElementPtrInst *) { return true; }
  static inline bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Instruction::GetElementPtr;
  }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::GetElementPtr;
  }
  static inline bool classof(const Value *V) {
    return (isa<Instruction>(V) && classof(cast<Instruction>(V))) ||
           (isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V)));
  }
};

} // End llvm namespace

#endif
