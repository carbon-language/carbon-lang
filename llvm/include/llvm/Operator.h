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
/// which may exhibit overflow - Add, Sub, and Mul.
///
class OverflowingBinaryOperator : public Operator {
public:
  /// hasNoSignedOverflow - Test whether this operation is known to never
  /// undergo signed overflow.
  bool hasNoSignedOverflow() const {
    return SubclassOptionalData & (1 << 0);
  }
  void setHasNoSignedOverflow(bool B) {
    SubclassOptionalData = (SubclassOptionalData & ~(1 << 0)) | (B << 0);
  }

  /// hasNoUnsignedOverflow - Test whether this operation is known to never
  /// undergo unsigned overflow.
  bool hasNoUnsignedOverflow() const {
    return SubclassOptionalData & (1 << 1);
  }
  void setHasNoUnsignedOverflow(bool B) {
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

/// UDivOperator - An Operator with opcode Instruction::UDiv.
///
class UDivOperator : public Operator {
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
  static inline bool classof(const UDivOperator *) { return true; }
  static inline bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Instruction::UDiv;
  }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::UDiv;
  }
  static inline bool classof(const Value *V) {
    return (isa<Instruction>(V) && classof(cast<Instruction>(V))) ||
           (isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V)));
  }
};

class GEPOperator : public Operator {
public:
  /// hasNoPointerOverflow - Return true if this GetElementPtr is known to
  /// never have overflow in the pointer addition portions of its effective
  /// computation. GetElementPtr computation involves several phases;
  /// overflow can be considered to occur in index typecasting, array index
  /// scaling, and the addition of the base pointer with offsets. This flag
  /// only applies to the last of these. The operands are added to the base
  /// pointer one at a time from left to right. This function returns false
  /// if any of these additions results in an address value which is not
  /// known to be within the allocated address space that the base pointer
  /// points into, or within one element (of the original allocation) past
  /// the end.
  bool hasNoPointerOverflow() const {
    return SubclassOptionalData & (1 << 0);
  }
  void setHasNoPointerOverflow(bool B) {
    SubclassOptionalData = (SubclassOptionalData & ~(1 << 0)) | (B << 0);
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
    return isa<GetElementPtrInst>(V) ||
           (isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V)));
  }
};

} // End llvm namespace

#endif
