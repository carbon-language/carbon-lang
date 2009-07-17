//===-- llvm/BinaryOperators.h - BinaryOperator subclasses ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines various classes for working with specific BinaryOperators,
// exposing special properties that individual operations have.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BINARY_OPTERATORS_H
#define LLVM_BINARY_OPTERATORS_H

#include "llvm/InstrTypes.h"

namespace llvm {

//===----------------------------------------------------------------------===//
//                      SpecificBinaryOperator Class
//===----------------------------------------------------------------------===//

/// SpecificBinaryOperator - This is a base class for utility classes that
/// provide additional information about binary operator instructions with
/// specific opcodes.
///
class SpecificBinaryOperator : public BinaryOperator {
private:
  // Do not implement any of these. The SpecificBinaryOperator class is
  // intended to be used as a utility, and is never itself instantiated.
  void *operator new(size_t, unsigned);
  void *operator new(size_t s);
  void Create(BinaryOps Op, Value *S1, Value *S2,
              const std::string &Name = "",
              Instruction *InsertBefore = 0);
  void Create(BinaryOps Op, Value *S1, Value *S2,
              const std::string &Name,
              BasicBlock *InsertAtEnd);
  SpecificBinaryOperator();
  ~SpecificBinaryOperator();
  void init(BinaryOps iType);
};

/// OverflowingBinaryOperator - Base class for integer arithmetic operators
/// which may exhibit overflow - Add, Sub, and Mul.
///
class OverflowingBinaryOperator : public SpecificBinaryOperator {
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
  static inline bool classof(const BinaryOperator *BO) {
    return BO->getOpcode() == Instruction::Add ||
           BO->getOpcode() == Instruction::Sub ||
           BO->getOpcode() == Instruction::Mul;
  }
  static inline bool classof(const Instruction *I) {
    return I->isBinaryOp() && classof(cast<BinaryOperator>(I));
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

/// UDivInst - BinaryOperators with opcode Instruction::UDiv.
///
class UDivInst : public SpecificBinaryOperator {
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
  static inline bool classof(const UDivInst *) { return true; }
  static inline bool classof(const BinaryOperator *BO) {
    return BO->getOpcode() == Instruction::UDiv;
  }
  static inline bool classof(const Instruction *I) {
    return I->isBinaryOp() && classof(cast<BinaryOperator>(I));
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

} // End llvm namespace

#endif
