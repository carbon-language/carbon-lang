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

#include "llvm/Constants.h"
#include "llvm/Instruction.h"

namespace llvm {

class GetElementPtrInst;
class BinaryOperator;
class ConstantExpr;

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
  static inline bool classof(const Instruction *) { return true; }
  static inline bool classof(const ConstantExpr *) { return true; }
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
  enum {
    NoUnsignedWrap = (1 << 0),
    NoSignedWrap   = (1 << 1)
  };

private:
  ~OverflowingBinaryOperator(); // do not implement

  friend class BinaryOperator;
  friend class ConstantExpr;
  void setHasNoUnsignedWrap(bool B) {
    SubclassOptionalData =
      (SubclassOptionalData & ~NoUnsignedWrap) | (B * NoUnsignedWrap);
  }
  void setHasNoSignedWrap(bool B) {
    SubclassOptionalData =
      (SubclassOptionalData & ~NoSignedWrap) | (B * NoSignedWrap);
  }

public:
  /// hasNoUnsignedWrap - Test whether this operation is known to never
  /// undergo unsigned overflow, aka the nuw property.
  bool hasNoUnsignedWrap() const {
    return SubclassOptionalData & NoUnsignedWrap;
  }

  /// hasNoSignedWrap - Test whether this operation is known to never
  /// undergo signed overflow, aka the nsw property.
  bool hasNoSignedWrap() const {
    return (SubclassOptionalData & NoSignedWrap) != 0;
  }

  static inline bool classof(const OverflowingBinaryOperator *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Add ||
           I->getOpcode() == Instruction::Sub ||
           I->getOpcode() == Instruction::Mul ||
           I->getOpcode() == Instruction::Shl;
  }
  static inline bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Instruction::Add ||
           CE->getOpcode() == Instruction::Sub ||
           CE->getOpcode() == Instruction::Mul ||
           CE->getOpcode() == Instruction::Shl;
  }
  static inline bool classof(const Value *V) {
    return (isa<Instruction>(V) && classof(cast<Instruction>(V))) ||
           (isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V)));
  }
};

/// PossiblyExactOperator - A udiv or sdiv instruction, which can be marked as
/// "exact", indicating that no bits are destroyed.
class PossiblyExactOperator : public Operator {
public:
  enum {
    IsExact = (1 << 0)
  };
  
  friend class BinaryOperator;
  friend class ConstantExpr;
  void setIsExact(bool B) {
    SubclassOptionalData = (SubclassOptionalData & ~IsExact) | (B * IsExact);
  }
  
private:
  ~PossiblyExactOperator(); // do not implement
public:
  /// isExact - Test whether this division is known to be exact, with
  /// zero remainder.
  bool isExact() const {
    return SubclassOptionalData & IsExact;
  }
  
  static bool isPossiblyExactOpcode(unsigned OpC) {
    return OpC == Instruction::SDiv ||
           OpC == Instruction::UDiv ||
           OpC == Instruction::AShr ||
           OpC == Instruction::LShr;
  }
  static inline bool classof(const ConstantExpr *CE) {
    return isPossiblyExactOpcode(CE->getOpcode());
  }
  static inline bool classof(const Instruction *I) {
    return isPossiblyExactOpcode(I->getOpcode());
  }
  static inline bool classof(const Value *V) {
    return (isa<Instruction>(V) && classof(cast<Instruction>(V))) ||
           (isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V)));
  }
};
  

  
/// ConcreteOperator - A helper template for defining operators for individual
/// opcodes.
template<typename SuperClass, unsigned Opc>
class ConcreteOperator : public SuperClass {
  ~ConcreteOperator(); // DO NOT IMPLEMENT
public:
  static inline bool classof(const ConcreteOperator<SuperClass, Opc> *) {
    return true;
  }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Opc;
  }
  static inline bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Opc;
  }
  static inline bool classof(const Value *V) {
    return (isa<Instruction>(V) && classof(cast<Instruction>(V))) ||
           (isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V)));
  }
};

class FAddOperator : public ConcreteOperator<Operator, Instruction::FAdd> {
  ~FAddOperator(); // DO NOT IMPLEMENT
};
class FSubOperator : public ConcreteOperator<Operator, Instruction::FSub> {
  ~FSubOperator(); // DO NOT IMPLEMENT
};
class FMulOperator : public ConcreteOperator<Operator, Instruction::FMul> {
  ~FMulOperator(); // DO NOT IMPLEMENT
};
class FDivOperator : public ConcreteOperator<Operator, Instruction::FDiv> {
  ~FDivOperator(); // DO NOT IMPLEMENT
};
class URemOperator : public ConcreteOperator<Operator, Instruction::URem> {
  ~URemOperator(); // DO NOT IMPLEMENT
};
class SRemOperator : public ConcreteOperator<Operator, Instruction::SRem> {
  ~SRemOperator(); // DO NOT IMPLEMENT
};
class FRemOperator : public ConcreteOperator<Operator, Instruction::FRem> {
  ~FRemOperator(); // DO NOT IMPLEMENT
};
class AndOperator : public ConcreteOperator<Operator, Instruction::And> {
  ~AndOperator(); // DO NOT IMPLEMENT
};
class OrOperator : public ConcreteOperator<Operator, Instruction::Or> {
  ~OrOperator(); // DO NOT IMPLEMENT
};
class XorOperator : public ConcreteOperator<Operator, Instruction::Xor> {
  ~XorOperator(); // DO NOT IMPLEMENT
};
class TruncOperator : public ConcreteOperator<Operator, Instruction::Trunc> {
  ~TruncOperator(); // DO NOT IMPLEMENT
};
class ZExtOperator : public ConcreteOperator<Operator, Instruction::ZExt> {
  ~ZExtOperator(); // DO NOT IMPLEMENT
};
class SExtOperator : public ConcreteOperator<Operator, Instruction::SExt> {
  ~SExtOperator(); // DO NOT IMPLEMENT
};
class FPToUIOperator : public ConcreteOperator<Operator, Instruction::FPToUI> {
  ~FPToUIOperator(); // DO NOT IMPLEMENT
};
class FPToSIOperator : public ConcreteOperator<Operator, Instruction::FPToSI> {
  ~FPToSIOperator(); // DO NOT IMPLEMENT
};
class UIToFPOperator : public ConcreteOperator<Operator, Instruction::UIToFP> {
  ~UIToFPOperator(); // DO NOT IMPLEMENT
};
class SIToFPOperator : public ConcreteOperator<Operator, Instruction::SIToFP> {
  ~SIToFPOperator(); // DO NOT IMPLEMENT
};
class FPTruncOperator
  : public ConcreteOperator<Operator, Instruction::FPTrunc> {
  ~FPTruncOperator(); // DO NOT IMPLEMENT
};
class FPExtOperator : public ConcreteOperator<Operator, Instruction::FPExt> {
  ~FPExtOperator(); // DO NOT IMPLEMENT
};
class PtrToIntOperator
  : public ConcreteOperator<Operator, Instruction::PtrToInt> {
  ~PtrToIntOperator(); // DO NOT IMPLEMENT
};
class IntToPtrOperator
  : public ConcreteOperator<Operator, Instruction::IntToPtr> {
  ~IntToPtrOperator(); // DO NOT IMPLEMENT
};
class BitCastOperator
  : public ConcreteOperator<Operator, Instruction::BitCast> {
  ~BitCastOperator(); // DO NOT IMPLEMENT
};
class ICmpOperator : public ConcreteOperator<Operator, Instruction::ICmp> {
  ~ICmpOperator(); // DO NOT IMPLEMENT
};
class FCmpOperator : public ConcreteOperator<Operator, Instruction::FCmp> {
  ~FCmpOperator(); // DO NOT IMPLEMENT
};
class SelectOperator : public ConcreteOperator<Operator, Instruction::Select> {
  ~SelectOperator(); // DO NOT IMPLEMENT
};
class ExtractElementOperator
  : public ConcreteOperator<Operator, Instruction::ExtractElement> {
  ~ExtractElementOperator(); // DO NOT IMPLEMENT
};
class InsertElementOperator
  : public ConcreteOperator<Operator, Instruction::InsertElement> {
  ~InsertElementOperator(); // DO NOT IMPLEMENT
};
class ShuffleVectorOperator
  : public ConcreteOperator<Operator, Instruction::ShuffleVector> {
  ~ShuffleVectorOperator(); // DO NOT IMPLEMENT
};
class ExtractValueOperator
  : public ConcreteOperator<Operator, Instruction::ExtractValue> {
  ~ExtractValueOperator(); // DO NOT IMPLEMENT
};
class InsertValueOperator
  : public ConcreteOperator<Operator, Instruction::InsertValue> {
  ~InsertValueOperator(); // DO NOT IMPLEMENT
};

class AddOperator
  : public ConcreteOperator<OverflowingBinaryOperator, Instruction::Add> {
  ~AddOperator(); // DO NOT IMPLEMENT
};
class SubOperator
  : public ConcreteOperator<OverflowingBinaryOperator, Instruction::Sub> {
  ~SubOperator(); // DO NOT IMPLEMENT
};
class MulOperator
  : public ConcreteOperator<OverflowingBinaryOperator, Instruction::Mul> {
  ~MulOperator(); // DO NOT IMPLEMENT
};
class ShlOperator
  : public ConcreteOperator<OverflowingBinaryOperator, Instruction::Shl> {
  ~ShlOperator(); // DO NOT IMPLEMENT
};

  
class SDivOperator
  : public ConcreteOperator<PossiblyExactOperator, Instruction::SDiv> {
  ~SDivOperator(); // DO NOT IMPLEMENT
};
class UDivOperator
  : public ConcreteOperator<PossiblyExactOperator, Instruction::UDiv> {
  ~UDivOperator(); // DO NOT IMPLEMENT
};
class AShrOperator
  : public ConcreteOperator<PossiblyExactOperator, Instruction::AShr> {
  ~AShrOperator(); // DO NOT IMPLEMENT
};
class LShrOperator
  : public ConcreteOperator<PossiblyExactOperator, Instruction::LShr> {
  ~LShrOperator(); // DO NOT IMPLEMENT
};
  
  
  
class GEPOperator
  : public ConcreteOperator<Operator, Instruction::GetElementPtr> {
  ~GEPOperator(); // DO NOT IMPLEMENT

  enum {
    IsInBounds = (1 << 0)
  };

  friend class GetElementPtrInst;
  friend class ConstantExpr;
  void setIsInBounds(bool B) {
    SubclassOptionalData =
      (SubclassOptionalData & ~IsInBounds) | (B * IsInBounds);
  }

public:
  /// isInBounds - Test whether this is an inbounds GEP, as defined
  /// by LangRef.html.
  bool isInBounds() const {
    return SubclassOptionalData & IsInBounds;
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
      if (ConstantInt *C = dyn_cast<ConstantInt>(I))
        if (C->isZero())
          continue;
      return false;
    }
    return true;
  }

  /// hasAllConstantIndices - Return true if all of the indices of this GEP are
  /// constant integers.  If so, the result pointer and the first operand have
  /// a constant offset between them.
  bool hasAllConstantIndices() const {
    for (const_op_iterator I = idx_begin(), E = idx_end(); I != E; ++I) {
      if (!isa<ConstantInt>(I))
        return false;
    }
    return true;
  }
};

} // End llvm namespace

#endif
