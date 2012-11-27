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
#include "llvm/DerivedTypes.h"
#include "llvm/Instruction.h"
#include "llvm/Type.h"

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
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
  void *operator new(size_t s) LLVM_DELETED_FUNCTION;
  Operator() LLVM_DELETED_FUNCTION;

protected:
  // NOTE: Cannot use LLVM_DELETED_FUNCTION because it's not legal to delete
  // an overridden method that's not deleted in the base class. Cannot leave
  // this unimplemented because that leads to an ODR-violation.
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

private:
  friend class BinaryOperator;
  friend class ConstantExpr;
  void setIsExact(bool B) {
    SubclassOptionalData = (SubclassOptionalData & ~IsExact) | (B * IsExact);
  }

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

/// Convenience struct for specifying and reasoning about fast-math flags.
struct FastMathFlags {
  bool UnsafeAlgebra   : 1;
  bool NoNaNs          : 1;
  bool NoInfs          : 1;
  bool NoSignedZeros   : 1;
  bool AllowReciprocal : 1;

  FastMathFlags() : UnsafeAlgebra(false), NoNaNs(false), NoInfs(false),
                    NoSignedZeros(false), AllowReciprocal(false)
  { }

  bool any() {
    return UnsafeAlgebra || NoNaNs || NoInfs || NoSignedZeros ||
      AllowReciprocal;
  }
};


/// FPMathOperator - Utility class for floating point operations which can have
/// information about relaxed accuracy requirements attached to them.
class FPMathOperator : public Operator {
public:
  enum {
    UnsafeAlgebra   = (1 << 0),
    NoNaNs          = (1 << 1),
    NoInfs          = (1 << 2),
    NoSignedZeros   = (1 << 3),
    AllowReciprocal = (1 << 4)
  };

private:
  friend class Instruction;

  void setHasUnsafeAlgebra(bool B) {
    SubclassOptionalData =
      (SubclassOptionalData & ~UnsafeAlgebra) | (B * UnsafeAlgebra);

    // Unsafe algebra implies all the others
    if (B) {
      setHasNoNaNs(true);
      setHasNoInfs(true);
      setHasNoSignedZeros(true);
      setHasAllowReciprocal(true);
    }
  }
  void setHasNoNaNs(bool B) {
    SubclassOptionalData =
      (SubclassOptionalData & ~NoNaNs) | (B * NoNaNs);
  }
  void setHasNoInfs(bool B) {
    SubclassOptionalData =
      (SubclassOptionalData & ~NoInfs) | (B * NoInfs);
  }
  void setHasNoSignedZeros(bool B) {
    SubclassOptionalData =
      (SubclassOptionalData & ~NoSignedZeros) | (B * NoSignedZeros);
  }
  void setHasAllowReciprocal(bool B) {
    SubclassOptionalData =
      (SubclassOptionalData & ~AllowReciprocal) | (B * AllowReciprocal);
  }

  /// Convenience function for setting all the fast-math flags
  void setFastMathFlags(FastMathFlags FMF) {
    if (FMF.UnsafeAlgebra) {
      // Set all the bits to true
      setHasUnsafeAlgebra(true);
      return;
    }

    setHasUnsafeAlgebra(FMF.UnsafeAlgebra);
    setHasNoNaNs(FMF.NoNaNs);
    setHasNoInfs(FMF.NoInfs);
    setHasNoSignedZeros(FMF.NoSignedZeros);
    setHasAllowReciprocal(FMF.AllowReciprocal);
  }

public:
  /// Test whether this operation is permitted to be
  /// algebraically transformed, aka the 'A' fast-math property.
  bool hasUnsafeAlgebra() const {
    return (SubclassOptionalData & UnsafeAlgebra) != 0;
  }

  /// Test whether this operation's arguments and results are to be
  /// treated as non-NaN, aka the 'N' fast-math property.
  bool hasNoNaNs() const {
    return (SubclassOptionalData & NoNaNs) != 0;
  }

  /// Test whether this operation's arguments and results are to be
  /// treated as NoN-Inf, aka the 'I' fast-math property.
  bool hasNoInfs() const {
    return (SubclassOptionalData & NoInfs) != 0;
  }

  /// Test whether this operation can treat the sign of zero
  /// as insignificant, aka the 'S' fast-math property.
  bool hasNoSignedZeros() const {
    return (SubclassOptionalData & NoSignedZeros) != 0;
  }

  /// Test whether this operation is permitted to use
  /// reciprocal instead of division, aka the 'R' fast-math property.
  bool hasAllowReciprocal() const {
    return (SubclassOptionalData & AllowReciprocal) != 0;
  }

  /// Convenience function for getting all the fast-math flags
  FastMathFlags getFastMathFlags() const {
    FastMathFlags FMF;
    FMF.UnsafeAlgebra   = hasUnsafeAlgebra();
    FMF.NoNaNs          = hasNoNaNs();
    FMF.NoInfs          = hasNoInfs();
    FMF.NoSignedZeros   = hasNoSignedZeros();
    FMF.AllowReciprocal = hasAllowReciprocal();
    return FMF;
  }


  /// \brief Get the maximum error permitted by this operation in ULPs.  An
  /// accuracy of 0.0 means that the operation should be performed with the
  /// default precision.
  float getFPAccuracy() const;

  static inline bool classof(const Instruction *I) {
    return I->getType()->isFPOrFPVectorTy();
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


/// ConcreteOperator - A helper template for defining operators for individual
/// opcodes.
template<typename SuperClass, unsigned Opc>
class ConcreteOperator : public SuperClass {
public:
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

class AddOperator
  : public ConcreteOperator<OverflowingBinaryOperator, Instruction::Add> {
};
class SubOperator
  : public ConcreteOperator<OverflowingBinaryOperator, Instruction::Sub> {
};
class MulOperator
  : public ConcreteOperator<OverflowingBinaryOperator, Instruction::Mul> {
};
class ShlOperator
  : public ConcreteOperator<OverflowingBinaryOperator, Instruction::Shl> {
};


class SDivOperator
  : public ConcreteOperator<PossiblyExactOperator, Instruction::SDiv> {
};
class UDivOperator
  : public ConcreteOperator<PossiblyExactOperator, Instruction::UDiv> {
};
class AShrOperator
  : public ConcreteOperator<PossiblyExactOperator, Instruction::AShr> {
};
class LShrOperator
  : public ConcreteOperator<PossiblyExactOperator, Instruction::LShr> {
};



class GEPOperator
  : public ConcreteOperator<Operator, Instruction::GetElementPtr> {
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
  Type *getPointerOperandType() const {
    return getPointerOperand()->getType();
  }

  /// getPointerAddressSpace - Method to return the address space of the
  /// pointer operand.
  unsigned getPointerAddressSpace() const {
    return cast<PointerType>(getPointerOperandType())->getAddressSpace();
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
