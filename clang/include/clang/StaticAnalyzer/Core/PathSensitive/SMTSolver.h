//== SMTSolver.h ------------------------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a SMT generic Solver API, which will be the base class
//  for every SMT solver specific class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SMTSOLVER_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SMTSOLVER_H

#include "clang/StaticAnalyzer/Core/PathSensitive/SMTExpr.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SMTSort.h"

namespace clang {
namespace ento {

class SMTSolver {
public:
  SMTSolver() = default;
  virtual ~SMTSolver() = default;

  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }

  // Return an appropriate floating-point sort for the given bitwidth.
  SMTSortRef getFloatSort(unsigned BitWidth) {
    switch (BitWidth) {
    case 16:
      return getFloat16Sort();
    case 32:
      return getFloat32Sort();
    case 64:
      return getFloat64Sort();
    case 128:
      return getFloat128Sort();
    default:;
    }
    llvm_unreachable("Unsupported floating-point bitwidth!");
  }

  // Return an appropriate sort, given a QualType
  SMTSortRef mkSort(const QualType &Ty, unsigned BitWidth) {
    if (Ty->isBooleanType())
      return getBoolSort();

    if (Ty->isRealFloatingType())
      return getFloatSort(BitWidth);

    return getBitvectorSort(BitWidth);
  }

  /// Construct a Z3Expr from a unary operator, given a Z3_context.
  SMTExprRef fromUnOp(const UnaryOperator::Opcode Op, const SMTExprRef &Exp) {
    switch (Op) {
    case UO_Minus:
      return mkBVNeg(Exp);

    case UO_Not:
      return mkBVNot(Exp);

    case UO_LNot:
      return mkNot(Exp);

    default:;
    }
    llvm_unreachable("Unimplemented opcode");
  }

  /// Construct a Z3Expr from a floating-point unary operator, given a
  /// Z3_context.
  SMTExprRef fromFloatUnOp(const UnaryOperator::Opcode Op,
                           const SMTExprRef &Exp) {
    switch (Op) {
    case UO_Minus:
      return mkFPNeg(Exp);

    case UO_LNot:
      return fromUnOp(Op, Exp);

    default:;
    }
    llvm_unreachable("Unimplemented opcode");
  }

  /// Construct a Z3Expr from a n-ary binary operator.
  SMTExprRef fromNBinOp(const BinaryOperator::Opcode Op,
                        const std::vector<SMTExprRef> &ASTs) {
    assert(!ASTs.empty());

    if (Op != BO_LAnd && Op != BO_LOr)
      llvm_unreachable("Unimplemented opcode");

    SMTExprRef res = ASTs.front();
    for (std::size_t i = 1; i < ASTs.size(); ++i)
      res = (Op == BO_LAnd) ? mkAnd(res, ASTs[i]) : mkOr(res, ASTs[i]);
    return res;
  }

  /// Construct a Z3Expr from a binary operator, given a Z3_context.
  SMTExprRef fromBinOp(const SMTExprRef &LHS, const BinaryOperator::Opcode Op,
                       const SMTExprRef &RHS, bool isSigned) {
    assert(*getSort(LHS) == *getSort(RHS) && "AST's must have the same sort!");

    switch (Op) {
    // Multiplicative operators
    case BO_Mul:
      return mkBVMul(LHS, RHS);

    case BO_Div:
      return isSigned ? mkBVSDiv(LHS, RHS) : mkBVUDiv(LHS, RHS);

    case BO_Rem:
      return isSigned ? mkBVSRem(LHS, RHS) : mkBVURem(LHS, RHS);

      // Additive operators
    case BO_Add:
      return mkBVAdd(LHS, RHS);

    case BO_Sub:
      return mkBVSub(LHS, RHS);

      // Bitwise shift operators
    case BO_Shl:
      return mkBVShl(LHS, RHS);

    case BO_Shr:
      return isSigned ? mkBVAshr(LHS, RHS) : mkBVLshr(LHS, RHS);

      // Relational operators
    case BO_LT:
      return isSigned ? mkBVSlt(LHS, RHS) : mkBVUlt(LHS, RHS);

    case BO_GT:
      return isSigned ? mkBVSgt(LHS, RHS) : mkBVUgt(LHS, RHS);

    case BO_LE:
      return isSigned ? mkBVSle(LHS, RHS) : mkBVUle(LHS, RHS);

    case BO_GE:
      return isSigned ? mkBVSge(LHS, RHS) : mkBVUge(LHS, RHS);

      // Equality operators
    case BO_EQ:
      return mkEqual(LHS, RHS);

    case BO_NE:
      return fromUnOp(UO_LNot, fromBinOp(LHS, BO_EQ, RHS, isSigned));

      // Bitwise operators
    case BO_And:
      return mkBVAnd(LHS, RHS);

    case BO_Xor:
      return mkBVXor(LHS, RHS);

    case BO_Or:
      return mkBVOr(LHS, RHS);

      // Logical operators
    case BO_LAnd:
      return mkAnd(LHS, RHS);

    case BO_LOr:
      return mkOr(LHS, RHS);

    default:;
    }
    llvm_unreachable("Unimplemented opcode");
  }

  /// Construct a Z3Expr from a special floating-point binary operator, given
  /// a Z3_context.
  SMTExprRef fromFloatSpecialBinOp(const SMTExprRef &LHS,
                                   const BinaryOperator::Opcode Op,
                                   const llvm::APFloat::fltCategory &RHS) {
    switch (Op) {
    // Equality operators
    case BO_EQ:
      switch (RHS) {
      case llvm::APFloat::fcInfinity:
        return mkFPIsInfinite(LHS);

      case llvm::APFloat::fcNaN:
        return mkFPIsNaN(LHS);

      case llvm::APFloat::fcNormal:
        return mkFPIsNormal(LHS);

      case llvm::APFloat::fcZero:
        return mkFPIsZero(LHS);
      }
      break;

    case BO_NE:
      return fromFloatUnOp(UO_LNot, fromFloatSpecialBinOp(LHS, BO_EQ, RHS));

    default:;
    }

    llvm_unreachable("Unimplemented opcode");
  }

  /// Construct a Z3Expr from a floating-point binary operator, given a
  /// Z3_context.
  SMTExprRef fromFloatBinOp(const SMTExprRef &LHS,
                            const BinaryOperator::Opcode Op,
                            const SMTExprRef &RHS) {
    assert(*getSort(LHS) == *getSort(RHS) && "AST's must have the same sort!");

    switch (Op) {
    // Multiplicative operators
    case BO_Mul:
      return mkFPMul(LHS, RHS);

    case BO_Div:
      return mkFPDiv(LHS, RHS);

    case BO_Rem:
      return mkFPRem(LHS, RHS);

      // Additive operators
    case BO_Add:
      return mkFPAdd(LHS, RHS);

    case BO_Sub:
      return mkFPSub(LHS, RHS);

      // Relational operators
    case BO_LT:
      return mkFPLt(LHS, RHS);

    case BO_GT:
      return mkFPGt(LHS, RHS);

    case BO_LE:
      return mkFPLe(LHS, RHS);

    case BO_GE:
      return mkFPGe(LHS, RHS);

      // Equality operators
    case BO_EQ:
      return mkFPEqual(LHS, RHS);

    case BO_NE:
      return fromFloatUnOp(UO_LNot, fromFloatBinOp(LHS, BO_EQ, RHS));

      // Logical operators
    case BO_LAnd:
    case BO_LOr:
      return fromBinOp(LHS, Op, RHS, false);

    default:;
    }

    llvm_unreachable("Unimplemented opcode");
  }

  /// Construct a Z3Expr from a SymbolCast, given a Z3_context.
  SMTExprRef fromCast(const SMTExprRef &Exp, QualType ToTy, uint64_t ToBitWidth,
                      QualType FromTy, uint64_t FromBitWidth) {
    if ((FromTy->isIntegralOrEnumerationType() &&
         ToTy->isIntegralOrEnumerationType()) ||
        (FromTy->isAnyPointerType() ^ ToTy->isAnyPointerType()) ||
        (FromTy->isBlockPointerType() ^ ToTy->isBlockPointerType()) ||
        (FromTy->isReferenceType() ^ ToTy->isReferenceType())) {

      if (FromTy->isBooleanType()) {
        assert(ToBitWidth > 0 && "BitWidth must be positive!");
        return mkIte(Exp, mkBitvector(llvm::APSInt("1"), ToBitWidth),
                     mkBitvector(llvm::APSInt("0"), ToBitWidth));
      }

      if (ToBitWidth > FromBitWidth)
        return FromTy->isSignedIntegerOrEnumerationType()
                   ? mkSignExt(ToBitWidth - FromBitWidth, Exp)
                   : mkZeroExt(ToBitWidth - FromBitWidth, Exp);

      if (ToBitWidth < FromBitWidth)
        return mkExtract(ToBitWidth - 1, 0, Exp);

      // Both are bitvectors with the same width, ignore the type cast
      return Exp;
    }

    if (FromTy->isRealFloatingType() && ToTy->isRealFloatingType()) {
      if (ToBitWidth != FromBitWidth)
        return mkFPtoFP(Exp, getFloatSort(ToBitWidth));

      return Exp;
    }

    if (FromTy->isIntegralOrEnumerationType() && ToTy->isRealFloatingType()) {
      SMTSortRef Sort = getFloatSort(ToBitWidth);
      return FromTy->isSignedIntegerOrEnumerationType() ? mkFPtoSBV(Exp, Sort)
                                                        : mkFPtoUBV(Exp, Sort);
    }

    if (FromTy->isRealFloatingType() && ToTy->isIntegralOrEnumerationType())
      return ToTy->isSignedIntegerOrEnumerationType()
                 ? mkSBVtoFP(Exp, ToBitWidth)
                 : mkUBVtoFP(Exp, ToBitWidth);

    llvm_unreachable("Unsupported explicit type cast!");
  }

  // Callback function for doCast parameter on APSInt type.
  llvm::APSInt castAPSInt(const llvm::APSInt &V, QualType ToTy,
                          uint64_t ToWidth, QualType FromTy,
                          uint64_t FromWidth) {
    APSIntType TargetType(ToWidth, !ToTy->isSignedIntegerOrEnumerationType());
    return TargetType.convert(V);
  }

  // Return a boolean sort.
  virtual SMTSortRef getBoolSort() = 0;

  // Return an appropriate bitvector sort for the given bitwidth.
  virtual SMTSortRef getBitvectorSort(const unsigned BitWidth) = 0;

  // Return a floating-point sort of width 16
  virtual SMTSortRef getFloat16Sort() = 0;

  // Return a floating-point sort of width 32
  virtual SMTSortRef getFloat32Sort() = 0;

  // Return a floating-point sort of width 64
  virtual SMTSortRef getFloat64Sort() = 0;

  // Return a floating-point sort of width 128
  virtual SMTSortRef getFloat128Sort() = 0;

  // Return an appropriate sort for the given AST.
  virtual SMTSortRef getSort(const SMTExprRef &AST) = 0;

  // Return a new SMTExprRef from an SMTExpr
  virtual SMTExprRef newExprRef(const SMTExpr &E) const = 0;

  /// Given a constraint, add it to the solver
  virtual void addConstraint(const SMTExprRef &Exp) const = 0;

  /// Create a bitvector addition operation
  virtual SMTExprRef mkBVAdd(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector subtraction operation
  virtual SMTExprRef mkBVSub(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector multiplication operation
  virtual SMTExprRef mkBVMul(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector signed modulus operation
  virtual SMTExprRef mkBVSRem(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector unsigned modulus operation
  virtual SMTExprRef mkBVURem(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector signed division operation
  virtual SMTExprRef mkBVSDiv(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector unsigned division operation
  virtual SMTExprRef mkBVUDiv(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector logical shift left operation
  virtual SMTExprRef mkBVShl(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector arithmetic shift right operation
  virtual SMTExprRef mkBVAshr(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector logical shift right operation
  virtual SMTExprRef mkBVLshr(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector negation operation
  virtual SMTExprRef mkBVNeg(const SMTExprRef &Exp) = 0;

  /// Create a bitvector not operation
  virtual SMTExprRef mkBVNot(const SMTExprRef &Exp) = 0;

  /// Create a bitvector xor operation
  virtual SMTExprRef mkBVXor(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector or operation
  virtual SMTExprRef mkBVOr(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector and operation
  virtual SMTExprRef mkBVAnd(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector unsigned less-than operation
  virtual SMTExprRef mkBVUlt(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector signed less-than operation
  virtual SMTExprRef mkBVSlt(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector unsigned greater-than operation
  virtual SMTExprRef mkBVUgt(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector signed greater-than operation
  virtual SMTExprRef mkBVSgt(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector unsigned less-equal-than operation
  virtual SMTExprRef mkBVUle(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector signed less-equal-than operation
  virtual SMTExprRef mkBVSle(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector unsigned greater-equal-than operation
  virtual SMTExprRef mkBVUge(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a bitvector signed greater-equal-than operation
  virtual SMTExprRef mkBVSge(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Create a boolean not operation
  virtual SMTExprRef mkNot(const SMTExprRef &Exp) = 0;

  /// Create a bitvector equality operation
  virtual SMTExprRef mkEqual(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  virtual SMTExprRef mkAnd(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  virtual SMTExprRef mkOr(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  virtual SMTExprRef mkIte(const SMTExprRef &Cond, const SMTExprRef &T,
                           const SMTExprRef &F) = 0;

  virtual SMTExprRef mkSignExt(unsigned i, const SMTExprRef &Exp) = 0;

  virtual SMTExprRef mkZeroExt(unsigned i, const SMTExprRef &Exp) = 0;

  virtual SMTExprRef mkExtract(unsigned High, unsigned Low,
                               const SMTExprRef &Exp) = 0;

  virtual SMTExprRef mkConcat(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  virtual SMTExprRef mkFPNeg(const SMTExprRef &Exp) = 0;

  virtual SMTExprRef mkFPIsInfinite(const SMTExprRef &Exp) = 0;

  virtual SMTExprRef mkFPIsNaN(const SMTExprRef &Exp) = 0;

  virtual SMTExprRef mkFPIsNormal(const SMTExprRef &Exp) = 0;

  virtual SMTExprRef mkFPIsZero(const SMTExprRef &Exp) = 0;

  virtual SMTExprRef mkFPMul(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  virtual SMTExprRef mkFPDiv(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  virtual SMTExprRef mkFPRem(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  virtual SMTExprRef mkFPAdd(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  virtual SMTExprRef mkFPSub(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  virtual SMTExprRef mkFPLt(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  virtual SMTExprRef mkFPGt(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  virtual SMTExprRef mkFPLe(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  virtual SMTExprRef mkFPGe(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  virtual SMTExprRef mkFPEqual(const SMTExprRef &LHS,
                               const SMTExprRef &RHS) = 0;

  virtual SMTExprRef mkFPtoFP(const SMTExprRef &From, const SMTSortRef &To) = 0;

  virtual SMTExprRef mkFPtoSBV(const SMTExprRef &From,
                               const SMTSortRef &To) = 0;

  virtual SMTExprRef mkFPtoUBV(const SMTExprRef &From,
                               const SMTSortRef &To) = 0;

  virtual SMTExprRef mkSBVtoFP(const SMTExprRef &From, unsigned ToWidth) = 0;

  virtual SMTExprRef mkUBVtoFP(const SMTExprRef &From, unsigned ToWidth) = 0;

  virtual SMTExprRef mkSymbol(const char *Name, SMTSortRef Sort) = 0;

  // Return an appropriate floating-point rounding mode.
  virtual SMTExprRef getFloatRoundingMode() = 0;

  virtual const llvm::APSInt getBitvector(const SMTExprRef &Exp) = 0;

  virtual bool getBoolean(const SMTExprRef &Exp) = 0;

  /// Construct a const SMTExprRef &From a boolean.
  virtual SMTExprRef mkBoolean(const bool b) = 0;

  /// Construct a const SMTExprRef &From a finite APFloat.
  virtual SMTExprRef mkFloat(const llvm::APFloat Float) = 0;

  /// Construct a const SMTExprRef &From an APSInt.
  virtual SMTExprRef mkBitvector(const llvm::APSInt Int, unsigned BitWidth) = 0;

  SMTExprRef mkBitvector(const llvm::APSInt Int) {
    return mkBitvector(Int, Int.getBitWidth());
  }

  /// Given an expression, extract the value of this operand in the model.
  virtual bool getInterpretation(const SMTExprRef &Exp, llvm::APSInt &Int) = 0;

  /// Given an expression extract the value of this operand in the model.
  virtual bool getInterpretation(const SMTExprRef &Exp,
                                 llvm::APFloat &Float) = 0;

  /// Construct a Z3Expr from a boolean, given a Z3_context.
  virtual SMTExprRef fromBoolean(const bool Bool) = 0;
  /// Construct a Z3Expr from a finite APFloat, given a Z3_context.
  virtual SMTExprRef fromAPFloat(const llvm::APFloat &Float) = 0;

  /// Construct a Z3Expr from an APSInt, given a Z3_context.
  virtual SMTExprRef fromAPSInt(const llvm::APSInt &Int) = 0;

  /// Construct a Z3Expr from an integer, given a Z3_context.
  virtual SMTExprRef fromInt(const char *Int, uint64_t BitWidth) = 0;

  /// Construct a const SMTExprRef &From a SymbolData, given a SMT_context.
  virtual SMTExprRef fromData(const SymbolID ID, const QualType &Ty,
                              uint64_t BitWidth) = 0;

  /// Check if the constraints are satisfiable
  virtual ConditionTruthVal check() const = 0;

  /// Push the current solver state
  virtual void push() = 0;

  /// Pop the previous solver state
  virtual void pop(unsigned NumStates = 1) = 0;

  /// Reset the solver and remove all constraints.
  virtual void reset() const = 0;

  virtual void print(raw_ostream &OS) const = 0;
};

using SMTSolverRef = std::shared_ptr<SMTSolver>;

} // namespace ento
} // namespace clang

#endif
