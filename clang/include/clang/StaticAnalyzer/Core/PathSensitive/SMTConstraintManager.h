//== SMTConstraintManager.h -------------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a SMT generic API, which will be the base class for
//  every SMT solver specific class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SMTCONSTRAINTMANAGER_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SMTCONSTRAINTMANAGER_H

#include "clang/StaticAnalyzer/Core/PathSensitive/RangedConstraintManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SMTSolver.h"

namespace clang {
namespace ento {

class SMTConstraintManager : public clang::ento::SimpleConstraintManager {
  SMTSolverRef &Solver;

public:
  SMTConstraintManager(clang::ento::SubEngine *SE, clang::ento::SValBuilder &SB,
                       SMTSolverRef &S)
      : SimpleConstraintManager(SE, SB), Solver(S) {}
  virtual ~SMTConstraintManager() = default;

  //===------------------------------------------------------------------===//
  // Implementation for interface from SimpleConstraintManager.
  //===------------------------------------------------------------------===//

  ProgramStateRef assumeSym(ProgramStateRef state, SymbolRef Sym,
                            bool Assumption) override;

  ProgramStateRef assumeSymInclusiveRange(ProgramStateRef State, SymbolRef Sym,
                                          const llvm::APSInt &From,
                                          const llvm::APSInt &To,
                                          bool InRange) override;

  ProgramStateRef assumeSymUnsupported(ProgramStateRef State, SymbolRef Sym,
                                       bool Assumption) override;

  //===------------------------------------------------------------------===//
  // Implementation for interface from ConstraintManager.
  //===------------------------------------------------------------------===//

  ConditionTruthVal checkNull(ProgramStateRef State, SymbolRef Sym) override;

  const llvm::APSInt *getSymVal(ProgramStateRef State,
                                SymbolRef Sym) const override;

  /// Converts the ranged constraints of a set of symbols to SMT
  ///
  /// \param CR The set of constraints.
  void addRangeConstraints(clang::ento::ConstraintRangeTy CR);

  /// Checks if the added constraints are satisfiable
  clang::ento::ConditionTruthVal isModelFeasible();

  /// Dumps SMT formula
  LLVM_DUMP_METHOD void dump() const { Solver->dump(); }

protected:
  //===------------------------------------------------------------------===//
  // Internal implementation.
  //===------------------------------------------------------------------===//

  // Check whether a new model is satisfiable, and update the program state.
  virtual ProgramStateRef assumeExpr(ProgramStateRef State, SymbolRef Sym,
                                     const SMTExprRef &Exp) = 0;

  /// Given a program state, construct the logical conjunction and add it to
  /// the solver
  virtual void addStateConstraints(ProgramStateRef State) const = 0;

  // Generate and check a Z3 model, using the given constraint.
  ConditionTruthVal checkModel(ProgramStateRef State,
                               const SMTExprRef &Exp) const;

  // Generate a Z3Expr that represents the given symbolic expression.
  // Sets the hasComparison parameter if the expression has a comparison
  // operator.
  // Sets the RetTy parameter to the final return type after promotions and
  // casts.
  SMTExprRef getExpr(SymbolRef Sym, QualType *RetTy = nullptr,
                     bool *hasComparison = nullptr) const;

  // Generate a Z3Expr that takes the logical not of an expression.
  SMTExprRef getNotExpr(const SMTExprRef &Exp) const;

  // Generate a Z3Expr that compares the expression to zero.
  SMTExprRef getZeroExpr(const SMTExprRef &Exp, QualType RetTy,
                         bool Assumption) const;

  // Recursive implementation to unpack and generate symbolic expression.
  // Sets the hasComparison and RetTy parameters. See getZ3Expr().
  SMTExprRef getSymExpr(SymbolRef Sym, QualType *RetTy,
                        bool *hasComparison) const;

  // Wrapper to generate Z3Expr from SymbolData.
  SMTExprRef getDataExpr(const SymbolID ID, QualType Ty) const;

  // Wrapper to generate Z3Expr from SymbolCast.
  SMTExprRef getCastExpr(const SMTExprRef &Exp, QualType FromTy,
                         QualType Ty) const;

  // Wrapper to generate Z3Expr from BinarySymExpr.
  // Sets the hasComparison and RetTy parameters. See getZ3Expr().
  SMTExprRef getSymBinExpr(const BinarySymExpr *BSE, bool *hasComparison,
                           QualType *RetTy) const;

  // Wrapper to generate Z3Expr from unpacked binary symbolic expression.
  // Sets the RetTy parameter. See getZ3Expr().
  SMTExprRef getBinExpr(const SMTExprRef &LHS, QualType LTy,
                        BinaryOperator::Opcode Op, const SMTExprRef &RHS,
                        QualType RTy, QualType *RetTy) const;

  // Wrapper to generate Z3Expr from a range. If From == To, an equality will
  // be created instead.
  SMTExprRef getRangeExpr(SymbolRef Sym, const llvm::APSInt &From,
                          const llvm::APSInt &To, bool InRange);

  //===------------------------------------------------------------------===//
  // Helper functions.
  //===------------------------------------------------------------------===//

  // Recover the QualType of an APSInt.
  // TODO: Refactor to put elsewhere
  QualType getAPSIntType(const llvm::APSInt &Int) const;

  // Get the QualTy for the input APSInt, and fix it if it has a bitwidth of 1.
  std::pair<llvm::APSInt, QualType> fixAPSInt(const llvm::APSInt &Int) const;

  // Perform implicit type conversion on binary symbolic expressions.
  // May modify all input parameters.
  // TODO: Refactor to use built-in conversion functions
  void doTypeConversion(SMTExprRef &LHS, SMTExprRef &RHS, QualType &LTy,
                        QualType &RTy) const;

  // Perform implicit integer type conversion.
  // May modify all input parameters.
  // TODO: Refactor to use Sema::handleIntegerConversion()
  template <typename T, T (SMTSolver::*doCast)(const T &, QualType, uint64_t,
                                               QualType, uint64_t)>
  void doIntTypeConversion(T &LHS, QualType &LTy, T &RHS, QualType &RTy) const {
    ASTContext &Ctx = getBasicVals().getContext();
    uint64_t LBitWidth = Ctx.getTypeSize(LTy);
    uint64_t RBitWidth = Ctx.getTypeSize(RTy);

    assert(!LTy.isNull() && !RTy.isNull() && "Input type is null!");
    // Always perform integer promotion before checking type equality.
    // Otherwise, e.g. (bool) a + (bool) b could trigger a backend assertion
    if (LTy->isPromotableIntegerType()) {
      QualType NewTy = Ctx.getPromotedIntegerType(LTy);
      uint64_t NewBitWidth = Ctx.getTypeSize(NewTy);
      LHS = ((*Solver).*doCast)(LHS, NewTy, NewBitWidth, LTy, LBitWidth);
      LTy = NewTy;
      LBitWidth = NewBitWidth;
    }
    if (RTy->isPromotableIntegerType()) {
      QualType NewTy = Ctx.getPromotedIntegerType(RTy);
      uint64_t NewBitWidth = Ctx.getTypeSize(NewTy);
      RHS = ((*Solver).*doCast)(RHS, NewTy, NewBitWidth, RTy, RBitWidth);
      RTy = NewTy;
      RBitWidth = NewBitWidth;
    }

    if (LTy == RTy)
      return;

    // Perform integer type conversion
    // Note: Safe to skip updating bitwidth because this must terminate
    bool isLSignedTy = LTy->isSignedIntegerOrEnumerationType();
    bool isRSignedTy = RTy->isSignedIntegerOrEnumerationType();

    int order = Ctx.getIntegerTypeOrder(LTy, RTy);
    if (isLSignedTy == isRSignedTy) {
      // Same signedness; use the higher-ranked type
      if (order == 1) {
        RHS = ((*Solver).*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
        RTy = LTy;
      } else {
        LHS = ((*Solver).*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
        LTy = RTy;
      }
    } else if (order != (isLSignedTy ? 1 : -1)) {
      // The unsigned type has greater than or equal rank to the
      // signed type, so use the unsigned type
      if (isRSignedTy) {
        RHS = ((*Solver).*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
        RTy = LTy;
      } else {
        LHS = ((*Solver).*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
        LTy = RTy;
      }
    } else if (LBitWidth != RBitWidth) {
      // The two types are different widths; if we are here, that
      // means the signed type is larger than the unsigned type, so
      // use the signed type.
      if (isLSignedTy) {
        RHS = ((*Solver).*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
        RTy = LTy;
      } else {
        LHS = ((*Solver).*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
        LTy = RTy;
      }
    } else {
      // The signed type is higher-ranked than the unsigned type,
      // but isn't actually any bigger (like unsigned int and long
      // on most 32-bit systems).  Use the unsigned type corresponding
      // to the signed type.
      QualType NewTy =
          Ctx.getCorrespondingUnsignedType(isLSignedTy ? LTy : RTy);
      RHS = ((*Solver).*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
      RTy = NewTy;
      LHS = ((*Solver).*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
      LTy = NewTy;
    }
  }

  // Perform implicit floating-point type conversion.
  // May modify all input parameters.
  // TODO: Refactor to use Sema::handleFloatConversion()
  template <typename T, T (SMTSolver::*doCast)(const T &, QualType, uint64_t,
                                               QualType, uint64_t)>
  void doFloatTypeConversion(T &LHS, QualType &LTy, T &RHS,
                             QualType &RTy) const {
    ASTContext &Ctx = getBasicVals().getContext();

    uint64_t LBitWidth = Ctx.getTypeSize(LTy);
    uint64_t RBitWidth = Ctx.getTypeSize(RTy);

    // Perform float-point type promotion
    if (!LTy->isRealFloatingType()) {
      LHS = ((*Solver).*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
      LTy = RTy;
      LBitWidth = RBitWidth;
    }
    if (!RTy->isRealFloatingType()) {
      RHS = ((*Solver).*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
      RTy = LTy;
      RBitWidth = LBitWidth;
    }

    if (LTy == RTy)
      return;

    // If we have two real floating types, convert the smaller operand to the
    // bigger result
    // Note: Safe to skip updating bitwidth because this must terminate
    int order = Ctx.getFloatingTypeOrder(LTy, RTy);
    if (order > 0) {
      RHS = ((*Solver).*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
      RTy = LTy;
    } else if (order == 0) {
      LHS = ((*Solver).*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
      LTy = RTy;
    } else {
      llvm_unreachable("Unsupported floating-point type cast!");
    }
  }
}; // end class SMTConstraintManager

} // namespace ento
} // namespace clang

#endif
