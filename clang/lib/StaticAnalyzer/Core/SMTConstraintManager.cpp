//== SMTConstraintManager.cpp -----------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/SMTConstraintManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"

using namespace clang;
using namespace ento;

void SMTConstraintManager::addRangeConstraints(ConstraintRangeTy CR) {
  Solver->reset();

  for (const auto &I : CR) {
    SymbolRef Sym = I.first;

    SMTExprRef Constraints = Solver->fromBoolean(false);
    for (const auto &Range : I.second) {
      SMTExprRef SymRange =
          getRangeExpr(Sym, Range.From(), Range.To(), /*InRange=*/true);

      // FIXME: the last argument (isSigned) is not used when generating the
      // or expression, as both arguments are booleans
      Constraints =
          Solver->fromBinOp(Constraints, BO_LOr, SymRange, /*IsSigned=*/true);
    }
    Solver->addConstraint(Constraints);
  }
}

clang::ento::ConditionTruthVal SMTConstraintManager::isModelFeasible() {
  return Solver->check();
}

ProgramStateRef SMTConstraintManager::assumeSym(ProgramStateRef State,
                                                SymbolRef Sym,
                                                bool Assumption) {
  QualType RetTy;
  bool hasComparison;

  SMTExprRef Exp = getExpr(Sym, &RetTy, &hasComparison);
  // Create zero comparison for implicit boolean cast, with reversed assumption
  if (!hasComparison && !RetTy->isBooleanType())
    return assumeExpr(State, Sym, getZeroExpr(Exp, RetTy, !Assumption));

  return assumeExpr(State, Sym, Assumption ? Exp : getNotExpr(Exp));
}

ProgramStateRef SMTConstraintManager::assumeSymInclusiveRange(
    ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
    const llvm::APSInt &To, bool InRange) {
  return assumeExpr(State, Sym, getRangeExpr(Sym, From, To, InRange));
}

ProgramStateRef
SMTConstraintManager::assumeSymUnsupported(ProgramStateRef State, SymbolRef Sym,
                                           bool Assumption) {
  // Skip anything that is unsupported
  return State;
}

ConditionTruthVal SMTConstraintManager::checkNull(ProgramStateRef State,
                                                  SymbolRef Sym) {
  QualType RetTy;
  // The expression may be casted, so we cannot call getZ3DataExpr() directly
  SMTExprRef VarExp = getExpr(Sym, &RetTy);
  SMTExprRef Exp = getZeroExpr(VarExp, RetTy, true);

  // Negate the constraint
  SMTExprRef NotExp = getZeroExpr(VarExp, RetTy, false);

  Solver->reset();
  addStateConstraints(State);

  Solver->push();
  Solver->addConstraint(Exp);
  ConditionTruthVal isSat = Solver->check();

  Solver->pop();
  Solver->addConstraint(NotExp);
  ConditionTruthVal isNotSat = Solver->check();

  // Zero is the only possible solution
  if (isSat.isConstrainedTrue() && isNotSat.isConstrainedFalse())
    return true;

  // Zero is not a solution
  if (isSat.isConstrainedFalse() && isNotSat.isConstrainedTrue())
    return false;

  // Zero may be a solution
  return ConditionTruthVal();
}

const llvm::APSInt *SMTConstraintManager::getSymVal(ProgramStateRef State,
                                                    SymbolRef Sym) const {
  BasicValueFactory &BVF = getBasicVals();
  ASTContext &Ctx = BVF.getContext();

  if (const SymbolData *SD = dyn_cast<SymbolData>(Sym)) {
    QualType Ty = Sym->getType();
    assert(!Ty->isRealFloatingType());
    llvm::APSInt Value(Ctx.getTypeSize(Ty),
                       !Ty->isSignedIntegerOrEnumerationType());

    SMTExprRef Exp = getDataExpr(SD->getSymbolID(), Ty);

    Solver->reset();
    addStateConstraints(State);

    // Constraints are unsatisfiable
    ConditionTruthVal isSat = Solver->check();
    if (!isSat.isConstrainedTrue())
      return nullptr;

    // Model does not assign interpretation
    if (!Solver->getInterpretation(Exp, Value))
      return nullptr;

    // A value has been obtained, check if it is the only value
    SMTExprRef NotExp = Solver->fromBinOp(
        Exp, BO_NE,
        Ty->isBooleanType() ? Solver->fromBoolean(Value.getBoolValue())
                            : Solver->fromAPSInt(Value),
        false);

    Solver->addConstraint(NotExp);

    ConditionTruthVal isNotSat = Solver->check();
    if (isNotSat.isConstrainedTrue())
      return nullptr;

    // This is the only solution, store it
    return &BVF.getValue(Value);
  }

  if (const SymbolCast *SC = dyn_cast<SymbolCast>(Sym)) {
    SymbolRef CastSym = SC->getOperand();
    QualType CastTy = SC->getType();
    // Skip the void type
    if (CastTy->isVoidType())
      return nullptr;

    const llvm::APSInt *Value;
    if (!(Value = getSymVal(State, CastSym)))
      return nullptr;
    return &BVF.Convert(SC->getType(), *Value);
  }

  if (const BinarySymExpr *BSE = dyn_cast<BinarySymExpr>(Sym)) {
    const llvm::APSInt *LHS, *RHS;
    if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(BSE)) {
      LHS = getSymVal(State, SIE->getLHS());
      RHS = &SIE->getRHS();
    } else if (const IntSymExpr *ISE = dyn_cast<IntSymExpr>(BSE)) {
      LHS = &ISE->getLHS();
      RHS = getSymVal(State, ISE->getRHS());
    } else if (const SymSymExpr *SSM = dyn_cast<SymSymExpr>(BSE)) {
      // Early termination to avoid expensive call
      LHS = getSymVal(State, SSM->getLHS());
      RHS = LHS ? getSymVal(State, SSM->getRHS()) : nullptr;
    } else {
      llvm_unreachable("Unsupported binary expression to get symbol value!");
    }

    if (!LHS || !RHS)
      return nullptr;

    llvm::APSInt ConvertedLHS, ConvertedRHS;
    QualType LTy, RTy;
    std::tie(ConvertedLHS, LTy) = fixAPSInt(*LHS);
    std::tie(ConvertedRHS, RTy) = fixAPSInt(*RHS);
    doIntTypeConversion<llvm::APSInt, &SMTSolver::castAPSInt>(
        ConvertedLHS, LTy, ConvertedRHS, RTy);
    return BVF.evalAPSInt(BSE->getOpcode(), ConvertedLHS, ConvertedRHS);
  }

  llvm_unreachable("Unsupported expression to get symbol value!");
}

//===------------------------------------------------------------------===//
// Internal implementation.
//===------------------------------------------------------------------===//

ConditionTruthVal
SMTConstraintManager::checkModel(ProgramStateRef State,
                                 const SMTExprRef &Exp) const {
  Solver->reset();
  Solver->addConstraint(Exp);
  addStateConstraints(State);
  return Solver->check();
}

SMTExprRef SMTConstraintManager::getExpr(SymbolRef Sym, QualType *RetTy,
                                         bool *hasComparison) const {
  if (hasComparison) {
    *hasComparison = false;
  }

  return getSymExpr(Sym, RetTy, hasComparison);
}

SMTExprRef SMTConstraintManager::getNotExpr(const SMTExprRef &Exp) const {
  return Solver->fromUnOp(UO_LNot, Exp);
}

SMTExprRef SMTConstraintManager::getZeroExpr(const SMTExprRef &Exp, QualType Ty,
                                             bool Assumption) const {
  ASTContext &Ctx = getBasicVals().getContext();
  if (Ty->isRealFloatingType()) {
    llvm::APFloat Zero = llvm::APFloat::getZero(Ctx.getFloatTypeSemantics(Ty));
    return Solver->fromFloatBinOp(Exp, Assumption ? BO_EQ : BO_NE,
                                  Solver->fromAPFloat(Zero));
  }

  if (Ty->isIntegralOrEnumerationType() || Ty->isAnyPointerType() ||
      Ty->isBlockPointerType() || Ty->isReferenceType()) {

    // Skip explicit comparison for boolean types
    bool isSigned = Ty->isSignedIntegerOrEnumerationType();
    if (Ty->isBooleanType())
      return Assumption ? getNotExpr(Exp) : Exp;

    return Solver->fromBinOp(Exp, Assumption ? BO_EQ : BO_NE,
                             Solver->fromInt("0", Ctx.getTypeSize(Ty)),
                             isSigned);
  }

  llvm_unreachable("Unsupported type for zero value!");
}

SMTExprRef SMTConstraintManager::getSymExpr(SymbolRef Sym, QualType *RetTy,
                                            bool *hasComparison) const {
  if (const SymbolData *SD = dyn_cast<SymbolData>(Sym)) {
    if (RetTy)
      *RetTy = Sym->getType();

    return getDataExpr(SD->getSymbolID(), Sym->getType());
  }

  if (const SymbolCast *SC = dyn_cast<SymbolCast>(Sym)) {
    if (RetTy)
      *RetTy = Sym->getType();

    QualType FromTy;
    SMTExprRef Exp = getSymExpr(SC->getOperand(), &FromTy, hasComparison);
    // Casting an expression with a comparison invalidates it. Note that this
    // must occur after the recursive call above.
    // e.g. (signed char) (x > 0)
    if (hasComparison)
      *hasComparison = false;
    return getCastExpr(Exp, FromTy, Sym->getType());
  }

  if (const BinarySymExpr *BSE = dyn_cast<BinarySymExpr>(Sym)) {
    SMTExprRef Exp = getSymBinExpr(BSE, hasComparison, RetTy);
    // Set the hasComparison parameter, in post-order traversal order.
    if (hasComparison)
      *hasComparison = BinaryOperator::isComparisonOp(BSE->getOpcode());
    return Exp;
  }

  llvm_unreachable("Unsupported SymbolRef type!");
}

SMTExprRef SMTConstraintManager::getDataExpr(const SymbolID ID,
                                             QualType Ty) const {
  ASTContext &Ctx = getBasicVals().getContext();
  return Solver->fromData(ID, Ty, Ctx.getTypeSize(Ty));
}

SMTExprRef SMTConstraintManager::getCastExpr(const SMTExprRef &Exp,
                                             QualType FromTy,
                                             QualType ToTy) const {
  ASTContext &Ctx = getBasicVals().getContext();
  return Solver->fromCast(Exp, ToTy, Ctx.getTypeSize(ToTy), FromTy,
                          Ctx.getTypeSize(FromTy));
}

SMTExprRef SMTConstraintManager::getSymBinExpr(const BinarySymExpr *BSE,
                                               bool *hasComparison,
                                               QualType *RetTy) const {
  QualType LTy, RTy;
  BinaryOperator::Opcode Op = BSE->getOpcode();

  if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(BSE)) {
    SMTExprRef LHS = getSymExpr(SIE->getLHS(), &LTy, hasComparison);
    llvm::APSInt NewRInt;
    std::tie(NewRInt, RTy) = fixAPSInt(SIE->getRHS());
    SMTExprRef RHS = Solver->fromAPSInt(NewRInt);
    return getBinExpr(LHS, LTy, Op, RHS, RTy, RetTy);
  }

  if (const IntSymExpr *ISE = dyn_cast<IntSymExpr>(BSE)) {
    llvm::APSInt NewLInt;
    std::tie(NewLInt, LTy) = fixAPSInt(ISE->getLHS());
    SMTExprRef LHS = Solver->fromAPSInt(NewLInt);
    SMTExprRef RHS = getSymExpr(ISE->getRHS(), &RTy, hasComparison);
    return getBinExpr(LHS, LTy, Op, RHS, RTy, RetTy);
  }

  if (const SymSymExpr *SSM = dyn_cast<SymSymExpr>(BSE)) {
    SMTExprRef LHS = getSymExpr(SSM->getLHS(), &LTy, hasComparison);
    SMTExprRef RHS = getSymExpr(SSM->getRHS(), &RTy, hasComparison);
    return getBinExpr(LHS, LTy, Op, RHS, RTy, RetTy);
  }

  llvm_unreachable("Unsupported BinarySymExpr type!");
}

SMTExprRef SMTConstraintManager::getBinExpr(const SMTExprRef &LHS, QualType LTy,
                                            BinaryOperator::Opcode Op,
                                            const SMTExprRef &RHS, QualType RTy,
                                            QualType *RetTy) const {
  SMTExprRef NewLHS = LHS;
  SMTExprRef NewRHS = RHS;
  doTypeConversion(NewLHS, NewRHS, LTy, RTy);

  // Update the return type parameter if the output type has changed.
  if (RetTy) {
    // A boolean result can be represented as an integer type in C/C++, but at
    // this point we only care about the Z3 type. Set it as a boolean type to
    // avoid subsequent Z3 errors.
    if (BinaryOperator::isComparisonOp(Op) || BinaryOperator::isLogicalOp(Op)) {
      ASTContext &Ctx = getBasicVals().getContext();
      *RetTy = Ctx.BoolTy;
    } else {
      *RetTy = LTy;
    }

    // If the two operands are pointers and the operation is a subtraction, the
    // result is of type ptrdiff_t, which is signed
    if (LTy->isAnyPointerType() && RTy->isAnyPointerType() && Op == BO_Sub) {
      *RetTy = getBasicVals().getContext().getPointerDiffType();
    }
  }

  return LTy->isRealFloatingType()
             ? Solver->fromFloatBinOp(NewLHS, Op, NewRHS)
             : Solver->fromBinOp(NewLHS, Op, NewRHS,
                                 LTy->isSignedIntegerOrEnumerationType());
}

SMTExprRef SMTConstraintManager::getRangeExpr(SymbolRef Sym,
                                              const llvm::APSInt &From,
                                              const llvm::APSInt &To,
                                              bool InRange) {
  // Convert lower bound
  QualType FromTy;
  llvm::APSInt NewFromInt;
  std::tie(NewFromInt, FromTy) = fixAPSInt(From);
  SMTExprRef FromExp = Solver->fromAPSInt(NewFromInt);

  // Convert symbol
  QualType SymTy;
  SMTExprRef Exp = getExpr(Sym, &SymTy);

  // Construct single (in)equality
  if (From == To)
    return getBinExpr(Exp, SymTy, InRange ? BO_EQ : BO_NE, FromExp, FromTy,
                      /*RetTy=*/nullptr);

  QualType ToTy;
  llvm::APSInt NewToInt;
  std::tie(NewToInt, ToTy) = fixAPSInt(To);
  SMTExprRef ToExp = Solver->fromAPSInt(NewToInt);
  assert(FromTy == ToTy && "Range values have different types!");

  // Construct two (in)equalities, and a logical and/or
  SMTExprRef LHS = getBinExpr(Exp, SymTy, InRange ? BO_GE : BO_LT, FromExp,
                              FromTy, /*RetTy=*/nullptr);
  SMTExprRef RHS = getBinExpr(Exp, SymTy, InRange ? BO_LE : BO_GT, ToExp, ToTy,
                              /*RetTy=*/nullptr);

  return Solver->fromBinOp(LHS, InRange ? BO_LAnd : BO_LOr, RHS,
                           SymTy->isSignedIntegerOrEnumerationType());
}

//===------------------------------------------------------------------===//
// Helper functions.
//===------------------------------------------------------------------===//

QualType SMTConstraintManager::getAPSIntType(const llvm::APSInt &Int) const {
  ASTContext &Ctx = getBasicVals().getContext();
  return Ctx.getIntTypeForBitwidth(Int.getBitWidth(), Int.isSigned());
}

std::pair<llvm::APSInt, QualType>
SMTConstraintManager::fixAPSInt(const llvm::APSInt &Int) const {
  llvm::APSInt NewInt;

  // FIXME: This should be a cast from a 1-bit integer type to a boolean type,
  // but the former is not available in Clang. Instead, extend the APSInt
  // directly.
  if (Int.getBitWidth() == 1 && getAPSIntType(Int).isNull()) {
    ASTContext &Ctx = getBasicVals().getContext();
    NewInt = Int.extend(Ctx.getTypeSize(Ctx.BoolTy));
  } else
    NewInt = Int;

  return std::make_pair(NewInt, getAPSIntType(NewInt));
}

void SMTConstraintManager::doTypeConversion(SMTExprRef &LHS, SMTExprRef &RHS,
                                            QualType &LTy,
                                            QualType &RTy) const {
  assert(!LTy.isNull() && !RTy.isNull() && "Input type is null!");

  ASTContext &Ctx = getBasicVals().getContext();

  // Perform type conversion
  if ((LTy->isIntegralOrEnumerationType() &&
       RTy->isIntegralOrEnumerationType()) &&
      (LTy->isArithmeticType() && RTy->isArithmeticType())) {
    doIntTypeConversion<SMTExprRef, &SMTSolver::fromCast>(LHS, LTy, RHS, RTy);
    return;
  }

  if (LTy->isRealFloatingType() || RTy->isRealFloatingType()) {
    doFloatTypeConversion<SMTExprRef, &SMTSolver::fromCast>(LHS, LTy, RHS, RTy);
    return;
  }

  if ((LTy->isAnyPointerType() || RTy->isAnyPointerType()) ||
      (LTy->isBlockPointerType() || RTy->isBlockPointerType()) ||
      (LTy->isReferenceType() || RTy->isReferenceType())) {
    // TODO: Refactor to Sema::FindCompositePointerType(), and
    // Sema::CheckCompareOperands().

    uint64_t LBitWidth = Ctx.getTypeSize(LTy);
    uint64_t RBitWidth = Ctx.getTypeSize(RTy);

    // Cast the non-pointer type to the pointer type.
    // TODO: Be more strict about this.
    if ((LTy->isAnyPointerType() ^ RTy->isAnyPointerType()) ||
        (LTy->isBlockPointerType() ^ RTy->isBlockPointerType()) ||
        (LTy->isReferenceType() ^ RTy->isReferenceType())) {
      if (LTy->isNullPtrType() || LTy->isBlockPointerType() ||
          LTy->isReferenceType()) {
        LHS = Solver->fromCast(LHS, RTy, RBitWidth, LTy, LBitWidth);
        LTy = RTy;
      } else {
        RHS = Solver->fromCast(RHS, LTy, LBitWidth, RTy, RBitWidth);
        RTy = LTy;
      }
    }

    // Cast the void pointer type to the non-void pointer type.
    // For void types, this assumes that the casted value is equal to the value
    // of the original pointer, and does not account for alignment requirements.
    if (LTy->isVoidPointerType() ^ RTy->isVoidPointerType()) {
      assert((Ctx.getTypeSize(LTy) == Ctx.getTypeSize(RTy)) &&
             "Pointer types have different bitwidths!");
      if (RTy->isVoidPointerType())
        RTy = LTy;
      else
        LTy = RTy;
    }

    if (LTy == RTy)
      return;
  }

  // Fallback: for the solver, assume that these types don't really matter
  if ((LTy.getCanonicalType() == RTy.getCanonicalType()) ||
      (LTy->isObjCObjectPointerType() && RTy->isObjCObjectPointerType())) {
    LTy = RTy;
    return;
  }

  // TODO: Refine behavior for invalid type casts
}
