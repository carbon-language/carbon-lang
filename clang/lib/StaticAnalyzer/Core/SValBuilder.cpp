//===- SValBuilder.cpp - Basic class for all SValBuilder implementations --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines SValBuilder, the base class for all (complete) SValBuilder
//  implementations.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/SValBuilder.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/APSIntType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/BasicValueFactory.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState_Fwd.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/Store.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymExpr.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <tuple>

using namespace clang;
using namespace ento;

//===----------------------------------------------------------------------===//
// Basic SVal creation.
//===----------------------------------------------------------------------===//

void SValBuilder::anchor() {}

SValBuilder::SValBuilder(llvm::BumpPtrAllocator &alloc, ASTContext &context,
                         ProgramStateManager &stateMgr)
    : Context(context), BasicVals(context, alloc),
      SymMgr(context, BasicVals, alloc), MemMgr(context, alloc),
      StateMgr(stateMgr),
      AnOpts(
          stateMgr.getOwningEngine().getAnalysisManager().getAnalyzerOptions()),
      ArrayIndexTy(context.LongLongTy),
      ArrayIndexWidth(context.getTypeSize(ArrayIndexTy)) {}

DefinedOrUnknownSVal SValBuilder::makeZeroVal(QualType type) {
  if (Loc::isLocType(type))
    return makeNull();

  if (type->isIntegralOrEnumerationType())
    return makeIntVal(0, type);

  if (type->isArrayType() || type->isRecordType() || type->isVectorType() ||
      type->isAnyComplexType())
    return makeCompoundVal(type, BasicVals.getEmptySValList());

  // FIXME: Handle floats.
  return UnknownVal();
}

NonLoc SValBuilder::makeNonLoc(const SymExpr *lhs, BinaryOperator::Opcode op,
                                const llvm::APSInt& rhs, QualType type) {
  // The Environment ensures we always get a persistent APSInt in
  // BasicValueFactory, so we don't need to get the APSInt from
  // BasicValueFactory again.
  assert(lhs);
  assert(!Loc::isLocType(type));
  return nonloc::SymbolVal(SymMgr.getSymIntExpr(lhs, op, rhs, type));
}

NonLoc SValBuilder::makeNonLoc(const llvm::APSInt& lhs,
                               BinaryOperator::Opcode op, const SymExpr *rhs,
                               QualType type) {
  assert(rhs);
  assert(!Loc::isLocType(type));
  return nonloc::SymbolVal(SymMgr.getIntSymExpr(lhs, op, rhs, type));
}

NonLoc SValBuilder::makeNonLoc(const SymExpr *lhs, BinaryOperator::Opcode op,
                               const SymExpr *rhs, QualType type) {
  assert(lhs && rhs);
  assert(!Loc::isLocType(type));
  return nonloc::SymbolVal(SymMgr.getSymSymExpr(lhs, op, rhs, type));
}

NonLoc SValBuilder::makeNonLoc(const SymExpr *operand,
                               QualType fromTy, QualType toTy) {
  assert(operand);
  assert(!Loc::isLocType(toTy));
  return nonloc::SymbolVal(SymMgr.getCastSymbol(operand, fromTy, toTy));
}

SVal SValBuilder::convertToArrayIndex(SVal val) {
  if (val.isUnknownOrUndef())
    return val;

  // Common case: we have an appropriately sized integer.
  if (Optional<nonloc::ConcreteInt> CI = val.getAs<nonloc::ConcreteInt>()) {
    const llvm::APSInt& I = CI->getValue();
    if (I.getBitWidth() == ArrayIndexWidth && I.isSigned())
      return val;
  }

  return evalCast(val, ArrayIndexTy, QualType{});
}

nonloc::ConcreteInt SValBuilder::makeBoolVal(const CXXBoolLiteralExpr *boolean){
  return makeTruthVal(boolean->getValue());
}

DefinedOrUnknownSVal
SValBuilder::getRegionValueSymbolVal(const TypedValueRegion *region) {
  QualType T = region->getValueType();

  if (T->isNullPtrType())
    return makeZeroVal(T);

  if (!SymbolManager::canSymbolicate(T))
    return UnknownVal();

  SymbolRef sym = SymMgr.getRegionValueSymbol(region);

  if (Loc::isLocType(T))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedOrUnknownSVal SValBuilder::conjureSymbolVal(const void *SymbolTag,
                                                   const Expr *Ex,
                                                   const LocationContext *LCtx,
                                                   unsigned Count) {
  QualType T = Ex->getType();

  if (T->isNullPtrType())
    return makeZeroVal(T);

  // Compute the type of the result. If the expression is not an R-value, the
  // result should be a location.
  QualType ExType = Ex->getType();
  if (Ex->isGLValue())
    T = LCtx->getAnalysisDeclContext()->getASTContext().getPointerType(ExType);

  return conjureSymbolVal(SymbolTag, Ex, LCtx, T, Count);
}

DefinedOrUnknownSVal SValBuilder::conjureSymbolVal(const void *symbolTag,
                                                   const Expr *expr,
                                                   const LocationContext *LCtx,
                                                   QualType type,
                                                   unsigned count) {
  if (type->isNullPtrType())
    return makeZeroVal(type);

  if (!SymbolManager::canSymbolicate(type))
    return UnknownVal();

  SymbolRef sym = SymMgr.conjureSymbol(expr, LCtx, type, count, symbolTag);

  if (Loc::isLocType(type))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedOrUnknownSVal SValBuilder::conjureSymbolVal(const Stmt *stmt,
                                                   const LocationContext *LCtx,
                                                   QualType type,
                                                   unsigned visitCount) {
  if (type->isNullPtrType())
    return makeZeroVal(type);

  if (!SymbolManager::canSymbolicate(type))
    return UnknownVal();

  SymbolRef sym = SymMgr.conjureSymbol(stmt, LCtx, type, visitCount);

  if (Loc::isLocType(type))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedOrUnknownSVal
SValBuilder::getConjuredHeapSymbolVal(const Expr *E,
                                      const LocationContext *LCtx,
                                      unsigned VisitCount) {
  QualType T = E->getType();
  return getConjuredHeapSymbolVal(E, LCtx, T, VisitCount);
}

DefinedOrUnknownSVal
SValBuilder::getConjuredHeapSymbolVal(const Expr *E,
                                      const LocationContext *LCtx,
                                      QualType type, unsigned VisitCount) {
  assert(Loc::isLocType(type));
  assert(SymbolManager::canSymbolicate(type));
  if (type->isNullPtrType())
    return makeZeroVal(type);

  SymbolRef sym = SymMgr.conjureSymbol(E, LCtx, type, VisitCount);
  return loc::MemRegionVal(MemMgr.getSymbolicHeapRegion(sym));
}

DefinedSVal SValBuilder::getMetadataSymbolVal(const void *symbolTag,
                                              const MemRegion *region,
                                              const Expr *expr, QualType type,
                                              const LocationContext *LCtx,
                                              unsigned count) {
  assert(SymbolManager::canSymbolicate(type) && "Invalid metadata symbol type");

  SymbolRef sym =
      SymMgr.getMetadataSymbol(region, expr, type, LCtx, count, symbolTag);

  if (Loc::isLocType(type))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedOrUnknownSVal
SValBuilder::getDerivedRegionValueSymbolVal(SymbolRef parentSymbol,
                                             const TypedValueRegion *region) {
  QualType T = region->getValueType();

  if (T->isNullPtrType())
    return makeZeroVal(T);

  if (!SymbolManager::canSymbolicate(T))
    return UnknownVal();

  SymbolRef sym = SymMgr.getDerivedSymbol(parentSymbol, region);

  if (Loc::isLocType(T))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedSVal SValBuilder::getMemberPointer(const NamedDecl *ND) {
  assert(!ND || (isa<CXXMethodDecl, FieldDecl, IndirectFieldDecl>(ND)));

  if (const auto *MD = dyn_cast_or_null<CXXMethodDecl>(ND)) {
    // Sema treats pointers to static member functions as have function pointer
    // type, so return a function pointer for the method.
    // We don't need to play a similar trick for static member fields
    // because these are represented as plain VarDecls and not FieldDecls
    // in the AST.
    if (MD->isStatic())
      return getFunctionPointer(MD);
  }

  return nonloc::PointerToMember(ND);
}

DefinedSVal SValBuilder::getFunctionPointer(const FunctionDecl *func) {
  return loc::MemRegionVal(MemMgr.getFunctionCodeRegion(func));
}

DefinedSVal SValBuilder::getBlockPointer(const BlockDecl *block,
                                         CanQualType locTy,
                                         const LocationContext *locContext,
                                         unsigned blockCount) {
  const BlockCodeRegion *BC =
    MemMgr.getBlockCodeRegion(block, locTy, locContext->getAnalysisDeclContext());
  const BlockDataRegion *BD = MemMgr.getBlockDataRegion(BC, locContext,
                                                        blockCount);
  return loc::MemRegionVal(BD);
}

Optional<loc::MemRegionVal>
SValBuilder::getCastedMemRegionVal(const MemRegion *R, QualType Ty) {
  if (auto OptR = StateMgr.getStoreManager().castRegion(R, Ty))
    return loc::MemRegionVal(*OptR);
  return None;
}

/// Return a memory region for the 'this' object reference.
loc::MemRegionVal SValBuilder::getCXXThis(const CXXMethodDecl *D,
                                          const StackFrameContext *SFC) {
  return loc::MemRegionVal(
      getRegionManager().getCXXThisRegion(D->getThisType(), SFC));
}

/// Return a memory region for the 'this' object reference.
loc::MemRegionVal SValBuilder::getCXXThis(const CXXRecordDecl *D,
                                          const StackFrameContext *SFC) {
  const Type *T = D->getTypeForDecl();
  QualType PT = getContext().getPointerType(QualType(T, 0));
  return loc::MemRegionVal(getRegionManager().getCXXThisRegion(PT, SFC));
}

Optional<SVal> SValBuilder::getConstantVal(const Expr *E) {
  E = E->IgnoreParens();

  switch (E->getStmtClass()) {
  // Handle expressions that we treat differently from the AST's constant
  // evaluator.
  case Stmt::AddrLabelExprClass:
    return makeLoc(cast<AddrLabelExpr>(E));

  case Stmt::CXXScalarValueInitExprClass:
  case Stmt::ImplicitValueInitExprClass:
    return makeZeroVal(E->getType());

  case Stmt::ObjCStringLiteralClass: {
    const auto *SL = cast<ObjCStringLiteral>(E);
    return makeLoc(getRegionManager().getObjCStringRegion(SL));
  }

  case Stmt::StringLiteralClass: {
    const auto *SL = cast<StringLiteral>(E);
    return makeLoc(getRegionManager().getStringRegion(SL));
  }

  case Stmt::PredefinedExprClass: {
    const auto *PE = cast<PredefinedExpr>(E);
    assert(PE->getFunctionName() &&
           "Since we analyze only instantiated functions, PredefinedExpr "
           "should have a function name.");
    return makeLoc(getRegionManager().getStringRegion(PE->getFunctionName()));
  }

  // Fast-path some expressions to avoid the overhead of going through the AST's
  // constant evaluator
  case Stmt::CharacterLiteralClass: {
    const auto *C = cast<CharacterLiteral>(E);
    return makeIntVal(C->getValue(), C->getType());
  }

  case Stmt::CXXBoolLiteralExprClass:
    return makeBoolVal(cast<CXXBoolLiteralExpr>(E));

  case Stmt::TypeTraitExprClass: {
    const auto *TE = cast<TypeTraitExpr>(E);
    return makeTruthVal(TE->getValue(), TE->getType());
  }

  case Stmt::IntegerLiteralClass:
    return makeIntVal(cast<IntegerLiteral>(E));

  case Stmt::ObjCBoolLiteralExprClass:
    return makeBoolVal(cast<ObjCBoolLiteralExpr>(E));

  case Stmt::CXXNullPtrLiteralExprClass:
    return makeNull();

  case Stmt::CStyleCastExprClass:
  case Stmt::CXXFunctionalCastExprClass:
  case Stmt::CXXConstCastExprClass:
  case Stmt::CXXReinterpretCastExprClass:
  case Stmt::CXXStaticCastExprClass:
  case Stmt::ImplicitCastExprClass: {
    const auto *CE = cast<CastExpr>(E);
    switch (CE->getCastKind()) {
    default:
      break;
    case CK_ArrayToPointerDecay:
    case CK_IntegralToPointer:
    case CK_NoOp:
    case CK_BitCast: {
      const Expr *SE = CE->getSubExpr();
      Optional<SVal> Val = getConstantVal(SE);
      if (!Val)
        return None;
      return evalCast(*Val, CE->getType(), SE->getType());
    }
    }
    // FALLTHROUGH
    LLVM_FALLTHROUGH;
  }

  // If we don't have a special case, fall back to the AST's constant evaluator.
  default: {
    // Don't try to come up with a value for materialized temporaries.
    if (E->isGLValue())
      return None;

    ASTContext &Ctx = getContext();
    Expr::EvalResult Result;
    if (E->EvaluateAsInt(Result, Ctx))
      return makeIntVal(Result.Val.getInt());

    if (Loc::isLocType(E->getType()))
      if (E->isNullPointerConstant(Ctx, Expr::NPC_ValueDependentIsNotNull))
        return makeNull();

    return None;
  }
  }
}

SVal SValBuilder::makeSymExprValNN(BinaryOperator::Opcode Op,
                                   NonLoc LHS, NonLoc RHS,
                                   QualType ResultTy) {
  SymbolRef symLHS = LHS.getAsSymbol();
  SymbolRef symRHS = RHS.getAsSymbol();

  // TODO: When the Max Complexity is reached, we should conjure a symbol
  // instead of generating an Unknown value and propagate the taint info to it.
  const unsigned MaxComp = AnOpts.MaxSymbolComplexity;

  if (symLHS && symRHS &&
      (symLHS->computeComplexity() + symRHS->computeComplexity()) <  MaxComp)
    return makeNonLoc(symLHS, Op, symRHS, ResultTy);

  if (symLHS && symLHS->computeComplexity() < MaxComp)
    if (Optional<nonloc::ConcreteInt> rInt = RHS.getAs<nonloc::ConcreteInt>())
      return makeNonLoc(symLHS, Op, rInt->getValue(), ResultTy);

  if (symRHS && symRHS->computeComplexity() < MaxComp)
    if (Optional<nonloc::ConcreteInt> lInt = LHS.getAs<nonloc::ConcreteInt>())
      return makeNonLoc(lInt->getValue(), Op, symRHS, ResultTy);

  return UnknownVal();
}

SVal SValBuilder::evalBinOp(ProgramStateRef state, BinaryOperator::Opcode op,
                            SVal lhs, SVal rhs, QualType type) {
  if (lhs.isUndef() || rhs.isUndef())
    return UndefinedVal();

  if (lhs.isUnknown() || rhs.isUnknown())
    return UnknownVal();

  if (lhs.getAs<nonloc::LazyCompoundVal>() ||
      rhs.getAs<nonloc::LazyCompoundVal>()) {
    return UnknownVal();
  }

  if (op == BinaryOperatorKind::BO_Cmp) {
    // We can't reason about C++20 spaceship operator yet.
    //
    // FIXME: Support C++20 spaceship operator.
    //        The main problem here is that the result is not integer.
    return UnknownVal();
  }

  if (Optional<Loc> LV = lhs.getAs<Loc>()) {
    if (Optional<Loc> RV = rhs.getAs<Loc>())
      return evalBinOpLL(state, op, *LV, *RV, type);

    return evalBinOpLN(state, op, *LV, rhs.castAs<NonLoc>(), type);
  }

  if (const Optional<Loc> RV = rhs.getAs<Loc>()) {
    const auto IsCommutative = [](BinaryOperatorKind Op) {
      return Op == BO_Mul || Op == BO_Add || Op == BO_And || Op == BO_Xor ||
             Op == BO_Or;
    };

    if (IsCommutative(op)) {
      // Swap operands.
      return evalBinOpLN(state, op, *RV, lhs.castAs<NonLoc>(), type);
    }

    // If the right operand is a concrete int location then we have nothing
    // better but to treat it as a simple nonloc.
    if (auto RV = rhs.getAs<loc::ConcreteInt>()) {
      const nonloc::ConcreteInt RhsAsLoc = makeIntVal(RV->getValue());
      return evalBinOpNN(state, op, lhs.castAs<NonLoc>(), RhsAsLoc, type);
    }
  }

  return evalBinOpNN(state, op, lhs.castAs<NonLoc>(), rhs.castAs<NonLoc>(),
                     type);
}

ConditionTruthVal SValBuilder::areEqual(ProgramStateRef state, SVal lhs,
                                        SVal rhs) {
  return state->isNonNull(evalEQ(state, lhs, rhs));
}

SVal SValBuilder::evalEQ(ProgramStateRef state, SVal lhs, SVal rhs) {
  return evalBinOp(state, BO_EQ, lhs, rhs, getConditionType());
}

DefinedOrUnknownSVal SValBuilder::evalEQ(ProgramStateRef state,
                                         DefinedOrUnknownSVal lhs,
                                         DefinedOrUnknownSVal rhs) {
  return evalEQ(state, static_cast<SVal>(lhs), static_cast<SVal>(rhs))
      .castAs<DefinedOrUnknownSVal>();
}

/// Recursively check if the pointer types are equal modulo const, volatile,
/// and restrict qualifiers. Also, assume that all types are similar to 'void'.
/// Assumes the input types are canonical.
static bool shouldBeModeledWithNoOp(ASTContext &Context, QualType ToTy,
                                                         QualType FromTy) {
  while (Context.UnwrapSimilarTypes(ToTy, FromTy)) {
    Qualifiers Quals1, Quals2;
    ToTy = Context.getUnqualifiedArrayType(ToTy, Quals1);
    FromTy = Context.getUnqualifiedArrayType(FromTy, Quals2);

    // Make sure that non-cvr-qualifiers the other qualifiers (e.g., address
    // spaces) are identical.
    Quals1.removeCVRQualifiers();
    Quals2.removeCVRQualifiers();
    if (Quals1 != Quals2)
      return false;
  }

  // If we are casting to void, the 'From' value can be used to represent the
  // 'To' value.
  //
  // FIXME: Doing this after unwrapping the types doesn't make any sense. A
  // cast from 'int**' to 'void**' is not special in the way that a cast from
  // 'int*' to 'void*' is.
  if (ToTy->isVoidType())
    return true;

  if (ToTy != FromTy)
    return false;

  return true;
}

// Handles casts of type CK_IntegralCast.
// At the moment, this function will redirect to evalCast, except when the range
// of the original value is known to be greater than the max of the target type.
SVal SValBuilder::evalIntegralCast(ProgramStateRef state, SVal val,
                                   QualType castTy, QualType originalTy) {
  // No truncations if target type is big enough.
  if (getContext().getTypeSize(castTy) >= getContext().getTypeSize(originalTy))
    return evalCast(val, castTy, originalTy);

  SymbolRef se = val.getAsSymbol();
  if (!se) // Let evalCast handle non symbolic expressions.
    return evalCast(val, castTy, originalTy);

  // Find the maximum value of the target type.
  APSIntType ToType(getContext().getTypeSize(castTy),
                    castTy->isUnsignedIntegerType());
  llvm::APSInt ToTypeMax = ToType.getMaxValue();
  NonLoc ToTypeMaxVal =
      makeIntVal(ToTypeMax.isUnsigned() ? ToTypeMax.getZExtValue()
                                        : ToTypeMax.getSExtValue(),
                 castTy)
          .castAs<NonLoc>();
  // Check the range of the symbol being casted against the maximum value of the
  // target type.
  NonLoc FromVal = val.castAs<NonLoc>();
  QualType CmpTy = getConditionType();
  NonLoc CompVal =
      evalBinOpNN(state, BO_LE, FromVal, ToTypeMaxVal, CmpTy).castAs<NonLoc>();
  ProgramStateRef IsNotTruncated, IsTruncated;
  std::tie(IsNotTruncated, IsTruncated) = state->assume(CompVal);
  if (!IsNotTruncated && IsTruncated) {
    // Symbol is truncated so we evaluate it as a cast.
    NonLoc CastVal = makeNonLoc(se, originalTy, castTy);
    return CastVal;
  }
  return evalCast(val, castTy, originalTy);
}

//===----------------------------------------------------------------------===//
// Cast methods.
// `evalCast` is the main method
// `evalCastKind` and `evalCastSubKind` are helpers
//===----------------------------------------------------------------------===//

/// Cast a given SVal to another SVal using given QualType's.
/// \param V -- SVal that should be casted.
/// \param CastTy -- QualType that V should be casted according to.
/// \param OriginalTy -- QualType which is associated to V. It provides
/// additional information about what type the cast performs from.
/// \returns the most appropriate casted SVal.
/// Note: Many cases don't use an exact OriginalTy. It can be extracted
/// from SVal or the cast can performs unconditionaly. Always pass OriginalTy!
/// It can be crucial in certain cases and generates different results.
/// FIXME: If `OriginalTy.isNull()` is true, then cast performs based on CastTy
/// only. This behavior is uncertain and should be improved.
SVal SValBuilder::evalCast(SVal V, QualType CastTy, QualType OriginalTy) {
  if (CastTy.isNull())
    return V;

  CastTy = Context.getCanonicalType(CastTy);

  const bool IsUnknownOriginalType = OriginalTy.isNull();
  if (!IsUnknownOriginalType) {
    OriginalTy = Context.getCanonicalType(OriginalTy);

    if (CastTy == OriginalTy)
      return V;

    // FIXME: Move this check to the most appropriate
    // evalCastKind/evalCastSubKind function. For const casts, casts to void,
    // just propagate the value.
    if (!CastTy->isVariableArrayType() && !OriginalTy->isVariableArrayType())
      if (shouldBeModeledWithNoOp(Context, Context.getPointerType(CastTy),
                                  Context.getPointerType(OriginalTy)))
        return V;
  }

  // Cast SVal according to kinds.
  switch (V.getBaseKind()) {
  case SVal::UndefinedValKind:
    return evalCastKind(V.castAs<UndefinedVal>(), CastTy, OriginalTy);
  case SVal::UnknownValKind:
    return evalCastKind(V.castAs<UnknownVal>(), CastTy, OriginalTy);
  case SVal::LocKind:
    return evalCastKind(V.castAs<Loc>(), CastTy, OriginalTy);
  case SVal::NonLocKind:
    return evalCastKind(V.castAs<NonLoc>(), CastTy, OriginalTy);
  }

  llvm_unreachable("Unknown SVal kind");
}

SVal SValBuilder::evalCastKind(UndefinedVal V, QualType CastTy,
                               QualType OriginalTy) {
  return V;
}

SVal SValBuilder::evalCastKind(UnknownVal V, QualType CastTy,
                               QualType OriginalTy) {
  return V;
}

SVal SValBuilder::evalCastKind(Loc V, QualType CastTy, QualType OriginalTy) {
  switch (V.getSubKind()) {
  case loc::ConcreteIntKind:
    return evalCastSubKind(V.castAs<loc::ConcreteInt>(), CastTy, OriginalTy);
  case loc::GotoLabelKind:
    return evalCastSubKind(V.castAs<loc::GotoLabel>(), CastTy, OriginalTy);
  case loc::MemRegionValKind:
    return evalCastSubKind(V.castAs<loc::MemRegionVal>(), CastTy, OriginalTy);
  }

  llvm_unreachable("Unknown SVal kind");
}

SVal SValBuilder::evalCastKind(NonLoc V, QualType CastTy, QualType OriginalTy) {
  switch (V.getSubKind()) {
  case nonloc::CompoundValKind:
    return evalCastSubKind(V.castAs<nonloc::CompoundVal>(), CastTy, OriginalTy);
  case nonloc::ConcreteIntKind:
    return evalCastSubKind(V.castAs<nonloc::ConcreteInt>(), CastTy, OriginalTy);
  case nonloc::LazyCompoundValKind:
    return evalCastSubKind(V.castAs<nonloc::LazyCompoundVal>(), CastTy,
                           OriginalTy);
  case nonloc::LocAsIntegerKind:
    return evalCastSubKind(V.castAs<nonloc::LocAsInteger>(), CastTy,
                           OriginalTy);
  case nonloc::SymbolValKind:
    return evalCastSubKind(V.castAs<nonloc::SymbolVal>(), CastTy, OriginalTy);
  case nonloc::PointerToMemberKind:
    return evalCastSubKind(V.castAs<nonloc::PointerToMember>(), CastTy,
                           OriginalTy);
  }

  llvm_unreachable("Unknown SVal kind");
}

SVal SValBuilder::evalCastSubKind(loc::ConcreteInt V, QualType CastTy,
                                  QualType OriginalTy) {
  // Pointer to bool.
  if (CastTy->isBooleanType())
    return makeTruthVal(V.getValue().getBoolValue(), CastTy);

  // Pointer to integer.
  if (CastTy->isIntegralOrEnumerationType()) {
    llvm::APSInt Value = V.getValue();
    BasicVals.getAPSIntType(CastTy).apply(Value);
    return makeIntVal(Value);
  }

  // Pointer to any pointer.
  if (Loc::isLocType(CastTy))
    return V;

  // Pointer to whatever else.
  return UnknownVal();
}

SVal SValBuilder::evalCastSubKind(loc::GotoLabel V, QualType CastTy,
                                  QualType OriginalTy) {
  // Pointer to bool.
  if (CastTy->isBooleanType())
    // Labels are always true.
    return makeTruthVal(true, CastTy);

  // Pointer to integer.
  if (CastTy->isIntegralOrEnumerationType()) {
    const unsigned BitWidth = Context.getIntWidth(CastTy);
    return makeLocAsInteger(V, BitWidth);
  }

  const bool IsUnknownOriginalType = OriginalTy.isNull();
  if (!IsUnknownOriginalType) {
    // Array to pointer.
    if (isa<ArrayType>(OriginalTy))
      if (CastTy->isPointerType() || CastTy->isReferenceType())
        return UnknownVal();
  }

  // Pointer to any pointer.
  if (Loc::isLocType(CastTy))
    return V;

  // Pointer to whatever else.
  return UnknownVal();
}

static bool hasSameUnqualifiedPointeeType(QualType ty1, QualType ty2) {
  return ty1->getPointeeType().getCanonicalType().getTypePtr() ==
         ty2->getPointeeType().getCanonicalType().getTypePtr();
}

SVal SValBuilder::evalCastSubKind(loc::MemRegionVal V, QualType CastTy,
                                  QualType OriginalTy) {
  // Pointer to bool.
  if (CastTy->isBooleanType()) {
    const MemRegion *R = V.getRegion();
    if (const FunctionCodeRegion *FTR = dyn_cast<FunctionCodeRegion>(R))
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(FTR->getDecl()))
        if (FD->isWeak())
          // FIXME: Currently we are using an extent symbol here,
          // because there are no generic region address metadata
          // symbols to use, only content metadata.
          return nonloc::SymbolVal(SymMgr.getExtentSymbol(FTR));

    if (const SymbolicRegion *SymR = R->getSymbolicBase()) {
      SymbolRef Sym = SymR->getSymbol();
      QualType Ty = Sym->getType();
      // This change is needed for architectures with varying
      // pointer widths. See the amdgcn opencl reproducer with
      // this change as an example: solver-sym-simplification-ptr-bool.cl
      // FIXME: Cleanup remainder of `getZeroWithPtrWidth ()`
      //        and `getIntWithPtrWidth()` functions to prevent future
      //        confusion
      if (!Ty->isReferenceType())
        return makeNonLoc(Sym, BO_NE, BasicVals.getZeroWithTypeSize(Ty),
                          CastTy);
    }
    // Non-symbolic memory regions are always true.
    return makeTruthVal(true, CastTy);
  }

  const bool IsUnknownOriginalType = OriginalTy.isNull();
  // Try to cast to array
  const auto *ArrayTy =
      IsUnknownOriginalType
          ? nullptr
          : dyn_cast<ArrayType>(OriginalTy.getCanonicalType());

  // Pointer to integer.
  if (CastTy->isIntegralOrEnumerationType()) {
    SVal Val = V;
    // Array to integer.
    if (ArrayTy) {
      // We will always decay to a pointer.
      QualType ElemTy = ArrayTy->getElementType();
      Val = StateMgr.ArrayToPointer(V, ElemTy);
      // FIXME: Keep these here for now in case we decide soon that we
      // need the original decayed type.
      //    QualType elemTy = cast<ArrayType>(originalTy)->getElementType();
      //    QualType pointerTy = C.getPointerType(elemTy);
    }
    const unsigned BitWidth = Context.getIntWidth(CastTy);
    return makeLocAsInteger(Val.castAs<Loc>(), BitWidth);
  }

  // Pointer to pointer.
  if (Loc::isLocType(CastTy)) {

    if (IsUnknownOriginalType) {
      // When retrieving symbolic pointer and expecting a non-void pointer,
      // wrap them into element regions of the expected type if necessary.
      // It is necessary to make sure that the retrieved value makes sense,
      // because there's no other cast in the AST that would tell us to cast
      // it to the correct pointer type. We might need to do that for non-void
      // pointers as well.
      // FIXME: We really need a single good function to perform casts for us
      // correctly every time we need it.
      const MemRegion *R = V.getRegion();
      if (CastTy->isPointerType() && !CastTy->isVoidPointerType()) {
        if (const auto *SR = dyn_cast<SymbolicRegion>(R)) {
          QualType SRTy = SR->getSymbol()->getType();
          if (!hasSameUnqualifiedPointeeType(SRTy, CastTy)) {
            if (auto OptMemRegV = getCastedMemRegionVal(SR, CastTy))
              return *OptMemRegV;
          }
        }
      }
      // Next fixes pointer dereference using type different from its initial
      // one. See PR37503 and PR49007 for details.
      if (const auto *ER = dyn_cast<ElementRegion>(R)) {
        if (auto OptMemRegV = getCastedMemRegionVal(ER, CastTy))
          return *OptMemRegV;
      }

      return V;
    }

    if (OriginalTy->isIntegralOrEnumerationType() ||
        OriginalTy->isBlockPointerType() || OriginalTy->isFunctionPointerType())
      return V;

    // Array to pointer.
    if (ArrayTy) {
      // Are we casting from an array to a pointer?  If so just pass on
      // the decayed value.
      if (CastTy->isPointerType() || CastTy->isReferenceType()) {
        // We will always decay to a pointer.
        QualType ElemTy = ArrayTy->getElementType();
        return StateMgr.ArrayToPointer(V, ElemTy);
      }
      // Are we casting from an array to an integer?  If so, cast the decayed
      // pointer value to an integer.
      assert(CastTy->isIntegralOrEnumerationType());
    }

    // Other pointer to pointer.
    assert(Loc::isLocType(OriginalTy) || OriginalTy->isFunctionType() ||
           CastTy->isReferenceType());

    // We get a symbolic function pointer for a dereference of a function
    // pointer, but it is of function type. Example:

    //  struct FPRec {
    //    void (*my_func)(int * x);
    //  };
    //
    //  int bar(int x);
    //
    //  int f1_a(struct FPRec* foo) {
    //    int x;
    //    (*foo->my_func)(&x);
    //    return bar(x)+1; // no-warning
    //  }

    // Get the result of casting a region to a different type.
    const MemRegion *R = V.getRegion();
    if (auto OptMemRegV = getCastedMemRegionVal(R, CastTy))
      return *OptMemRegV;
  }

  // Pointer to whatever else.
  // FIXME: There can be gross cases where one casts the result of a
  // function (that returns a pointer) to some other value that happens to
  // fit within that pointer value.  We currently have no good way to model
  // such operations.  When this happens, the underlying operation is that
  // the caller is reasoning about bits.  Conceptually we are layering a
  // "view" of a location on top of those bits.  Perhaps we need to be more
  // lazy about mutual possible views, even on an SVal?  This may be
  // necessary for bit-level reasoning as well.
  return UnknownVal();
}

SVal SValBuilder::evalCastSubKind(nonloc::CompoundVal V, QualType CastTy,
                                  QualType OriginalTy) {
  // Compound to whatever.
  return UnknownVal();
}

SVal SValBuilder::evalCastSubKind(nonloc::ConcreteInt V, QualType CastTy,
                                  QualType OriginalTy) {
  auto CastedValue = [V, CastTy, this]() {
    llvm::APSInt Value = V.getValue();
    BasicVals.getAPSIntType(CastTy).apply(Value);
    return Value;
  };

  // Integer to bool.
  if (CastTy->isBooleanType())
    return makeTruthVal(V.getValue().getBoolValue(), CastTy);

  // Integer to pointer.
  if (CastTy->isIntegralOrEnumerationType())
    return makeIntVal(CastedValue());

  // Integer to pointer.
  if (Loc::isLocType(CastTy))
    return makeIntLocVal(CastedValue());

  // Pointer to whatever else.
  return UnknownVal();
}

SVal SValBuilder::evalCastSubKind(nonloc::LazyCompoundVal V, QualType CastTy,
                                  QualType OriginalTy) {
  // Compound to whatever.
  return UnknownVal();
}

SVal SValBuilder::evalCastSubKind(nonloc::LocAsInteger V, QualType CastTy,
                                  QualType OriginalTy) {
  Loc L = V.getLoc();

  // Pointer as integer to bool.
  if (CastTy->isBooleanType())
    // Pass to Loc function.
    return evalCastKind(L, CastTy, OriginalTy);

  const bool IsUnknownOriginalType = OriginalTy.isNull();
  // Pointer as integer to pointer.
  if (!IsUnknownOriginalType && Loc::isLocType(CastTy) &&
      OriginalTy->isIntegralOrEnumerationType()) {
    if (const MemRegion *R = L.getAsRegion())
      if (auto OptMemRegV = getCastedMemRegionVal(R, CastTy))
        return *OptMemRegV;
    return L;
  }

  // Pointer as integer with region to integer/pointer.
  const MemRegion *R = L.getAsRegion();
  if (!IsUnknownOriginalType && R) {
    if (CastTy->isIntegralOrEnumerationType())
      return evalCastSubKind(loc::MemRegionVal(R), CastTy, OriginalTy);

    if (Loc::isLocType(CastTy)) {
      assert(Loc::isLocType(OriginalTy) || OriginalTy->isFunctionType() ||
             CastTy->isReferenceType());
      // Delegate to store manager to get the result of casting a region to a
      // different type. If the MemRegion* returned is NULL, this expression
      // Evaluates to UnknownVal.
      if (auto OptMemRegV = getCastedMemRegionVal(R, CastTy))
        return *OptMemRegV;
    }
  } else {
    if (Loc::isLocType(CastTy)) {
      if (IsUnknownOriginalType)
        return evalCastSubKind(loc::MemRegionVal(R), CastTy, OriginalTy);
      return L;
    }

    SymbolRef SE = nullptr;
    if (R) {
      if (const SymbolicRegion *SR =
              dyn_cast<SymbolicRegion>(R->StripCasts())) {
        SE = SR->getSymbol();
      }
    }

    if (!CastTy->isFloatingType() || !SE || SE->getType()->isFloatingType()) {
      // FIXME: Correctly support promotions/truncations.
      const unsigned CastSize = Context.getIntWidth(CastTy);
      if (CastSize == V.getNumBits())
        return V;

      return makeLocAsInteger(L, CastSize);
    }
  }

  // Pointer as integer to whatever else.
  return UnknownVal();
}

SVal SValBuilder::evalCastSubKind(nonloc::SymbolVal V, QualType CastTy,
                                  QualType OriginalTy) {
  SymbolRef SE = V.getSymbol();

  const bool IsUnknownOriginalType = OriginalTy.isNull();
  // Symbol to bool.
  if (!IsUnknownOriginalType && CastTy->isBooleanType()) {
    // Non-float to bool.
    if (Loc::isLocType(OriginalTy) ||
        OriginalTy->isIntegralOrEnumerationType() ||
        OriginalTy->isMemberPointerType()) {
      BasicValueFactory &BVF = getBasicValueFactory();
      return makeNonLoc(SE, BO_NE, BVF.getValue(0, SE->getType()), CastTy);
    }
  } else {
    // Symbol to integer, float.
    QualType T = Context.getCanonicalType(SE->getType());
    // If types are the same or both are integers, ignore the cast.
    // FIXME: Remove this hack when we support symbolic truncation/extension.
    // HACK: If both castTy and T are integers, ignore the cast.  This is
    // not a permanent solution.  Eventually we want to precisely handle
    // extension/truncation of symbolic integers.  This prevents us from losing
    // precision when we assign 'x = y' and 'y' is symbolic and x and y are
    // different integer types.
    if (haveSameType(T, CastTy))
      return V;
    if (!Loc::isLocType(CastTy))
      if (!IsUnknownOriginalType || !CastTy->isFloatingType() ||
          T->isFloatingType())
        return makeNonLoc(SE, T, CastTy);
  }

  // Symbol to pointer and whatever else.
  return UnknownVal();
}

SVal SValBuilder::evalCastSubKind(nonloc::PointerToMember V, QualType CastTy,
                                  QualType OriginalTy) {
  // Member pointer to whatever.
  return V;
}
