//===--------- SCEVAffinator.cpp  - Create Scops from LLVM IR -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Create a polyhedral description for a SCEV value.
//
//===----------------------------------------------------------------------===//

#include "polly/Support/SCEVAffinator.h"

#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ScopHelper.h"
#include "polly/Support/SCEVValidator.h"

#include "isl/aff.h"
#include "isl/set.h"
#include "isl/val.h"
#include "isl/local_space.h"

using namespace llvm;
using namespace polly;

SCEVAffinator::SCEVAffinator(Scop *S)
    : S(S), Ctx(S->getIslCtx()), R(S->getRegion()), SE(*S->getSE()) {}

SCEVAffinator::~SCEVAffinator() {
  for (const auto &CachedPair : CachedExpressions)
    isl_pw_aff_free(CachedPair.second);
}

__isl_give isl_pw_aff *SCEVAffinator::getPwAff(const SCEV *Expr,
                                               const ScopStmt *Stmt) {
  this->Stmt = Stmt;

  if (Stmt)
    NumIterators = Stmt->getNumIterators();
  else
    NumIterators = 0;

  S->addParams(getParamsInAffineExpr(&R, Expr, SE));

  return visit(Expr);
}

__isl_give isl_pw_aff *SCEVAffinator::visit(const SCEV *Expr) {

  auto Key = std::make_pair(Expr, Stmt);
  isl_pw_aff *PWA = CachedExpressions[Key];

  if (PWA)
    return isl_pw_aff_copy(PWA);

  // In case the scev is a valid parameter, we do not further analyze this
  // expression, but create a new parameter in the isl_pw_aff. This allows us
  // to treat subexpressions that we cannot translate into an piecewise affine
  // expression, as constant parameters of the piecewise affine expression.
  if (isl_id *Id = S->getIdForParam(Expr)) {
    isl_space *Space = isl_space_set_alloc(Ctx, 1, NumIterators);
    Space = isl_space_set_dim_id(Space, isl_dim_param, 0, Id);

    isl_set *Domain = isl_set_universe(isl_space_copy(Space));
    isl_aff *Affine = isl_aff_zero_on_domain(isl_local_space_from_space(Space));
    Affine = isl_aff_add_coefficient_si(Affine, isl_dim_param, 0, 1);

    PWA = isl_pw_aff_alloc(Domain, Affine);
    CachedExpressions[Key] = PWA;
    return isl_pw_aff_copy(PWA);
  }

  PWA = SCEVVisitor<SCEVAffinator, isl_pw_aff *>::visit(Expr);
  CachedExpressions[Key] = PWA;
  return isl_pw_aff_copy(PWA);
}

__isl_give isl_pw_aff *SCEVAffinator::visitConstant(const SCEVConstant *Expr) {
  ConstantInt *Value = Expr->getValue();
  isl_val *v;

  // LLVM does not define if an integer value is interpreted as a signed or
  // unsigned value. Hence, without further information, it is unknown how
  // this value needs to be converted to GMP. At the moment, we only support
  // signed operations. So we just interpret it as signed. Later, there are
  // two options:
  //
  // 1. We always interpret any value as signed and convert the values on
  //    demand.
  // 2. We pass down the signedness of the calculation and use it to interpret
  //    this constant correctly.
  v = isl_valFromAPInt(Ctx, Value->getValue(), /* isSigned */ true);

  isl_space *Space = isl_space_set_alloc(Ctx, 0, NumIterators);
  isl_local_space *ls = isl_local_space_from_space(Space);
  return isl_pw_aff_from_aff(isl_aff_val_on_domain(ls, v));
}

__isl_give isl_pw_aff *
SCEVAffinator::visitTruncateExpr(const SCEVTruncateExpr *Expr) {
  llvm_unreachable("SCEVTruncateExpr not yet supported");
}

__isl_give isl_pw_aff *
SCEVAffinator::visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr) {
  llvm_unreachable("SCEVZeroExtendExpr not yet supported");
}

__isl_give isl_pw_aff *
SCEVAffinator::visitSignExtendExpr(const SCEVSignExtendExpr *Expr) {
  // Assuming the value is signed, a sign extension is basically a noop.
  // TODO: Reconsider this as soon as we support unsigned values.
  return visit(Expr->getOperand());
}

__isl_give isl_pw_aff *SCEVAffinator::visitAddExpr(const SCEVAddExpr *Expr) {
  isl_pw_aff *Sum = visit(Expr->getOperand(0));

  for (int i = 1, e = Expr->getNumOperands(); i < e; ++i) {
    isl_pw_aff *NextSummand = visit(Expr->getOperand(i));
    Sum = isl_pw_aff_add(Sum, NextSummand);
  }

  // TODO: Check for NSW and NUW.

  return Sum;
}

__isl_give isl_pw_aff *SCEVAffinator::visitMulExpr(const SCEVMulExpr *Expr) {
  // Divide Expr into a constant part and the rest. Then visit both and multiply
  // the result to obtain the representation for Expr. While the second part of
  // ConstantAndLeftOverPair might still be a SCEVMulExpr we will not get to
  // this point again. The reason is that if it is a multiplication it consists
  // only of parameters and we will stop in the visit(const SCEV *) function and
  // return the isl_pw_aff for that parameter.
  auto ConstantAndLeftOverPair = extractConstantFactor(Expr, *S->getSE());
  return isl_pw_aff_mul(visit(ConstantAndLeftOverPair.first),
                        visit(ConstantAndLeftOverPair.second));
}

__isl_give isl_pw_aff *SCEVAffinator::visitUDivExpr(const SCEVUDivExpr *Expr) {
  llvm_unreachable("SCEVUDivExpr not yet supported");
}

__isl_give isl_pw_aff *
SCEVAffinator::visitAddRecExpr(const SCEVAddRecExpr *Expr) {
  assert(Expr->isAffine() && "Only affine AddRecurrences allowed");

  auto Flags = Expr->getNoWrapFlags();

  // Directly generate isl_pw_aff for Expr if 'start' is zero.
  if (Expr->getStart()->isZero()) {
    assert(S->getRegion().contains(Expr->getLoop()) &&
           "Scop does not contain the loop referenced in this AddRec");

    isl_pw_aff *Step = visit(Expr->getOperand(1));
    isl_space *Space = isl_space_set_alloc(Ctx, 0, NumIterators);
    isl_local_space *LocalSpace = isl_local_space_from_space(Space);

    int loopDimension = getLoopDepth(Expr->getLoop());

    isl_aff *LAff = isl_aff_set_coefficient_si(
        isl_aff_zero_on_domain(LocalSpace), isl_dim_in, loopDimension, 1);
    isl_pw_aff *LPwAff = isl_pw_aff_from_aff(LAff);

    // TODO: Do we need to check for NSW and NUW?
    return isl_pw_aff_mul(Step, LPwAff);
  }

  // Translate AddRecExpr from '{start, +, inc}' into 'start + {0, +, inc}'
  // if 'start' is not zero.
  // TODO: Using the original SCEV no-wrap flags is not always safe, however
  //       as our code generation is reordering the expression anyway it doesn't
  //       really matter.
  ScalarEvolution &SE = *S->getSE();
  const SCEV *ZeroStartExpr =
      SE.getAddRecExpr(SE.getConstant(Expr->getStart()->getType(), 0),
                       Expr->getStepRecurrence(SE), Expr->getLoop(), Flags);

  isl_pw_aff *ZeroStartResult = visit(ZeroStartExpr);
  isl_pw_aff *Start = visit(Expr->getStart());

  return isl_pw_aff_add(ZeroStartResult, Start);
}

__isl_give isl_pw_aff *SCEVAffinator::visitSMaxExpr(const SCEVSMaxExpr *Expr) {
  isl_pw_aff *Max = visit(Expr->getOperand(0));

  for (int i = 1, e = Expr->getNumOperands(); i < e; ++i) {
    isl_pw_aff *NextOperand = visit(Expr->getOperand(i));
    Max = isl_pw_aff_max(Max, NextOperand);
  }

  return Max;
}

__isl_give isl_pw_aff *SCEVAffinator::visitUMaxExpr(const SCEVUMaxExpr *Expr) {
  llvm_unreachable("SCEVUMaxExpr not yet supported");
}

__isl_give isl_pw_aff *SCEVAffinator::visitSDivInstruction(Instruction *SDiv) {
  assert(SDiv->getOpcode() == Instruction::SDiv && "Assumed SDiv instruction!");
  auto *SE = S->getSE();

  auto *Divisor = SDiv->getOperand(1);
  auto *DivisorSCEV = SE->getSCEV(Divisor);
  auto *DivisorPWA = visit(DivisorSCEV);
  assert(isa<ConstantInt>(Divisor) &&
         "SDiv is no parameter but has a non-constant RHS.");

  auto *Dividend = SDiv->getOperand(0);
  auto *DividendSCEV = SE->getSCEV(Dividend);
  auto *DividendPWA = visit(DividendSCEV);
  return isl_pw_aff_tdiv_q(DividendPWA, DivisorPWA);
}

__isl_give isl_pw_aff *SCEVAffinator::visitSRemInstruction(Instruction *SRem) {
  assert(SRem->getOpcode() == Instruction::SRem && "Assumed SRem instruction!");
  auto *SE = S->getSE();

  auto *Divisor = dyn_cast<ConstantInt>(SRem->getOperand(1));
  assert(Divisor && "SRem is no parameter but has a non-constant RHS.");
  auto *DivisorVal = isl_valFromAPInt(Ctx, Divisor->getValue(),
                                      /* isSigned */ true);

  auto *Dividend = SRem->getOperand(0);
  auto *DividendSCEV = SE->getSCEV(Dividend);
  auto *DividendPWA = visit(DividendSCEV);

  return isl_pw_aff_mod_val(DividendPWA, isl_val_abs(DivisorVal));
}

__isl_give isl_pw_aff *SCEVAffinator::visitUnknown(const SCEVUnknown *Expr) {
  if (Instruction *I = dyn_cast<Instruction>(Expr->getValue())) {
    switch (I->getOpcode()) {
    case Instruction::SDiv:
      return visitSDivInstruction(I);
    case Instruction::SRem:
      return visitSRemInstruction(I);
    default:
      break; // Fall through.
    }
  }

  llvm_unreachable(
      "Unknowns SCEV was neither parameter nor a valid instruction.");
}

int SCEVAffinator::getLoopDepth(const Loop *L) {
  Loop *outerLoop = S->getRegion().outermostLoopInRegion(const_cast<Loop *>(L));
  assert(outerLoop && "Scop does not contain this loop");
  return L->getLoopDepth() - outerLoop->getLoopDepth();
}
