//===------ polly/SCEVAffinator.h - Create isl expressions from SCEVs -----===//
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

#ifndef POLLY_SCEV_AFFINATOR_H
#define POLLY_SCEV_AFFINATOR_H

#include "llvm/Analysis/ScalarEvolutionExpressions.h"

#include "isl/ctx.h"

struct isl_ctx;
struct isl_map;
struct isl_basic_map;
struct isl_id;
struct isl_set;
struct isl_union_set;
struct isl_union_map;
struct isl_space;
struct isl_ast_build;
struct isl_constraint;
struct isl_pw_aff;
struct isl_schedule;

namespace llvm {
class Region;
class ScalarEvolution;
}

namespace polly {
class Scop;
class ScopStmt;

/// Translate a SCEV to an isl_pw_aff.
struct SCEVAffinator : public llvm::SCEVVisitor<SCEVAffinator, isl_pw_aff *> {
public:
  SCEVAffinator(Scop *S);

  /// @brief Translate a SCEV to an isl_pw_aff.
  ///
  /// @param E    The expression that is translated.
  /// @param Stmt The SCoP statement surrounding @p E.
  ///
  /// @returns The isl representation of the SCEV @p E in @p Stmt.
  __isl_give isl_pw_aff *getPwAff(const llvm::SCEV *E, const ScopStmt *Stmt);

private:
  Scop *S;
  isl_ctx *Ctx;
  unsigned NumIterators;
  const llvm::Region &R;
  llvm::ScalarEvolution &SE;

  int getLoopDepth(const llvm::Loop *L);

  __isl_give isl_pw_aff *visit(const llvm::SCEV *E);
  __isl_give isl_pw_aff *visitConstant(const llvm::SCEVConstant *E);
  __isl_give isl_pw_aff *visitTruncateExpr(const llvm::SCEVTruncateExpr *E);
  __isl_give isl_pw_aff *visitZeroExtendExpr(const llvm::SCEVZeroExtendExpr *E);
  __isl_give isl_pw_aff *visitSignExtendExpr(const llvm::SCEVSignExtendExpr *E);
  __isl_give isl_pw_aff *visitAddExpr(const llvm::SCEVAddExpr *E);
  __isl_give isl_pw_aff *visitMulExpr(const llvm::SCEVMulExpr *E);
  __isl_give isl_pw_aff *visitUDivExpr(const llvm::SCEVUDivExpr *E);
  __isl_give isl_pw_aff *visitAddRecExpr(const llvm::SCEVAddRecExpr *E);
  __isl_give isl_pw_aff *visitSMaxExpr(const llvm::SCEVSMaxExpr *E);
  __isl_give isl_pw_aff *visitUMaxExpr(const llvm::SCEVUMaxExpr *E);
  __isl_give isl_pw_aff *visitUnknown(const llvm::SCEVUnknown *E);
  __isl_give isl_pw_aff *visitSDivInstruction(llvm::Instruction *SDiv);
  __isl_give isl_pw_aff *visitSRemInstruction(llvm::Instruction *SRem);

  friend struct llvm::SCEVVisitor<SCEVAffinator, isl_pw_aff *>;
};
}

#endif
