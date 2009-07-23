//==- CheckSecuritySyntaxOnly.cpp - Basic security checks --------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a set of flow-insensitive security checks.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/LocalCheckers.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {
class VISIBILITY_HIDDEN WalkAST : public StmtVisitor<WalkAST> {
  BugReporter &BR;
public:
  WalkAST(BugReporter &br) : BR(br) {}
  
  // Statement visitor methods.
  void VisitForStmt(ForStmt *S);
  void VisitStmt(Stmt *S) { VisitChildren(S); }

  void VisitChildren(Stmt *S);
  
  // Checker-specific methods.
  void CheckLoopConditionForFloat(const ForStmt *FS);
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// AST walking.
//===----------------------------------------------------------------------===//

void WalkAST::VisitChildren(Stmt *S) {
  for (Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I!=E; ++I)
    if (Stmt *child = *I)
      Visit(child);
}

void WalkAST::VisitForStmt(ForStmt *FS) {
  CheckLoopConditionForFloat(FS);  

  // Recurse and check children.
  VisitChildren(FS);
}

//===----------------------------------------------------------------------===//
// Check: floating poing variable used as loop counter.
//===----------------------------------------------------------------------===//

static const DeclRefExpr*
GetIncrementedVar(const Expr *expr, const VarDecl *x, const VarDecl *y) {
  expr = expr->IgnoreParenCasts();
  
  if (const BinaryOperator *B = dyn_cast<BinaryOperator>(expr)) {      
    if (!(B->isAssignmentOp() || B->isCompoundAssignmentOp() ||
          B->getOpcode() == BinaryOperator::Comma))
      return NULL;
      
    if (const DeclRefExpr *lhs = GetIncrementedVar(B->getLHS(), x, y))
      return lhs;
      
    if (const DeclRefExpr *rhs = GetIncrementedVar(B->getRHS(), x, y))
      return rhs;
      
    return NULL;
  }
    
  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(expr)) {
    const NamedDecl *ND = DR->getDecl();
    return ND == x || ND == y ? DR : NULL;
  }
   
  if (const UnaryOperator *U = dyn_cast<UnaryOperator>(expr))
    return U->isIncrementDecrementOp()
      ? GetIncrementedVar(U->getSubExpr(), x, y) : NULL;

  return NULL;
}

/// CheckLoopConditionForFloat - This check looks for 'for' statements that
///  use a floating point variable as a loop counter.
///  CERT: FLP30-C, FLP30-CPP.
///
void WalkAST::CheckLoopConditionForFloat(const ForStmt *FS) {
  // Does the loop have a condition?
  const Expr *condition = FS->getCond();
  
  if (!condition)
    return;

  // Does the loop have an increment?
  const Expr *increment = FS->getInc();
  
  if (!increment)
    return;
    
  // Strip away '()' and casts.
  condition = condition->IgnoreParenCasts();
  increment = increment->IgnoreParenCasts();
  
  // Is the loop condition a comparison?
  const BinaryOperator *B = dyn_cast<BinaryOperator>(condition);

  if (!B)
    return;
  
  // The actual error condition.
  if (!((B->isRelationalOp() || B->isEqualityOp()) &&
        ((B->getLHS()->getType()->isFloatingType() ||
          B->getRHS()->getType()->isFloatingType()))))
    return;
  
  // Are we comparing variables?
  const DeclRefExpr *drLHS = dyn_cast<DeclRefExpr>(B->getLHS()->IgnoreParens());
  const DeclRefExpr *drRHS = dyn_cast<DeclRefExpr>(B->getRHS()->IgnoreParens());
  
  if (!drLHS && !drRHS)
    return;

  const VarDecl *vdLHS = drLHS ? dyn_cast<VarDecl>(drLHS->getDecl()) : NULL;
  const VarDecl *vdRHS = drRHS ? dyn_cast<VarDecl>(drRHS->getDecl()) : NULL;
  
  if (!vdLHS && !vdRHS)
    return;  
  
  // Does either variable appear in increment?
  const DeclRefExpr *drInc = GetIncrementedVar(increment, vdLHS, vdRHS);
  
  if (!drInc)
    return;
  
  // Emit the error.  First figure out which DeclRefExpr in the condition
  // referenced the compared variable.
  const DeclRefExpr *drCond = vdLHS == drInc->getDecl() ? drLHS : drRHS;

  llvm::SmallVector<SourceRange, 2> ranges;  
  std::string sbuf;
  llvm::raw_string_ostream os(sbuf);
  
  os << "Variable '" << drCond->getDecl()->getNameAsCString()
     << "' with floating point type '" << drCond->getType().getAsString()
     << "' should not be used as a loop counter";

  ranges.push_back(drCond->getSourceRange());
  ranges.push_back(drInc->getSourceRange());
  
  const char *bugType = "Floating point variable used as loop counter";
  BR.EmitBasicReport(bugType, "Security", os.str().c_str(),
                     FS->getLocStart(), ranges.data(), ranges.size());
}

//===----------------------------------------------------------------------===//
// Entry point for check.
//===----------------------------------------------------------------------===//

void clang::CheckSecuritySyntaxOnly(Decl *D, BugReporter &BR) {  
  WalkAST walker(BR);
  walker.Visit(D->getBody());  
}
