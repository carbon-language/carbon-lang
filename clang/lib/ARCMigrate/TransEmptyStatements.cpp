//===--- TransEmptyStatements.cpp - Tranformations to ARC mode ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// removeEmptyStatements:
//
// Removes empty statements that are leftovers from previous transformations.
// e.g for
//
//  [x retain];
//
// removeRetainReleaseDealloc will leave an empty ";" that removeEmptyStatements
// will remove.
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"
#include "Internals.h"
#include "clang/AST/StmtVisitor.h"

using namespace clang;
using namespace arcmt;
using namespace trans;
using llvm::StringRef;

namespace {

class EmptyStatementsRemover :
                            public RecursiveASTVisitor<EmptyStatementsRemover> {
  MigrationPass &Pass;
  llvm::DenseSet<unsigned> MacroLocs;

public:
  EmptyStatementsRemover(MigrationPass &pass) : Pass(pass) {
    for (unsigned i = 0, e = Pass.ARCMTMacroLocs.size(); i != e; ++i)
      MacroLocs.insert(Pass.ARCMTMacroLocs[i].getRawEncoding());
  }

  bool TraverseStmtExpr(StmtExpr *E) {
    CompoundStmt *S = E->getSubStmt();
    for (CompoundStmt::body_iterator
           I = S->body_begin(), E = S->body_end(); I != E; ++I) {
      if (I != E - 1)
        check(*I);
      TraverseStmt(*I);
    }
    return true;
  }

  bool VisitCompoundStmt(CompoundStmt *S) {
    for (CompoundStmt::body_iterator
           I = S->body_begin(), E = S->body_end(); I != E; ++I)
      check(*I);
    return true;
  }

  bool isMacroLoc(SourceLocation loc) {
    if (loc.isInvalid()) return false;
    return MacroLocs.count(loc.getRawEncoding());
  }

  ASTContext &getContext() { return Pass.Ctx; }

private:
  /// \brief Returns true if the statement became empty due to previous
  /// transformations.
  class EmptyChecker : public StmtVisitor<EmptyChecker, bool> {
    EmptyStatementsRemover &Trans;

  public:
    EmptyChecker(EmptyStatementsRemover &trans) : Trans(trans) { }

    bool VisitNullStmt(NullStmt *S) {
      return Trans.isMacroLoc(S->getLeadingEmptyMacroLoc());
    }
    bool VisitCompoundStmt(CompoundStmt *S) {
      if (S->body_empty())
        return false; // was already empty, not because of transformations.
      for (CompoundStmt::body_iterator
             I = S->body_begin(), E = S->body_end(); I != E; ++I)
        if (!Visit(*I))
          return false;
      return true;
    }
    bool VisitIfStmt(IfStmt *S) {
      if (S->getConditionVariable())
        return false;
      Expr *condE = S->getCond();
      if (!condE)
        return false;
      if (hasSideEffects(condE, Trans.getContext()))
        return false;
      if (!S->getThen() || !Visit(S->getThen()))
        return false;
      if (S->getElse() && !Visit(S->getElse()))
        return false;
      return true;
    }
    bool VisitWhileStmt(WhileStmt *S) {
      if (S->getConditionVariable())
        return false;
      Expr *condE = S->getCond();
      if (!condE)
        return false;
      if (hasSideEffects(condE, Trans.getContext()))
        return false;
      if (!S->getBody())
        return false;
      return Visit(S->getBody());
    }
    bool VisitDoStmt(DoStmt *S) {
      Expr *condE = S->getCond();
      if (!condE)
        return false;
      if (hasSideEffects(condE, Trans.getContext()))
        return false;
      if (!S->getBody())
        return false;
      return Visit(S->getBody());
    }
    bool VisitObjCForCollectionStmt(ObjCForCollectionStmt *S) {
      Expr *Exp = S->getCollection();
      if (!Exp)
        return false;
      if (hasSideEffects(Exp, Trans.getContext()))
        return false;
      if (!S->getBody())
        return false;
      return Visit(S->getBody());
    }
    bool VisitObjCAutoreleasePoolStmt(ObjCAutoreleasePoolStmt *S) {
      if (!S->getSubStmt())
        return false;
      return Visit(S->getSubStmt());
    }
  };

  void check(Stmt *S) {
    if (!S) return;
    if (EmptyChecker(*this).Visit(S)) {
      Transaction Trans(Pass.TA);
      Pass.TA.removeStmt(S);
    }
  }
};

} // anonymous namespace

void trans::removeEmptyStatements(MigrationPass &pass) {
  EmptyStatementsRemover(pass).TraverseDecl(pass.Ctx.getTranslationUnitDecl());

  for (unsigned i = 0, e = pass.ARCMTMacroLocs.size(); i != e; ++i) {
    Transaction Trans(pass.TA);
    pass.TA.remove(pass.ARCMTMacroLocs[i]);
  }
}
