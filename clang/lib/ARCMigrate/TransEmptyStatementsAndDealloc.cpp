//===--- TransEmptyStatements.cpp - Tranformations to ARC mode ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// removeEmptyStatementsAndDealloc:
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

namespace {

/// \brief Returns true if the statement became empty due to previous
/// transformations.
class EmptyChecker : public StmtVisitor<EmptyChecker, bool> {
  ASTContext &Ctx;
  llvm::DenseSet<unsigned> &MacroLocs;

public:
  EmptyChecker(ASTContext &ctx, llvm::DenseSet<unsigned> &macroLocs)
    : Ctx(ctx), MacroLocs(macroLocs) { }

  bool VisitNullStmt(NullStmt *S) {
    return isMacroLoc(S->getLeadingEmptyMacroLoc());
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
    if (hasSideEffects(condE, Ctx))
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
    if (hasSideEffects(condE, Ctx))
      return false;
    if (!S->getBody())
      return false;
    return Visit(S->getBody());
  }
  bool VisitDoStmt(DoStmt *S) {
    Expr *condE = S->getCond();
    if (!condE)
      return false;
    if (hasSideEffects(condE, Ctx))
      return false;
    if (!S->getBody())
      return false;
    return Visit(S->getBody());
  }
  bool VisitObjCForCollectionStmt(ObjCForCollectionStmt *S) {
    Expr *Exp = S->getCollection();
    if (!Exp)
      return false;
    if (hasSideEffects(Exp, Ctx))
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

private:
  bool isMacroLoc(SourceLocation loc) {
    if (loc.isInvalid()) return false;
    return MacroLocs.count(loc.getRawEncoding());
  }
};

class EmptyStatementsRemover :
                            public RecursiveASTVisitor<EmptyStatementsRemover> {
  MigrationPass &Pass;
  llvm::DenseSet<unsigned> &MacroLocs;

public:
  EmptyStatementsRemover(MigrationPass &pass,
                         llvm::DenseSet<unsigned> &macroLocs)
    : Pass(pass), MacroLocs(macroLocs) { }

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
  void check(Stmt *S) {
    if (!S) return;
    if (EmptyChecker(Pass.Ctx, MacroLocs).Visit(S)) {
      Transaction Trans(Pass.TA);
      Pass.TA.removeStmt(S);
    }
  }
};

} // anonymous namespace

static bool isBodyEmpty(CompoundStmt *body,
                        ASTContext &Ctx, llvm::DenseSet<unsigned> &MacroLocs) {
  for (CompoundStmt::body_iterator
         I = body->body_begin(), E = body->body_end(); I != E; ++I)
    if (!EmptyChecker(Ctx, MacroLocs).Visit(*I))
      return false;

  return true;
}

static void removeDeallocMethod(MigrationPass &pass,
                                llvm::DenseSet<unsigned> &MacroLocs) {
  ASTContext &Ctx = pass.Ctx;
  TransformActions &TA = pass.TA;
  DeclContext *DC = Ctx.getTranslationUnitDecl();

  typedef DeclContext::specific_decl_iterator<ObjCImplementationDecl>
    impl_iterator;
  for (impl_iterator I = impl_iterator(DC->decls_begin()),
                     E = impl_iterator(DC->decls_end()); I != E; ++I) {
    for (ObjCImplementationDecl::instmeth_iterator
           MI = (*I)->instmeth_begin(),
           ME = (*I)->instmeth_end(); MI != ME; ++MI) {
      ObjCMethodDecl *MD = *MI;
      if (MD->getMethodFamily() == OMF_dealloc) {
        if (MD->hasBody() &&
            isBodyEmpty(MD->getCompoundBody(), Ctx, MacroLocs)) {
          Transaction Trans(TA);
          TA.remove(MD->getSourceRange());
        }
        break;
      }
    }
  }
}

void trans::removeEmptyStatementsAndDealloc(MigrationPass &pass) {
  llvm::DenseSet<unsigned> MacroLocs;
  for (unsigned i = 0, e = pass.ARCMTMacroLocs.size(); i != e; ++i)
    MacroLocs.insert(pass.ARCMTMacroLocs[i].getRawEncoding());

  EmptyStatementsRemover(pass, MacroLocs)
    .TraverseDecl(pass.Ctx.getTranslationUnitDecl());

  removeDeallocMethod(pass, MacroLocs);

  for (unsigned i = 0, e = pass.ARCMTMacroLocs.size(); i != e; ++i) {
    Transaction Trans(pass.TA);
    pass.TA.remove(pass.ARCMTMacroLocs[i]);
  }
}
