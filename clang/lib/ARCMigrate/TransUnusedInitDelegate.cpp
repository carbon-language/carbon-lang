//===--- TransUnusedInitDelegate.cpp - Tranformations to ARC mode ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Transformations:
//===----------------------------------------------------------------------===//
//
// rewriteUnusedInitDelegate:
//
// Rewrites an unused result of calling a delegate initialization, to assigning
// the result to self.
// e.g
//  [self init];
// ---->
//  self = [self init];
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"
#include "Internals.h"
#include "clang/Sema/SemaDiagnostic.h"

using namespace clang;
using namespace arcmt;
using namespace trans;
using llvm::StringRef;

namespace {

class UnusedInitRewriter : public RecursiveASTVisitor<UnusedInitRewriter> {
  Stmt *Body;
  MigrationPass &Pass;

  ExprSet Removables;

public:
  UnusedInitRewriter(MigrationPass &pass)
    : Body(0), Pass(pass) { }

  void transformBody(Stmt *body) {
    Body = body;
    collectRemovables(body, Removables);
    TraverseStmt(body);
  }

  bool VisitObjCMessageExpr(ObjCMessageExpr *ME) {
    if (ME->isDelegateInitCall() &&
        isRemovable(ME) &&
        Pass.TA.hasDiagnostic(diag::err_arc_unused_init_message,
                              ME->getExprLoc())) {
      Transaction Trans(Pass.TA);
      Pass.TA.clearDiagnostic(diag::err_arc_unused_init_message,
                              ME->getExprLoc());
      Pass.TA.insert(ME->getExprLoc(), "self = ");
    }
    return true;
  }

private:
  bool isRemovable(Expr *E) const {
    return Removables.count(E);
  }
};

} // anonymous namespace

void trans::rewriteUnusedInitDelegate(MigrationPass &pass) {
  BodyTransform<UnusedInitRewriter> trans(pass);
  trans.TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}
