//===--- TransGCCalls.cpp - Tranformations to ARC mode --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"
#include "Internals.h"
#include "clang/Sema/SemaDiagnostic.h"

using namespace clang;
using namespace arcmt;
using namespace trans;

namespace {

class GCCollectableCallsChecker :
                         public RecursiveASTVisitor<GCCollectableCallsChecker> {
  MigrationContext &MigrateCtx;
  ParentMap &PMap;
  IdentifierInfo *NSMakeCollectableII;

public:
  GCCollectableCallsChecker(MigrationContext &ctx, ParentMap &map)
    : MigrateCtx(ctx), PMap(map) {
    NSMakeCollectableII =
        &MigrateCtx.getPass().Ctx.Idents.get("NSMakeCollectable");
  }

  bool VisitCallExpr(CallExpr *E) {
    Expr *CEE = E->getCallee()->IgnoreParenImpCasts();
    if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CEE)) {
      if (FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(DRE->getDecl())) {
        if (FD->getDeclContext()->getRedeclContext()->isFileContext() &&
            FD->getIdentifier() == NSMakeCollectableII) {
          TransformActions &TA = MigrateCtx.getPass().TA;
          Transaction Trans(TA);
          TA.clearDiagnostic(diag::err_unavailable,
                             diag::err_unavailable_message,
                             DRE->getSourceRange());
          TA.replace(DRE->getSourceRange(), "CFBridgingRelease");
        }
      }
    }

    return true;
  }
};

} // anonymous namespace

void GCCollectableCallsTraverser::traverseBody(BodyContext &BodyCtx) {
  GCCollectableCallsChecker(BodyCtx.getMigrationContext(),
                            BodyCtx.getParentMap())
                                            .TraverseStmt(BodyCtx.getTopStmt());
}
