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
  IdentifierInfo *CFMakeCollectableII;

public:
  GCCollectableCallsChecker(MigrationContext &ctx, ParentMap &map)
    : MigrateCtx(ctx), PMap(map) {
    IdentifierTable &Ids = MigrateCtx.getPass().Ctx.Idents;
    NSMakeCollectableII = &Ids.get("NSMakeCollectable");
    CFMakeCollectableII = &Ids.get("CFMakeCollectable");
  }

  bool VisitCallExpr(CallExpr *E) {
    TransformActions &TA = MigrateCtx.getPass().TA;

    Expr *CEE = E->getCallee()->IgnoreParenImpCasts();
    if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CEE)) {
      if (FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(DRE->getDecl())) {
        if (!FD->getDeclContext()->getRedeclContext()->isFileContext())
          return true;

        if (FD->getIdentifier() == NSMakeCollectableII) {
          Transaction Trans(TA);
          TA.clearDiagnostic(diag::err_unavailable,
                             diag::err_unavailable_message,
                             diag::err_ovl_deleted_call, // ObjC++
                             DRE->getSourceRange());
          TA.replace(DRE->getSourceRange(), "CFBridgingRelease");

        } else if (FD->getIdentifier() == CFMakeCollectableII) {
          TA.reportError("CFMakeCollectable will leak the object that it "
                         "receives in ARC", DRE->getLocation(),
                         DRE->getSourceRange());
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
