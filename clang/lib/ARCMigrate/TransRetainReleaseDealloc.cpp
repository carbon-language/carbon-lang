//===--- TransRetainReleaseDealloc.cpp - Tranformations to ARC mode -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// removeRetainReleaseDealloc:
//
// Removes retain/release/autorelease/dealloc messages.
//
//  return [[foo retain] autorelease];
// ---->
//  return foo;
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"
#include "Internals.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/AST/ParentMap.h"

using namespace clang;
using namespace arcmt;
using namespace trans;
using llvm::StringRef;

namespace {

class RetainReleaseDeallocRemover :
                       public RecursiveASTVisitor<RetainReleaseDeallocRemover> {
  Stmt *Body;
  MigrationPass &Pass;

  ExprSet Removables;
  llvm::OwningPtr<ParentMap> StmtMap;

public:
  RetainReleaseDeallocRemover(MigrationPass &pass)
    : Body(0), Pass(pass) { }

  void transformBody(Stmt *body) {
    Body = body;
    collectRemovables(body, Removables);
    StmtMap.reset(new ParentMap(body));
    TraverseStmt(body);
  }

  bool VisitObjCMessageExpr(ObjCMessageExpr *E) {
    switch (E->getMethodFamily()) {
    default:
      return true;
    case OMF_autorelease:
      if (isRemovable(E)) {
        // An unused autorelease is badness. If we remove it the receiver
        // will likely die immediately while previously it was kept alive
        // by the autorelease pool. This is bad practice in general, leave it
        // and emit an error to force the user to restructure his code.
        std::string err = "it is not safe to remove an unused '";
        err += E->getSelector().getAsString() + "'; its receiver may be "
            "destroyed immediately";
        Pass.TA.reportError(err, E->getLocStart(), E->getSourceRange());
        return true;
      }
      // Pass through.
    case OMF_retain:
    case OMF_release:
      if (E->getReceiverKind() == ObjCMessageExpr::Instance)
        if (Expr *rec = E->getInstanceReceiver()) {
          rec = rec->IgnoreParenImpCasts();
          if (rec->getType().getObjCLifetime() == Qualifiers::OCL_ExplicitNone &&
              (E->getMethodFamily() != OMF_retain || isRemovable(E))) {
            std::string err = "it is not safe to remove '";
            err += E->getSelector().getAsString() + "' message on "
                "an __unsafe_unretained type";
            Pass.TA.reportError(err, rec->getLocStart());
            return true;
          }

          if (isGlobalVar(rec) &&
              (E->getMethodFamily() != OMF_retain || isRemovable(E))) {
            std::string err = "it is not safe to remove '";
            err += E->getSelector().getAsString() + "' message on "
                "a global variable";
            Pass.TA.reportError(err, rec->getLocStart());
            return true;
          }
        }
    case OMF_dealloc:
      break;
    }

    switch (E->getReceiverKind()) {
    default:
      return true;
    case ObjCMessageExpr::SuperInstance: {
      Transaction Trans(Pass.TA);
      clearDiagnostics(E->getSuperLoc());
      if (tryRemoving(E))
        return true;
      Pass.TA.replace(E->getSourceRange(), "self");
      return true;
    }
    case ObjCMessageExpr::Instance:
      break;
    }

    Expr *rec = E->getInstanceReceiver();
    if (!rec) return true;

    Transaction Trans(Pass.TA);
    clearDiagnostics(rec->getExprLoc());

    if (E->getMethodFamily() == OMF_release &&
        isRemovable(E) && isInAtFinally(E)) {
      // Change the -release to "receiver = 0" in a finally to avoid a leak
      // when an exception is thrown.
      Pass.TA.replace(E->getSourceRange(), rec->getSourceRange());
      Pass.TA.insertAfterToken(rec->getLocEnd(), " = 0");
      return true;
    }

    if (!hasSideEffects(E, Pass.Ctx)) {
      if (tryRemoving(E))
        return true;
    }
    Pass.TA.replace(E->getSourceRange(), rec->getSourceRange());

    return true;
  }

private:
  void clearDiagnostics(SourceLocation loc) const {
    Pass.TA.clearDiagnostic(diag::err_arc_illegal_explicit_message,
                            diag::err_unavailable,
                            diag::err_unavailable_message,
                            loc);
  }

  bool isInAtFinally(Expr *E) const {
    assert(E);
    Stmt *S = E;
    while (S) {
      if (isa<ObjCAtFinallyStmt>(S))
        return true;
      S = StmtMap->getParent(S);
    }

    return false;
  }

  bool isRemovable(Expr *E) const {
    return Removables.count(E);
  }
  
  bool tryRemoving(Expr *E) const {
    if (isRemovable(E)) {
      Pass.TA.removeStmt(E);
      return true;
    }

    Stmt *parent = StmtMap->getParent(E);

    if (ImplicitCastExpr *castE = dyn_cast_or_null<ImplicitCastExpr>(parent))
      return tryRemoving(castE);

    if (ParenExpr *parenE = dyn_cast_or_null<ParenExpr>(parent))
      return tryRemoving(parenE);

    if (BinaryOperator *
          bopE = dyn_cast_or_null<BinaryOperator>(parent)) {
      if (bopE->getOpcode() == BO_Comma && bopE->getLHS() == E &&
          isRemovable(bopE)) {
        Pass.TA.replace(bopE->getSourceRange(), bopE->getRHS()->getSourceRange());
        return true;
      }
    }

    return false;
  }

};

} // anonymous namespace

void trans::removeRetainReleaseDealloc(MigrationPass &pass) {
  BodyTransform<RetainReleaseDeallocRemover> trans(pass);
  trans.TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}
