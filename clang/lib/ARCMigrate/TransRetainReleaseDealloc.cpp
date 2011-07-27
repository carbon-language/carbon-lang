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

namespace {

class RetainReleaseDeallocRemover :
                       public RecursiveASTVisitor<RetainReleaseDeallocRemover> {
  Stmt *Body;
  MigrationPass &Pass;

  ExprSet Removables;
  llvm::OwningPtr<ParentMap> StmtMap;

  Selector DelegateSel;

public:
  RetainReleaseDeallocRemover(MigrationPass &pass)
    : Body(0), Pass(pass) {
    DelegateSel =
        Pass.Ctx.Selectors.getNullarySelector(&Pass.Ctx.Idents.get("delegate"));
  }

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
        Pass.TA.reportError("it is not safe to remove an unused 'autorelease' "
            "message; its receiver may be destroyed immediately",
            E->getLocStart(), E->getSourceRange());
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

          if (E->getMethodFamily() == OMF_release && isDelegateMessage(rec)) {
            Pass.TA.reportError("it is not safe to remove 'retain' "
                "message on the result of a 'delegate' message; "
                "the object that was passed to 'setDelegate:' may not be "
                "properly retained", rec->getLocStart());
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
      // Change the -release to "receiver = nil" in a finally to avoid a leak
      // when an exception is thrown.
      Pass.TA.replace(E->getSourceRange(), rec->getSourceRange());
      std::string str = " = ";
      str += getNilString(Pass.Ctx);
      Pass.TA.insertAfterToken(rec->getLocEnd(), str);
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

  bool isDelegateMessage(Expr *E) const {
    if (!E) return false;

    E = E->IgnoreParenCasts();
    if (ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(E))
      return (ME->isInstanceMessage() && ME->getSelector() == DelegateSel);

    if (ObjCPropertyRefExpr *propE = dyn_cast<ObjCPropertyRefExpr>(E))
      return propE->getGetterSelector() == DelegateSel;

    return false;
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
