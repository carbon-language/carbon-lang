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
  Decl *Dcl;
  Stmt *Body;
  MigrationPass &Pass;

  ExprSet Removables;
  llvm::OwningPtr<ParentMap> StmtMap;

public:
  RetainReleaseDeallocRemover(Decl *D, MigrationPass &pass)
    : Dcl(D), Body(0), Pass(pass) { }

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
    case OMF_retain:
    case OMF_release:
    case OMF_autorelease:
      if (E->getReceiverKind() == ObjCMessageExpr::Instance)
        if (Expr *rec = E->getInstanceReceiver()) {
          rec = rec->IgnoreParenImpCasts();
          if (rec->getType().getObjCLifetime() == Qualifiers::OCL_ExplicitNone){
            std::string err = "It is not safe to remove '";
            err += E->getSelector().getAsString() + "' message on "
                "an __unsafe_unretained type";
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
      Pass.TA.clearDiagnostic(diag::err_arc_illegal_explicit_message,
                              diag::err_unavailable,
                              diag::err_unavailable_message,
                              E->getSuperLoc());
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
    Pass.TA.clearDiagnostic(diag::err_arc_illegal_explicit_message,
                            diag::err_unavailable,
                            diag::err_unavailable_message,
                            rec->getExprLoc());
    if (!hasSideEffects(E, Pass.Ctx)) {
      if (tryRemoving(E))
        return true;
    }
    Pass.TA.replace(E->getSourceRange(), rec->getSourceRange());

    return true;
  }

private:
  bool isRemovable(Expr *E) const {
    return Removables.count(E);
  }
  
  bool tryRemoving(Expr *E) const {
    if (isRemovable(E)) {
      Pass.TA.removeStmt(E);
      return true;
    }

    if (ParenExpr *parenE = dyn_cast_or_null<ParenExpr>(StmtMap->getParent(E)))
      return tryRemoving(parenE);

    if (BinaryOperator *
          bopE = dyn_cast_or_null<BinaryOperator>(StmtMap->getParent(E))) {
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
