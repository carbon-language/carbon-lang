//===--- TransAllocCopyWithZone.cpp - Tranformations to ARC mode ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// rewriteAllocCopyWithZone:
//
// Calls to +allocWithZone/-copyWithZone/-mutableCopyWithZone are changed to
// +alloc/-copy/-mutableCopy if we can safely remove the given parameter.
//
//  Foo *foo1 = [[Foo allocWithZone:[self zone]] init];
// ---->
//  Foo *foo1 = [[Foo alloc] init];
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

class AllocCopyWithZoneRewriter :
                         public RecursiveASTVisitor<AllocCopyWithZoneRewriter> {
  Decl *Dcl;
  Stmt *Body;
  MigrationPass &Pass;

  Selector allocWithZoneSel;
  Selector copyWithZoneSel;
  Selector mutableCopyWithZoneSel;
  Selector zoneSel;
  IdentifierInfo *NSZoneII;

  std::vector<DeclStmt *> NSZoneVars;
  std::vector<Expr *> Removals;

public:
  AllocCopyWithZoneRewriter(Decl *D, MigrationPass &pass)
    : Dcl(D), Body(0), Pass(pass) {
    SelectorTable &sels = pass.Ctx.Selectors;
    IdentifierTable &ids = pass.Ctx.Idents; 
    allocWithZoneSel = sels.getUnarySelector(&ids.get("allocWithZone"));
    copyWithZoneSel = sels.getUnarySelector(&ids.get("copyWithZone"));
    mutableCopyWithZoneSel = sels.getUnarySelector(
                                               &ids.get("mutableCopyWithZone"));
    zoneSel = sels.getNullarySelector(&ids.get("zone"));
    NSZoneII = &ids.get("_NSZone");
  }

  void transformBody(Stmt *body) {
    Body = body;
    // Don't change allocWithZone/copyWithZone messages inside
    // custom implementations of such methods, it can lead to infinite loops.
    if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(Dcl)) {
      Selector sel = MD->getSelector();
      if (sel == allocWithZoneSel ||
          sel == copyWithZoneSel ||
          sel == mutableCopyWithZoneSel ||
          sel == zoneSel)
        return;
    }

    TraverseStmt(body);
  }

  ~AllocCopyWithZoneRewriter() {
    for (std::vector<DeclStmt *>::reverse_iterator
           I = NSZoneVars.rbegin(), E = NSZoneVars.rend(); I != E; ++I) {
      DeclStmt *DS = *I;
      DeclGroupRef group = DS->getDeclGroup();
      std::vector<Expr *> varRemovals = Removals;

      bool areAllVarsUnused = true;
      for (std::reverse_iterator<DeclGroupRef::iterator>
             DI(group.end()), DE(group.begin()); DI != DE; ++DI) {
        VarDecl *VD = cast<VarDecl>(*DI);
        if (isNSZoneVarUsed(VD, varRemovals)) {
          areAllVarsUnused = false;
          break;
        }
        varRemovals.push_back(VD->getInit());
      }

      if (areAllVarsUnused) {
        Transaction Trans(Pass.TA);
        clearUnavailableDiags(DS);
        Pass.TA.removeStmt(DS);
        Removals.swap(varRemovals);
      }
    }
  }

  bool VisitObjCMessageExpr(ObjCMessageExpr *E) {
    if (!isAllocCopyWithZoneCall(E))
      return true;
    Expr *arg = E->getArg(0);
    if (paramToAllocWithZoneHasSideEffects(arg))
      return true;

    Pass.TA.startTransaction();

    clearUnavailableDiags(arg);
    Pass.TA.clearDiagnostic(diag::err_unavailable_message,
                            E->getReceiverRange().getBegin());

    Pass.TA.remove(SourceRange(E->getSelectorLoc(), arg->getLocEnd()));
    StringRef rewrite;
    if (E->getSelector() == allocWithZoneSel)
      rewrite = "alloc";
    else if (E->getSelector() == copyWithZoneSel)
      rewrite = "copy";
    else {
      assert(E->getSelector() == mutableCopyWithZoneSel);
      rewrite = "mutableCopy";
    }
    Pass.TA.insert(E->getSelectorLoc(), rewrite);

    bool failed = Pass.TA.commitTransaction();
    if (!failed)
      Removals.push_back(arg);

    return true;
  }

  bool VisitDeclStmt(DeclStmt *DS) {
    DeclGroupRef group = DS->getDeclGroup();
    if (group.begin() == group.end())
      return true;
    for (DeclGroupRef::iterator
           DI = group.begin(), DE = group.end(); DI != DE; ++DI)
      if (!isRemovableNSZoneVar(*DI))
        return true;

    NSZoneVars.push_back(DS);
    return true;
  }

private:
  bool isRemovableNSZoneVar(Decl *D) {
    if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
      if (isNSZone(VD->getType()))
        return !paramToAllocWithZoneHasSideEffects(VD->getInit());
    }
    return false;
  }

  bool isNSZone(RecordDecl *RD) {
    return RD && RD->getIdentifier() == NSZoneII;
  }

  bool isNSZone(QualType Ty) {
    QualType pointee = Ty->getPointeeType();
    if (pointee.isNull())
      return false;
    if (const RecordType *recT = pointee->getAsStructureType())
      return isNSZone(recT->getDecl());
    return false;
  }

  bool isNSZoneVarUsed(VarDecl *D, std::vector<Expr *> &removals) {
    ExprSet refs;
    collectRefs(D, Body, refs);
    clearRefsIn(removals.begin(), removals.end(), refs);

    return !refs.empty();
  }

  bool isAllocCopyWithZoneCall(ObjCMessageExpr *E) {
    if (E->getNumArgs() == 1 &&
        E->getSelector() == allocWithZoneSel &&
        (E->isClassMessage() ||
         Pass.TA.hasDiagnostic(diag::err_unavailable_message,
                               E->getReceiverRange().getBegin())))
      return true;

    return E->isInstanceMessage() &&
           E->getNumArgs() == 1   &&
           (E->getSelector() == copyWithZoneSel ||
            E->getSelector() == mutableCopyWithZoneSel);
  }

  bool isZoneCall(ObjCMessageExpr *E) {
    return E->isInstanceMessage() &&
           E->getNumArgs() == 0   &&
           E->getSelector() == zoneSel;
  }

  bool paramToAllocWithZoneHasSideEffects(Expr *E) {
    if (!hasSideEffects(E, Pass.Ctx))
      return false;
    E = E->IgnoreParenCasts();
    ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(E);
    if (!ME)
      return true;
    if (!isZoneCall(ME))
      return true;
    return hasSideEffects(ME->getInstanceReceiver(), Pass.Ctx);
  }

  void clearUnavailableDiags(Stmt *S) {
    if (S)
      Pass.TA.clearDiagnostic(diag::err_unavailable,
                              diag::err_unavailable_message,
                              S->getSourceRange());
  }
};

} // end anonymous namespace

void trans::rewriteAllocCopyWithZone(MigrationPass &pass) {
  BodyTransform<AllocCopyWithZoneRewriter> trans(pass);
  trans.TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}
