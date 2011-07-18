//===--- TransAPIUses.cpp - Tranformations to ARC mode --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// checkAPIUses:
//
// Emits error with some API uses that are not safe in ARC mode:
//
// - NSInvocation's [get/set]ReturnValue and [get/set]Argument are only safe
//   with __unsafe_unretained objects.
// - When a NSData's 'bytes' family of methods are used on a local var,
//   add __attribute__((objc_precise_lifetime)) to make it safer.
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"
#include "Internals.h"

using namespace clang;
using namespace arcmt;
using namespace trans;
using llvm::StringRef;

namespace {

class APIChecker : public RecursiveASTVisitor<APIChecker> {
  MigrationPass &Pass;

  Selector getReturnValueSel, setReturnValueSel;
  Selector getArgumentSel, setArgumentSel;

  Selector bytesSel, getBytesSel, getBytesLengthSel, getBytesRangeSel;

  llvm::DenseSet<VarDecl *> ChangedNSDataVars;
public:
  APIChecker(MigrationPass &pass) : Pass(pass) {
    SelectorTable &sels = Pass.Ctx.Selectors;
    IdentifierTable &ids = Pass.Ctx.Idents;
    getReturnValueSel = sels.getUnarySelector(&ids.get("getReturnValue"));
    setReturnValueSel = sels.getUnarySelector(&ids.get("setReturnValue"));

    IdentifierInfo *selIds[2];
    selIds[0] = &ids.get("getArgument");
    selIds[1] = &ids.get("atIndex");
    getArgumentSel = sels.getSelector(2, selIds);
    selIds[0] = &ids.get("setArgument");
    setArgumentSel = sels.getSelector(2, selIds);

    bytesSel = sels.getNullarySelector(&ids.get("bytes"));
    getBytesSel = sels.getUnarySelector(&ids.get("getBytes"));
    selIds[0] = &ids.get("getBytes");
    selIds[1] = &ids.get("length");
    getBytesLengthSel = sels.getSelector(2, selIds);
    selIds[1] = &ids.get("range");
    getBytesRangeSel = sels.getSelector(2, selIds);
  }

  bool VisitObjCMessageExpr(ObjCMessageExpr *E) {
    if (E->isInstanceMessage() &&
        E->getReceiverInterface() &&
        E->getReceiverInterface()->getName() == "NSInvocation") {
      StringRef selName;
      if (E->getSelector() == getReturnValueSel)
        selName = "getReturnValue";
      else if (E->getSelector() == setReturnValueSel)
        selName = "setReturnValue";
      else if (E->getSelector() == getArgumentSel)
        selName = "getArgument";
      else if (E->getSelector() == setArgumentSel)
        selName = "setArgument";

      if (selName.empty())
        return true;

      Expr *parm = E->getArg(0)->IgnoreParenCasts();
      QualType pointee = parm->getType()->getPointeeType();
      if (pointee.isNull())
        return true;

      if (pointee.getObjCLifetime() > Qualifiers::OCL_ExplicitNone) {
        std::string err = "NSInvocation's ";
        err += selName;
        err += " is not safe to be used with an object with ownership other "
            "than __unsafe_unretained";
        Pass.TA.reportError(err, parm->getLocStart(), parm->getSourceRange());
      }
      return true;
    }

    if (E->isInstanceMessage() &&
        E->getReceiverInterface() &&
        E->getReceiverInterface()->getName() == "NSData" &&
        E->getInstanceReceiver() &&
        (E->getSelector() == bytesSel ||
         E->getSelector() == getBytesSel ||
         E->getSelector() == getBytesLengthSel ||
         E->getSelector() == getBytesRangeSel)) {
      Expr *rec = E->getInstanceReceiver();
      rec = rec->IgnoreParenCasts();
      if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(rec))
        if (VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl()))
          if (VD->hasLocalStorage() && !ChangedNSDataVars.count(VD)) {
            Transaction Trans(Pass.TA);
            Pass.TA.insertAfterToken(VD->getLocation(),
                                     " __attribute__((objc_precise_lifetime))");
            ChangedNSDataVars.insert(VD);
          }
    }

    return true;
  }
};

} // anonymous namespace

void trans::checkAPIUses(MigrationPass &pass) {
  APIChecker(pass).TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}
