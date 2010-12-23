//=== CastToStructChecker.cpp - Fixed address usage checker ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This files defines CastToStructChecker, a builtin checker that checks for
// cast from non-struct pointer to struct pointer.
// This check corresponds to CWE-588.
//
//===----------------------------------------------------------------------===//

#include "clang/EntoSA/BugReporter/BugType.h"
#include "clang/EntoSA/PathSensitive/CheckerVisitor.h"
#include "ExprEngineInternalChecks.h"

using namespace clang;
using namespace ento;

namespace {
class CastToStructChecker 
  : public CheckerVisitor<CastToStructChecker> {
  BuiltinBug *BT;
public:
  CastToStructChecker() : BT(0) {}
  static void *getTag();
  void PreVisitCastExpr(CheckerContext &C, const CastExpr *B);
};
}

void *CastToStructChecker::getTag() {
  static int x;
  return &x;
}

void CastToStructChecker::PreVisitCastExpr(CheckerContext &C,
                                           const CastExpr *CE) {
  const Expr *E = CE->getSubExpr();
  ASTContext &Ctx = C.getASTContext();
  QualType OrigTy = Ctx.getCanonicalType(E->getType());
  QualType ToTy = Ctx.getCanonicalType(CE->getType());

  PointerType *OrigPTy = dyn_cast<PointerType>(OrigTy.getTypePtr());
  PointerType *ToPTy = dyn_cast<PointerType>(ToTy.getTypePtr());

  if (!ToPTy || !OrigPTy)
    return;

  QualType OrigPointeeTy = OrigPTy->getPointeeType();
  QualType ToPointeeTy = ToPTy->getPointeeType();

  if (!ToPointeeTy->isStructureOrClassType())
    return;

  // We allow cast from void*.
  if (OrigPointeeTy->isVoidType())
    return;

  // Now the cast-to-type is struct pointer, the original type is not void*.
  if (!OrigPointeeTy->isRecordType()) {
    if (ExplodedNode *N = C.generateNode()) {
      if (!BT)
        BT = new BuiltinBug("Cast from non-struct type to struct type",
                            "Casting a non-structure type to a structure type "
                            "and accessing a field can lead to memory access "
                            "errors or data corruption.");
      RangedBugReport *R = new RangedBugReport(*BT,BT->getDescription(), N);
      R->addRange(CE->getSourceRange());
      C.EmitReport(R);
    }
  }
}

void ento::RegisterCastToStructChecker(ExprEngine &Eng) {
  Eng.registerCheck(new CastToStructChecker());
}
