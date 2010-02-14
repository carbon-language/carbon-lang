//=== LLVMConventionsChecker.cpp - Check LLVM codebase conventions ---*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines LLVMConventionsChecker, a bunch of small little checks
// for checking specific coding conventions in the LLVM/Clang codebase.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Checker/Checkers/LocalCheckers.h"
#include "clang/Checker/BugReporter/BugReporter.h"
#include <string>
#include <llvm/ADT/StringRef.h>

using namespace clang;

//===----------------------------------------------------------------------===//
// Generic type checking routines.
//===----------------------------------------------------------------------===//

static bool IsStringRef(QualType T) {
  const RecordType *RT = T->getAs<RecordType>();
  if (!RT)
    return false;

  return llvm::StringRef(QualType(RT, 0).getAsString()) ==
  "class llvm::StringRef";
}

static bool IsStdString(QualType T) {
  if (const QualifiedNameType *QT = T->getAs<QualifiedNameType>())
    T = QT->getNamedType();

  const TypedefType *TT = T->getAs<TypedefType>();
  if (!TT)
    return false;

  const TypedefDecl *TD = TT->getDecl();    
  const NamespaceDecl *ND = dyn_cast<NamespaceDecl>(TD->getDeclContext());
  if (!ND)
    return false;
  const IdentifierInfo *II = ND->getIdentifier();
  if (!II || II->getName() != "std")
    return false;

  DeclarationName N = TD->getDeclName();
  return llvm::StringRef(N.getAsString()) == "string";
}

//===----------------------------------------------------------------------===//
// CHECK: a llvm::StringRef should not be bound to a temporary std::string whose
// lifetime is shorter than the StringRef's.
//===----------------------------------------------------------------------===//

namespace {
class StringRefCheckerVisitor : public StmtVisitor<StringRefCheckerVisitor> {
  BugReporter &BR;
public:
  StringRefCheckerVisitor(BugReporter &br) : BR(br) {}
  void VisitChildren(Stmt *S) {
    for (Stmt::child_iterator I = S->child_begin(), E = S->child_end() ;
      I != E; ++I)
      if (Stmt *child = *I)
        Visit(child);
  }
  void VisitStmt(Stmt *S) { VisitChildren(S); }
  void VisitDeclStmt(DeclStmt *DS);
private:
  void VisitVarDecl(VarDecl *VD);
  void CheckStringRefBoundtoTemporaryString(VarDecl *VD);
};
} // end anonymous namespace

static void CheckStringRefAssignedTemporary(const Decl *D, BugReporter &BR) {
  StringRefCheckerVisitor walker(BR);
  walker.Visit(D->getBody());
}

void StringRefCheckerVisitor::VisitDeclStmt(DeclStmt *S) {
  VisitChildren(S);

  for (DeclStmt::decl_iterator I = S->decl_begin(), E = S->decl_end();I!=E; ++I)
    if (VarDecl *VD = dyn_cast<VarDecl>(*I))
      VisitVarDecl(VD);
}

void StringRefCheckerVisitor::VisitVarDecl(VarDecl *VD) {
  Expr *Init = VD->getInit();
  if (!Init)
    return; 

  // Pattern match for:
  // llvm::StringRef x = call() (where call returns std::string)
  if (!IsStringRef(VD->getType()))
    return;
  CXXExprWithTemporaries *Ex1 = dyn_cast<CXXExprWithTemporaries>(Init);
  if (!Ex1)
    return;
  CXXConstructExpr *Ex2 = dyn_cast<CXXConstructExpr>(Ex1->getSubExpr());
  if (!Ex2 || Ex2->getNumArgs() != 1)
    return;
  ImplicitCastExpr *Ex3 = dyn_cast<ImplicitCastExpr>(Ex2->getArg(0));
  if (!Ex3)
    return;
  CXXConstructExpr *Ex4 = dyn_cast<CXXConstructExpr>(Ex3->getSubExpr());
  if (!Ex4 || Ex4->getNumArgs() != 1)
    return;
  ImplicitCastExpr *Ex5 = dyn_cast<ImplicitCastExpr>(Ex4->getArg(0));
  if (!Ex5)
    return;
  CXXBindTemporaryExpr *Ex6 = dyn_cast<CXXBindTemporaryExpr>(Ex5->getSubExpr());
  if (!Ex6 || !IsStdString(Ex6->getType()))
    return;

  // Okay, badness!  Report an error.
  BR.EmitBasicReport("StringRef should not be bound to temporary "
                     "std::string that it outlives", "LLVM Conventions",
                     VD->getLocStart(), Init->getSourceRange());
}

//===----------------------------------------------------------------------===//
// Entry point for all checks.
//===----------------------------------------------------------------------===//

void clang::CheckLLVMConventions(const Decl *D, BugReporter &BR) {
  CheckStringRefAssignedTemporary(D, BR);
}
