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

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/StaticAnalyzer/Checkers/LocalCheckers.h"
#include "clang/StaticAnalyzer/BugReporter/BugReporter.h"
#include <string>
#include "llvm/ADT/StringRef.h"

using namespace clang;
using namespace ento;

//===----------------------------------------------------------------------===//
// Generic type checking routines.
//===----------------------------------------------------------------------===//

static bool IsLLVMStringRef(QualType T) {
  const RecordType *RT = T->getAs<RecordType>();
  if (!RT)
    return false;

  return llvm::StringRef(QualType(RT, 0).getAsString()) ==
          "class llvm::StringRef";
}

/// Check whether the declaration is semantically inside the top-level
/// namespace named by ns.
static bool InNamespace(const Decl *D, llvm::StringRef NS) {
  const NamespaceDecl *ND = dyn_cast<NamespaceDecl>(D->getDeclContext());
  if (!ND)
    return false;
  const IdentifierInfo *II = ND->getIdentifier();
  if (!II || !II->getName().equals(NS))
    return false;
  return isa<TranslationUnitDecl>(ND->getDeclContext());
}

static bool IsStdString(QualType T) {
  if (const ElaboratedType *QT = T->getAs<ElaboratedType>())
    T = QT->getNamedType();

  const TypedefType *TT = T->getAs<TypedefType>();
  if (!TT)
    return false;

  const TypedefDecl *TD = TT->getDecl();

  if (!InNamespace(TD, "std"))
    return false;

  return TD->getName() == "string";
}

static bool IsClangType(const RecordDecl *RD) {
  return RD->getName() == "Type" && InNamespace(RD, "clang");
}

static bool IsClangDecl(const RecordDecl *RD) {
  return RD->getName() == "Decl" && InNamespace(RD, "clang");
}

static bool IsClangStmt(const RecordDecl *RD) {
  return RD->getName() == "Stmt" && InNamespace(RD, "clang");
}

static bool IsClangAttr(const RecordDecl *RD) {
  return RD->getName() == "Attr" && InNamespace(RD, "clang");
}

static bool IsStdVector(QualType T) {
  const TemplateSpecializationType *TS = T->getAs<TemplateSpecializationType>();
  if (!TS)
    return false;

  TemplateName TM = TS->getTemplateName();
  TemplateDecl *TD = TM.getAsTemplateDecl();

  if (!TD || !InNamespace(TD, "std"))
    return false;

  return TD->getName() == "vector";
}

static bool IsSmallVector(QualType T) {
  const TemplateSpecializationType *TS = T->getAs<TemplateSpecializationType>();
  if (!TS)
    return false;

  TemplateName TM = TS->getTemplateName();
  TemplateDecl *TD = TM.getAsTemplateDecl();

  if (!TD || !InNamespace(TD, "llvm"))
    return false;

  return TD->getName() == "SmallVector";
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
  if (!IsLLVMStringRef(VD->getType()))
    return;
  ExprWithCleanups *Ex1 = dyn_cast<ExprWithCleanups>(Init);
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
  const char *desc = "StringRef should not be bound to temporary "
                     "std::string that it outlives";

  BR.EmitBasicReport(desc, "LLVM Conventions", desc,
                     VD->getLocStart(), Init->getSourceRange());
}

//===----------------------------------------------------------------------===//
// CHECK: Clang AST nodes should not have fields that can allocate
//   memory.
//===----------------------------------------------------------------------===//

static bool AllocatesMemory(QualType T) {
  return IsStdVector(T) || IsStdString(T) || IsSmallVector(T);
}

// This type checking could be sped up via dynamic programming.
static bool IsPartOfAST(const CXXRecordDecl *R) {
  if (IsClangStmt(R) || IsClangType(R) || IsClangDecl(R) || IsClangAttr(R))
    return true;

  for (CXXRecordDecl::base_class_const_iterator I = R->bases_begin(),
                                                E = R->bases_end(); I!=E; ++I) {
    CXXBaseSpecifier BS = *I;
    QualType T = BS.getType();
    if (const RecordType *baseT = T->getAs<RecordType>()) {
      CXXRecordDecl *baseD = cast<CXXRecordDecl>(baseT->getDecl());
      if (IsPartOfAST(baseD))
        return true;
    }
  }

  return false;
}

namespace {
class ASTFieldVisitor {
  llvm::SmallVector<FieldDecl*, 10> FieldChain;
  CXXRecordDecl *Root;
  BugReporter &BR;
public:
  ASTFieldVisitor(CXXRecordDecl *root, BugReporter &br)
    : Root(root), BR(br) {}

  void Visit(FieldDecl *D);
  void ReportError(QualType T);
};
} // end anonymous namespace

static void CheckASTMemory(CXXRecordDecl *R, BugReporter &BR) {
  if (!IsPartOfAST(R))
    return;

  for (RecordDecl::field_iterator I = R->field_begin(), E = R->field_end();
       I != E; ++I) {
    ASTFieldVisitor walker(R, BR);
    walker.Visit(*I);
  }
}

void ASTFieldVisitor::Visit(FieldDecl *D) {
  FieldChain.push_back(D);

  QualType T = D->getType();

  if (AllocatesMemory(T))
    ReportError(T);

  if (const RecordType *RT = T->getAs<RecordType>()) {
    const RecordDecl *RD = RT->getDecl()->getDefinition();
    for (RecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end();
         I != E; ++I)
      Visit(*I);
  }

  FieldChain.pop_back();
}

void ASTFieldVisitor::ReportError(QualType T) {
  llvm::SmallString<1024> buf;
  llvm::raw_svector_ostream os(buf);

  os << "AST class '" << Root->getName() << "' has a field '"
     << FieldChain.front()->getName() << "' that allocates heap memory";
  if (FieldChain.size() > 1) {
    os << " via the following chain: ";
    bool isFirst = true;
    for (llvm::SmallVectorImpl<FieldDecl*>::iterator I=FieldChain.begin(),
         E=FieldChain.end(); I!=E; ++I) {
      if (!isFirst)
        os << '.';
      else
        isFirst = false;
      os << (*I)->getName();
    }
  }
  os << " (type " << FieldChain.back()->getType().getAsString() << ")";
  os.flush();

  // Note that this will fire for every translation unit that uses this
  // class.  This is suboptimal, but at least scan-build will merge
  // duplicate HTML reports.  In the future we need a unified way of merging
  // duplicate reports across translation units.  For C++ classes we cannot
  // just report warnings when we see an out-of-line method definition for a
  // class, as that heuristic doesn't always work (the complete definition of
  // the class may be in the header file, for example).
  BR.EmitBasicReport("AST node allocates heap memory", "LLVM Conventions",
                     os.str(), FieldChain.front()->getLocStart());
}

//===----------------------------------------------------------------------===//
// Entry point for all checks.
//===----------------------------------------------------------------------===//

static void ScanCodeDecls(DeclContext *DC, BugReporter &BR) {
  for (DeclContext::decl_iterator I=DC->decls_begin(), E=DC->decls_end();
       I!=E ; ++I) {

    Decl *D = *I;

    if (D->hasBody())
      CheckStringRefAssignedTemporary(D, BR);

    if (CXXRecordDecl *R = dyn_cast<CXXRecordDecl>(D))
      if (R->isDefinition())
        CheckASTMemory(R, BR);

    if (DeclContext *DC_child = dyn_cast<DeclContext>(D))
      ScanCodeDecls(DC_child, BR);
  }
}

void ento::CheckLLVMConventions(TranslationUnitDecl &TU,
                                 BugReporter &BR) {
  ScanCodeDecls(&TU, BR);
}

