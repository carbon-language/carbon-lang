//===--- ASTLocation.cpp - A <Decl, Stmt> pair ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  ASTLocation is Decl or a Stmt and its immediate Decl parent.
//
//===----------------------------------------------------------------------===//

#include "clang/Index/ASTLocation.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprObjC.h"
using namespace clang;
using namespace idx;

static Decl *getDeclFromExpr(Stmt *E) {
  if (DeclRefExpr *RefExpr = dyn_cast<DeclRefExpr>(E))
    return RefExpr->getDecl();
  if (MemberExpr *ME = dyn_cast<MemberExpr>(E))
    return ME->getMemberDecl();
  if (ObjCIvarRefExpr *RE = dyn_cast<ObjCIvarRefExpr>(E))
    return RE->getDecl();

  if (CallExpr *CE = dyn_cast<CallExpr>(E))
    return getDeclFromExpr(CE->getCallee());
  if (CastExpr *CE = dyn_cast<CastExpr>(E))
    return getDeclFromExpr(CE->getSubExpr());

  return 0;
}

Decl *ASTLocation::getReferencedDecl() {
  if (isInvalid())
    return 0;
  if (isDecl())
    return getDecl();

  assert(getStmt());
  return getDeclFromExpr(getStmt());
}


static bool isContainedInStatement(const Stmt *Node, const Stmt *Parent) {
  assert(Node && Parent && "Passed null Node or Parent");

  if (Node == Parent)
    return true;

  for (Stmt::const_child_iterator
         I = Parent->child_begin(), E = Parent->child_end(); I != E; ++I) {
    if (*I)
      if (isContainedInStatement(Node, *I))
        return true;
  }

  return false;
}

const Decl *ASTLocation::FindImmediateParent(const Decl *D, const Stmt *Node) {
  assert(D && Node && "Passed null Decl or null Stmt");

  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    const Expr *Init = VD->getInit();
    if (Init == 0)
      return 0;
    return isContainedInStatement(Node, Init) ? D : 0;
  }

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (!FD->isThisDeclarationADefinition())
      return 0;

    for (DeclContext::decl_iterator
           I = FD->decls_begin(), E = FD->decls_end(); I != E; ++I) {
      const Decl *Child = FindImmediateParent(*I, Node);
      if (Child)
        return Child;
    }

    assert(FD->getBody() && "If not definition we should have exited already");
    return isContainedInStatement(Node, FD->getBody()) ? D : 0;
  }

  if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
    if (!MD->getBody())
      return 0;

    for (DeclContext::decl_iterator
           I = MD->decls_begin(), E = MD->decls_end(); I != E; ++I) {
      const Decl *Child = FindImmediateParent(*I, Node);
      if (Child)
        return Child;
    }

    assert(MD->getBody() && "If not definition we should have exited already");
    return isContainedInStatement(Node, MD->getBody()) ? D : 0;
  }

  if (const BlockDecl *BD = dyn_cast<BlockDecl>(D)) {
    for (DeclContext::decl_iterator
           I = BD->decls_begin(), E = BD->decls_end(); I != E; ++I) {
      const Decl *Child = FindImmediateParent(*I, Node);
      if (Child)
        return Child;
    }

    assert(BD->getBody() && "BlockDecl without body ?");
    return isContainedInStatement(Node, BD->getBody()) ? D : 0;
  }

  return 0;
}

bool ASTLocation::isImmediateParent(const Decl *D, const Stmt *Node) {
  assert(D && Node && "Passed null Decl or null Stmt");
  return D == FindImmediateParent(D, Node);
}

SourceRange ASTLocation::getSourceRange() const {
  if (isInvalid())
    return SourceRange();
  return isDecl() ? getDecl()->getSourceRange() : getStmt()->getSourceRange();
}

void ASTLocation::print(llvm::raw_ostream &OS) const {
  if (isInvalid()) {
    OS << "<< Invalid ASTLocation >>\n";
    return;
  }

  OS << "[Decl: " << getDecl()->getDeclKindName() << " ";
  if (const NamedDecl *ND = dyn_cast<NamedDecl>(getDecl()))
    OS << ND->getNameAsString();

  if (getStmt()) {
    ASTContext &Ctx = getDecl()->getASTContext();
    OS << " | Stmt: " << getStmt()->getStmtClassName() << " ";
    getStmt()->printPretty(OS, Ctx, 0, PrintingPolicy(Ctx.getLangOptions()));
  }

  OS << "] <";

  SourceRange Range = getSourceRange();
  SourceManager &SourceMgr = getDecl()->getASTContext().getSourceManager();
  Range.getBegin().print(OS, SourceMgr);
  OS << ", ";
  Range.getEnd().print(OS, SourceMgr);
  OS << ">\n";
}
