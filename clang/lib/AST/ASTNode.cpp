//===--- ASTNode.h - A <Decl, Stmt> pair ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  ASTNode is Decl or a Stmt and its immediate Decl parent.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTNode.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"
using namespace clang;

static bool isContainedInStatement(Stmt *Node, Stmt *Parent) {
  assert(Node && Parent && "Passed null Node or Parent");
  
  if (Node == Parent)
    return true;
  
  for (Stmt::child_iterator
         I = Parent->child_begin(), E = Parent->child_end(); I != E; ++I) {
    if (isContainedInStatement(Node, *I))
      return true;
  }
  
  return false;
}

static Decl *FindImmediateParent(Decl *D, Stmt *Node) {
  assert(D && Node && "Passed null Decl or null Stmt");

  if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
    Expr *Init = VD->getInit();
    if (Init == 0)
      return 0;
    return isContainedInStatement(Node, Init) ? D : 0;
  }
  
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (!FD->isThisDeclarationADefinition())
      return 0;
    
    for (DeclContext::decl_iterator
           I = FD->decls_begin(), E = FD->decls_end(); I != E; ++I) {
      Decl *Child = FindImmediateParent(*I, Node);
      if (Child)
        return Child;
    }
    
    assert(FD->getBody() && "If not definition we should have exited already");
    return isContainedInStatement(Node, FD->getBody()) ? D : 0;
  }
  
  return 0;
}

bool ASTNode::isImmediateParent(Decl *D, Stmt *Node) {
  assert(D && Node && "Passed null Decl or null Stmt");
  return D == FindImmediateParent(D, Node);
}

void ASTNode::print(llvm::raw_ostream &OS) {
  assert(isValid() && "ASTNode is not valid");

  OS << "[Decl: " << getDecl()->getDeclKindName() << " ";
  if (NamedDecl *ND = dyn_cast<NamedDecl>(getDecl()))
    OS << ND->getNameAsString();
  
  if (getStmt()) {
    ASTContext &Ctx = getDecl()->getASTContext();
    OS << " | Stmt: " << getStmt()->getStmtClassName() << " ";
    getStmt()->printPretty(OS, Ctx, 0, PrintingPolicy(Ctx.getLangOptions()));
  }

  OS << "] <";
  
  SourceRange Range = hasStmt() ? getStmt()->getSourceRange()
                                : getDecl()->getSourceRange();
  SourceManager &SourceMgr = getDecl()->getASTContext().getSourceManager();
  Range.getBegin().print(OS, SourceMgr);
  OS << ", ";
  Range.getEnd().print(OS, SourceMgr);
  OS << ">\n";
}
