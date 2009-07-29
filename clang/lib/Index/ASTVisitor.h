//===--- ASTVisitor.h - Visitor for an ASTContext ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTVisitor interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_ASTVISITOR_H
#define LLVM_CLANG_INDEX_ASTVISITOR_H

#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"

namespace clang {

namespace idx {

/// \brief Traverses the full AST, both Decls and Stmts.
template<typename ImplClass>
class ASTVisitor : public DeclVisitor<ImplClass>,
                   public StmtVisitor<ImplClass> {
public:
  ASTVisitor() : CurrentDecl(0) { }

  Decl *CurrentDecl;

  typedef ASTVisitor<ImplClass>  Base;
  typedef DeclVisitor<ImplClass> BaseDeclVisitor;
  typedef StmtVisitor<ImplClass> BaseStmtVisitor;

  using BaseStmtVisitor::Visit;

  //===--------------------------------------------------------------------===//
  // DeclVisitor
  //===--------------------------------------------------------------------===//

  void Visit(Decl *D) {
    Decl *PrevDecl = CurrentDecl;
    CurrentDecl = D;
    BaseDeclVisitor::Visit(D);
    CurrentDecl = PrevDecl;
  }

  void VisitFunctionDecl(FunctionDecl *D) {
    BaseDeclVisitor::VisitValueDecl(D);
    if (D->isThisDeclarationADefinition())
      Visit(D->getBody());
  }

  void VisitObjCMethodDecl(ObjCMethodDecl *D) {
    BaseDeclVisitor::VisitNamedDecl(D);
    if (D->getBody())
      Visit(D->getBody());
  }

  void VisitBlockDecl(BlockDecl *D) {
    BaseDeclVisitor::VisitDecl(D);
    Visit(D->getBody());
  }

  void VisitVarDecl(VarDecl *D) {
    BaseDeclVisitor::VisitValueDecl(D);
    if (Expr *Init = D->getInit())
      Visit(Init);
  }

  void VisitDecl(Decl *D) {
    if (isa<FunctionDecl>(D) || isa<ObjCMethodDecl>(D) || isa<BlockDecl>(D))
      return;

    if (DeclContext *DC = dyn_cast<DeclContext>(D))
      static_cast<ImplClass*>(this)->VisitDeclContext(DC);
  }

  void VisitDeclContext(DeclContext *DC) {
    for (DeclContext::decl_iterator
           I = DC->decls_begin(), E = DC->decls_end(); I != E; ++I)
      Visit(*I);
  }

  //===--------------------------------------------------------------------===//
  // StmtVisitor
  //===--------------------------------------------------------------------===//

  void VisitDeclStmt(DeclStmt *Node) {
    for (DeclStmt::decl_iterator
           I = Node->decl_begin(), E = Node->decl_end(); I != E; ++I)
      Visit(*I);
  }

  void VisitBlockExpr(BlockExpr *Node) {
    Visit(Node->getBlockDecl());
  }

  void VisitStmt(Stmt *Node) {
    for (Stmt::child_iterator
           I = Node->child_begin(), E = Node->child_end(); I != E; ++I)
      if (*I)
        Visit(*I);
  }
};

} // namespace idx

} // namespace clang

#endif
