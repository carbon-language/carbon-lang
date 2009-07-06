//===--- DeclReferenceMap.h - Map Decls to their references -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  DeclReferenceMap creates a mapping from Decls to the ASTLocations that
//  reference them.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclReferenceMap.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/ASTLocation.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/Support/Compiler.h"
using namespace clang;

namespace {

class VISIBILITY_HIDDEN StmtMapper : public StmtVisitor<StmtMapper> {
  DeclReferenceMap::MapTy &Map;
  Decl *Parent;

public:
  StmtMapper(DeclReferenceMap::MapTy &map, Decl *parent)
    : Map(map), Parent(parent) { }

  void VisitDeclStmt(DeclStmt *Node);
  void VisitDeclRefExpr(DeclRefExpr *Node);
  void VisitStmt(Stmt *Node);
};

class VISIBILITY_HIDDEN DeclMapper : public DeclVisitor<DeclMapper> {
  DeclReferenceMap::MapTy &Map;
  
public:
  DeclMapper(DeclReferenceMap::MapTy &map)
    : Map(map) { }

  void VisitDeclContext(DeclContext *DC);
  void VisitVarDecl(VarDecl *D);
  void VisitFunctionDecl(FunctionDecl *D);
  void VisitBlockDecl(BlockDecl *D);
  void VisitDecl(Decl *D);
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// StmtMapper Implementation
//===----------------------------------------------------------------------===//

void StmtMapper::VisitDeclStmt(DeclStmt *Node) {
  DeclMapper Mapper(Map);
  for (DeclStmt::decl_iterator
         I = Node->decl_begin(), E = Node->decl_end(); I != E; ++I)
    Mapper.Visit(*I);
}

void StmtMapper::VisitDeclRefExpr(DeclRefExpr *Node) {
  NamedDecl *PrimD = cast<NamedDecl>(Node->getDecl()->getPrimaryDecl());
  Map.insert(std::make_pair(PrimD, ASTLocation(Parent, Node)));
}

void StmtMapper::VisitStmt(Stmt *Node) {
  for (Stmt::child_iterator
         I = Node->child_begin(), E = Node->child_end(); I != E; ++I)
    Visit(*I);
}

//===----------------------------------------------------------------------===//
// DeclMapper Implementation
//===----------------------------------------------------------------------===//

void DeclMapper::VisitDeclContext(DeclContext *DC) {
  for (DeclContext::decl_iterator
         I = DC->decls_begin(), E = DC->decls_end(); I != E; ++I)
    Visit(*I);
}

void DeclMapper::VisitFunctionDecl(FunctionDecl *D) {
  if (!D->isThisDeclarationADefinition())
    return;
  
  StmtMapper(Map, D).Visit(D->getBody());
}

void DeclMapper::VisitBlockDecl(BlockDecl *D) {
  StmtMapper(Map, D).Visit(D->getBody());
}

void DeclMapper::VisitVarDecl(VarDecl *D) {
  if (Expr *Init = D->getInit())
    StmtMapper(Map, D).Visit(Init);
}

void DeclMapper::VisitDecl(Decl *D) {
  if (DeclContext *DC = dyn_cast<DeclContext>(D))
    VisitDeclContext(DC);
}

//===----------------------------------------------------------------------===//
// DeclReferenceMap Implementation
//===----------------------------------------------------------------------===//

DeclReferenceMap::DeclReferenceMap(ASTContext &Ctx) {
  DeclMapper(Map).Visit(Ctx.getTranslationUnitDecl());
}

DeclReferenceMap::astlocation_iterator
DeclReferenceMap::refs_begin(NamedDecl *D) const {
  NamedDecl *Prim = cast<NamedDecl>(D->getPrimaryDecl());
  return astlocation_iterator(Map.lower_bound(Prim));  
}

DeclReferenceMap::astlocation_iterator
DeclReferenceMap::refs_end(NamedDecl *D) const {
  NamedDecl *Prim = cast<NamedDecl>(D->getPrimaryDecl());
  return astlocation_iterator(Map.upper_bound(Prim));  
}

bool DeclReferenceMap::refs_empty(NamedDecl *D) const {
  NamedDecl *Prim = cast<NamedDecl>(D->getPrimaryDecl());
  return refs_begin(Prim) == refs_end(Prim);  
}
