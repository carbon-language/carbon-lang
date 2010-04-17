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

  switch (getKind()) {
  default: assert(0 && "Invalid Kind");
  case N_Type:
    return 0;
  case N_Decl:
    return D;
  case N_NamedRef:
    return NDRef.ND;
  case N_Stmt:
    return getDeclFromExpr(Stm);
  }
  
  return 0;
}

SourceRange ASTLocation::getSourceRange() const {
  if (isInvalid())
    return SourceRange();

  switch (getKind()) {
  default: assert(0 && "Invalid Kind");
    return SourceRange();
  case N_Decl:
    return D->getSourceRange();
  case N_Stmt:
    return Stm->getSourceRange();
  case N_NamedRef:
    return SourceRange(AsNamedRef().Loc, AsNamedRef().Loc);
  case N_Type:
    return AsTypeLoc().getSourceRange();
  }
  
  return SourceRange();
}

void ASTLocation::print(llvm::raw_ostream &OS) const {
  if (isInvalid()) {
    OS << "<< Invalid ASTLocation >>\n";
    return;
  }
  
  ASTContext &Ctx = getParentDecl()->getASTContext();

  switch (getKind()) {
  case N_Decl:
    OS << "[Decl: " << AsDecl()->getDeclKindName() << " ";
    if (const NamedDecl *ND = dyn_cast<NamedDecl>(AsDecl()))
      OS << ND;
    break;

  case N_Stmt:
    OS << "[Stmt: " << AsStmt()->getStmtClassName() << " ";
    AsStmt()->printPretty(OS, Ctx, 0, PrintingPolicy(Ctx.getLangOptions()));
    break;
    
  case N_NamedRef:
    OS << "[NamedRef: " << AsNamedRef().ND->getDeclKindName() << " ";
    OS << AsNamedRef().ND;
    break;
    
  case N_Type: {
    QualType T = AsTypeLoc().getType();
    OS << "[Type: " << T->getTypeClassName() << " " << T.getAsString();
  }
  }

  OS << "] <";

  SourceRange Range = getSourceRange();
  SourceManager &SourceMgr = Ctx.getSourceManager();
  Range.getBegin().print(OS, SourceMgr);
  OS << ", ";
  Range.getEnd().print(OS, SourceMgr);
  OS << ">\n";
}
