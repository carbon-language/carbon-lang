//===-- Transforms.h - Tranformations to ARC mode ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_ARCMIGRATE_TRANSFORMS_H
#define LLVM_CLANG_LIB_ARCMIGRATE_TRANSFORMS_H

#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/ADT/DenseSet.h"

namespace clang {
  class Decl;
  class Stmt;
  class BlockDecl;
  class ObjCMethodDecl;
  class FunctionDecl;

namespace arcmt {
  class MigrationPass;

namespace trans {

//===----------------------------------------------------------------------===//
// Transformations.
//===----------------------------------------------------------------------===//

void rewriteAutoreleasePool(MigrationPass &pass);
void rewriteUnbridgedCasts(MigrationPass &pass);
void rewriteAllocCopyWithZone(MigrationPass &pass);
void makeAssignARCSafe(MigrationPass &pass);
void removeRetainReleaseDealloc(MigrationPass &pass);
void removeZeroOutPropsInDealloc(MigrationPass &pass);
void changeIvarsOfAssignProperties(MigrationPass &pass);
void rewriteBlockObjCVariable(MigrationPass &pass);
void rewriteUnusedInitDelegate(MigrationPass &pass);

void removeEmptyStatementsAndDealloc(MigrationPass &pass);

//===----------------------------------------------------------------------===//
// Helpers.
//===----------------------------------------------------------------------===//

/// \brief 'Loc' is the end of a statement range. This returns the location
/// immediately after the semicolon following the statement.
/// If no semicolon is found or the location is inside a macro, the returned
/// source location will be invalid.
SourceLocation findLocationAfterSemi(SourceLocation loc, ASTContext &Ctx);

bool hasSideEffects(Expr *E, ASTContext &Ctx);

template <typename BODY_TRANS>
class BodyTransform : public RecursiveASTVisitor<BodyTransform<BODY_TRANS> > {
  MigrationPass &Pass;

public:
  BodyTransform(MigrationPass &pass) : Pass(pass) { }

  void handleBody(Decl *D) {
    Stmt *body = D->getBody();
    if (body) {
      BODY_TRANS(D, Pass).transformBody(body);
    }
  }

  bool TraverseBlockDecl(BlockDecl *D) {
    handleBody(D);
    return true;
  }
  bool TraverseObjCMethodDecl(ObjCMethodDecl *D) {
    if (D->isThisDeclarationADefinition())
      handleBody(D);
    return true;
  }
  bool TraverseFunctionDecl(FunctionDecl *D) {
    if (D->isThisDeclarationADefinition())
      handleBody(D);
    return true;
  }
};

typedef llvm::DenseSet<Expr *> ExprSet;

void clearRefsIn(Stmt *S, ExprSet &refs);
template <typename iterator>
void clearRefsIn(iterator begin, iterator end, ExprSet &refs) {
  for (; begin != end; ++begin)
    clearRefsIn(*begin, refs);
}

void collectRefs(ValueDecl *D, Stmt *S, ExprSet &refs);

void collectRemovables(Stmt *S, ExprSet &exprs);

} // end namespace trans

} // end namespace arcmt

} // end namespace clang

#endif
