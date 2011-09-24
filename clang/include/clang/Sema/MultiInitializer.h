//===--- MultiInitializer.h - Initializer expression group ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MultiInitializer class, which can represent a list
// initializer or a parentheses-wrapped group of expressions in a C++ member
// initializer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_MULTIINITIALIZER_H
#define LLVM_CLANG_SEMA_MULTIINITIALIZER_H

#include "clang/Sema/Ownership.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/PointerUnion.h"

namespace clang {
  class ASTContext;
  class Expr;
  class InitializationKind;
  class InitializedEntity;
  class InitListExpr;
  class Sema;

class MultiInitializer {
  llvm::PointerUnion<Expr*, Expr**> InitListOrExpressions;
  unsigned NumInitializers;
  SourceLocation LParenLoc;
  SourceLocation RParenLoc;

  InitListExpr *getInitList() const;
  Expr **getExpressions() const { return InitListOrExpressions.get<Expr**>(); }

public:
  MultiInitializer(Expr* InitList)
    : InitListOrExpressions(InitList)
  {}

  MultiInitializer(SourceLocation LParenLoc, Expr **Exprs, unsigned NumInits,
                   SourceLocation RParenLoc)
    : InitListOrExpressions(Exprs), NumInitializers(NumInits),
    LParenLoc(LParenLoc), RParenLoc(RParenLoc)
  {}

  bool isInitializerList() const { return InitListOrExpressions.is<Expr*>(); }

  SourceLocation getStartLoc() const;
  SourceLocation getEndLoc() const;

  typedef Expr **iterator;
  iterator begin() const;
  iterator end() const;

  bool isTypeDependent() const;

  bool DiagnoseUnexpandedParameterPack(Sema &SemaRef) const;

  // Return the InitListExpr or create a ParenListExpr.
  Expr *CreateInitExpr(ASTContext &Ctx, QualType T) const;

  ExprResult PerformInit(Sema &SemaRef, InitializedEntity Entity,
                         InitializationKind Kind) const;
};
}

#endif
