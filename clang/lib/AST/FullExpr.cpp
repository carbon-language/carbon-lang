//===--- FullExpr.cpp - C++ full expression class ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the FullExpr interface, to be used for type safe handling
//  of full expressions.
//
//  Full expressions are described in C++ [intro.execution]p12.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/FullExpr.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "llvm/Support/AlignOf.h"
using namespace clang;

FullExpr FullExpr::Create(ASTContext &Context, Expr *SubExpr,
                          CXXTemporary **Temporaries, unsigned NumTemporaries) {
  FullExpr E;

  if (!NumTemporaries) {
      E.SubExpr = SubExpr;
      return E;
  }

  unsigned Size = sizeof(FullExpr) 
      + sizeof(CXXTemporary *) * NumTemporaries;

  unsigned Align = llvm::AlignOf<ExprAndTemporaries>::Alignment;
  ExprAndTemporaries *ET = 
      static_cast<ExprAndTemporaries *>(Context.Allocate(Size, Align));

  ET->SubExpr = SubExpr;
  std::copy(Temporaries, Temporaries + NumTemporaries, ET->temps_begin());
    
  return E;
}

void FullExpr::Destroy(ASTContext &Context) {
  if (Expr *E = SubExpr.dyn_cast<Expr *>()) {
    E->Destroy(Context);
    return;
  }
  
  ExprAndTemporaries *ET = SubExpr.get<ExprAndTemporaries *>();
  for (ExprAndTemporaries::temps_iterator i = ET->temps_begin(), 
       e = ET->temps_end(); i != e; ++i)
    (*i)->Destroy(Context);

  Context.Deallocate(ET);
}
