//===--- FullExpr.h - C++ full expression class -----------------*- C++ -*-===//
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
#ifndef LLVM_CLANG_AST_FULLEXPR_H
#define LLVM_CLANG_AST_FULLEXPR_H

#include "llvm/ADT/PointerUnion.h"

namespace clang {
  class ASTContext;
  class CXXTemporary;
  class Expr;

  class FullExpr {
    struct ExprAndTemporaries {
      Expr *SubExpr;
      
      unsigned NumTemps;
      
      typedef CXXTemporary** temps_iterator;
      
      temps_iterator temps_begin() { 
        return reinterpret_cast<CXXTemporary **>(this + 1); 
      }
      temps_iterator temps_end() { 
        return temps_begin() + NumTemps;
      }
    };
  
    llvm::PointerUnion<Expr *, ExprAndTemporaries *> SubExpr;
    
    FullExpr() { }

  public:
    static FullExpr Create(ASTContext &Context, Expr *SubExpr, 
                           CXXTemporary **Temps, unsigned NumTemps);
    void Destroy(ASTContext &Context);
    
    Expr *getExpr() {
      if (Expr *E = SubExpr.dyn_cast<Expr *>())
        return E;
      
      return SubExpr.get<ExprAndTemporaries *>()->SubExpr;
    }
    
    const Expr *getExpr() const { 
      return const_cast<FullExpr*>(this)->getExpr();
    }
    
    typedef CXXTemporary** temps_iterator;

    temps_iterator temps_begin() {
      if (ExprAndTemporaries *ET = SubExpr.dyn_cast<ExprAndTemporaries *>())
        return ET->temps_begin();
      
      return 0;
    }
    temps_iterator temps_end() {
      if (ExprAndTemporaries *ET = SubExpr.dyn_cast<ExprAndTemporaries *>())
        return ET->temps_end();
      
      return 0;
    }
  };
  

}  // end namespace clang

#endif
