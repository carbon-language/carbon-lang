//=- ExprDeclBitVector.h - Dataflow types for Bitvector Analysis --*- C++ --*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides definition of dataflow types used by analyses such
// as LiveVariables and UninitializedValues.  The underlying dataflow values
// are implemented as bitvectors, but the definitions in this file include
// the necessary boilerplate to use with our dataflow framework.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_EXPRDECLBVDVAL_H
#define LLVM_CLANG_EXPRDECLBVDVAL_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {
  
  class Expr;
  class ScopedDecl;

struct ExprDeclBitVector_Types {
  
  //===--------------------------------------------------------------------===//
  // AnalysisDataTy - Whole-function meta data.
  //===--------------------------------------------------------------------===//

  class AnalysisDataTy {
  public:
    typedef llvm::DenseMap<const ScopedDecl*, unsigned > DMapTy;
    typedef llvm::DenseMap<const Expr*, unsigned > EMapTy;    
    typedef DMapTy::const_iterator decl_iterator;
    typedef EMapTy::const_iterator expr_iterator;

  protected:
    EMapTy EMap;
    DMapTy DMap;    
    unsigned NDecls;
    unsigned NExprs;

  public:
    
    AnalysisDataTy() : NDecls(0), NExprs(0) {}
    virtual ~AnalysisDataTy() {}
    
    bool isTracked(const ScopedDecl* SD) { return DMap.find(SD) != DMap.end(); }
    bool isTracked(const Expr* E) { return EMap.find(E) != EMap.end(); }

    unsigned operator[](const ScopedDecl* SD) const {
      DMapTy::const_iterator I = DMap.find(SD);
      assert (I != DMap.end());
      return I->second;
    }
    
    unsigned operator[](const Expr* E) const {
      EMapTy::const_iterator I = EMap.find(E);
      assert (I != EMap.end());
      return I->second;
    }
    
    unsigned getNumDecls() const { return NDecls; }
    unsigned getNumExprs() const { return NExprs; }
    
    void Register(const ScopedDecl* SD) {
      if (!isTracked(SD)) DMap[SD] = NDecls++;
    }
    
    void Register(const Expr* E) {
      if (!isTracked(E)) EMap[E] = NExprs++;
    }
    
    decl_iterator begin_decl() const { return DMap.begin(); }
    decl_iterator end_decl() const { return DMap.end(); }
    expr_iterator begin_expr() const { return EMap.begin(); }
    expr_iterator end_expr() const { return EMap.end(); }
  };

  //===--------------------------------------------------------------------===//
  // ValTy - Dataflow value.
  //===--------------------------------------------------------------------===//

  class ValTy {
    llvm::BitVector DeclBV;
    llvm::BitVector ExprBV;
  public:
    
    void resetValues(AnalysisDataTy& AD) {
      DeclBV.resize(AD.getNumDecls()); 
      DeclBV.reset();
      ExprBV.resize(AD.getNumExprs());
      ExprBV.reset();
    }
    
    bool operator==(const ValTy& RHS) const { 
      assert (sizesEqual(RHS));
      return DeclBV == RHS.DeclBV && ExprBV == RHS.ExprBV; 
    }
    
    void copyValues(const ValTy& RHS) {
      DeclBV = RHS.DeclBV;
      ExprBV = RHS.ExprBV;
    }
    
    llvm::BitVector::reference
    operator()(const ScopedDecl* SD, const AnalysisDataTy& AD) {
      return DeclBV[AD[SD]];      
    }
    const llvm::BitVector::reference
    operator()(const ScopedDecl* SD, const AnalysisDataTy& AD) const {
      return const_cast<ValTy&>(*this)(SD,AD);
    }
    
    llvm::BitVector::reference
    operator()(const Expr* E, const AnalysisDataTy& AD) {
      return ExprBV[AD[E]];      
    }    
    const llvm::BitVector::reference
    operator()(const Expr* E, const AnalysisDataTy& AD) const {
      return const_cast<ValTy&>(*this)(E,AD);
    }
    
    llvm::BitVector::reference getDeclBit(unsigned i) { return DeclBV[i]; }    
    const llvm::BitVector::reference getDeclBit(unsigned i) const {
      return const_cast<llvm::BitVector&>(DeclBV)[i];
    }
    
    llvm::BitVector::reference getExprBit(unsigned i) { return ExprBV[i]; }    
    const llvm::BitVector::reference getExprBit(unsigned i) const {
      return const_cast<llvm::BitVector&>(ExprBV)[i];
    }
    
    ValTy& operator|=(const ValTy& RHS) {
      assert (sizesEqual(RHS));
      DeclBV |= RHS.DeclBV;
      ExprBV |= RHS.ExprBV;
      return *this;
    }
    
    ValTy& operator&=(const ValTy& RHS) {
      assert (sizesEqual(RHS));
      DeclBV &= RHS.DeclBV;
      ExprBV &= RHS.ExprBV;
      return *this;
    }
    
    bool sizesEqual(const ValTy& RHS) const {
      return DeclBV.size() == RHS.DeclBV.size()
          && ExprBV.size() == RHS.ExprBV.size();
    }
  };
  
  //===--------------------------------------------------------------------===//
  // Some useful merge operations.
  //===--------------------------------------------------------------------===//
  
  struct Union { void operator()(ValTy& Dst, ValTy& Src) { Dst |= Src; } };
  struct Intersect { void operator()(ValTy& Dst, ValTy& Src) { Dst &= Src; } };
  
};
} // end namespace clang

#endif
