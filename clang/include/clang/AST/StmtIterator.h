//===--- StmtIterator.h - Iterators for Statements ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the StmtIterator and ConstStmtIterator classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMT_ITR_H
#define LLVM_CLANG_AST_STMT_ITR_H

#include "llvm/ADT/iterator"

namespace clang {

class Stmt;
class ScopedDecl;
  
class StmtIterator : public bidirectional_iterator<Stmt*, ptrdiff_t> {
  Stmt** S;
  ScopedDecl* D;
  
  void NextDecl();
  void PrevDecl();
  Stmt*& GetInitializer() const;
public:  
  StmtIterator(Stmt** s, ScopedDecl* d = NULL) : S(s), D(d) {}
  
  StmtIterator& operator++() { 
    if (D) NextDecl();
    else ++S;
      
    return *this;
  }
    
  StmtIterator operator++(int) {
    StmtIterator tmp = *this;
    operator++();
    return tmp;
  }
  
  StmtIterator& operator--() {
    if (D) PrevDecl();
    else --S;
    
    return *this;
  }
  
  StmtIterator operator--(int) {
    StmtIterator tmp = *this;
    operator--();
    return tmp;
  }
  
  reference operator*() const { return D ? GetInitializer() : *S; }
  pointer operator->() const { return D ? &GetInitializer() : S; }

  bool operator==(const StmtIterator& RHS) const {
    return D == RHS.D && S == RHS.S;
  }
  
  bool operator!=(const StmtIterator& RHS) const {
    return D != RHS.D || S != RHS.S;
  }
};
  
class ConstStmtIterator: public bidirectional_iterator<const Stmt*, ptrdiff_t> {
  StmtIterator I;
public:
  explicit ConstStmtIterator(const StmtIterator& i) : I(i) {}

  ConstStmtIterator& operator++() { ++I; return *this; }
  ConstStmtIterator& operator--() { --I; return *this; }

  ConstStmtIterator operator++(int) {
    ConstStmtIterator tmp = *this;
    operator++();
    return tmp;
  }
  
  ConstStmtIterator operator--(int) {
    ConstStmtIterator tmp = *this;
    operator--();
    return tmp;
  }
  
  reference operator*() const { return const_cast<reference>(*I); }
  pointer operator->() const { return const_cast<pointer>(I.operator->()); }
  
  bool operator==(const ConstStmtIterator& RHS) const { return I == RHS.I; }
  bool operator!=(const ConstStmtIterator& RHS) const { return I != RHS.I; }
};
  
} // end namespace clang

#endif
