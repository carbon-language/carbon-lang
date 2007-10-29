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
class VariableArrayType;
  
class StmtIteratorBase {
protected:
  union { Stmt** stmt; ScopedDecl* decl; };
  ScopedDecl* FirstDecl;
  VariableArrayType* vat;
  
  void NextDecl();
  void PrevDecl();
  Stmt*& GetDeclExpr() const;

  StmtIteratorBase(Stmt** s) : stmt(s), FirstDecl(NULL), vat(NULL) {}
  StmtIteratorBase(ScopedDecl* d);
  StmtIteratorBase() : stmt(NULL), FirstDecl(NULL), vat(NULL) {}
};
  
  
template <typename DERIVED, typename REFERENCE>
class StmtIteratorImpl : public StmtIteratorBase, 
                         public std::iterator<std::bidirectional_iterator_tag,
                                              REFERENCE, ptrdiff_t, 
                                              REFERENCE, REFERENCE> {  
protected:
  StmtIteratorImpl(const StmtIteratorBase& RHS) : StmtIteratorBase(RHS) {}
public:
  StmtIteratorImpl() {}                                                
  StmtIteratorImpl(Stmt** s) : StmtIteratorBase(s) {}
  StmtIteratorImpl(ScopedDecl* d) : StmtIteratorBase(d) {}

  
  DERIVED& operator++() { 
    if (FirstDecl) NextDecl();
    else ++stmt;
      
    return static_cast<DERIVED&>(*this);
  }
    
  DERIVED operator++(int) {
    DERIVED tmp = static_cast<DERIVED&>(*this);
    operator++();
    return tmp;
  }
  
  DERIVED& operator--() {
    if (FirstDecl) PrevDecl();
    else --stmt;
    
    return static_cast<DERIVED&>(*this);
  }
  
  DERIVED operator--(int) {
    DERIVED tmp = static_cast<DERIVED&>(*this);
    operator--();
    return tmp;
  }

  bool operator==(const DERIVED& RHS) const {
    return FirstDecl == RHS.FirstDecl && stmt == RHS.stmt;
  }
  
  bool operator!=(const DERIVED& RHS) const {
    return FirstDecl != RHS.FirstDecl || stmt != RHS.stmt;
  }
  
  REFERENCE operator*() const { 
    return (REFERENCE) (FirstDecl ? GetDeclExpr() : *stmt);
  }
  
  REFERENCE operator->() const { return operator*(); }   
};

struct StmtIterator : public StmtIteratorImpl<StmtIterator,Stmt*&> {
  explicit StmtIterator() : StmtIteratorImpl<StmtIterator,Stmt*&>() {}
  StmtIterator(Stmt** S) : StmtIteratorImpl<StmtIterator,Stmt*&>(S) {}
  StmtIterator(ScopedDecl* D) : StmtIteratorImpl<StmtIterator,Stmt*&>(D) {}
};

struct ConstStmtIterator : public StmtIteratorImpl<ConstStmtIterator,
                                                   const Stmt*> {
  explicit ConstStmtIterator() : 
    StmtIteratorImpl<ConstStmtIterator,const Stmt*>() {}

  ConstStmtIterator(const StmtIterator& RHS) : 
    StmtIteratorImpl<ConstStmtIterator,const Stmt*>(RHS) {}
};

} // end namespace clang

#endif
