//===-- llvm/CFG.h - CFG definitions and useful classes ----------*- C++ -*--=//
//
// This file contains the class definitions useful for operating on the control
// flow graph.
//
// Currently it contains functionality for these three applications:
//
//  1. Iterate over the predecessors of a basic block:
//     pred_iterator, pred_const_iterator, pred_begin, pred_end
//  2. Iterate over the successors of a basic block:
//     succ_iterator, succ_const_iterator, succ_begin, succ_end
//  3. Iterate over the basic blocks of a method in depth first ordering or 
//     reverse depth first order.  df_iterator, df_const_iterator, 
//     df_begin, df_end.  df_begin takes an arg to specify reverse or not.
//  4. Iterator over the basic blocks of a method in post order.
//  5. Iterator over a method in reverse post order.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CFG_H
#define LLVM_CFG_H

#include "llvm/CFGdecls.h"      // See this file for concise interface info
#include "llvm/BasicBlock.h"
#include "llvm/InstrTypes.h"
#include "llvm/Type.h"
#include <iterator>

namespace cfg {

//===----------------------------------------------------------------------===//
//                                Implementation
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Basic Block Predecessor Iterator
//

template <class _Ptr,  class _USE_iterator> // Predecessor Iterator
class PredIterator : public std::bidirectional_iterator<_Ptr, ptrdiff_t> {
  _Ptr *BB;
  _USE_iterator It;
public:
  typedef PredIterator<_Ptr,_USE_iterator> _Self;
  
  inline void advancePastConstPool() {
    // TODO: This is bad
    // Loop to ignore constant pool references
    while (It != BB->use_end() && 
	   ((!(*It)->isInstruction()) ||
	    !(((Instruction*)(*It))->isTerminator())))
      ++It;
  }
  
  inline PredIterator(_Ptr *bb) : BB(bb), It(bb->use_begin()) {
    advancePastConstPool();
  }
  inline PredIterator(_Ptr *bb, bool) : BB(bb), It(bb->use_end()) {}
  
  inline bool operator==(const _Self& x) const { return It == x.It; }
  inline bool operator!=(const _Self& x) const { return !operator==(x); }
  
  inline pointer operator*() const { 
    return (*It)->castInstructionAsserting()->getParent(); 
  }
  inline pointer *operator->() const { return &(operator*()); }
  
  inline _Self& operator++() {   // Preincrement
    ++It; advancePastConstPool();
    return *this; 
  }
  
  inline _Self operator++(int) { // Postincrement
    _Self tmp = *this; ++*this; return tmp; 
  }
  
  inline _Self& operator--() { --It; return *this; }  // Predecrement
  inline _Self operator--(int) { // Postdecrement
    _Self tmp = *this; --*this; return tmp;
  }
};

inline pred_iterator       pred_begin(      BasicBlock *BB) {
  return pred_iterator(BB);
}
inline pred_const_iterator pred_begin(const BasicBlock *BB) {
  return pred_const_iterator(BB);
}
inline pred_iterator       pred_end(      BasicBlock *BB) {
  return pred_iterator(BB,true);
}
inline pred_const_iterator pred_end(const BasicBlock *BB) {
  return pred_const_iterator(BB,true);
}


//===----------------------------------------------------------------------===//
// Basic Block Successor Iterator
//

template <class _Term, class _BB>           // Successor Iterator
class SuccIterator : public std::bidirectional_iterator<_BB, ptrdiff_t> {
  const _Term Term;
  unsigned idx;
public:
  typedef SuccIterator<_Term, _BB> _Self;
  // TODO: This can be random access iterator, need operator+ and stuff tho
  
  inline SuccIterator(_Term T) : Term(T), idx(0) {         // begin iterator
    assert(T && "getTerminator returned null!");
  }
  inline SuccIterator(_Term T, bool)                       // end iterator
    : Term(T), idx(Term->getNumSuccessors()) {
    assert(T && "getTerminator returned null!");
  }
  
  inline bool operator==(const _Self& x) const { return idx == x.idx; }
  inline bool operator!=(const _Self& x) const { return !operator==(x); }
  
  inline pointer operator*() const { return Term->getSuccessor(idx); }
  inline pointer operator->() const { return operator*(); }
  
  inline _Self& operator++() { ++idx; return *this; } // Preincrement
  inline _Self operator++(int) { // Postincrement
    _Self tmp = *this; ++*this; return tmp; 
  }
  
  inline _Self& operator--() { --idx; return *this; }  // Predecrement
  inline _Self operator--(int) { // Postdecrement
    _Self tmp = *this; --*this; return tmp;
  }
};

inline succ_iterator       succ_begin(      BasicBlock *BB) {
  return succ_iterator(BB->getTerminator());
}
inline succ_const_iterator succ_begin(const BasicBlock *BB) {
  return succ_const_iterator(BB->getTerminator());
}
inline succ_iterator       succ_end(      BasicBlock *BB) {
  return succ_iterator(BB->getTerminator(),true);
}
inline succ_const_iterator succ_end(const BasicBlock *BB) {
  return succ_const_iterator(BB->getTerminator(),true);
}

}    // End namespace cfg

#endif
