//===- llvm/Support/GetElementPtrTypeIterator.h -----------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements an iterator for walking through the types indexed by
// getelementptr instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_GETELEMENTPTRTYPE_H
#define LLVM_SUPPORT_GETELEMENTPTRTYPE_H

#include "llvm/User.h"
#include "llvm/DerivedTypes.h"

namespace llvm {
  class gep_type_iterator
    : public forward_iterator<const Type *, ptrdiff_t> {
    typedef forward_iterator<const Type*, ptrdiff_t> super;

    User::op_iterator OpIt;
    const Type *CurTy;
    gep_type_iterator() {}
  public:

    static gep_type_iterator begin(const Type *Ty, User::op_iterator It) {
      gep_type_iterator I;
      I.CurTy = Ty;
      I.OpIt = It;
      return I;
    }
    static gep_type_iterator end(User::op_iterator It) {
      gep_type_iterator I;
      I.CurTy = 0;
      I.OpIt = It;
      return I;
    }

    bool operator==(const gep_type_iterator& x) const { 
      return OpIt == x.OpIt;
    }
    bool operator!=(const gep_type_iterator& x) const {
      return !operator==(x);
    }

    const Type *operator*() const { 
      return CurTy;
    }

    // This is a non-standard operator->.  It allows you to call methods on the
    // current type directly.
    const Type *operator->() const { return operator*(); }
    
    Value *getOperand() const { return *OpIt; }

    gep_type_iterator& operator++() {   // Preincrement
      if (const CompositeType *CT = dyn_cast<CompositeType>(CurTy)) {
        CurTy = CT->getTypeAtIndex(getOperand());
      } else {
        CurTy = 0;
      }
      ++OpIt;
      return *this; 
    }

    gep_type_iterator operator++(int) { // Postincrement
      gep_type_iterator tmp = *this; ++*this; return tmp; 
    }
  };

  inline gep_type_iterator gep_type_begin(User *GEP) {
    return gep_type_iterator::begin(GEP->getOperand(0)->getType(),
                                    GEP->op_begin()+1);
  }
  inline gep_type_iterator gep_type_end(User *GEP) {
    return gep_type_iterator::end(GEP->op_end());
  }
  inline gep_type_iterator gep_type_begin(User &GEP) {
    return gep_type_iterator::begin(GEP.getOperand(0)->getType(),
                                    GEP.op_begin()+1);
  }
  inline gep_type_iterator gep_type_end(User &GEP) {
    return gep_type_iterator::end(GEP.op_end());
  }
  inline gep_type_iterator gep_type_begin(const Type *Op0, User::op_iterator I,
                                          User::op_iterator E) {
    return gep_type_iterator::begin(Op0, I);
  }
  inline gep_type_iterator gep_type_end(const Type *Op0, User::op_iterator I,
                                        User::op_iterator E) {
    return gep_type_iterator::end(E);
  }
} // end namespace llvm

#endif
