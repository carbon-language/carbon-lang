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

#include "Support/iterator"
#include "llvm/iMemory.h"
#include "llvm/DerivedTypes.h"

namespace llvm {
  class GetElementPtrTypeIterator
    : public forward_iterator<const Type *, ptrdiff_t> {
    typedef forward_iterator<const Type*, ptrdiff_t> super;

    GetElementPtrInst *TheGEP;
    const Type *CurTy;
    unsigned Operand;
    
    GetElementPtrTypeIterator() {}
  public:

    static GetElementPtrTypeIterator begin(GetElementPtrInst *gep) {
      GetElementPtrTypeIterator I;
      I.TheGEP = gep;
      I.CurTy = gep->getOperand(0)->getType();
      I.Operand = 1;
      return I;
    }
    static GetElementPtrTypeIterator end(GetElementPtrInst *gep) {
      GetElementPtrTypeIterator I;
      I.TheGEP = gep;
      I.CurTy = 0;
      I.Operand = gep->getNumOperands();
      return I;
    }

    bool operator==(const GetElementPtrTypeIterator& x) const { 
      return Operand == x.Operand;
    }
    bool operator!=(const GetElementPtrTypeIterator& x) const {
      return !operator==(x);
    }

    const Type *operator*() const { 
      return CurTy;
    }

    // This is a non-standard operator->.  It allows you to call methods on the
    // current type directly.
    const Type *operator->() const { return operator*(); }
    
    unsigned getOperandNum() const { return Operand; }

    Value *getOperand() const { return TheGEP->getOperand(Operand); }

    GetElementPtrTypeIterator& operator++() {   // Preincrement
      if (const CompositeType *CT = dyn_cast<CompositeType>(CurTy)) {
        CurTy = CT->getTypeAtIndex(getOperand());
      } else {
        CurTy = 0;
      }
      ++Operand;
      return *this; 
    }

    GetElementPtrTypeIterator operator++(int) { // Postincrement
      GetElementPtrTypeIterator tmp = *this; ++*this; return tmp; 
    }
  };

  inline GetElementPtrTypeIterator gep_type_begin(GetElementPtrInst *GEP) {
    return GetElementPtrTypeIterator::begin(GEP);
  }

  inline GetElementPtrTypeIterator gep_type_end(GetElementPtrInst *GEP) {
    return GetElementPtrTypeIterator::end(GEP);
  }
} // end namespace llvm

#endif
