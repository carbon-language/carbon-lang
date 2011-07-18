//===- llvm/ADT/TinyPtrVector.h - 'Normally tiny' vectors -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Type class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_TINYPTRVECTOR_H
#define LLVM_ADT_TINYPTRVECTOR_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/PointerUnion.h"

namespace llvm {
  
/// TinyPtrVector - This class is specialized for cases where there are
/// normally 0 or 1 element in a vector, but is general enough to go beyond that
/// when required.
///
/// NOTE: This container doesn't allow you to store a null pointer into it.
///
template <typename EltTy>
class TinyPtrVector {
public:
  typedef llvm::SmallVector<EltTy, 4> VecTy;
  llvm::PointerUnion<EltTy, VecTy*> Val;
  
  TinyPtrVector() {}
  TinyPtrVector(const TinyPtrVector &RHS) : Val(RHS.Val) {
    if (VecTy *V = Val.template dyn_cast<VecTy*>())
      Val = new VecTy(*V);
  }
  ~TinyPtrVector() {
    if (VecTy *V = Val.template dyn_cast<VecTy*>())
      delete V;
  }
  
  /// empty() - This vector can be empty if it contains no element, or if it
  /// contains a pointer to an empty vector.
  bool empty() const {
    if (Val.isNull()) return true;
    if (VecTy *Vec = Val.template dyn_cast<VecTy*>())
      return Vec->empty();
    return false;
  }
  
  unsigned size() const {
    if (empty())
      return 0;
    if (Val. template is<EltTy>())
      return 1;
    return Val. template get<VecTy*>()->size();
  }
  
  EltTy operator[](unsigned i) const {
    assert(!Val.isNull() && "can't index into an empty vector");
    if (EltTy V = Val.template dyn_cast<EltTy>()) {
      assert(i == 0 && "tinyvector index out of range");
      return V;
    }
    
    assert(i < Val. template get<VecTy*>()->size() && 
           "tinyvector index out of range");
    return (*Val. template get<VecTy*>())[i];
  }
  
  EltTy front() const {
    assert(!empty() && "vector empty");
    if (EltTy V = Val.template dyn_cast<EltTy>())
      return V;
    return Val.template get<VecTy*>()->front();
  }
  
  void push_back(EltTy NewVal) {
    assert(NewVal != 0 && "Can't add a null value");
    
    // If we have nothing, add something.
    if (Val.isNull()) {
      Val = NewVal;
      return;
    }
    
    // If we have a single value, convert to a vector.
    if (EltTy V = Val.template  dyn_cast<EltTy>()) {
      Val = new VecTy();
      Val.template get<VecTy*>()->push_back(V);
    }
    
    // Add the new value, we know we have a vector.
    Val.template get<VecTy*>()->push_back(NewVal);
  }
  
  void clear() {
    // If we have a single value, convert to empty.
    if (EltTy V = Val.template dyn_cast<EltTy>()) {
      Val = (EltTy)0;
    } else if (VecTy *Vec = Val.template dyn_cast<VecTy*>()) {
      // If we have a vector form, just clear it.
      Vec->clear();
    }
    // Otherwise, we're already empty.
  }
  
private:
  void operator=(const TinyPtrVector&); // NOT IMPLEMENTED YET.
};
} // end namespace llvm

#endif
