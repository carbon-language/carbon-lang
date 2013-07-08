//===- llvm/ADT/NullablePtr.h - A pointer that allows null ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines and implements the NullablePtr class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_NULLABLEPTR_H
#define LLVM_ADT_NULLABLEPTR_H

#include "llvm/Support/type_traits.h"
#include <cassert>
#include <cstddef>

namespace llvm {
/// NullablePtr pointer wrapper - NullablePtr is used for APIs where a
/// potentially-null pointer gets passed around that must be explicitly handled
/// in lots of places.  By putting a wrapper around the null pointer, it makes
/// it more likely that the null pointer case will be handled correctly.
template<class T>
class NullablePtr {
  T *Ptr;
  struct PlaceHolder {};

public:
  NullablePtr(T *P = 0) : Ptr(P) {}

  template<typename OtherT>
  NullablePtr(NullablePtr<OtherT> Other,
              typename enable_if<
                is_base_of<T, OtherT>,
                PlaceHolder
              >::type = PlaceHolder()) : Ptr(Other.getPtrOrNull()) {}
  
  bool isNull() const { return Ptr == 0; }
  bool isNonNull() const { return Ptr != 0; }

  /// get - Return the pointer if it is non-null.
  const T *get() const {
    assert(Ptr && "Pointer wasn't checked for null!");
    return Ptr;
  }

  /// get - Return the pointer if it is non-null.
  T *get() {
    assert(Ptr && "Pointer wasn't checked for null!");
    return Ptr;
  }
  
  T *getPtrOrNull() { return Ptr; }
  const T *getPtrOrNull() const { return Ptr; }
};
  
} // end namespace llvm

#endif
