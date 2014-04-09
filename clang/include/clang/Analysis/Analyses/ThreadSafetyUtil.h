//===- ThreadSafetyUtil.h --------------------------------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines some basic utility classes for use by ThreadSafetyTIL.h
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_THREAD_SAFETY_UTIL_H
#define LLVM_CLANG_THREAD_SAFETY_UTIL_H

#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Allocator.h"

#include <cassert>
#include <cstddef>
#include <utility>

namespace clang {
namespace threadSafety {
namespace til {

// Simple wrapper class to abstract away from the details of memory management.
// SExprs are allocated in pools, and deallocated all at once.
class MemRegionRef {
private:
  union AlignmentType {
    double d;
    void *p;
    long double dd;
    long long ii;
  };

public:
  MemRegionRef() : Allocator(nullptr) {}
  MemRegionRef(llvm::BumpPtrAllocator *A) : Allocator(A) {}

  void *allocate(size_t Sz) {
    return Allocator->Allocate(Sz, llvm::AlignOf<AlignmentType>::Alignment);
  }

  template <typename T> T *allocateT() { return Allocator->Allocate<T>(); }

  template <typename T> T *allocateT(size_t NumElems) {
    return Allocator->Allocate<T>(NumElems);
  }

private:
  llvm::BumpPtrAllocator *Allocator;
};


} // end namespace til
} // end namespace threadSafety
} // end namespace clang


inline void *operator new(size_t Sz,
                          clang::threadSafety::til::MemRegionRef &R) {
  return R.allocate(Sz);
}


namespace clang {
namespace threadSafety {
namespace til {

using llvm::StringRef;

// A simple fixed size array class that does not manage its own memory,
// suitable for use with bump pointer allocation.
template <class T> class SimpleArray {
public:
  SimpleArray() : Data(nullptr), Size(0), Capacity(0) {}
  SimpleArray(T *Dat, size_t Cp, size_t Sz = 0)
      : Data(Dat), Size(0), Capacity(Cp) {}
  SimpleArray(MemRegionRef A, size_t Cp)
      : Data(A.allocateT<T>(Cp)), Size(0), Capacity(Cp) {}
  SimpleArray(SimpleArray<T> &&A)
      : Data(A.Data), Size(A.Size), Capacity(A.Capacity) {
    A.Data = nullptr;
    A.Size = 0;
    A.Capacity = 0;
  }

  T *resize(size_t Ncp, MemRegionRef A) {
    T *Odata = Data;
    Data = A.allocateT<T>(Ncp);
    memcpy(Data, Odata, sizeof(T) * Size);
    return Odata;
  }

  typedef T *iterator;
  typedef const T *const_iterator;

  size_t size() const { return Size; }
  size_t capacity() const { return Capacity; }

  T &operator[](unsigned I) { return Data[I]; }
  const T &operator[](unsigned I) const { return Data[I]; }

  iterator begin() { return Data; }
  iterator end() { return Data + Size; }

  const_iterator cbegin() const { return Data; }
  const_iterator cend() const { return Data + Size; }

  void push_back(const T &Elem) {
    assert(Size < Capacity);
    Data[Size++] = Elem;
  }

  template <class Iter> unsigned append(Iter I, Iter E) {
    size_t Osz = Size;
    size_t J = Osz;
    for (; J < Capacity && I != E; ++J, ++I)
      Data[J] = *I;
    Size = J;
    return J - Osz;
  }

private:
  SimpleArray(const SimpleArray<T> &A) { }

  T *Data;
  size_t Size;
  size_t Capacity;
};


} // end namespace til
} // end namespace threadSafety
} // end namespace clang

#endif  // LLVM_CLANG_THREAD_SAFETY_UTIL_H
