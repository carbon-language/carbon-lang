//===--- Allocator.h - Simple memory allocation abstraction -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MallocAllocator and BumpPtrAllocator interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ALLOCATOR_H
#define LLVM_SUPPORT_ALLOCATOR_H

#include "llvm/Support/AlignOf.h"
#include <cstdlib>

namespace llvm {

class MallocAllocator {
public:
  MallocAllocator() {}
  ~MallocAllocator() {}

  void Reset() {}

  void *Allocate(size_t Size, size_t /*Alignment*/) { return malloc(Size); }

  template <typename T>
  T *Allocate() { return static_cast<T*>(malloc(sizeof(T))); }

  template <typename T>
  T *Allocate(size_t Num) {
    return static_cast<T*>(malloc(sizeof(T)*Num));
  }

  void Deallocate(const void *Ptr) { free(const_cast<void*>(Ptr)); }

  void PrintStats() const {}
};

/// BumpPtrAllocator - This allocator is useful for containers that need very
/// simple memory allocation strategies.  In particular, this just keeps
/// allocating memory, and never deletes it until the entire block is dead. This
/// makes allocation speedy, but must only be used when the trade-off is ok.
class BumpPtrAllocator {
  BumpPtrAllocator(const BumpPtrAllocator &); // do not implement
  void operator=(const BumpPtrAllocator &);   // do not implement

  void *TheMemory;
public:
  BumpPtrAllocator();
  ~BumpPtrAllocator();

  void Reset();

  void *Allocate(size_t Size, size_t Alignment);

  /// Allocate space, but do not construct, one object.
  ///
  template <typename T>
  T *Allocate() {
    return static_cast<T*>(Allocate(sizeof(T),AlignOf<T>::Alignment));
  }

  /// Allocate space for an array of objects.  This does not construct the
  /// objects though.
  template <typename T>
  T *Allocate(size_t Num) {
    return static_cast<T*>(Allocate(Num * sizeof(T), AlignOf<T>::Alignment));
  }

  /// Allocate space for a specific count of elements and with a specified
  /// alignment.
  template <typename T>
  T *Allocate(size_t Num, size_t Alignment) {
    // Round EltSize up to the specified alignment.
    size_t EltSize = (sizeof(T)+Alignment-1)&(-Alignment);
    return static_cast<T*>(Allocate(Num * EltSize, Alignment));
  }

  void Deallocate(const void * /*Ptr*/) {}

  void PrintStats() const;
};

}  // end namespace llvm

#endif
