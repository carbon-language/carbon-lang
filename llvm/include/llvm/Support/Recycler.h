//==- llvm/Support/Recycler.h - Recycling Allocator --------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Recycler class template.  See the doxygen comment for
// Recycler for more details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_RECYCLER_H
#define LLVM_SUPPORT_RECYCLER_H

#include "llvm/ADT/alist_node.h"

namespace llvm {

/// PrintRecyclingAllocatorStats - Helper for RecyclingAllocator for
/// printing statistics.
///
void PrintRecyclerStats(size_t LargestTypeSize, size_t FreeListSize);

/// Recycler - This class manages a linked-list of deallocated nodes
/// and facilitates reusing deallocated memory in place of allocating
/// new memory. The objects it allocates are stored in alist_node
/// containers, so they may be used in alists.
///
template<class T, class LargestT = T>
class Recycler {
  typedef alist_node<T, LargestT> NodeTy;

  /// FreeListTraits - ilist traits for FreeList.
  ///
  struct FreeListTraits : ilist_traits<alist_node<T, LargestT> > {
    NodeTy &getSentinel() { return this->Sentinel; }
  };

  /// FreeList - Doubly-linked list of nodes that have deleted contents and
  /// are not in active use.
  ///
  iplist<NodeTy, FreeListTraits> FreeList;

  /// CreateNewNode - Allocate a new node object and initialize its
  /// prev and next pointers to 0.
  ///
  template<class AllocatorType>
  NodeTy *CreateNewNode(AllocatorType &Allocator) {
    // Note that we're calling new on the *node*, to initialize its
    // Next/Prev pointers, not new on the end-user object.
    return new (Allocator.Allocate<NodeTy>()) NodeTy();
  }

public:
  ~Recycler() { assert(FreeList.empty()); }

  template<class AllocatorType>
  void clear(AllocatorType &Allocator) {
    while (!FreeList.empty())
      Allocator.Deallocate(FreeList.remove(FreeList.begin()));
  }

  template<class SubClass, class AllocatorType>
  SubClass *Allocate(AllocatorType &Allocator) {
    NodeTy *N = !FreeList.empty() ?
                FreeList.remove(FreeList.front()) :
                CreateNewNode(Allocator);
    assert(N->getPrev() == 0);
    assert(N->getNext() == 0);
    return N->getElement((SubClass*)0);
  }

  template<class AllocatorType>
  T *Allocate(AllocatorType &Allocator) {
    return Allocate<T>(Allocator);
  }

  template<class SubClass, class AllocatorType>
  void Deallocate(AllocatorType & /*Allocator*/, SubClass* Element) {
    NodeTy *N = NodeTy::getNode(Element);
    assert(N->getPrev() == 0);
    assert(N->getNext() == 0);
    FreeList.push_front(N);
  }

  void PrintStats() {
    PrintRecyclerStats(sizeof(LargestT), FreeList.size());
  }
};

}

#endif
