//===--- Allocator.h - Simple memory allocation abstraction -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MallocAllocator and BumpPtrAllocator interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ALLOCATOR_H
#define LLVM_SUPPORT_ALLOCATOR_H

#include <cstdlib>

namespace llvm {
    
class MallocAllocator {
public:
  MallocAllocator() {}
  ~MallocAllocator() {}
  
  void *Allocate(unsigned Size, unsigned Alignment) { return malloc(Size); }
  void Deallocate(void *Ptr) { free(Ptr); }
  void PrintStats() const {}
};

/// BumpPtrAllocator - This allocator is useful for containers that need very
/// simple memory allocation strategies.  In particular, this just keeps
/// allocating memory, and never deletes it until the entire block is dead. This
/// makes allocation speedy, but must only be used when the trade-off is ok.
class BumpPtrAllocator {
  void *TheMemory;
public:
  BumpPtrAllocator();
  ~BumpPtrAllocator();
  
  void *Allocate(unsigned Size, unsigned Alignment);
  void Deallocate(void *Ptr) {}
  void PrintStats() const;
};

}  // end namespace clang

#endif
