//===--- Allocator.cpp - Simple memory allocation abstraction -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the BumpPtrAllocator interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Recycler.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>

namespace llvm {

SlabAllocator::~SlabAllocator() { }

MallocSlabAllocator::~MallocSlabAllocator() { }

MemSlab *MallocSlabAllocator::Allocate(size_t Size) {
  MemSlab *Slab = (MemSlab*)Allocator.Allocate(Size, 0);
  Slab->Size = Size;
  Slab->NextPtr = nullptr;
  return Slab;
}

void MallocSlabAllocator::Deallocate(MemSlab *Slab) {
  Allocator.Deallocate(Slab);
}

void BumpPtrAllocatorBase::PrintStats() const {
  unsigned NumSlabs = 0;
  size_t TotalMemory = 0;
  for (MemSlab *Slab = CurSlab; Slab; Slab = Slab->NextPtr) {
    TotalMemory += Slab->Size;
    ++NumSlabs;
  }

  errs() << "\nNumber of memory regions: " << NumSlabs << '\n'
         << "Bytes used: " << BytesAllocated << '\n'
         << "Bytes allocated: " << TotalMemory << '\n'
         << "Bytes wasted: " << (TotalMemory - BytesAllocated)
         << " (includes alignment, etc)\n";
}

size_t BumpPtrAllocatorBase::getTotalMemory() const {
  size_t TotalMemory = 0;
  for (MemSlab *Slab = CurSlab; Slab; Slab = Slab->NextPtr) {
    TotalMemory += Slab->Size;
  }
  return TotalMemory;
}

void PrintRecyclerStats(size_t Size,
                        size_t Align,
                        size_t FreeListSize) {
  errs() << "Recycler element size: " << Size << '\n'
         << "Recycler element alignment: " << Align << '\n'
         << "Number of elements free for recycling: " << FreeListSize << '\n';
}

}
