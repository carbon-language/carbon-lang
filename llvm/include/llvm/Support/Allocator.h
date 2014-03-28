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
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>

namespace llvm {
template <typename T> struct ReferenceAdder {
  typedef T &result;
};
template <typename T> struct ReferenceAdder<T &> {
  typedef T result;
};

class MallocAllocator {
public:
  MallocAllocator() {}
  ~MallocAllocator() {}

  void Reset() {}

  void *Allocate(size_t Size, size_t /*Alignment*/) { return malloc(Size); }

  template <typename T> T *Allocate() {
    return static_cast<T *>(malloc(sizeof(T)));
  }

  template <typename T> T *Allocate(size_t Num) {
    return static_cast<T *>(malloc(sizeof(T) * Num));
  }

  void Deallocate(const void *Ptr) { free(const_cast<void *>(Ptr)); }

  void PrintStats() const {}
};

/// MemSlab - This structure lives at the beginning of every slab allocated by
/// the bump allocator.
class MemSlab {
public:
  size_t Size;
  MemSlab *NextPtr;
};

/// SlabAllocator - This class can be used to parameterize the underlying
/// allocation strategy for the bump allocator.  In particular, this is used
/// by the JIT to allocate contiguous swathes of executable memory.  The
/// interface uses MemSlab's instead of void *'s so that the allocator
/// doesn't have to remember the size of the pointer it allocated.
class SlabAllocator {
public:
  virtual ~SlabAllocator();
  virtual MemSlab *Allocate(size_t Size) = 0;
  virtual void Deallocate(MemSlab *Slab) = 0;
};

/// MallocSlabAllocator - The default slab allocator for the bump allocator
/// is an adapter class for MallocAllocator that just forwards the method
/// calls and translates the arguments.
class MallocSlabAllocator : public SlabAllocator {
  /// Allocator - The underlying allocator that we forward to.
  ///
  MallocAllocator Allocator;

public:
  MallocSlabAllocator() : Allocator() {}
  virtual ~MallocSlabAllocator();
  MemSlab *Allocate(size_t Size) override;
  void Deallocate(MemSlab *Slab) override;
};

/// \brief Allocate memory in an ever growing pool, as if by bump-pointer.
///
/// This isn't strictly a bump-pointer allocator as it uses backing slabs of
/// memory rather than relying on boundless contiguous heap. However, it has
/// bump-pointer semantics in that is a monotonically growing pool of memory
/// where every allocation is found by merely allocating the next N bytes in
/// the slab, or the next N bytes in the next slab.
///
/// Note that this also has a threshold for forcing allocations above a certain
/// size into their own slab.
class BumpPtrAllocator {
  BumpPtrAllocator(const BumpPtrAllocator &) LLVM_DELETED_FUNCTION;
  void operator=(const BumpPtrAllocator &) LLVM_DELETED_FUNCTION;

public:
  BumpPtrAllocator(size_t size = 4096, size_t threshold = 4096);
  BumpPtrAllocator(size_t size, size_t threshold, SlabAllocator &allocator);
  ~BumpPtrAllocator();

  /// \brief Deallocate all but the current slab and reset the current pointer
  /// to the beginning of it, freeing all memory allocated so far.
  void Reset();

  /// \brief Allocate space at the specified alignment.
  void *Allocate(size_t Size, size_t Alignment);

  /// \brief Allocate space for one object without constructing it.
  template <typename T> T *Allocate() {
    return static_cast<T *>(Allocate(sizeof(T), AlignOf<T>::Alignment));
  }

  /// \brief Allocate space for an array of objects without constructing them.
  template <typename T> T *Allocate(size_t Num) {
    return static_cast<T *>(Allocate(Num * sizeof(T), AlignOf<T>::Alignment));
  }

  /// \brief Allocate space for an array of objects with the specified alignment
  /// and without constructing them.
  template <typename T> T *Allocate(size_t Num, size_t Alignment) {
    // Round EltSize up to the specified alignment.
    size_t EltSize = (sizeof(T) + Alignment - 1) & (-Alignment);
    return static_cast<T *>(Allocate(Num * EltSize, Alignment));
  }

  void Deallocate(const void * /*Ptr*/) {}

  size_t GetNumSlabs() const { return NumSlabs; }

  void PrintStats() const;

  /// \brief Returns the total physical memory allocated by this allocator.
  size_t getTotalMemory() const;

private:
  /// \brief Allocate at least this many bytes of memory in a slab.
  size_t SlabSize;

  /// \brief Threshold above which allocations to go into a dedicated slab.
  size_t SizeThreshold;

  /// \brief The default allocator used if one is not provided.
  MallocSlabAllocator DefaultSlabAllocator;

  /// \brief The underlying allocator we use to get slabs of memory.
  ///
  /// This defaults to MallocSlabAllocator, which wraps malloc, but it could be
  /// changed to use a custom allocator.
  SlabAllocator &Allocator;

  /// \brief The slab that we are currently allocating into.
  MemSlab *CurSlab;

  /// \brief The current pointer into the current slab.
  ///
  /// This points to the next free byte in the slab.
  char *CurPtr;

  /// \brief The end of the current slab.
  char *End;

  /// \brief How many bytes we've allocated.
  ///
  /// Used so that we can compute how much space was wasted.
  size_t BytesAllocated;

  /// \brief How many slabs we've allocated.
  ///
  /// Used to scale the size of each slab and reduce the number of allocations
  /// for extremely heavy memory use scenarios.
  size_t NumSlabs;

  /// \brief Allocate a new slab and move the bump pointers over into the new
  /// slab, modifying CurPtr and End.
  void StartNewSlab();

  /// \brief Deallocate all memory slabs after and including this one.
  void DeallocateSlabs(MemSlab *Slab);

  template <typename T> friend class SpecificBumpPtrAllocator;
};

/// \brief A BumpPtrAllocator that allows only elements of a specific type to be
/// allocated.
///
/// This allows calling the destructor in DestroyAll() and when the allocator is
/// destroyed.
template <typename T> class SpecificBumpPtrAllocator {
  BumpPtrAllocator Allocator;

public:
  SpecificBumpPtrAllocator(size_t size = 4096, size_t threshold = 4096)
      : Allocator(size, threshold) {}
  SpecificBumpPtrAllocator(size_t size, size_t threshold,
                           SlabAllocator &allocator)
      : Allocator(size, threshold, allocator) {}

  ~SpecificBumpPtrAllocator() { DestroyAll(); }

  /// Call the destructor of each allocated object and deallocate all but the
  /// current slab and reset the current pointer to the beginning of it, freeing
  /// all memory allocated so far.
  void DestroyAll() {
    MemSlab *Slab = Allocator.CurSlab;
    while (Slab) {
      char *End = Slab == Allocator.CurSlab ? Allocator.CurPtr
                                            : (char *)Slab + Slab->Size;
      for (char *Ptr = (char *)(Slab + 1); Ptr < End; Ptr += sizeof(T)) {
        Ptr = alignPtr(Ptr, alignOf<T>());
        if (Ptr + sizeof(T) <= End)
          reinterpret_cast<T *>(Ptr)->~T();
      }
      Slab = Slab->NextPtr;
    }
    Allocator.Reset();
  }

  /// \brief Allocate space for an array of objects without constructing them.
  T *Allocate(size_t num = 1) { return Allocator.Allocate<T>(num); }
};

}  // end namespace llvm

inline void *operator new(size_t Size, llvm::BumpPtrAllocator &Allocator) {
  struct S {
    char c;
    union {
      double D;
      long double LD;
      long long L;
      void *P;
    } x;
  };
  return Allocator.Allocate(Size, std::min((size_t)llvm::NextPowerOf2(Size),
                                           offsetof(S, x)));
}

inline void operator delete(void *, llvm::BumpPtrAllocator &) {}

#endif // LLVM_SUPPORT_ALLOCATOR_H
