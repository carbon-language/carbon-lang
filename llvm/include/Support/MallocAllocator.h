//===-- Support/MallocAllocator.h - Allocator using malloc/free -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines MallocAllocator class, an STL compatible allocator which
// just uses malloc/free to get and release memory.  The default allocator uses
// the STL pool allocator runtime library, this explicitly avoids it.
//
// This file is used for variety of purposes, including the pool allocator
// project and testing, regardless of whether or not it's used directly in the
// LLVM code, so don't delete this from CVS if you think it's unused!
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MALLOCALLOCATOR_H
#define SUPPORT_MALLOCALLOCATOR_H

#include <cstdlib>
#include <memory>

template<typename T>
struct MallocAllocator {
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef T value_type;
  template <class U> struct rebind {
    typedef MallocAllocator<U> other;
  };

  template<typename R>
  MallocAllocator(const MallocAllocator<R> &) {}
  MallocAllocator() {}

  pointer address(reference x) const { return &x; }
  const_pointer address(const_reference x) const { return &x; }
  size_type max_size() const { return ~0 / sizeof(T); }
  
  static pointer allocate(size_t n, void* hint = 0) {
    return (pointer)malloc(n*sizeof(T));
  }

  static void deallocate(pointer p, size_t n) {
    free((void*)p);
  }

  void construct(pointer p, const T &val) {
    new((void*)p) T(val);
  }
  void destroy(pointer p) {
    p->~T();
  }
};

template<typename T>
inline bool operator==(const MallocAllocator<T> &, const MallocAllocator<T> &) {
  return true;
}
template<typename T>
inline bool operator!=(const MallocAllocator<T>&, const MallocAllocator<T>&) {
  return false;
}

namespace std {
  template<typename Type, typename Type2>
  struct _Alloc_traits<Type, ::MallocAllocator<Type2> > {
    static const bool _S_instanceless = true;
    typedef ::MallocAllocator<Type> base_alloc_type;
    typedef ::MallocAllocator<Type> _Alloc_type;
    typedef ::MallocAllocator<Type> allocator_type;
  };
}


#endif
