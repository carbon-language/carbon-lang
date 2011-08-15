//ProgramStateTrait.h - Partial implementations of ProgramStateTrait -*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines partial implementations of template specializations of
//  the class ProgramStateTrait<>.  ProgramStateTrait<> is used by ProgramState 
//  to implement set/get methods for mapulating a ProgramState's
// generic data map.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_CLANG_GR_PROGRAMSTATETRAIT_H
#define LLVM_CLANG_GR_PROGRAMSTATETRAIT_H

namespace llvm {
  class BumpPtrAllocator;
  template <typename K, typename D, typename I> class ImmutableMap;
  template <typename K, typename I> class ImmutableSet;
  template <typename T> class ImmutableList;
  template <typename T> class ImmutableListImpl;
}

namespace clang {

namespace ento {
  template <typename T> struct ProgramStatePartialTrait;

  // Partial-specialization for ImmutableMap.

  template <typename Key, typename Data, typename Info>
  struct ProgramStatePartialTrait< llvm::ImmutableMap<Key,Data,Info> > {
    typedef llvm::ImmutableMap<Key,Data,Info> data_type;
    typedef typename data_type::Factory&      context_type;
    typedef Key                               key_type;
    typedef Data                              value_type;
    typedef const value_type*                 lookup_type;

    static inline data_type MakeData(void *const* p) {
      return p ? data_type((typename data_type::TreeTy*) *p) : data_type(0);
    }
    static inline void *MakeVoidPtr(data_type B) {
      return B.getRoot();
    }
    static lookup_type Lookup(data_type B, key_type K) {
      return B.lookup(K);
    }
    static data_type Set(data_type B, key_type K, value_type E,context_type F){
      return F.add(B, K, E);
    }

    static data_type Remove(data_type B, key_type K, context_type F) {
      return F.remove(B, K);
    }

    static inline context_type MakeContext(void *p) {
      return *((typename data_type::Factory*) p);
    }

    static void *CreateContext(llvm::BumpPtrAllocator& Alloc) {
      return new typename data_type::Factory(Alloc);
    }

    static void DeleteContext(void *Ctx) {
      delete (typename data_type::Factory*) Ctx;
    }
  };


  // Partial-specialization for ImmutableSet.

  template <typename Key, typename Info>
  struct ProgramStatePartialTrait< llvm::ImmutableSet<Key,Info> > {
    typedef llvm::ImmutableSet<Key,Info>      data_type;
    typedef typename data_type::Factory&      context_type;
    typedef Key                               key_type;

    static inline data_type MakeData(void *const* p) {
      return p ? data_type((typename data_type::TreeTy*) *p) : data_type(0);
    }

    static inline void *MakeVoidPtr(data_type B) {
      return B.getRoot();
    }

    static data_type Add(data_type B, key_type K, context_type F) {
      return F.add(B, K);
    }

    static data_type Remove(data_type B, key_type K, context_type F) {
      return F.remove(B, K);
    }

    static bool Contains(data_type B, key_type K) {
      return B.contains(K);
    }

    static inline context_type MakeContext(void *p) {
      return *((typename data_type::Factory*) p);
    }

    static void *CreateContext(llvm::BumpPtrAllocator& Alloc) {
      return new typename data_type::Factory(Alloc);
    }

    static void DeleteContext(void *Ctx) {
      delete (typename data_type::Factory*) Ctx;
    }
  };

  // Partial-specialization for ImmutableList.

  template <typename T>
  struct ProgramStatePartialTrait< llvm::ImmutableList<T> > {
    typedef llvm::ImmutableList<T>            data_type;
    typedef T                                 key_type;
    typedef typename data_type::Factory&      context_type;

    static data_type Add(data_type L, key_type K, context_type F) {
      return F.add(K, L);
    }

    static bool Contains(data_type L, key_type K) {
      return L.contains(K);
    }

    static inline data_type MakeData(void *const* p) {
      return p ? data_type((const llvm::ImmutableListImpl<T>*) *p)
               : data_type(0);
    }

    static inline void *MakeVoidPtr(data_type D) {
      return  (void*) D.getInternalPointer();
    }

    static inline context_type MakeContext(void *p) {
      return *((typename data_type::Factory*) p);
    }

    static void *CreateContext(llvm::BumpPtrAllocator& Alloc) {
      return new typename data_type::Factory(Alloc);
    }

    static void DeleteContext(void *Ctx) {
      delete (typename data_type::Factory*) Ctx;
    }
  };
  
  // Partial specialization for bool.
  template <> struct ProgramStatePartialTrait<bool> {
    typedef bool data_type;

    static inline data_type MakeData(void *const* p) {
      return p ? (data_type) (uintptr_t) *p
               : data_type();
    }
    static inline void *MakeVoidPtr(data_type d) {
      return (void*) (uintptr_t) d;
    }
  };
  
  // Partial specialization for unsigned.
  template <> struct ProgramStatePartialTrait<unsigned> {
    typedef unsigned data_type;

    static inline data_type MakeData(void *const* p) {
      return p ? (data_type) (uintptr_t) *p
               : data_type();
    }
    static inline void *MakeVoidPtr(data_type d) {
      return (void*) (uintptr_t) d;
    }
  };
  
} // end GR namespace

} // end clang namespace

#endif
