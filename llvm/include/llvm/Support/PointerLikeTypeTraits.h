//===- llvm/Support/PointerLikeTypeTraits.h - Pointer Traits ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PointerLikeTypeTraits class.  This allows data
// structures to reason about pointers and other things that are pointer sized.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_POINTERLIKETYPETRAITS_H
#define LLVM_SUPPORT_POINTERLIKETYPETRAITS_H

namespace llvm {
  
/// PointerLikeTypeTraits - This is a traits object that is used to handle
/// pointer types and things that are just wrappers for pointers as a uniform
/// entity.
template <typename T>
class PointerLikeTypeTraits {
  // getAsVoidPointer
  // getFromVoidPointer
  // getNumLowBitsAvailable
};

// Provide PointerLikeTypeTraits for non-cvr pointers.
template<typename T>
class PointerLikeTypeTraits<T*> {
public:
  static inline void *getAsVoidPointer(T* P) { return P; }
  static inline T *getFromVoidPointer(void *P) {
    return static_cast<T*>(P);
  }
  
  /// Note, we assume here that malloc returns objects at least 8-byte aligned.
  /// However, this may be wrong, or pointers may be from something other than
  /// malloc.  In this case, you should specialize this template to reduce this.
  ///
  /// All clients should use assertions to do a run-time check to ensure that
  /// this is actually true.
  static inline unsigned getNumLowBitsAvailable() { return 3; }
};
  
// Provide PointerLikeTypeTraits for const pointers.
template<typename T>
class PointerLikeTypeTraits<const T*> {
public:
  static inline const void *getAsVoidPointer(const T* P) { return P; }
  static inline const T *getFromVoidPointer(const void *P) {
    return static_cast<const T*>(P);
  }
  static inline unsigned getNumLowBitsAvailable() { return 3; }
};
  
} // end namespace llvm

#endif
