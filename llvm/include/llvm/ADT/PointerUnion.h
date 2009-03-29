//===- llvm/ADT/PointerUnion.h - Discriminated Union of 2 Ptrs --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PointerUnion class, which is a discriminated union of
// pointer types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_POINTERUNION_H
#define LLVM_ADT_POINTERUNION_H

#include "llvm/ADT/PointerIntPair.h"

namespace llvm {

  /// getPointerUnionTypeNum - If the argument has type PT1* or PT2* return
  /// false or true respectively.
  template <typename PT1, typename PT2>
  static inline bool getPointerUnionTypeNum(PT1 *P) { return false; }
  template <typename PT1, typename PT2>
  static inline bool getPointerUnionTypeNum(PT2 *P) { return true; }
  // Enable, if we could use static_assert.
  //template <typename PT1, typename PT2>
  //static inline bool getPointerUnionTypeNum(...) { abort() }
  
  
  /// Provide PointerLikeTypeTraits for void* that is used by PointerUnion
  /// for the two template arguments.
  template <typename PT1, typename PT2>
  class PointerUnionUIntTraits {
  public:
    static inline void *getAsVoidPointer(void *P) { return P; }
    static inline void *getFromVoidPointer(void *P) { return P; }
    enum {
      PT1BitsAv = PointerLikeTypeTraits<PT1>::NumLowBitsAvailable,
      PT2BitsAv = PointerLikeTypeTraits<PT2>::NumLowBitsAvailable,
      NumLowBitsAvailable = PT1BitsAv < PT2BitsAv ? PT1BitsAv : PT2BitsAv
    };
  };
  
  /// PointerUnion - This implements a discriminated union of two pointer types,
  /// and keeps the discriminator bit-mangled into the low bits of the pointer.
  /// This allows the implementation to be extremely efficient in space, but
  /// permits a very natural and type-safe API.
  ///
  /// Common use patterns would be something like this:
  ///    PointerUnion<int*, float*> P;
  ///    P = (int*)0;
  ///    printf("%d %d", P.is<int*>(), P.is<float*>());  // prints "1 0"
  ///    X = P.get<int*>();     // ok.
  ///    Y = P.get<float*>();   // runtime assertion failure.
  ///    Z = P.get<double*>();  // does not compile.
  ///    P = (float*)0;
  ///    Y = P.get<float*>();   // ok.
  ///    X = P.get<int*>();     // runtime assertion failure.
  template <typename PT1, typename PT2>
  class PointerUnion {
  public:
    typedef PointerIntPair<void*, 1, bool, 
                           PointerUnionUIntTraits<PT1,PT2> > ValTy;
  private:
    ValTy Val;
  public:
    PointerUnion() {}
    
    PointerUnion(PT1 V) {
      Val.setPointer(V);
      Val.setInt(0);
    }
    PointerUnion(PT2 V) {
      Val.setPointer(V);
      Val.setInt(1);
    }
    
    bool isNull() const { return Val.getPointer() == 0; }
    
    template<typename T>
    int is() const {
      return Val.getInt() == ::llvm::getPointerUnionTypeNum<PT1, PT2>((T*)0);
    }
    template<typename T>
    T get() const {
      assert(is<T>() && "Invalid accessor called");
      return static_cast<T>(Val.getPointer());
    }
    
    const PointerUnion &operator=(const PT1 &RHS) {
      Val.setPointer(RHS);
      Val.setInt(0);
      return *this;
    }
    const PointerUnion &operator=(const PT2 &RHS) {
      Val.setPointer(RHS);
      Val.setInt(1);
      return *this;
    }
    
    void *getOpaqueValue() const { return Val.getOpaqueValue(); }
    static PointerUnion getFromOpaqueValue(void *VP) {
      PointerUnion V;
      V.Val = ValTy::getFromOpaqueValue(VP);
      return V;
    }
  };
  
  // Teach SmallPtrSet that PointerIntPair is "basically a pointer", that has
  // # low bits available = min(PT1bits,PT2bits)-1.
  template<typename PT1, typename PT2>
  class PointerLikeTypeTraits<PointerUnion<PT1, PT2> > {
  public:
    static inline void *
    getAsVoidPointer(const PointerUnion<PT1, PT2> &P) {
      return P.getOpaqueValue();
    }
    static inline PointerUnion<PT1, PT2>
    getFromVoidPointer(void *P) {
      return PointerUnion<PT1, PT2>::getFromOpaqueValue(P);
    }
    
    // The number of bits available are the min of the two pointer types.
    enum {
      NumLowBitsAvailable = 
        PointerUnion<PT1,PT2>::ValTy::NumLowBitsAvailable
    };
  };
}

#endif
