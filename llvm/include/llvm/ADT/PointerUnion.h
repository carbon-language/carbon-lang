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
  static inline int getPointerUnionTypeNum(PT1 *P) { return 0; }
  template <typename PT1, typename PT2>
  static inline int getPointerUnionTypeNum(PT2 *P) { return 1; }
  template <typename PT1, typename PT2>
  static inline int getPointerUnionTypeNum(...) { return -1; }
  
  
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
  ///    Z = P.get<double*>();  // runtime assertion failure (regardless of tag)
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
      Val.setPointer(
         const_cast<void *>(PointerLikeTypeTraits<PT1>::getAsVoidPointer(V)));
      Val.setInt(0);
    }
    PointerUnion(PT2 V) {
      Val.setPointer(
         const_cast<void *>(PointerLikeTypeTraits<PT2>::getAsVoidPointer(V)));
      Val.setInt(1);
    }
    
    /// isNull - Return true if the pointer held in the union is null,
    /// regardless of which type it is.
    bool isNull() const { return Val.getPointer() == 0; }
    operator bool() const { return !isNull(); }

    /// is<T>() return true if the Union currently holds the type matching T.
    template<typename T>
    int is() const {
      int TyNo = ::llvm::getPointerUnionTypeNum<PT1, PT2>((T*)0);
      assert(TyNo != -1 && "Type query could never succeed on PointerUnion!");
      return static_cast<int>(Val.getInt()) == TyNo;
    }
    
    /// get<T>() - Return the value of the specified pointer type. If the
    /// specified pointer type is incorrect, assert.
    template<typename T>
    T get() const {
      assert(is<T>() && "Invalid accessor called");
      return PointerLikeTypeTraits<T>::getFromVoidPointer(Val.getPointer());
    }
    
    /// dyn_cast<T>() - If the current value is of the specified pointer type,
    /// return it, otherwise return null.
    template<typename T>
    T dyn_cast() const {
      if (is<T>()) return get<T>();
      return T();
    }
    
    /// Assignment operators - Allow assigning into this union from either
    /// pointer type, setting the discriminator to remember what it came from.
    const PointerUnion &operator=(const PT1 &RHS) {
      Val.setPointer(
         const_cast<void *>(PointerLikeTypeTraits<PT1>::getAsVoidPointer(RHS)));
      Val.setInt(0);
      return *this;
    }
    const PointerUnion &operator=(const PT2 &RHS) {
      Val.setPointer(
        const_cast<void *>(PointerLikeTypeTraits<PT2>::getAsVoidPointer(RHS)));
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
  
  // Teach SmallPtrSet that PointerUnion is "basically a pointer", that has
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
        PointerLikeTypeTraits<typename PointerUnion<PT1,PT2>::ValTy>
          ::NumLowBitsAvailable
    };
  };
  
  
  /// PointerUnion3 - This is a pointer union of three pointer types.  See
  /// documentation for PointerUnion for usage.
  template <typename PT1, typename PT2, typename PT3>
  class PointerUnion3 {
  public:
    typedef PointerUnion<PT1, PT2> InnerUnion;
    typedef PointerUnion<InnerUnion, PT3> ValTy;
  private:
    ValTy Val;
  public:
    PointerUnion3() {}
    
    PointerUnion3(PT1 V) {
      Val = InnerUnion(V);
    }
    PointerUnion3(PT2 V) {
      Val = InnerUnion(V);
    }
    PointerUnion3(PT3 V) {
      Val = V;
    }
    
    /// isNull - Return true if the pointer held in the union is null,
    /// regardless of which type it is.
    bool isNull() const { return Val.isNull(); }
    operator bool() const { return !isNull(); }
    
    /// is<T>() return true if the Union currently holds the type matching T.
    template<typename T>
    int is() const {
      // Is it PT1/PT2?
      if (::llvm::getPointerUnionTypeNum<PT1, PT2>((T*)0) != -1)
        return Val.template is<InnerUnion>() && 
               Val.template get<InnerUnion>().template is<T>();
      return Val.template is<T>();
    }
    
    /// get<T>() - Return the value of the specified pointer type. If the
    /// specified pointer type is incorrect, assert.
    template<typename T>
    T get() const {
      assert(is<T>() && "Invalid accessor called");
      // Is it PT1/PT2?
      if (::llvm::getPointerUnionTypeNum<PT1, PT2>((T*)0) != -1)
        return Val.template get<InnerUnion>().template get<T>();
      
      return Val.template get<T>();
    }
    
    /// dyn_cast<T>() - If the current value is of the specified pointer type,
    /// return it, otherwise return null.
    template<typename T>
    T dyn_cast() const {
      if (is<T>()) return get<T>();
      return T();
    }
    
    /// Assignment operators - Allow assigning into this union from either
    /// pointer type, setting the discriminator to remember what it came from.
    const PointerUnion3 &operator=(const PT1 &RHS) {
      Val = InnerUnion(RHS);
      return *this;
    }
    const PointerUnion3 &operator=(const PT2 &RHS) {
      Val = InnerUnion(RHS);
      return *this;
    }
    const PointerUnion3 &operator=(const PT3 &RHS) {
      Val = RHS;
      return *this;
    }
    
    void *getOpaqueValue() const { return Val.getOpaqueValue(); }
    static PointerUnion3 getFromOpaqueValue(void *VP) {
      PointerUnion3 V;
      V.Val = ValTy::getFromOpaqueValue(VP);
      return V;
    }
  };
 
  // Teach SmallPtrSet that PointerUnion3 is "basically a pointer", that has
  // # low bits available = min(PT1bits,PT2bits,PT2bits)-2.
  template<typename PT1, typename PT2, typename PT3>
  class PointerLikeTypeTraits<PointerUnion3<PT1, PT2, PT3> > {
  public:
    static inline void *
    getAsVoidPointer(const PointerUnion3<PT1, PT2, PT3> &P) {
      return P.getOpaqueValue();
    }
    static inline PointerUnion3<PT1, PT2, PT3>
    getFromVoidPointer(void *P) {
      return PointerUnion3<PT1, PT2, PT3>::getFromOpaqueValue(P);
    }
    
    // The number of bits available are the min of the two pointer types.
    enum {
      NumLowBitsAvailable = 
        PointerLikeTypeTraits<typename PointerUnion3<PT1, PT2, PT3>::ValTy>
          ::NumLowBitsAvailable
    };
  };

  /// PointerUnion4 - This is a pointer union of four pointer types.  See
  /// documentation for PointerUnion for usage.
  template <typename PT1, typename PT2, typename PT3, typename PT4>
  class PointerUnion4 {
  public:
    typedef PointerUnion<PT1, PT2> InnerUnion1;
    typedef PointerUnion<PT3, PT4> InnerUnion2;
    typedef PointerUnion<InnerUnion1, InnerUnion2> ValTy;
  private:
    ValTy Val;
  public:
    PointerUnion4() {}
    
    PointerUnion4(PT1 V) {
      Val = InnerUnion1(V);
    }
    PointerUnion4(PT2 V) {
      Val = InnerUnion1(V);
    }
    PointerUnion4(PT3 V) {
      Val = InnerUnion2(V);
    }
    PointerUnion4(PT4 V) {
      Val = InnerUnion2(V);
    }
    
    /// isNull - Return true if the pointer held in the union is null,
    /// regardless of which type it is.
    bool isNull() const { return Val.isNull(); }
    operator bool() const { return !isNull(); }
    
    /// is<T>() return true if the Union currently holds the type matching T.
    template<typename T>
    int is() const {
      // Is it PT1/PT2?
      if (::llvm::getPointerUnionTypeNum<PT1, PT2>((T*)0) != -1)
        return Val.template is<InnerUnion1>() && 
               Val.template get<InnerUnion1>().template is<T>();
      return Val.template is<InnerUnion2>() && 
             Val.template get<InnerUnion2>().template is<T>();
    }
    
    /// get<T>() - Return the value of the specified pointer type. If the
    /// specified pointer type is incorrect, assert.
    template<typename T>
    T get() const {
      assert(is<T>() && "Invalid accessor called");
      // Is it PT1/PT2?
      if (::llvm::getPointerUnionTypeNum<PT1, PT2>((T*)0) != -1)
        return Val.template get<InnerUnion1>().template get<T>();
      
      return Val.template get<InnerUnion2>().template get<T>();
    }
    
    /// dyn_cast<T>() - If the current value is of the specified pointer type,
    /// return it, otherwise return null.
    template<typename T>
    T dyn_cast() const {
      if (is<T>()) return get<T>();
      return T();
    }
    
    /// Assignment operators - Allow assigning into this union from either
    /// pointer type, setting the discriminator to remember what it came from.
    const PointerUnion4 &operator=(const PT1 &RHS) {
      Val = InnerUnion1(RHS);
      return *this;
    }
    const PointerUnion4 &operator=(const PT2 &RHS) {
      Val = InnerUnion1(RHS);
      return *this;
    }
    const PointerUnion4 &operator=(const PT3 &RHS) {
      Val = InnerUnion2(RHS);
      return *this;
    }
    const PointerUnion4 &operator=(const PT4 &RHS) {
      Val = InnerUnion2(RHS);
      return *this;
    }
    
    void *getOpaqueValue() const { return Val.getOpaqueValue(); }
    static PointerUnion4 getFromOpaqueValue(void *VP) {
      PointerUnion4 V;
      V.Val = ValTy::getFromOpaqueValue(VP);
      return V;
    }
  };
  
  // Teach SmallPtrSet that PointerUnion4 is "basically a pointer", that has
  // # low bits available = min(PT1bits,PT2bits,PT2bits)-2.
  template<typename PT1, typename PT2, typename PT3, typename PT4>
  class PointerLikeTypeTraits<PointerUnion4<PT1, PT2, PT3, PT4> > {
  public:
    static inline void *
    getAsVoidPointer(const PointerUnion4<PT1, PT2, PT3, PT4> &P) {
      return P.getOpaqueValue();
    }
    static inline PointerUnion4<PT1, PT2, PT3, PT4>
    getFromVoidPointer(void *P) {
      return PointerUnion4<PT1, PT2, PT3, PT4>::getFromOpaqueValue(P);
    }
    
    // The number of bits available are the min of the two pointer types.
    enum {
      NumLowBitsAvailable = 
        PointerLikeTypeTraits<typename PointerUnion4<PT1, PT2, PT3, PT4>::ValTy>
          ::NumLowBitsAvailable
    };
  };
}

#endif
