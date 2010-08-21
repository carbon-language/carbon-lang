//===-- CGValue.h - LLVM CodeGen wrappers for llvm::Value* ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These classes implement wrappers around llvm::Value in order to
// fully represent the range of values for C L- and R- values.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGVALUE_H
#define CLANG_CODEGEN_CGVALUE_H

#include "clang/AST/Type.h"

namespace llvm {
  class Constant;
  class Value;
}

namespace clang {
  class ObjCPropertyRefExpr;
  class ObjCImplicitSetterGetterRefExpr;

namespace CodeGen {
  class CGBitFieldInfo;

/// RValue - This trivial value class is used to represent the result of an
/// expression that is evaluated.  It can be one of three things: either a
/// simple LLVM SSA value, a pair of SSA values for complex numbers, or the
/// address of an aggregate value in memory.
class RValue {
  enum Flavor { Scalar, Complex, Aggregate };

  // Stores first value and flavor.
  llvm::PointerIntPair<llvm::Value *, 2, Flavor> V1;
  // Stores second value and volatility.
  llvm::PointerIntPair<llvm::Value *, 1, bool> V2;

public:
  bool isScalar() const { return V1.getInt() == Scalar; }
  bool isComplex() const { return V1.getInt() == Complex; }
  bool isAggregate() const { return V1.getInt() == Aggregate; }

  bool isVolatileQualified() const { return V2.getInt(); }

  /// getScalarVal() - Return the Value* of this scalar value.
  llvm::Value *getScalarVal() const {
    assert(isScalar() && "Not a scalar!");
    return V1.getPointer();
  }

  /// getComplexVal - Return the real/imag components of this complex value.
  ///
  std::pair<llvm::Value *, llvm::Value *> getComplexVal() const {
    return std::make_pair(V1.getPointer(), V2.getPointer());
  }

  /// getAggregateAddr() - Return the Value* of the address of the aggregate.
  llvm::Value *getAggregateAddr() const {
    assert(isAggregate() && "Not an aggregate!");
    return V1.getPointer();
  }

  static RValue get(llvm::Value *V) {
    RValue ER;
    ER.V1.setPointer(V);
    ER.V1.setInt(Scalar);
    ER.V2.setInt(false);
    return ER;
  }
  static RValue getComplex(llvm::Value *V1, llvm::Value *V2) {
    RValue ER;
    ER.V1.setPointer(V1);
    ER.V2.setPointer(V2);
    ER.V1.setInt(Complex);
    ER.V2.setInt(false);
    return ER;
  }
  static RValue getComplex(const std::pair<llvm::Value *, llvm::Value *> &C) {
    return getComplex(C.first, C.second);
  }
  // FIXME: Aggregate rvalues need to retain information about whether they are
  // volatile or not.  Remove default to find all places that probably get this
  // wrong.
  static RValue getAggregate(llvm::Value *V, bool Volatile = false) {
    RValue ER;
    ER.V1.setPointer(V);
    ER.V1.setInt(Aggregate);
    ER.V2.setInt(Volatile);
    return ER;
  }
};


/// LValue - This represents an lvalue references.  Because C/C++ allow
/// bitfields, this is not a simple LLVM pointer, it may be a pointer plus a
/// bitrange.
class LValue {
  // FIXME: alignment?

  enum {
    Simple,       // This is a normal l-value, use getAddress().
    VectorElt,    // This is a vector element l-value (V[i]), use getVector*
    BitField,     // This is a bitfield l-value, use getBitfield*.
    ExtVectorElt, // This is an extended vector subset, use getExtVectorComp
    PropertyRef,  // This is an Objective-C property reference, use
                  // getPropertyRefExpr
    KVCRef        // This is an objective-c 'implicit' property ref,
                  // use getKVCRefExpr
  } LVType;

  llvm::Value *V;

  union {
    // Index into a vector subscript: V[i]
    llvm::Value *VectorIdx;

    // ExtVector element subset: V.xyx
    llvm::Constant *VectorElts;

    // BitField start bit and size
    const CGBitFieldInfo *BitFieldInfo;

    // Obj-C property reference expression
    const ObjCPropertyRefExpr *PropertyRefExpr;

    // ObjC 'implicit' property reference expression
    const ObjCImplicitSetterGetterRefExpr *KVCRefExpr;
  };

  // 'const' is unused here
  Qualifiers Quals;

  // objective-c's ivar
  bool Ivar:1;
  
  // objective-c's ivar is an array
  bool ObjIsArray:1;

  // LValue is non-gc'able for any reason, including being a parameter or local
  // variable.
  bool NonGC: 1;

  // Lvalue is a global reference of an objective-c object
  bool GlobalObjCRef : 1;
  
  // Lvalue is a thread local reference
  bool ThreadLocalRef : 1;

  Expr *BaseIvarExp;
private:
  void Initialize(Qualifiers Quals) {
    this->Quals = Quals;
    
    // FIXME: Convenient place to set objc flags to 0. This should really be
    // done in a user-defined constructor instead.
    this->Ivar = this->ObjIsArray = this->NonGC = this->GlobalObjCRef = false;
    this->ThreadLocalRef = false;
    this->BaseIvarExp = 0;
  }

public:
  bool isSimple() const { return LVType == Simple; }
  bool isVectorElt() const { return LVType == VectorElt; }
  bool isBitField() const { return LVType == BitField; }
  bool isExtVectorElt() const { return LVType == ExtVectorElt; }
  bool isPropertyRef() const { return LVType == PropertyRef; }
  bool isKVCRef() const { return LVType == KVCRef; }

  bool isVolatileQualified() const { return Quals.hasVolatile(); }
  bool isRestrictQualified() const { return Quals.hasRestrict(); }
  unsigned getVRQualifiers() const {
    return Quals.getCVRQualifiers() & ~Qualifiers::Const;
  }

  bool isObjCIvar() const { return Ivar; }
  bool isObjCArray() const { return ObjIsArray; }
  bool isNonGC () const { return NonGC; }
  bool isGlobalObjCRef() const { return GlobalObjCRef; }
  bool isThreadLocalRef() const { return ThreadLocalRef; }
  bool isObjCWeak() const { return Quals.getObjCGCAttr() == Qualifiers::Weak; }
  bool isObjCStrong() const { return Quals.getObjCGCAttr() == Qualifiers::Strong; }
  
  Expr *getBaseIvarExp() const { return BaseIvarExp; }
  void setBaseIvarExp(Expr *V) { BaseIvarExp = V; }

  unsigned getAddressSpace() const { return Quals.getAddressSpace(); }

  static void SetObjCIvar(LValue& R, bool iValue) {
    R.Ivar = iValue;
  }
  static void SetObjCArray(LValue& R, bool iValue) {
    R.ObjIsArray = iValue;
  }
  static void SetGlobalObjCRef(LValue& R, bool iValue) {
    R.GlobalObjCRef = iValue;
  }
  static void SetThreadLocalRef(LValue& R, bool iValue) {
    R.ThreadLocalRef = iValue;
  }
  static void SetObjCNonGC(LValue& R, bool iValue) {
    R.NonGC = iValue;
  }

  // simple lvalue
  llvm::Value *getAddress() const { assert(isSimple()); return V; }

  // vector elt lvalue
  llvm::Value *getVectorAddr() const { assert(isVectorElt()); return V; }
  llvm::Value *getVectorIdx() const { assert(isVectorElt()); return VectorIdx; }

  // extended vector elements.
  llvm::Value *getExtVectorAddr() const { assert(isExtVectorElt()); return V; }
  llvm::Constant *getExtVectorElts() const {
    assert(isExtVectorElt());
    return VectorElts;
  }

  // bitfield lvalue
  llvm::Value *getBitFieldBaseAddr() const {
    assert(isBitField());
    return V;
  }
  const CGBitFieldInfo &getBitFieldInfo() const {
    assert(isBitField());
    return *BitFieldInfo;
  }

  // property ref lvalue
  const ObjCPropertyRefExpr *getPropertyRefExpr() const {
    assert(isPropertyRef());
    return PropertyRefExpr;
  }

  // 'implicit' property ref lvalue
  const ObjCImplicitSetterGetterRefExpr *getKVCRefExpr() const {
    assert(isKVCRef());
    return KVCRefExpr;
  }

  static LValue MakeAddr(llvm::Value *V, Qualifiers Quals) {
    LValue R;
    R.LVType = Simple;
    R.V = V;
    R.Initialize(Quals);
    return R;
  }

  static LValue MakeVectorElt(llvm::Value *Vec, llvm::Value *Idx,
                              unsigned CVR) {
    LValue R;
    R.LVType = VectorElt;
    R.V = Vec;
    R.VectorIdx = Idx;
    R.Initialize(Qualifiers::fromCVRMask(CVR));
    return R;
  }

  static LValue MakeExtVectorElt(llvm::Value *Vec, llvm::Constant *Elts,
                                 unsigned CVR) {
    LValue R;
    R.LVType = ExtVectorElt;
    R.V = Vec;
    R.VectorElts = Elts;
    R.Initialize(Qualifiers::fromCVRMask(CVR));
    return R;
  }

  /// \brief Create a new object to represent a bit-field access.
  ///
  /// \param BaseValue - The base address of the structure containing the
  /// bit-field.
  /// \param Info - The information describing how to perform the bit-field
  /// access.
  static LValue MakeBitfield(llvm::Value *BaseValue, const CGBitFieldInfo &Info,
                             unsigned CVR) {
    LValue R;
    R.LVType = BitField;
    R.V = BaseValue;
    R.BitFieldInfo = &Info;
    R.Initialize(Qualifiers::fromCVRMask(CVR));
    return R;
  }

  // FIXME: It is probably bad that we aren't emitting the target when we build
  // the lvalue. However, this complicates the code a bit, and I haven't figured
  // out how to make it go wrong yet.
  static LValue MakePropertyRef(const ObjCPropertyRefExpr *E,
                                unsigned CVR) {
    LValue R;
    R.LVType = PropertyRef;
    R.PropertyRefExpr = E;
    R.Initialize(Qualifiers::fromCVRMask(CVR));
    return R;
  }

  static LValue MakeKVCRef(const ObjCImplicitSetterGetterRefExpr *E,
                           unsigned CVR) {
    LValue R;
    R.LVType = KVCRef;
    R.KVCRefExpr = E;
    R.Initialize(Qualifiers::fromCVRMask(CVR));
    return R;
  }
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
