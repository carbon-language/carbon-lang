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

#include "clang/AST/ASTContext.h"
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

  /// The alignment to use when accessing this lvalue.
  unsigned short Alignment;

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

  /// TBAAInfo - TBAA information to attach to dereferences of this LValue.
  llvm::MDNode *TBAAInfo;

private:
  void Initialize(Qualifiers Quals, unsigned Alignment = 0,
                  llvm::MDNode *TBAAInfo = 0) {
    this->Quals = Quals;
    this->Alignment = Alignment;
    assert(this->Alignment == Alignment && "Alignment exceeds allowed max!");

    // Initialize Objective-C flags.
    this->Ivar = this->ObjIsArray = this->NonGC = this->GlobalObjCRef = false;
    this->ThreadLocalRef = false;
    this->BaseIvarExp = 0;
    this->TBAAInfo = TBAAInfo;
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
  void setObjCIvar(bool Value) { Ivar = Value; }

  bool isObjCArray() const { return ObjIsArray; }
  void setObjCArray(bool Value) { ObjIsArray = Value; }

  bool isNonGC () const { return NonGC; }
  void setNonGC(bool Value) { NonGC = Value; }

  bool isGlobalObjCRef() const { return GlobalObjCRef; }
  void setGlobalObjCRef(bool Value) { GlobalObjCRef = Value; }

  bool isThreadLocalRef() const { return ThreadLocalRef; }
  void setThreadLocalRef(bool Value) { ThreadLocalRef = Value;}

  bool isObjCWeak() const {
    return Quals.getObjCGCAttr() == Qualifiers::Weak;
  }
  bool isObjCStrong() const {
    return Quals.getObjCGCAttr() == Qualifiers::Strong;
  }
  
  Expr *getBaseIvarExp() const { return BaseIvarExp; }
  void setBaseIvarExp(Expr *V) { BaseIvarExp = V; }

  llvm::MDNode *getTBAAInfo() const { return TBAAInfo; }
  void setTBAAInfo(llvm::MDNode *N) { TBAAInfo = N; }

  const Qualifiers &getQuals() const { return Quals; }
  Qualifiers &getQuals() { return Quals; }

  unsigned getAddressSpace() const { return Quals.getAddressSpace(); }

  unsigned getAlignment() const { return Alignment; }

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

  static LValue MakeAddr(llvm::Value *V, QualType T, unsigned Alignment,
                         ASTContext &Context,
                         llvm::MDNode *TBAAInfo = 0) {
    Qualifiers Quals = Context.getCanonicalType(T).getQualifiers();
    Quals.setObjCGCAttr(Context.getObjCGCAttrKind(T));

    LValue R;
    R.LVType = Simple;
    R.V = V;
    R.Initialize(Quals, Alignment, TBAAInfo);
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

/// An aggregate value slot.
class AggValueSlot {
  /// The address.
  llvm::Value *Addr;
  
  // Associated flags.
  bool VolatileFlag : 1;
  bool LifetimeFlag : 1;
  bool RequiresGCollection : 1;

public:
  /// ignored - Returns an aggregate value slot indicating that the
  /// aggregate value is being ignored.
  static AggValueSlot ignored() {
    AggValueSlot AV;
    AV.Addr = 0;
    AV.VolatileFlag = AV.LifetimeFlag = AV.RequiresGCollection = 0;
    return AV;
  }

  /// forAddr - Make a slot for an aggregate value.
  ///
  /// \param Volatile - true if the slot should be volatile-initialized
  /// \param LifetimeExternallyManaged - true if the slot's lifetime
  ///   is being externally managed; false if a destructor should be
  ///   registered for any temporaries evaluated into the slot
  /// \param RequiresGCollection - true if the slot is located
  ///   somewhere that ObjC GC calls should be emitted for
  static AggValueSlot forAddr(llvm::Value *Addr, bool Volatile,
                              bool LifetimeExternallyManaged,
                              bool RequiresGCollection=false) {
    AggValueSlot AV;
    AV.Addr = Addr;
    AV.VolatileFlag = Volatile;
    AV.LifetimeFlag = LifetimeExternallyManaged;
    AV.RequiresGCollection = RequiresGCollection;
    return AV;
  }

  static AggValueSlot forLValue(LValue LV, bool LifetimeExternallyManaged,
                                bool RequiresGCollection=false) {
    return forAddr(LV.getAddress(), LV.isVolatileQualified(),
                   LifetimeExternallyManaged, RequiresGCollection);
  }

  bool isLifetimeExternallyManaged() const {
    return LifetimeFlag;
  }
  void setLifetimeExternallyManaged() {
    LifetimeFlag = true;
  }

  bool isVolatile() const {
    return VolatileFlag;
  }

  bool requiresGCollection() const {
    return RequiresGCollection;
  }
  
  llvm::Value *getAddr() const {
    return Addr;
  }

  bool isIgnored() const {
    return Addr == 0;
  }

  RValue asRValue() const {
    return RValue::getAggregate(getAddr(), isVolatile());
  }
  
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
