//===- ConstantFolding.cpp - LLVM constant folder -------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements folding of constants for LLVM.  This implements the
// (internal) ConstantFolding.h interface, which is used by the
// ConstantExpr::get* methods to automatically fold constants when possible.
//
// The current constant folding implementation is implemented in two pieces: the
// template-based folder for simple primitive constants like ConstantInt, and
// the special case hackery that we use to symbolically evaluate expressions
// that use ConstantExprs.
//
//===----------------------------------------------------------------------===//

#include "ConstantFolding.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include <cmath>
using namespace llvm;

namespace {
  struct ConstRules {
    ConstRules() {}
    
    // Binary Operators...
    virtual Constant *add(const Constant *V1, const Constant *V2) const = 0;
    virtual Constant *sub(const Constant *V1, const Constant *V2) const = 0;
    virtual Constant *mul(const Constant *V1, const Constant *V2) const = 0;
    virtual Constant *div(const Constant *V1, const Constant *V2) const = 0;
    virtual Constant *rem(const Constant *V1, const Constant *V2) const = 0;
    virtual Constant *op_and(const Constant *V1, const Constant *V2) const = 0;
    virtual Constant *op_or (const Constant *V1, const Constant *V2) const = 0;
    virtual Constant *op_xor(const Constant *V1, const Constant *V2) const = 0;
    virtual Constant *shl(const Constant *V1, const Constant *V2) const = 0;
    virtual Constant *shr(const Constant *V1, const Constant *V2) const = 0;
    virtual Constant *lessthan(const Constant *V1, const Constant *V2) const =0;
    virtual Constant *equalto(const Constant *V1, const Constant *V2) const = 0;

    // Casting operators.
    virtual Constant *castToBool  (const Constant *V) const = 0;
    virtual Constant *castToSByte (const Constant *V) const = 0;
    virtual Constant *castToUByte (const Constant *V) const = 0;
    virtual Constant *castToShort (const Constant *V) const = 0;
    virtual Constant *castToUShort(const Constant *V) const = 0;
    virtual Constant *castToInt   (const Constant *V) const = 0;
    virtual Constant *castToUInt  (const Constant *V) const = 0;
    virtual Constant *castToLong  (const Constant *V) const = 0;
    virtual Constant *castToULong (const Constant *V) const = 0;
    virtual Constant *castToFloat (const Constant *V) const = 0;
    virtual Constant *castToDouble(const Constant *V) const = 0;
    virtual Constant *castToPointer(const Constant *V,
                                    const PointerType *Ty) const = 0;
    
    // ConstRules::get - Return an instance of ConstRules for the specified
    // constant operands.
    //
    static ConstRules &get(const Constant *V1, const Constant *V2);
  private:
    ConstRules(const ConstRules &);             // Do not implement
    ConstRules &operator=(const ConstRules &);  // Do not implement
  };
}


//===----------------------------------------------------------------------===//
//                             TemplateRules Class
//===----------------------------------------------------------------------===//
//
// TemplateRules - Implement a subclass of ConstRules that provides all 
// operations as noops.  All other rules classes inherit from this class so 
// that if functionality is needed in the future, it can simply be added here 
// and to ConstRules without changing anything else...
// 
// This class also provides subclasses with typesafe implementations of methods
// so that don't have to do type casting.
//
template<class ArgType, class SubClassName>
class TemplateRules : public ConstRules {

  //===--------------------------------------------------------------------===//
  // Redirecting functions that cast to the appropriate types
  //===--------------------------------------------------------------------===//

  virtual Constant *add(const Constant *V1, const Constant *V2) const { 
    return SubClassName::Add((const ArgType *)V1, (const ArgType *)V2);  
  }
  virtual Constant *sub(const Constant *V1, const Constant *V2) const { 
    return SubClassName::Sub((const ArgType *)V1, (const ArgType *)V2);  
  }
  virtual Constant *mul(const Constant *V1, const Constant *V2) const { 
    return SubClassName::Mul((const ArgType *)V1, (const ArgType *)V2);  
  }
  virtual Constant *div(const Constant *V1, const Constant *V2) const { 
    return SubClassName::Div((const ArgType *)V1, (const ArgType *)V2);  
  }
  virtual Constant *rem(const Constant *V1, const Constant *V2) const { 
    return SubClassName::Rem((const ArgType *)V1, (const ArgType *)V2);  
  }
  virtual Constant *op_and(const Constant *V1, const Constant *V2) const { 
    return SubClassName::And((const ArgType *)V1, (const ArgType *)V2);  
  }
  virtual Constant *op_or(const Constant *V1, const Constant *V2) const { 
    return SubClassName::Or((const ArgType *)V1, (const ArgType *)V2);  
  }
  virtual Constant *op_xor(const Constant *V1, const Constant *V2) const { 
    return SubClassName::Xor((const ArgType *)V1, (const ArgType *)V2);  
  }
  virtual Constant *shl(const Constant *V1, const Constant *V2) const { 
    return SubClassName::Shl((const ArgType *)V1, (const ArgType *)V2);  
  }
  virtual Constant *shr(const Constant *V1, const Constant *V2) const { 
    return SubClassName::Shr((const ArgType *)V1, (const ArgType *)V2);  
  }

  virtual Constant *lessthan(const Constant *V1, const Constant *V2) const { 
    return SubClassName::LessThan((const ArgType *)V1, (const ArgType *)V2);
  }
  virtual Constant *equalto(const Constant *V1, const Constant *V2) const { 
    return SubClassName::EqualTo((const ArgType *)V1, (const ArgType *)V2);
  }

  // Casting operators.  ick
  virtual Constant *castToBool(const Constant *V) const {
    return SubClassName::CastToBool((const ArgType*)V);
  }
  virtual Constant *castToSByte(const Constant *V) const {
    return SubClassName::CastToSByte((const ArgType*)V);
  }
  virtual Constant *castToUByte(const Constant *V) const {
    return SubClassName::CastToUByte((const ArgType*)V);
  }
  virtual Constant *castToShort(const Constant *V) const {
    return SubClassName::CastToShort((const ArgType*)V);
  }
  virtual Constant *castToUShort(const Constant *V) const {
    return SubClassName::CastToUShort((const ArgType*)V);
  }
  virtual Constant *castToInt(const Constant *V) const {
    return SubClassName::CastToInt((const ArgType*)V);
  }
  virtual Constant *castToUInt(const Constant *V) const {
    return SubClassName::CastToUInt((const ArgType*)V);
  }
  virtual Constant *castToLong(const Constant *V) const {
    return SubClassName::CastToLong((const ArgType*)V);
  }
  virtual Constant *castToULong(const Constant *V) const {
    return SubClassName::CastToULong((const ArgType*)V);
  }
  virtual Constant *castToFloat(const Constant *V) const {
    return SubClassName::CastToFloat((const ArgType*)V);
  }
  virtual Constant *castToDouble(const Constant *V) const {
    return SubClassName::CastToDouble((const ArgType*)V);
  }
  virtual Constant *castToPointer(const Constant *V, 
                                  const PointerType *Ty) const {
    return SubClassName::CastToPointer((const ArgType*)V, Ty);
  }

  //===--------------------------------------------------------------------===//
  // Default "noop" implementations
  //===--------------------------------------------------------------------===//

  static Constant *Add(const ArgType *V1, const ArgType *V2) { return 0; }
  static Constant *Sub(const ArgType *V1, const ArgType *V2) { return 0; }
  static Constant *Mul(const ArgType *V1, const ArgType *V2) { return 0; }
  static Constant *Div(const ArgType *V1, const ArgType *V2) { return 0; }
  static Constant *Rem(const ArgType *V1, const ArgType *V2) { return 0; }
  static Constant *And(const ArgType *V1, const ArgType *V2) { return 0; }
  static Constant *Or (const ArgType *V1, const ArgType *V2) { return 0; }
  static Constant *Xor(const ArgType *V1, const ArgType *V2) { return 0; }
  static Constant *Shl(const ArgType *V1, const ArgType *V2) { return 0; }
  static Constant *Shr(const ArgType *V1, const ArgType *V2) { return 0; }
  static Constant *LessThan(const ArgType *V1, const ArgType *V2) {
    return 0;
  }
  static Constant *EqualTo(const ArgType *V1, const ArgType *V2) {
    return 0;
  }

  // Casting operators.  ick
  static Constant *CastToBool  (const Constant *V) { return 0; }
  static Constant *CastToSByte (const Constant *V) { return 0; }
  static Constant *CastToUByte (const Constant *V) { return 0; }
  static Constant *CastToShort (const Constant *V) { return 0; }
  static Constant *CastToUShort(const Constant *V) { return 0; }
  static Constant *CastToInt   (const Constant *V) { return 0; }
  static Constant *CastToUInt  (const Constant *V) { return 0; }
  static Constant *CastToLong  (const Constant *V) { return 0; }
  static Constant *CastToULong (const Constant *V) { return 0; }
  static Constant *CastToFloat (const Constant *V) { return 0; }
  static Constant *CastToDouble(const Constant *V) { return 0; }
  static Constant *CastToPointer(const Constant *,
                                 const PointerType *) {return 0;}
};



//===----------------------------------------------------------------------===//
//                             EmptyRules Class
//===----------------------------------------------------------------------===//
//
// EmptyRules provides a concrete base class of ConstRules that does nothing
//
struct EmptyRules : public TemplateRules<Constant, EmptyRules> {
  static Constant *EqualTo(const Constant *V1, const Constant *V2) {
    if (V1 == V2) return ConstantBool::True;
    return 0;
  }
};



//===----------------------------------------------------------------------===//
//                              BoolRules Class
//===----------------------------------------------------------------------===//
//
// BoolRules provides a concrete base class of ConstRules for the 'bool' type.
//
struct BoolRules : public TemplateRules<ConstantBool, BoolRules> {

  static Constant *LessThan(const ConstantBool *V1, const ConstantBool *V2){
    return ConstantBool::get(V1->getValue() < V2->getValue());
  }

  static Constant *EqualTo(const Constant *V1, const Constant *V2) {
    return ConstantBool::get(V1 == V2);
  }

  static Constant *And(const ConstantBool *V1, const ConstantBool *V2) {
    return ConstantBool::get(V1->getValue() & V2->getValue());
  }

  static Constant *Or(const ConstantBool *V1, const ConstantBool *V2) {
    return ConstantBool::get(V1->getValue() | V2->getValue());
  }

  static Constant *Xor(const ConstantBool *V1, const ConstantBool *V2) {
    return ConstantBool::get(V1->getValue() ^ V2->getValue());
  }

  // Casting operators.  ick
#define DEF_CAST(TYPE, CLASS, CTYPE) \
  static Constant *CastTo##TYPE  (const ConstantBool *V) {    \
    return CLASS::get(Type::TYPE##Ty, (CTYPE)(bool)V->getValue()); \
  }

  DEF_CAST(Bool  , ConstantBool, bool)
  DEF_CAST(SByte , ConstantSInt, signed char)
  DEF_CAST(UByte , ConstantUInt, unsigned char)
  DEF_CAST(Short , ConstantSInt, signed short)
  DEF_CAST(UShort, ConstantUInt, unsigned short)
  DEF_CAST(Int   , ConstantSInt, signed int)
  DEF_CAST(UInt  , ConstantUInt, unsigned int)
  DEF_CAST(Long  , ConstantSInt, int64_t)
  DEF_CAST(ULong , ConstantUInt, uint64_t)
  DEF_CAST(Float , ConstantFP  , float)
  DEF_CAST(Double, ConstantFP  , double)
#undef DEF_CAST
};


//===----------------------------------------------------------------------===//
//                            NullPointerRules Class
//===----------------------------------------------------------------------===//
//
// NullPointerRules provides a concrete base class of ConstRules for null
// pointers.
//
struct NullPointerRules : public TemplateRules<ConstantPointerNull,
                                               NullPointerRules> {
  static Constant *EqualTo(const Constant *V1, const Constant *V2) {
    return ConstantBool::True;  // Null pointers are always equal
  }
  static Constant *CastToBool(const Constant *V) {
    return ConstantBool::False;
  }
  static Constant *CastToSByte (const Constant *V) {
    return ConstantSInt::get(Type::SByteTy, 0);
  }
  static Constant *CastToUByte (const Constant *V) {
    return ConstantUInt::get(Type::UByteTy, 0);
  }
  static Constant *CastToShort (const Constant *V) {
    return ConstantSInt::get(Type::ShortTy, 0);
  }
  static Constant *CastToUShort(const Constant *V) {
    return ConstantUInt::get(Type::UShortTy, 0);
  }
  static Constant *CastToInt   (const Constant *V) {
    return ConstantSInt::get(Type::IntTy, 0);
  }
  static Constant *CastToUInt  (const Constant *V) {
    return ConstantUInt::get(Type::UIntTy, 0);
  }
  static Constant *CastToLong  (const Constant *V) {
    return ConstantSInt::get(Type::LongTy, 0);
  }
  static Constant *CastToULong (const Constant *V) {
    return ConstantUInt::get(Type::ULongTy, 0);
  }
  static Constant *CastToFloat (const Constant *V) {
    return ConstantFP::get(Type::FloatTy, 0);
  }
  static Constant *CastToDouble(const Constant *V) {
    return ConstantFP::get(Type::DoubleTy, 0);
  }

  static Constant *CastToPointer(const ConstantPointerNull *V,
                                 const PointerType *PTy) {
    return ConstantPointerNull::get(PTy);
  }
};


//===----------------------------------------------------------------------===//
//                             DirectRules Class
//===----------------------------------------------------------------------===//
//
// DirectRules provides a concrete base classes of ConstRules for a variety of
// different types.  This allows the C++ compiler to automatically generate our
// constant handling operations in a typesafe and accurate manner.
//
template<class ConstantClass, class BuiltinType, Type **Ty, class SuperClass>
struct DirectRules : public TemplateRules<ConstantClass, SuperClass> {
  static Constant *Add(const ConstantClass *V1, const ConstantClass *V2) {
    BuiltinType R = (BuiltinType)V1->getValue() + (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, R);
  }

  static Constant *Sub(const ConstantClass *V1, const ConstantClass *V2) {
    BuiltinType R = (BuiltinType)V1->getValue() - (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, R);
  }

  static Constant *Mul(const ConstantClass *V1, const ConstantClass *V2) {
    BuiltinType R = (BuiltinType)V1->getValue() * (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, R);
  }

  static Constant *Div(const ConstantClass *V1, const ConstantClass *V2) {
    if (V2->isNullValue()) return 0;
    BuiltinType R = (BuiltinType)V1->getValue() / (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, R);
  }

  static Constant *LessThan(const ConstantClass *V1, const ConstantClass *V2) {
    bool R = (BuiltinType)V1->getValue() < (BuiltinType)V2->getValue();
    return ConstantBool::get(R);
  } 

  static Constant *EqualTo(const ConstantClass *V1, const ConstantClass *V2) {
    bool R = (BuiltinType)V1->getValue() == (BuiltinType)V2->getValue();
    return ConstantBool::get(R);
  }

  static Constant *CastToPointer(const ConstantClass *V,
                                 const PointerType *PTy) {
    if (V->isNullValue())    // Is it a FP or Integral null value?
      return ConstantPointerNull::get(PTy);
    return 0;  // Can't const prop other types of pointers
  }

  // Casting operators.  ick
#define DEF_CAST(TYPE, CLASS, CTYPE) \
  static Constant *CastTo##TYPE  (const ConstantClass *V) {    \
    return CLASS::get(Type::TYPE##Ty, (CTYPE)(BuiltinType)V->getValue()); \
  }

  DEF_CAST(Bool  , ConstantBool, bool)
  DEF_CAST(SByte , ConstantSInt, signed char)
  DEF_CAST(UByte , ConstantUInt, unsigned char)
  DEF_CAST(Short , ConstantSInt, signed short)
  DEF_CAST(UShort, ConstantUInt, unsigned short)
  DEF_CAST(Int   , ConstantSInt, signed int)
  DEF_CAST(UInt  , ConstantUInt, unsigned int)
  DEF_CAST(Long  , ConstantSInt, int64_t)
  DEF_CAST(ULong , ConstantUInt, uint64_t)
  DEF_CAST(Float , ConstantFP  , float)
  DEF_CAST(Double, ConstantFP  , double)
#undef DEF_CAST
};


//===----------------------------------------------------------------------===//
//                           DirectIntRules Class
//===----------------------------------------------------------------------===//
//
// DirectIntRules provides implementations of functions that are valid on
// integer types, but not all types in general.
//
template <class ConstantClass, class BuiltinType, Type **Ty>
struct DirectIntRules
  : public DirectRules<ConstantClass, BuiltinType, Ty,
                       DirectIntRules<ConstantClass, BuiltinType, Ty> > {

  static Constant *Div(const ConstantClass *V1, const ConstantClass *V2) {
    if (V2->isNullValue()) return 0;
    if (V2->isAllOnesValue() &&              // MIN_INT / -1
        (BuiltinType)V1->getValue() == -(BuiltinType)V1->getValue())
      return 0;
    BuiltinType R = (BuiltinType)V1->getValue() / (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, R);
  }

  static Constant *Rem(const ConstantClass *V1,
                       const ConstantClass *V2) {
    if (V2->isNullValue()) return 0;         // X / 0
    if (V2->isAllOnesValue() &&              // MIN_INT / -1
        (BuiltinType)V1->getValue() == -(BuiltinType)V1->getValue())
      return 0;
    BuiltinType R = (BuiltinType)V1->getValue() % (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, R);
  }

  static Constant *And(const ConstantClass *V1, const ConstantClass *V2) {
    BuiltinType R = (BuiltinType)V1->getValue() & (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, R);
  }
  static Constant *Or(const ConstantClass *V1, const ConstantClass *V2) {
    BuiltinType R = (BuiltinType)V1->getValue() | (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, R);
  }
  static Constant *Xor(const ConstantClass *V1, const ConstantClass *V2) {
    BuiltinType R = (BuiltinType)V1->getValue() ^ (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, R);
  }

  static Constant *Shl(const ConstantClass *V1, const ConstantClass *V2) {
    BuiltinType R = (BuiltinType)V1->getValue() << (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, R);
  }

  static Constant *Shr(const ConstantClass *V1, const ConstantClass *V2) {
    BuiltinType R = (BuiltinType)V1->getValue() >> (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, R);
  }
};


//===----------------------------------------------------------------------===//
//                           DirectFPRules Class
//===----------------------------------------------------------------------===//
//
/// DirectFPRules provides implementations of functions that are valid on
/// floating point types, but not all types in general.
///
template <class ConstantClass, class BuiltinType, Type **Ty>
struct DirectFPRules
  : public DirectRules<ConstantClass, BuiltinType, Ty,
                       DirectFPRules<ConstantClass, BuiltinType, Ty> > {
  static Constant *Rem(const ConstantClass *V1, const ConstantClass *V2) {
    if (V2->isNullValue()) return 0;
    BuiltinType Result = std::fmod((BuiltinType)V1->getValue(),
                                   (BuiltinType)V2->getValue());
    return ConstantClass::get(*Ty, Result);
  }
};


/// ConstRules::get - This method returns the constant rules implementation that
/// implements the semantics of the two specified constants.
ConstRules &ConstRules::get(const Constant *V1, const Constant *V2) {
  static EmptyRules       EmptyR;
  static BoolRules        BoolR;
  static NullPointerRules NullPointerR;
  static DirectIntRules<ConstantSInt,   signed char , &Type::SByteTy>  SByteR;
  static DirectIntRules<ConstantUInt, unsigned char , &Type::UByteTy>  UByteR;
  static DirectIntRules<ConstantSInt,   signed short, &Type::ShortTy>  ShortR;
  static DirectIntRules<ConstantUInt, unsigned short, &Type::UShortTy> UShortR;
  static DirectIntRules<ConstantSInt,   signed int  , &Type::IntTy>    IntR;
  static DirectIntRules<ConstantUInt, unsigned int  , &Type::UIntTy>   UIntR;
  static DirectIntRules<ConstantSInt,  int64_t      , &Type::LongTy>   LongR;
  static DirectIntRules<ConstantUInt, uint64_t      , &Type::ULongTy>  ULongR;
  static DirectFPRules <ConstantFP  , float         , &Type::FloatTy>  FloatR;
  static DirectFPRules <ConstantFP  , double        , &Type::DoubleTy> DoubleR;

  if (isa<ConstantExpr>(V1) || isa<ConstantExpr>(V2) ||
      isa<ConstantPointerRef>(V1) || isa<ConstantPointerRef>(V2))
    return EmptyR;

  switch (V1->getType()->getTypeID()) {
  default: assert(0 && "Unknown value type for constant folding!");
  case Type::BoolTyID:    return BoolR;
  case Type::PointerTyID: return NullPointerR;
  case Type::SByteTyID:   return SByteR;
  case Type::UByteTyID:   return UByteR;
  case Type::ShortTyID:   return ShortR;
  case Type::UShortTyID:  return UShortR;
  case Type::IntTyID:     return IntR;
  case Type::UIntTyID:    return UIntR;
  case Type::LongTyID:    return LongR;
  case Type::ULongTyID:   return ULongR;
  case Type::FloatTyID:   return FloatR;
  case Type::DoubleTyID:  return DoubleR;
  }
}


//===----------------------------------------------------------------------===//
//                ConstantFold*Instruction Implementations
//===----------------------------------------------------------------------===//
//
// These methods contain the special case hackery required to symbolically
// evaluate some constant expression cases, and use the ConstantRules class to
// evaluate normal constants.
//
static unsigned getSize(const Type *Ty) {
  unsigned S = Ty->getPrimitiveSize();
  return S ? S : 8;  // Treat pointers at 8 bytes
}

Constant *llvm::ConstantFoldCastInstruction(const Constant *V,
                                            const Type *DestTy) {
  if (V->getType() == DestTy) return (Constant*)V;

  // Cast of a global address to boolean is always true.
  if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(V))
    if (DestTy == Type::BoolTy)
      // FIXME: When we support 'external weak' references, we have to prevent
      // this transformation from happening.  In the meantime we avoid folding
      // any cast of an external symbol.
      if (!CPR->getValue()->isExternal())
        return ConstantBool::True;

  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
    if (CE->getOpcode() == Instruction::Cast) {
      Constant *Op = const_cast<Constant*>(CE->getOperand(0));
      // Try to not produce a cast of a cast, which is almost always redundant.
      if (!Op->getType()->isFloatingPoint() &&
          !CE->getType()->isFloatingPoint() &&
          !DestTy->isFloatingPoint()) {
        unsigned S1 = getSize(Op->getType()), S2 = getSize(CE->getType());
        unsigned S3 = getSize(DestTy);
        if (Op->getType() == DestTy && S3 >= S2)
          return Op;
        if (S1 >= S2 && S2 >= S3)
          return ConstantExpr::getCast(Op, DestTy);
        if (S1 <= S2 && S2 >= S3 && S1 <= S3)
          return ConstantExpr::getCast(Op, DestTy);
      }
    } else if (CE->getOpcode() == Instruction::GetElementPtr) {
      // If all of the indexes in the GEP are null values, there is no pointer
      // adjustment going on.  We might as well cast the source pointer.
      bool isAllNull = true;
      for (unsigned i = 1, e = CE->getNumOperands(); i != e; ++i)
        if (!CE->getOperand(i)->isNullValue()) {
          isAllNull = false;
          break;
        }
      if (isAllNull)
        return ConstantExpr::getCast(CE->getOperand(0), DestTy);
    }

  ConstRules &Rules = ConstRules::get(V, V);

  switch (DestTy->getTypeID()) {
  case Type::BoolTyID:    return Rules.castToBool(V);
  case Type::UByteTyID:   return Rules.castToUByte(V);
  case Type::SByteTyID:   return Rules.castToSByte(V);
  case Type::UShortTyID:  return Rules.castToUShort(V);
  case Type::ShortTyID:   return Rules.castToShort(V);
  case Type::UIntTyID:    return Rules.castToUInt(V);
  case Type::IntTyID:     return Rules.castToInt(V);
  case Type::ULongTyID:   return Rules.castToULong(V);
  case Type::LongTyID:    return Rules.castToLong(V);
  case Type::FloatTyID:   return Rules.castToFloat(V);
  case Type::DoubleTyID:  return Rules.castToDouble(V);
  case Type::PointerTyID:
    return Rules.castToPointer(V, cast<PointerType>(DestTy));
  default: return 0;
  }
}

Constant *llvm::ConstantFoldSelectInstruction(const Constant *Cond,
                                              const Constant *V1,
                                              const Constant *V2) {
  if (Cond == ConstantBool::True)
    return const_cast<Constant*>(V1);
  else if (Cond == ConstantBool::False)
    return const_cast<Constant*>(V2);
  return 0;
}


/// IdxCompare - Compare the two constants as though they were getelementptr
/// indices.  This allows coersion of the types to be the same thing.
///
/// If the two constants are the "same" (after coersion), return 0.  If the
/// first is less than the second, return -1, if the second is less than the
/// first, return 1.  If the constants are not integral, return -2.
///
static int IdxCompare(Constant *C1, Constant *C2) {
  if (C1 == C2) return 0;

  // Ok, we found a different index.  Are either of the operands
  // ConstantExprs?  If so, we can't do anything with them.
  if (!isa<ConstantInt>(C1) || !isa<ConstantInt>(C2))
    return -2; // don't know!
  
  // Ok, we have two differing integer indices.  Sign extend them to be the same
  // type.  Long is always big enough, so we use it.
  C1 = ConstantExpr::getSignExtend(C1, Type::LongTy);
  C2 = ConstantExpr::getSignExtend(C2, Type::LongTy);
  if (C1 == C2) return 0;  // Are they just differing types?

  // If they are really different, now that they are the same type, then we
  // found a difference!
  if (cast<ConstantSInt>(C1)->getValue() < cast<ConstantSInt>(C2)->getValue())
    return -1;
  else
    return 1;
}

/// evaluateRelation - This function determines if there is anything we can
/// decide about the two constants provided.  This doesn't need to handle simple
/// things like integer comparisons, but should instead handle ConstantExpr's
/// and ConstantPointerRef's.  If we can determine that the two constants have a
/// particular relation to each other, we should return the corresponding SetCC
/// code, otherwise return Instruction::BinaryOpsEnd.
///
/// To simplify this code we canonicalize the relation so that the first
/// operand is always the most "complex" of the two.  We consider simple
/// constants (like ConstantInt) to be the simplest, followed by
/// ConstantPointerRef's, followed by ConstantExpr's (the most complex).
///
static Instruction::BinaryOps evaluateRelation(const Constant *V1,
                                               const Constant *V2) {
  assert(V1->getType() == V2->getType() &&
         "Cannot compare different types of values!");
  if (V1 == V2) return Instruction::SetEQ;

  if (!isa<ConstantExpr>(V1) && !isa<ConstantPointerRef>(V1)) {
    // If the first operand is simple, swap operands.
    assert((isa<ConstantPointerRef>(V2) || isa<ConstantExpr>(V2)) &&
           "Simple cases should have been handled by caller!");
    Instruction::BinaryOps SwappedRelation = evaluateRelation(V2, V1);
    if (SwappedRelation != Instruction::BinaryOpsEnd)
      return SetCondInst::getSwappedCondition(SwappedRelation);

  } else if (const ConstantPointerRef *CPR1 = dyn_cast<ConstantPointerRef>(V1)){
    if (isa<ConstantExpr>(V2)) {  // Swap as necessary.
    Instruction::BinaryOps SwappedRelation = evaluateRelation(V2, V1);
    if (SwappedRelation != Instruction::BinaryOpsEnd)
      return SetCondInst::getSwappedCondition(SwappedRelation);
    else
      return Instruction::BinaryOpsEnd;
    }

    // Now we know that the RHS is a ConstantPointerRef or simple constant,
    // which (since the types must match) means that it's a ConstantPointerNull.
    if (const ConstantPointerRef *CPR2 = dyn_cast<ConstantPointerRef>(V2)) {
      assert(CPR1->getValue() != CPR2->getValue() &&
             "CPRs for the same value exist at different addresses??");
      // FIXME: If both globals are external weak, they might both be null!
      return Instruction::SetNE;
    } else {
      assert(isa<ConstantPointerNull>(V2) && "Canonicalization guarantee!");
      // Global can never be null.  FIXME: if we implement external weak
      // linkage, this is not necessarily true!
      return Instruction::SetNE;
    }

  } else {
    // Ok, the LHS is known to be a constantexpr.  The RHS can be any of a
    // constantexpr, a CPR, or a simple constant.
    const ConstantExpr *CE1 = cast<ConstantExpr>(V1);
    Constant *CE1Op0 = CE1->getOperand(0);

    switch (CE1->getOpcode()) {
    case Instruction::Cast:
      // If the cast is not actually changing bits, and the second operand is a
      // null pointer, do the comparison with the pre-casted value.
      if (V2->isNullValue() &&
          CE1->getType()->isLosslesslyConvertibleTo(CE1Op0->getType()))
        return evaluateRelation(CE1Op0,
                                Constant::getNullValue(CE1Op0->getType()));
      break;

    case Instruction::GetElementPtr:
      // Ok, since this is a getelementptr, we know that the constant has a
      // pointer type.  Check the various cases.
      if (isa<ConstantPointerNull>(V2)) {
        // If we are comparing a GEP to a null pointer, check to see if the base
        // of the GEP equals the null pointer.
        if (isa<ConstantPointerRef>(CE1Op0)) {
          // FIXME: this is not true when we have external weak references!
          // No offset can go from a global to a null pointer.
          return Instruction::SetGT;
        } else if (isa<ConstantPointerNull>(CE1Op0)) {
          // If we are indexing from a null pointer, check to see if we have any
          // non-zero indices.
          for (unsigned i = 1, e = CE1->getNumOperands(); i != e; ++i)
            if (!CE1->getOperand(i)->isNullValue())
              // Offsetting from null, must not be equal.
              return Instruction::SetGT;
          // Only zero indexes from null, must still be zero.
          return Instruction::SetEQ;
        }
        // Otherwise, we can't really say if the first operand is null or not.
      } else if (const ConstantPointerRef *CPR2 =
                                             dyn_cast<ConstantPointerRef>(V2)) {
        if (isa<ConstantPointerNull>(CE1Op0)) {
          // FIXME: This is not true with external weak references.
          return Instruction::SetLT;
        } else if (const ConstantPointerRef *CPR1 =
                   dyn_cast<ConstantPointerRef>(CE1Op0)) {
          if (CPR1 == CPR2) {
            // If this is a getelementptr of the same global, then it must be
            // different.  Because the types must match, the getelementptr could
            // only have at most one index, and because we fold getelementptr's
            // with a single zero index, it must be nonzero.
            assert(CE1->getNumOperands() == 2 &&
                   !CE1->getOperand(1)->isNullValue() &&
                   "Suprising getelementptr!");
            return Instruction::SetGT;
          } else {
            // If they are different globals, we don't know what the value is,
            // but they can't be equal.
            return Instruction::SetNE;
          }
        }
      } else {
        const ConstantExpr *CE2 = cast<ConstantExpr>(V2);
        const Constant *CE2Op0 = CE2->getOperand(0);

        // There are MANY other foldings that we could perform here.  They will
        // probably be added on demand, as they seem needed.
        switch (CE2->getOpcode()) {
        default: break;
        case Instruction::GetElementPtr:
          // By far the most common case to handle is when the base pointers are
          // obviously to the same or different globals.
          if (isa<ConstantPointerRef>(CE1Op0) &&
              isa<ConstantPointerRef>(CE2Op0)) {
            if (CE1Op0 != CE2Op0) // Don't know relative ordering, but not equal
              return Instruction::SetNE;
            // Ok, we know that both getelementptr instructions are based on the
            // same global.  From this, we can precisely determine the relative
            // ordering of the resultant pointers.
            unsigned i = 1;
            
            // Compare all of the operands the GEP's have in common.
            for (;i != CE1->getNumOperands() && i != CE2->getNumOperands(); ++i)
              switch (IdxCompare(CE1->getOperand(i), CE2->getOperand(i))) {
              case -1: return Instruction::SetLT;
              case 1:  return Instruction::SetGT;
              case -2: return Instruction::BinaryOpsEnd;
              }

            // Ok, we ran out of things they have in common.  If any leftovers
            // are non-zero then we have a difference, otherwise we are equal.
            for (; i < CE1->getNumOperands(); ++i)
              if (!CE1->getOperand(i)->isNullValue())
                return Instruction::SetGT;
            for (; i < CE2->getNumOperands(); ++i)
              if (!CE2->getOperand(i)->isNullValue())
                return Instruction::SetLT;
            return Instruction::SetEQ;
          }
        }
      }
      
    default:
      break;
    }
  }

  return Instruction::BinaryOpsEnd;
}

Constant *llvm::ConstantFoldBinaryInstruction(unsigned Opcode,
                                              const Constant *V1,
                                              const Constant *V2) {
  Constant *C = 0;
  switch (Opcode) {
  default:                   break;
  case Instruction::Add:     C = ConstRules::get(V1, V2).add(V1, V2); break;
  case Instruction::Sub:     C = ConstRules::get(V1, V2).sub(V1, V2); break;
  case Instruction::Mul:     C = ConstRules::get(V1, V2).mul(V1, V2); break;
  case Instruction::Div:     C = ConstRules::get(V1, V2).div(V1, V2); break;
  case Instruction::Rem:     C = ConstRules::get(V1, V2).rem(V1, V2); break;
  case Instruction::And:     C = ConstRules::get(V1, V2).op_and(V1, V2); break;
  case Instruction::Or:      C = ConstRules::get(V1, V2).op_or (V1, V2); break;
  case Instruction::Xor:     C = ConstRules::get(V1, V2).op_xor(V1, V2); break;
  case Instruction::Shl:     C = ConstRules::get(V1, V2).shl(V1, V2); break;
  case Instruction::Shr:     C = ConstRules::get(V1, V2).shr(V1, V2); break;
  case Instruction::SetEQ:   C = ConstRules::get(V1, V2).equalto(V1, V2); break;
  case Instruction::SetLT:   C = ConstRules::get(V1, V2).lessthan(V1, V2);break;
  case Instruction::SetGT:   C = ConstRules::get(V1, V2).lessthan(V2, V1);break;
  case Instruction::SetNE:   // V1 != V2  ===  !(V1 == V2)
    C = ConstRules::get(V1, V2).equalto(V1, V2);
    if (C) return ConstantExpr::get(Instruction::Xor, C, ConstantBool::True);
    break;
  case Instruction::SetLE:   // V1 <= V2  ===  !(V2 < V1)
    C = ConstRules::get(V1, V2).lessthan(V2, V1);
    if (C) return ConstantExpr::get(Instruction::Xor, C, ConstantBool::True);
    break;
  case Instruction::SetGE:   // V1 >= V2  ===  !(V1 < V2)
    C = ConstRules::get(V1, V2).lessthan(V1, V2);
    if (C) return ConstantExpr::get(Instruction::Xor, C, ConstantBool::True);
    break;
  }

  // If we successfully folded the expression, return it now.
  if (C) return C;

  if (SetCondInst::isRelational(Opcode))
    switch (evaluateRelation(V1, V2)) {
    default: assert(0 && "Unknown relational!");
    case Instruction::BinaryOpsEnd:
      break;  // Couldn't determine anything about these constants.
    case Instruction::SetEQ:   // We know the constants are equal!
      // If we know the constants are equal, we can decide the result of this
      // computation precisely.
      return ConstantBool::get(Opcode == Instruction::SetEQ ||
                               Opcode == Instruction::SetLE ||
                               Opcode == Instruction::SetGE);
    case Instruction::SetLT:
      // If we know that V1 < V2, we can decide the result of this computation
      // precisely.
      return ConstantBool::get(Opcode == Instruction::SetLT ||
                               Opcode == Instruction::SetNE ||
                               Opcode == Instruction::SetLE);
    case Instruction::SetGT:
      // If we know that V1 > V2, we can decide the result of this computation
      // precisely.
      return ConstantBool::get(Opcode == Instruction::SetGT ||
                               Opcode == Instruction::SetNE ||
                               Opcode == Instruction::SetGE);
    case Instruction::SetLE:
      // If we know that V1 <= V2, we can only partially decide this relation.
      if (Opcode == Instruction::SetGT) return ConstantBool::False;
      if (Opcode == Instruction::SetLT) return ConstantBool::True;
      break;

    case Instruction::SetGE:
      // If we know that V1 >= V2, we can only partially decide this relation.
      if (Opcode == Instruction::SetLT) return ConstantBool::False;
      if (Opcode == Instruction::SetGT) return ConstantBool::True;
      break;
      
    case Instruction::SetNE:
      // If we know that V1 != V2, we can only partially decide this relation.
      if (Opcode == Instruction::SetEQ) return ConstantBool::False;
      if (Opcode == Instruction::SetNE) return ConstantBool::True;
      break;
    }

  if (const ConstantExpr *CE1 = dyn_cast<ConstantExpr>(V1)) {
    if (const ConstantExpr *CE2 = dyn_cast<ConstantExpr>(V2)) {
      // There are many possible foldings we could do here.  We should probably
      // at least fold add of a pointer with an integer into the appropriate
      // getelementptr.  This will improve alias analysis a bit.




    } else {
      // Just implement a couple of simple identities.
      switch (Opcode) {
      case Instruction::Add:
        if (V2->isNullValue()) return const_cast<Constant*>(V1);  // X + 0 == X
        break;
      case Instruction::Sub:
        if (V2->isNullValue()) return const_cast<Constant*>(V1);  // X - 0 == X
        break;
      case Instruction::Mul:
        if (V2->isNullValue()) return const_cast<Constant*>(V2);  // X * 0 == 0
        if (const ConstantInt *CI = dyn_cast<ConstantInt>(V2))
          if (CI->getRawValue() == 1)
            return const_cast<Constant*>(V1);                     // X * 1 == X
        break;
      case Instruction::Div:
        if (const ConstantInt *CI = dyn_cast<ConstantInt>(V2))
          if (CI->getRawValue() == 1)
            return const_cast<Constant*>(V1);                     // X / 1 == X
        break;
      case Instruction::Rem:
        if (const ConstantInt *CI = dyn_cast<ConstantInt>(V2))
          if (CI->getRawValue() == 1)
            return Constant::getNullValue(CI->getType()); // X % 1 == 0
        break;
      case Instruction::And:
        if (cast<ConstantIntegral>(V2)->isAllOnesValue())
          return const_cast<Constant*>(V1);                       // X & -1 == X
        if (V2->isNullValue()) return const_cast<Constant*>(V2);  // X & 0 == 0
        if (CE1->getOpcode() == Instruction::Cast &&
            isa<ConstantPointerRef>(CE1->getOperand(0))) {
          ConstantPointerRef *CPR =cast<ConstantPointerRef>(CE1->getOperand(0));

          // Functions are at least 4-byte aligned.  If and'ing the address of a
          // function with a constant < 4, fold it to zero.
          if (const ConstantInt *CI = dyn_cast<ConstantInt>(V2))
            if (CI->getRawValue() < 4 && isa<Function>(CPR->getValue()))
              return Constant::getNullValue(CI->getType());
        }
        break;
      case Instruction::Or:
        if (V2->isNullValue()) return const_cast<Constant*>(V1);  // X | 0 == X
        if (cast<ConstantIntegral>(V2)->isAllOnesValue())
          return const_cast<Constant*>(V2);  // X | -1 == -1
        break;
      case Instruction::Xor:
        if (V2->isNullValue()) return const_cast<Constant*>(V1);  // X ^ 0 == X
        break;
      }
    }

  } else if (const ConstantExpr *CE2 = dyn_cast<ConstantExpr>(V2)) {
    // If V2 is a constant expr and V1 isn't, flop them around and fold the
    // other way if possible.
    switch (Opcode) {
    case Instruction::Add:
    case Instruction::Mul:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::SetEQ:
    case Instruction::SetNE:
      // No change of opcode required.
      return ConstantFoldBinaryInstruction(Opcode, V2, V1);

    case Instruction::SetLT:
    case Instruction::SetGT:
    case Instruction::SetLE:
    case Instruction::SetGE:
      // Change the opcode as necessary to swap the operands.
      Opcode = SetCondInst::getSwappedCondition((Instruction::BinaryOps)Opcode);
      return ConstantFoldBinaryInstruction(Opcode, V2, V1);

    case Instruction::Shl:
    case Instruction::Shr:
    case Instruction::Sub:
    case Instruction::Div:
    case Instruction::Rem:
    default:  // These instructions cannot be flopped around.
      break;
    }
  }
  return 0;
}

Constant *llvm::ConstantFoldGetElementPtr(const Constant *C,
                                        const std::vector<Constant*> &IdxList) {
  if (IdxList.size() == 0 ||
      (IdxList.size() == 1 && IdxList[0]->isNullValue()))
    return const_cast<Constant*>(C);

  if (C->isNullValue()) {
    bool isNull = true;
    for (unsigned i = 0, e = IdxList.size(); i != e; ++i)
      if (!IdxList[i]->isNullValue()) {
        isNull = false;
        break;
      }
    if (isNull) {
      std::vector<Value*> VIdxList(IdxList.begin(), IdxList.end());
      const Type *Ty = GetElementPtrInst::getIndexedType(C->getType(), VIdxList,
                                                         true);
      assert(Ty != 0 && "Invalid indices for GEP!");
      return ConstantPointerNull::get(PointerType::get(Ty));
    }
  }

  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(const_cast<Constant*>(C))) {
    // Combine Indices - If the source pointer to this getelementptr instruction
    // is a getelementptr instruction, combine the indices of the two
    // getelementptr instructions into a single instruction.
    //
    if (CE->getOpcode() == Instruction::GetElementPtr) {
      const Type *LastTy = 0;
      for (gep_type_iterator I = gep_type_begin(CE), E = gep_type_end(CE);
           I != E; ++I)
        LastTy = *I;

      if ((LastTy && isa<ArrayType>(LastTy)) || IdxList[0]->isNullValue()) {
        std::vector<Constant*> NewIndices;
        NewIndices.reserve(IdxList.size() + CE->getNumOperands());
        for (unsigned i = 1, e = CE->getNumOperands()-1; i != e; ++i)
          NewIndices.push_back(cast<Constant>(CE->getOperand(i)));

        // Add the last index of the source with the first index of the new GEP.
        // Make sure to handle the case when they are actually different types.
        Constant *Combined = CE->getOperand(CE->getNumOperands()-1);
        if (!IdxList[0]->isNullValue())   // Otherwise it must be an array
          Combined = 
            ConstantExpr::get(Instruction::Add,
                              ConstantExpr::getCast(IdxList[0], Type::LongTy),
                              ConstantExpr::getCast(Combined, Type::LongTy));
        
        NewIndices.push_back(Combined);
        NewIndices.insert(NewIndices.end(), IdxList.begin()+1, IdxList.end());
        return ConstantExpr::getGetElementPtr(CE->getOperand(0), NewIndices);
      }
    }

    // Implement folding of:
    //    int* getelementptr ([2 x int]* cast ([3 x int]* %X to [2 x int]*),
    //                        long 0, long 0)
    // To: int* getelementptr ([3 x int]* %X, long 0, long 0)
    //
    if (CE->getOpcode() == Instruction::Cast && IdxList.size() > 1 &&
        IdxList[0]->isNullValue())
      if (const PointerType *SPT = 
          dyn_cast<PointerType>(CE->getOperand(0)->getType()))
        if (const ArrayType *SAT = dyn_cast<ArrayType>(SPT->getElementType()))
          if (const ArrayType *CAT =
              dyn_cast<ArrayType>(cast<PointerType>(C->getType())->getElementType()))
            if (CAT->getElementType() == SAT->getElementType())
              return ConstantExpr::getGetElementPtr(
                      (Constant*)CE->getOperand(0), IdxList);
  }
  return 0;
}

