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
#include "llvm/iPHINode.h"
#include "llvm/InstrTypes.h"
#include "llvm/DerivedTypes.h"
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

  switch (V1->getType()->getPrimitiveID()) {
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

  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
    if (CE->getOpcode() == Instruction::Cast) {
      Constant *Op = const_cast<Constant*>(CE->getOperand(0));
      // Try to not produce a cast of a cast, which is almost always redundant.
      if (!Op->getType()->isFloatingPoint() &&
          !CE->getType()->isFloatingPoint() &&
          !DestTy->getType()->isFloatingPoint()) {
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

  switch (DestTy->getPrimitiveID()) {
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

Constant *llvm::ConstantFoldBinaryInstruction(unsigned Opcode,
                                              const Constant *V1,
                                              const Constant *V2) {
  Constant *C;
  switch (Opcode) {
  default:                   return 0;
  case Instruction::Add:     return ConstRules::get(V1, V2).add(V1, V2);
  case Instruction::Sub:     return ConstRules::get(V1, V2).sub(V1, V2);
  case Instruction::Mul:     return ConstRules::get(V1, V2).mul(V1, V2);
  case Instruction::Div:     return ConstRules::get(V1, V2).div(V1, V2);
  case Instruction::Rem:     return ConstRules::get(V1, V2).rem(V1, V2);
  case Instruction::And:     return ConstRules::get(V1, V2).op_and(V1, V2);
  case Instruction::Or:      return ConstRules::get(V1, V2).op_or (V1, V2);
  case Instruction::Xor:     return ConstRules::get(V1, V2).op_xor(V1, V2);

  case Instruction::Shl:     return ConstRules::get(V1, V2).shl(V1, V2);
  case Instruction::Shr:     return ConstRules::get(V1, V2).shr(V1, V2);

  case Instruction::SetEQ:   return ConstRules::get(V1, V2).equalto(V1, V2);
  case Instruction::SetLT:   return ConstRules::get(V1, V2).lessthan(V1, V2);
  case Instruction::SetGT:   return ConstRules::get(V1, V2).lessthan(V2, V1);
  case Instruction::SetNE:   // V1 != V2  ===  !(V1 == V2)
    C = ConstRules::get(V1, V2).equalto(V1, V2);
    break;
  case Instruction::SetLE:   // V1 <= V2  ===  !(V2 < V1)
    C = ConstRules::get(V1, V2).lessthan(V2, V1);
    break;
  case Instruction::SetGE:   // V1 >= V2  ===  !(V1 < V2)
    C = ConstRules::get(V1, V2).lessthan(V1, V2);
    break;
  }

  // If the folder broke out of the switch statement, invert the boolean
  // constant value, if it exists, and return it.
  if (!C) return 0;
  return ConstantExpr::get(Instruction::Xor, ConstantBool::True, C);
}

Constant *llvm::ConstantFoldGetElementPtr(const Constant *C,
                                        const std::vector<Constant*> &IdxList) {
  if (IdxList.size() == 0 ||
      (IdxList.size() == 1 && IdxList[0]->isNullValue()))
    return const_cast<Constant*>(C);

  // TODO If C is null and all idx's are null, return null of the right type.


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

