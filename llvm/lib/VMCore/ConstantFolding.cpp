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
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Visibility.h"
#include <limits>
#include <cmath>
using namespace llvm;

namespace {
  struct VISIBILITY_HIDDEN ConstRules {
    ConstRules() {}
    virtual ~ConstRules() {}

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
namespace {
template<class ArgType, class SubClassName>
class VISIBILITY_HIDDEN TemplateRules : public ConstRules {


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

public:
  virtual ~TemplateRules() {}
};
}  // end anonymous namespace


//===----------------------------------------------------------------------===//
//                             EmptyRules Class
//===----------------------------------------------------------------------===//
//
// EmptyRules provides a concrete base class of ConstRules that does nothing
//
namespace {
struct VISIBILITY_HIDDEN EmptyRules
  : public TemplateRules<Constant, EmptyRules> {
  static Constant *EqualTo(const Constant *V1, const Constant *V2) {
    if (V1 == V2) return ConstantBool::True;
    return 0;
  }
};
}  // end anonymous namespace



//===----------------------------------------------------------------------===//
//                              BoolRules Class
//===----------------------------------------------------------------------===//
//
// BoolRules provides a concrete base class of ConstRules for the 'bool' type.
//
namespace {
struct VISIBILITY_HIDDEN BoolRules
  : public TemplateRules<ConstantBool, BoolRules> {

  static Constant *LessThan(const ConstantBool *V1, const ConstantBool *V2) {
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
}  // end anonymous namespace


//===----------------------------------------------------------------------===//
//                            NullPointerRules Class
//===----------------------------------------------------------------------===//
//
// NullPointerRules provides a concrete base class of ConstRules for null
// pointers.
//
namespace {
struct VISIBILITY_HIDDEN NullPointerRules
  : public TemplateRules<ConstantPointerNull, NullPointerRules> {
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
}  // end anonymous namespace

//===----------------------------------------------------------------------===//
//                          ConstantPackedRules Class
//===----------------------------------------------------------------------===//

/// DoVectorOp - Given two packed constants and a function pointer, apply the
/// function pointer to each element pair, producing a new ConstantPacked
/// constant.
static Constant *EvalVectorOp(const ConstantPacked *V1, 
                              const ConstantPacked *V2,
                              Constant *(*FP)(Constant*, Constant*)) {
  std::vector<Constant*> Res;
  for (unsigned i = 0, e = V1->getNumOperands(); i != e; ++i)
    Res.push_back(FP(const_cast<Constant*>(V1->getOperand(i)),
                     const_cast<Constant*>(V2->getOperand(i))));
  return ConstantPacked::get(Res);
}

/// PackedTypeRules provides a concrete base class of ConstRules for
/// ConstantPacked operands.
///
namespace {
struct VISIBILITY_HIDDEN ConstantPackedRules
  : public TemplateRules<ConstantPacked, ConstantPackedRules> {
  
  static Constant *Add(const ConstantPacked *V1, const ConstantPacked *V2) {
    return EvalVectorOp(V1, V2, ConstantExpr::getAdd);
  }
  static Constant *Sub(const ConstantPacked *V1, const ConstantPacked *V2) {
    return EvalVectorOp(V1, V2, ConstantExpr::getSub);
  }
  static Constant *Mul(const ConstantPacked *V1, const ConstantPacked *V2) {
    return EvalVectorOp(V1, V2, ConstantExpr::getMul);
  }
  static Constant *Div(const ConstantPacked *V1, const ConstantPacked *V2) {
    return EvalVectorOp(V1, V2, ConstantExpr::getDiv);
  }
  static Constant *Rem(const ConstantPacked *V1, const ConstantPacked *V2) {
    return EvalVectorOp(V1, V2, ConstantExpr::getRem);
  }
  static Constant *And(const ConstantPacked *V1, const ConstantPacked *V2) {
    return EvalVectorOp(V1, V2, ConstantExpr::getAnd);
  }
  static Constant *Or (const ConstantPacked *V1, const ConstantPacked *V2) {
    return EvalVectorOp(V1, V2, ConstantExpr::getOr);
  }
  static Constant *Xor(const ConstantPacked *V1, const ConstantPacked *V2) {
    return EvalVectorOp(V1, V2, ConstantExpr::getXor);
  }
  static Constant *Shl(const ConstantPacked *V1, const ConstantPacked *V2) {
    return EvalVectorOp(V1, V2, ConstantExpr::getShl);
  }
  static Constant *Shr(const ConstantPacked *V1, const ConstantPacked *V2) {
    return EvalVectorOp(V1, V2, ConstantExpr::getShr);
  }
  static Constant *LessThan(const ConstantPacked *V1, const ConstantPacked *V2){
    return 0;
  }
  static Constant *EqualTo(const ConstantPacked *V1, const ConstantPacked *V2) {
    for (unsigned i = 0, e = V1->getNumOperands(); i != e; ++i) {
      Constant *C = 
        ConstantExpr::getSetEQ(const_cast<Constant*>(V1->getOperand(i)),
                               const_cast<Constant*>(V2->getOperand(i)));
      if (ConstantBool *CB = dyn_cast<ConstantBool>(C))
        return CB;
    }
    // Otherwise, could not decide from any element pairs.
    return 0;
  }
};
}  // end anonymous namespace


//===----------------------------------------------------------------------===//
//                          GeneralPackedRules Class
//===----------------------------------------------------------------------===//

/// GeneralPackedRules provides a concrete base class of ConstRules for
/// PackedType operands, where both operands are not ConstantPacked.  The usual
/// cause for this is that one operand is a ConstantAggregateZero.
///
namespace {
struct VISIBILITY_HIDDEN GeneralPackedRules
  : public TemplateRules<Constant, GeneralPackedRules> {
};
}  // end anonymous namespace


//===----------------------------------------------------------------------===//
//                             DirectRules Class
//===----------------------------------------------------------------------===//
//
// DirectRules provides a concrete base classes of ConstRules for a variety of
// different types.  This allows the C++ compiler to automatically generate our
// constant handling operations in a typesafe and accurate manner.
//
namespace {
template<class ConstantClass, class BuiltinType, Type **Ty, class SuperClass>
struct VISIBILITY_HIDDEN DirectRules
  : public TemplateRules<ConstantClass, SuperClass> {
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
}  // end anonymous namespace


//===----------------------------------------------------------------------===//
//                           DirectIntRules Class
//===----------------------------------------------------------------------===//
//
// DirectIntRules provides implementations of functions that are valid on
// integer types, but not all types in general.
//
namespace {
template <class ConstantClass, class BuiltinType, Type **Ty>
struct VISIBILITY_HIDDEN DirectIntRules
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
}  // end anonymous namespace


//===----------------------------------------------------------------------===//
//                           DirectFPRules Class
//===----------------------------------------------------------------------===//
//
/// DirectFPRules provides implementations of functions that are valid on
/// floating point types, but not all types in general.
///
namespace {
template <class ConstantClass, class BuiltinType, Type **Ty>
struct VISIBILITY_HIDDEN DirectFPRules
  : public DirectRules<ConstantClass, BuiltinType, Ty,
                       DirectFPRules<ConstantClass, BuiltinType, Ty> > {
  static Constant *Rem(const ConstantClass *V1, const ConstantClass *V2) {
    if (V2->isNullValue()) return 0;
    BuiltinType Result = std::fmod((BuiltinType)V1->getValue(),
                                   (BuiltinType)V2->getValue());
    return ConstantClass::get(*Ty, Result);
  }
  static Constant *Div(const ConstantClass *V1, const ConstantClass *V2) {
    BuiltinType inf = std::numeric_limits<BuiltinType>::infinity();
    if (V2->isExactlyValue(0.0)) return ConstantClass::get(*Ty, inf);
    if (V2->isExactlyValue(-0.0)) return ConstantClass::get(*Ty, -inf);
    BuiltinType R = (BuiltinType)V1->getValue() / (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, R);
  }
};
}  // end anonymous namespace


/// ConstRules::get - This method returns the constant rules implementation that
/// implements the semantics of the two specified constants.
ConstRules &ConstRules::get(const Constant *V1, const Constant *V2) {
  static EmptyRules       EmptyR;
  static BoolRules        BoolR;
  static NullPointerRules NullPointerR;
  static ConstantPackedRules ConstantPackedR;
  static GeneralPackedRules GeneralPackedR;
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
      isa<GlobalValue>(V1) || isa<GlobalValue>(V2) ||
      isa<UndefValue>(V1) || isa<UndefValue>(V2))
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
  case Type::PackedTyID:
    if (isa<ConstantPacked>(V1) && isa<ConstantPacked>(V2))
      return ConstantPackedR;
    return GeneralPackedR;  // Constant folding rules for ConstantAggregateZero.
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

/// CastConstantPacked - Convert the specified ConstantPacked node to the
/// specified packed type.  At this point, we know that the elements of the
/// input packed constant are all simple integer or FP values.
static Constant *CastConstantPacked(ConstantPacked *CP,
                                    const PackedType *DstTy) {
  unsigned SrcNumElts = CP->getType()->getNumElements();
  unsigned DstNumElts = DstTy->getNumElements();
  const Type *SrcEltTy = CP->getType()->getElementType();
  const Type *DstEltTy = DstTy->getElementType();
  
  // If both vectors have the same number of elements (thus, the elements
  // are the same size), perform the conversion now.
  if (SrcNumElts == DstNumElts) {
    std::vector<Constant*> Result;
    
    // If the src and dest elements are both integers, just cast each one
    // which will do the appropriate bit-convert.
    if (SrcEltTy->isIntegral() && DstEltTy->isIntegral()) {
      for (unsigned i = 0; i != SrcNumElts; ++i)
        Result.push_back(ConstantExpr::getCast(CP->getOperand(i),
                                               DstEltTy));
      return ConstantPacked::get(Result);
    }
    
    if (SrcEltTy->isIntegral()) {
      // Otherwise, this is an int-to-fp cast.
      assert(DstEltTy->isFloatingPoint());
      if (DstEltTy->getTypeID() == Type::DoubleTyID) {
        for (unsigned i = 0; i != SrcNumElts; ++i) {
          double V =
            BitsToDouble(cast<ConstantInt>(CP->getOperand(i))->getRawValue());
          Result.push_back(ConstantFP::get(Type::DoubleTy, V));
        }
        return ConstantPacked::get(Result);
      }
      assert(DstEltTy == Type::FloatTy && "Unknown fp type!");
      for (unsigned i = 0; i != SrcNumElts; ++i) {
        float V =
        BitsToFloat(cast<ConstantInt>(CP->getOperand(i))->getRawValue());
        Result.push_back(ConstantFP::get(Type::FloatTy, V));
      }
      return ConstantPacked::get(Result);
    }
    
    // Otherwise, this is an fp-to-int cast.
    assert(SrcEltTy->isFloatingPoint() && DstEltTy->isIntegral());
    
    if (SrcEltTy->getTypeID() == Type::DoubleTyID) {
      for (unsigned i = 0; i != SrcNumElts; ++i) {
        uint64_t V =
          DoubleToBits(cast<ConstantFP>(CP->getOperand(i))->getValue());
        Constant *C = ConstantUInt::get(Type::ULongTy, V);
        Result.push_back(ConstantExpr::getCast(C, DstEltTy));
      }
      return ConstantPacked::get(Result);
    }

    assert(SrcEltTy->getTypeID() == Type::FloatTyID);
    for (unsigned i = 0; i != SrcNumElts; ++i) {
      unsigned V = FloatToBits(cast<ConstantFP>(CP->getOperand(i))->getValue());
      Constant *C = ConstantUInt::get(Type::UIntTy, V);
      Result.push_back(ConstantExpr::getCast(C, DstEltTy));
    }
    return ConstantPacked::get(Result);
  }
  
  // Otherwise, this is a cast that changes element count and size.  Handle
  // casts which shrink the elements here.
  
  // FIXME: We need to know endianness to do this!
  
  return 0;
}


Constant *llvm::ConstantFoldCastInstruction(const Constant *V,
                                            const Type *DestTy) {
  if (V->getType() == DestTy) return (Constant*)V;

  // Cast of a global address to boolean is always true.
  if (const GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    if (DestTy == Type::BoolTy)
      // FIXME: When we support 'external weak' references, we have to prevent
      // this transformation from happening.  This code will need to be updated
      // to ignore external weak symbols when we support it.
      return ConstantBool::True;
  } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
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
  } else if (isa<UndefValue>(V)) {
    return UndefValue::get(DestTy);
  }

  // Check to see if we are casting an pointer to an aggregate to a pointer to
  // the first element.  If so, return the appropriate GEP instruction.
  if (const PointerType *PTy = dyn_cast<PointerType>(V->getType()))
    if (const PointerType *DPTy = dyn_cast<PointerType>(DestTy)) {
      std::vector<Value*> IdxList;
      IdxList.push_back(Constant::getNullValue(Type::IntTy));
      const Type *ElTy = PTy->getElementType();
      while (ElTy != DPTy->getElementType()) {
        if (const StructType *STy = dyn_cast<StructType>(ElTy)) {
          if (STy->getNumElements() == 0) break;
          ElTy = STy->getElementType(0);
          IdxList.push_back(Constant::getNullValue(Type::UIntTy));
        } else if (const SequentialType *STy = dyn_cast<SequentialType>(ElTy)) {
          if (isa<PointerType>(ElTy)) break;  // Can't index into pointers!
          ElTy = STy->getElementType();
          IdxList.push_back(IdxList[0]);
        } else {
          break;
        }
      }

      if (ElTy == DPTy->getElementType())
        return ConstantExpr::getGetElementPtr(const_cast<Constant*>(V),IdxList);
    }
      
  // Handle casts from one packed constant to another.  We know that the src and
  // dest type have the same size.
  if (const PackedType *DestPTy = dyn_cast<PackedType>(DestTy)) {
    if (const PackedType *SrcTy = dyn_cast<PackedType>(V->getType())) {
      assert(DestPTy->getElementType()->getPrimitiveSizeInBits() * 
                 DestPTy->getNumElements()  ==
             SrcTy->getElementType()->getPrimitiveSizeInBits() * 
             SrcTy->getNumElements() && "Not cast between same sized vectors!");
      if (isa<ConstantAggregateZero>(V))
        return Constant::getNullValue(DestTy);
      if (isa<UndefValue>(V))
        return UndefValue::get(DestTy);
      if (const ConstantPacked *CP = dyn_cast<ConstantPacked>(V)) {
        // This is a cast from a ConstantPacked of one type to a ConstantPacked
        // of another type.  Check to see if all elements of the input are
        // simple.
        bool AllSimpleConstants = true;
        for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i) {
          if (!isa<ConstantInt>(CP->getOperand(i)) &&
              !isa<ConstantFP>(CP->getOperand(i))) {
            AllSimpleConstants = false;
            break;
          }
        }
            
        // If all of the elements are simple constants, we can fold this.
        if (AllSimpleConstants)
          return CastConstantPacked(const_cast<ConstantPacked*>(CP), DestPTy);
      }
    }
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

  if (isa<UndefValue>(V1)) return const_cast<Constant*>(V2);
  if (isa<UndefValue>(V2)) return const_cast<Constant*>(V1);
  if (isa<UndefValue>(Cond)) return const_cast<Constant*>(V1);
  if (V1 == V2) return const_cast<Constant*>(V1);
  return 0;
}

Constant *llvm::ConstantFoldExtractElementInstruction(const Constant *Val,
                                                      const Constant *Idx) {
  if (isa<UndefValue>(Val))  // ee(undef, x) -> undef
    return UndefValue::get(cast<PackedType>(Val->getType())->getElementType());
  if (Val->isNullValue())  // ee(zero, x) -> zero
    return Constant::getNullValue(
                          cast<PackedType>(Val->getType())->getElementType());
  
  if (const ConstantPacked *CVal = dyn_cast<ConstantPacked>(Val)) {
    if (const ConstantUInt *CIdx = dyn_cast<ConstantUInt>(Idx)) {
      return const_cast<Constant*>(CVal->getOperand(CIdx->getValue()));
    } else if (isa<UndefValue>(Idx)) {
      // ee({w,x,y,z}, undef) -> w (an arbitrary value).
      return const_cast<Constant*>(CVal->getOperand(0));
    }
  }
  return 0;
}

Constant *llvm::ConstantFoldInsertElementInstruction(const Constant *Val,
                                                     const Constant *Elt,
                                                     const Constant *Idx) {
  const ConstantUInt *CIdx = dyn_cast<ConstantUInt>(Idx);
  if (!CIdx) return 0;
  unsigned idxVal = CIdx->getValue();
  if (const UndefValue *UVal = dyn_cast<UndefValue>(Val)) {
    // Insertion of scalar constant into packed undef
    // Optimize away insertion of undef
    if (isa<UndefValue>(Elt))
      return const_cast<Constant*>(Val);
    // Otherwise break the aggregate undef into multiple undefs and do
    // the insertion
    unsigned numOps = 
      cast<PackedType>(Val->getType())->getNumElements();
    std::vector<Constant*> Ops; 
    Ops.reserve(numOps);
    for (unsigned i = 0; i < numOps; ++i) {
      const Constant *Op =
        (i == idxVal) ? Elt : UndefValue::get(Elt->getType());
      Ops.push_back(const_cast<Constant*>(Op));
    }
    return ConstantPacked::get(Ops);
  }
  if (const ConstantAggregateZero *CVal =
      dyn_cast<ConstantAggregateZero>(Val)) {
    // Insertion of scalar constant into packed aggregate zero
    // Optimize away insertion of zero
    if (Elt->isNullValue())
      return const_cast<Constant*>(Val);
    // Otherwise break the aggregate zero into multiple zeros and do
    // the insertion
    unsigned numOps = 
      cast<PackedType>(Val->getType())->getNumElements();
    std::vector<Constant*> Ops; 
    Ops.reserve(numOps);
    for (unsigned i = 0; i < numOps; ++i) {
      const Constant *Op =
        (i == idxVal) ? Elt : Constant::getNullValue(Elt->getType());
      Ops.push_back(const_cast<Constant*>(Op));
    }
    return ConstantPacked::get(Ops);
  }
  if (const ConstantPacked *CVal = dyn_cast<ConstantPacked>(Val)) {
    // Insertion of scalar constant into packed constant
    std::vector<Constant*> Ops; 
    Ops.reserve(CVal->getNumOperands());
    for (unsigned i = 0; i < CVal->getNumOperands(); ++i) {
      const Constant *Op =
        (i == idxVal) ? Elt : cast<Constant>(CVal->getOperand(i));
      Ops.push_back(const_cast<Constant*>(Op));
    }
    return ConstantPacked::get(Ops);
  }
  return 0;
}

Constant *llvm::ConstantFoldShuffleVectorInstruction(const Constant *V1,
                                                     const Constant *V2,
                                                     const Constant *Mask) {
  // TODO:
  return 0;
}


/// isZeroSizedType - This type is zero sized if its an array or structure of
/// zero sized types.  The only leaf zero sized type is an empty structure.
static bool isMaybeZeroSizedType(const Type *Ty) {
  if (isa<OpaqueType>(Ty)) return true;  // Can't say.
  if (const StructType *STy = dyn_cast<StructType>(Ty)) {

    // If all of elements have zero size, this does too.
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i)
      if (!isMaybeZeroSizedType(STy->getElementType(i))) return false;
    return true;

  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    return isMaybeZeroSizedType(ATy->getElementType());
  }
  return false;
}

/// IdxCompare - Compare the two constants as though they were getelementptr
/// indices.  This allows coersion of the types to be the same thing.
///
/// If the two constants are the "same" (after coersion), return 0.  If the
/// first is less than the second, return -1, if the second is less than the
/// first, return 1.  If the constants are not integral, return -2.
///
static int IdxCompare(Constant *C1, Constant *C2, const Type *ElTy) {
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

  // If the type being indexed over is really just a zero sized type, there is
  // no pointer difference being made here.
  if (isMaybeZeroSizedType(ElTy))
    return -2; // dunno.

  // If they are really different, now that they are the same type, then we
  // found a difference!
  if (cast<ConstantSInt>(C1)->getValue() < cast<ConstantSInt>(C2)->getValue())
    return -1;
  else
    return 1;
}

/// evaluateRelation - This function determines if there is anything we can
/// decide about the two constants provided.  This doesn't need to handle simple
/// things like integer comparisons, but should instead handle ConstantExprs
/// and GlobalValuess.  If we can determine that the two constants have a
/// particular relation to each other, we should return the corresponding SetCC
/// code, otherwise return Instruction::BinaryOpsEnd.
///
/// To simplify this code we canonicalize the relation so that the first
/// operand is always the most "complex" of the two.  We consider simple
/// constants (like ConstantInt) to be the simplest, followed by
/// GlobalValues, followed by ConstantExpr's (the most complex).
///
static Instruction::BinaryOps evaluateRelation(Constant *V1, Constant *V2) {
  assert(V1->getType() == V2->getType() &&
         "Cannot compare different types of values!");
  if (V1 == V2) return Instruction::SetEQ;

  if (!isa<ConstantExpr>(V1) && !isa<GlobalValue>(V1)) {
    if (!isa<GlobalValue>(V2) && !isa<ConstantExpr>(V2)) {
      // We distilled this down to a simple case, use the standard constant
      // folder.
      ConstantBool *R = dyn_cast<ConstantBool>(ConstantExpr::getSetEQ(V1, V2));
      if (R == ConstantBool::True) return Instruction::SetEQ;
      R = dyn_cast<ConstantBool>(ConstantExpr::getSetLT(V1, V2));
      if (R == ConstantBool::True) return Instruction::SetLT;
      R = dyn_cast<ConstantBool>(ConstantExpr::getSetGT(V1, V2));
      if (R == ConstantBool::True) return Instruction::SetGT;
      
      // If we couldn't figure it out, bail.
      return Instruction::BinaryOpsEnd;
    }
    
    // If the first operand is simple, swap operands.
    Instruction::BinaryOps SwappedRelation = evaluateRelation(V2, V1);
    if (SwappedRelation != Instruction::BinaryOpsEnd)
      return SetCondInst::getSwappedCondition(SwappedRelation);

  } else if (const GlobalValue *CPR1 = dyn_cast<GlobalValue>(V1)) {
    if (isa<ConstantExpr>(V2)) {  // Swap as necessary.
      Instruction::BinaryOps SwappedRelation = evaluateRelation(V2, V1);
      if (SwappedRelation != Instruction::BinaryOpsEnd)
        return SetCondInst::getSwappedCondition(SwappedRelation);
      else
        return Instruction::BinaryOpsEnd;
    }

    // Now we know that the RHS is a GlobalValue or simple constant,
    // which (since the types must match) means that it's a ConstantPointerNull.
    if (const GlobalValue *CPR2 = dyn_cast<GlobalValue>(V2)) {
      assert(CPR1 != CPR2 &&
             "GVs for the same value exist at different addresses??");
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
    ConstantExpr *CE1 = cast<ConstantExpr>(V1);
    Constant *CE1Op0 = CE1->getOperand(0);

    switch (CE1->getOpcode()) {
    case Instruction::Cast:
      // If the cast is not actually changing bits, and the second operand is a
      // null pointer, do the comparison with the pre-casted value.
      if (V2->isNullValue() &&
          (isa<PointerType>(CE1->getType()) || CE1->getType()->isIntegral()))
        return evaluateRelation(CE1Op0,
                                Constant::getNullValue(CE1Op0->getType()));

      // If the dest type is a pointer type, and the RHS is a constantexpr cast
      // from the same type as the src of the LHS, evaluate the inputs.  This is
      // important for things like "seteq (cast 4 to int*), (cast 5 to int*)",
      // which happens a lot in compilers with tagged integers.
      if (ConstantExpr *CE2 = dyn_cast<ConstantExpr>(V2))
        if (isa<PointerType>(CE1->getType()) && 
            CE2->getOpcode() == Instruction::Cast &&
            CE1->getOperand(0)->getType() == CE2->getOperand(0)->getType() &&
            CE1->getOperand(0)->getType()->isIntegral()) {
          return evaluateRelation(CE1->getOperand(0), CE2->getOperand(0));
        }
      break;

    case Instruction::GetElementPtr:
      // Ok, since this is a getelementptr, we know that the constant has a
      // pointer type.  Check the various cases.
      if (isa<ConstantPointerNull>(V2)) {
        // If we are comparing a GEP to a null pointer, check to see if the base
        // of the GEP equals the null pointer.
        if (isa<GlobalValue>(CE1Op0)) {
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
      } else if (const GlobalValue *CPR2 = dyn_cast<GlobalValue>(V2)) {
        if (isa<ConstantPointerNull>(CE1Op0)) {
          // FIXME: This is not true with external weak references.
          return Instruction::SetLT;
        } else if (const GlobalValue *CPR1 = dyn_cast<GlobalValue>(CE1Op0)) {
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
          if (isa<GlobalValue>(CE1Op0) && isa<GlobalValue>(CE2Op0)) {
            if (CE1Op0 != CE2Op0) // Don't know relative ordering, but not equal
              return Instruction::SetNE;
            // Ok, we know that both getelementptr instructions are based on the
            // same global.  From this, we can precisely determine the relative
            // ordering of the resultant pointers.
            unsigned i = 1;

            // Compare all of the operands the GEP's have in common.
            gep_type_iterator GTI = gep_type_begin(CE1);
            for (;i != CE1->getNumOperands() && i != CE2->getNumOperands();
                 ++i, ++GTI)
              switch (IdxCompare(CE1->getOperand(i), CE2->getOperand(i),
                                 GTI.getIndexedType())) {
              case -1: return Instruction::SetLT;
              case 1:  return Instruction::SetGT;
              case -2: return Instruction::BinaryOpsEnd;
              }

            // Ok, we ran out of things they have in common.  If any leftovers
            // are non-zero then we have a difference, otherwise we are equal.
            for (; i < CE1->getNumOperands(); ++i)
              if (!CE1->getOperand(i)->isNullValue())
                if (isa<ConstantIntegral>(CE1->getOperand(i)))
                  return Instruction::SetGT;
                else
                  return Instruction::BinaryOpsEnd; // Might be equal.

            for (; i < CE2->getNumOperands(); ++i)
              if (!CE2->getOperand(i)->isNullValue())
                if (isa<ConstantIntegral>(CE2->getOperand(i)))
                  return Instruction::SetLT;
                else
                  return Instruction::BinaryOpsEnd; // Might be equal.
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
    if (C) return ConstantExpr::getNot(C);
    break;
  case Instruction::SetLE:   // V1 <= V2  ===  !(V2 < V1)
    C = ConstRules::get(V1, V2).lessthan(V2, V1);
    if (C) return ConstantExpr::getNot(C);
    break;
  case Instruction::SetGE:   // V1 >= V2  ===  !(V1 < V2)
    C = ConstRules::get(V1, V2).lessthan(V1, V2);
    if (C) return ConstantExpr::getNot(C);
    break;
  }

  // If we successfully folded the expression, return it now.
  if (C) return C;

  if (SetCondInst::isRelational(Opcode)) {
    if (isa<UndefValue>(V1) || isa<UndefValue>(V2))
      return UndefValue::get(Type::BoolTy);
    switch (evaluateRelation(const_cast<Constant*>(V1),
                             const_cast<Constant*>(V2))) {
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
  }

  if (isa<UndefValue>(V1) || isa<UndefValue>(V2)) {
    switch (Opcode) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Xor:
      return UndefValue::get(V1->getType());

    case Instruction::Mul:
    case Instruction::And:
      return Constant::getNullValue(V1->getType());
    case Instruction::Div:
    case Instruction::Rem:
      if (!isa<UndefValue>(V2))     // undef/X -> 0
        return Constant::getNullValue(V1->getType());
      return const_cast<Constant*>(V2);                // X/undef -> undef
    case Instruction::Or:           // X|undef -> -1
      return ConstantInt::getAllOnesValue(V1->getType());
    case Instruction::Shr:
      if (!isa<UndefValue>(V2)) {
        if (V1->getType()->isSigned())
          return const_cast<Constant*>(V1);  // undef >>s X -> undef
        // undef >>u X -> 0
      } else if (isa<UndefValue>(V1)) {
        return const_cast<Constant*>(V1);   //  undef >> undef -> undef
      } else {
        if (V1->getType()->isSigned())
          return const_cast<Constant*>(V1);  // X >>s undef -> X
        // X >>u undef -> 0
      }
      return Constant::getNullValue(V1->getType());

    case Instruction::Shl:
      // undef << X -> 0   X << undef -> 0
      return Constant::getNullValue(V1->getType());
    }
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
            isa<GlobalValue>(CE1->getOperand(0))) {
          GlobalValue *CPR = cast<GlobalValue>(CE1->getOperand(0));

          // Functions are at least 4-byte aligned.  If and'ing the address of a
          // function with a constant < 4, fold it to zero.
          if (const ConstantInt *CI = dyn_cast<ConstantInt>(V2))
            if (CI->getRawValue() < 4 && isa<Function>(CPR))
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
                                          const std::vector<Value*> &IdxList) {
  if (IdxList.size() == 0 ||
      (IdxList.size() == 1 && cast<Constant>(IdxList[0])->isNullValue()))
    return const_cast<Constant*>(C);

  if (isa<UndefValue>(C)) {
    const Type *Ty = GetElementPtrInst::getIndexedType(C->getType(), IdxList,
                                                       true);
    assert(Ty != 0 && "Invalid indices for GEP!");
    return UndefValue::get(PointerType::get(Ty));
  }

  Constant *Idx0 = cast<Constant>(IdxList[0]);
  if (C->isNullValue()) {
    bool isNull = true;
    for (unsigned i = 0, e = IdxList.size(); i != e; ++i)
      if (!cast<Constant>(IdxList[i])->isNullValue()) {
        isNull = false;
        break;
      }
    if (isNull) {
      const Type *Ty = GetElementPtrInst::getIndexedType(C->getType(), IdxList,
                                                         true);
      assert(Ty != 0 && "Invalid indices for GEP!");
      return ConstantPointerNull::get(PointerType::get(Ty));
    }

    if (IdxList.size() == 1) {
      const Type *ElTy = cast<PointerType>(C->getType())->getElementType();
      if (unsigned ElSize = ElTy->getPrimitiveSize()) {
        // gep null, C is equal to C*sizeof(nullty).  If nullty is a known llvm
        // type, we can statically fold this.
        Constant *R = ConstantUInt::get(Type::UIntTy, ElSize);
        R = ConstantExpr::getCast(R, Idx0->getType());
        R = ConstantExpr::getMul(R, Idx0);
        return ConstantExpr::getCast(R, C->getType());
      }
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

      if ((LastTy && isa<ArrayType>(LastTy)) || Idx0->isNullValue()) {
        std::vector<Value*> NewIndices;
        NewIndices.reserve(IdxList.size() + CE->getNumOperands());
        for (unsigned i = 1, e = CE->getNumOperands()-1; i != e; ++i)
          NewIndices.push_back(CE->getOperand(i));

        // Add the last index of the source with the first index of the new GEP.
        // Make sure to handle the case when they are actually different types.
        Constant *Combined = CE->getOperand(CE->getNumOperands()-1);
        // Otherwise it must be an array.
        if (!Idx0->isNullValue()) {
          const Type *IdxTy = Combined->getType();
          if (IdxTy != Idx0->getType()) IdxTy = Type::LongTy;
          Combined =
            ConstantExpr::get(Instruction::Add,
                              ConstantExpr::getCast(Idx0, IdxTy),
                              ConstantExpr::getCast(Combined, IdxTy));
        }

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
        Idx0->isNullValue())
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

