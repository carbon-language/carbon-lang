//===- ConstantHandling.cpp - Implement ConstantHandling.h ----------------===//
//
// This file implements the various intrinsic operations, on constant values.
//
//===----------------------------------------------------------------------===//

#include "llvm/Optimizations/ConstantHandling.h"

namespace opt {

AnnotationID ConstRules::AID(AnnotationManager::getID("opt::ConstRules",
						      &ConstRules::find));

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

  virtual ConstPoolVal *not(const ConstPoolVal *V) const {
    return SubClassName::Not((const ArgType *)V);
  }

  
  virtual ConstPoolVal *add(const ConstPoolVal *V1, 
                            const ConstPoolVal *V2) const { 
    return SubClassName::Add((const ArgType *)V1, (const ArgType *)V2);  
  }

  virtual ConstPoolVal *sub(const ConstPoolVal *V1, 
                            const ConstPoolVal *V2) const { 
    return SubClassName::Sub((const ArgType *)V1, (const ArgType *)V2);  
  }

  virtual ConstPoolVal *mul(const ConstPoolVal *V1, 
                            const ConstPoolVal *V2) const { 
    return SubClassName::Mul((const ArgType *)V1, (const ArgType *)V2);  
  }

  virtual ConstPoolBool *lessthan(const ConstPoolVal *V1, 
                                  const ConstPoolVal *V2) const { 
    return SubClassName::LessThan((const ArgType *)V1, (const ArgType *)V2);
  }

  // Casting operators.  ick
  virtual ConstPoolBool *castToBool(const ConstPoolVal *V) const {
    return SubClassName::CastToBool((const ArgType*)V);
  }
  virtual ConstPoolSInt *castToSByte(const ConstPoolVal *V) const {
    return SubClassName::CastToSByte((const ArgType*)V);
  }
  virtual ConstPoolUInt *castToUByte(const ConstPoolVal *V) const {
    return SubClassName::CastToUByte((const ArgType*)V);
  }
  virtual ConstPoolSInt *castToShort(const ConstPoolVal *V) const {
    return SubClassName::CastToShort((const ArgType*)V);
  }
  virtual ConstPoolUInt *castToUShort(const ConstPoolVal *V) const {
    return SubClassName::CastToUShort((const ArgType*)V);
  }
  virtual ConstPoolSInt *castToInt(const ConstPoolVal *V) const {
    return SubClassName::CastToInt((const ArgType*)V);
  }
  virtual ConstPoolUInt *castToUInt(const ConstPoolVal *V) const {
    return SubClassName::CastToUInt((const ArgType*)V);
  }
  virtual ConstPoolSInt *castToLong(const ConstPoolVal *V) const {
    return SubClassName::CastToLong((const ArgType*)V);
  }
  virtual ConstPoolUInt *castToULong(const ConstPoolVal *V) const {
    return SubClassName::CastToULong((const ArgType*)V);
  }
  virtual ConstPoolFP   *castToFloat(const ConstPoolVal *V) const {
    return SubClassName::CastToFloat((const ArgType*)V);
  }
  virtual ConstPoolFP   *castToDouble(const ConstPoolVal *V) const {
    return SubClassName::CastToDouble((const ArgType*)V);
  }

  //===--------------------------------------------------------------------===//
  // Default "noop" implementations
  //===--------------------------------------------------------------------===//

  inline static ConstPoolVal *Not(const ArgType *V) { return 0; }

  inline static ConstPoolVal *Add(const ArgType *V1, const ArgType *V2) {
    return 0;
  }
  inline static ConstPoolVal *Sub(const ArgType *V1, const ArgType *V2) {
    return 0;
  }
  inline static ConstPoolVal *Mul(const ArgType *V1, const ArgType *V2) {
    return 0;
  }
  inline static ConstPoolBool *LessThan(const ArgType *V1, const ArgType *V2) {
    return 0;
  }

  // Casting operators.  ick
  inline static ConstPoolBool *CastToBool  (const ConstPoolVal *V) { return 0; }
  inline static ConstPoolSInt *CastToSByte (const ConstPoolVal *V) { return 0; }
  inline static ConstPoolUInt *CastToUByte (const ConstPoolVal *V) { return 0; }
  inline static ConstPoolSInt *CastToShort (const ConstPoolVal *V) { return 0; }
  inline static ConstPoolUInt *CastToUShort(const ConstPoolVal *V) { return 0; }
  inline static ConstPoolSInt *CastToInt   (const ConstPoolVal *V) { return 0; }
  inline static ConstPoolUInt *CastToUInt  (const ConstPoolVal *V) { return 0; }
  inline static ConstPoolSInt *CastToLong  (const ConstPoolVal *V) { return 0; }
  inline static ConstPoolUInt *CastToULong (const ConstPoolVal *V) { return 0; }
  inline static ConstPoolFP   *CastToFloat (const ConstPoolVal *V) { return 0; }
  inline static ConstPoolFP   *CastToDouble(const ConstPoolVal *V) { return 0; }
};



//===----------------------------------------------------------------------===//
//                             EmptyRules Class
//===----------------------------------------------------------------------===//
//
// EmptyRules provides a concrete base class of ConstRules that does nothing
//
struct EmptyRules : public TemplateRules<ConstPoolVal, EmptyRules> {
};



//===----------------------------------------------------------------------===//
//                              BoolRules Class
//===----------------------------------------------------------------------===//
//
// BoolRules provides a concrete base class of ConstRules for the 'bool' type.
//
struct BoolRules : public TemplateRules<ConstPoolBool, BoolRules> {

  inline static ConstPoolVal *Not(const ConstPoolBool *V) { 
    return ConstPoolBool::get(!V->getValue());
  }

  inline static ConstPoolVal *Or(const ConstPoolBool *V1,
				 const ConstPoolBool *V2) {
    return ConstPoolBool::get(V1->getValue() | V2->getValue());
  }

  inline static ConstPoolVal *And(const ConstPoolBool *V1, 
                                  const ConstPoolBool *V2) {
    return ConstPoolBool::get(V1->getValue() & V2->getValue());
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
template<class ConstPoolClass, class BuiltinType, Type **Ty>
struct DirectRules 
  : public TemplateRules<ConstPoolClass, 
                         DirectRules<ConstPoolClass, BuiltinType, Ty> > {

  inline static ConstPoolVal *Not(const ConstPoolClass *V) { 
    return ConstPoolClass::get(*Ty, !(BuiltinType)V->getValue());;
  }

  inline static ConstPoolVal *Add(const ConstPoolClass *V1, 
                                  const ConstPoolClass *V2) {
    BuiltinType Result = (BuiltinType)V1->getValue() + 
                         (BuiltinType)V2->getValue();
    return ConstPoolClass::get(*Ty, Result);
  }

  inline static ConstPoolVal *Sub(const ConstPoolClass *V1, 
                                  const ConstPoolClass *V2) {
    BuiltinType Result = (BuiltinType)V1->getValue() -
                         (BuiltinType)V2->getValue();
    return ConstPoolClass::get(*Ty, Result);
  }

  inline static ConstPoolVal *Mul(const ConstPoolClass *V1, 
				   const ConstPoolClass *V2) {
    BuiltinType Result = (BuiltinType)V1->getValue() *
                         (BuiltinType)V2->getValue();
    return ConstPoolClass::get(*Ty, Result);
  }

  inline static ConstPoolBool *LessThan(const ConstPoolClass *V1, 
                                        const ConstPoolClass *V2) {
    bool Result = (BuiltinType)V1->getValue() < (BuiltinType)V2->getValue();
    return ConstPoolBool::get(Result);
  } 

  // Casting operators.  ick
#define DEF_CAST(TYPE, CLASS, CTYPE) \
  inline static CLASS *CastTo##TYPE  (const ConstPoolClass *V) {    \
    return CLASS::get(Type::TYPE##Ty, (CTYPE)(BuiltinType)V->getValue()); \
  }

  DEF_CAST(Bool  , ConstPoolBool, bool)
  DEF_CAST(SByte , ConstPoolSInt, signed char)
  DEF_CAST(UByte , ConstPoolUInt, unsigned char)
  DEF_CAST(Short , ConstPoolSInt, signed short)
  DEF_CAST(UShort, ConstPoolUInt, unsigned short)
  DEF_CAST(Int   , ConstPoolSInt, signed int)
  DEF_CAST(UInt  , ConstPoolUInt, unsigned int)
  DEF_CAST(Long  , ConstPoolSInt, int64_t)
  DEF_CAST(ULong , ConstPoolUInt, uint64_t)
  DEF_CAST(Float , ConstPoolFP  , float)
  DEF_CAST(Double, ConstPoolFP  , double)
#undef DEF_CAST
};

//===----------------------------------------------------------------------===//
//                            DirectRules Subclasses
//===----------------------------------------------------------------------===//
//
// Given the DirectRules class we can now implement lots of types with little
// code.  Thank goodness C++ compilers are great at stomping out layers of 
// templates... can you imagine having to do this all by hand? (/me is lazy :)
//

// ConstRules::find - Return the constant rules that take care of the specified
// type.
//
Annotation *ConstRules::find(AnnotationID AID, const Annotable *TyA, void *) {
  assert(AID == ConstRules::AID && "Bad annotation for factory!");
  const Type *Ty = ((const Value*)TyA)->castTypeAsserting();
  
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID: return new BoolRules();
  case Type::SByteTyID:
    return new DirectRules<ConstPoolSInt,   signed char , &Type::SByteTy>();
  case Type::UByteTyID:
    return new DirectRules<ConstPoolUInt, unsigned char , &Type::UByteTy>();
  case Type::ShortTyID:
    return new DirectRules<ConstPoolSInt,   signed short, &Type::ShortTy>();
  case Type::UShortTyID:
    return new DirectRules<ConstPoolUInt, unsigned short, &Type::UShortTy>();
  case Type::IntTyID:
    return new DirectRules<ConstPoolSInt,   signed int  , &Type::IntTy>();
  case Type::UIntTyID:
    return new DirectRules<ConstPoolUInt, unsigned int  , &Type::UIntTy>();
  case Type::LongTyID:
    return new DirectRules<ConstPoolSInt,  int64_t      , &Type::LongTy>();
  case Type::ULongTyID:
    return new DirectRules<ConstPoolUInt, uint64_t      , &Type::ULongTy>();
  case Type::FloatTyID:
    return new DirectRules<ConstPoolFP  , float         , &Type::FloatTy>();
  case Type::DoubleTyID:
    return new DirectRules<ConstPoolFP  , double        , &Type::DoubleTy>();
  default:
    return new EmptyRules();
  }
}


} // End namespace opt
