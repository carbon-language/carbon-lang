//===- ConstantHandling.cpp - Implement ConstantHandling.h ----------------===//
//
// This file implements the various intrinsic operations, on constant values.
//
//===----------------------------------------------------------------------===//

#include "llvm/Opt/ConstantHandling.h"

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

  virtual ConstPoolVal *neg(const ConstPoolVal *V) const {
    return SubClassName::Neg((const ArgType *)V);
  }

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

  virtual ConstPoolBool *lessthan(const ConstPoolVal *V1, 
                                  const ConstPoolVal *V2) const { 
    return SubClassName::LessThan((const ArgType *)V1, (const ArgType *)V2);
  }

  //===--------------------------------------------------------------------===//
  // Default "noop" implementations
  //===--------------------------------------------------------------------===//

  inline static ConstPoolVal *Neg(const ArgType *V) { return 0; }
  inline static ConstPoolVal *Not(const ArgType *V) { return 0; }

  inline static ConstPoolVal *Add(const ArgType *V1, const ArgType *V2) {
    return 0;
  }

  inline static ConstPoolVal *Sub(const ArgType *V1, const ArgType *V2) {
    return 0;
  }

  inline static ConstPoolBool *LessThan(const ArgType *V1, const ArgType *V2) {
    return 0;
  }
};



//===----------------------------------------------------------------------===//
//                             EmptyRules Class
//===----------------------------------------------------------------------===//
//
// EmptyRules provides a concrete base class of ConstRules that does nothing
//
static    // EmptyInst is static
struct EmptyRules : public TemplateRules<ConstPoolVal, EmptyRules> {
} EmptyInst;



//===----------------------------------------------------------------------===//
//                              BoolRules Class
//===----------------------------------------------------------------------===//
//
// BoolRules provides a concrete base class of ConstRules for the 'bool' type.
//
static   // BoolTyInst is static...
struct BoolRules : public TemplateRules<ConstPoolBool, BoolRules> {

  inline static ConstPoolVal *Not(const ConstPoolBool *V) { 
    return new ConstPoolBool(!V->getValue());
  }

  inline static ConstPoolVal *Or(const ConstPoolBool *V1, 
                                 const ConstPoolBool *V2) {
    bool Result = V1->getValue() | V2->getValue();
    return new ConstPoolBool(Result);
  }

  inline static ConstPoolVal *And(const ConstPoolBool *V1, 
                                  const ConstPoolBool *V2) {
    bool Result = V1->getValue() & V2->getValue();
    return new ConstPoolBool(Result);
  }
} BoolTyInst;


//===----------------------------------------------------------------------===//
//                             DirectRules Class
//===----------------------------------------------------------------------===//
//
// DirectRules provides a concrete base classes of ConstRules for a variety of
// different types.  This allows the C++ compiler to automatically generate our
// constant handling operations in a typesafe and accurate manner.
//
template<class ConstPoolClass, class BuiltinType, const Type **Ty>
struct DirectRules 
  : public TemplateRules<ConstPoolClass, 
                         DirectRules<ConstPoolClass, BuiltinType, Ty> > {

  inline static ConstPoolVal *Neg(const ConstPoolClass *V) { 
    return new ConstPoolClass(*Ty, -(BuiltinType)V->getValue());;
  }
  inline static ConstPoolVal *Not(const ConstPoolClass *V) { 
    return new ConstPoolClass(*Ty, !(BuiltinType)V->getValue());;
  }

  inline static ConstPoolVal *Add(const ConstPoolClass *V1, 
                                  const ConstPoolClass *V2) {
    BuiltinType Result = (BuiltinType)V1->getValue() + 
                         (BuiltinType)V2->getValue();
    return new ConstPoolClass(*Ty, Result);
  }

  inline static ConstPoolVal *Sub(const ConstPoolClass *V1, 
                                  const ConstPoolClass *V2) {
    BuiltinType Result = (BuiltinType)V1->getValue() -
                         (BuiltinType)V2->getValue();
    return new ConstPoolClass(*Ty, Result);
  }

  inline static ConstPoolBool *LessThan(const ConstPoolClass *V1, 
                                        const ConstPoolClass *V2) {
    bool Result = (BuiltinType)V1->getValue() < (BuiltinType)V2->getValue();
    return new ConstPoolBool(Result);
  } 
};

//===----------------------------------------------------------------------===//
//                            DirectRules Subclasses
//===----------------------------------------------------------------------===//
//
// Given the DirectRules class we can now implement lots of types with little
// code.  Thank goodness C++ compilers are great at stomping out layers of 
// templates... can you imagine having to do this all by hand? (/me is lazy :)
//
static DirectRules<ConstPoolSInt,   signed char , &Type::SByteTy>  SByteTyInst;
static DirectRules<ConstPoolUInt, unsigned char , &Type::UByteTy>  UByteTyInst;
static DirectRules<ConstPoolSInt,   signed short, &Type::ShortTy>  ShortTyInst;
static DirectRules<ConstPoolUInt, unsigned short, &Type::UShortTy> UShortTyInst;
static DirectRules<ConstPoolSInt,   signed int  , &Type::IntTy>    IntTyInst;
static DirectRules<ConstPoolUInt, unsigned int  , &Type::UIntTy>   UIntTyInst;
static DirectRules<ConstPoolSInt,  int64_t      , &Type::LongTy>   LongTyInst;
static DirectRules<ConstPoolUInt, uint64_t      , &Type::ULongTy>  ULongTyInst;
static DirectRules<ConstPoolFP  , float         , &Type::FloatTy>  FloatTyInst;
static DirectRules<ConstPoolFP  , double        , &Type::DoubleTy> DoubleTyInst;


// ConstRules::find - Return the constant rules that take care of the specified
// type.  Note that this is cached in the Type value itself, so switch statement
// is only hit at most once per type.
//
const ConstRules *ConstRules::find(const Type *Ty) {
  const ConstRules *Result;
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID:   Result = &BoolTyInst;   break;
  case Type::SByteTyID:  Result = &SByteTyInst;  break;
  case Type::UByteTyID:  Result = &UByteTyInst;  break;
  case Type::ShortTyID:  Result = &ShortTyInst;  break;
  case Type::UShortTyID: Result = &UShortTyInst; break;
  case Type::IntTyID:    Result = &IntTyInst;    break;
  case Type::UIntTyID:   Result = &UIntTyInst;   break;
  case Type::LongTyID:   Result = &LongTyInst;   break;
  case Type::ULongTyID:  Result = &ULongTyInst;  break;
  case Type::FloatTyID:  Result = &FloatTyInst;  break;
  case Type::DoubleTyID: Result = &DoubleTyInst; break;
  default:               Result = &EmptyInst;    break;
  }

  Ty->setConstRules(Result);   // Cache the value for future short circuiting!
  return Result;
}
