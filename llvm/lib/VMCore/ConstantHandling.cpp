//===- ConstantHandling.cpp - Implement ConstantHandling.h ----------------===//
//
// This file implements the various intrinsic operations, on constant values.
//
//===----------------------------------------------------------------------===//

#include "llvm/ConstantHandling.h"
#include "llvm/Instruction.h"
#include <cmath>

AnnotationID ConstRules::AID(AnnotationManager::getID("opt::ConstRules",
						      &ConstRules::find));

// ConstantFoldInstruction - Attempt to constant fold the specified instruction.
// If successful, the constant result is returned, if not, null is returned.
//
Constant *ConstantFoldInstruction(Instruction *I) {
  Constant *Op0 = 0;
  Constant *Op1 = 0;

  if (I->getNumOperands() != 0) {    // Get first operand if it's a constant...
    Op0 = dyn_cast<Constant>(I->getOperand(0));
    if (Op0 == 0) return 0;          // Not a constant?, can't fold

    if (I->getNumOperands() != 1) {  // Get second operand if it's a constant...
      Op1 = dyn_cast<Constant>(I->getOperand(1));
      if (Op1 == 0) return 0;        // Not a constant?, can't fold
    }
  }

  switch (I->getOpcode()) {
  case Instruction::Cast:
    return ConstRules::get(*Op0)->castTo(Op0, I->getType());
  case Instruction::Not:     return ~*Op0;
  case Instruction::Add:     return *Op0 + *Op1;
  case Instruction::Sub:     return *Op0 - *Op1;
  case Instruction::Mul:     return *Op0 * *Op1;
  case Instruction::Div:     return *Op0 / *Op1;
  case Instruction::Rem:     return *Op0 % *Op1;

  case Instruction::SetEQ:   return *Op0 == *Op1;
  case Instruction::SetNE:   return *Op0 != *Op1;
  case Instruction::SetLE:   return *Op0 <= *Op1;
  case Instruction::SetGE:   return *Op0 >= *Op1;
  case Instruction::SetLT:   return *Op0 <  *Op1;
  case Instruction::SetGT:   return *Op0 >  *Op1;
  case Instruction::Shl:     return *Op0 << *Op1;
  case Instruction::Shr:     return *Op0 >> *Op1;
  default:
    return 0;
  }
}

Constant *ConstantFoldCastInstruction(const Constant *V, const Type *DestTy) {
  return ConstRules::get(*V)->castTo(V, DestTy);
}

Constant *ConstantFoldUnaryInstruction(unsigned Opcode, const Constant *V) {
  switch (Opcode) {
  case Instruction::Not:  return ~*V;
  }
  return 0;
}

Constant *ConstantFoldBinaryInstruction(unsigned Opcode, const Constant *V1,
                                        const Constant *V2) {
  switch (Opcode) {
  case Instruction::Add:     return *V1 + *V2;
  case Instruction::Sub:     return *V1 - *V2;
  case Instruction::Mul:     return *V1 * *V2;
  case Instruction::Div:     return *V1 / *V2;
  case Instruction::Rem:     return *V1 % *V2;

  case Instruction::SetEQ:   return *V1 == *V2;
  case Instruction::SetNE:   return *V1 != *V2;
  case Instruction::SetLE:   return *V1 <= *V2;
  case Instruction::SetGE:   return *V1 >= *V2;
  case Instruction::SetLT:   return *V1 <  *V2;
  case Instruction::SetGT:   return *V1 >  *V2;
  }
  return 0;
}

Constant *ConstantFoldShiftInstruction(unsigned Opcode, const Constant *V1, 
                                       const Constant *V2) {
  switch (Opcode) {
  case Instruction::Shl:     return *V1 << *V2;
  case Instruction::Shr:     return *V1 >> *V2;
  default:                   return 0;
  }
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

  virtual Constant *op_not(const Constant *V) const {
    return SubClassName::Not((const ArgType *)V);
  }

  
  virtual Constant *add(const Constant *V1, 
                        const Constant *V2) const { 
    return SubClassName::Add((const ArgType *)V1, (const ArgType *)V2);  
  }

  virtual Constant *sub(const Constant *V1, 
                        const Constant *V2) const { 
    return SubClassName::Sub((const ArgType *)V1, (const ArgType *)V2);  
  }

  virtual Constant *mul(const Constant *V1, 
                        const Constant *V2) const { 
    return SubClassName::Mul((const ArgType *)V1, (const ArgType *)V2);  
  }
  virtual Constant *div(const Constant *V1, 
                        const Constant *V2) const { 
    return SubClassName::Div((const ArgType *)V1, (const ArgType *)V2);  
  }
  virtual Constant *rem(const Constant *V1, 
                        const Constant *V2) const { 
    return SubClassName::Rem((const ArgType *)V1, (const ArgType *)V2);  
  }
  virtual Constant *shl(const Constant *V1, 
                        const Constant *V2) const { 
    return SubClassName::Shl((const ArgType *)V1, (const ArgType *)V2);  
  }
  virtual Constant *shr(const Constant *V1, 
                        const Constant *V2) const { 
    return SubClassName::Shr((const ArgType *)V1, (const ArgType *)V2);  
  }

  virtual ConstantBool *lessthan(const Constant *V1, 
                                 const Constant *V2) const { 
    return SubClassName::LessThan((const ArgType *)V1, (const ArgType *)V2);
  }

  // Casting operators.  ick
  virtual ConstantBool *castToBool(const Constant *V) const {
    return SubClassName::CastToBool((const ArgType*)V);
  }
  virtual ConstantSInt *castToSByte(const Constant *V) const {
    return SubClassName::CastToSByte((const ArgType*)V);
  }
  virtual ConstantUInt *castToUByte(const Constant *V) const {
    return SubClassName::CastToUByte((const ArgType*)V);
  }
  virtual ConstantSInt *castToShort(const Constant *V) const {
    return SubClassName::CastToShort((const ArgType*)V);
  }
  virtual ConstantUInt *castToUShort(const Constant *V) const {
    return SubClassName::CastToUShort((const ArgType*)V);
  }
  virtual ConstantSInt *castToInt(const Constant *V) const {
    return SubClassName::CastToInt((const ArgType*)V);
  }
  virtual ConstantUInt *castToUInt(const Constant *V) const {
    return SubClassName::CastToUInt((const ArgType*)V);
  }
  virtual ConstantSInt *castToLong(const Constant *V) const {
    return SubClassName::CastToLong((const ArgType*)V);
  }
  virtual ConstantUInt *castToULong(const Constant *V) const {
    return SubClassName::CastToULong((const ArgType*)V);
  }
  virtual ConstantFP   *castToFloat(const Constant *V) const {
    return SubClassName::CastToFloat((const ArgType*)V);
  }
  virtual ConstantFP   *castToDouble(const Constant *V) const {
    return SubClassName::CastToDouble((const ArgType*)V);
  }
  virtual ConstantPointer *castToPointer(const Constant *V, 
                                         const PointerType *Ty) const {
    return SubClassName::CastToPointer((const ArgType*)V, Ty);
  }

  //===--------------------------------------------------------------------===//
  // Default "noop" implementations
  //===--------------------------------------------------------------------===//

  inline static Constant *Not(const ArgType *V) { return 0; }

  inline static Constant *Add(const ArgType *V1, const ArgType *V2) {
    return 0;
  }
  inline static Constant *Sub(const ArgType *V1, const ArgType *V2) {
    return 0;
  }
  inline static Constant *Mul(const ArgType *V1, const ArgType *V2) {
    return 0;
  }
  inline static Constant *Div(const ArgType *V1, const ArgType *V2) {
    return 0;
  }
  inline static Constant *Rem(const ArgType *V1, const ArgType *V2) {
    return 0;
  }
  inline static Constant *Shl(const ArgType *V1, const ArgType *V2) {
    return 0;
  }
  inline static Constant *Shr(const ArgType *V1, const ArgType *V2) {
    return 0;
  }
  inline static ConstantBool *LessThan(const ArgType *V1, const ArgType *V2) {
    return 0;
  }

  // Casting operators.  ick
  inline static ConstantBool *CastToBool  (const Constant *V) { return 0; }
  inline static ConstantSInt *CastToSByte (const Constant *V) { return 0; }
  inline static ConstantUInt *CastToUByte (const Constant *V) { return 0; }
  inline static ConstantSInt *CastToShort (const Constant *V) { return 0; }
  inline static ConstantUInt *CastToUShort(const Constant *V) { return 0; }
  inline static ConstantSInt *CastToInt   (const Constant *V) { return 0; }
  inline static ConstantUInt *CastToUInt  (const Constant *V) { return 0; }
  inline static ConstantSInt *CastToLong  (const Constant *V) { return 0; }
  inline static ConstantUInt *CastToULong (const Constant *V) { return 0; }
  inline static ConstantFP   *CastToFloat (const Constant *V) { return 0; }
  inline static ConstantFP   *CastToDouble(const Constant *V) { return 0; }
  inline static ConstantPointer *CastToPointer(const Constant *,
                                               const PointerType *) {return 0;}
};



//===----------------------------------------------------------------------===//
//                             EmptyRules Class
//===----------------------------------------------------------------------===//
//
// EmptyRules provides a concrete base class of ConstRules that does nothing
//
struct EmptyRules : public TemplateRules<Constant, EmptyRules> {
};



//===----------------------------------------------------------------------===//
//                              BoolRules Class
//===----------------------------------------------------------------------===//
//
// BoolRules provides a concrete base class of ConstRules for the 'bool' type.
//
struct BoolRules : public TemplateRules<ConstantBool, BoolRules> {

  inline static Constant *Not(const ConstantBool *V) { 
    return ConstantBool::get(!V->getValue());
  }

  inline static Constant *Or(const ConstantBool *V1,
                             const ConstantBool *V2) {
    return ConstantBool::get(V1->getValue() | V2->getValue());
  }

  inline static Constant *And(const ConstantBool *V1, 
                              const ConstantBool *V2) {
    return ConstantBool::get(V1->getValue() & V2->getValue());
  }
};


//===----------------------------------------------------------------------===//
//                            PointerRules Class
//===----------------------------------------------------------------------===//
//
// PointerRules provides a concrete base class of ConstRules for pointer types
//
struct PointerRules : public TemplateRules<ConstantPointer, PointerRules> {
  inline static ConstantBool *CastToBool  (const Constant *V) {
    if (V->isNullValue()) return ConstantBool::False;
    return 0;  // Can't const prop other types of pointers
  }
  inline static ConstantSInt *CastToSByte (const Constant *V) {
    if (V->isNullValue()) return ConstantSInt::get(Type::SByteTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  inline static ConstantUInt *CastToUByte (const Constant *V) {
    if (V->isNullValue()) return ConstantUInt::get(Type::UByteTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  inline static ConstantSInt *CastToShort (const Constant *V) {
    if (V->isNullValue()) return ConstantSInt::get(Type::ShortTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  inline static ConstantUInt *CastToUShort(const Constant *V) {
    if (V->isNullValue()) return ConstantUInt::get(Type::UShortTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  inline static ConstantSInt *CastToInt   (const Constant *V) {
    if (V->isNullValue()) return ConstantSInt::get(Type::IntTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  inline static ConstantUInt *CastToUInt  (const Constant *V) {
    if (V->isNullValue()) return ConstantUInt::get(Type::UIntTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  inline static ConstantSInt *CastToLong  (const Constant *V) {
    if (V->isNullValue()) return ConstantSInt::get(Type::LongTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  inline static ConstantUInt *CastToULong (const Constant *V) {
    if (V->isNullValue()) return ConstantUInt::get(Type::ULongTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  inline static ConstantFP   *CastToFloat (const Constant *V) {
    if (V->isNullValue()) return ConstantFP::get(Type::FloatTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  inline static ConstantFP   *CastToDouble(const Constant *V) {
    if (V->isNullValue()) return ConstantFP::get(Type::DoubleTy, 0);
    return 0;  // Can't const prop other types of pointers
  }

  inline static ConstantPointer *CastToPointer(const ConstantPointer *V,
                                               const PointerType *PTy) {
    if (V->getType() == PTy)
      return const_cast<ConstantPointer*>(V);  // Allow cast %PTy %ptr to %PTy
    if (V->isNullValue())
      return ConstantPointerNull::get(PTy);
    return 0;  // Can't const prop other types of pointers
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
  inline static Constant *Add(const ConstantClass *V1, 
                              const ConstantClass *V2) {
    BuiltinType Result = (BuiltinType)V1->getValue() + 
                         (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, Result);
  }

  inline static Constant *Sub(const ConstantClass *V1, 
                              const ConstantClass *V2) {
    BuiltinType Result = (BuiltinType)V1->getValue() -
                         (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, Result);
  }

  inline static Constant *Mul(const ConstantClass *V1, 
                              const ConstantClass *V2) {
    BuiltinType Result = (BuiltinType)V1->getValue() *
                         (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, Result);
  }

  inline static Constant *Div(const ConstantClass *V1,
                              const ConstantClass *V2) {
    if (V2->isNullValue()) return 0;
    BuiltinType Result = (BuiltinType)V1->getValue() /
                         (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, Result);
  }

  inline static ConstantBool *LessThan(const ConstantClass *V1, 
                                       const ConstantClass *V2) {
    bool Result = (BuiltinType)V1->getValue() < (BuiltinType)V2->getValue();
    return ConstantBool::get(Result);
  } 

  inline static ConstantPointer *CastToPointer(const ConstantClass *V,
                                               const PointerType *PTy) {
    if (V->isNullValue())    // Is it a FP or Integral null value?
      return ConstantPointerNull::get(PTy);
    return 0;  // Can't const prop other types of pointers
  }

  // Casting operators.  ick
#define DEF_CAST(TYPE, CLASS, CTYPE) \
  inline static CLASS *CastTo##TYPE  (const ConstantClass *V) {    \
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
  inline static Constant *Not(const ConstantClass *V) { 
    return ConstantClass::get(*Ty, ~(BuiltinType)V->getValue());;
  }

  inline static Constant *Rem(const ConstantClass *V1,
                              const ConstantClass *V2) {
    if (V2->isNullValue()) return 0;
    BuiltinType Result = (BuiltinType)V1->getValue() %
                         (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, Result);
  }

  inline static Constant *Shl(const ConstantClass *V1,
                              const ConstantClass *V2) {
    BuiltinType Result = (BuiltinType)V1->getValue() <<
                         (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, Result);
  }

  inline static Constant *Shr(const ConstantClass *V1,
                              const ConstantClass *V2) {
    BuiltinType Result = (BuiltinType)V1->getValue() >>
                         (BuiltinType)V2->getValue();
    return ConstantClass::get(*Ty, Result);
  }
};


//===----------------------------------------------------------------------===//
//                           DirectFPRules Class
//===----------------------------------------------------------------------===//
//
// DirectFPRules provides implementations of functions that are valid on
// floating point types, but not all types in general.
//
template <class ConstantClass, class BuiltinType, Type **Ty>
struct DirectFPRules
  : public DirectRules<ConstantClass, BuiltinType, Ty,
                       DirectFPRules<ConstantClass, BuiltinType, Ty> > {
  inline static Constant *Rem(const ConstantClass *V1,
                              const ConstantClass *V2) {
    if (V2->isNullValue()) return 0;
    BuiltinType Result = std::fmod((BuiltinType)V1->getValue(),
                                   (BuiltinType)V2->getValue());
    return ConstantClass::get(*Ty, Result);
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

// ConstRules::find - Return the constant rules that take care of the specified
// type.
//
Annotation *ConstRules::find(AnnotationID AID, const Annotable *TyA, void *) {
  assert(AID == ConstRules::AID && "Bad annotation for factory!");
  const Type *Ty = cast<Type>((const Value*)TyA);
  
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID:    return new BoolRules();
  case Type::PointerTyID: return new PointerRules();
  case Type::SByteTyID:
    return new DirectIntRules<ConstantSInt,   signed char , &Type::SByteTy>();
  case Type::UByteTyID:
    return new DirectIntRules<ConstantUInt, unsigned char , &Type::UByteTy>();
  case Type::ShortTyID:
    return new DirectIntRules<ConstantSInt,   signed short, &Type::ShortTy>();
  case Type::UShortTyID:
    return new DirectIntRules<ConstantUInt, unsigned short, &Type::UShortTy>();
  case Type::IntTyID:
    return new DirectIntRules<ConstantSInt,   signed int  , &Type::IntTy>();
  case Type::UIntTyID:
    return new DirectIntRules<ConstantUInt, unsigned int  , &Type::UIntTy>();
  case Type::LongTyID:
    return new DirectIntRules<ConstantSInt,  int64_t      , &Type::LongTy>();
  case Type::ULongTyID:
    return new DirectIntRules<ConstantUInt, uint64_t      , &Type::ULongTy>();
  case Type::FloatTyID:
    return new DirectFPRules<ConstantFP  , float         , &Type::FloatTy>();
  case Type::DoubleTyID:
    return new DirectFPRules<ConstantFP  , double        , &Type::DoubleTy>();
  default:
    return new EmptyRules();
  }
}
