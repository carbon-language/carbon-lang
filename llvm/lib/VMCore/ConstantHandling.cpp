//===- ConstantHandling.cpp - Implement ConstantHandling.h ----------------===//
//
// This file implements the various intrinsic operations, on constant values.
//
//===----------------------------------------------------------------------===//

#include "llvm/ConstantHandling.h"
#include "llvm/iPHINode.h"
#include "llvm/DerivedTypes.h"
#include <cmath>

AnnotationID ConstRules::AID(AnnotationManager::getID("opt::ConstRules",
						      &ConstRules::find));

// ConstantFoldInstruction - Attempt to constant fold the specified instruction.
// If successful, the constant result is returned, if not, null is returned.
//
Constant *ConstantFoldInstruction(Instruction *I) {
  if (PHINode *PN = dyn_cast<PHINode>(I)) {
    if (PN->getNumIncomingValues() == 0)
      return Constant::getNullValue(PN->getType());
    
    Constant *Result = dyn_cast<Constant>(PN->getIncomingValue(0));
    if (Result == 0) return 0;

    // Handle PHI nodes specially here...
    for (unsigned i = 1, e = PN->getNumIncomingValues(); i != e; ++i)
      if (PN->getIncomingValue(i) != Result)
        return 0;   // Not all the same incoming constants...

    // If we reach here, all incoming values are the same constant.
    return Result;
  }

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
    return ConstRules::get(*Op0, *Op0)->castTo(Op0, I->getType());
  case Instruction::Add:     return *Op0 + *Op1;
  case Instruction::Sub:     return *Op0 - *Op1;
  case Instruction::Mul:     return *Op0 * *Op1;
  case Instruction::Div:     return *Op0 / *Op1;
  case Instruction::Rem:     return *Op0 % *Op1;
  case Instruction::And:     return *Op0 & *Op1;
  case Instruction::Or:      return *Op0 | *Op1;
  case Instruction::Xor:     return *Op0 ^ *Op1;

  case Instruction::SetEQ:   return *Op0 == *Op1;
  case Instruction::SetNE:   return *Op0 != *Op1;
  case Instruction::SetLE:   return *Op0 <= *Op1;
  case Instruction::SetGE:   return *Op0 >= *Op1;
  case Instruction::SetLT:   return *Op0 <  *Op1;
  case Instruction::SetGT:   return *Op0 >  *Op1;
  case Instruction::Shl:     return *Op0 << *Op1;
  case Instruction::Shr:     return *Op0 >> *Op1;
  case Instruction::GetElementPtr: {
    std::vector<Constant*> IdxList;
    IdxList.reserve(I->getNumOperands()-1);
    if (Op1) IdxList.push_back(Op1);
    for (unsigned i = 2, e = I->getNumOperands(); i != e; ++i)
      if (Constant *C = dyn_cast<Constant>(I->getOperand(i)))
        IdxList.push_back(C);
      else
        return 0;  // Non-constant operand
    return ConstantFoldGetElementPtr(Op0, IdxList);
  }
  default:
    return 0;
  }
}

static unsigned getSize(const Type *Ty) {
  unsigned S = Ty->getPrimitiveSize();
  return S ? S : 8;  // Treat pointers at 8 bytes
}

Constant *ConstantFoldCastInstruction(const Constant *V, const Type *DestTy) {
  if (V->getType() == DestTy) return (Constant*)V;

  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
    if (CE->getOpcode() == Instruction::Cast) {
      Constant *Op = (Constant*)cast<Constant>(CE->getOperand(0));
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
    }

  return ConstRules::get(*V, *V)->castTo(V, DestTy);
}

Constant *ConstantFoldBinaryInstruction(unsigned Opcode, const Constant *V1,
                                        const Constant *V2) {
  switch (Opcode) {
  case Instruction::Add:     return *V1 + *V2;
  case Instruction::Sub:     return *V1 - *V2;
  case Instruction::Mul:     return *V1 * *V2;
  case Instruction::Div:     return *V1 / *V2;
  case Instruction::Rem:     return *V1 % *V2;
  case Instruction::And:     return *V1 & *V2;
  case Instruction::Or:      return *V1 | *V2;
  case Instruction::Xor:     return *V1 ^ *V2;

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

Constant *ConstantFoldGetElementPtr(const Constant *C,
                                    const std::vector<Constant*> &IdxList) {
  if (IdxList.size() == 0 ||
      (IdxList.size() == 1 && IdxList[0]->isNullValue()))
    return const_cast<Constant*>(C);

  // If C is null and all idx's are null, return null of the right type.

  // FIXME: Implement folding of GEP constant exprs the same as instcombine does

  // Implement folding of:
  //    int* getelementptr ([2 x int]* cast ([3 x int]* %X to [2 x int]*),
  //                        long 0, long 0)
  // To: int* getelementptr ([3 x int]* %X, long 0, long 0)
  //
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
    if (CE->getOpcode() == Instruction::Cast && IdxList.size() > 1 &&
        IdxList[0]->isNullValue())
      if (const PointerType *SPT = 
          dyn_cast<PointerType>(CE->getOperand(0)->getType()))
        if (const ArrayType *SAT = dyn_cast<ArrayType>(SPT->getElementType()))
          if (const ArrayType *CAT =
              dyn_cast<ArrayType>(cast<PointerType>(C->getType())->getElementType()))
            if (CAT->getElementType() == SAT->getElementType())
              return ConstantExpr::getGetElementPtr(
                      (Constant*)cast<Constant>(CE->getOperand(0)), IdxList);
  return 0;
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
  static ConstantBool *LessThan(const ArgType *V1, const ArgType *V2) {
    return 0;
  }

  // Casting operators.  ick
  static ConstantBool *CastToBool  (const Constant *V) { return 0; }
  static ConstantSInt *CastToSByte (const Constant *V) { return 0; }
  static ConstantUInt *CastToUByte (const Constant *V) { return 0; }
  static ConstantSInt *CastToShort (const Constant *V) { return 0; }
  static ConstantUInt *CastToUShort(const Constant *V) { return 0; }
  static ConstantSInt *CastToInt   (const Constant *V) { return 0; }
  static ConstantUInt *CastToUInt  (const Constant *V) { return 0; }
  static ConstantSInt *CastToLong  (const Constant *V) { return 0; }
  static ConstantUInt *CastToULong (const Constant *V) { return 0; }
  static ConstantFP   *CastToFloat (const Constant *V) { return 0; }
  static ConstantFP   *CastToDouble(const Constant *V) { return 0; }
  static Constant     *CastToPointer(const Constant *,
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

  static ConstantBool *LessThan(const ConstantBool *V1, const ConstantBool *V2){
    return ConstantBool::get(V1->getValue() < V2->getValue());
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
};


//===----------------------------------------------------------------------===//
//                            PointerRules Class
//===----------------------------------------------------------------------===//
//
// PointerRules provides a concrete base class of ConstRules for pointer types
//
struct PointerRules : public TemplateRules<ConstantPointer, PointerRules> {
  static ConstantBool *CastToBool  (const Constant *V) {
    if (V->isNullValue()) return ConstantBool::False;
    return 0;  // Can't const prop other types of pointers
  }
  static ConstantSInt *CastToSByte (const Constant *V) {
    if (V->isNullValue()) return ConstantSInt::get(Type::SByteTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  static ConstantUInt *CastToUByte (const Constant *V) {
    if (V->isNullValue()) return ConstantUInt::get(Type::UByteTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  static ConstantSInt *CastToShort (const Constant *V) {
    if (V->isNullValue()) return ConstantSInt::get(Type::ShortTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  static ConstantUInt *CastToUShort(const Constant *V) {
    if (V->isNullValue()) return ConstantUInt::get(Type::UShortTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  static ConstantSInt *CastToInt   (const Constant *V) {
    if (V->isNullValue()) return ConstantSInt::get(Type::IntTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  static ConstantUInt *CastToUInt  (const Constant *V) {
    if (V->isNullValue()) return ConstantUInt::get(Type::UIntTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  static ConstantSInt *CastToLong  (const Constant *V) {
    if (V->isNullValue()) return ConstantSInt::get(Type::LongTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  static ConstantUInt *CastToULong (const Constant *V) {
    if (V->isNullValue()) return ConstantUInt::get(Type::ULongTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  static ConstantFP   *CastToFloat (const Constant *V) {
    if (V->isNullValue()) return ConstantFP::get(Type::FloatTy, 0);
    return 0;  // Can't const prop other types of pointers
  }
  static ConstantFP   *CastToDouble(const Constant *V) {
    if (V->isNullValue()) return ConstantFP::get(Type::DoubleTy, 0);
    return 0;  // Can't const prop other types of pointers
  }

  static Constant *CastToPointer(const ConstantPointer *V,
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

  static ConstantBool *LessThan(const ConstantClass *V1,
                                const ConstantClass *V2) {
    bool R = (BuiltinType)V1->getValue() < (BuiltinType)V2->getValue();
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
  static CLASS *CastTo##TYPE  (const ConstantClass *V) {    \
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
// DirectFPRules provides implementations of functions that are valid on
// floating point types, but not all types in general.
//
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

ConstRules *ConstRules::getConstantExprRules() {
  static EmptyRules CERules;
  return &CERules;
}
