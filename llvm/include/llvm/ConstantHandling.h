//===-- ConstantHandling.h - Stuff for manipulating constants ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of some cool operators that allow you
// to do natural things with constant pool values.
//
// Unfortunately we can't overload operators on pointer types (like this:)
//
//      inline bool operator==(const Constant *V1, const Constant *V2)
//
// so we must make due with references, even though it leads to some butt ugly
// looking code downstream.  *sigh*  (ex:  Constant *Result = *V1 + *v2; )
//
//===----------------------------------------------------------------------===//
//
// WARNING: These operators may return a null object if I don't know how to 
//          perform the specified operation on the specified constant types.
//
//===----------------------------------------------------------------------===//
//
// Implementation notes:
//   This library is implemented this way for a reason: In most cases, we do
//   not want to have to link the constant mucking code into an executable.
//   We do, however want to tie some of this into the main type system, as an
//   optional component.  By using a mutable cache member in the Type class, we
//   get exactly the kind of behavior we want.
//
// In the end, we get performance almost exactly the same as having a virtual
// function dispatch, but we don't have to put our virtual functions into the
// "Type" class, and we can implement functionality with templates. Good deal.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CONSTANTHANDLING_H
#define LLVM_CONSTANTHANDLING_H

#include "llvm/Constants.h"
#include "llvm/Type.h"

namespace llvm {

class PointerType;

//===----------------------------------------------------------------------===//
//  Implement == and != directly...
//===----------------------------------------------------------------------===//

inline ConstantBool *operator==(const Constant &V1, const Constant &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return ConstantBool::get(&V1 == &V2);
}

inline ConstantBool *operator!=(const Constant &V1, const Constant &V2) {
  return ConstantBool::get(&V1 != &V2);
}

//===----------------------------------------------------------------------===//
//  Implement all other operators indirectly through TypeRules system
//===----------------------------------------------------------------------===//

class ConstRules : public Annotation {
protected:
  inline ConstRules() : Annotation(AID) {}  // Can only be subclassed...
public:
  static AnnotationID AID;    // AnnotationID for this class

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

  virtual ConstantBool *lessthan(const Constant *V1, 
                                 const Constant *V2) const = 0;

  // Casting operators.  ick
  virtual ConstantBool *castToBool  (const Constant *V) const = 0;
  virtual ConstantSInt *castToSByte (const Constant *V) const = 0;
  virtual ConstantUInt *castToUByte (const Constant *V) const = 0;
  virtual ConstantSInt *castToShort (const Constant *V) const = 0;
  virtual ConstantUInt *castToUShort(const Constant *V) const = 0;
  virtual ConstantSInt *castToInt   (const Constant *V) const = 0;
  virtual ConstantUInt *castToUInt  (const Constant *V) const = 0;
  virtual ConstantSInt *castToLong  (const Constant *V) const = 0;
  virtual ConstantUInt *castToULong (const Constant *V) const = 0;
  virtual ConstantFP   *castToFloat (const Constant *V) const = 0;
  virtual ConstantFP   *castToDouble(const Constant *V) const = 0;
  virtual Constant     *castToPointer(const Constant *V,
                                      const PointerType *Ty) const = 0;

  inline Constant *castTo(const Constant *V, const Type *Ty) const {
    switch (Ty->getPrimitiveID()) {
    case Type::BoolTyID:   return castToBool(V);
    case Type::UByteTyID:  return castToUByte(V);
    case Type::SByteTyID:  return castToSByte(V);
    case Type::UShortTyID: return castToUShort(V);
    case Type::ShortTyID:  return castToShort(V);
    case Type::UIntTyID:   return castToUInt(V);
    case Type::IntTyID:    return castToInt(V);
    case Type::ULongTyID:  return castToULong(V);
    case Type::LongTyID:   return castToLong(V);
    case Type::FloatTyID:  return castToFloat(V);
    case Type::DoubleTyID: return castToDouble(V);
    case Type::PointerTyID:return castToPointer(V, (PointerType*)Ty);
    default: return 0;
    }
  }

  // ConstRules::get - A type will cache its own type rules if one is needed...
  // we just want to make sure to hit the cache instead of doing it indirectly,
  //  if possible...
  //
  static inline ConstRules *get(const Constant &V1, const Constant &V2) {
    if (isa<ConstantExpr>(V1) || isa<ConstantExpr>(V2))
      return getConstantExprRules();
    return (ConstRules*)V1.getType()->getOrCreateAnnotation(AID);
  }
private:
  static ConstRules *getConstantExprRules();
  static Annotation *find(AnnotationID AID, const Annotable *Ty, void *);

  ConstRules(const ConstRules &);             // Do not implement
  ConstRules &operator=(const ConstRules &);  // Do not implement
};

// Unary operators...
inline Constant *operator~(const Constant &V) {
  assert(V.getType()->isIntegral() && "Cannot invert non-integral constant!");
  return ConstRules::get(V, V)->op_xor(&V,
                                    ConstantInt::getAllOnesValue(V.getType()));
}

inline Constant *operator-(const Constant &V) {
  return ConstRules::get(V, V)->sub(Constant::getNullValue(V.getType()), &V);
}

// Standard binary operators...
inline Constant *operator+(const Constant &V1, const Constant &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return ConstRules::get(V1, V2)->add(&V1, &V2);
}

inline Constant *operator-(const Constant &V1, const Constant &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return ConstRules::get(V1, V2)->sub(&V1, &V2);
}

inline Constant *operator*(const Constant &V1, const Constant &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return ConstRules::get(V1, V2)->mul(&V1, &V2);
}

inline Constant *operator/(const Constant &V1, const Constant &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return ConstRules::get(V1, V2)->div(&V1, &V2);
}

inline Constant *operator%(const Constant &V1, const Constant &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return ConstRules::get(V1, V2)->rem(&V1, &V2);
}

// Logical Operators...
inline Constant *operator&(const Constant &V1, const Constant &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return ConstRules::get(V1, V2)->op_and(&V1, &V2);
}

inline Constant *operator|(const Constant &V1, const Constant &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return ConstRules::get(V1, V2)->op_or(&V1, &V2);
}

inline Constant *operator^(const Constant &V1, const Constant &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return ConstRules::get(V1, V2)->op_xor(&V1, &V2);
}

// Shift Instructions...
inline Constant *operator<<(const Constant &V1, const Constant &V2) {
  assert(V1.getType()->isInteger() && V2.getType() == Type::UByteTy);
  return ConstRules::get(V1, V2)->shl(&V1, &V2);
}

inline Constant *operator>>(const Constant &V1, const Constant &V2) {
  assert(V1.getType()->isInteger() && V2.getType() == Type::UByteTy);
  return ConstRules::get(V1, V2)->shr(&V1, &V2);
}

inline ConstantBool *operator<(const Constant &V1, 
                               const Constant &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return ConstRules::get(V1, V2)->lessthan(&V1, &V2);
}


//===----------------------------------------------------------------------===//
//  Implement 'derived' operators based on what we already have...
//===----------------------------------------------------------------------===//

inline ConstantBool *operator>(const Constant &V1, 
                               const Constant &V2) {
  return V2 < V1;
}

inline ConstantBool *operator>=(const Constant &V1, 
                                const Constant &V2) {
  if (ConstantBool *V = (V1 < V2))
    return V->inverted();                // !(V1 < V2)
  return 0;
}

inline ConstantBool *operator<=(const Constant &V1, 
                                const Constant &V2) {
  if (ConstantBool *V = (V1 > V2))
    return V->inverted();                // !(V1 > V2)
  return 0;
}


//===----------------------------------------------------------------------===//
//  Implement higher level instruction folding type instructions
//===----------------------------------------------------------------------===//

// ConstantFoldInstruction - Attempt to constant fold the specified instruction.
// If successful, the constant result is returned, if not, null is returned.
//
Constant *ConstantFoldInstruction(Instruction *I);

// Constant fold various types of instruction...
Constant *ConstantFoldCastInstruction(const Constant *V, const Type *DestTy);
Constant *ConstantFoldBinaryInstruction(unsigned Opcode, const Constant *V1,
                                        const Constant *V2);
Constant *ConstantFoldShiftInstruction(unsigned Opcode, const Constant *V1,
                                       const Constant *V2);
Constant *ConstantFoldGetElementPtr(const Constant *C,
                                    const std::vector<Constant*> &IdxList);

} // End llvm namespace

#endif
