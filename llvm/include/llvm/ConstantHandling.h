//===-- ConstantHandling.h - Stuff for manipulating constants ----*- C++ -*--=//
//
// This file contains the declarations of some cool operators that allow you
// to do natural things with constant pool values.
//
// Unfortunately we can't overload operators on pointer types (like this:)
//
//      inline bool operator==(const ConstPoolVal *V1, const ConstPoolVal *V2)
//
// so we must make due with references, even though it leads to some butt ugly
// looking code downstream.  *sigh*  (ex:  ConstPoolVal *Result = *V1 + *v2; )
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

#ifndef LLVM_OPT_CONSTANTHANDLING_H
#define LLVM_OPT_CONSTANTHANDLING_H

#include "llvm/ConstPoolVals.h"
#include "llvm/Instruction.h"
#include "llvm/Type.h"

namespace opt {

//===----------------------------------------------------------------------===//
//  Implement == and != directly...
//===----------------------------------------------------------------------===//

inline ConstPoolBool *operator==(const ConstPoolVal &V1, 
                                 const ConstPoolVal &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return ConstPoolBool::get(&V1 == &V2);
}

inline ConstPoolBool *operator!=(const ConstPoolVal &V1, 
                                 const ConstPoolVal &V2) {
  return ConstPoolBool::get(&V1 != &V2);
}

//===----------------------------------------------------------------------===//
//  Implement all other operators indirectly through TypeRules system
//===----------------------------------------------------------------------===//

class ConstRules : public Annotation {
protected:
  inline ConstRules() : Annotation(AID) {}  // Can only be subclassed...
public:
  static AnnotationID AID;    // AnnotationID for this class

  // Unary Operators...
  virtual ConstPoolVal *op_not(const ConstPoolVal *V) const = 0;

  // Binary Operators...
  virtual ConstPoolVal *add(const ConstPoolVal *V1, 
                            const ConstPoolVal *V2) const = 0;
  virtual ConstPoolVal *sub(const ConstPoolVal *V1, 
                            const ConstPoolVal *V2) const = 0;
  virtual ConstPoolVal *mul(const ConstPoolVal *V1, 
			    const ConstPoolVal *V2) const = 0;

  virtual ConstPoolBool *lessthan(const ConstPoolVal *V1, 
                                  const ConstPoolVal *V2) const = 0;

  // Casting operators.  ick
  virtual ConstPoolBool *castToBool  (const ConstPoolVal *V) const = 0;
  virtual ConstPoolSInt *castToSByte (const ConstPoolVal *V) const = 0;
  virtual ConstPoolUInt *castToUByte (const ConstPoolVal *V) const = 0;
  virtual ConstPoolSInt *castToShort (const ConstPoolVal *V) const = 0;
  virtual ConstPoolUInt *castToUShort(const ConstPoolVal *V) const = 0;
  virtual ConstPoolSInt *castToInt   (const ConstPoolVal *V) const = 0;
  virtual ConstPoolUInt *castToUInt  (const ConstPoolVal *V) const = 0;
  virtual ConstPoolSInt *castToLong  (const ConstPoolVal *V) const = 0;
  virtual ConstPoolUInt *castToULong (const ConstPoolVal *V) const = 0;
  virtual ConstPoolFP   *castToFloat (const ConstPoolVal *V) const = 0;
  virtual ConstPoolFP   *castToDouble(const ConstPoolVal *V) const = 0;

  inline ConstPoolVal *castTo(const ConstPoolVal *V, const Type *Ty) const {
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
    default: return 0;
    }
  }

  // ConstRules::get - A type will cache its own type rules if one is needed...
  // we just want to make sure to hit the cache instead of doing it indirectly,
  //  if possible...
  //
  static inline ConstRules *get(const ConstPoolVal &V) {
    return (ConstRules*)V.getType()->getOrCreateAnnotation(AID);
  }
private :
  static Annotation *find(AnnotationID AID, const Annotable *Ty, void *);

  ConstRules(const ConstRules &);             // Do not implement
  ConstRules &operator=(const ConstRules &);  // Do not implement
};


inline ConstPoolVal *operator!(const ConstPoolVal &V) {
  return ConstRules::get(V)->op_not(&V);
}



inline ConstPoolVal *operator+(const ConstPoolVal &V1, const ConstPoolVal &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return ConstRules::get(V1)->add(&V1, &V2);
}

inline ConstPoolVal *operator-(const ConstPoolVal &V1, const ConstPoolVal &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return ConstRules::get(V1)->sub(&V1, &V2);
}

inline ConstPoolVal *operator*(const ConstPoolVal &V1, const ConstPoolVal &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return ConstRules::get(V1)->mul(&V1, &V2);
}

inline ConstPoolBool *operator<(const ConstPoolVal &V1, 
                                const ConstPoolVal &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return ConstRules::get(V1)->lessthan(&V1, &V2);
}


//===----------------------------------------------------------------------===//
//  Implement 'derived' operators based on what we already have...
//===----------------------------------------------------------------------===//

inline ConstPoolBool *operator>(const ConstPoolVal &V1, 
                                const ConstPoolVal &V2) {
  return V2 < V1;
}

inline ConstPoolBool *operator>=(const ConstPoolVal &V1, 
                                 const ConstPoolVal &V2) {
  return (V1 < V2)->inverted();      // !(V1 < V2)
}

inline ConstPoolBool *operator<=(const ConstPoolVal &V1, 
                                 const ConstPoolVal &V2) {
  return (V1 > V2)->inverted();      // !(V1 > V2)
}


//===----------------------------------------------------------------------===//
//  Implement higher level instruction folding type instructions
//===----------------------------------------------------------------------===//

inline ConstPoolVal *ConstantFoldUnaryInstruction(unsigned Opcode, 
                                                  ConstPoolVal *V) {
  switch (Opcode) {
  case Instruction::Not:  return !*V;
  }
  return 0;
}

inline ConstPoolVal *ConstantFoldBinaryInstruction(unsigned Opcode,
                                                   ConstPoolVal *V1, 
                                                   ConstPoolVal *V2) {
  switch (Opcode) {
  case Instruction::Add:     return *V1 + *V2;
  case Instruction::Sub:     return *V1 - *V2;

  case Instruction::SetEQ:   return *V1 == *V2;
  case Instruction::SetNE:   return *V1 != *V2;
  case Instruction::SetLE:   return *V1 <= *V2;
  case Instruction::SetGE:   return *V1 >= *V2;
  case Instruction::SetLT:   return *V1 <  *V2;
  case Instruction::SetGT:   return *V1 >  *V2;
  }
  return 0;
}

} // end namespace opt
#endif
