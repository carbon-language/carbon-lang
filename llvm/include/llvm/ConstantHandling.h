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
// WARNING: These operators return pointers to newly 'new'd objects.  You MUST
//          make sure to free them if you don't want them hanging around. Also,
//          note that these may return a null object if I don't know how to 
//          perform those operations on the specified constant types.
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
#include "llvm/Type.h"

//===----------------------------------------------------------------------===//
//  Implement == directly...
//===----------------------------------------------------------------------===//

inline ConstPoolBool *operator==(const ConstPoolVal &V1, 
                                 const ConstPoolVal &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return new ConstPoolBool(V1.equals(&V2));
}

//===----------------------------------------------------------------------===//
//  Implement all other operators indirectly through TypeRules system
//===----------------------------------------------------------------------===//

class ConstRules {
protected:
  inline ConstRules() {}  // Can only be subclassed...
public:
  // Unary Operators...
  virtual ConstPoolVal *neg(const ConstPoolVal *V) const = 0;
  virtual ConstPoolVal *not(const ConstPoolVal *V) const = 0;

  // Binary Operators...
  virtual ConstPoolVal *add(const ConstPoolVal *V1, 
                            const ConstPoolVal *V2) const = 0;
  virtual ConstPoolVal *sub(const ConstPoolVal *V1, 
                            const ConstPoolVal *V2) const = 0;

  virtual ConstPoolBool *lessthan(const ConstPoolVal *V1, 
                                  const ConstPoolVal *V2) const = 0;

  // ConstRules::get - A type will cache its own type rules if one is needed...
  // we just want to make sure to hit the cache instead of doing it indirectly,
  //  if possible...
  //
  static inline const ConstRules *get(const ConstPoolVal &V) {
    const ConstRules *Result = V.getType()->getConstRules();
    return Result ? Result : find(V.getType());
  }
private :
  static const ConstRules *find(const Type *Ty);

  ConstRules(const ConstRules &);             // Do not implement
  ConstRules &operator=(const ConstRules &);  // Do not implement
};


inline ConstPoolVal *operator-(const ConstPoolVal &V) {
  return ConstRules::get(V)->neg(&V);
}

inline ConstPoolVal *operator!(const ConstPoolVal &V) {
  return ConstRules::get(V)->not(&V);
}



inline ConstPoolVal *operator+(const ConstPoolVal &V1, const ConstPoolVal &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return ConstRules::get(V1)->add(&V1, &V2);
}

inline ConstPoolVal *operator-(const ConstPoolVal &V1, const ConstPoolVal &V2) {
  assert(V1.getType() == V2.getType() && "Constant types must be identical!");
  return ConstRules::get(V1)->sub(&V1, &V2);
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

inline ConstPoolBool *operator!=(const ConstPoolVal &V1, 
                                 const ConstPoolVal &V2) {
  ConstPoolBool *Result = V1 == V2;
  Result->setValue(!Result->getValue());     // Invert value
  return Result;     // !(V1 == V2)
}

inline ConstPoolBool *operator>=(const ConstPoolVal &V1, 
                                 const ConstPoolVal &V2) {
  ConstPoolBool *Result = V1 < V2;
  Result->setValue(!Result->getValue());     // Invert value
  return Result;      // !(V1 < V2)
}

inline ConstPoolBool *operator<=(const ConstPoolVal &V1, 
                                 const ConstPoolVal &V2) {
  ConstPoolBool *Result = V1 > V2;
  Result->setValue(!Result->getValue());     // Invert value
  return Result;      // !(V1 > V2)
}

#endif
