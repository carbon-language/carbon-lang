//===-- Support/Casting.h - Allow flexible, checked, casts -------*- C++ -*--=//
//
// This file defines the isa<X>(), cast<X>(), dyn_cast<X>(), cast_or_null<X>(),
// and dyn_cast_or_null<X>() templates.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CASTING_H
#define SUPPORT_CASTING_H

// real_type - Provide a macro to get the real type of a value that might be 
// a use.  This provides a typedef 'Type' that is the argument type for all
// non UseTy types, and is the contained pointer type of the use if it is a
// UseTy.
//
template <class X> class real_type { typedef X Type; };

//===----------------------------------------------------------------------===//
//                          Type Checking Templates
//===----------------------------------------------------------------------===//

// isa<X> - Return true if the parameter to the template is an instance of the
// template type argument.  Used like this:
//
//  if (isa<Type>(myVal)) { ... }
//
template <class X, class Y>
inline bool isa(Y Val) {
  assert(Val && "isa<Ty>(NULL) invoked!");
  return X::classof(Val);
}


// cast<X> - Return the argument parameter cast to the specified type.  This
// casting operator asserts that the type is correct, so it does not return null
// on failure.  But it will correctly return NULL when the input is NULL.
// Used Like this:
//
//  cast<      Instruction>(myVal)->getParent()
//  cast<const Instruction>(myVal)->getParent()
//
template <class X, class Y>
inline X *cast(Y Val) {
  assert(isa<X>(Val) && "cast<Ty>() argument of uncompatible type!");
  return (X*)(real_type<Y>::Type)Val;
}

// cast_or_null<X> - Functionally identical to cast, except that a null value is
// accepted.
//
template <class X, class Y>
inline X *cast_or_null(Y Val) {
  assert((Val == 0 || isa<X>(Val)) &&
         "cast_or_null<Ty>() argument of uncompatible type!");
  return (X*)(real_type<Y>::Type)Val;
}


// dyn_cast<X> - Return the argument parameter cast to the specified type.  This
// casting operator returns null if the argument is of the wrong type, so it can
// be used to test for a type as well as cast if successful.  This should be
// used in the context of an if statement like this:
//
//  if (const Instruction *I = dyn_cast<const Instruction>(myVal)) { ... }
//

template <class X, class Y>
inline X *dyn_cast(Y Val) {
  return isa<X>(Val) ? cast<X>(Val) : 0;
}

// dyn_cast_or_null<X> - Functionally identical to dyn_cast, except that a null
// value is accepted.
//
template <class X, class Y>
inline X *dyn_cast_or_null(Y Val) {
  return (Val && isa<X>(Val)) ? cast<X>(Val) : 0;
}

#endif
