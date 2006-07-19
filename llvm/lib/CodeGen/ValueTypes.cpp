//===-- ValueTypes.cpp - Implementation of MVT::ValueType methods ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements methods in the CodeGen/ValueTypes.h header.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Type.h"
using namespace llvm;

/// MVT::getValueTypeString - This function returns value type as a string,
/// e.g. "i32".
const char *MVT::getValueTypeString(MVT::ValueType VT) {
  switch (VT) {
  default: assert(0 && "Invalid ValueType!");
  case MVT::i1:    return "i1";
  case MVT::i8:    return "i8";
  case MVT::i16:   return "i16";
  case MVT::i32:   return "i32";
  case MVT::i64:   return "i64";
  case MVT::i128:  return "i128";
  case MVT::f32:   return "f32";
  case MVT::f64:   return "f64";
  case MVT::f80:   return "f80";
  case MVT::f128:  return "f128";
  case MVT::isVoid:return "isVoid";
  case MVT::Other: return "ch";
  case MVT::Flag:  return "flag";
  case MVT::Vector:return "vec";
  case MVT::v8i8:  return "v8i8";
  case MVT::v4i16: return "v4i16";
  case MVT::v2i32: return "v2i32";
  case MVT::v16i8: return "v16i8";
  case MVT::v8i16: return "v8i16";
  case MVT::v4i32: return "v4i32";
  case MVT::v2i64: return "v2i64";
  case MVT::v4f32: return "v4f32";
  case MVT::v2f64: return "v2f64";
  }
}

/// MVT::getVectorType - Returns the ValueType that represents a vector
/// NumElements in length, where each element is of type VT.  If there is no
/// ValueType that represents this vector, a ValueType of Other is returned.
///
MVT::ValueType MVT::getVectorType(ValueType VT, unsigned NumElements) {
  switch (VT) {
  default: 
    break;
  case MVT::i8:
    if (NumElements == 8)  return MVT::v8i8;
    if (NumElements == 16) return MVT::v16i8;
    break;
  case MVT::i16:
    if (NumElements == 4)  return MVT::v4i16;
    if (NumElements == 8)  return MVT::v8i16;
    break;
  case MVT::i32:
    if (NumElements == 2)  return MVT::v2i32;
    if (NumElements == 4)  return MVT::v4i32;
    break;
  case MVT::i64:
    if (NumElements == 2)  return MVT::v2i64;
    break;
  case MVT::f32:
    if (NumElements == 2)  return MVT::v2f32;
    if (NumElements == 4)  return MVT::v4f32;
    break;
  case MVT::f64:
    if (NumElements == 2)  return MVT::v2f64;
    break;
  }
  return MVT::Other;
}

/// MVT::getTypeForValueType - This method returns an LLVM type corresponding
/// to the specified ValueType.  For integer types, this returns an unsigned
/// type.  Note that this will abort for types that cannot be represented.
const Type *MVT::getTypeForValueType(MVT::ValueType VT) {
  switch (VT) {
  default: assert(0 && "ValueType does not correspond to LLVM type!");
  case MVT::isVoid:return Type::VoidTy;
  case MVT::i1:    return Type::BoolTy;
  case MVT::i8:    return Type::UByteTy;
  case MVT::i16:   return Type::UShortTy;
  case MVT::i32:   return Type::UIntTy;
  case MVT::i64:   return Type::ULongTy;
  case MVT::f32:   return Type::FloatTy;
  case MVT::f64:   return Type::DoubleTy;
  }
}
