//===- CodeGen/ValueTypes.h - Low-Level Target independ. types --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the set of low-level target independent types which various
// values in the code generator are.  This allows the target specific behavior
// of instructions to be described to target independent passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_VALUETYPES_H
#define LLVM_CODEGEN_VALUETYPES_H

#include <cassert>
#include "llvm/Support/DataTypes.h"

namespace llvm {
  class Type;

/// MVT namespace - This namespace defines the ValueType enum, which contains
/// the various low-level value types.
///
namespace MVT {  // MVT = Machine Value Types
  enum ValueType {
    // If you change this numbering, you must change the values in ValueTypes.td
    // well!
    Other          =   0,   // This is a non-standard value
    i1             =   1,   // This is a 1 bit integer value
    i8             =   2,   // This is an 8 bit integer value
    i16            =   3,   // This is a 16 bit integer value
    i32            =   4,   // This is a 32 bit integer value
    i64            =   5,   // This is a 64 bit integer value
    i128           =   6,   // This is a 128 bit integer value

    f32            =   7,   // This is a 32 bit floating point value
    f64            =   8,   // This is a 64 bit floating point value
    f80            =   9,   // This is a 80 bit floating point value
    f128           =  10,   // This is a 128 bit floating point value
    Flag           =  11,   // This is a condition code or machine flag.

    isVoid         =  12,   // This has no value
    
    Vector         =  13,   // This is an abstract vector type, which will
                            // be expanded into a target vector type, or scalars
                            // if no matching vector type is available.

    v8i8           =  14,   //  8 x i8
    v4i16          =  15,   //  4 x i16
    v2i32          =  16,   //  2 x i32
    v16i8          =  17,   // 16 x i8
    v8i16          =  18,   //  8 x i16
    v4i32          =  19,   //  4 x i32
    v2i64          =  20,   //  2 x i64

    v2f32          =  21,   //  2 x f32
    v4f32          =  22,   //  4 x f32
    v2f64          =  23,   //  2 x f64
    FIRST_VECTOR_VALUETYPE = v8i8,
    LAST_VECTOR_VALUETYPE  = v2f64,

    LAST_VALUETYPE =  24,   // This always remains at the end of the list.

    // iPTR - An int value the size of the pointer of the current
    // target.  This should only be used internal to tblgen!
    iPTR           = 255
  };

  /// MVT::isInteger - Return true if this is a simple integer, or a packed
  /// vector integer type.
  static inline bool isInteger(ValueType VT) {
    return (VT >= i1 && VT <= i128) || (VT >= v8i8 && VT <= v2i64);
  }

  /// MVT::isFloatingPoint - Return true if this is a simple FP, or a packed
  /// vector FP type.
  static inline bool isFloatingPoint(ValueType VT) {
    return (VT >= f32 && VT <= f128) || (VT >= v4f32 && VT <= v2f64);
  }
  
  /// MVT::isVector - Return true if this is a packed vector type (i.e. not 
  /// MVT::Vector).
  static inline bool isVector(ValueType VT) {
    return VT >= FIRST_VECTOR_VALUETYPE && VT <= LAST_VECTOR_VALUETYPE;
  }
  
  /// MVT::getSizeInBits - Return the size of the specified value type in bits.
  ///
  static inline unsigned getSizeInBits(ValueType VT) {
    switch (VT) {
    default: assert(0 && "ValueType has no known size!");
    case MVT::i1  :  return 1;
    case MVT::i8  :  return 8;
    case MVT::i16 :  return 16;
    case MVT::f32 :
    case MVT::i32 :  return 32;
    case MVT::f64 :
    case MVT::i64 :
    case MVT::v8i8:
    case MVT::v4i16:
    case MVT::v2i32: 
    case MVT::v2f32: return 64;
    case MVT::f80 :  return 80;
    case MVT::f128:
    case MVT::i128: 
    case MVT::v16i8:
    case MVT::v8i16:
    case MVT::v4i32:
    case MVT::v2i64:
    case MVT::v4f32:
    case MVT::v2f64: return 128;
    }
  }
  
  /// MVT::getVectorType - Returns the ValueType that represents a vector
  /// NumElements in length, where each element is of type VT.  If there is no
  /// ValueType that represents this vector, a ValueType of Other is returned.
  ///
  ValueType getVectorType(ValueType VT, unsigned NumElements);
    
  /// MVT::getVectorBaseType - Given a packed vector type, return the type of
  /// each element.
  static inline ValueType getVectorBaseType(ValueType VT) {
    switch (VT) {
    default: assert(0 && "Invalid vector type!");
    case v8i8 :
    case v16i8: return i8;
    case v4i16:
    case v8i16: return i16; 
    case v2i32:
    case v4i32: return i32;
    case v2i64: return i64;
    case v2f32:
    case v4f32: return f32;
    case v2f64: return f64;
    }
  }
  
  /// MVT::getVectorNumElements - Given a packed vector type, return the number
  /// of elements it contains.
  static inline unsigned getVectorNumElements(ValueType VT) {
    switch (VT) {
      default: assert(0 && "Invalid vector type!");
      case v16i8: return 16;
      case v8i8 :
      case v8i16: return 8;
      case v4i16:
      case v4i32: 
      case v4f32: return 4;
      case v2i32:
      case v2i64:
      case v2f32:
      case v2f64: return 2;
    }
  }
  
  /// MVT::getIntVectorWithNumElements - Return any integer vector type that has
  /// the specified number of elements.
  static inline ValueType getIntVectorWithNumElements(unsigned NumElts) {
    switch (NumElts) {
    default: assert(0 && "Invalid vector type!");
    case  2: return v2i32;
    case  4: return v4i16;
    case  8: return v8i8;
    case 16: return v16i8;
    }
  }
  
  
  /// MVT::getIntVTBitMask - Return an integer with 1's every place there are
  /// bits in the specified integer value type.
  static inline uint64_t getIntVTBitMask(ValueType VT) {
    assert(isInteger(VT) && !isVector(VT) && "Only applies to int scalars!");
    return ~uint64_t(0UL) >> (64-getSizeInBits(VT));
  }
  /// MVT::getIntVTSignBit - Return an integer with a 1 in the position of the
  /// sign bit for the specified integer value type.
  static inline uint64_t getIntVTSignBit(ValueType VT) {
    assert(isInteger(VT) && !isVector(VT) && "Only applies to int scalars!");
    return uint64_t(1UL) << (getSizeInBits(VT)-1);
  }

  /// MVT::getValueTypeString - This function returns value type as a string,
  /// e.g. "i32".
  const char *getValueTypeString(ValueType VT);

  /// MVT::getTypeForValueType - This method returns an LLVM type corresponding
  /// to the specified ValueType.  For integer types, this returns an unsigned
  /// type.  Note that this will abort for types that cannot be represented.
  const Type *getTypeForValueType(ValueType VT);
}

} // End llvm namespace

#endif
