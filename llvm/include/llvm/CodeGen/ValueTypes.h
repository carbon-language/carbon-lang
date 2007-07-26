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
#include <string>
#include "llvm/Support/DataTypes.h"

namespace llvm {
  class Type;

/// MVT namespace - This namespace defines the SimpleValueType enum, which
/// contains the various low-level value types, and the ValueType typedef.
///
namespace MVT {  // MVT = Machine Value Types
  enum SimpleValueType {
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
    
    v8i8           =  13,   //  8 x i8
    v4i16          =  14,   //  4 x i16
    v2i32          =  15,   //  2 x i32
    v1i64          =  16,   //  1 x i64
    v16i8          =  17,   // 16 x i8
    v8i16          =  18,   //  8 x i16
    v3i32           = 19,   //  3 x i32
    v4i32          =  20,   //  4 x i32
    v2i64          =  21,   //  2 x i64

    v2f32          =  22,   //  2 x f32
    v3f32           = 23,   //  3 x f32
    v4f32          =  24,   //  4 x f32
    v2f64          =  25,   //  2 x f64
    
    FIRST_VECTOR_VALUETYPE = v8i8,
    LAST_VECTOR_VALUETYPE  = v2f64,

    LAST_VALUETYPE =  26,   // This always remains at the end of the list.

    // iAny - An integer value of any bit width. This is used for intrinsics
    // that have overloadings based on integer bit widths. This is only for
    // tblgen's consumption!
    iAny           = 254,   

    // iPTR - An int value the size of the pointer of the current
    // target.  This should only be used internal to tblgen!
    iPTR           = 255
  };

  /// MVT::ValueType - This type holds low-level value types. Valid values
  /// include any of the values in the SimpleValueType enum, or any value
  /// returned from a function in the MVT namespace that has a ValueType
  /// return type. Any value type equal to one of the SimpleValueType enum
  /// values is a "simple" value type. All other value types are "extended".
  ///
  /// Note that simple doesn't necessary mean legal for the target machine.
  /// All legal value types must be simple, but often there are some simple
  /// value types that are not legal.
  ///
  /// @internal
  /// Currently extended types are always vector types. Extended types are
  /// encoded by having the first SimpleTypeBits bits encode the vector
  /// element type (which must be a scalar type) and the remaining upper
  /// bits encode the vector length, offset by one.
  typedef uint32_t ValueType;

  static const int SimpleTypeBits = 8;

  static const uint32_t SimpleTypeMask =
    (~uint32_t(0) << (32 - SimpleTypeBits)) >> (32 - SimpleTypeBits);

  /// MVT::isExtendedVT - Test if the given ValueType is extended
  /// (as opposed to being simple).
  static inline bool isExtendedVT(ValueType VT) {
    return VT > SimpleTypeMask;
  }

  /// MVT::isInteger - Return true if this is an integer, or a vector integer
  /// type.
  static inline bool isInteger(ValueType VT) {
    ValueType SVT = VT & SimpleTypeMask;
    return (SVT >= i1 && SVT <= i128) || (SVT >= v8i8 && SVT <= v2i64);
  }
  
  /// MVT::isFloatingPoint - Return true if this is an FP, or a vector FP type.
  static inline bool isFloatingPoint(ValueType VT) {
    ValueType SVT = VT & SimpleTypeMask;
    return (SVT >= f32 && SVT <= f128) || (SVT >= v2f32 && SVT <= v2f64);
  }
  
  /// MVT::isVector - Return true if this is a vector value type.
  static inline bool isVector(ValueType VT) {
    return (VT >= FIRST_VECTOR_VALUETYPE && VT <= LAST_VECTOR_VALUETYPE) ||
           isExtendedVT(VT);
  }
  
  /// MVT::getVectorElementType - Given a vector type, return the type of
  /// each element.
  static inline ValueType getVectorElementType(ValueType VT) {
    switch (VT) {
    default:
      if (isExtendedVT(VT))
        return VT & SimpleTypeMask;
      assert(0 && "Invalid vector type!");
    case v8i8 :
    case v16i8: return i8;
    case v4i16:
    case v8i16: return i16; 
    case v2i32:
    case v3i32:
    case v4i32: return i32;
    case v1i64:
    case v2i64: return i64;
    case v2f32:
    case v3f32:
    case v4f32: return f32;
    case v2f64: return f64;
    }
  }
  
  /// MVT::getVectorNumElements - Given a vector type, return the
  /// number of elements it contains.
  static inline unsigned getVectorNumElements(ValueType VT) {
    switch (VT) {
    default:
      if (isExtendedVT(VT))
        return ((VT & ~SimpleTypeMask) >> SimpleTypeBits) - 1;
      assert(0 && "Invalid vector type!");
    case v16i8: return 16;
    case v8i8 :
    case v8i16: return 8;
    case v4i16:
    case v4i32: 
    case v4f32: return 4;
    case v3i32:
    case v3f32: return 3;
    case v2i32:
    case v2i64:
    case v2f32:
    case v2f64: return 2;
    case v1i64: return 1;
    }
  }
  
  /// MVT::getSizeInBits - Return the size of the specified value type
  /// in bits.
  ///
  static inline unsigned getSizeInBits(ValueType VT) {
    switch (VT) {
    default:
      if (isExtendedVT(VT))
        return getSizeInBits(getVectorElementType(VT)) *
               getVectorNumElements(VT);
      assert(0 && "ValueType has no known size!");
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
    case MVT::v1i64:
    case MVT::v2f32: return 64;
    case MVT::f80 :  return 80;
    case MVT::v3i32:
    case MVT::v3f32: return 96;
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
  /// NumElements in length, where each element is of type VT.
  ///
  static inline ValueType getVectorType(ValueType VT, unsigned NumElements) {
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
      if (NumElements == 3)  return MVT::v3i32;
      if (NumElements == 4)  return MVT::v4i32;
      break;
    case MVT::i64:
      if (NumElements == 1)  return MVT::v1i64;
      if (NumElements == 2)  return MVT::v2i64;
      break;
    case MVT::f32:
      if (NumElements == 2)  return MVT::v2f32;
      if (NumElements == 3)  return MVT::v3f32;
      if (NumElements == 4)  return MVT::v4f32;
      break;
    case MVT::f64:
      if (NumElements == 2)  return MVT::v2f64;
      break;
    }
    ValueType Result = VT | ((NumElements + 1) << SimpleTypeBits);
    assert(getVectorElementType(Result) == VT &&
           "Bad vector element type!");
    assert(getVectorNumElements(Result) == NumElements &&
           "Bad vector length!");
    return Result;
  }

  /// MVT::getIntVectorWithNumElements - Return any integer vector type that has
  /// the specified number of elements.
  static inline ValueType getIntVectorWithNumElements(unsigned NumElts) {
    switch (NumElts) {
    default: return getVectorType(i8, NumElts);
    case  1: return v1i64;
    case  2: return v2i32;
    case  3: return v3i32;
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
  std::string getValueTypeString(ValueType VT);

  /// MVT::getTypeForValueType - This method returns an LLVM type corresponding
  /// to the specified ValueType.  For integer types, this returns an unsigned
  /// type.  Note that this will abort for types that cannot be represented.
  const Type *getTypeForValueType(ValueType VT);
  
  /// MVT::getValueType - Return the value type corresponding to the specified
  /// type.  This returns all pointers as MVT::iPTR.  If HandleUnknown is true,
  /// unknown types are returned as Other, otherwise they are invalid.
  ValueType getValueType(const Type *Ty, bool HandleUnknown = false);
}

} // End llvm namespace

#endif
