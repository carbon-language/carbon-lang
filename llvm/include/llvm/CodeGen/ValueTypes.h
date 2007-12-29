//===- CodeGen/ValueTypes.h - Low-Level Target independ. types --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Support/MathExtras.h"

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

    FIRST_INTEGER_VALUETYPE = i1,
    LAST_INTEGER_VALUETYPE  = i128,

    f32            =   7,   // This is a 32 bit floating point value
    f64            =   8,   // This is a 64 bit floating point value
    f80            =   9,   // This is a 80 bit floating point value
    f128           =  10,   // This is a 128 bit floating point value
    ppcf128        =  11,   // This is a PPC 128-bit floating point value
    Flag           =  12,   // This is a condition code or machine flag.

    isVoid         =  13,   // This has no value

    v8i8           =  14,   //  8 x i8
    v4i16          =  15,   //  4 x i16
    v2i32          =  16,   //  2 x i32
    v1i64          =  17,   //  1 x i64
    v16i8          =  18,   // 16 x i8
    v8i16          =  19,   //  8 x i16
    v3i32          =  20,   //  3 x i32
    v4i32          =  21,   //  4 x i32
    v2i64          =  22,   //  2 x i64

    v2f32          =  23,   //  2 x f32
    v3f32          =  24,   //  3 x f32
    v4f32          =  25,   //  4 x f32
    v2f64          =  26,   //  2 x f64

    FIRST_VECTOR_VALUETYPE = v8i8,
    LAST_VECTOR_VALUETYPE  = v2f64,

    LAST_VALUETYPE =  27,   // This always remains at the end of the list.

    // fAny - Any floating-point or vector floating-point value. This is used
    // for intrinsics that have overloadings based on floating-point types.
    // This is only for tblgen's consumption!
    fAny           =  253,

    // iAny - An integer or vector integer value of any bit width. This is
    // used for intrinsics that have overloadings based on integer bit widths.
    // This is only for tblgen's consumption!
    iAny           =  254,

    // iPTR - An int value the size of the pointer of the current
    // target.  This should only be used internal to tblgen!
    iPTR           =  255
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
  /// Extended types are either vector types or arbitrary precision integers.
  /// Arbitrary precision integers have iAny in the first SimpleTypeBits bits,
  /// and the bit-width in the next PrecisionBits bits, offset by minus one.
  /// Vector types are encoded by having the first SimpleTypeBits+PrecisionBits
  /// bits encode the vector element type (which must be a scalar type, possibly
  /// an arbitrary precision integer) and the remaining VectorBits upper bits
  /// encode the vector length, offset by one.
  ///
  /// 31--------------16-----------8-------------0
  ///  | Vector length | Precision | Simple type |
  ///  |               |      Vector element     |
  ///
  /// Note that the verifier currently requires the top bit to be zero.

  typedef uint32_t ValueType;

  static const int SimpleTypeBits = 8;
  static const int PrecisionBits  = 8;
  static const int VectorBits     = 32 - SimpleTypeBits - PrecisionBits;

  static const uint32_t SimpleTypeMask =
    (~uint32_t(0) << (32 - SimpleTypeBits)) >> (32 - SimpleTypeBits);

  static const uint32_t PrecisionMask =
    ((~uint32_t(0) << VectorBits) >> (32 - PrecisionBits)) << SimpleTypeBits;

  static const uint32_t VectorMask =
    (~uint32_t(0) >> (32 - VectorBits)) << (32 - VectorBits);

  static const uint32_t ElementMask =
    (~uint32_t(0) << VectorBits) >> VectorBits;

  /// MVT::isExtendedVT - Test if the given ValueType is extended
  /// (as opposed to being simple).
  static inline bool isExtendedVT(ValueType VT) {
    return VT > SimpleTypeMask;
  }

  /// MVT::isInteger - Return true if this is an integer, or a vector integer
  /// type.
  static inline bool isInteger(ValueType VT) {
    ValueType SVT = VT & SimpleTypeMask;
    return (SVT >= FIRST_INTEGER_VALUETYPE && SVT <= LAST_INTEGER_VALUETYPE) ||
      (SVT >= v8i8 && SVT <= v2i64) || (SVT == iAny && (VT & PrecisionMask));
  }

  /// MVT::isFloatingPoint - Return true if this is an FP, or a vector FP type.
  static inline bool isFloatingPoint(ValueType VT) {
    ValueType SVT = VT & SimpleTypeMask;
    return (SVT >= f32 && SVT <= ppcf128) || (SVT >= v2f32 && SVT <= v2f64);
  }

  /// MVT::isVector - Return true if this is a vector value type.
  static inline bool isVector(ValueType VT) {
    return (VT >= FIRST_VECTOR_VALUETYPE && VT <= LAST_VECTOR_VALUETYPE) ||
           (VT & VectorMask);
  }

  /// MVT::getVectorElementType - Given a vector type, return the type of
  /// each element.
  static inline ValueType getVectorElementType(ValueType VT) {
    assert(isVector(VT) && "Invalid vector type!");
    switch (VT) {
    default:
      assert(isExtendedVT(VT) && "Unknown simple vector type!");
      return VT & ElementMask;
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
    assert(isVector(VT) && "Invalid vector type!");
    switch (VT) {
    default:
      assert(isExtendedVT(VT) && "Unknown simple vector type!");
      return ((VT & VectorMask) >> (32 - VectorBits)) - 1;
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
      assert(isExtendedVT(VT) && "ValueType has no known size!");
      if (isVector(VT))
        return getSizeInBits(getVectorElementType(VT)) *
               getVectorNumElements(VT);
      if (isInteger(VT))
        return ((VT & PrecisionMask) >> SimpleTypeBits) + 1;
      assert(0 && "Unknown value type!");
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
    case MVT::ppcf128:
    case MVT::i128:
    case MVT::v16i8:
    case MVT::v8i16:
    case MVT::v4i32:
    case MVT::v2i64:
    case MVT::v4f32:
    case MVT::v2f64: return 128;
    }
  }

  /// MVT::getStoreSizeInBits - Return the number of bits overwritten by a
  /// store of the specified value type.
  ///
  static inline unsigned getStoreSizeInBits(ValueType VT) {
    return (getSizeInBits(VT) + 7)/8*8;
  }

  /// MVT::getIntegerType - Returns the ValueType that represents an integer
  /// with the given number of bits.
  ///
  static inline ValueType getIntegerType(unsigned BitWidth) {
    switch (BitWidth) {
    default:
      break;
    case 1:
      return MVT::i1;
    case 8:
      return MVT::i8;
    case 16:
      return MVT::i16;
    case 32:
      return MVT::i32;
    case 64:
      return MVT::i64;
    case 128:
      return MVT::i128;
    }
    ValueType Result = iAny |
      (((BitWidth - 1) << SimpleTypeBits) & PrecisionMask);
    assert(getSizeInBits(Result) == BitWidth && "Bad bit width!");
    return Result;
  }

  /// MVT::RoundIntegerType - Rounds the bit-width of the given integer
  /// ValueType up to the nearest power of two (and at least to eight),
  /// and returns the integer ValueType with that number of bits.
  ///
  static inline ValueType RoundIntegerType(ValueType VT) {
    assert(isInteger(VT) && !isVector(VT) && "Invalid integer type!");
    unsigned BitWidth = getSizeInBits(VT);
    if (BitWidth <= 8)
      return MVT::i8;
    else
      return getIntegerType(1 << Log2_32_Ceil(BitWidth));
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
    // Set the length with the top bit forced to zero (needed by the verifier).
    ValueType Result = VT | (((NumElements + 1) << (33 - VectorBits)) >> 1);
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
