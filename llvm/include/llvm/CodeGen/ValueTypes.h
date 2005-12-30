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

namespace llvm {
  class Type;

/// MVT namespace - This namespace defines the ValueType enum, which contains
/// the various low-level value types.
///
namespace MVT {  // MVT = Machine Value Types
  enum ValueType {
    // If you change this numbering, you must change the values in Target.td as
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
    v16i8          =  14,   // 16 x i8
    v8i16          =  15,   //  8 x i16
    v4i32          =  16,   //  4 x i32
    v2i64          =  17,   //  2 x i64

    v4f32          =  18,   //  4 x f32
    v2f64          =  19,   //  2 x f64

    LAST_VALUETYPE,         // This always remains at the end of the list.
  };

  static inline bool isInteger(ValueType VT) {
    return (VT >= i1 && VT <= i128) || (VT >= v16i8 && VT <= v2i64);
  }
  static inline bool isFloatingPoint(ValueType VT) {
    return (VT >= f32 && VT <= f128) || (VT >= v4f32 && VT <= v2f64);
  }
  static inline bool isVector(ValueType VT) {
    return (VT >= v16i8 && VT <= v2f64);
  }

  /// getVectorType - Returns the ValueType that represents a vector NumElements
  /// in length, where each element is of type VT.  If there is no ValueType
  /// that represents this vector, a ValueType of Other is returned.
  ///
  static inline ValueType getVectorType(ValueType VT, unsigned NumElements) {
    switch (VT) {
    default: 
      break;
    case MVT::i32:
      if (NumElements == 4) return MVT::v4i32;
      break;
    case MVT::f32:
      if (NumElements == 4) return MVT::v4f32;
      break;
    }
    return MVT::Other;
  }
  
  static inline unsigned getSizeInBits(ValueType VT) {
    switch (VT) {
    default: assert(0 && "ValueType has no known size!");
    case MVT::i1  : return 1;
    case MVT::i8  : return 8;
    case MVT::i16 : return 16;
    case MVT::f32 :
    case MVT::i32 : return 32;
    case MVT::f64 :
    case MVT::i64 : return 64;
    case MVT::f80 : return 80;
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

  /// MVT::getValueTypeString - This function returns value type as a string,
  /// e.g. "i32".
  const char *getValueTypeString(ValueType VT);

  /// MVT::getTypeForValueType - This method returns an LLVM type corresponding
  /// to the specified ValueType.  For integer types, this returns an unsigned
  /// type.  Note that this will abort for types that cannot be represented.
  const Type *getTypeForValueType(ValueType VT);
};

} // End llvm namespace

#endif
