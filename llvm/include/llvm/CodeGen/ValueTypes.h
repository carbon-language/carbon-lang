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
  class LLVMContext;

  struct MVT { // MVT = Machine Value Type
  public:
    enum SimpleValueType {
      // If you change this numbering, you must change the values in
      // ValueTypes.td as well!
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

      v2i8           =  14,   //  2 x i8
      v4i8           =  15,   //  4 x i8
      v8i8           =  16,   //  8 x i8
      v16i8          =  17,   // 16 x i8
      v24i8          =  18,   // 24 x i8
      v32i8          =  19,   // 32 x i8
      v48i8          =  20,   // 48 x i8
      v64i8          =  21,   // 64 x i8

      v2i16          =  22,   //  2 x i16
      v4i16          =  23,   //  4 x i16
      v8i16          =  24,   //  8 x i16
      v12i16         =  25,   // 12 x i16
      v16i16         =  26,   // 16 x i16
      v24i16         =  27,   // 24 x i16
      v32i16         =  28,   // 32 x i16

      v2i32          =  29,   //  2 x i32
      v3i32          =  30,   //  3 x i32
      v4i32          =  31,   //  4 x i32
      v6i32          =  32,   //  6 x i32
      v8i32          =  33,   //  8 x i32
      v12i32         =  34,   // 12 x i32
      v16i32         =  35,   // 16 x i32

      v1i64          =  36,   //  1 x i64
      v2i64          =  37,   //  2 x i64
      v3i64          =  38,   //  3 x i64
      v4i64          =  39,   //  4 x i64
      v6i64          =  40,   //  6 x i64
      v8i64          =  41,   //  8 x i64

      v2f32          =  42,   //  2 x f32
      v3f32          =  43,   //  3 x f32
      v4f32          =  44,   //  4 x f32
      v6f32          =  45,   //  6 x f32
      v8f32          =  46,   //  8 x f32
      v12f32         =  47,   // 12 x f32
      v16f32         =  48,   // 16 x f32

      v2f64          =  49,   //  2 x f64
      v4f64          =  50,   //  4 x f64
  
      FIRST_VECTOR_VALUETYPE = v2i8,
      LAST_VECTOR_VALUETYPE  = v4f64,

      LAST_VALUETYPE =  51,   // This always remains at the end of the list.

      // This is the current maximum for LAST_VALUETYPE.
      // MVT::MAX_ALLOWED_VALUETYPE is used for asserts and to size bit vectors
      // This value must be a multiple of 32.
      MAX_ALLOWED_VALUETYPE = 64,

      // Metadata - This is MDNode or MDString. 
      Metadata       = 251,

      // iPTRAny - An int value the size of the pointer of the current
      // target to any address space. This must only be used internal to
      // tblgen. Other than for overloading, we treat iPTRAny the same as iPTR.
      iPTRAny        =  252,

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
      iPTR           =  255,

      // LastSimpleValueType - The greatest valid SimpleValueType value.
      LastSimpleValueType = 255
    };

  private:
    /// This union holds low-level value types. Valid values include any of
    /// the values in the SimpleValueType enum, or any value returned from one
    /// of the MVT methods.  Any value type equal to one of the SimpleValueType
    /// enum values is a "simple" value type.  All others are "extended".
    ///
    /// Note that simple doesn't necessary mean legal for the target machine.
    /// All legal value types must be simple, but often there are some simple
    /// value types that are not legal.
    ///
    union {
      uintptr_t V;
      const Type *LLVMTy;
    };

  public:
    MVT() {}
    MVT(SimpleValueType S) : V(S) {}

    bool operator==(const MVT VT) const {
      return getRawBits() == VT.getRawBits();
    }
    bool operator!=(const MVT VT) const {
      return getRawBits() != VT.getRawBits();
    }

    /// getFloatingPointVT - Returns the MVT that represents a floating point
    /// type with the given number of bits.  There are two floating point types
    /// with 128 bits - this returns f128 rather than ppcf128.
    static MVT getFloatingPointVT(unsigned BitWidth) {
      switch (BitWidth) {
      default:
        assert(false && "Bad bit width!");
      case 32:
        return f32;
      case 64:
        return f64;
      case 80:
        return f80;
      case 128:
        return f128;
      }
    }

    /// getIntegerVT - Returns the MVT that represents an integer with the given
    /// number of bits.
    static MVT getIntegerVT(unsigned BitWidth) {
      switch (BitWidth) {
      default:
        break;
      case 1:
        return i1;
      case 8:
        return i8;
      case 16:
        return i16;
      case 32:
        return i32;
      case 64:
        return i64;
      case 128:
        return i128;
      }
      return getExtendedIntegerVT(BitWidth);
    }

    /// getVectorVT - Returns the MVT that represents a vector NumElements in
    /// length, where each element is of type VT.
    static MVT getVectorVT(MVT VT, unsigned NumElements) {
      switch (VT.V) {
      default:
        break;
      case i8:
        if (NumElements == 2)  return v2i8;
        if (NumElements == 4)  return v4i8;
        if (NumElements == 8)  return v8i8;
        if (NumElements == 16) return v16i8;
        if (NumElements == 24) return v24i8;
        if (NumElements == 32) return v32i8;
        if (NumElements == 48) return v48i8;
        if (NumElements == 64) return v64i8;
        break;
      case i16:
        if (NumElements == 2)  return v2i16;
        if (NumElements == 4)  return v4i16;
        if (NumElements == 8)  return v8i16;
        if (NumElements == 12) return v12i16;
        if (NumElements == 16) return v16i16;
        if (NumElements == 24) return v24i16;
        if (NumElements == 32) return v32i16;
        break;
      case i32:
        if (NumElements == 2)  return v2i32;
        if (NumElements == 3)  return v3i32;
        if (NumElements == 4)  return v4i32;
        if (NumElements == 6)  return v6i32;
        if (NumElements == 8)  return v8i32;
        if (NumElements == 12) return v12i32;
        if (NumElements == 16) return v16i32;
        break;
      case i64:
        if (NumElements == 1)  return v1i64;
        if (NumElements == 2)  return v2i64;
        if (NumElements == 3)  return v3i64;
        if (NumElements == 4)  return v4i64;
        if (NumElements == 6)  return v6i64;
        if (NumElements == 8)  return v8i64;
        break;
      case f32:
        if (NumElements == 2)  return v2f32;
        if (NumElements == 3)  return v3f32;
        if (NumElements == 4)  return v4f32;
        if (NumElements == 6)  return v6f32;
        if (NumElements == 8)  return v8f32;
        if (NumElements == 12) return v12f32;
        if (NumElements == 16) return v16f32;
        break;
      case f64:
        if (NumElements == 2)  return v2f64;
        if (NumElements == 4)  return v4f64;
        break;
      }
      return getExtendedVectorVT(VT, NumElements);
    }

    /// getIntVectorWithNumElements - Return any integer vector type that has
    /// the specified number of elements.
    static MVT getIntVectorWithNumElements(unsigned NumElts) {
      switch (NumElts) {
      default: return getVectorVT(i8, NumElts);
      case  1: return v1i64;
      case  2: return v2i32;
      case  3: return v3i32;
      case  4: return v4i16;
      case  8: return v8i8;
      case 16: return v16i8;
      }
    }

    /// isSimple - Test if the given MVT is simple (as opposed to being
    /// extended).
    bool isSimple() const {
      return V <= LastSimpleValueType;
    }

    /// isExtended - Test if the given MVT is extended (as opposed to
    /// being simple).
    bool isExtended() const {
      return !isSimple();
    }

    /// isFloatingPoint - Return true if this is a FP, or a vector FP type.
    bool isFloatingPoint() const {
      return isSimple() ?
             ((V >= f32 && V <= ppcf128) ||
              (V >= v2f32 && V <= v4f64)) : isExtendedFloatingPoint();
    }

    /// isInteger - Return true if this is an integer, or a vector integer type.
    bool isInteger() const {
      return isSimple() ?
             ((V >= FIRST_INTEGER_VALUETYPE && V <= LAST_INTEGER_VALUETYPE) ||
              (V >= v2i8 && V <= v8i64)) : isExtendedInteger();
    }

    /// isVector - Return true if this is a vector value type.
    bool isVector() const {
      return isSimple() ?
             (V >= FIRST_VECTOR_VALUETYPE && V <= LAST_VECTOR_VALUETYPE) :
             isExtendedVector();
    }

    /// is64BitVector - Return true if this is a 64-bit vector type.
    bool is64BitVector() const {
      return isSimple() ?
             (V==v8i8 || V==v4i16 || V==v2i32 || V==v1i64 || V==v2f32) :
             isExtended64BitVector();
    }

    /// is128BitVector - Return true if this is a 128-bit vector type.
    bool is128BitVector() const {
      return isSimple() ?
             (V==v16i8 || V==v8i16 || V==v4i32 ||
              V==v2i64 || V==v4f32 || V==v2f64) :
             isExtended128BitVector();
    }

    /// is256BitVector - Return true if this is a 256-bit vector type.
    inline bool is256BitVector() const {
      return isSimple() ? 
             (V==v8f32 || V==v4f64 || V==v32i8 || V==v16i16 || V==v8i32 ||
              V==v4i64) : isExtended256BitVector();
    }

    /// isByteSized - Return true if the bit size is a multiple of 8.
    bool isByteSized() const {
      return (getSizeInBits() & 7) == 0;
    }

    /// isRound - Return true if the size is a power-of-two number of bytes.
    bool isRound() const {
      unsigned BitSize = getSizeInBits();
      return BitSize >= 8 && !(BitSize & (BitSize - 1));
    }

    /// bitsEq - Return true if this has the same number of bits as VT.
    bool bitsEq(MVT VT) const {
      return getSizeInBits() == VT.getSizeInBits();
    }

    /// bitsGT - Return true if this has more bits than VT.
    bool bitsGT(MVT VT) const {
      return getSizeInBits() > VT.getSizeInBits();
    }

    /// bitsGE - Return true if this has no less bits than VT.
    bool bitsGE(MVT VT) const {
      return getSizeInBits() >= VT.getSizeInBits();
    }

    /// bitsLT - Return true if this has less bits than VT.
    bool bitsLT(MVT VT) const {
      return getSizeInBits() < VT.getSizeInBits();
    }

    /// bitsLE - Return true if this has no more bits than VT.
    bool bitsLE(MVT VT) const {
      return getSizeInBits() <= VT.getSizeInBits();
    }


    /// getSimpleVT - Return the SimpleValueType held in the specified
    /// simple MVT.
    SimpleValueType getSimpleVT() const {
      assert(isSimple() && "Expected a SimpleValueType!");
      return SimpleValueType(V);
    }

    /// getVectorElementType - Given a vector type, return the type of
    /// each element.
    MVT getVectorElementType() const {
      assert(isVector() && "Invalid vector type!");
      switch (V) {
      default:
        return getExtendedVectorElementType();
      case v2i8 :
      case v4i8 :
      case v8i8 :
      case v16i8:
      case v24i8:
      case v32i8:
      case v48i8:
      case v64i8: return i8;
      case v2i16:
      case v4i16:
      case v8i16:
      case v12i16:
      case v16i16:
      case v24i16:
      case v32i16: return i16;
      case v2i32:
      case v3i32:
      case v4i32:
      case v6i32:
      case v8i32:
      case v12i32:
      case v16i32: return i32;
      case v1i64:
      case v2i64:
      case v3i64:
      case v4i64:
      case v6i64:
      case v8i64: return i64;
      case v2f32:
      case v3f32:
      case v4f32:
      case v6f32:
      case v8f32:
      case v12f32:
      case v16f32: return f32;
      case v2f64:
      case v4f64: return f64;
      }
    }

    /// getVectorNumElements - Given a vector type, return the number of
    /// elements it contains.
    unsigned getVectorNumElements() const {
      assert(isVector() && "Invalid vector type!");
      switch (V) {
      default:
        return getExtendedVectorNumElements();
      case v64i8: return 64;
      case v48i8: return 48;
      case v32i8:
      case v32i16: return 32;
      case v24i8:
      case v24i16: return 24;
      case v16i8:
      case v16i16:
      case v16i32:
      case v16f32: return 16;
      case v12i16:
      case v12i32:
      case v12f32: return 12;
      case v8i8:
      case v8i16:
      case v8i32:
      case v8i64:
      case v8f32: return 8;
      case v6i32:
      case v6i64:
      case v6f32: return 6;
      case v4i8:
      case v4i16:
      case v4i32:
      case v4i64:
      case v4f32:
      case v4f64: return 4;
      case v3i32:
      case v3i64:
      case v3f32: return 3;
      case v2i8:
      case v2i16:
      case v2i32:
      case v2i64:
      case v2f32:
      case v2f64: return 2;
      case v1i64: return 1;
      }
    }

    /// getSizeInBits - Return the size of the specified value type in bits.
    unsigned getSizeInBits() const {
      switch (V) {
      case iPTR:
        assert(0 && "Value type size is target-dependent. Ask TLI.");
      case iPTRAny:
      case iAny:
      case fAny:
        assert(0 && "Value type is overloaded.");
      default:
        return getExtendedSizeInBits();
      case i1  :  return 1;
      case i8  :  return 8;
      case i16 :
      case v2i8:  return 16;
      case f32 :
      case i32 :
      case v4i8:
      case v2i16: return 32;
      case f64 :
      case i64 :
      case v8i8:
      case v4i16:
      case v2i32:
      case v1i64:
      case v2f32: return 64;
      case f80 :  return 80;
      case v3i32:
      case v3f32: return 96;
      case f128:
      case ppcf128:
      case i128:
      case v16i8:
      case v8i16:
      case v4i32:
      case v2i64:
      case v4f32:
      case v2f64: return 128;
      case v24i8:
      case v12i16:
      case v6i32:
      case v3i64:
      case v6f32: return 192;
      case v32i8:
      case v16i16:
      case v8i32:
      case v4i64:
      case v8f32:
      case v4f64: return 256;
      case v48i8:
      case v24i16:
      case v12i32:
      case v6i64:
      case v12f32: return 384;
      case v64i8:
      case v32i16:
      case v16i32:
      case v8i64:
      case v16f32: return 512;
      }
    }

    /// getStoreSizeInBits - Return the number of bits overwritten by a store
    /// of the specified value type.
    unsigned getStoreSizeInBits() const {
      return (getSizeInBits() + 7)/8*8;
    }

    /// getRoundIntegerType - Rounds the bit-width of the given integer MVT up
    /// to the nearest power of two (and at least to eight), and returns the
    /// integer MVT with that number of bits.
    MVT getRoundIntegerType() const {
      assert(isInteger() && !isVector() && "Invalid integer type!");
      unsigned BitWidth = getSizeInBits();
      if (BitWidth <= 8)
        return i8;
      else
        return getIntegerVT(1 << Log2_32_Ceil(BitWidth));
    }

    /// isPow2VectorType - Retuns true if the given vector is a power of 2.
    bool isPow2VectorType() const {
      unsigned NElts = getVectorNumElements();
      return !(NElts & (NElts - 1));
    }

    /// getPow2VectorType - Widens the length of the given vector MVT up to
    /// the nearest power of 2 and returns that type.
    MVT getPow2VectorType() const {
      if (!isPow2VectorType()) {
        unsigned NElts = getVectorNumElements();
        unsigned Pow2NElts = 1 <<  Log2_32_Ceil(NElts);
        return MVT::getVectorVT(getVectorElementType(), Pow2NElts);
      }
      else {
        return *this;
      }
    }

    /// getMVTString - This function returns value type as a string,
    /// e.g. "i32".
    std::string getMVTString() const;

    /// getTypeForMVT - This method returns an LLVM type corresponding to the
    /// specified MVT.  For integer types, this returns an unsigned type.  Note
    /// that this will abort for types that cannot be represented.
    const Type *getTypeForMVT(LLVMContext &Context) const;

    /// getMVT - Return the value type corresponding to the specified type.
    /// This returns all pointers as iPTR.  If HandleUnknown is true, unknown
    /// types are returned as Other, otherwise they are invalid.
    static MVT getMVT(const Type *Ty, bool HandleUnknown = false);

    /// getRawBits - Represent the type as a bunch of bits.
    uintptr_t getRawBits() const { return V; }

    /// compareRawBits - A meaningless but well-behaved order, useful for
    /// constructing containers.
    struct compareRawBits {
      bool operator()(MVT L, MVT R) const {
        return L.getRawBits() < R.getRawBits();
      }
    };

  private:
    // Methods for handling the Extended-type case in functions above.
    // These are all out-of-line to prevent users of this header file
    // from having a dependency on Type.h.
    static MVT getExtendedIntegerVT(unsigned BitWidth);
    static MVT getExtendedVectorVT(MVT VT, unsigned NumElements);
    bool isExtendedFloatingPoint() const;
    bool isExtendedInteger() const;
    bool isExtendedVector() const;
    bool isExtended64BitVector() const;
    bool isExtended128BitVector() const;
    bool isExtended256BitVector() const;
    MVT getExtendedVectorElementType() const;
    unsigned getExtendedVectorNumElements() const;
    unsigned getExtendedSizeInBits() const;
  };

} // End llvm namespace

#endif
