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

  struct MVT { // MVT = Machine Value Type
  public:

    enum SimpleValueType {
      // If you change this numbering, you must change the values in
      // ValueTypes.td well!
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

    /// MVT - This type holds low-level value types. Valid values include any of
    /// the values in the SimpleValueType enum, or any value returned from one
    /// of the MVT methods.  Any value type equal to one of the SimpleValueType
    /// enum values is a "simple" value type.  All others are "extended".
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

  private:

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

    uint32_t V;

  public:

    MVT() {}
    MVT(SimpleValueType S) { V = S; }

    inline bool operator== (const MVT VT) const { return V == VT.V; }
    inline bool operator!= (const MVT VT) const { return V != VT.V; }

    /// getIntegerVT - Returns the MVT that represents an integer with the given
    /// number of bits.
    static inline MVT getIntegerVT(unsigned BitWidth) {
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
      MVT VT;
      VT.V = iAny | (((BitWidth - 1) << SimpleTypeBits) & PrecisionMask);
      assert(VT.getSizeInBits() == BitWidth && "Bad bit width!");
      return VT;
    }

    /// getVectorVT - Returns the MVT that represents a vector NumElements in
    /// length, where each element is of type VT.
    static inline MVT getVectorVT(MVT VT, unsigned NumElements) {
      switch (VT.V) {
      default:
        break;
      case i8:
        if (NumElements == 8)  return v8i8;
        if (NumElements == 16) return v16i8;
        break;
      case i16:
        if (NumElements == 4)  return v4i16;
        if (NumElements == 8)  return v8i16;
        break;
      case i32:
        if (NumElements == 2)  return v2i32;
        if (NumElements == 3)  return v3i32;
        if (NumElements == 4)  return v4i32;
        break;
      case i64:
        if (NumElements == 1)  return v1i64;
        if (NumElements == 2)  return v2i64;
        break;
      case f32:
        if (NumElements == 2)  return v2f32;
        if (NumElements == 3)  return v3f32;
        if (NumElements == 4)  return v4f32;
        break;
      case f64:
        if (NumElements == 2)  return v2f64;
        break;
      }
      MVT Result;
      Result.V = VT.V | ((NumElements + 1) << (32 - VectorBits));
      assert(Result.getVectorElementType() == VT &&
             "Bad vector element type!");
      assert(Result.getVectorNumElements() == NumElements &&
             "Bad vector length!");
      return Result;
    }

    /// getIntVectorWithNumElements - Return any integer vector type that has
    /// the specified number of elements.
    static inline MVT getIntVectorWithNumElements(unsigned NumElts) {
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
    inline bool isSimple() const {
      return V <= SimpleTypeMask;
    }

    /// isExtended - Test if the given MVT is extended (as opposed to
    /// being simple).
    inline bool isExtended() const {
      return !isSimple();
    }

    /// isFloatingPoint - Return true if this is a FP, or a vector FP type.
    inline bool isFloatingPoint() const {
      uint32_t SVT = V & SimpleTypeMask;
      return (SVT >= f32 && SVT <= ppcf128) || (SVT >= v2f32 && SVT <= v2f64);
    }

    /// isInteger - Return true if this is an integer, or a vector integer type.
    inline bool isInteger() const {
      uint32_t SVT = V & SimpleTypeMask;
      return (SVT >= FIRST_INTEGER_VALUETYPE && SVT <= LAST_INTEGER_VALUETYPE) ||
        (SVT >= v8i8 && SVT <= v2i64) || (SVT == iAny && (V & PrecisionMask));
    }

    /// isVector - Return true if this is a vector value type.
    inline bool isVector() const {
      return (V >= FIRST_VECTOR_VALUETYPE && V <= LAST_VECTOR_VALUETYPE) ||
             (V & VectorMask);
    }

    /// is64BitVector - Return true if this is a 64-bit vector type.
    inline bool is64BitVector() const {
      return (V==v8i8 || V==v4i16 || V==v2i32 || V==v1i64 || V==v2f32 ||
              (isExtended() && isVector() && getSizeInBits()==64));
    }

    /// is128BitVector - Return true if this is a 128-bit vector type.
    inline bool is128BitVector() const {
      return (V==v16i8 || V==v8i16 || V==v4i32 || V==v2i64 ||
              V==v4f32 || V==v2f64 ||
              (isExtended() && isVector() && getSizeInBits()==128));
    }


    /// bitsGT - Return true if this has more bits than VT.
    inline bool bitsGT(MVT VT) const {
      return getSizeInBits() > VT.getSizeInBits();
    }

    /// bitsGE - Return true if this has no less bits than VT.
    inline bool bitsGE(MVT VT) const {
      return getSizeInBits() >= VT.getSizeInBits();
    }

    /// bitsLT - Return true if this has less bits than VT.
    inline bool bitsLT(MVT VT) const {
      return getSizeInBits() < VT.getSizeInBits();
    }

    /// bitsLE - Return true if this has no more bits than VT.
    inline bool bitsLE(MVT VT) const {
      return getSizeInBits() <= VT.getSizeInBits();
    }


    /// getSimpleVT - Return the SimpleValueType held in the specified
    /// simple MVT.
    inline SimpleValueType getSimpleVT() const {
      assert(isSimple() && "Expected a SimpleValueType!");
      return (SimpleValueType)V;
    }

    /// getVectorElementType - Given a vector type, return the type of
    /// each element.
    inline MVT getVectorElementType() const {
      assert(isVector() && "Invalid vector type!");
      switch (V) {
      default: {
        assert(isExtended() && "Unknown simple vector type!");
        MVT VT;
        VT.V = V & ElementMask;
        return VT;
      }
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

    /// getVectorNumElements - Given a vector type, return the number of
    /// elements it contains.
    inline unsigned getVectorNumElements() const {
      assert(isVector() && "Invalid vector type!");
      switch (V) {
      default:
        assert(isExtended() && "Unknown simple vector type!");
        return ((V & VectorMask) >> (32 - VectorBits)) - 1;
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

    /// getSizeInBits - Return the size of the specified value type in bits.
    inline unsigned getSizeInBits() const {
      switch (V) {
      default:
        assert(isExtended() && "MVT has no known size!");
        if (isVector())
          return getVectorElementType().getSizeInBits()*getVectorNumElements();
        if (isInteger())
          return ((V & PrecisionMask) >> SimpleTypeBits) + 1;
        assert(false && "Unknown value type!");
        return 0;
      case i1  :  return 1;
      case i8  :  return 8;
      case i16 :  return 16;
      case f32 :
      case i32 :  return 32;
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
      }
    }

    /// getStoreSizeInBits - Return the number of bits overwritten by a store
    /// of the specified value type.
    inline unsigned getStoreSizeInBits() const {
      return (getSizeInBits() + 7)/8*8;
    }

    /// getRoundIntegerType - Rounds the bit-width of the given integer MVT up
    /// to the nearest power of two (and at least to eight), and returns the
    /// integer MVT with that number of bits.
    inline MVT getRoundIntegerType() const {
      assert(isInteger() && !isVector() && "Invalid integer type!");
      unsigned BitWidth = getSizeInBits();
      if (BitWidth <= 8)
        return i8;
      else
        return getIntegerVT(1 << Log2_32_Ceil(BitWidth));
    }

    /// getIntegerVTBitMask - Return an integer with 1's every place there are
    /// bits in the specified integer value type. FIXME: Should return an apint.
    inline uint64_t getIntegerVTBitMask() const {
      assert(isInteger() && !isVector() && "Only applies to int scalars!");
      return ~uint64_t(0UL) >> (64-getSizeInBits());
    }

    /// getIntegerVTSignBit - Return an integer with a 1 in the position of the
    /// sign bit for the specified integer value type. FIXME: Should return an
    /// apint.
    inline uint64_t getIntegerVTSignBit() const {
      assert(isInteger() && !isVector() && "Only applies to int scalars!");
      return uint64_t(1UL) << (getSizeInBits()-1);
    }

    /// getMVTString - This function returns value type as a string,
    /// e.g. "i32".
    std::string getMVTString() const;

    /// getTypeForMVT - This method returns an LLVM type corresponding to the
    /// specified MVT.  For integer types, this returns an unsigned type.  Note
    /// that this will abort for types that cannot be represented.
    const Type *getTypeForMVT() const;

    /// getMVT - Return the value type corresponding to the specified type.
    /// This returns all pointers as iPTR.  If HandleUnknown is true, unknown
    /// types are returned as Other, otherwise they are invalid.
    static MVT getMVT(const Type *Ty, bool HandleUnknown = false);

    /// getRawBits - Represent the type as a bunch of bits.
    uint32_t getRawBits() const { return V; }

    /// compareRawBits - A meaningless but well-behaved order, useful for
    /// constructing containers.
    struct compareRawBits {
      bool operator()(MVT L, MVT R) const {
        return L.getRawBits() < R.getRawBits();
      }
    };
  };

} // End llvm namespace

#endif
