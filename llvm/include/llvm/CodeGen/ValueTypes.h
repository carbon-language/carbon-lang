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
  struct EVT;

  class MVT { // MVT = Machine Value Type
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
      v32i8          =  18,   // 32 x i8
      v2i16          =  19,   //  2 x i16
      v4i16          =  20,   //  4 x i16
      v8i16          =  21,   //  8 x i16
      v16i16         =  22,   // 16 x i16
      v2i32          =  23,   //  2 x i32
      v4i32          =  24,   //  4 x i32
      v8i32          =  25,   //  8 x i32
      v1i64          =  26,   //  1 x i64
      v2i64          =  27,   //  2 x i64
      v4i64          =  28,   //  4 x i64

      v2f32          =  29,   //  2 x f32
      v4f32          =  30,   //  4 x f32
      v8f32          =  31,   //  8 x f32
      v2f64          =  32,   //  2 x f64
      v4f64          =  33,   //  4 x f64

      FIRST_VECTOR_VALUETYPE = v2i8,
      LAST_VECTOR_VALUETYPE  = v4f64,

      LAST_VALUETYPE =  34,   // This always remains at the end of the list.

      // This is the current maximum for LAST_VALUETYPE.
      // EVT::MAX_ALLOWED_VALUETYPE is used for asserts and to size bit vectors
      // This value must be a multiple of 32.
      MAX_ALLOWED_VALUETYPE = 64,

      // Metadata - This is MDNode or MDString.
      Metadata       = 250,

      // iPTRAny - An int value the size of the pointer of the current
      // target to any address space. This must only be used internal to
      // tblgen. Other than for overloading, we treat iPTRAny the same as iPTR.
      iPTRAny        = 251,

      // vAny - A vector with any length and element size. This is used
      // for intrinsics that have overloadings based on vector types.
      // This is only for tblgen's consumption!
      vAny           = 252,

      // fAny - Any floating-point or vector floating-point value. This is used
      // for intrinsics that have overloadings based on floating-point types.
      // This is only for tblgen's consumption!
      fAny           = 253,

      // iAny - An integer or vector integer value of any bit width. This is
      // used for intrinsics that have overloadings based on integer bit widths.
      // This is only for tblgen's consumption!
      iAny           = 254,

      // iPTR - An int value the size of the pointer of the current
      // target.  This should only be used internal to tblgen!
      iPTR           = 255,

      // LastSimpleValueType - The greatest valid SimpleValueType value.
      LastSimpleValueType = 255
    };

    SimpleValueType SimpleTy;

    MVT() : SimpleTy((SimpleValueType)(LastSimpleValueType+1)) {}
    MVT(SimpleValueType SVT) : SimpleTy(SVT) { }
    
    bool operator>(const MVT& S)  const { return SimpleTy >  S.SimpleTy; }
    bool operator<(const MVT& S)  const { return SimpleTy <  S.SimpleTy; }
    bool operator==(const MVT& S) const { return SimpleTy == S.SimpleTy; }
    bool operator>=(const MVT& S) const { return SimpleTy >= S.SimpleTy; }
    bool operator<=(const MVT& S) const { return SimpleTy <= S.SimpleTy; }
    
    /// isFloatingPoint - Return true if this is a FP, or a vector FP type.
    bool isFloatingPoint() const {
      return ((SimpleTy >= MVT::f32 && SimpleTy <= MVT::ppcf128) ||
        (SimpleTy >= MVT::v2f32 && SimpleTy <= MVT::v4f64));
    }

    /// isInteger - Return true if this is an integer, or a vector integer type.
    bool isInteger() const {
      return ((SimpleTy >= MVT::FIRST_INTEGER_VALUETYPE &&
               SimpleTy <= MVT::LAST_INTEGER_VALUETYPE) ||
               (SimpleTy >= MVT::v2i8 && SimpleTy <= MVT::v4i64));
    }

    /// isVector - Return true if this is a vector value type.
    bool isVector() const {
      return (SimpleTy >= MVT::FIRST_VECTOR_VALUETYPE &&
              SimpleTy <= MVT::LAST_VECTOR_VALUETYPE);
    }
    
    MVT getVectorElementType() const {
      switch (SimpleTy) {
      default:
        return (MVT::SimpleValueType)(MVT::LastSimpleValueType+1);
      case v2i8 :
      case v4i8 :
      case v8i8 :
      case v16i8:
      case v32i8: return i8;
      case v2i16:
      case v4i16:
      case v8i16:
      case v16i16: return i16;
      case v2i32:
      case v4i32:
      case v8i32: return i32;
      case v1i64:
      case v2i64:
      case v4i64: return i64;
      case v2f32:
      case v4f32:
      case v8f32: return f32;
      case v2f64:
      case v4f64: return f64;
      }
    }
    
    unsigned getVectorNumElements() const {
      switch (SimpleTy) {
      default:
        return ~0U;
      case v32i8: return 32;
      case v16i8:
      case v16i16: return 16;
      case v8i8 :
      case v8i16:
      case v8i32:
      case v8f32: return 8;
      case v4i8:
      case v4i16:
      case v4i32:
      case v4i64:
      case v4f32:
      case v4f64: return 4;
      case v2i8:
      case v2i16:
      case v2i32:
      case v2i64:
      case v2f32:
      case v2f64: return 2;
      case v1i64: return 1;
      }
    }
    
    unsigned getSizeInBits() const {
      switch (SimpleTy) {
      case iPTR:
        assert(0 && "Value type size is target-dependent. Ask TLI.");
      case iPTRAny:
      case iAny:
      case fAny:
        assert(0 && "Value type is overloaded.");
      default:
        assert(0 && "getSizeInBits called on extended MVT.");
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
      case f128:
      case ppcf128:
      case i128:
      case v16i8:
      case v8i16:
      case v4i32:
      case v2i64:
      case v4f32:
      case v2f64: return 128;
      case v32i8:
      case v16i16:
      case v8i32:
      case v4i64:
      case v8f32:
      case v4f64: return 256;
      }
    }
    
    static MVT getFloatingPointVT(unsigned BitWidth) {
      switch (BitWidth) {
      default:
        assert(false && "Bad bit width!");
      case 32:
        return MVT::f32;
      case 64:
        return MVT::f64;
      case 80:
        return MVT::f80;
      case 128:
        return MVT::f128;
      }
    }
    
    static MVT getIntegerVT(unsigned BitWidth) {
      switch (BitWidth) {
      default:
        return (MVT::SimpleValueType)(MVT::LastSimpleValueType+1);
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
    }
    
    static MVT getVectorVT(MVT VT, unsigned NumElements) {
      switch (VT.SimpleTy) {
      default:
        break;
      case MVT::i8:
        if (NumElements == 2)  return MVT::v2i8;
        if (NumElements == 4)  return MVT::v4i8;
        if (NumElements == 8)  return MVT::v8i8;
        if (NumElements == 16) return MVT::v16i8;
        if (NumElements == 32) return MVT::v32i8;
        break;
      case MVT::i16:
        if (NumElements == 2)  return MVT::v2i16;
        if (NumElements == 4)  return MVT::v4i16;
        if (NumElements == 8)  return MVT::v8i16;
        if (NumElements == 16) return MVT::v16i16;
        break;
      case MVT::i32:
        if (NumElements == 2)  return MVT::v2i32;
        if (NumElements == 4)  return MVT::v4i32;
        if (NumElements == 8)  return MVT::v8i32;
        break;
      case MVT::i64:
        if (NumElements == 1)  return MVT::v1i64;
        if (NumElements == 2)  return MVT::v2i64;
        if (NumElements == 4)  return MVT::v4i64;
        break;
      case MVT::f32:
        if (NumElements == 2)  return MVT::v2f32;
        if (NumElements == 4)  return MVT::v4f32;
        if (NumElements == 8)  return MVT::v8f32;
        break;
      case MVT::f64:
        if (NumElements == 2)  return MVT::v2f64;
        if (NumElements == 4)  return MVT::v4f64;
        break;
      }
      return (MVT::SimpleValueType)(MVT::LastSimpleValueType+1);
    }
    
    static MVT getIntVectorWithNumElements(unsigned NumElts) {
      switch (NumElts) {
      default: return (MVT::SimpleValueType)(MVT::LastSimpleValueType+1);
      case  1: return MVT::v1i64;
      case  2: return MVT::v2i32;
      case  4: return MVT::v4i16;
      case  8: return MVT::v8i8;
      case 16: return MVT::v16i8;
      }
    }
  };

  struct EVT { // EVT = Extended Value Type
  private:
    MVT V;
    const Type *LLVMTy;

  public:
    EVT() : V((MVT::SimpleValueType)(MVT::LastSimpleValueType+1)) {}
    EVT(MVT::SimpleValueType SVT) : V(SVT) { }
    EVT(MVT S) : V(S) {}

    bool operator==(const EVT VT) const {
      if (V.SimpleTy == VT.V.SimpleTy) {
        if (V.SimpleTy == MVT::LastSimpleValueType+1)
          return LLVMTy == VT.LLVMTy;
        return true;
      }
      return false;
    }
    bool operator!=(const EVT VT) const {
      if (V.SimpleTy == VT.V.SimpleTy) {
        if (V.SimpleTy == MVT::LastSimpleValueType+1)
          return LLVMTy != VT.LLVMTy;
        return false;
      }
      return true;
    }

    /// getFloatingPointVT - Returns the EVT that represents a floating point
    /// type with the given number of bits.  There are two floating point types
    /// with 128 bits - this returns f128 rather than ppcf128.
    static EVT getFloatingPointVT(unsigned BitWidth) {
      return MVT::getFloatingPointVT(BitWidth);
    }

    /// getIntegerVT - Returns the EVT that represents an integer with the given
    /// number of bits.
    static EVT getIntegerVT(unsigned BitWidth) {
      MVT M = MVT::getIntegerVT(BitWidth);
      if (M.SimpleTy == MVT::LastSimpleValueType+1)
        return getExtendedIntegerVT(BitWidth);
      else
        return M;
    }

    /// getVectorVT - Returns the EVT that represents a vector NumElements in
    /// length, where each element is of type VT.
    static EVT getVectorVT(EVT VT, unsigned NumElements) {
      MVT M = MVT::getVectorVT(VT.V, NumElements);
      if (M.SimpleTy == MVT::LastSimpleValueType+1)
        return getExtendedVectorVT(VT, NumElements);
      else
        return M;
    }

    /// getIntVectorWithNumElements - Return any integer vector type that has
    /// the specified number of elements.
    static EVT getIntVectorWithNumElements(unsigned NumElts) {
      MVT M = MVT::getIntVectorWithNumElements(NumElts);
      if (M.SimpleTy == MVT::LastSimpleValueType+1)
        return getVectorVT(EVT(MVT::i8), NumElts);
      else
        return M;
    }

    /// isSimple - Test if the given EVT is simple (as opposed to being
    /// extended).
    bool isSimple() const {
      return V.SimpleTy <= MVT::LastSimpleValueType;
    }

    /// isExtended - Test if the given EVT is extended (as opposed to
    /// being simple).
    bool isExtended() const {
      return !isSimple();
    }

    /// isFloatingPoint - Return true if this is a FP, or a vector FP type.
    bool isFloatingPoint() const {
      return isSimple() ?
             ((V >= MVT::f32 && V <= MVT::ppcf128) ||
              (V >= MVT::v2f32 && V <= MVT::v4f64)) : isExtendedFloatingPoint();
    }

    /// isInteger - Return true if this is an integer, or a vector integer type.
    bool isInteger() const {
      return isSimple() ?
             ((V >= MVT::FIRST_INTEGER_VALUETYPE &&
               V <= MVT::LAST_INTEGER_VALUETYPE) ||
              (V >= MVT::v2i8 && V <= MVT::v4i64)) : isExtendedInteger();
    }

    /// isVector - Return true if this is a vector value type.
    bool isVector() const {
      return isSimple() ?
             (V >= MVT::FIRST_VECTOR_VALUETYPE && V <= 
                   MVT::LAST_VECTOR_VALUETYPE) :
             isExtendedVector();
    }

    /// is64BitVector - Return true if this is a 64-bit vector type.
    bool is64BitVector() const {
      return isSimple() ?
             (V==MVT::v8i8 || V==MVT::v4i16 || V==MVT::v2i32 ||
              V==MVT::v1i64 || V==MVT::v2f32) :
             isExtended64BitVector();
    }

    /// is128BitVector - Return true if this is a 128-bit vector type.
    bool is128BitVector() const {
      return isSimple() ?
             (V==MVT::v16i8 || V==MVT::v8i16 || V==MVT::v4i32 ||
              V==MVT::v2i64 || V==MVT::v4f32 || V==MVT::v2f64) :
             isExtended128BitVector();
    }

    /// is256BitVector - Return true if this is a 256-bit vector type.
    inline bool is256BitVector() const {
      return isSimple() ?
             (V==MVT::v8f32 || V==MVT::v4f64 || V==MVT::v32i8 ||
              V==MVT::v16i16 || V==MVT::v8i32 || V==MVT::v4i64) : 
            isExtended256BitVector();
    }

    /// isOverloaded - Return true if this is an overloaded type for TableGen.
    bool isOverloaded() const {
      return (V==MVT::iAny || V==MVT::fAny || V==MVT::vAny || V==MVT::iPTRAny);
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
    bool bitsEq(EVT VT) const {
      return getSizeInBits() == VT.getSizeInBits();
    }

    /// bitsGT - Return true if this has more bits than VT.
    bool bitsGT(EVT VT) const {
      return getSizeInBits() > VT.getSizeInBits();
    }

    /// bitsGE - Return true if this has no less bits than VT.
    bool bitsGE(EVT VT) const {
      return getSizeInBits() >= VT.getSizeInBits();
    }

    /// bitsLT - Return true if this has less bits than VT.
    bool bitsLT(EVT VT) const {
      return getSizeInBits() < VT.getSizeInBits();
    }

    /// bitsLE - Return true if this has no more bits than VT.
    bool bitsLE(EVT VT) const {
      return getSizeInBits() <= VT.getSizeInBits();
    }


    /// getSimpleVT - Return the SimpleValueType held in the specified
    /// simple EVT.
    MVT getSimpleVT() const {
      assert(isSimple() && "Expected a SimpleValueType!");
      return V;
    }

    /// getVectorElementType - Given a vector type, return the type of
    /// each element.
    EVT getVectorElementType() const {
      assert(isVector() && "Invalid vector type!");
      if (isSimple())
        return V.getVectorElementType();
      else
        return getExtendedVectorElementType();
    }

    /// getVectorNumElements - Given a vector type, return the number of
    /// elements it contains.
    unsigned getVectorNumElements() const {
      assert(isVector() && "Invalid vector type!");
      if (isSimple())
        return V.getVectorNumElements();
      else
        return getExtendedVectorNumElements();
    }

    /// getSizeInBits - Return the size of the specified value type in bits.
    unsigned getSizeInBits() const {
      if (isSimple())
        return V.getSizeInBits();
      else
        return getExtendedSizeInBits();
    }

    /// getStoreSizeInBits - Return the number of bits overwritten by a store
    /// of the specified value type.
    unsigned getStoreSizeInBits() const {
      return (getSizeInBits() + 7)/8*8;
    }

    /// getRoundIntegerType - Rounds the bit-width of the given integer EVT up
    /// to the nearest power of two (and at least to eight), and returns the
    /// integer EVT with that number of bits.
    EVT getRoundIntegerType() const {
      assert(isInteger() && !isVector() && "Invalid integer type!");
      unsigned BitWidth = getSizeInBits();
      if (BitWidth <= 8)
        return EVT(MVT::i8);
      else
        return getIntegerVT(1 << Log2_32_Ceil(BitWidth));
    }

    /// isPow2VectorType - Retuns true if the given vector is a power of 2.
    bool isPow2VectorType() const {
      unsigned NElts = getVectorNumElements();
      return !(NElts & (NElts - 1));
    }

    /// getPow2VectorType - Widens the length of the given vector EVT up to
    /// the nearest power of 2 and returns that type.
    EVT getPow2VectorType() const {
      if (!isPow2VectorType()) {
        unsigned NElts = getVectorNumElements();
        unsigned Pow2NElts = 1 <<  Log2_32_Ceil(NElts);
        return EVT::getVectorVT(getVectorElementType(), Pow2NElts);
      }
      else {
        return *this;
      }
    }

    /// getEVTString - This function returns value type as a string,
    /// e.g. "i32".
    std::string getEVTString() const;

    /// getTypeForEVT - This method returns an LLVM type corresponding to the
    /// specified EVT.  For integer types, this returns an unsigned type.  Note
    /// that this will abort for types that cannot be represented.
    const Type *getTypeForEVT() const;

    /// getEVT - Return the value type corresponding to the specified type.
    /// This returns all pointers as iPTR.  If HandleUnknown is true, unknown
    /// types are returned as Other, otherwise they are invalid.
    static EVT getEVT(const Type *Ty, bool HandleUnknown = false);

    intptr_t getRawBits() {
      if (V.SimpleTy <= MVT::LastSimpleValueType)
        return V.SimpleTy;
      else
        return (intptr_t)(LLVMTy);
    }

    /// compareRawBits - A meaningless but well-behaved order, useful for
    /// constructing containers.
    struct compareRawBits {
      bool operator()(EVT L, EVT R) const {
        if (L.V.SimpleTy == R.V.SimpleTy)
          return L.LLVMTy < R.LLVMTy;
        else
          return L.V.SimpleTy < R.V.SimpleTy;
      }
    };

  private:
    // Methods for handling the Extended-type case in functions above.
    // These are all out-of-line to prevent users of this header file
    // from having a dependency on Type.h.
    static EVT getExtendedIntegerVT(unsigned BitWidth);
    static EVT getExtendedVectorVT(EVT VT, unsigned NumElements);
    bool isExtendedFloatingPoint() const;
    bool isExtendedInteger() const;
    bool isExtendedVector() const;
    bool isExtended64BitVector() const;
    bool isExtended128BitVector() const;
    bool isExtended256BitVector() const;
    EVT getExtendedVectorElementType() const;
    unsigned getExtendedVectorNumElements() const;
    unsigned getExtendedSizeInBits() const;
  };

} // End llvm namespace

#endif
