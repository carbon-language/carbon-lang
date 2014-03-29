//===- ARM64AddressingModes.h - ARM64 Addressing Modes ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM64 addressing mode implementation stuff.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_ARM64_ARM64ADDRESSINGMODES_H
#define LLVM_TARGET_ARM64_ARM64ADDRESSINGMODES_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

namespace llvm {

/// ARM64_AM - ARM64 Addressing Mode Stuff
namespace ARM64_AM {

//===----------------------------------------------------------------------===//
// Shifts
//

enum ShiftType {
  InvalidShift = -1,
  LSL = 0,
  LSR = 1,
  ASR = 2,
  ROR = 3,
  MSL = 4
};

/// getShiftName - Get the string encoding for the shift type.
static inline const char *getShiftName(ARM64_AM::ShiftType ST) {
  switch (ST) {
  default: assert(false && "unhandled shift type!");
  case ARM64_AM::LSL: return "lsl";
  case ARM64_AM::LSR: return "lsr";
  case ARM64_AM::ASR: return "asr";
  case ARM64_AM::ROR: return "ror";
  case ARM64_AM::MSL: return "msl";
  }
  return 0;
}

/// getShiftType - Extract the shift type.
static inline ARM64_AM::ShiftType getShiftType(unsigned Imm) {
  return ARM64_AM::ShiftType((Imm >> 6) & 0x7);
}

/// getShiftValue - Extract the shift value.
static inline unsigned getShiftValue(unsigned Imm) {
  return Imm & 0x3f;
}

/// getShifterImm - Encode the shift type and amount:
///   imm:     6-bit shift amount
///   shifter: 000 ==> lsl
///            001 ==> lsr
///            010 ==> asr
///            011 ==> ror
///            100 ==> msl
///   {8-6}  = shifter
///   {5-0}  = imm
static inline unsigned getShifterImm(ARM64_AM::ShiftType ST, unsigned Imm) {
  assert((Imm & 0x3f) == Imm && "Illegal shifted immedate value!");
  return (unsigned(ST) << 6) | (Imm & 0x3f);
}

//===----------------------------------------------------------------------===//
// Extends
//

enum ExtendType {
  InvalidExtend = -1,
  UXTB = 0,
  UXTH = 1,
  UXTW = 2,
  UXTX = 3,
  SXTB = 4,
  SXTH = 5,
  SXTW = 6,
  SXTX = 7
};

/// getExtendName - Get the string encoding for the extend type.
static inline const char *getExtendName(ARM64_AM::ExtendType ET) {
  switch (ET) {
  default: assert(false && "unhandled extend type!");
  case ARM64_AM::UXTB: return "uxtb";
  case ARM64_AM::UXTH: return "uxth";
  case ARM64_AM::UXTW: return "uxtw";
  case ARM64_AM::UXTX: return "uxtx";
  case ARM64_AM::SXTB: return "sxtb";
  case ARM64_AM::SXTH: return "sxth";
  case ARM64_AM::SXTW: return "sxtw";
  case ARM64_AM::SXTX: return "sxtx";
  }
  return 0;
}

/// getArithShiftValue - get the arithmetic shift value.
static inline unsigned getArithShiftValue(unsigned Imm) {
  return Imm & 0x7;
}

/// getExtendType - Extract the extend type for operands of arithmetic ops.
static inline ARM64_AM::ExtendType getArithExtendType(unsigned Imm) {
  return ARM64_AM::ExtendType((Imm >> 3) & 0x7);
}

/// getArithExtendImm - Encode the extend type and shift amount for an
///                     arithmetic instruction:
///   imm:     3-bit extend amount
///   shifter: 000 ==> uxtb
///            001 ==> uxth
///            010 ==> uxtw
///            011 ==> uxtx
///            100 ==> sxtb
///            101 ==> sxth
///            110 ==> sxtw
///            111 ==> sxtx
///   {5-3}  = shifter
///   {2-0}  = imm3
static inline unsigned getArithExtendImm(ARM64_AM::ExtendType ET,
                                         unsigned Imm) {
  assert((Imm & 0x7) == Imm && "Illegal shifted immedate value!");
  return (unsigned(ET) << 3) | (Imm & 0x7);
}

/// getMemDoShift - Extract the "do shift" flag value for load/store
/// instructions.
static inline bool getMemDoShift(unsigned Imm) {
  return (Imm & 0x1) != 0;
}

/// getExtendType - Extract the extend type for the offset operand of
/// loads/stores.
static inline ARM64_AM::ExtendType getMemExtendType(unsigned Imm) {
  return ARM64_AM::ExtendType((Imm >> 1) & 0x7);
}

/// getExtendImm - Encode the extend type and amount for a load/store inst:
///   imm:     3-bit extend amount
///   shifter: 000 ==> uxtb
///            001 ==> uxth
///            010 ==> uxtw
///            011 ==> uxtx
///            100 ==> sxtb
///            101 ==> sxth
///            110 ==> sxtw
///            111 ==> sxtx
///   {3-1}  = shifter
///   {0}  = imm3
static inline unsigned getMemExtendImm(ARM64_AM::ExtendType ET, bool Imm) {
  assert((Imm & 0x7) == Imm && "Illegal shifted immedate value!");
  return (unsigned(ET) << 1) | (Imm & 0x7);
}

//===----------------------------------------------------------------------===//
// Prefetch
//

/// Pre-fetch operator names.
/// The enum values match the encoding values:
///   prfop<4:3> 00=preload data, 10=prepare for store
///   prfop<2:1> 00=target L1 cache, 01=target L2 cache, 10=target L3 cache,
///   prfop<0> 0=non-streaming (temporal), 1=streaming (non-temporal)
enum PrefetchOp {
  InvalidPrefetchOp = -1,
  PLDL1KEEP = 0x00,
  PLDL1STRM = 0x01,
  PLDL2KEEP = 0x02,
  PLDL2STRM = 0x03,
  PLDL3KEEP = 0x04,
  PLDL3STRM = 0x05,
  PSTL1KEEP = 0x10,
  PSTL1STRM = 0x11,
  PSTL2KEEP = 0x12,
  PSTL2STRM = 0x13,
  PSTL3KEEP = 0x14,
  PSTL3STRM = 0x15
};

/// isNamedPrefetchOp - Check if the prefetch-op 5-bit value has a name.
static inline bool isNamedPrefetchOp(unsigned prfop) {
  switch (prfop) {
  default: return false;
  case ARM64_AM::PLDL1KEEP: case ARM64_AM::PLDL1STRM: case ARM64_AM::PLDL2KEEP:
  case ARM64_AM::PLDL2STRM: case ARM64_AM::PLDL3KEEP: case ARM64_AM::PLDL3STRM:
  case ARM64_AM::PSTL1KEEP: case ARM64_AM::PSTL1STRM: case ARM64_AM::PSTL2KEEP:
  case ARM64_AM::PSTL2STRM: case ARM64_AM::PSTL3KEEP: case ARM64_AM::PSTL3STRM:
    return true;
  }
}


/// getPrefetchOpName - Get the string encoding for the prefetch operator.
static inline const char *getPrefetchOpName(ARM64_AM::PrefetchOp prfop) {
  switch (prfop) {
  default: assert(false && "unhandled prefetch-op type!");
  case ARM64_AM::PLDL1KEEP: return "pldl1keep";
  case ARM64_AM::PLDL1STRM: return "pldl1strm";
  case ARM64_AM::PLDL2KEEP: return "pldl2keep";
  case ARM64_AM::PLDL2STRM: return "pldl2strm";
  case ARM64_AM::PLDL3KEEP: return "pldl3keep";
  case ARM64_AM::PLDL3STRM: return "pldl3strm";
  case ARM64_AM::PSTL1KEEP: return "pstl1keep";
  case ARM64_AM::PSTL1STRM: return "pstl1strm";
  case ARM64_AM::PSTL2KEEP: return "pstl2keep";
  case ARM64_AM::PSTL2STRM: return "pstl2strm";
  case ARM64_AM::PSTL3KEEP: return "pstl3keep";
  case ARM64_AM::PSTL3STRM: return "pstl3strm";
  }
  return 0;
}

static inline uint64_t ror(uint64_t elt, unsigned size) {
  return ((elt & 1) << (size-1)) | (elt >> 1);
}

/// processLogicalImmediate - Determine if an immediate value can be encoded
/// as the immediate operand of a logical instruction for the given register
/// size.  If so, return true with "encoding" set to the encoded value in
/// the form N:immr:imms.
static inline bool processLogicalImmediate(uint64_t imm, unsigned regSize,
                                           uint64_t &encoding) {
  if (imm == 0ULL || imm == ~0ULL ||
      (regSize != 64 && (imm >> regSize != 0 || imm == ~0U)))
    return false;

  unsigned size = 2;
  uint64_t eltVal = imm;

  // First, determine the element size.
  while (size < regSize) {
    unsigned numElts = regSize / size;
    unsigned mask = (1ULL << size) - 1;
    uint64_t lowestEltVal = imm & mask;

    bool allMatched = true;
    for (unsigned i = 1; i < numElts; ++i) {
     uint64_t currEltVal = (imm >> (i*size)) & mask;
      if (currEltVal != lowestEltVal) {
        allMatched = false;
        break;
      }
    }

    if (allMatched) {
      eltVal = lowestEltVal;
      break;
    }

    size *= 2;
  }

  // Second, determine the rotation to make the element be: 0^m 1^n.
  for (unsigned i = 0; i < size; ++i) {
    eltVal = ror(eltVal, size);
    uint32_t clz = countLeadingZeros(eltVal) - (64 - size);
    uint32_t cto = CountTrailingOnes_64(eltVal);

    if (clz + cto == size) {
      // Encode in immr the number of RORs it would take to get *from* this
      // element value to our target value, where i+1 is the number of RORs
      // to go the opposite direction.
      unsigned immr = size - (i + 1);

      // If size has a 1 in the n'th bit, create a value that has zeroes in
      // bits [0, n] and ones above that.
      uint64_t nimms = ~(size-1) << 1;

      // Or the CTO value into the low bits, which must be below the Nth bit
      // bit mentioned above.
      nimms |= (cto-1);

      // Extract the seventh bit and toggle it to create the N field.
      unsigned N = ((nimms >> 6) & 1) ^ 1;

      encoding = (N << 12) | (immr << 6) | (nimms & 0x3f);
      return true;
    }
  }

  return false;
}

/// isLogicalImmediate - Return true if the immediate is valid for a logical
/// immediate instruction of the given register size. Return false otherwise.
static inline bool isLogicalImmediate(uint64_t imm, unsigned regSize) {
  uint64_t encoding;
  return processLogicalImmediate(imm, regSize, encoding);
}

/// encodeLogicalImmediate - Return the encoded immediate value for a logical
/// immediate instruction of the given register size.
static inline uint64_t encodeLogicalImmediate(uint64_t imm, unsigned regSize) {
  uint64_t encoding = 0;
  bool res = processLogicalImmediate(imm, regSize, encoding);
  assert(res && "invalid logical immediate");
  (void)res;
  return encoding;
}

/// decodeLogicalImmediate - Decode a logical immediate value in the form
/// "N:immr:imms" (where the immr and imms fields are each 6 bits) into the
/// integer value it represents with regSize bits.
static inline uint64_t decodeLogicalImmediate(uint64_t val, unsigned regSize) {
  // Extract the N, imms, and immr fields.
  unsigned N = (val >> 12) & 1;
  unsigned immr = (val >> 6) & 0x3f;
  unsigned imms = val & 0x3f;

  assert((regSize == 64 || N == 0) && "undefined logical immediate encoding");
  int len = 31 - countLeadingZeros((N << 6) | (~imms & 0x3f));
  assert(len >= 0 && "undefined logical immediate encoding");
  unsigned size = (1 << len);
  unsigned R = immr & (size - 1);
  unsigned S = imms & (size - 1);
  assert(S != size - 1 && "undefined logical immediate encoding");
  uint64_t pattern = (1ULL << (S + 1)) - 1;
  for (unsigned i = 0; i < R; ++i)
    pattern = ror(pattern, size);

  // Replicate the pattern to fill the regSize.
  while (size != regSize) {
    pattern |= (pattern << size);
    size *= 2;
  }
  return pattern;
}

/// isValidDecodeLogicalImmediate - Check to see if the logical immediate value
/// in the form "N:immr:imms" (where the immr and imms fields are each 6 bits)
/// is a valid encoding for an integer value with regSize bits.
static inline bool isValidDecodeLogicalImmediate(uint64_t val,
                                                 unsigned regSize) {
  // Extract the N and imms fields needed for checking.
  unsigned N = (val >> 12) & 1;
  unsigned imms = val & 0x3f;

  if (regSize == 32 && N != 0) // undefined logical immediate encoding
    return false;
  int len = 31 - countLeadingZeros((N << 6) | (~imms & 0x3f));
  if (len < 0) // undefined logical immediate encoding
    return false;
  unsigned size = (1 << len);
  unsigned S = imms & (size - 1);
  if (S == size - 1) // undefined logical immediate encoding
    return false;

  return true;
}

//===----------------------------------------------------------------------===//
// Floating-point Immediates
//
static inline float getFPImmFloat(unsigned Imm) {
  // We expect an 8-bit binary encoding of a floating-point number here.
  union {
    uint32_t I;
    float F;
  } FPUnion;

  uint8_t Sign = (Imm >> 7) & 0x1;
  uint8_t Exp = (Imm >> 4) & 0x7;
  uint8_t Mantissa = Imm & 0xf;

  //   8-bit FP    iEEEE Float Encoding
  //   abcd efgh   aBbbbbbc defgh000 00000000 00000000
  //
  // where B = NOT(b);

  FPUnion.I = 0;
  FPUnion.I |= Sign << 31;
  FPUnion.I |= ((Exp & 0x4) != 0 ? 0 : 1) << 30;
  FPUnion.I |= ((Exp & 0x4) != 0 ? 0x1f : 0) << 25;
  FPUnion.I |= (Exp & 0x3) << 23;
  FPUnion.I |= Mantissa << 19;
  return FPUnion.F;
}

/// getFP32Imm - Return an 8-bit floating-point version of the 32-bit
/// floating-point value. If the value cannot be represented as an 8-bit
/// floating-point value, then return -1.
static inline int getFP32Imm(const APInt &Imm) {
  uint32_t Sign = Imm.lshr(31).getZExtValue() & 1;
  int32_t Exp = (Imm.lshr(23).getSExtValue() & 0xff) - 127;  // -126 to 127
  int64_t Mantissa = Imm.getZExtValue() & 0x7fffff;  // 23 bits

  // We can handle 4 bits of mantissa.
  // mantissa = (16+UInt(e:f:g:h))/16.
  if (Mantissa & 0x7ffff)
    return -1;
  Mantissa >>= 19;
  if ((Mantissa & 0xf) != Mantissa)
    return -1;

  // We can handle 3 bits of exponent: exp == UInt(NOT(b):c:d)-3
  if (Exp < -3 || Exp > 4)
    return -1;
  Exp = ((Exp+3) & 0x7) ^ 4;

  return ((int)Sign << 7) | (Exp << 4) | Mantissa;
}

static inline int getFP32Imm(const APFloat &FPImm) {
  return getFP32Imm(FPImm.bitcastToAPInt());
}

/// getFP64Imm - Return an 8-bit floating-point version of the 64-bit
/// floating-point value. If the value cannot be represented as an 8-bit
/// floating-point value, then return -1.
static inline int getFP64Imm(const APInt &Imm) {
  uint64_t Sign = Imm.lshr(63).getZExtValue() & 1;
  int64_t Exp = (Imm.lshr(52).getSExtValue() & 0x7ff) - 1023;   // -1022 to 1023
  uint64_t Mantissa = Imm.getZExtValue() & 0xfffffffffffffULL;

  // We can handle 4 bits of mantissa.
  // mantissa = (16+UInt(e:f:g:h))/16.
  if (Mantissa & 0xffffffffffffULL)
    return -1;
  Mantissa >>= 48;
  if ((Mantissa & 0xf) != Mantissa)
    return -1;

  // We can handle 3 bits of exponent: exp == UInt(NOT(b):c:d)-3
  if (Exp < -3 || Exp > 4)
    return -1;
  Exp = ((Exp+3) & 0x7) ^ 4;

  return ((int)Sign << 7) | (Exp << 4) | Mantissa;
}

static inline int getFP64Imm(const APFloat &FPImm) {
  return getFP64Imm(FPImm.bitcastToAPInt());
}

//===--------------------------------------------------------------------===//
// AdvSIMD Modified Immediates
//===--------------------------------------------------------------------===//

// 0x00 0x00 0x00 abcdefgh 0x00 0x00 0x00 abcdefgh
static inline bool isAdvSIMDModImmType1(uint64_t Imm) {
  return ((Imm >> 32) == (Imm & 0xffffffffULL)) &&
         ((Imm & 0xffffff00ffffff00ULL) == 0);
}

static inline uint8_t encodeAdvSIMDModImmType1(uint64_t Imm) {
  return (Imm & 0xffULL);
}

static inline uint64_t decodeAdvSIMDModImmType1(uint8_t Imm) {
  uint64_t EncVal = Imm;
  return (EncVal << 32) | EncVal;
}

// 0x00 0x00 abcdefgh 0x00 0x00 0x00 abcdefgh 0x00
static inline bool isAdvSIMDModImmType2(uint64_t Imm) {
  return ((Imm >> 32) == (Imm & 0xffffffffULL)) &&
         ((Imm & 0xffff00ffffff00ffULL) == 0);
}

static inline uint8_t encodeAdvSIMDModImmType2(uint64_t Imm) {
  return (Imm & 0xff00ULL) >> 8;
}

static inline uint64_t decodeAdvSIMDModImmType2(uint8_t Imm) {
  uint64_t EncVal = Imm;
  return (EncVal << 40) | (EncVal << 8);
}

// 0x00 abcdefgh 0x00 0x00 0x00 abcdefgh 0x00 0x00
static inline bool isAdvSIMDModImmType3(uint64_t Imm) {
  return ((Imm >> 32) == (Imm & 0xffffffffULL)) &&
         ((Imm & 0xff00ffffff00ffffULL) == 0);
}

static inline uint8_t encodeAdvSIMDModImmType3(uint64_t Imm) {
  return (Imm & 0xff0000ULL) >> 16;
}

static inline uint64_t decodeAdvSIMDModImmType3(uint8_t Imm) {
  uint64_t EncVal = Imm;
  return (EncVal << 48) | (EncVal << 16);
}

// abcdefgh 0x00 0x00 0x00 abcdefgh 0x00 0x00 0x00
static inline bool isAdvSIMDModImmType4(uint64_t Imm) {
  return ((Imm >> 32) == (Imm & 0xffffffffULL)) &&
         ((Imm & 0x00ffffff00ffffffULL) == 0);
}

static inline uint8_t encodeAdvSIMDModImmType4(uint64_t Imm) {
  return (Imm & 0xff000000ULL) >> 24;
}

static inline uint64_t decodeAdvSIMDModImmType4(uint8_t Imm) {
  uint64_t EncVal = Imm;
  return (EncVal << 56) | (EncVal << 24);
}

// 0x00 abcdefgh 0x00 abcdefgh 0x00 abcdefgh 0x00 abcdefgh
static inline bool isAdvSIMDModImmType5(uint64_t Imm) {
  return ((Imm >> 32) == (Imm & 0xffffffffULL)) &&
         (((Imm & 0x00ff0000ULL) >> 16) == (Imm & 0x000000ffULL)) &&
         ((Imm & 0xff00ff00ff00ff00ULL) == 0);
}

static inline uint8_t encodeAdvSIMDModImmType5(uint64_t Imm) {
  return (Imm & 0xffULL);
}

static inline uint64_t decodeAdvSIMDModImmType5(uint8_t Imm) {
  uint64_t EncVal = Imm;
  return (EncVal << 48) | (EncVal << 32) | (EncVal << 16) | EncVal;
}

// abcdefgh 0x00 abcdefgh 0x00 abcdefgh 0x00 abcdefgh 0x00
static inline bool isAdvSIMDModImmType6(uint64_t Imm) {
  return ((Imm >> 32) == (Imm & 0xffffffffULL)) &&
         (((Imm & 0xff000000ULL) >> 16) == (Imm & 0x0000ff00ULL)) &&
         ((Imm & 0x00ff00ff00ff00ffULL) == 0);
}

static inline uint8_t encodeAdvSIMDModImmType6(uint64_t Imm) {
  return (Imm & 0xff00ULL) >> 8;
}

static inline uint64_t decodeAdvSIMDModImmType6(uint8_t Imm) {
  uint64_t EncVal = Imm;
  return (EncVal << 56) | (EncVal << 40) | (EncVal << 24) | (EncVal << 8);
}

// 0x00 0x00 abcdefgh 0xFF 0x00 0x00 abcdefgh 0xFF
static inline bool isAdvSIMDModImmType7(uint64_t Imm) {
  return ((Imm >> 32) == (Imm & 0xffffffffULL)) &&
         ((Imm & 0xffff00ffffff00ffULL) == 0x000000ff000000ffULL);
}

static inline uint8_t encodeAdvSIMDModImmType7(uint64_t Imm) {
  return (Imm & 0xff00ULL) >> 8;
}

static inline uint64_t decodeAdvSIMDModImmType7(uint8_t Imm) {
  uint64_t EncVal = Imm;
  return (EncVal << 40) | (EncVal << 8) | 0x000000ff000000ffULL;
}

// 0x00 abcdefgh 0xFF 0xFF 0x00 abcdefgh 0xFF 0xFF
static inline bool isAdvSIMDModImmType8(uint64_t Imm) {
  return ((Imm >> 32) == (Imm & 0xffffffffULL)) &&
         ((Imm & 0xff00ffffff00ffffULL) == 0x0000ffff0000ffffULL);
}

static inline uint64_t decodeAdvSIMDModImmType8(uint8_t Imm) {
  uint64_t EncVal = Imm;
  return (EncVal << 48) | (EncVal << 16) | 0x0000ffff0000ffffULL;
}

static inline uint8_t encodeAdvSIMDModImmType8(uint64_t Imm) {
  return (Imm & 0x00ff0000ULL) >> 16;
}

// abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh
static inline bool isAdvSIMDModImmType9(uint64_t Imm) {
  return ((Imm >> 32) == (Imm & 0xffffffffULL)) &&
         ((Imm >> 48) == (Imm & 0x0000ffffULL)) &&
         ((Imm >> 56) == (Imm & 0x000000ffULL));
}

static inline uint8_t encodeAdvSIMDModImmType9(uint64_t Imm) {
  return (Imm & 0xffULL);
}

static inline uint64_t decodeAdvSIMDModImmType9(uint8_t Imm) {
  uint64_t EncVal = Imm;
  EncVal |= (EncVal << 8);
  EncVal |= (EncVal << 16);
  EncVal |= (EncVal << 32);
  return EncVal;
}

// aaaaaaaa bbbbbbbb cccccccc dddddddd eeeeeeee ffffffff gggggggg hhhhhhhh
// cmode: 1110, op: 1
static inline bool isAdvSIMDModImmType10(uint64_t Imm) {
  uint64_t ByteA = Imm & 0xff00000000000000ULL;
  uint64_t ByteB = Imm & 0x00ff000000000000ULL;
  uint64_t ByteC = Imm & 0x0000ff0000000000ULL;
  uint64_t ByteD = Imm & 0x000000ff00000000ULL;
  uint64_t ByteE = Imm & 0x00000000ff000000ULL;
  uint64_t ByteF = Imm & 0x0000000000ff0000ULL;
  uint64_t ByteG = Imm & 0x000000000000ff00ULL;
  uint64_t ByteH = Imm & 0x00000000000000ffULL;

  return (ByteA == 0ULL || ByteA == 0xff00000000000000ULL) &&
         (ByteB == 0ULL || ByteB == 0x00ff000000000000ULL) &&
         (ByteC == 0ULL || ByteC == 0x0000ff0000000000ULL) &&
         (ByteD == 0ULL || ByteD == 0x000000ff00000000ULL) &&
         (ByteE == 0ULL || ByteE == 0x00000000ff000000ULL) &&
         (ByteF == 0ULL || ByteF == 0x0000000000ff0000ULL) &&
         (ByteG == 0ULL || ByteG == 0x000000000000ff00ULL) &&
         (ByteH == 0ULL || ByteH == 0x00000000000000ffULL);
}

static inline uint8_t encodeAdvSIMDModImmType10(uint64_t Imm) {
  bool BitA = Imm & 0xff00000000000000ULL;
  bool BitB = Imm & 0x00ff000000000000ULL;
  bool BitC = Imm & 0x0000ff0000000000ULL;
  bool BitD = Imm & 0x000000ff00000000ULL;
  bool BitE = Imm & 0x00000000ff000000ULL;
  bool BitF = Imm & 0x0000000000ff0000ULL;
  bool BitG = Imm & 0x000000000000ff00ULL;
  bool BitH = Imm & 0x00000000000000ffULL;

  unsigned EncVal = BitA;
  EncVal <<= 1;
  EncVal |= BitB;
  EncVal <<= 1;
  EncVal |= BitC;
  EncVal <<= 1;
  EncVal |= BitD;
  EncVal <<= 1;
  EncVal |= BitE;
  EncVal <<= 1;
  EncVal |= BitF;
  EncVal <<= 1;
  EncVal |= BitG;
  EncVal <<= 1;
  EncVal |= BitH;
  return EncVal;
}

static inline uint64_t decodeAdvSIMDModImmType10(uint8_t Imm) {
  uint64_t EncVal = 0;
  if (Imm & 0x80) EncVal |= 0xff00000000000000ULL;
  if (Imm & 0x40) EncVal |= 0x00ff000000000000ULL;
  if (Imm & 0x20) EncVal |= 0x0000ff0000000000ULL;
  if (Imm & 0x10) EncVal |= 0x000000ff00000000ULL;
  if (Imm & 0x08) EncVal |= 0x00000000ff000000ULL;
  if (Imm & 0x04) EncVal |= 0x0000000000ff0000ULL;
  if (Imm & 0x02) EncVal |= 0x000000000000ff00ULL;
  if (Imm & 0x01) EncVal |= 0x00000000000000ffULL;
  return EncVal;
}

// aBbbbbbc defgh000 0x00 0x00 aBbbbbbc defgh000 0x00 0x00
static inline bool isAdvSIMDModImmType11(uint64_t Imm) {
  uint64_t BString = (Imm & 0x7E000000ULL) >> 25;
  return ((Imm >> 32) == (Imm & 0xffffffffULL)) &&
         (BString == 0x1f || BString == 0x20) &&
         ((Imm & 0x0007ffff0007ffffULL) == 0);
}

static inline uint8_t encodeAdvSIMDModImmType11(uint64_t Imm) {
  bool BitA = (Imm & 0x80000000ULL);
  bool BitB = (Imm & 0x20000000ULL);
  bool BitC = (Imm & 0x01000000ULL);
  bool BitD = (Imm & 0x00800000ULL);
  bool BitE = (Imm & 0x00400000ULL);
  bool BitF = (Imm & 0x00200000ULL);
  bool BitG = (Imm & 0x00100000ULL);
  bool BitH = (Imm & 0x00080000ULL);

  unsigned EncVal = BitA;
  EncVal <<= 1;
  EncVal |= BitB;
  EncVal <<= 1;
  EncVal |= BitC;
  EncVal <<= 1;
  EncVal |= BitD;
  EncVal <<= 1;
  EncVal |= BitE;
  EncVal <<= 1;
  EncVal |= BitF;
  EncVal <<= 1;
  EncVal |= BitG;
  EncVal <<= 1;
  EncVal |= BitH;
  return EncVal;
}

static inline uint64_t decodeAdvSIMDModImmType11(uint8_t Imm) {
  uint64_t EncVal = 0;
  if (Imm & 0x80) EncVal |= 0x80000000ULL;
  if (Imm & 0x40) EncVal |= 0x3e000000ULL;
  else            EncVal |= 0x40000000ULL;
  if (Imm & 0x20) EncVal |= 0x01000000ULL;
  if (Imm & 0x10) EncVal |= 0x00800000ULL;
  if (Imm & 0x08) EncVal |= 0x00400000ULL;
  if (Imm & 0x04) EncVal |= 0x00200000ULL;
  if (Imm & 0x02) EncVal |= 0x00100000ULL;
  if (Imm & 0x01) EncVal |= 0x00080000ULL;
  return (EncVal << 32) | EncVal;
}

// aBbbbbbb bbcdefgh 0x00 0x00 0x00 0x00 0x00 0x00
static inline bool isAdvSIMDModImmType12(uint64_t Imm) {
  uint64_t BString = (Imm & 0x7fc0000000000000ULL) >> 54;
  return ((BString == 0xff || BString == 0x100) &&
         ((Imm & 0x0000ffffffffffffULL) == 0));
}

static inline uint8_t encodeAdvSIMDModImmType12(uint64_t Imm) {
  bool BitA = (Imm & 0x8000000000000000ULL);
  bool BitB = (Imm & 0x0040000000000000ULL);
  bool BitC = (Imm & 0x0020000000000000ULL);
  bool BitD = (Imm & 0x0010000000000000ULL);
  bool BitE = (Imm & 0x0008000000000000ULL);
  bool BitF = (Imm & 0x0004000000000000ULL);
  bool BitG = (Imm & 0x0002000000000000ULL);
  bool BitH = (Imm & 0x0001000000000000ULL);

  unsigned EncVal = BitA;
  EncVal <<= 1;
  EncVal |= BitB;
  EncVal <<= 1;
  EncVal |= BitC;
  EncVal <<= 1;
  EncVal |= BitD;
  EncVal <<= 1;
  EncVal |= BitE;
  EncVal <<= 1;
  EncVal |= BitF;
  EncVal <<= 1;
  EncVal |= BitG;
  EncVal <<= 1;
  EncVal |= BitH;
  return EncVal;
}

static inline uint64_t decodeAdvSIMDModImmType12(uint8_t Imm) {
  uint64_t EncVal = 0;
  if (Imm & 0x80) EncVal |= 0x8000000000000000ULL;
  if (Imm & 0x40) EncVal |= 0x3fc0000000000000ULL;
  else            EncVal |= 0x4000000000000000ULL;
  if (Imm & 0x20) EncVal |= 0x0020000000000000ULL;
  if (Imm & 0x10) EncVal |= 0x0010000000000000ULL;
  if (Imm & 0x08) EncVal |= 0x0008000000000000ULL;
  if (Imm & 0x04) EncVal |= 0x0004000000000000ULL;
  if (Imm & 0x02) EncVal |= 0x0002000000000000ULL;
  if (Imm & 0x01) EncVal |= 0x0001000000000000ULL;
  return (EncVal << 32) | EncVal;
}

} // end namespace ARM64_AM

} // end namespace llvm

#endif
