//===-- llvm/Support/MathExtras.h - Useful math functions -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains some functions that are useful for math stuff.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MATHEXTRAS_H
#define LLVM_SUPPORT_MATHEXTRAS_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

// NOTE: The following support functions use the _32/_64 extensions instead of  
// type overloading so that signed and unsigned integers can be used without
// ambiguity.


// Hi_32 - This function returns the high 32 bits of a 64 bit value.
inline unsigned Hi_32(uint64_t Value) {
  return (unsigned)(Value >> 32);
}

// Lo_32 - This function returns the low 32 bits of a 64 bit value.
inline unsigned Lo_32(uint64_t Value) {
  return (unsigned)Value;
}

// is?Type - these functions produce optimal testing for integer data types.
inline bool isInt8  (int Value)     { return (  signed char )Value == Value; }
inline bool isUInt8 (int Value)     { return (unsigned char )Value == Value; }
inline bool isInt16 (int Value)     { return (  signed short)Value == Value; }
inline bool isUInt16(int Value)     { return (unsigned short)Value == Value; }
inline bool isInt32 (int64_t Value) { return (  signed int  )Value == Value; }
inline bool isUInt32(int64_t Value) { return (unsigned int  )Value == Value; }

// isMask_32 - This function returns true if the argument is a sequence of ones  
// starting at the least significant bit with the remainder zero (32 bit version.)
// Ex. isMask_32(0x0000FFFFU) == true.
inline const bool isMask_32(unsigned Value) {
  return Value && ((Value + 1) & Value) == 0;
}

// isMask_64 - This function returns true if the argument is a sequence of ones  
// starting at the least significant bit with the remainder zero (64 bit version.)
inline const bool isMask_64(uint64_t Value) {
  return Value && ((Value + 1) & Value) == 0;
}

// isShiftedMask_32 - This function returns true if the argument contains a  
// sequence of ones with the remainder zero (32 bit version.)
// Ex. isShiftedMask_32(0x0000FF00U) == true.
inline const bool isShiftedMask_32(unsigned Value) {
  return isMask_32((Value - 1) | Value);
}

// isShiftedMask_64 - This function returns true if the argument contains a  
// sequence of ones with the remainder zero (64 bit version.)
inline const bool isShiftedMask_64(uint64_t Value) {
  return isMask_64((Value - 1) | Value);
}

// isPowerOf2_32 - This function returns true if the argument is a power of 
// two > 0. Ex. isPowerOf2_32(0x00100000U) == true (32 bit edition.)
inline bool isPowerOf2_32(unsigned Value) {
  return Value && !(Value & (Value - 1));
}

// isPowerOf2_64 - This function returns true if the argument is a power of two
// > 0 (64 bit edition.)
inline bool isPowerOf2_64(uint64_t Value) {
  return Value && !(Value & (Value - 1LL));
}

// CountLeadingZeros_32 - this function performs the platform optimal form of
// counting the number of zeros from the most significant bit to the first one
// bit.  Ex. CountLeadingZeros_32(0x00F000FF) == 8.
// Returns 32 if the word is zero.
// CountLeadingZeros_32 - this function performs the platform optimal form
// of counting the number of zeros from the most significant bit to the first
// one bit.  Ex. CountLeadingZeros_32(0x00F000FF) == 8.
// Returns 32 if the word is zero.
inline unsigned CountLeadingZeros_32(unsigned Value) {
  unsigned Count; // result
#if __GNUC__ >= 4
  // PowerPC is defined for __builtin_clz(0)
#if !defined(__ppc__) && !defined(__ppc64__)
  if (!Value) return 32;
#endif
  Count = __builtin_clz(Value);
#else
  if (!Value) return 32;
  Count = 0;
  // bisecton method for count leading zeros
  for (unsigned Shift = 32 >> 1; Shift; Shift >>= 1) {
    unsigned Tmp = Value >> Shift;
    if (Tmp) {
      Value = Tmp;
    } else {
      Count |= Shift;
    }
  }
#endif
  return Count;
}

// CountLeadingZeros_64 - This function performs the platform optimal form
// of counting the number of zeros from the most significant bit to the first 
// one bit (64 bit edition.)
// Returns 64 if the word is zero.
inline unsigned CountLeadingZeros_64(uint64_t Value) {
  unsigned Count; // result
#if __GNUC__ >= 4
  // PowerPC is defined for __builtin_clzll(0)
#if !defined(__ppc__) && !defined(__ppc64__)
  if (!Value) return 64;
#endif
  Count = __builtin_clzll(Value);
#else
  if (sizeof(long) == sizeof(int64_t)) {
    if (!Value) return 64;
    Count = 0;
    // bisecton method for count leading zeros
    for (uint64_t Shift = 64 >> 1; Shift; Shift >>= 1) {
      uint64_t Tmp = Value >> Shift;
      if (Tmp) {
        Value = Tmp;
      } else {
        Count |= Shift;
      }
    }
  } else {
    // get hi portion
    unsigned Hi = Hi_32(Value);

    // if some bits in hi portion
    if (Hi) {
        // leading zeros in hi portion plus all bits in lo portion
        Count = CountLeadingZeros_32(Hi);
    } else {
        // get lo portion
        unsigned Lo = Lo_32(Value);
        // same as 32 bit value
        Count = CountLeadingZeros_32(Lo)+32;
    }
  }
#endif
  return Count;
}

// Log2_32 - This function returns the floor log base 2 of the specified value, 
// -1 if the value is zero. (32 bit edition.)
// Ex. Log2_32(32) == 5, Log2_32(1) == 0, Log2_32(0) == -1
inline unsigned Log2_32(unsigned Value) {
    return 31 - CountLeadingZeros_32(Value);
  }

// Log2_64 - This function returns the floor log base 2 of the specified value, 
// -1 if the value is zero. (64 bit edition.)
inline unsigned Log2_64(unsigned Value) {
    return 63 - CountLeadingZeros_64(Value);
}

// Platform-independent wrappers for the C99 isnan() function.
int IsNAN (float f);
int IsNAN (double d);

// Platform-independent wrappers for the C99 isinf() function.
int IsInf (float f);
int IsInf (double d);

} // End llvm namespace

#endif
