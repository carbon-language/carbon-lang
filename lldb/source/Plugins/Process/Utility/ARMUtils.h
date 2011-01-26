//===-- lldb_ARMUtils.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_ARMUtils_h_
#define lldb_ARMUtils_h_

#include "InstructionUtils.h"

// Common utilities for the ARM/Thumb Instruction Set Architecture.

namespace lldb_private {

static inline uint32_t bits(const uint32_t val, const uint32_t msbit, const uint32_t lsbit)
{
    return Bits32(val, msbit, lsbit);
}

static inline uint32_t bit(const uint32_t val, const uint32_t msbit)
{
    return bits(val, msbit, msbit);
}

static uint32_t ror(uint32_t val, uint32_t N, uint32_t shift)
{
    uint32_t m = shift % N;
    return (val >> m) | (val << (N - m));
}

static inline uint32_t ARMExpandImm(uint32_t val)
{
    uint32_t imm = bits(val, 7, 0);      // immediate value
    uint32_t rot = 2 * bits(val, 11, 8); // rotate amount
    return (imm >> rot) | (imm << (32 - rot));
}

static inline uint32_t ThumbExpandImm(uint32_t val)
{
  uint32_t imm32 = 0;
  const uint32_t i = bit(val, 26);
  const uint32_t imm3 = bits(val, 14, 12);
  const uint32_t abcdefgh = bits(val, 7, 0);
  const uint32_t imm12 = i << 11 | imm3 << 8 | abcdefgh;

  if (bits(imm12, 10, 11) == 0)
  {
      switch (bits(imm12, 8, 9)) {
      case 0:
          imm32 = abcdefgh;
          break;

      case 1:
          imm32 = abcdefgh << 16 | abcdefgh;
          break;

      case 2:
          imm32 = abcdefgh << 24 | abcdefgh << 8;
          break;

      case 3:
          imm32 = abcdefgh  << 24 | abcdefgh << 16 | abcdefgh << 8 | abcdefgh; 
          break;
      }
  }
  else
  {
      const uint32_t unrotated_value = 0x80 | bits(imm12, 0, 6);
      imm32 = ror(unrotated_value, 32, bits(imm12, 7, 11));
  }
  return imm32;
}

// imm32 = ZeroExtend(i:imm3:imm8, 32)
static inline uint32_t ThumbImm12(uint32_t val)
{
  const uint32_t i = bit(val, 26);
  const uint32_t imm3 = bits(val, 14, 12);
  const uint32_t imm8 = bits(val, 7, 0);
  const uint32_t imm12 = i << 11 | imm3 << 8 | imm8;
  return imm12;
}

// imm32 = ZeroExtend(imm7:'00', 32)
static inline uint32_t ThumbImmScaled(uint32_t val)
{
  const uint32_t imm7 = bits(val, 6, 0);
  return imm7 * 4;
}

// This function performs the check for the register numbers 13 and 15 that are
// not permitted for many Thumb register specifiers.
static inline bool BadReg(uint32_t n) { return n == 13 || n == 15; }

}   // namespace lldb_private

#endif  // lldb_ARMUtils_h_
