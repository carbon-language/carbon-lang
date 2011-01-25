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

// Common utilities for the ARM/Thumb Instruction Set Architecture.

namespace lldb_private {

// ARM conditions
#define COND_EQ     0x0
#define COND_NE     0x1
#define COND_CS     0x2
#define COND_HS     0x2
#define COND_CC     0x3
#define COND_LO     0x3
#define COND_MI     0x4
#define COND_PL     0x5
#define COND_VS     0x6
#define COND_VC     0x7
#define COND_HI     0x8
#define COND_LS     0x9
#define COND_GE     0xA
#define COND_LT     0xB
#define COND_GT     0xC
#define COND_LE     0xD
#define COND_AL     0xE
#define COND_UNCOND 0xF

// Masks for CPSR
#define MASK_CPSR_MODE_MASK     (0x0000001fu)
#define MASK_CPSR_T         (1u << 5)
#define MASK_CPSR_F         (1u << 6)
#define MASK_CPSR_I         (1u << 7)
#define MASK_CPSR_A         (1u << 8)
#define MASK_CPSR_E         (1u << 9)
#define MASK_CPSR_GE_MASK   (0x000f0000u)
#define MASK_CPSR_J         (1u << 24)
#define MASK_CPSR_Q         (1u << 27)
#define MASK_CPSR_V         (1u << 28)
#define MASK_CPSR_C         (1u << 29)
#define MASK_CPSR_Z         (1u << 30)
#define MASK_CPSR_N         (1u << 31)


static inline uint32_t bits(const uint32_t val, const uint32_t msbit, const uint32_t lsbit)
{
    assert(msbit < 32 && lsbit <= msbit);
    return (val >> lsbit) & ((1u << (msbit - lsbit + 1)) - 1);
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

// This function performs the check for the register numbers 13 and 15 that are
// not permitted for many Thumb register specifiers.
static inline bool BadReg(uint32_t n) { return n == 13 || n == 15; }

// Returns an integer result equal to the number of bits of x that are ones.
static inline uint32_t BitCount(uint32_t x)
{
    // c accumulates the total bits set in x
    uint32_t c;
    for (c = 0; x; ++c)
    {
        x &= x - 1; // clear the least significant bit set
    }
    return c;
}

}   // namespace lldb_private

#endif  // lldb_ARMUtils_h_
