//===-- lldb_ARMDefines.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_ARMDefines_h_
#define lldb_ARMDefines_h_

// Common defintions for the ARM/Thumb Instruction Set Architecture.

namespace lldb_private {

// ARM shifter types
typedef enum
{
    SRType_LSL,
    SRType_LSR,
    SRType_ASR,
    SRType_ROR,
    SRType_RRX
} ARM_ShifterType;

// ARM conditions          // Meaning (integer)         Meaning (floating-point)      Condition flags
#define COND_EQ     0x0    // Equal                     Equal                         Z == 1
#define COND_NE     0x1    // Not equal                 Not equal, or unordered       Z == 0
#define COND_CS     0x2    // Carry set                 >, ==, or unordered           C == 1
#define COND_HS     0x2
#define COND_CC     0x3    // Carry clear               Less than                     C == 0
#define COND_LO     0x3
#define COND_MI     0x4    // Minus, negative           Less than                     N == 1
#define COND_PL     0x5    // Plus, positive or zero    >, ==, or unordered           N == 0
#define COND_VS     0x6    // Overflow                  Unordered                     V == 1
#define COND_VC     0x7    // No overflow               Not unordered                 V == 0
#define COND_HI     0x8    // Unsigned higher           Greater than, or unordered    C == 1 and Z == 0
#define COND_LS     0x9    // Unsigned lower or same    Less than or equal            C == 0 or Z == 1
#define COND_GE     0xA    // Greater than or equal     Greater than or equal         N == V
#define COND_LT     0xB    // Less than                 Less than, or unordered       N != V
#define COND_GT     0xC    // Greater than              Greater than                  Z == 0 and N == V
#define COND_LE     0xD    // Less than or equal        <, ==, or unordered           Z == 1 or N != V
#define COND_AL     0xE    // Always (unconditional)    Always (unconditional)        Any
#define COND_UNCOND 0xF

static inline const char *ARMCondCodeToString(uint32_t CC)
{
    switch (CC) {
    default: assert(0 && "Unknown condition code");
    case COND_EQ:  return "eq";
    case COND_NE:  return "ne";
    case COND_HS:  return "hs";
    case COND_LO:  return "lo";
    case COND_MI:  return "mi";
    case COND_PL:  return "pl";
    case COND_VS:  return "vs";
    case COND_VC:  return "vc";
    case COND_HI:  return "hi";
    case COND_LS:  return "ls";
    case COND_GE:  return "ge";
    case COND_LT:  return "lt";
    case COND_GT:  return "gt";
    case COND_LE:  return "le";
    case COND_AL:  return "al";
    }
}

// Bit positions for CPSR
#define CPSR_T  5
#define CPSR_F  6
#define CPSR_I  7
#define CPSR_A  8
#define CPSR_E  9
#define CPSR_J 24
#define CPSR_Q 27
#define CPSR_V 28
#define CPSR_C 29
#define CPSR_Z 30
#define CPSR_N 31

// Masks for CPSR
#define MASK_CPSR_MODE_MASK     (0x0000001fu)
#define MASK_CPSR_T         (1u << CPSR_T)
#define MASK_CPSR_F         (1u << CPSR_F)
#define MASK_CPSR_I         (1u << CPSR_I)
#define MASK_CPSR_A         (1u << CPSR_A)
#define MASK_CPSR_E         (1u << CPSR_E)
#define MASK_CPSR_GE_MASK   (0x000f0000u)
#define MASK_CPSR_J         (1u << CPSR_J)
#define MASK_CPSR_Q         (1u << CPSR_Q)
#define MASK_CPSR_V         (1u << CPSR_V)
#define MASK_CPSR_C         (1u << CPSR_C)
#define MASK_CPSR_Z         (1u << CPSR_Z)
#define MASK_CPSR_N         (1u << CPSR_N)

}   // namespace lldb_private

#endif  // lldb_ARMDefines_h_
