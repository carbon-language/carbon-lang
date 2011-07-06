//===-- ARMBaseInfo.h - Top level definitions for ARM -------- --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the ARM target useful for the compiler back-end and the MC libraries.
// As such, it deliberately does not include references to LLVM core
// code gen types, passes, etc..
//
//===----------------------------------------------------------------------===//

#ifndef ARMBASEINFO_H
#define ARMBASEINFO_H

#include "MCTargetDesc/ARMMCTargetDesc.h"
#include "llvm/Support/ErrorHandling.h"

// Note that the following auto-generated files only defined enum types, and
// so are safe to include here.

namespace llvm {

// Enums corresponding to ARM condition codes
namespace ARMCC {
  // The CondCodes constants map directly to the 4-bit encoding of the
  // condition field for predicated instructions.
  enum CondCodes { // Meaning (integer)          Meaning (floating-point)
    EQ,            // Equal                      Equal
    NE,            // Not equal                  Not equal, or unordered
    HS,            // Carry set                  >, ==, or unordered
    LO,            // Carry clear                Less than
    MI,            // Minus, negative            Less than
    PL,            // Plus, positive or zero     >, ==, or unordered
    VS,            // Overflow                   Unordered
    VC,            // No overflow                Not unordered
    HI,            // Unsigned higher            Greater than, or unordered
    LS,            // Unsigned lower or same     Less than or equal
    GE,            // Greater than or equal      Greater than or equal
    LT,            // Less than                  Less than, or unordered
    GT,            // Greater than               Greater than
    LE,            // Less than or equal         <, ==, or unordered
    AL             // Always (unconditional)     Always (unconditional)
  };

  inline static CondCodes getOppositeCondition(CondCodes CC) {
    switch (CC) {
    default: llvm_unreachable("Unknown condition code");
    case EQ: return NE;
    case NE: return EQ;
    case HS: return LO;
    case LO: return HS;
    case MI: return PL;
    case PL: return MI;
    case VS: return VC;
    case VC: return VS;
    case HI: return LS;
    case LS: return HI;
    case GE: return LT;
    case LT: return GE;
    case GT: return LE;
    case LE: return GT;
    }
  }
} // namespace ARMCC

inline static const char *ARMCondCodeToString(ARMCC::CondCodes CC) {
  switch (CC) {
  default: llvm_unreachable("Unknown condition code");
  case ARMCC::EQ:  return "eq";
  case ARMCC::NE:  return "ne";
  case ARMCC::HS:  return "hs";
  case ARMCC::LO:  return "lo";
  case ARMCC::MI:  return "mi";
  case ARMCC::PL:  return "pl";
  case ARMCC::VS:  return "vs";
  case ARMCC::VC:  return "vc";
  case ARMCC::HI:  return "hi";
  case ARMCC::LS:  return "ls";
  case ARMCC::GE:  return "ge";
  case ARMCC::LT:  return "lt";
  case ARMCC::GT:  return "gt";
  case ARMCC::LE:  return "le";
  case ARMCC::AL:  return "al";
  }
}

namespace ARM_PROC {
  enum IMod {
    IE = 2,
    ID = 3
  };

  enum IFlags {
    F = 1,
    I = 2,
    A = 4
  };

  inline static const char *IFlagsToString(unsigned val) {
    switch (val) {
    default: llvm_unreachable("Unknown iflags operand");
    case F: return "f";
    case I: return "i";
    case A: return "a";
    }
  }

  inline static const char *IModToString(unsigned val) {
    switch (val) {
    default: llvm_unreachable("Unknown imod operand");
    case IE: return "ie";
    case ID: return "id";
    }
  }
}

namespace ARM_MB {
  // The Memory Barrier Option constants map directly to the 4-bit encoding of
  // the option field for memory barrier operations.
  enum MemBOpt {
    SY    = 15,
    ST    = 14,
    ISH   = 11,
    ISHST = 10,
    NSH   = 7,
    NSHST = 6,
    OSH   = 3,
    OSHST = 2
  };

  inline static const char *MemBOptToString(unsigned val) {
    switch (val) {
    default: llvm_unreachable("Unknown memory operation");
    case SY:    return "sy";
    case ST:    return "st";
    case ISH:   return "ish";
    case ISHST: return "ishst";
    case NSH:   return "nsh";
    case NSHST: return "nshst";
    case OSH:   return "osh";
    case OSHST: return "oshst";
    }
  }
} // namespace ARM_MB

/// getARMRegisterNumbering - Given the enum value for some register, e.g.
/// ARM::LR, return the number that it corresponds to (e.g. 14).
inline static unsigned getARMRegisterNumbering(unsigned Reg) {
  using namespace ARM;
  switch (Reg) {
  default:
    llvm_unreachable("Unknown ARM register!");
  case R0:  case S0:  case D0:  case Q0:  return 0;
  case R1:  case S1:  case D1:  case Q1:  return 1;
  case R2:  case S2:  case D2:  case Q2:  return 2;
  case R3:  case S3:  case D3:  case Q3:  return 3;
  case R4:  case S4:  case D4:  case Q4:  return 4;
  case R5:  case S5:  case D5:  case Q5:  return 5;
  case R6:  case S6:  case D6:  case Q6:  return 6;
  case R7:  case S7:  case D7:  case Q7:  return 7;
  case R8:  case S8:  case D8:  case Q8:  return 8;
  case R9:  case S9:  case D9:  case Q9:  return 9;
  case R10: case S10: case D10: case Q10: return 10;
  case R11: case S11: case D11: case Q11: return 11;
  case R12: case S12: case D12: case Q12: return 12;
  case SP:  case S13: case D13: case Q13: return 13;
  case LR:  case S14: case D14: case Q14: return 14;
  case PC:  case S15: case D15: case Q15: return 15;

  case S16: case D16: return 16;
  case S17: case D17: return 17;
  case S18: case D18: return 18;
  case S19: case D19: return 19;
  case S20: case D20: return 20;
  case S21: case D21: return 21;
  case S22: case D22: return 22;
  case S23: case D23: return 23;
  case S24: case D24: return 24;
  case S25: case D25: return 25;
  case S26: case D26: return 26;
  case S27: case D27: return 27;
  case S28: case D28: return 28;
  case S29: case D29: return 29;
  case S30: case D30: return 30;
  case S31: case D31: return 31;
  }
}

namespace ARMII {

  /// ARM Index Modes
  enum IndexMode {
    IndexModeNone  = 0,
    IndexModePre   = 1,
    IndexModePost  = 2,
    IndexModeUpd   = 3
  };

  /// ARM Addressing Modes
  enum AddrMode {
    AddrModeNone    = 0,
    AddrMode1       = 1,
    AddrMode2       = 2,
    AddrMode3       = 3,
    AddrMode4       = 4,
    AddrMode5       = 5,
    AddrMode6       = 6,
    AddrModeT1_1    = 7,
    AddrModeT1_2    = 8,
    AddrModeT1_4    = 9,
    AddrModeT1_s    = 10, // i8 * 4 for pc and sp relative data
    AddrModeT2_i12  = 11,
    AddrModeT2_i8   = 12,
    AddrModeT2_so   = 13,
    AddrModeT2_pc   = 14, // +/- i12 for pc relative data
    AddrModeT2_i8s4 = 15, // i8 * 4
    AddrMode_i12    = 16
  };

  inline static const char *AddrModeToString(AddrMode addrmode) {
    switch (addrmode) {
    default: llvm_unreachable("Unknown memory operation");
    case AddrModeNone:    return "AddrModeNone";
    case AddrMode1:       return "AddrMode1";
    case AddrMode2:       return "AddrMode2";
    case AddrMode3:       return "AddrMode3";
    case AddrMode4:       return "AddrMode4";
    case AddrMode5:       return "AddrMode5";
    case AddrMode6:       return "AddrMode6";
    case AddrModeT1_1:    return "AddrModeT1_1";
    case AddrModeT1_2:    return "AddrModeT1_2";
    case AddrModeT1_4:    return "AddrModeT1_4";
    case AddrModeT1_s:    return "AddrModeT1_s";
    case AddrModeT2_i12:  return "AddrModeT2_i12";
    case AddrModeT2_i8:   return "AddrModeT2_i8";
    case AddrModeT2_so:   return "AddrModeT2_so";
    case AddrModeT2_pc:   return "AddrModeT2_pc";
    case AddrModeT2_i8s4: return "AddrModeT2_i8s4";
    case AddrMode_i12:    return "AddrMode_i12";
    }
  }

  /// Target Operand Flag enum.
  enum TOF {
    //===------------------------------------------------------------------===//
    // ARM Specific MachineOperand flags.

    MO_NO_FLAG,

    /// MO_LO16 - On a symbol operand, this represents a relocation containing
    /// lower 16 bit of the address. Used only via movw instruction.
    MO_LO16,

    /// MO_HI16 - On a symbol operand, this represents a relocation containing
    /// higher 16 bit of the address. Used only via movt instruction.
    MO_HI16,

    /// MO_LO16_NONLAZY - On a symbol operand "FOO", this represents a
    /// relocation containing lower 16 bit of the non-lazy-ptr indirect symbol,
    /// i.e. "FOO$non_lazy_ptr".
    /// Used only via movw instruction.
    MO_LO16_NONLAZY,

    /// MO_HI16_NONLAZY - On a symbol operand "FOO", this represents a
    /// relocation containing lower 16 bit of the non-lazy-ptr indirect symbol,
    /// i.e. "FOO$non_lazy_ptr". Used only via movt instruction.
    MO_HI16_NONLAZY,

    /// MO_LO16_NONLAZY_PIC - On a symbol operand "FOO", this represents a
    /// relocation containing lower 16 bit of the PC relative address of the
    /// non-lazy-ptr indirect symbol, i.e. "FOO$non_lazy_ptr - LABEL".
    /// Used only via movw instruction.
    MO_LO16_NONLAZY_PIC,

    /// MO_HI16_NONLAZY_PIC - On a symbol operand "FOO", this represents a
    /// relocation containing lower 16 bit of the PC relative address of the
    /// non-lazy-ptr indirect symbol, i.e. "FOO$non_lazy_ptr - LABEL".
    /// Used only via movt instruction.
    MO_HI16_NONLAZY_PIC,

    /// MO_PLT - On a symbol operand, this represents an ELF PLT reference on a
    /// call operand.
    MO_PLT
  };
} // end namespace ARMII

} // end namespace llvm;

#endif
