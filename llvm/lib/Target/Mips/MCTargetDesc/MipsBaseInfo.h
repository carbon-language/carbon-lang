//===-- MipsBaseInfo.h - Top level definitions for ARM ------- --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the Mips target useful for the compiler back-end and the MC libraries.
//
//===----------------------------------------------------------------------===//
#ifndef MIPSBASEINFO_H
#define MIPSBASEINFO_H

#include "MipsMCTargetDesc.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
/// getMipsRegisterNumbering - Given the enum value for some register,
/// return the number that it corresponds to.
inline static unsigned getMipsRegisterNumbering(unsigned RegEnum)
{
  switch (RegEnum) {
  case Mips::ZERO: case Mips::ZERO_64: case Mips::F0: case Mips::D0_64:
  case Mips::D0:
    return 0;
  case Mips::AT: case Mips::AT_64: case Mips::F1: case Mips::D1_64:
    return 1;
  case Mips::V0: case Mips::V0_64: case Mips::F2: case Mips::D2_64:
  case Mips::D1:
    return 2;
  case Mips::V1: case Mips::V1_64: case Mips::F3: case Mips::D3_64:
    return 3;
  case Mips::A0: case Mips::A0_64: case Mips::F4: case Mips::D4_64:
  case Mips::D2:
    return 4;
  case Mips::A1: case Mips::A1_64: case Mips::F5: case Mips::D5_64:
    return 5;
  case Mips::A2: case Mips::A2_64: case Mips::F6: case Mips::D6_64:
  case Mips::D3:
    return 6;
  case Mips::A3: case Mips::A3_64: case Mips::F7: case Mips::D7_64:
    return 7;
  case Mips::T0: case Mips::T0_64: case Mips::F8: case Mips::D8_64:
  case Mips::D4:
    return 8;
  case Mips::T1: case Mips::T1_64: case Mips::F9: case Mips::D9_64:
    return 9;
  case Mips::T2: case Mips::T2_64: case Mips::F10: case Mips::D10_64:
  case Mips::D5:
    return 10;
  case Mips::T3: case Mips::T3_64: case Mips::F11: case Mips::D11_64:
    return 11;
  case Mips::T4: case Mips::T4_64: case Mips::F12: case Mips::D12_64:
  case Mips::D6:
    return 12;
  case Mips::T5: case Mips::T5_64: case Mips::F13: case Mips::D13_64:
    return 13;
  case Mips::T6: case Mips::T6_64: case Mips::F14: case Mips::D14_64:
  case Mips::D7:
    return 14;
  case Mips::T7: case Mips::T7_64: case Mips::F15: case Mips::D15_64:
    return 15;
  case Mips::S0: case Mips::S0_64: case Mips::F16: case Mips::D16_64:
  case Mips::D8:
    return 16;
  case Mips::S1: case Mips::S1_64: case Mips::F17: case Mips::D17_64:
    return 17;
  case Mips::S2: case Mips::S2_64: case Mips::F18: case Mips::D18_64:
  case Mips::D9:
    return 18;
  case Mips::S3: case Mips::S3_64: case Mips::F19: case Mips::D19_64:
    return 19;
  case Mips::S4: case Mips::S4_64: case Mips::F20: case Mips::D20_64:
  case Mips::D10:
    return 20;
  case Mips::S5: case Mips::S5_64: case Mips::F21: case Mips::D21_64:
    return 21;
  case Mips::S6: case Mips::S6_64: case Mips::F22: case Mips::D22_64:
  case Mips::D11:
    return 22;
  case Mips::S7: case Mips::S7_64: case Mips::F23: case Mips::D23_64:
    return 23;
  case Mips::T8: case Mips::T8_64: case Mips::F24: case Mips::D24_64:
  case Mips::D12:
    return 24;
  case Mips::T9: case Mips::T9_64: case Mips::F25: case Mips::D25_64:
    return 25;
  case Mips::K0: case Mips::K0_64: case Mips::F26: case Mips::D26_64:
  case Mips::D13:
    return 26;
  case Mips::K1: case Mips::K1_64: case Mips::F27: case Mips::D27_64:
    return 27;
  case Mips::GP: case Mips::GP_64: case Mips::F28: case Mips::D28_64:
  case Mips::D14:
    return 28;
  case Mips::SP: case Mips::SP_64: case Mips::F29: case Mips::D29_64:
    return 29;
  case Mips::FP: case Mips::FP_64: case Mips::F30: case Mips::D30_64:
  case Mips::D15: 
    return 30;
  case Mips::RA: case Mips::RA_64: case Mips::F31: case Mips::D31_64:
    return 31;
  default: llvm_unreachable("Unknown register number!");
  }
  return 0; // Not reached
}
}

#endif
