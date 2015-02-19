//===- HexagonMCInstrInfo.cpp - Hexagon sub-class of MCInst ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class extends MCInstrInfo to allow Hexagon specific MCInstr queries
//
//===----------------------------------------------------------------------===//

#include "HexagonMCInstrInfo.h"
#include "HexagonBaseInfo.h"
#include "llvm/MC/MCInstrInfo.h"

namespace llvm {
MCInstrDesc const &HexagonMCInstrInfo::getDesc(MCInstrInfo const &MCII,
                                               MCInst const &MCI) {
  return (MCII.get(MCI.getOpcode()));
}
// Return the max value that a constant extendable operand can have
// without being extended.
int HexagonMCInstrInfo::getMaxValue(MCInstrInfo const &MCII,
                                    MCInst const &MCI) {
  const uint64_t F = HexagonMCInstrInfo::getDesc(MCII, MCI).TSFlags;
  unsigned isSigned =
      (F >> HexagonII::ExtentSignedPos) & HexagonII::ExtentSignedMask;
  unsigned bits = (F >> HexagonII::ExtentBitsPos) & HexagonII::ExtentBitsMask;

  if (isSigned) // if value is signed
    return ~(-1U << (bits - 1));
  else
    return ~(-1U << bits);
}
}
