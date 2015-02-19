//===- HexagonMCInstrInfo.cpp - Hexagon sub-class of MCInst ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Utility functions for Hexagon specific MCInst queries
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCINSTRINFO_H
#define LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCINSTRINFO_H

namespace llvm {
class MCInstrDesc;
class MCInstrInfo;
class MCInst;
namespace HexagonMCInstrInfo {
MCInstrDesc const &getDesc(MCInstrInfo const &MCII, MCInst const &MCI);
// Return the max value that a constant extendable operand can have
// without being extended.
int getMaxValue(MCInstrInfo const &MCII, MCInst const &MCI);
}
}

#endif // LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCINSTRINFO_H
