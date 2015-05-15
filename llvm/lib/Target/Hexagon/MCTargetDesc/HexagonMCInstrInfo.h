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

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"

#include <bitset>

namespace llvm {
class MCInstrDesc;
class MCInstrInfo;
class MCInst;
class MCOperand;
namespace HexagonII {
enum class MemAccessSize;
}
namespace HexagonMCInstrInfo {
void AppendImplicitOperands(MCInst &MCI);

// Return memory access size
HexagonII::MemAccessSize getAccessSize(MCInstrInfo const &MCII,
                                       MCInst const &MCI);

// Return number of bits in the constant extended operand.
unsigned getBitCount(MCInstrInfo const &MCII, MCInst const &MCI);

// Return constant extended operand number.
unsigned short getCExtOpNum(MCInstrInfo const &MCII, MCInst const &MCI);

MCInstrDesc const &getDesc(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the implicit alignment of the extendable operand
unsigned getExtentAlignment(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the number of logical bits of the extendable operand
unsigned getExtentBits(MCInstrInfo const &MCII, MCInst const &MCI);

std::bitset<16> GetImplicitBits(MCInst const &MCI);

// Return the max value that a constant extendable operand can have
// without being extended.
int getMaxValue(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the min value that a constant extendable operand can have
// without being extended.
int getMinValue(MCInstrInfo const &MCII, MCInst const &MCI);

// Return instruction name
char const *getName(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the operand that consumes or produces a new value.
MCOperand const &getNewValue(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the Hexagon ISA class for the insn.
unsigned getType(MCInstrInfo const &MCII, MCInst const &MCI);

// Return whether the instruction is a legal new-value producer.
bool hasNewValue(MCInstrInfo const &MCII, MCInst const &MCI);

// Return whether the insn is an actual insn.
bool isCanon(MCInstrInfo const &MCII, MCInst const &MCI);

// Return whether the instruction needs to be constant extended.
bool isConstExtended(MCInstrInfo const &MCII, MCInst const &MCI);

// Return true if the insn may be extended based on the operand value.
bool isExtendable(MCInstrInfo const &MCII, MCInst const &MCI);

// Return whether the instruction must be always extended.
bool isExtended(MCInstrInfo const &MCII, MCInst const &MCI);

// Return whether the insn is a new-value consumer.
bool isNewValue(MCInstrInfo const &MCII, MCInst const &MCI);

// Return true if the operand can be constant extended.
bool isOperandExtended(MCInstrInfo const &MCII, MCInst const &MCI,
                       unsigned short OperandNum);

bool isPacketBegin(MCInst const &MCI);

bool isPacketEnd(MCInst const &MCI);

// Return whether the insn is a prefix.
bool isPrefix(MCInstrInfo const &MCII, MCInst const &MCI);

// Return whether the insn is solo, i.e., cannot be in a packet.
bool isSolo(MCInstrInfo const &MCII, MCInst const &MCI);

static const size_t packetBeginIndex = 0;
static const size_t packetEndIndex = 1;

void resetPacket(MCInst &MCI);

inline void SanityCheckImplicitOperands(MCInst const &MCI) {
  assert(MCI.getNumOperands() >= 2 && "At least the two implicit operands");
  assert(MCI.getOperand(MCI.getNumOperands() - 1).isInst() &&
         "Implicit bits and flags");
  assert(MCI.getOperand(MCI.getNumOperands() - 2).isImm() && "Parent pointer");
}

void SetImplicitBits(MCInst &MCI, std::bitset<16> Bits);

void setPacketBegin(MCInst &MCI, bool Y);

void setPacketEnd(MCInst &MCI, bool Y);
}
}

#endif // LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCINSTRINFO_H
