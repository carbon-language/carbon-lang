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
size_t const innerLoopOffset = 0;
int64_t const innerLoopMask = 1 << innerLoopOffset;

size_t const outerLoopOffset = 1;
int64_t const outerLoopMask = 1 << outerLoopOffset;

size_t const bundleInstructionsOffset = 1;

// Returns the number of instructions in the bundle
size_t bundleSize(MCInst const &MCI);

// Returns a iterator range of instructions in this bundle
iterator_range<MCInst::const_iterator> bundleInstructions(MCInst const &MCI);

// Return memory access size
HexagonII::MemAccessSize getAccessSize(MCInstrInfo const &MCII,
                                       MCInst const &MCI);

// Return number of bits in the constant extended operand.
unsigned getBitCount(MCInstrInfo const &MCII, MCInst const &MCI);

// Return constant extended operand number.
unsigned short getCExtOpNum(MCInstrInfo const &MCII, MCInst const &MCI);

MCInstrDesc const &getDesc(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the index of the extendable operand
unsigned short getExtendableOp(MCInstrInfo const &MCII, MCInst const &MCI);

// Return a reference to the extendable operand
MCOperand const &getExtendableOperand(MCInstrInfo const &MCII,
                                      MCInst const &MCI);

// Return the implicit alignment of the extendable operand
unsigned getExtentAlignment(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the number of logical bits of the extendable operand
unsigned getExtentBits(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the max value that a constant extendable operand can have
// without being extended.
int getMaxValue(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the min value that a constant extendable operand can have
// without being extended.
int getMinValue(MCInstrInfo const &MCII, MCInst const &MCI);

// Return instruction name
char const *getName(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the operand index for the new value.
unsigned short getNewValueOp(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the operand that consumes or produces a new value.
MCOperand const &getNewValueOperand(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the Hexagon ISA class for the insn.
unsigned getType(MCInstrInfo const &MCII, MCInst const &MCI);

// Return whether the instruction is a legal new-value producer.
bool hasNewValue(MCInstrInfo const &MCII, MCInst const &MCI);

// Return the instruction at Index
MCInst const &instruction(MCInst const &MCB, size_t Index);

// Returns whether this MCInst is a wellformed bundle
bool isBundle(MCInst const &MCI);

// Return whether the insn is an actual insn.
bool isCanon(MCInstrInfo const &MCII, MCInst const &MCI);

// Return whether the instruction needs to be constant extended.
bool isConstExtended(MCInstrInfo const &MCII, MCInst const &MCI);

// Return true if the insn may be extended based on the operand value.
bool isExtendable(MCInstrInfo const &MCII, MCInst const &MCI);

// Return whether the instruction must be always extended.
bool isExtended(MCInstrInfo const &MCII, MCInst const &MCI);

// Returns whether this instruction is an immediate extender
bool isImmext(MCInst const &MCI);

// Returns whether this bundle is an endloop0
bool isInnerLoop(MCInst const &MCI);

// Return whether the insn is a new-value consumer.
bool isNewValue(MCInstrInfo const &MCII, MCInst const &MCI);

// Return true if the operand can be constant extended.
bool isOperandExtended(MCInstrInfo const &MCII, MCInst const &MCI,
                       unsigned short OperandNum);

// Returns whether this bundle is an endloop1
bool isOuterLoop(MCInst const &MCI);

// Return whether this instruction is predicated
bool isPredicated(MCInstrInfo const &MCII, MCInst const &MCI);

// Return whether the predicate sense is true
bool isPredicatedTrue(MCInstrInfo const &MCII, MCInst const &MCI);

// Return whether the insn is a prefix.
bool isPrefix(MCInstrInfo const &MCII, MCInst const &MCI);

// Return whether the insn is solo, i.e., cannot be in a packet.
bool isSolo(MCInstrInfo const &MCII, MCInst const &MCI);

// Pad the bundle with nops to satisfy endloop requirements
void padEndloop(MCInst &MCI);

// Marks a bundle as endloop0
void setInnerLoop(MCInst &MCI);

// Marks a bundle as endloop1
void setOuterLoop(MCInst &MCI);
}
}

#endif // LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCINSTRINFO_H
