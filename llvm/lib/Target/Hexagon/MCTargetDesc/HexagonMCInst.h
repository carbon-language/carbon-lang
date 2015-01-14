//===- HexagonMCInst.h - Hexagon sub-class of MCInst ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class extends MCInst to allow some VLIW annotations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCINST_H
#define LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCINST_H

#include "HexagonTargetMachine.h"
#include "llvm/MC/MCInst.h"
#include <memory>

extern "C" void LLVMInitializeHexagonTargetMC();
namespace llvm {
class MCOperand;

class HexagonMCInst : public MCInst {
  friend void ::LLVMInitializeHexagonTargetMC();
  // Used to access TSFlags
  static std::unique_ptr <MCInstrInfo const> MCII;

public:
  explicit HexagonMCInst();
  HexagonMCInst(const MCInstrDesc &mcid);

  static void AppendImplicitOperands(MCInst &MCI);
  static std::bitset<16> GetImplicitBits(MCInst const &MCI);
  static void SetImplicitBits(MCInst &MCI, std::bitset<16> Bits);
  static void SanityCheckImplicitOperands(MCInst const &MCI) {
    assert(MCI.getNumOperands() >= 2 && "At least the two implicit operands");
    assert(MCI.getOperand(MCI.getNumOperands() - 1).isInst() &&
           "Implicit bits and flags");
    assert(MCI.getOperand(MCI.getNumOperands() - 2).isImm() &&
           "Parent pointer");
  }

  void setPacketBegin(bool Y);
  bool isPacketBegin() const;
  static const size_t packetBeginIndex = 0;
  void setPacketEnd(bool Y);
  bool isPacketEnd() const;
  static const size_t packetEndIndex = 1;
  void resetPacket();

  // Return the slots used by the insn.
  unsigned getUnits(const HexagonTargetMachine *TM) const;

  // Return the Hexagon ISA class for the insn.
  unsigned getType() const;

  MCInstrDesc const &getDesc() const;

  // Return whether the insn is an actual insn.
  bool isCanon() const;

  // Return whether the insn is a prefix.
  bool isPrefix() const;

  // Return whether the insn is solo, i.e., cannot be in a packet.
  bool isSolo() const;

  // Return whether the instruction needs to be constant extended.
  bool isConstExtended() const;

  // Return constant extended operand number.
  unsigned short getCExtOpNum(void) const;

  // Return whether the insn is a new-value consumer.
  bool isNewValue() const;

  // Return whether the instruction is a legal new-value producer.
  bool hasNewValue() const;

  // Return the operand that consumes or produces a new value.
  const MCOperand &getNewValue() const;

  // Return number of bits in the constant extended operand.
  unsigned getBitCount(void) const;

private:
  // Return whether the instruction must be always extended.
  bool isExtended() const;

  // Return true if the insn may be extended based on the operand value.
  bool isExtendable() const;

  // Return true if the operand can be constant extended.
  bool isOperandExtended(const unsigned short OperandNum) const;

  // Return the min value that a constant extendable operand can have
  // without being extended.
  int getMinValue() const;

  // Return the max value that a constant extendable operand can have
  // without being extended.
  int getMaxValue() const;
};
}

#endif
