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

#include "llvm/MC/MCInst.h"

namespace llvm {
class MCInstrDesc;
class MCOperand;
class HexagonTargetMachine;

class HexagonMCInst : public MCInst {
public:
  explicit HexagonMCInst(unsigned op);

  /// 10.6 Instruction Packets
  bool isPacketBegin() const;
  /// \brief Is this marked as last in packet.
  bool isPacketEnd() const;
  void setPacketBegin(bool Y);
  /// \brief Mark this as last in packet.
  void setPacketEnd(bool Y);
  /// \brief Return the slots used.
  unsigned getUnits(HexagonTargetMachine const &TM) const;
  bool isConstExtended() const;
  /// \brief Return constant extended operand number.
  unsigned short getCExtOpNum(void) const;
  /// \brief Return whether this is a new-value consumer.
  bool isNewValue() const;
  /// \brief Return whether this is a legal new-value producer.
  bool hasNewValue() const;
  /// \brief Return the operand that consumes or produces a new value.
  MCOperand const &getNewValue() const;
  /// \brief Return number of bits in the constant extended operand.
  unsigned getBitCount(void) const;

private:
  /// \brief Return whether this must be always extended.
  bool isExtended() const;
  /// \brief Return true if this may be extended based on the operand value.
  bool isExtendable() const;
  ///  \brief Return if the operand can be constant extended.
  bool isOperandExtended(unsigned short const OperandNum) const;
  /// \brief Return the min value that a constant extendable operand can have
  /// without being extended.
  int getMinValue() const;
  /// \brief Return the max value that a constant extendable operand can have
  /// without being extended.
  int getMaxValue() const;
  bool packetBegin;
  bool packetEnd;
  MCInstrDesc const &MCID;
};
}

#endif
