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

#ifndef HEXAGONMCINST_H
#define HEXAGONMCINST_H

#include "HexagonTargetMachine.h"
#include "llvm/MC/MCInst.h"

namespace llvm {
  class MCOperand;

  class HexagonMCInst: public MCInst {
    // MCID is set during instruction lowering.
    // It is needed in order to access TSFlags for
    // use in checking MC instruction properties.
    const MCInstrDesc *MCID;

    // Packet start and end markers
    unsigned packetStart: 1, packetEnd: 1;

  public:
    explicit HexagonMCInst():
      MCInst(), MCID(nullptr), packetStart(0), packetEnd(0) {};
    HexagonMCInst(const MCInstrDesc& mcid):
      MCInst(), MCID(&mcid), packetStart(0), packetEnd(0) {};

    bool isPacketStart() const { return (packetStart); };
    bool isPacketEnd() const { return (packetEnd); };
    void setPacketStart(bool Y) { packetStart = Y; };
    void setPacketEnd(bool Y) { packetEnd = Y; };
    void resetPacket() { setPacketStart(false); setPacketEnd(false); };

    // Return the slots used by the insn.
    unsigned getUnits(const HexagonTargetMachine* TM) const;

    // Return the Hexagon ISA class for the insn.
    unsigned getType() const;

    void setDesc(const MCInstrDesc& mcid) { MCID = &mcid; };
    const MCInstrDesc& getDesc(void) const { return *MCID; };

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
    const MCOperand& getNewValue() const;

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
