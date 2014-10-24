//===- HexagonInstPrinter.cpp - Convert Hexagon MCInst to assembly syntax -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an Hexagon MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "HexagonAsmPrinter.h"
#include "Hexagon.h"
#include "HexagonInstPrinter.h"
#include "MCTargetDesc/HexagonMCInst.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#define GET_INSTRUCTION_NAME
#include "HexagonGenAsmWriter.inc"

const char HexagonInstPrinter::PacketPadding = '\t';
// Return the minimum value that a constant extendable operand can have
// without being extended.
static int getMinValue(uint64_t TSFlags) {
  unsigned isSigned =
      (TSFlags >> HexagonII::ExtentSignedPos) & HexagonII::ExtentSignedMask;
  unsigned bits =
      (TSFlags >> HexagonII::ExtentBitsPos) & HexagonII::ExtentBitsMask;

  if (isSigned)
    return -1U << (bits - 1);

  return 0;
}

// Return the maximum value that a constant extendable operand can have
// without being extended.
static int getMaxValue(uint64_t TSFlags) {
  unsigned isSigned =
      (TSFlags >> HexagonII::ExtentSignedPos) & HexagonII::ExtentSignedMask;
  unsigned bits =
      (TSFlags >> HexagonII::ExtentBitsPos) & HexagonII::ExtentBitsMask;

  if (isSigned)
    return ~(-1U << (bits - 1));

  return ~(-1U << bits);
}

// Return true if the instruction must be extended.
static bool isExtended(uint64_t TSFlags) {
  return (TSFlags >> HexagonII::ExtendedPos) & HexagonII::ExtendedMask;
}

// Currently just used in an assert statement
static bool isExtendable(uint64_t TSFlags) LLVM_ATTRIBUTE_UNUSED;
// Return true if the instruction may be extended based on the operand value.
static bool isExtendable(uint64_t TSFlags) {
  return (TSFlags >> HexagonII::ExtendablePos) & HexagonII::ExtendableMask;
}

StringRef HexagonInstPrinter::getOpcodeName(unsigned Opcode) const {
  return MII.getName(Opcode);
}

StringRef HexagonInstPrinter::getRegName(unsigned RegNo) const {
  return getRegisterName(RegNo);
}

void HexagonInstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                                   StringRef Annot) {
  printInst((const HexagonMCInst*)(MI), O, Annot);
}

void HexagonInstPrinter::printInst(const HexagonMCInst *MI, raw_ostream &O,
                                   StringRef Annot) {
  const char startPacket = '{',
             endPacket = '}';
  // TODO: add outer HW loop when it's supported too.
  if (MI->getOpcode() == Hexagon::ENDLOOP0) {
    // Ending a harware loop is different from ending an regular packet.
    assert(MI->isPacketEnd() && "Loop-end must also end the packet");

    if (MI->isPacketBegin()) {
      // There must be a packet to end a loop.
      // FIXME: when shuffling is always run, this shouldn't be needed.
      HexagonMCInst Nop (Hexagon::NOP);
      StringRef NoAnnot;

      Nop.setPacketBegin (MI->isPacketBegin());
      printInst (&Nop, O, NoAnnot);
    }

    // Close the packet.
    if (MI->isPacketEnd())
      O << PacketPadding << endPacket;

    printInstruction(MI, O);
  }
  else {
    // Prefix the insn opening the packet.
    if (MI->isPacketBegin())
      O << PacketPadding << startPacket << '\n';

    printInstruction(MI, O);

    // Suffix the insn closing the packet.
    if (MI->isPacketEnd())
      // Suffix the packet in a new line always, since the GNU assembler has
      // issues with a closing brace on the same line as CONST{32,64}.
      O << '\n' << PacketPadding << endPacket;
  }

  printAnnotation(O, Annot);
}

void HexagonInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                      raw_ostream &O) const {
  const MCOperand& MO = MI->getOperand(OpNo);

  if (MO.isReg()) {
    O << getRegisterName(MO.getReg());
  } else if(MO.isExpr()) {
    O << *MO.getExpr();
  } else if(MO.isImm()) {
    printImmOperand(MI, OpNo, O);
  } else {
    llvm_unreachable("Unknown operand");
  }
}

void HexagonInstPrinter::printImmOperand(const MCInst *MI, unsigned OpNo,
                                         raw_ostream &O) const {
  const MCOperand& MO = MI->getOperand(OpNo);

  if(MO.isExpr()) {
    O << *MO.getExpr();
  } else if(MO.isImm()) {
    O << MI->getOperand(OpNo).getImm();
  } else {
    llvm_unreachable("Unknown operand");
  }
}

void HexagonInstPrinter::printExtOperand(const MCInst *MI, unsigned OpNo,
                                         raw_ostream &O) const {
  const MCOperand &MO = MI->getOperand(OpNo);
  const MCInstrDesc &MII = getMII().get(MI->getOpcode());

  assert((isExtendable(MII.TSFlags) || isExtended(MII.TSFlags)) &&
         "Expecting an extendable operand");

  if (MO.isExpr() || isExtended(MII.TSFlags)) {
    O << "#";
  } else if (MO.isImm()) {
    int ImmValue = MO.getImm();
    if (ImmValue < getMinValue(MII.TSFlags) ||
        ImmValue > getMaxValue(MII.TSFlags))
      O << "#";
  }
  printOperand(MI, OpNo, O);
}

void HexagonInstPrinter::printUnsignedImmOperand(const MCInst *MI,
                                    unsigned OpNo, raw_ostream &O) const {
  O << MI->getOperand(OpNo).getImm();
}

void HexagonInstPrinter::printNegImmOperand(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &O) const {
  O << -MI->getOperand(OpNo).getImm();
}

void HexagonInstPrinter::printNOneImmOperand(const MCInst *MI, unsigned OpNo,
                                             raw_ostream &O) const {
  O << -1;
}

void HexagonInstPrinter::printMEMriOperand(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) const {
  const MCOperand& MO0 = MI->getOperand(OpNo);
  const MCOperand& MO1 = MI->getOperand(OpNo + 1);

  O << getRegisterName(MO0.getReg());
  O << " + #" << MO1.getImm();
}

void HexagonInstPrinter::printFrameIndexOperand(const MCInst *MI, unsigned OpNo,
                                                raw_ostream &O) const {
  const MCOperand& MO0 = MI->getOperand(OpNo);
  const MCOperand& MO1 = MI->getOperand(OpNo + 1);

  O << getRegisterName(MO0.getReg()) << ", #" << MO1.getImm();
}

void HexagonInstPrinter::printGlobalOperand(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &O) const {
  assert(MI->getOperand(OpNo).isExpr() && "Expecting expression");

  printOperand(MI, OpNo, O);
}

void HexagonInstPrinter::printJumpTable(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) const {
  assert(MI->getOperand(OpNo).isExpr() && "Expecting expression");

  printOperand(MI, OpNo, O);
}

void HexagonInstPrinter::printConstantPool(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) const {
  assert(MI->getOperand(OpNo).isExpr() && "Expecting expression");

  printOperand(MI, OpNo, O);
}

void HexagonInstPrinter::printBranchOperand(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &O) const {
  // Branches can take an immediate operand.  This is used by the branch
  // selection pass to print $+8, an eight byte displacement from the PC.
  llvm_unreachable("Unknown branch operand.");
}

void HexagonInstPrinter::printCallOperand(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) const {
}

void HexagonInstPrinter::printAbsAddrOperand(const MCInst *MI, unsigned OpNo,
                                             raw_ostream &O) const {
}

void HexagonInstPrinter::printPredicateOperand(const MCInst *MI, unsigned OpNo,
                                               raw_ostream &O) const {
}

void HexagonInstPrinter::printSymbol(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O, bool hi) const {
  assert(MI->getOperand(OpNo).isImm() && "Unknown symbol operand");

  O << '#' << (hi ? "HI" : "LO") << "(#";
  printOperand(MI, OpNo, O);
  O << ')';
}
