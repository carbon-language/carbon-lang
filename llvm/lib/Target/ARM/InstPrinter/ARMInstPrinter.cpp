//===-- ARMInstPrinter.cpp - Convert ARM MCInst to assembly syntax --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an ARM MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "ARMInstPrinter.h"
#include "MCTargetDesc/ARMBaseInfo.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define GET_INSTRUCTION_NAME
#include "ARMGenAsmWriter.inc"

/// translateShiftImm - Convert shift immediate from 0-31 to 1-32 for printing.
///
/// getSORegOffset returns an integer from 0-31, but '0' should actually be printed
/// 32 as the immediate shouldbe within the range 1-32.
static unsigned translateShiftImm(unsigned imm) {
  if (imm == 0)
    return 32;
  return imm;
}


ARMInstPrinter::ARMInstPrinter(const MCAsmInfo &MAI,
                               const MCSubtargetInfo &STI) :
  MCInstPrinter(MAI) {
  // Initialize the set of available features.
  setAvailableFeatures(STI.getFeatureBits());
}

StringRef ARMInstPrinter::getOpcodeName(unsigned Opcode) const {
  return getInstructionName(Opcode);
}

void ARMInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  OS << getRegisterName(RegNo);
}

void ARMInstPrinter::printInst(const MCInst *MI, raw_ostream &O) {
  unsigned Opcode = MI->getOpcode();

  // Check for MOVs and print canonical forms, instead.
  if (Opcode == ARM::MOVsr) {
    // FIXME: Thumb variants?
    const MCOperand &Dst = MI->getOperand(0);
    const MCOperand &MO1 = MI->getOperand(1);
    const MCOperand &MO2 = MI->getOperand(2);
    const MCOperand &MO3 = MI->getOperand(3);

    O << '\t' << ARM_AM::getShiftOpcStr(ARM_AM::getSORegShOp(MO3.getImm()));
    printSBitModifierOperand(MI, 6, O);
    printPredicateOperand(MI, 4, O);

    O << '\t' << getRegisterName(Dst.getReg())
      << ", " << getRegisterName(MO1.getReg());

    O << ", " << getRegisterName(MO2.getReg());
    assert(ARM_AM::getSORegOffset(MO3.getImm()) == 0);
    return;
  }

  if (Opcode == ARM::MOVsi) {
    // FIXME: Thumb variants?
    const MCOperand &Dst = MI->getOperand(0);
    const MCOperand &MO1 = MI->getOperand(1);
    const MCOperand &MO2 = MI->getOperand(2);

    O << '\t' << ARM_AM::getShiftOpcStr(ARM_AM::getSORegShOp(MO2.getImm()));
    printSBitModifierOperand(MI, 5, O);
    printPredicateOperand(MI, 3, O);

    O << '\t' << getRegisterName(Dst.getReg())
      << ", " << getRegisterName(MO1.getReg());

    if (ARM_AM::getSORegShOp(MO2.getImm()) == ARM_AM::rrx)
      return;

    O << ", #" << translateShiftImm(ARM_AM::getSORegOffset(MO2.getImm()));
    return;
  }


  // A8.6.123 PUSH
  if ((Opcode == ARM::STMDB_UPD || Opcode == ARM::t2STMDB_UPD) &&
      MI->getOperand(0).getReg() == ARM::SP) {
    O << '\t' << "push";
    printPredicateOperand(MI, 2, O);
    if (Opcode == ARM::t2STMDB_UPD)
      O << ".w";
    O << '\t';
    printRegisterList(MI, 4, O);
    return;
  }
  if (Opcode == ARM::STR_PRE_IMM && MI->getOperand(2).getReg() == ARM::SP &&
      MI->getOperand(3).getImm() == -4) {
    O << '\t' << "push";
    printPredicateOperand(MI, 4, O);
    O << "\t{" << getRegisterName(MI->getOperand(1).getReg()) << "}";
    return;
  }

  // A8.6.122 POP
  if ((Opcode == ARM::LDMIA_UPD || Opcode == ARM::t2LDMIA_UPD) &&
      MI->getOperand(0).getReg() == ARM::SP) {
    O << '\t' << "pop";
    printPredicateOperand(MI, 2, O);
    if (Opcode == ARM::t2LDMIA_UPD)
      O << ".w";
    O << '\t';
    printRegisterList(MI, 4, O);
    return;
  }
  if (Opcode == ARM::LDR_POST_IMM && MI->getOperand(2).getReg() == ARM::SP &&
      MI->getOperand(4).getImm() == 4) {
    O << '\t' << "pop";
    printPredicateOperand(MI, 5, O);
    O << "\t{" << getRegisterName(MI->getOperand(0).getReg()) << "}";
    return;
  }


  // A8.6.355 VPUSH
  if ((Opcode == ARM::VSTMSDB_UPD || Opcode == ARM::VSTMDDB_UPD) &&
      MI->getOperand(0).getReg() == ARM::SP) {
    O << '\t' << "vpush";
    printPredicateOperand(MI, 2, O);
    O << '\t';
    printRegisterList(MI, 4, O);
    return;
  }

  // A8.6.354 VPOP
  if ((Opcode == ARM::VLDMSIA_UPD || Opcode == ARM::VLDMDIA_UPD) &&
      MI->getOperand(0).getReg() == ARM::SP) {
    O << '\t' << "vpop";
    printPredicateOperand(MI, 2, O);
    O << '\t';
    printRegisterList(MI, 4, O);
    return;
  }

  if (Opcode == ARM::tLDMIA) {
    bool Writeback = true;
    unsigned BaseReg = MI->getOperand(0).getReg();
    for (unsigned i = 3; i < MI->getNumOperands(); ++i) {
      if (MI->getOperand(i).getReg() == BaseReg)
        Writeback = false;
    }

    O << "\tldm";

    printPredicateOperand(MI, 1, O);
    O << '\t' << getRegisterName(BaseReg);
    if (Writeback) O << "!";
    O << ", ";
    printRegisterList(MI, 3, O);
    return;
  }

  // Thumb1 NOP
  if (Opcode == ARM::tMOVr && MI->getOperand(0).getReg() == ARM::R8 &&
      MI->getOperand(1).getReg() == ARM::R8) {
    O << "\tnop";
    printPredicateOperand(MI, 2, O);
    return;
  }

  printInstruction(MI, O);
}

void ARMInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    unsigned Reg = Op.getReg();
    O << getRegisterName(Reg);
  } else if (Op.isImm()) {
    O << '#' << Op.getImm();
  } else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    O << *Op.getExpr();
  }
}

// so_reg is a 4-operand unit corresponding to register forms of the A5.1
// "Addressing Mode 1 - Data-processing operands" forms.  This includes:
//    REG 0   0           - e.g. R5
//    REG REG 0,SH_OPC    - e.g. R5, ROR R3
//    REG 0   IMM,SH_OPC  - e.g. R5, LSL #3
void ARMInstPrinter::printSORegRegOperand(const MCInst *MI, unsigned OpNum,
                                       raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);
  const MCOperand &MO3 = MI->getOperand(OpNum+2);

  O << getRegisterName(MO1.getReg());

  // Print the shift opc.
  ARM_AM::ShiftOpc ShOpc = ARM_AM::getSORegShOp(MO3.getImm());
  O << ", " << ARM_AM::getShiftOpcStr(ShOpc);
  if (ShOpc == ARM_AM::rrx)
    return;
  
  O << ' ' << getRegisterName(MO2.getReg());
  assert(ARM_AM::getSORegOffset(MO3.getImm()) == 0);
}

void ARMInstPrinter::printSORegImmOperand(const MCInst *MI, unsigned OpNum,
                                       raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  O << getRegisterName(MO1.getReg());

  // Print the shift opc.
  ARM_AM::ShiftOpc ShOpc = ARM_AM::getSORegShOp(MO2.getImm());
  O << ", " << ARM_AM::getShiftOpcStr(ShOpc);
  if (ShOpc == ARM_AM::rrx)
    return;
  O << " #" << translateShiftImm(ARM_AM::getSORegOffset(MO2.getImm()));
}


//===--------------------------------------------------------------------===//
// Addressing Mode #2
//===--------------------------------------------------------------------===//

void ARMInstPrinter::printAM2PreOrOffsetIndexOp(const MCInst *MI, unsigned Op,
                                                raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(Op);
  const MCOperand &MO2 = MI->getOperand(Op+1);
  const MCOperand &MO3 = MI->getOperand(Op+2);

  O << "[" << getRegisterName(MO1.getReg());

  if (!MO2.getReg()) {
    if (ARM_AM::getAM2Offset(MO3.getImm())) // Don't print +0.
      O << ", #"
        << ARM_AM::getAddrOpcStr(ARM_AM::getAM2Op(MO3.getImm()))
        << ARM_AM::getAM2Offset(MO3.getImm());
    O << "]";
    return;
  }

  O << ", "
    << ARM_AM::getAddrOpcStr(ARM_AM::getAM2Op(MO3.getImm()))
    << getRegisterName(MO2.getReg());

  if (unsigned ShImm = ARM_AM::getAM2Offset(MO3.getImm()))
    O << ", "
    << ARM_AM::getShiftOpcStr(ARM_AM::getAM2ShiftOpc(MO3.getImm()))
    << " #" << ShImm;
  O << "]";
}

void ARMInstPrinter::printAM2PostIndexOp(const MCInst *MI, unsigned Op,
                                         raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(Op);
  const MCOperand &MO2 = MI->getOperand(Op+1);
  const MCOperand &MO3 = MI->getOperand(Op+2);

  O << "[" << getRegisterName(MO1.getReg()) << "], ";

  if (!MO2.getReg()) {
    unsigned ImmOffs = ARM_AM::getAM2Offset(MO3.getImm());
    O << '#'
      << ARM_AM::getAddrOpcStr(ARM_AM::getAM2Op(MO3.getImm()))
      << ImmOffs;
    return;
  }

  O << ARM_AM::getAddrOpcStr(ARM_AM::getAM2Op(MO3.getImm()))
    << getRegisterName(MO2.getReg());

  if (unsigned ShImm = ARM_AM::getAM2Offset(MO3.getImm()))
    O << ", "
    << ARM_AM::getShiftOpcStr(ARM_AM::getAM2ShiftOpc(MO3.getImm()))
    << " #" << ShImm;
}

void ARMInstPrinter::printAddrMode2Operand(const MCInst *MI, unsigned Op,
                                           raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(Op);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, Op, O);
    return;
  }

  const MCOperand &MO3 = MI->getOperand(Op+2);
  unsigned IdxMode = ARM_AM::getAM2IdxMode(MO3.getImm());

  if (IdxMode == ARMII::IndexModePost) {
    printAM2PostIndexOp(MI, Op, O);
    return;
  }
  printAM2PreOrOffsetIndexOp(MI, Op, O);
}

void ARMInstPrinter::printAddrMode2OffsetOperand(const MCInst *MI,
                                                 unsigned OpNum,
                                                 raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  if (!MO1.getReg()) {
    unsigned ImmOffs = ARM_AM::getAM2Offset(MO2.getImm());
    O << '#'
      << ARM_AM::getAddrOpcStr(ARM_AM::getAM2Op(MO2.getImm()))
      << ImmOffs;
    return;
  }

  O << ARM_AM::getAddrOpcStr(ARM_AM::getAM2Op(MO2.getImm()))
    << getRegisterName(MO1.getReg());

  if (unsigned ShImm = ARM_AM::getAM2Offset(MO2.getImm()))
    O << ", "
    << ARM_AM::getShiftOpcStr(ARM_AM::getAM2ShiftOpc(MO2.getImm()))
    << " #" << ShImm;
}

//===--------------------------------------------------------------------===//
// Addressing Mode #3
//===--------------------------------------------------------------------===//

void ARMInstPrinter::printAM3PostIndexOp(const MCInst *MI, unsigned Op,
                                         raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(Op);
  const MCOperand &MO2 = MI->getOperand(Op+1);
  const MCOperand &MO3 = MI->getOperand(Op+2);

  O << "[" << getRegisterName(MO1.getReg()) << "], ";

  if (MO2.getReg()) {
    O << (char)ARM_AM::getAM3Op(MO3.getImm())
    << getRegisterName(MO2.getReg());
    return;
  }

  unsigned ImmOffs = ARM_AM::getAM3Offset(MO3.getImm());
  O << '#'
    << ARM_AM::getAddrOpcStr(ARM_AM::getAM3Op(MO3.getImm()))
    << ImmOffs;
}

void ARMInstPrinter::printAM3PreOrOffsetIndexOp(const MCInst *MI, unsigned Op,
                                                raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(Op);
  const MCOperand &MO2 = MI->getOperand(Op+1);
  const MCOperand &MO3 = MI->getOperand(Op+2);

  O << '[' << getRegisterName(MO1.getReg());

  if (MO2.getReg()) {
    O << ", " << getAddrOpcStr(ARM_AM::getAM3Op(MO3.getImm()))
      << getRegisterName(MO2.getReg()) << ']';
    return;
  }

  if (unsigned ImmOffs = ARM_AM::getAM3Offset(MO3.getImm()))
    O << ", #"
      << ARM_AM::getAddrOpcStr(ARM_AM::getAM3Op(MO3.getImm()))
      << ImmOffs;
  O << ']';
}

void ARMInstPrinter::printAddrMode3Operand(const MCInst *MI, unsigned Op,
                                           raw_ostream &O) {
  const MCOperand &MO3 = MI->getOperand(Op+2);
  unsigned IdxMode = ARM_AM::getAM3IdxMode(MO3.getImm());

  if (IdxMode == ARMII::IndexModePost) {
    printAM3PostIndexOp(MI, Op, O);
    return;
  }
  printAM3PreOrOffsetIndexOp(MI, Op, O);
}

void ARMInstPrinter::printAddrMode3OffsetOperand(const MCInst *MI,
                                                 unsigned OpNum,
                                                 raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  if (MO1.getReg()) {
    O << getAddrOpcStr(ARM_AM::getAM3Op(MO2.getImm()))
      << getRegisterName(MO1.getReg());
    return;
  }

  unsigned ImmOffs = ARM_AM::getAM3Offset(MO2.getImm());
  O << '#'
    << ARM_AM::getAddrOpcStr(ARM_AM::getAM3Op(MO2.getImm()))
    << ImmOffs;
}

void ARMInstPrinter::printPostIdxImm8Operand(const MCInst *MI,
                                             unsigned OpNum,
                                             raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  unsigned Imm = MO.getImm();
  O << '#' << ((Imm & 256) ? "" : "-") << (Imm & 0xff);
}

void ARMInstPrinter::printPostIdxRegOperand(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  O << (MO2.getImm() ? "" : "-") << getRegisterName(MO1.getReg());
}

void ARMInstPrinter::printPostIdxImm8s4Operand(const MCInst *MI,
                                             unsigned OpNum,
                                             raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  unsigned Imm = MO.getImm();
  O << '#' << ((Imm & 256) ? "" : "-") << ((Imm & 0xff) << 2);
}


void ARMInstPrinter::printLdStmModeOperand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MI->getOperand(OpNum)
                                                 .getImm());
  O << ARM_AM::getAMSubModeStr(Mode);
}

void ARMInstPrinter::printAddrMode5Operand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, OpNum, O);
    return;
  }

  O << "[" << getRegisterName(MO1.getReg());

  unsigned ImmOffs = ARM_AM::getAM5Offset(MO2.getImm());
  unsigned Op = ARM_AM::getAM5Op(MO2.getImm());
  if (ImmOffs || Op == ARM_AM::sub) {
    O << ", #"
      << ARM_AM::getAddrOpcStr(ARM_AM::getAM5Op(MO2.getImm()))
      << ImmOffs * 4;
  }
  O << "]";
}

void ARMInstPrinter::printAddrMode6Operand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  O << "[" << getRegisterName(MO1.getReg());
  if (MO2.getImm()) {
    // FIXME: Both darwin as and GNU as violate ARM docs here.
    O << ", :" << (MO2.getImm() << 3);
  }
  O << "]";
}

void ARMInstPrinter::printAddrMode7Operand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  O << "[" << getRegisterName(MO1.getReg()) << "]";
}

void ARMInstPrinter::printAddrMode6OffsetOperand(const MCInst *MI,
                                                 unsigned OpNum,
                                                 raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  if (MO.getReg() == 0)
    O << "!";
  else
    O << ", " << getRegisterName(MO.getReg());
}

void ARMInstPrinter::printBitfieldInvMaskImmOperand(const MCInst *MI,
                                                    unsigned OpNum,
                                                    raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  uint32_t v = ~MO.getImm();
  int32_t lsb = CountTrailingZeros_32(v);
  int32_t width = (32 - CountLeadingZeros_32 (v)) - lsb;
  assert(MO.isImm() && "Not a valid bf_inv_mask_imm value!");
  O << '#' << lsb << ", #" << width;
}

void ARMInstPrinter::printMemBOption(const MCInst *MI, unsigned OpNum,
                                     raw_ostream &O) {
  unsigned val = MI->getOperand(OpNum).getImm();
  O << ARM_MB::MemBOptToString(val);
}

void ARMInstPrinter::printShiftImmOperand(const MCInst *MI, unsigned OpNum,
                                          raw_ostream &O) {
  unsigned ShiftOp = MI->getOperand(OpNum).getImm();
  bool isASR = (ShiftOp & (1 << 5)) != 0;
  unsigned Amt = ShiftOp & 0x1f;
  if (isASR)
    O << ", asr #" << (Amt == 0 ? 32 : Amt);
  else if (Amt)
    O << ", lsl #" << Amt;
}

void ARMInstPrinter::printPKHLSLShiftImm(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNum).getImm();
  if (Imm == 0)
    return;
  assert(Imm > 0 && Imm < 32 && "Invalid PKH shift immediate value!");
  O << ", lsl #" << Imm;
}

void ARMInstPrinter::printPKHASRShiftImm(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNum).getImm();
  // A shift amount of 32 is encoded as 0.
  if (Imm == 0)
    Imm = 32;
  assert(Imm > 0 && Imm <= 32 && "Invalid PKH shift immediate value!");
  O << ", asr #" << Imm;
}

void ARMInstPrinter::printRegisterList(const MCInst *MI, unsigned OpNum,
                                       raw_ostream &O) {
  O << "{";
  for (unsigned i = OpNum, e = MI->getNumOperands(); i != e; ++i) {
    if (i != OpNum) O << ", ";
    O << getRegisterName(MI->getOperand(i).getReg());
  }
  O << "}";
}

void ARMInstPrinter::printSetendOperand(const MCInst *MI, unsigned OpNum,
                                        raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNum);
  if (Op.getImm())
    O << "be";
  else
    O << "le";
}

void ARMInstPrinter::printCPSIMod(const MCInst *MI, unsigned OpNum,
                                  raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNum);
  O << ARM_PROC::IModToString(Op.getImm());
}

void ARMInstPrinter::printCPSIFlag(const MCInst *MI, unsigned OpNum,
                                   raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNum);
  unsigned IFlags = Op.getImm();
  for (int i=2; i >= 0; --i)
    if (IFlags & (1 << i))
      O << ARM_PROC::IFlagsToString(1 << i);
}

void ARMInstPrinter::printMSRMaskOperand(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNum);
  unsigned SpecRegRBit = Op.getImm() >> 4;
  unsigned Mask = Op.getImm() & 0xf;

  // As special cases, CPSR_f, CPSR_s and CPSR_fs prefer printing as
  // APSR_nzcvq, APSR_g and APSRnzcvqg, respectively.
  if (!SpecRegRBit && (Mask == 8 || Mask == 4 || Mask == 12)) {
    O << "APSR_";
    switch (Mask) {
    default: assert(0);
    case 4:  O << "g"; return;
    case 8:  O << "nzcvq"; return;
    case 12: O << "nzcvqg"; return;
    }
    llvm_unreachable("Unexpected mask value!");
  }

  if (SpecRegRBit)
    O << "SPSR";
  else
    O << "CPSR";

  if (Mask) {
    O << '_';
    if (Mask & 8) O << 'f';
    if (Mask & 4) O << 's';
    if (Mask & 2) O << 'x';
    if (Mask & 1) O << 'c';
  }
}

void ARMInstPrinter::printPredicateOperand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  ARMCC::CondCodes CC = (ARMCC::CondCodes)MI->getOperand(OpNum).getImm();
  if (CC != ARMCC::AL)
    O << ARMCondCodeToString(CC);
}

void ARMInstPrinter::printMandatoryPredicateOperand(const MCInst *MI,
                                                    unsigned OpNum,
                                                    raw_ostream &O) {
  ARMCC::CondCodes CC = (ARMCC::CondCodes)MI->getOperand(OpNum).getImm();
  O << ARMCondCodeToString(CC);
}

void ARMInstPrinter::printSBitModifierOperand(const MCInst *MI, unsigned OpNum,
                                              raw_ostream &O) {
  if (MI->getOperand(OpNum).getReg()) {
    assert(MI->getOperand(OpNum).getReg() == ARM::CPSR &&
           "Expect ARM CPSR register!");
    O << 's';
  }
}

void ARMInstPrinter::printNoHashImmediate(const MCInst *MI, unsigned OpNum,
                                          raw_ostream &O) {
  O << MI->getOperand(OpNum).getImm();
}

void ARMInstPrinter::printPImmediate(const MCInst *MI, unsigned OpNum,
                                          raw_ostream &O) {
  O << "p" << MI->getOperand(OpNum).getImm();
}

void ARMInstPrinter::printCImmediate(const MCInst *MI, unsigned OpNum,
                                          raw_ostream &O) {
  O << "c" << MI->getOperand(OpNum).getImm();
}

void ARMInstPrinter::printPCLabel(const MCInst *MI, unsigned OpNum,
                                  raw_ostream &O) {
  llvm_unreachable("Unhandled PC-relative pseudo-instruction!");
}

void ARMInstPrinter::printThumbS4ImmOperand(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O) {
  O << "#" << MI->getOperand(OpNum).getImm() * 4;
}

void ARMInstPrinter::printThumbSRImm(const MCInst *MI, unsigned OpNum,
                                     raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNum).getImm();
  O << "#" << (Imm == 0 ? 32 : Imm);
}

void ARMInstPrinter::printThumbITMask(const MCInst *MI, unsigned OpNum,
                                      raw_ostream &O) {
  // (3 - the number of trailing zeros) is the number of then / else.
  unsigned Mask = MI->getOperand(OpNum).getImm();
  unsigned CondBit0 = Mask >> 4 & 1;
  unsigned NumTZ = CountTrailingZeros_32(Mask);
  assert(NumTZ <= 3 && "Invalid IT mask!");
  for (unsigned Pos = 3, e = NumTZ; Pos > e; --Pos) {
    bool T = ((Mask >> Pos) & 1) == CondBit0;
    if (T)
      O << 't';
    else
      O << 'e';
  }
}

void ARMInstPrinter::printThumbAddrModeRROperand(const MCInst *MI, unsigned Op,
                                                 raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(Op);
  const MCOperand &MO2 = MI->getOperand(Op + 1);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, Op, O);
    return;
  }

  O << "[" << getRegisterName(MO1.getReg());
  if (unsigned RegNum = MO2.getReg())
    O << ", " << getRegisterName(RegNum);
  O << "]";
}

void ARMInstPrinter::printThumbAddrModeImm5SOperand(const MCInst *MI,
                                                    unsigned Op,
                                                    raw_ostream &O,
                                                    unsigned Scale) {
  const MCOperand &MO1 = MI->getOperand(Op);
  const MCOperand &MO2 = MI->getOperand(Op + 1);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, Op, O);
    return;
  }

  O << "[" << getRegisterName(MO1.getReg());
  if (unsigned ImmOffs = MO2.getImm())
    O << ", #" << ImmOffs * Scale;
  O << "]";
}

void ARMInstPrinter::printThumbAddrModeImm5S1Operand(const MCInst *MI,
                                                     unsigned Op,
                                                     raw_ostream &O) {
  printThumbAddrModeImm5SOperand(MI, Op, O, 1);
}

void ARMInstPrinter::printThumbAddrModeImm5S2Operand(const MCInst *MI,
                                                     unsigned Op,
                                                     raw_ostream &O) {
  printThumbAddrModeImm5SOperand(MI, Op, O, 2);
}

void ARMInstPrinter::printThumbAddrModeImm5S4Operand(const MCInst *MI,
                                                     unsigned Op,
                                                     raw_ostream &O) {
  printThumbAddrModeImm5SOperand(MI, Op, O, 4);
}

void ARMInstPrinter::printThumbAddrModeSPOperand(const MCInst *MI, unsigned Op,
                                                 raw_ostream &O) {
  printThumbAddrModeImm5SOperand(MI, Op, O, 4);
}

// Constant shifts t2_so_reg is a 2-operand unit corresponding to the Thumb2
// register with shift forms.
// REG 0   0           - e.g. R5
// REG IMM, SH_OPC     - e.g. R5, LSL #3
void ARMInstPrinter::printT2SOOperand(const MCInst *MI, unsigned OpNum,
                                      raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  unsigned Reg = MO1.getReg();
  O << getRegisterName(Reg);

  // Print the shift opc.
  assert(MO2.isImm() && "Not a valid t2_so_reg value!");
  ARM_AM::ShiftOpc ShOpc = ARM_AM::getSORegShOp(MO2.getImm());
  O << ", " << ARM_AM::getShiftOpcStr(ShOpc);
  if (ShOpc != ARM_AM::rrx)
    O << " #" << translateShiftImm(ARM_AM::getSORegOffset(MO2.getImm()));
}

void ARMInstPrinter::printAddrModeImm12Operand(const MCInst *MI, unsigned OpNum,
                                               raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, OpNum, O);
    return;
  }

  O << "[" << getRegisterName(MO1.getReg());

  int32_t OffImm = (int32_t)MO2.getImm();
  bool isSub = OffImm < 0;
  // Special value for #-0. All others are normal.
  if (OffImm == INT32_MIN)
    OffImm = 0;
  if (isSub)
    O << ", #-" << -OffImm;
  else if (OffImm > 0)
    O << ", #" << OffImm;
  O << "]";
}

void ARMInstPrinter::printT2AddrModeImm8Operand(const MCInst *MI,
                                                unsigned OpNum,
                                                raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  O << "[" << getRegisterName(MO1.getReg());

  int32_t OffImm = (int32_t)MO2.getImm();
  // Don't print +0.
  if (OffImm < 0)
    O << ", #-" << -OffImm;
  else if (OffImm > 0)
    O << ", #" << OffImm;
  O << "]";
}

void ARMInstPrinter::printT2AddrModeImm8s4Operand(const MCInst *MI,
                                                  unsigned OpNum,
                                                  raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  O << "[" << getRegisterName(MO1.getReg());

  int32_t OffImm = (int32_t)MO2.getImm() / 4;
  // Don't print +0.
  if (OffImm < 0)
    O << ", #-" << -OffImm * 4;
  else if (OffImm > 0)
    O << ", #" << OffImm * 4;
  O << "]";
}

void ARMInstPrinter::printT2AddrModeImm0_1020s4Operand(const MCInst *MI,
                                                       unsigned OpNum,
                                                       raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  O << "[" << getRegisterName(MO1.getReg());
  if (MO2.getImm())
    O << ", #" << MO2.getImm() * 4;
  O << "]";
}

void ARMInstPrinter::printT2AddrModeImm8OffsetOperand(const MCInst *MI,
                                                      unsigned OpNum,
                                                      raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  int32_t OffImm = (int32_t)MO1.getImm();
  // Don't print +0.
  if (OffImm < 0)
    O << "#-" << -OffImm;
  else if (OffImm > 0)
    O << "#" << OffImm;
}

void ARMInstPrinter::printT2AddrModeImm8s4OffsetOperand(const MCInst *MI,
                                                        unsigned OpNum,
                                                        raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  int32_t OffImm = (int32_t)MO1.getImm() / 4;
  // Don't print +0.
  if (OffImm < 0)
    O << "#-" << -OffImm * 4;
  else if (OffImm > 0)
    O << "#" << OffImm * 4;
}

void ARMInstPrinter::printT2AddrModeSoRegOperand(const MCInst *MI,
                                                 unsigned OpNum,
                                                 raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);
  const MCOperand &MO3 = MI->getOperand(OpNum+2);

  O << "[" << getRegisterName(MO1.getReg());

  assert(MO2.getReg() && "Invalid so_reg load / store address!");
  O << ", " << getRegisterName(MO2.getReg());

  unsigned ShAmt = MO3.getImm();
  if (ShAmt) {
    assert(ShAmt <= 3 && "Not a valid Thumb2 addressing mode!");
    O << ", lsl #" << ShAmt;
  }
  O << "]";
}

void ARMInstPrinter::printVFPf32ImmOperand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  O << '#';
  if (MO.isFPImm()) {
    O << (float)MO.getFPImm();
  } else {
    union {
      uint32_t I;
      float F;
    } FPUnion;

    FPUnion.I = MO.getImm();
    O << FPUnion.F;
  }
}

void ARMInstPrinter::printVFPf64ImmOperand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  O << '#';
  if (MO.isFPImm()) {
    O << MO.getFPImm();
  } else {
    // We expect the binary encoding of a floating point number here.
    union {
      uint64_t I;
      double D;
    } FPUnion;

    FPUnion.I = MO.getImm();
    O << FPUnion.D;
  }
}

void ARMInstPrinter::printNEONModImmOperand(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O) {
  unsigned EncodedImm = MI->getOperand(OpNum).getImm();
  unsigned EltBits;
  uint64_t Val = ARM_AM::decodeNEONModImm(EncodedImm, EltBits);
  O << "#0x" << utohexstr(Val);
}

void ARMInstPrinter::printImmPlusOneOperand(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNum).getImm();
  O << "#" << Imm + 1;
}

void ARMInstPrinter::printRotImmOperand(const MCInst *MI, unsigned OpNum,
                                        raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNum).getImm();
  if (Imm == 0)
    return;
  O << ", ror #";
  switch (Imm) {
  default: assert (0 && "illegal ror immediate!");
  case 1: O << "8"; break;
  case 2: O << "16"; break;
  case 3: O << "24"; break;
  }
}
