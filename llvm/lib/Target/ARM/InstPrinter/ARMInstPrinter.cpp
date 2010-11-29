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
#include "ARMBaseInfo.h"
#include "ARMInstPrinter.h"
#include "ARMAddressingModes.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define GET_INSTRUCTION_NAME
#include "ARMGenAsmWriter.inc"

StringRef ARMInstPrinter::getOpcodeName(unsigned Opcode) const {
  return getInstructionName(Opcode);
}


void ARMInstPrinter::printInst(const MCInst *MI, raw_ostream &O) {
  unsigned Opcode = MI->getOpcode();

  // Check for MOVs and print canonical forms, instead.
  if (Opcode == ARM::MOVs) {
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

    if (ARM_AM::getSORegShOp(MO3.getImm()) == ARM_AM::rrx)
      return;

    O << ", ";

    if (MO2.getReg()) {
      O << getRegisterName(MO2.getReg());
      assert(ARM_AM::getSORegOffset(MO3.getImm()) == 0);
    } else {
      O << "#" << ARM_AM::getSORegOffset(MO3.getImm());
    }
    return;
  }

  // A8.6.123 PUSH
  if ((Opcode == ARM::STMDB_UPD || Opcode == ARM::t2STMDB_UPD) &&
      MI->getOperand(0).getReg() == ARM::SP) {
    O << '\t' << "push";
    printPredicateOperand(MI, 2, O);
    O << '\t';
    printRegisterList(MI, 4, O);
    return;
  }

  // A8.6.122 POP
  if ((Opcode == ARM::LDMIA_UPD || Opcode == ARM::t2LDMIA_UPD) &&
      MI->getOperand(0).getReg() == ARM::SP) {
    O << '\t' << "pop";
    printPredicateOperand(MI, 2, O);
    O << '\t';
    printRegisterList(MI, 4, O);
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

static void printSOImm(raw_ostream &O, int64_t V, raw_ostream *CommentStream,
                       const MCAsmInfo *MAI) {
  // Break it up into two parts that make up a shifter immediate.
  V = ARM_AM::getSOImmVal(V);
  assert(V != -1 && "Not a valid so_imm value!");

  unsigned Imm = ARM_AM::getSOImmValImm(V);
  unsigned Rot = ARM_AM::getSOImmValRot(V);

  // Print low-level immediate formation info, per
  // A5.1.3: "Data-processing operands - Immediate".
  if (Rot) {
    O << "#" << Imm << ", " << Rot;
    // Pretty printed version.
    if (CommentStream)
      *CommentStream << (int)ARM_AM::rotr32(Imm, Rot) << "\n";
  } else {
    O << "#" << Imm;
  }
}


/// printSOImmOperand - SOImm is 4-bit rotate amount in bits 8-11 with 8-bit
/// immediate in bits 0-7.
void ARMInstPrinter::printSOImmOperand(const MCInst *MI, unsigned OpNum,
                                       raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  assert(MO.isImm() && "Not a valid so_imm value!");
  printSOImm(O, MO.getImm(), CommentStream, &MAI);
}

// so_reg is a 4-operand unit corresponding to register forms of the A5.1
// "Addressing Mode 1 - Data-processing operands" forms.  This includes:
//    REG 0   0           - e.g. R5
//    REG REG 0,SH_OPC    - e.g. R5, ROR R3
//    REG 0   IMM,SH_OPC  - e.g. R5, LSL #3
void ARMInstPrinter::printSORegOperand(const MCInst *MI, unsigned OpNum,
                                       raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);
  const MCOperand &MO3 = MI->getOperand(OpNum+2);

  O << getRegisterName(MO1.getReg());

  // Print the shift opc.
  ARM_AM::ShiftOpc ShOpc = ARM_AM::getSORegShOp(MO3.getImm());
  O << ", " << ARM_AM::getShiftOpcStr(ShOpc);
  if (MO2.getReg()) {
    O << ' ' << getRegisterName(MO2.getReg());
    assert(ARM_AM::getSORegOffset(MO3.getImm()) == 0);
  } else if (ShOpc != ARM_AM::rrx) {
    O << " #" << ARM_AM::getSORegOffset(MO3.getImm());
  }
}


void ARMInstPrinter::printAddrMode2Operand(const MCInst *MI, unsigned Op,
                                           raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(Op);
  const MCOperand &MO2 = MI->getOperand(Op+1);
  const MCOperand &MO3 = MI->getOperand(Op+2);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, Op, O);
    return;
  }

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

void ARMInstPrinter::printAddrMode3Operand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);
  const MCOperand &MO3 = MI->getOperand(OpNum+2);

  O << '[' << getRegisterName(MO1.getReg());

  if (MO2.getReg()) {
    O << ", " << (char)ARM_AM::getAM3Op(MO3.getImm())
      << getRegisterName(MO2.getReg()) << ']';
    return;
  }

  if (unsigned ImmOffs = ARM_AM::getAM3Offset(MO3.getImm()))
    O << ", #"
      << ARM_AM::getAddrOpcStr(ARM_AM::getAM3Op(MO3.getImm()))
      << ImmOffs;
  O << ']';
}

void ARMInstPrinter::printAddrMode3OffsetOperand(const MCInst *MI,
                                                 unsigned OpNum,
                                                 raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  if (MO1.getReg()) {
    O << (char)ARM_AM::getAM3Op(MO2.getImm())
    << getRegisterName(MO1.getReg());
    return;
  }

  unsigned ImmOffs = ARM_AM::getAM3Offset(MO2.getImm());
  O << '#'
    << ARM_AM::getAddrOpcStr(ARM_AM::getAM3Op(MO2.getImm()))
    << ImmOffs;
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

  if (unsigned ImmOffs = ARM_AM::getAM5Offset(MO2.getImm())) {
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
  ARM_AM::ShiftOpc Opc = ARM_AM::getSORegShOp(ShiftOp);
  switch (Opc) {
  case ARM_AM::no_shift:
    return;
  case ARM_AM::lsl:
    O << ", lsl #";
    break;
  case ARM_AM::asr:
    O << ", asr #";
    break;
  default:
    assert(0 && "unexpected shift opcode for shift immediate operand");
  }
  O << ARM_AM::getSORegOffset(ShiftOp);
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

void ARMInstPrinter::printCPSOptionOperand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNum);
  unsigned option = Op.getImm();
  unsigned mode = option & 31;
  bool changemode = option >> 5 & 1;
  unsigned AIF = option >> 6 & 7;
  unsigned imod = option >> 9 & 3;
  if (imod == 2)
    O << "ie";
  else if (imod == 3)
    O << "id";
  O << '\t';
  if (imod > 1) {
    if (AIF & 4) O << 'a';
    if (AIF & 2) O << 'i';
    if (AIF & 1) O << 'f';
    if (AIF > 0 && changemode) O << ", ";
  }
  if (changemode)
    O << '#' << mode;
}

void ARMInstPrinter::printMSRMaskOperand(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNum);
  unsigned Mask = Op.getImm();
  if (Mask) {
    O << '_';
    if (Mask & 8) O << 'f';
    if (Mask & 4) O << 's';
    if (Mask & 2) O << 'x';
    if (Mask & 1) O << 'c';
  }
}

void ARMInstPrinter::printNegZeroOperand(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNum);
  O << '#';
  if (Op.getImm() < 0)
    O << '-' << (-Op.getImm() - 1);
  else
    O << Op.getImm();
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

void ARMInstPrinter::printPCLabel(const MCInst *MI, unsigned OpNum,
                                  raw_ostream &O) {
  llvm_unreachable("Unhandled PC-relative pseudo-instruction!");
}

void ARMInstPrinter::printThumbS4ImmOperand(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O) {
  O << "#" <<  MI->getOperand(OpNum).getImm() * 4;
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
  const MCOperand &MO2 = MI->getOperand(Op+1);
  O << "[" << getRegisterName(MO1.getReg());
  O << ", " << getRegisterName(MO2.getReg()) << "]";
}

void ARMInstPrinter::printThumbAddrModeRI5Operand(const MCInst *MI, unsigned Op,
                                                  raw_ostream &O,
                                                  unsigned Scale) {
  const MCOperand &MO1 = MI->getOperand(Op);
  const MCOperand &MO2 = MI->getOperand(Op+1);
  const MCOperand &MO3 = MI->getOperand(Op+2);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, Op, O);
    return;
  }

  O << "[" << getRegisterName(MO1.getReg());
  if (MO3.getReg())
    O << ", " << getRegisterName(MO3.getReg());
  else if (unsigned ImmOffs = MO2.getImm())
    O << ", #" << ImmOffs * Scale;
  O << "]";
}

void ARMInstPrinter::printThumbAddrModeS1Operand(const MCInst *MI, unsigned Op,
                                                 raw_ostream &O) {
  printThumbAddrModeRI5Operand(MI, Op, O, 1);
}

void ARMInstPrinter::printThumbAddrModeS2Operand(const MCInst *MI, unsigned Op,
                                                 raw_ostream &O) {
  printThumbAddrModeRI5Operand(MI, Op, O, 2);
}

void ARMInstPrinter::printThumbAddrModeS4Operand(const MCInst *MI, unsigned Op,
                                                 raw_ostream &O) {
  printThumbAddrModeRI5Operand(MI, Op, O, 4);
}

void ARMInstPrinter::printThumbAddrModeSPOperand(const MCInst *MI, unsigned Op,
                                                 raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(Op);
  const MCOperand &MO2 = MI->getOperand(Op+1);
  O << "[" << getRegisterName(MO1.getReg());
  if (unsigned ImmOffs = MO2.getImm())
    O << ", #" << ImmOffs*4;
  O << "]";
}

void ARMInstPrinter::printTBAddrMode(const MCInst *MI, unsigned OpNum,
                                     raw_ostream &O) {
  O << "[pc, " << getRegisterName(MI->getOperand(OpNum).getReg());
  if (MI->getOpcode() == ARM::t2TBH_JT)
    O << ", lsl #1";
  O << ']';
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
    O << " #" << ARM_AM::getSORegOffset(MO2.getImm());
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
  O << '#' << (float)MI->getOperand(OpNum).getFPImm();
}

void ARMInstPrinter::printVFPf64ImmOperand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  O << '#' << MI->getOperand(OpNum).getFPImm();
}

void ARMInstPrinter::printNEONModImmOperand(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O) {
  unsigned EncodedImm = MI->getOperand(OpNum).getImm();
  unsigned EltBits;
  uint64_t Val = ARM_AM::decodeNEONModImm(EncodedImm, EltBits);
  O << "#0x" << utohexstr(Val);
}
