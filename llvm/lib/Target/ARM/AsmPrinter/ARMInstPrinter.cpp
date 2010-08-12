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
#include "ARM.h" // FIXME: FACTOR ENUMS BETTER.
#include "ARMInstPrinter.h"
#include "ARMAddressingModes.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

// Include the auto-generated portion of the assembly writer.
#define MachineInstr MCInst
#define ARMAsmPrinter ARMInstPrinter  // FIXME: REMOVE.
#include "ARMGenAsmWriter.inc"
#undef MachineInstr
#undef ARMAsmPrinter

static unsigned NextReg(unsigned Reg) {
  switch (Reg) {
  default:
    assert(0 && "Unexpected register enum");

  case ARM::D0:
    return ARM::D1;
  case ARM::D1:
    return ARM::D2;
  case ARM::D2:
    return ARM::D3;
  case ARM::D3:
    return ARM::D4;
  case ARM::D4:
    return ARM::D5;
  case ARM::D5:
    return ARM::D6;
  case ARM::D6:
    return ARM::D7;
  case ARM::D7:
    return ARM::D8;
  case ARM::D8:
    return ARM::D9;
  case ARM::D9:
    return ARM::D10;
  case ARM::D10:
    return ARM::D11;
  case ARM::D11:
    return ARM::D12;
  case ARM::D12:
    return ARM::D13;
  case ARM::D13:
    return ARM::D14;
  case ARM::D14:
    return ARM::D15;
  case ARM::D15:
    return ARM::D16;
  case ARM::D16:
    return ARM::D17;
  case ARM::D17:
    return ARM::D18;
  case ARM::D18:
    return ARM::D19;
  case ARM::D19:
    return ARM::D20;
  case ARM::D20:
    return ARM::D21;
  case ARM::D21:
    return ARM::D22;
  case ARM::D22:
    return ARM::D23;
  case ARM::D23:
    return ARM::D24;
  case ARM::D24:
    return ARM::D25;
  case ARM::D25:
    return ARM::D26;
  case ARM::D26:
    return ARM::D27;
  case ARM::D27:
    return ARM::D28;
  case ARM::D28:
    return ARM::D29;
  case ARM::D29:
    return ARM::D30;
  case ARM::D30:
    return ARM::D31;
  }
}

void ARMInstPrinter::printInst(const MCInst *MI, raw_ostream &O) {
  // Check for MOVs and print canonical forms, instead.
  if (MI->getOpcode() == ARM::MOVs) {
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
  if ((MI->getOpcode() == ARM::STM_UPD || MI->getOpcode() == ARM::t2STM_UPD) &&
      MI->getOperand(0).getReg() == ARM::SP) {
    const MCOperand &MO1 = MI->getOperand(2);
    if (ARM_AM::getAM4SubMode(MO1.getImm()) == ARM_AM::db) {
      O << '\t' << "push";
      printPredicateOperand(MI, 3, O);
      O << '\t';
      printRegisterList(MI, 5, O);
      return;
    }
  }

  // A8.6.122 POP
  if ((MI->getOpcode() == ARM::LDM_UPD || MI->getOpcode() == ARM::t2LDM_UPD) &&
      MI->getOperand(0).getReg() == ARM::SP) {
    const MCOperand &MO1 = MI->getOperand(2);
    if (ARM_AM::getAM4SubMode(MO1.getImm()) == ARM_AM::ia) {
      O << '\t' << "pop";
      printPredicateOperand(MI, 3, O);
      O << '\t';
      printRegisterList(MI, 5, O);
      return;
    }
  }

  // A8.6.355 VPUSH
  if ((MI->getOpcode() == ARM::VSTMS_UPD || MI->getOpcode() ==ARM::VSTMD_UPD) &&
      MI->getOperand(0).getReg() == ARM::SP) {
    const MCOperand &MO1 = MI->getOperand(2);
    if (ARM_AM::getAM5SubMode(MO1.getImm()) == ARM_AM::db) {
      O << '\t' << "vpush";
      printPredicateOperand(MI, 3, O);
      O << '\t';
      printRegisterList(MI, 5, O);
      return;
    }
  }

  // A8.6.354 VPOP
  if ((MI->getOpcode() == ARM::VLDMS_UPD || MI->getOpcode() ==ARM::VLDMD_UPD) &&
      MI->getOperand(0).getReg() == ARM::SP) {
    const MCOperand &MO1 = MI->getOperand(2);
    if (ARM_AM::getAM5SubMode(MO1.getImm()) == ARM_AM::ia) {
      O << '\t' << "vpop";
      printPredicateOperand(MI, 3, O);
      O << '\t';
      printRegisterList(MI, 5, O);
      return;
    }
  }

  printInstruction(MI, O);
 }

void ARMInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &O, const char *Modifier) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    unsigned Reg = Op.getReg();
    if (Modifier && strcmp(Modifier, "dregpair") == 0) {
      O << '{' << getRegisterName(Reg) << ", "
               << getRegisterName(NextReg(Reg)) << '}';
#if 0
      // FIXME: Breaks e.g. ARM/vmul.ll.
      assert(0);
      /*
      unsigned DRegLo = TRI->getSubReg(Reg, ARM::dsub_0);
      unsigned DRegHi = TRI->getSubReg(Reg, ARM::dsub_1);
      O << '{'
      << getRegisterName(DRegLo) << ',' << getRegisterName(DRegHi)
      << '}';*/
#endif
    } else if (Modifier && strcmp(Modifier, "lane") == 0) {
      assert(0);
      /*
      unsigned RegNum = ARMRegisterInfo::getRegisterNumbering(Reg);
      unsigned DReg = TRI->getMatchingSuperReg(Reg, RegNum & 1 ? 2 : 1,
                                               &ARM::DPR_VFP2RegClass);
      O << getRegisterName(DReg) << '[' << (RegNum & 1) << ']';
       */
    } else {
      O << getRegisterName(Reg);
    }
  } else if (Op.isImm()) {
    assert((Modifier && !strcmp(Modifier, "call")) ||
           ((Modifier == 0 || Modifier[0] == 0) && "No modifiers supported"));
    O << '#' << Op.getImm();
  } else {
    if (Modifier && Modifier[0] != 0 && strcmp(Modifier, "call") != 0)
      llvm_unreachable("Unsupported modifier");
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    O << *Op.getExpr();
  }
}

static void printSOImm(raw_ostream &O, int64_t V, bool VerboseAsm,
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
    if (VerboseAsm)
      O << ' ' << MAI->getCommentString()
      << ' ' << (int)ARM_AM::rotr32(Imm, Rot);
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
  printSOImm(O, MO.getImm(), VerboseAsm, &MAI);
}

/// printSOImm2PartOperand - SOImm is broken into two pieces using a 'mov'
/// followed by an 'orr' to materialize.
void ARMInstPrinter::printSOImm2PartOperand(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O) {
  // FIXME: REMOVE this method.
  abort();
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


void ARMInstPrinter::printAddrMode4Operand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O,
                                           const char *Modifier) {
  const MCOperand &MO2 = MI->getOperand(OpNum+1);
  ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MO2.getImm());
  if (Modifier && strcmp(Modifier, "submode") == 0) {
    O << ARM_AM::getAMSubModeStr(Mode);
  } else if (Modifier && strcmp(Modifier, "wide") == 0) {
    ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MO2.getImm());
    if (Mode == ARM_AM::ia)
      O << ".w";
  } else {
    printOperand(MI, OpNum, O);
  }
}

void ARMInstPrinter::printAddrMode5Operand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O,
                                           const char *Modifier) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);
  
  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, OpNum, O);
    return;
  }
  
  if (Modifier && strcmp(Modifier, "submode") == 0) {
    ARM_AM::AMSubMode Mode = ARM_AM::getAM5SubMode(MO2.getImm());
    O << ARM_AM::getAMSubModeStr(Mode);
    return;
  } else if (Modifier && strcmp(Modifier, "base") == 0) {
    // Used for FSTM{D|S} and LSTM{D|S} operations.
    O << getRegisterName(MO1.getReg());
    return;
  }
  
  O << "[" << getRegisterName(MO1.getReg());
  
  if (unsigned ImmOffs = ARM_AM::getAM5Offset(MO2.getImm())) {
    O << ", #"
      << ARM_AM::getAddrOpcStr(ARM_AM::getAM5Op(MO2.getImm()))
      << ImmOffs*4;
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

void ARMInstPrinter::printAddrModePCOperand(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O,
                                            const char *Modifier) {
  assert(0 && "FIXME: Implement printAddrModePCOperand");
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

void ARMInstPrinter::printSatShiftOperand(const MCInst *MI, unsigned OpNum,
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
    assert(0 && "unexpected shift opcode for saturate shift operand");
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



void ARMInstPrinter::printCPInstOperand(const MCInst *MI, unsigned OpNum,
                                        raw_ostream &O,
                                        const char *Modifier) {
  // FIXME: remove this.
  abort();
}

void ARMInstPrinter::printNoHashImmediate(const MCInst *MI, unsigned OpNum,
                                          raw_ostream &O) {
  O << MI->getOperand(OpNum).getImm();
}


void ARMInstPrinter::printPCLabel(const MCInst *MI, unsigned OpNum,
                                  raw_ostream &O) {
  // FIXME: remove this.
  abort();
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
  if (MI->getOpcode() == ARM::t2TBH)
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

void ARMInstPrinter::printT2AddrModeImm12Operand(const MCInst *MI,
                                                 unsigned OpNum,
                                                 raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  O << "[" << getRegisterName(MO1.getReg());

  unsigned OffImm = MO2.getImm();
  if (OffImm)  // Don't print +0.
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
  O << '#' << MI->getOperand(OpNum).getImm();
}

void ARMInstPrinter::printVFPf64ImmOperand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  O << '#' << MI->getOperand(OpNum).getImm();
}

void ARMInstPrinter::printNEONModImmOperand(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O) {
  unsigned EncodedImm = MI->getOperand(OpNum).getImm();
  unsigned EltBits;
  uint64_t Val = ARM_AM::decodeNEONModImm(EncodedImm, EltBits);
  O << "#0x" << utohexstr(Val);
}
