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
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

// Include the auto-generated portion of the assembly writer.
#define MachineInstr MCInst
#define ARMAsmPrinter ARMInstPrinter  // FIXME: REMOVE.
#define NO_ASM_WRITER_BOILERPLATE
#include "ARMGenAsmWriter.inc"
#undef MachineInstr
#undef ARMAsmPrinter

void ARMInstPrinter::printInst(const MCInst *MI) { printInstruction(MI); }

void ARMInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                  const char *Modifier) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    unsigned Reg = Op.getReg();
    if (Modifier && strcmp(Modifier, "dregpair") == 0) {
      // FIXME: Breaks e.g. ARM/vmul.ll.
      assert(0);
      /*
      unsigned DRegLo = TRI->getSubReg(Reg, 5); // arm_dsubreg_0
      unsigned DRegHi = TRI->getSubReg(Reg, 6); // arm_dsubreg_1
      O << '{'
      << getRegisterName(DRegLo) << ',' << getRegisterName(DRegHi)
      << '}';*/
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
    assert((Modifier == 0 || Modifier[0] == 0) && "No modifiers supported");
    O << '#' << Op.getImm();
  } else {
    assert((Modifier == 0 || Modifier[0] == 0) && "No modifiers supported");
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    Op.getExpr()->print(O, &MAI);
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
void ARMInstPrinter::printSOImmOperand(const MCInst *MI, unsigned OpNum) {
  const MCOperand &MO = MI->getOperand(OpNum);
  assert(MO.isImm() && "Not a valid so_imm value!");
  printSOImm(O, MO.getImm(), VerboseAsm, &MAI);
}

/// printSOImm2PartOperand - SOImm is broken into two pieces using a 'mov'
/// followed by an 'orr' to materialize.
void ARMInstPrinter::printSOImm2PartOperand(const MCInst *MI, unsigned OpNum) {
  // FIXME: REMOVE this method.
  abort();
}

// so_reg is a 4-operand unit corresponding to register forms of the A5.1
// "Addressing Mode 1 - Data-processing operands" forms.  This includes:
//    REG 0   0           - e.g. R5
//    REG REG 0,SH_OPC    - e.g. R5, ROR R3
//    REG 0   IMM,SH_OPC  - e.g. R5, LSL #3
void ARMInstPrinter::printSORegOperand(const MCInst *MI, unsigned OpNum) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);
  const MCOperand &MO3 = MI->getOperand(OpNum+2);
  
  O << getRegisterName(MO1.getReg());
  
  // Print the shift opc.
  O << ", "
    << ARM_AM::getShiftOpcStr(ARM_AM::getSORegShOp(MO3.getImm()))
    << ' ';
  
  if (MO2.getReg()) {
    O << getRegisterName(MO2.getReg());
    assert(ARM_AM::getSORegOffset(MO3.getImm()) == 0);
  } else {
    O << "#" << ARM_AM::getSORegOffset(MO3.getImm());
  }
}


void ARMInstPrinter::printAddrMode2Operand(const MCInst *MI, unsigned Op) {
  const MCOperand &MO1 = MI->getOperand(Op);
  const MCOperand &MO2 = MI->getOperand(Op+1);
  const MCOperand &MO3 = MI->getOperand(Op+2);
  
  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, Op);
    return;
  }
  
  O << "[" << getRegisterName(MO1.getReg());
  
  if (!MO2.getReg()) {
    if (ARM_AM::getAM2Offset(MO3.getImm()))  // Don't print +0.
      O << ", #"
      << (char)ARM_AM::getAM2Op(MO3.getImm())
      << ARM_AM::getAM2Offset(MO3.getImm());
    O << "]";
    return;
  }
  
  O << ", "
  << (char)ARM_AM::getAM2Op(MO3.getImm())
  << getRegisterName(MO2.getReg());
  
  if (unsigned ShImm = ARM_AM::getAM2Offset(MO3.getImm()))
    O << ", "
    << ARM_AM::getShiftOpcStr(ARM_AM::getAM2ShiftOpc(MO3.getImm()))
    << " #" << ShImm;
  O << "]";
}  

void ARMInstPrinter::printAddrMode2OffsetOperand(const MCInst *MI,
                                                 unsigned OpNum) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);
  
  if (!MO1.getReg()) {
    unsigned ImmOffs = ARM_AM::getAM2Offset(MO2.getImm());
    assert(ImmOffs && "Malformed indexed load / store!");
    O << '#' << (char)ARM_AM::getAM2Op(MO2.getImm()) << ImmOffs;
    return;
  }
  
  O << (char)ARM_AM::getAM2Op(MO2.getImm()) << getRegisterName(MO1.getReg());
  
  if (unsigned ShImm = ARM_AM::getAM2Offset(MO2.getImm()))
    O << ", "
    << ARM_AM::getShiftOpcStr(ARM_AM::getAM2ShiftOpc(MO2.getImm()))
    << " #" << ShImm;
}

void ARMInstPrinter::printAddrMode3Operand(const MCInst *MI, unsigned OpNum) {
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
    << (char)ARM_AM::getAM3Op(MO3.getImm())
    << ImmOffs;
  O << ']';
}

void ARMInstPrinter::printAddrMode3OffsetOperand(const MCInst *MI,
                                                 unsigned OpNum) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);
  
  if (MO1.getReg()) {
    O << (char)ARM_AM::getAM3Op(MO2.getImm())
    << getRegisterName(MO1.getReg());
    return;
  }
  
  unsigned ImmOffs = ARM_AM::getAM3Offset(MO2.getImm());
  assert(ImmOffs && "Malformed indexed load / store!");
  O << "#"
  << (char)ARM_AM::getAM3Op(MO2.getImm())
  << ImmOffs;
}


void ARMInstPrinter::printAddrMode4Operand(const MCInst *MI, unsigned OpNum,
                                           const char *Modifier) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);
  ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MO2.getImm());
  if (Modifier && strcmp(Modifier, "submode") == 0) {
    if (MO1.getReg() == ARM::SP) {
      // FIXME
      bool isLDM = (MI->getOpcode() == ARM::LDM ||
                    MI->getOpcode() == ARM::LDM_RET ||
                    MI->getOpcode() == ARM::t2LDM ||
                    MI->getOpcode() == ARM::t2LDM_RET);
      O << ARM_AM::getAMSubModeAltStr(Mode, isLDM);
    } else
      O << ARM_AM::getAMSubModeStr(Mode);
  } else if (Modifier && strcmp(Modifier, "wide") == 0) {
    ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MO2.getImm());
    if (Mode == ARM_AM::ia)
      O << ".w";
  } else {
    printOperand(MI, OpNum);
    if (ARM_AM::getAM4WBFlag(MO2.getImm()))
      O << "!";
  }
}

void ARMInstPrinter::printAddrMode5Operand(const MCInst *MI, unsigned OpNum,
                                           const char *Modifier) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);
  
  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, OpNum);
    return;
  }
  
  if (Modifier && strcmp(Modifier, "submode") == 0) {
    ARM_AM::AMSubMode Mode = ARM_AM::getAM5SubMode(MO2.getImm());
    if (MO1.getReg() == ARM::SP) {
      bool isFLDM = (MI->getOpcode() == ARM::FLDMD ||
                     MI->getOpcode() == ARM::FLDMS);
      O << ARM_AM::getAMSubModeAltStr(Mode, isFLDM);
    } else
      O << ARM_AM::getAMSubModeStr(Mode);
    return;
  } else if (Modifier && strcmp(Modifier, "base") == 0) {
    // Used for FSTM{D|S} and LSTM{D|S} operations.
    O << getRegisterName(MO1.getReg());
    if (ARM_AM::getAM5WBFlag(MO2.getImm()))
      O << "!";
    return;
  }
  
  O << "[" << getRegisterName(MO1.getReg());
  
  if (unsigned ImmOffs = ARM_AM::getAM5Offset(MO2.getImm())) {
    O << ", #"
      << (char)ARM_AM::getAM5Op(MO2.getImm())
      << ImmOffs*4;
  }
  O << "]";
}

void ARMInstPrinter::printAddrMode6Operand(const MCInst *MI, unsigned OpNum) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);
  const MCOperand &MO3 = MI->getOperand(OpNum+2);
  
  // FIXME: No support yet for specifying alignment.
  O << '[' << getRegisterName(MO1.getReg()) << ']';
  
  if (ARM_AM::getAM6WBFlag(MO3.getImm())) {
    if (MO2.getReg() == 0)
      O << '!';
    else
      O << ", " << getRegisterName(MO2.getReg());
  }
}

void ARMInstPrinter::printAddrModePCOperand(const MCInst *MI, unsigned OpNum,
                                            const char *Modifier) {
  assert(0 && "FIXME: Implement printAddrModePCOperand");
}

void ARMInstPrinter::printBitfieldInvMaskImmOperand (const MCInst *MI,
                                                     unsigned OpNum) {
  const MCOperand &MO = MI->getOperand(OpNum);
  uint32_t v = ~MO.getImm();
  int32_t lsb = CountTrailingZeros_32(v);
  int32_t width = (32 - CountLeadingZeros_32 (v)) - lsb;
  assert(MO.isImm() && "Not a valid bf_inv_mask_imm value!");
  O << '#' << lsb << ", #" << width;
}

void ARMInstPrinter::printRegisterList(const MCInst *MI, unsigned OpNum) {
  O << "{";
  // Always skip the first operand, it's the optional (and implicit writeback).
  for (unsigned i = OpNum+1, e = MI->getNumOperands(); i != e; ++i) {
    if (i != OpNum+1) O << ", ";
    O << getRegisterName(MI->getOperand(i).getReg());
  }
  O << "}";
}

void ARMInstPrinter::printPredicateOperand(const MCInst *MI, unsigned OpNum) {
  ARMCC::CondCodes CC = (ARMCC::CondCodes)MI->getOperand(OpNum).getImm();
  if (CC != ARMCC::AL)
    O << ARMCondCodeToString(CC);
}

void ARMInstPrinter::printSBitModifierOperand(const MCInst *MI, unsigned OpNum){
  if (MI->getOperand(OpNum).getReg()) {
    assert(MI->getOperand(OpNum).getReg() == ARM::CPSR &&
           "Expect ARM CPSR register!");
    O << 's';
  }
}



void ARMInstPrinter::printCPInstOperand(const MCInst *MI, unsigned OpNum,
                                        const char *Modifier) {
  // FIXME: remove this.
  abort();
}

void ARMInstPrinter::printNoHashImmediate(const MCInst *MI, unsigned OpNum) {
  O << MI->getOperand(OpNum).getImm();
}


void ARMInstPrinter::printPCLabel(const MCInst *MI, unsigned OpNum) {
  // FIXME: remove this.
  abort();
}
