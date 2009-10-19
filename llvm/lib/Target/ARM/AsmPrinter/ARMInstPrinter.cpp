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
#include "ARMAddressingModes.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/raw_ostream.h"
#include "ARMGenInstrNames.inc"
#include "ARMGenRegisterNames.inc"
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
  // FIXME: TURN ASSERT ON.
  //assert((Modifier == 0 || Modifier[0] == 0) && "Cannot print modifiers");
  
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    O << getRegisterName(Op.getReg());
  } else if (Op.isImm()) {
    O << '#' << Op.getImm();
  } else {
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


void ARMInstPrinter::printAddrMode4Operand(const MCInst *MI, unsigned OpNum,
                                           const char *Modifier) {
  // FIXME: ENABLE assert.
  //assert((Modifier == 0 || Modifier[0] == 0) && "Cannot print modifiers");
  
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);
  ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MO2.getImm());
  if (0 && Modifier && strcmp(Modifier, "submode") == 0) {
    if (MO1.getReg() == ARM::SP) {
      // FIXME
      bool isLDM = (MI->getOpcode() == ARM::LDM ||
                    MI->getOpcode() == ARM::LDM_RET ||
                    MI->getOpcode() == ARM::t2LDM ||
                    MI->getOpcode() == ARM::t2LDM_RET);
      O << ARM_AM::getAMSubModeAltStr(Mode, isLDM);
    } else
      O << ARM_AM::getAMSubModeStr(Mode);
  } else if (0 && Modifier && strcmp(Modifier, "wide") == 0) {
    ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MO2.getImm());
    if (Mode == ARM_AM::ia)
      O << ".w";
  } else {
    printOperand(MI, OpNum);
    if (ARM_AM::getAM4WBFlag(MO2.getImm()))
      O << "!";
  }
}

void ARMInstPrinter::printRegisterList(const MCInst *MI, unsigned OpNum) {
  O << "{";
  // Always skip the first operand, it's the optional (and implicit writeback).
  for (unsigned i = OpNum+1, e = MI->getNumOperands(); i != e; ++i) {
#if 0 // FIXME: HANDLE WHEN LOWERING??
    if (MI->getOperand(i).isImplicit())
      continue;
#endif
    if (i != OpNum+1) O << ", ";
    
    O << getRegisterName(MI->getOperand(i).getReg());
  }
  O << "}";
}


void ARMInstPrinter::printCPInstOperand(const MCInst *MI, unsigned OpNum,
                                        const char *Modifier) {
  // FIXME: remove this.
  abort();
}

void ARMInstPrinter::printPCLabel(const MCInst *MI, unsigned OpNum) {
  // FIXME: remove this.
  abort();
}
