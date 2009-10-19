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
