//===-- PPCInstPrinter.cpp - Convert PPC MCInst to assembly syntax --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an PPC MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "PPCInstPrinter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
//#include "llvm/MC/MCAsmInfo.h"
//#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "PPCGenRegisterNames.inc"
#include "PPCGenInstrNames.inc"
using namespace llvm;

#define GET_INSTRUCTION_NAME
#define PPCAsmPrinter PPCInstPrinter
#define MachineInstr MCInst
#include "PPCGenAsmWriter.inc"

StringRef PPCInstPrinter::getOpcodeName(unsigned Opcode) const {
  return getInstructionName(Opcode);
}


void PPCInstPrinter::printInst(const MCInst *MI, raw_ostream &O) {
  // TODO: pseudo ops.
  
  printInstruction(MI, O);
}

void PPCInstPrinter::printS5ImmOperand(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  char Value = MI->getOperand(OpNo).getImm();
  Value = (Value << (32-5)) >> (32-5);
  O << (int)Value;
}

void PPCInstPrinter::printU5ImmOperand(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  unsigned char Value = MI->getOperand(OpNo).getImm();
  assert(Value <= 31 && "Invalid u5imm argument!");
  O << (unsigned int)Value;
}

void PPCInstPrinter::printU6ImmOperand(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  unsigned char Value = MI->getOperand(OpNo).getImm();
  assert(Value <= 63 && "Invalid u6imm argument!");
  O << (unsigned int)Value;
}

void PPCInstPrinter::printS16ImmOperand(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  O << (short)MI->getOperand(OpNo).getImm();
}

void PPCInstPrinter::printU16ImmOperand(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  O << (unsigned short)MI->getOperand(OpNo).getImm();
}

void PPCInstPrinter::printS16X4ImmOperand(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) {
  if (MI->getOperand(OpNo).isImm()) {
    O << (short)(MI->getOperand(OpNo).getImm()*4);
    return;
  }
  
  assert(0 && "Unhandled operand");
#if 0
  O << "lo16(";
  printOp(MI->getOperand(OpNo), O);
  if (TM.getRelocationModel() == Reloc::PIC_)
    O << "-\"L" << getFunctionNumber() << "$pb\")";
  else
    O << ')';
#endif
}


/// stripRegisterPrefix - This method strips the character prefix from a
/// register name so that only the number is left.  Used by for linux asm.
const char *stripRegisterPrefix(const char *RegName) {
  switch (RegName[0]) {
  case 'r':
  case 'f':
  case 'v': return RegName + 1;
  case 'c': if (RegName[1] == 'r') return RegName + 2;
  }
  
  return RegName;
}

void PPCInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    const char *RegName = getRegisterName(Op.getReg());
    // The linux and AIX assembler does not take register prefixes.
    if (!isDarwinSyntax())
      RegName = stripRegisterPrefix(RegName);
    
    O << RegName;
    return;
  }
  
  if (Op.isImm()) {
    O << Op.getImm();
    return;
  }
  
  assert(Op.isExpr() && "unknown operand kind in printOperand");
  O << *Op.getExpr();
}
  
