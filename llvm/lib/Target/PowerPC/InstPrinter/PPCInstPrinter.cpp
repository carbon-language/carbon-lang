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
  
