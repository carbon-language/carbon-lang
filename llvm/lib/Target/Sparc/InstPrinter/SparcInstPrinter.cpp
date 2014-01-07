//===-- SparcInstPrinter.cpp - Convert Sparc MCInst to assembly syntax -----==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an Sparc MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "SparcInstPrinter.h"
#include "MCTargetDesc/SparcBaseInfo.h"
#include "Sparc.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define GET_INSTRUCTION_NAME
#include "SparcGenAsmWriter.inc"

void SparcInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const
{
  OS << '%' << StringRef(getRegisterName(RegNo)).lower();
}

void SparcInstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                               StringRef Annot)
{
  printInstruction(MI, O);
  printAnnotation(O, Annot);
}

void SparcInstPrinter::printOperand(const MCInst *MI, int opNum,
                                    raw_ostream &O)
{
  const MCOperand &MO = MI->getOperand (opNum);

  if (MO.isReg()) {
    printRegName(O, MO.getReg());
    return ;
  }

  if (MO.isImm()) {
    O << (int)MO.getImm();
    return;
  }

  assert(MO.isExpr() && "Unknown operand kind in printOperand");
  MO.getExpr()->print(O);
}

void SparcInstPrinter::printMemOperand(const MCInst *MI, int opNum,
                                      raw_ostream &O, const char *Modifier)
{
  printOperand(MI, opNum, O);

  // If this is an ADD operand, emit it like normal operands.
  if (Modifier && !strcmp(Modifier, "arith")) {
    O << ", ";
    printOperand(MI, opNum+1, O);
    return;
  }
  const MCOperand &MO = MI->getOperand(opNum+1);

  if (MO.isReg() && MO.getReg() == SP::G0)
    return;   // don't print "+%g0"
  if (MO.isImm() && MO.getImm() == 0)
    return;   // don't print "+0"

  O << "+";

  printOperand(MI, opNum+1, O);
}

void SparcInstPrinter::printCCOperand(const MCInst *MI, int opNum,
                                     raw_ostream &O)
{
  int CC = (int)MI->getOperand(opNum).getImm();
  O << SPARCCondCodeToString((SPCC::CondCodes)CC);
}

bool SparcInstPrinter::printGetPCX(const MCInst *MI, unsigned opNum,
                                  raw_ostream &O)
{
  assert(0 && "FIXME: Implement SparcInstPrinter::printGetPCX.");
  return true;
}
