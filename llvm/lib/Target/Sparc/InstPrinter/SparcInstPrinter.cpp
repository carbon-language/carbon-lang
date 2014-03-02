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
#include "Sparc.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

// The generated AsmMatcher SparcGenAsmWriter uses "Sparc" as the target
// namespace. But SPARC backend uses "SP" as its namespace.
namespace llvm {
namespace Sparc {
  using namespace SP;
}
}

#define GET_INSTRUCTION_NAME
#define PRINT_ALIAS_INSTR
#include "SparcGenAsmWriter.inc"

bool SparcInstPrinter::isV9() const {
  return (STI.getFeatureBits() & Sparc::FeatureV9) != 0;
}

void SparcInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const
{
  OS << '%' << StringRef(getRegisterName(RegNo)).lower();
}

void SparcInstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                               StringRef Annot)
{
  if (!printAliasInstr(MI, O) && !printSparcAliasInstr(MI, O))
    printInstruction(MI, O);
  printAnnotation(O, Annot);
}

bool SparcInstPrinter::printSparcAliasInstr(const MCInst *MI, raw_ostream &O)
{
  switch (MI->getOpcode()) {
  default: return false;
  case SP::JMPLrr:
  case SP::JMPLri: {
    if (MI->getNumOperands() != 3)
      return false;
    if (!MI->getOperand(0).isReg())
      return false;
    switch (MI->getOperand(0).getReg()) {
    default: return false;
    case SP::G0: // jmp $addr
      O << "\tjmp "; printMemOperand(MI, 1, O);
      return true;
    case SP::O7: // call $addr
      O << "\tcall "; printMemOperand(MI, 1, O);
      return true;
    }
  }
  case SP::V9FCMPS:
  case SP::V9FCMPD:
  case SP::V9FCMPQ: {
    if (isV9()
        || (MI->getNumOperands() != 3)
        || (!MI->getOperand(0).isReg())
        || (MI->getOperand(0).getReg() != SP::FCC0))
      return false;
    // if V8, skip printing %fcc0.
    switch(MI->getOpcode()) {
    default:
    case SP::V9FCMPS: O << "\tfcmps "; break;
    case SP::V9FCMPD: O << "\tfcmpd "; break;
    case SP::V9FCMPQ: O << "\tfcmpq "; break;
    }
    printOperand(MI, 1, O);
    O << ", ";
    printOperand(MI, 2, O);
    return true;
  }
  }
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
  switch (MI->getOpcode()) {
  default: break;
  case SP::FBCOND:
  case SP::FBCONDA:
  case SP::BPFCC:
  case SP::BPFCCA:
  case SP::BPFCCNT:
  case SP::BPFCCANT:
  case SP::MOVFCCrr:  case SP::V9MOVFCCrr:
  case SP::MOVFCCri:  case SP::V9MOVFCCri:
  case SP::FMOVS_FCC: case SP::V9FMOVS_FCC:
  case SP::FMOVD_FCC: case SP::V9FMOVD_FCC:
  case SP::FMOVQ_FCC: case SP::V9FMOVQ_FCC:
    // Make sure CC is a fp conditional flag.
    CC = (CC < 16) ? (CC + 16) : CC;
    break;
  }
  O << SPARCCondCodeToString((SPCC::CondCodes)CC);
}

bool SparcInstPrinter::printGetPCX(const MCInst *MI, unsigned opNum,
                                  raw_ostream &O)
{
  assert(0 && "FIXME: Implement SparcInstPrinter::printGetPCX.");
  return true;
}
