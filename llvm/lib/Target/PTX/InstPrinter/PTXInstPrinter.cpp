//===-- PTXInstPrinter.cpp - Convert PTX MCInst to assembly syntax --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints a PTX MCInst to a .ptx file.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "PTXInstPrinter.h"
#include "MCTargetDesc/PTXBaseInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define GET_INSTRUCTION_NAME
#include "PTXGenAsmWriter.inc"

PTXInstPrinter::PTXInstPrinter(const MCAsmInfo &MAI,
                               const MCSubtargetInfo &STI) :
  MCInstPrinter(MAI) {
  // Initialize the set of available features.
  setAvailableFeatures(STI.getFeatureBits());
}

StringRef PTXInstPrinter::getOpcodeName(unsigned Opcode) const {
  return getInstructionName(Opcode);
}

void PTXInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  OS << getRegisterName(RegNo);
}

void PTXInstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                               StringRef Annot) {
  printPredicate(MI, O);
  switch (MI->getOpcode()) {
  default:
    printInstruction(MI, O);
    break;
  case PTX::CALL:
    printCall(MI, O);
  }
  O << ";";
  printAnnotation(O, Annot);
}

void PTXInstPrinter::printPredicate(const MCInst *MI, raw_ostream &O) {
  // The last two operands are the predicate operands
  int RegIndex;
  int OpIndex;

  if (MI->getOpcode() == PTX::CALL) {
    RegIndex = 0;
    OpIndex  = 1;
  } else {
    RegIndex = MI->getNumOperands()-2;
    OpIndex = MI->getNumOperands()-1;
  }

  int PredOp = MI->getOperand(OpIndex).getImm();
  if (PredOp != PTX::PRED_NONE) {
    if (PredOp == PTX::PRED_NEGATE) {
      O << '!';
    } else {
      O << '@';
    }
    printOperand(MI, RegIndex, O);
  }
}

void PTXInstPrinter::printCall(const MCInst *MI, raw_ostream &O) {
  O << "\tcall.uni\t";
  // The first two operands are the predicate slot
  unsigned Index = 2;
  unsigned NumRets = MI->getOperand(Index++).getImm();
  for (unsigned i = 0; i < NumRets; ++i) {
    if (i == 0) {
      O << "(";
    } else {
      O << ", ";
    }
    printOperand(MI, Index++, O);
  }

  if (NumRets > 0) {
    O << "), ";
  }

  O << *(MI->getOperand(Index++).getExpr()) << ", (";

  unsigned NumArgs = MI->getOperand(Index++).getImm();
  for (unsigned i = 0; i < NumArgs; ++i) {
    printOperand(MI, Index++, O);
    if (i < NumArgs-1) {
      O << ", ";
    }
  }

  O << ")";
}

void PTXInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm()) {
    O << Op.getImm();
  } else if (Op.isFPImm()) {
    double Imm = Op.getFPImm();
    APFloat FPImm(Imm);
    APInt FPIntImm = FPImm.bitcastToAPInt();
    O << "0D";
    // PTX requires us to output the full 64 bits, even if the number is zero
    if (FPIntImm.getZExtValue() > 0) {
      O << FPIntImm.toString(16, false);
    } else {
      O << "0000000000000000";
    }
  } else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    const MCExpr *Expr = Op.getExpr();
    if (const MCSymbolRefExpr *SymRefExpr = dyn_cast<MCSymbolRefExpr>(Expr)) {
      const MCSymbol &Sym = SymRefExpr->getSymbol();
      O << Sym.getName();
    } else {
      O << *Op.getExpr();
    }
  }
}

void PTXInstPrinter::printMemOperand(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O) {
  printOperand(MI, OpNo, O);
  if (MI->getOperand(OpNo+1).isImm() && MI->getOperand(OpNo+1).getImm() == 0)
    return; // don't print "+0"
  O << "+";
  printOperand(MI, OpNo+1, O);
}

void PTXInstPrinter::printRoundingMode(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  assert (Op.isImm() && "Rounding modes must be immediate values");
  switch (Op.getImm()) {
  default:
    llvm_unreachable("Unknown rounding mode!");
  case PTXRoundingMode::RndDefault:
    llvm_unreachable("FP rounding-mode pass did not handle instruction!");
    break;
  case PTXRoundingMode::RndNone:
    // Do not print anything.
    break;
  case PTXRoundingMode::RndNearestEven:
    O << ".rn";
    break;
  case PTXRoundingMode::RndTowardsZero:
    O << ".rz";
    break;
  case PTXRoundingMode::RndNegInf:
    O << ".rm";
    break;
  case PTXRoundingMode::RndPosInf:
    O << ".rp";
    break;
  case PTXRoundingMode::RndApprox:
    O << ".approx";
    break;
  case PTXRoundingMode::RndNearestEvenInt:
    O << ".rni";
    break;
  case PTXRoundingMode::RndTowardsZeroInt:
    O << ".rzi";
    break;
  case PTXRoundingMode::RndNegInfInt:
    O << ".rmi";
    break;
  case PTXRoundingMode::RndPosInfInt:
    O << ".rpi";
    break;
  }
}

