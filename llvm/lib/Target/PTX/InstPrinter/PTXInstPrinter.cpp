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
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define GET_INSTRUCTION_NAME
#include "PTXGenAsmWriter.inc"

PTXInstPrinter::PTXInstPrinter(const MCAsmInfo &MAI,
                               const MCRegisterInfo &MRI,
                               const MCSubtargetInfo &STI) :
  MCInstPrinter(MAI, MRI) {
  // Initialize the set of available features.
  setAvailableFeatures(STI.getFeatureBits());
}

StringRef PTXInstPrinter::getOpcodeName(unsigned Opcode) const {
  return getInstructionName(Opcode);
}

void PTXInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  // Decode the register number into type and offset
  unsigned RegSpace  = RegNo & 0x7;
  unsigned RegType   = (RegNo >> 3) & 0x7;
  unsigned RegOffset = RegNo >> 6;

  // Print the register
  OS << "%";

  switch (RegSpace) {
  default:
    llvm_unreachable("Unknown register space!");
  case PTXRegisterSpace::Reg:
    switch (RegType) {
    default:
      llvm_unreachable("Unknown register type!");
    case PTXRegisterType::Pred:
      OS << "p";
      break;
    case PTXRegisterType::B16:
      OS << "rh";
      break;
    case PTXRegisterType::B32:
      OS << "r";
      break;
    case PTXRegisterType::B64:
      OS << "rd";
      break;
    case PTXRegisterType::F32:
      OS << "f";
      break;
    case PTXRegisterType::F64:
      OS << "fd";
      break;
    }
    break;
  case PTXRegisterSpace::Return:
    OS << "ret";
    break;
  case PTXRegisterSpace::Argument:
    OS << "arg";
    break;
  }

  OS << RegOffset;
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
  if (PredOp == PTXPredicate::None)
    return;

  if (PredOp == PTXPredicate::Negate)
    O << '!';
  else
    O << '@';

  printOperand(MI, RegIndex, O);
}

void PTXInstPrinter::printCall(const MCInst *MI, raw_ostream &O) {
  O << "\tcall.uni\t";
  // The first two operands are the predicate slot
  unsigned Index = 2;
  unsigned NumRets = MI->getOperand(Index++).getImm();

  if (NumRets > 0) {
    O << "(";
    printOperand(MI, Index++, O);
    for (unsigned i = 1; i < NumRets; ++i) {
      O << ", ";
      printOperand(MI, Index++, O);
    }
    O << "), ";
  }

  const MCExpr* Expr = MI->getOperand(Index++).getExpr();
  unsigned NumArgs = MI->getOperand(Index++).getImm();
  
  // if the function call is to printf or puts, change to vprintf
  if (const MCSymbolRefExpr *SymRefExpr = dyn_cast<MCSymbolRefExpr>(Expr)) {
    const MCSymbol &Sym = SymRefExpr->getSymbol();
    if (Sym.getName() == "printf" || Sym.getName() == "puts") {
      O << "vprintf";
    } else {
      O << Sym.getName();
    }
  } else {
    O << *Expr;
  }
  
  O << ", (";

  if (NumArgs > 0) {
    printOperand(MI, Index++, O);
    for (unsigned i = 1; i < NumArgs; ++i) {
      O << ", ";
      printOperand(MI, Index++, O);
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
  } else if (Op.isReg()) {
    printRegName(O, Op.getReg());
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
  // By definition, operand OpNo+1 is an i32imm
  const MCOperand &Op2 = MI->getOperand(OpNo+1);
  printOperand(MI, OpNo, O);
  if (Op2.getImm() == 0)
    return; // don't print "+0"
  O << "+" << Op2.getImm();
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

