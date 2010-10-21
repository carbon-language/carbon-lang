//===-- MBlazeInstPrinter.cpp - Convert MBlaze MCInst to assembly syntax --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an MBlaze MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "MBlaze.h"
#include "MBlazeInstPrinter.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;


// Include the auto-generated portion of the assembly writer.
#include "MBlazeGenAsmWriter.inc"

void MBlazeInstPrinter::printInst(const MCInst *MI, raw_ostream &O) {
  printInstruction(MI, O);
}

void MBlazeInstPrinter::printPCRelImmOperand(const MCInst *MI, unsigned OpNo,
                                             raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm())
    O << Op.getImm();
  else {
    assert(Op.isExpr() && "unknown pcrel immediate operand");
    O << *Op.getExpr();
  }
}

void MBlazeInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O, const char *Modifier) {
  assert((Modifier == 0 || Modifier[0] == 0) && "No modifiers supported");
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    O << getRegisterName(Op.getReg());
  } else if (Op.isImm()) {
    O << (int32_t)Op.getImm();
  } else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    O << *Op.getExpr();
  }
}

void MBlazeInstPrinter::printSrcMemOperand(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O,
                                           const char *Modifier) {
  const MCOperand &Base = MI->getOperand(OpNo);
  const MCOperand &Disp = MI->getOperand(OpNo+1);

  // Print displacement first

  // If the global address expression is a part of displacement field with a
  // register base, we should not emit any prefix symbol here, e.g.
  //   mov.w &foo, r1
  // vs
  //   mov.w glb(r1), r2
  // Otherwise (!) msp430-as will silently miscompile the output :(
  if (!Base.getReg())
    O << '&';

  if (Disp.isExpr())
    O << *Disp.getExpr();
  else {
    assert(Disp.isImm() && "Expected immediate in displacement field");
    O << Disp.getImm();
  }

  // Print register base field
  if (Base.getReg())
    O << getRegisterName(Base.getReg());
}

void MBlazeInstPrinter::printFSLImm(const MCInst *MI, int OpNo,
                                    raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNo);
  if (MO.isImm())
    O << "rfsl" << MO.getImm();
  else
    printOperand(MI, OpNo, O, NULL);
}

void MBlazeInstPrinter::printUnsignedImm(const MCInst *MI, int OpNo,
                                        raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNo);
  if (MO.isImm())
    O << MO.getImm();
  else
    printOperand(MI, OpNo, O, NULL);
}

void MBlazeInstPrinter::printMemOperand(const MCInst *MI, int OpNo,
                                        raw_ostream &O, const char *Modifier ) {
  printOperand(MI, OpNo+1, O, NULL);
  O << ", ";
  printOperand(MI, OpNo, O, NULL);
}

/*
void MBlazeInstPrinter::printCCOperand(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  unsigned CC = MI->getOperand(OpNo).getImm();

  switch (CC) {
  default:
   llvm_unreachable("Unsupported CC code");
   break;
  case MBlazeCC::COND_E:
   O << "eq";
   break;
  case MBlazeCC::COND_NE:
   O << "ne";
   break;
  case MBlazeCC::COND_HS:
   O << "hs";
   break;
  case MBlazeCC::COND_LO:
   O << "lo";
   break;
  case MBlazeCC::COND_GE:
   O << "ge";
   break;
  case MBlazeCC::COND_L:
   O << 'l';
   break;
  }
}
*/
