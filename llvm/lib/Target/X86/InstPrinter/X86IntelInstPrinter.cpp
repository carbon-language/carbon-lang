//===-- X86IntelInstPrinter.cpp - AT&T assembly instruction printing ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file includes code for rendering MCInst instances as AT&T-style
// assembly.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "X86IntelInstPrinter.h"
#include "X86InstComments.h"
#include "X86Subtarget.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "X86GenInstrNames.inc"
#include <cctype>
using namespace llvm;

// Include the auto-generated portion of the assembly writer.
#define GET_INSTRUCTION_NAME
#include "X86GenAsmWriter1.inc"

void X86IntelInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  OS << getRegisterName(RegNo);
}

void X86IntelInstPrinter::printInst(const MCInst *MI, raw_ostream &OS) {
  printInstruction(MI, OS);
  
  // If verbose assembly is enabled, we can print some informative comments.
  if (CommentStream)
    EmitAnyX86InstComments(MI, *CommentStream, getRegisterName);
}
StringRef X86IntelInstPrinter::getOpcodeName(unsigned Opcode) const {
  return getInstructionName(Opcode);
}

void X86IntelInstPrinter::printSSECC(const MCInst *MI, unsigned Op,
                                     raw_ostream &O) {
  switch (MI->getOperand(Op).getImm()) {
  default: assert(0 && "Invalid ssecc argument!");
  case 0: O << "eq"; break;
  case 1: O << "lt"; break;
  case 2: O << "le"; break;
  case 3: O << "unord"; break;
  case 4: O << "neq"; break;
  case 5: O << "nlt"; break;
  case 6: O << "nle"; break;
  case 7: O << "ord"; break;
  }
}

/// print_pcrel_imm - This is used to print an immediate value that ends up
/// being encoded as a pc-relative value.
void X86IntelInstPrinter::print_pcrel_imm(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm())
    O << Op.getImm();
  else {
    assert(Op.isExpr() && "unknown pcrel immediate operand");
    O << *Op.getExpr();
  }
}

static void PrintRegName(raw_ostream &O, StringRef RegName) {
  for (unsigned i = 0, e = RegName.size(); i != e; ++i)
    O << (char)toupper(RegName[i]);
}

void X86IntelInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    PrintRegName(O, getRegisterName(Op.getReg()));
  } else if (Op.isImm()) {
    O << Op.getImm();
  } else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    O << *Op.getExpr();
  }
}

void X86IntelInstPrinter::printMemReference(const MCInst *MI, unsigned Op,
                                            raw_ostream &O) {
  const MCOperand &BaseReg  = MI->getOperand(Op);
  unsigned ScaleVal         = MI->getOperand(Op+1).getImm();
  const MCOperand &IndexReg = MI->getOperand(Op+2);
  const MCOperand &DispSpec = MI->getOperand(Op+3);
  const MCOperand &SegReg   = MI->getOperand(Op+4);
  
  // If this has a segment register, print it.
  if (SegReg.getReg()) {
    printOperand(MI, Op+4, O);
    O << ':';
  }
  
  O << '[';
  
  bool NeedPlus = false;
  if (BaseReg.getReg()) {
    printOperand(MI, Op, O);
    NeedPlus = true;
  }
  
  if (IndexReg.getReg()) {
    if (NeedPlus) O << " + ";
    if (ScaleVal != 1)
      O << ScaleVal << '*';
    printOperand(MI, Op+2, O);
    NeedPlus = true;
  }
  
  
  if (!DispSpec.isImm()) {
    if (NeedPlus) O << " + ";
    assert(DispSpec.isExpr() && "non-immediate displacement for LEA?");
    O << *DispSpec.getExpr();
  } else {
    int64_t DispVal = DispSpec.getImm();
    if (DispVal || (!IndexReg.getReg() && !BaseReg.getReg())) {
      if (NeedPlus) {
        if (DispVal > 0)
          O << " + ";
        else {
          O << " - ";
          DispVal = -DispVal;
        }
      }
      O << DispVal;
    }
  }
  
  O << ']';
}
