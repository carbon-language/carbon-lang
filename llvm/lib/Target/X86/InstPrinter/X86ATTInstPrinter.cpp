//===-- X86ATTInstPrinter.cpp - AT&T assembly instruction printing --------===//
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
#include "X86ATTInstPrinter.h"
#include "X86InstComments.h"
#include "X86Subtarget.h"
#include "MCTargetDesc/X86TargetDesc.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormattedStream.h"
#include <map>
using namespace llvm;

// Include the auto-generated portion of the assembly writer.
#define GET_INSTRUCTION_NAME
#define PRINT_ALIAS_INSTR
#include "X86GenAsmWriter.inc"

X86ATTInstPrinter::X86ATTInstPrinter(TargetMachine &TM, const MCAsmInfo &MAI)
  : MCInstPrinter(MAI) {
  // Initialize the set of available features.
  setAvailableFeatures(ComputeAvailableFeatures(
            &TM.getSubtarget<X86Subtarget>()));
}

void X86ATTInstPrinter::printRegName(raw_ostream &OS,
                                     unsigned RegNo) const {
  OS << '%' << getRegisterName(RegNo);
}

void X86ATTInstPrinter::printInst(const MCInst *MI, raw_ostream &OS) {
  // Try to print any aliases first.
  if (!printAliasInstr(MI, OS))
    printInstruction(MI, OS);
  
  // If verbose assembly is enabled, we can print some informative comments.
  if (CommentStream)
    EmitAnyX86InstComments(MI, *CommentStream, getRegisterName);
}

StringRef X86ATTInstPrinter::getOpcodeName(unsigned Opcode) const {
  return getInstructionName(Opcode);
}

void X86ATTInstPrinter::printSSECC(const MCInst *MI, unsigned Op,
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
/// being encoded as a pc-relative value (e.g. for jumps and calls).  These
/// print slightly differently than normal immediates.  For example, a $ is not
/// emitted.
void X86ATTInstPrinter::print_pcrel_imm(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm())
    // Print this as a signed 32-bit value.
    O << (int)Op.getImm();
  else {
    assert(Op.isExpr() && "unknown pcrel immediate operand");
    O << *Op.getExpr();
  }
}

void X86ATTInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    O << '%' << getRegisterName(Op.getReg());
  } else if (Op.isImm()) {
    O << '$' << Op.getImm();
    
    if (CommentStream && (Op.getImm() > 255 || Op.getImm() < -256))
      *CommentStream << format("imm = 0x%llX\n", (long long)Op.getImm());
    
  } else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    O << '$' << *Op.getExpr();
  }
}

void X86ATTInstPrinter::printMemReference(const MCInst *MI, unsigned Op,
                                          raw_ostream &O) {
  const MCOperand &BaseReg  = MI->getOperand(Op);
  const MCOperand &IndexReg = MI->getOperand(Op+2);
  const MCOperand &DispSpec = MI->getOperand(Op+3);
  const MCOperand &SegReg = MI->getOperand(Op+4);
  
  // If this has a segment register, print it.
  if (SegReg.getReg()) {
    printOperand(MI, Op+4, O);
    O << ':';
  }
  
  if (DispSpec.isImm()) {
    int64_t DispVal = DispSpec.getImm();
    if (DispVal || (!IndexReg.getReg() && !BaseReg.getReg()))
      O << DispVal;
  } else {
    assert(DispSpec.isExpr() && "non-immediate displacement for LEA?");
    O << *DispSpec.getExpr();
  }
  
  if (IndexReg.getReg() || BaseReg.getReg()) {
    O << '(';
    if (BaseReg.getReg())
      printOperand(MI, Op, O);
    
    if (IndexReg.getReg()) {
      O << ',';
      printOperand(MI, Op+2, O);
      unsigned ScaleVal = MI->getOperand(Op+1).getImm();
      if (ScaleVal != 1)
        O << ',' << ScaleVal;
    }
    O << ')';
  }
}
