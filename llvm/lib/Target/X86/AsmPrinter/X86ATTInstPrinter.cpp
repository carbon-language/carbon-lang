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
#include "llvm/MC/MCInst.h"
#include "X86ATTAsmPrinter.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

// Include the auto-generated portion of the assembly writer.
#define MachineInstr MCInst
#define NO_ASM_WRITER_BOILERPLATE
#include "X86GenAsmWriter.inc"
#undef MachineInstr

void X86ATTAsmPrinter::printSSECC(const MCInst *MI, unsigned Op) {
  switch (MI->getOperand(Op).getImm()) {
  default: llvm_unreachable("Invalid ssecc argument!");
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


void X86ATTAsmPrinter::printPICLabel(const MCInst *MI, unsigned Op) {
  llvm_unreachable("This is only used for MOVPC32r,"
                   "should lower before asm printing!");
}


/// print_pcrel_imm - This is used to print an immediate value that ends up
/// being encoded as a pc-relative value.  These print slightly differently, for
/// example, a $ is not emitted.
void X86ATTAsmPrinter::print_pcrel_imm(const MCInst *MI, unsigned OpNo) {
  const MCOperand &Op = MI->getOperand(OpNo);
  
  if (Op.isImm())
    O << Op.getImm();
  else if (Op.isExpr())
    Op.getExpr()->print(O, MAI);
  else if (Op.isMBBLabel())
    // FIXME: Keep in sync with printBasicBlockLabel.  printBasicBlockLabel
    // should eventually call into this code, not the other way around.
    O << MAI->getPrivateGlobalPrefix() << "BB" << Op.getMBBLabelFunction()
      << '_' << Op.getMBBLabelBlock();
  else
    llvm_unreachable("Unknown pcrel immediate operand");
}


void X86ATTAsmPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                    const char *Modifier) {
  assert(Modifier == 0 && "Modifiers should not be used");
  
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    O << '%';
    unsigned Reg = Op.getReg();
#if 0
    if (Modifier && strncmp(Modifier, "subreg", strlen("subreg")) == 0) {
      EVT VT = (strcmp(Modifier+6,"64") == 0) ?
      EVT::i64 : ((strcmp(Modifier+6, "32") == 0) ? EVT::i32 :
                  ((strcmp(Modifier+6,"16") == 0) ? EVT::i16 : EVT::i8));
      Reg = getX86SubSuperRegister(Reg, VT);
    }
#endif
    O << TRI->getAsmName(Reg);
    return;
  } else if (Op.isImm()) {
    //if (!Modifier || (strcmp(Modifier, "debug") && strcmp(Modifier, "mem")))
    O << '$';
    O << Op.getImm();
    return;
  } else if (Op.isExpr()) {
    O << '$';
    Op.getExpr()->print(O, MAI);
    return;
  }
  
  O << "<<UNKNOWN OPERAND KIND>>";
}

void X86ATTAsmPrinter::printLeaMemReference(const MCInst *MI, unsigned Op) {

  const MCOperand &BaseReg  = MI->getOperand(Op);
  const MCOperand &IndexReg = MI->getOperand(Op+2);
  const MCOperand &DispSpec = MI->getOperand(Op+3);
  
  if (DispSpec.isImm()) {
    int64_t DispVal = DispSpec.getImm();
    if (DispVal || (!IndexReg.getReg() && !BaseReg.getReg()))
      O << DispVal;
  } else if (DispSpec.isExpr()) {
    DispSpec.getExpr()->print(O, MAI);
  } else {
    llvm_unreachable("non-immediate displacement for LEA?");
    //assert(DispSpec.isGlobal() || DispSpec.isCPI() ||
    //       DispSpec.isJTI() || DispSpec.isSymbol());
    //printOperand(MI, Op+3, "mem");
  }
  
  if (IndexReg.getReg() || BaseReg.getReg()) {
    // There are cases where we can end up with ESP/RSP in the indexreg slot.
    // If this happens, swap the base/index register to support assemblers that
    // don't work when the index is *SP.
    // FIXME: REMOVE THIS.
    assert(IndexReg.getReg() != X86::ESP && IndexReg.getReg() != X86::RSP);
    
    O << '(';
    if (BaseReg.getReg())
      printOperand(MI, Op);
    
    if (IndexReg.getReg()) {
      O << ',';
      printOperand(MI, Op+2);
      unsigned ScaleVal = MI->getOperand(Op+1).getImm();
      if (ScaleVal != 1)
        O << ',' << ScaleVal;
    }
    O << ')';
  }
}

void X86ATTAsmPrinter::printMemReference(const MCInst *MI, unsigned Op) {
  const MCOperand &Segment = MI->getOperand(Op+4);
  if (Segment.getReg()) {
    printOperand(MI, Op+4);
    O << ':';
  }
  printLeaMemReference(MI, Op);
}
