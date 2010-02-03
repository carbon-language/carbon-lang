//===-- MSP430AsmPrinter.cpp - MSP430 LLVM assembly writer ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to the MSP430 assembly language.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "MSP430.h"
#include "MSP430InstrInfo.h"
#include "MSP430InstPrinter.h"
#include "MSP430MCAsmInfo.h"
#include "MSP430MCInstLower.h"
#include "MSP430TargetMachine.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

namespace {
  class MSP430AsmPrinter : public AsmPrinter {
  public:
    MSP430AsmPrinter(formatted_raw_ostream &O, TargetMachine &TM,
                     MCContext &Ctx, MCStreamer &Streamer,
                     const MCAsmInfo *MAI)
      : AsmPrinter(O, TM, Ctx, Streamer, MAI) {}

    virtual const char *getPassName() const {
      return "MSP430 Assembly Printer";
    }

    void printMCInst(const MCInst *MI) {
      MSP430InstPrinter(O, *MAI).printInstruction(MI);
    }
    void printOperand(const MachineInstr *MI, int OpNum,
                      const char* Modifier = 0);
    void printPCRelImmOperand(const MachineInstr *MI, int OpNum) {
      printOperand(MI, OpNum);
    }
    void printSrcMemOperand(const MachineInstr *MI, int OpNum,
                            const char* Modifier = 0);
    void printCCOperand(const MachineInstr *MI, int OpNum);
    void printMachineInstruction(const MachineInstr * MI);
    bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                         unsigned AsmVariant,
                         const char *ExtraCode);
    bool PrintAsmMemoryOperand(const MachineInstr *MI,
                               unsigned OpNo, unsigned AsmVariant,
                               const char *ExtraCode);
    void EmitInstruction(const MachineInstr *MI);

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AsmPrinter::getAnalysisUsage(AU);
      AU.setPreservesAll();
    }
  };
} // end of anonymous namespace


void MSP430AsmPrinter::printOperand(const MachineInstr *MI, int OpNum,
                                    const char* Modifier) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  switch (MO.getType()) {
  default: assert(0 && "Not implemented yet!");
  case MachineOperand::MO_Register:
    O << MSP430InstPrinter::getRegisterName(MO.getReg());
    return;
  case MachineOperand::MO_Immediate:
    if (!Modifier || strcmp(Modifier, "nohash"))
      O << '#';
    O << MO.getImm();
    return;
  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol(OutContext);
    return;
  case MachineOperand::MO_GlobalAddress: {
    bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
    uint64_t Offset = MO.getOffset();

    O << (isMemOp ? '&' : '#');
    if (Offset)
      O << '(' << Offset << '+';

    O << *GetGlobalValueSymbol(MO.getGlobal());
    
    if (Offset)
      O << ')';

    return;
  }
  case MachineOperand::MO_ExternalSymbol: {
    bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
    O << (isMemOp ? '&' : '#');
    O << MAI->getGlobalPrefix() << MO.getSymbolName();
    return;
  }
  }
}

void MSP430AsmPrinter::printSrcMemOperand(const MachineInstr *MI, int OpNum,
                                          const char* Modifier) {
  const MachineOperand &Base = MI->getOperand(OpNum);
  const MachineOperand &Disp = MI->getOperand(OpNum+1);

  // Print displacement first
  if (!Disp.isImm()) {
    printOperand(MI, OpNum+1, "mem");
  } else {
    if (!Base.getReg())
      O << '&';

    printOperand(MI, OpNum+1, "nohash");
  }


  // Print register base field
  if (Base.getReg()) {
    O << '(';
    printOperand(MI, OpNum);
    O << ')';
  }
}

void MSP430AsmPrinter::printCCOperand(const MachineInstr *MI, int OpNum) {
  switch (MI->getOperand(OpNum).getImm()) {
  default: assert(0 && "Unknown cond");
  case MSP430CC::COND_E:  O << "eq"; break;
  case MSP430CC::COND_NE: O << "ne"; break;
  case MSP430CC::COND_HS: O << "hs"; break;
  case MSP430CC::COND_LO: O << "lo"; break;
  case MSP430CC::COND_GE: O << "ge"; break;
  case MSP430CC::COND_L:  O << 'l';  break;
  }
}

/// PrintAsmOperand - Print out an operand for an inline asm expression.
///
bool MSP430AsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                       unsigned AsmVariant,
                                       const char *ExtraCode) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0])
    return true; // Unknown modifier.

  printOperand(MI, OpNo);
  return false;
}

bool MSP430AsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                             unsigned OpNo, unsigned AsmVariant,
                                             const char *ExtraCode) {
  if (ExtraCode && ExtraCode[0]) {
    return true; // Unknown modifier.
  }
  printSrcMemOperand(MI, OpNo);
  return false;
}

//===----------------------------------------------------------------------===//
void MSP430AsmPrinter::EmitInstruction(const MachineInstr *MI) {
  MSP430MCInstLower MCInstLowering(OutContext, *Mang, *this);

  MCInst TmpInst;
  MCInstLowering.Lower(MI, TmpInst);
  printMCInst(&TmpInst);
  O << '\n';
}

static MCInstPrinter *createMSP430MCInstPrinter(const Target &T,
                                                unsigned SyntaxVariant,
                                                const MCAsmInfo &MAI,
                                                raw_ostream &O) {
  if (SyntaxVariant == 0)
    return new MSP430InstPrinter(O, MAI);
  return 0;
}

// Force static initialization.
extern "C" void LLVMInitializeMSP430AsmPrinter() {
  RegisterAsmPrinter<MSP430AsmPrinter> X(TheMSP430Target);
  TargetRegistry::RegisterMCInstPrinter(TheMSP430Target,
                                        createMSP430MCInstPrinter);
}
