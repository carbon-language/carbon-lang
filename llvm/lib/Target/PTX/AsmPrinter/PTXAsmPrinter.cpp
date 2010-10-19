//===-- PTXAsmPrinter.cpp - PTX LLVM assembly writer ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to PTX assembly language.
//
//===----------------------------------------------------------------------===//

#include "PTX.h"
#include "PTXTargetMachine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/TargetRegistry.h"

using namespace llvm;

namespace {
  class PTXAsmPrinter : public AsmPrinter {
    public:
      explicit PTXAsmPrinter(TargetMachine &TM, MCStreamer &Streamer) :
        AsmPrinter(TM, Streamer) {}
      const char *getPassName() const { return "PTX Assembly Printer"; }

      virtual void EmitInstruction(const MachineInstr *MI);

      void printOperand(const MachineInstr *MI, int opNum, raw_ostream &OS);

      // autogen'd.
      void printInstruction(const MachineInstr *MI, raw_ostream &OS);
      static const char *getRegisterName(unsigned RegNo);
  };
} // namespace

void PTXAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  SmallString<128> str;
  raw_svector_ostream OS(str);
  printInstruction(MI, OS);
  OS << ';';
  OutStreamer.EmitRawText(OS.str());
}

void PTXAsmPrinter::printOperand(const MachineInstr *MI, int opNum,
                                 raw_ostream &OS) {
  const MachineOperand &MO = MI->getOperand(opNum);

  switch (MO.getType()) {
    default:
      llvm_unreachable("<unknown operand type>");
      break;
    case MachineOperand::MO_Register:
      OS << getRegisterName(MO.getReg());
      break;
    case MachineOperand::MO_Immediate:
      OS << (int) MO.getImm();
      break;
  }
}

#include "PTXGenAsmWriter.inc"

// Force static initialization.
extern "C" void LLVMInitializePTXAsmPrinter() {
  RegisterAsmPrinter<PTXAsmPrinter> X(ThePTXTarget);
}
