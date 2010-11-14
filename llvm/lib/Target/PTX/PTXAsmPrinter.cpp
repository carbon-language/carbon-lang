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

#define DEBUG_TYPE "ptx-asm-printer"

#include "PTX.h"
#include "PTXMachineFunctionInfo.h"
#include "PTXTargetMachine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
class PTXAsmPrinter : public AsmPrinter {
public:
  explicit PTXAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
    : AsmPrinter(TM, Streamer) {}

  const char *getPassName() const { return "PTX Assembly Printer"; }

  virtual bool runOnMachineFunction(MachineFunction &MF);

  virtual void EmitFunctionBodyStart();
  virtual void EmitFunctionBodyEnd() { OutStreamer.EmitRawText(Twine("}")); }

  virtual void EmitInstruction(const MachineInstr *MI);

  void printOperand(const MachineInstr *MI, int opNum, raw_ostream &OS);

  // autogen'd.
  void printInstruction(const MachineInstr *MI, raw_ostream &OS);
  static const char *getRegisterName(unsigned RegNo);

private:
  void EmitFunctionDeclaration();
}; // class PTXAsmPrinter
} // namespace

static const char PARAM_PREFIX[] = "__param_";

static const char *getRegisterTypeName(unsigned RegNo){
#define TEST_REGCLS(cls, clsstr) \
  if (PTX::cls ## RegisterClass->contains(RegNo)) return # clsstr;
  TEST_REGCLS(RRegs32, .s32);
  TEST_REGCLS(Preds, .pred);
#undef TEST_REGCLS

  llvm_unreachable("Not in any register class!");
  return NULL;
}

bool PTXAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  SetupMachineFunction(MF);
  EmitFunctionDeclaration();
  EmitFunctionBody();
  return false;
}

void PTXAsmPrinter::EmitFunctionBodyStart() {
  OutStreamer.EmitRawText(Twine("{"));

  const PTXMachineFunctionInfo *MFI = MF->getInfo<PTXMachineFunctionInfo>();

  // Print local variable definition
  for (PTXMachineFunctionInfo::reg_iterator
       i = MFI->localVarRegBegin(), e = MFI->localVarRegEnd(); i != e; ++ i) {
    unsigned reg = *i;

    std::string def = "\t.reg ";
    def += getRegisterTypeName(reg);
    def += ' ';
    def += getRegisterName(reg);
    def += ';';
    OutStreamer.EmitRawText(Twine(def));
  }
}

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

void PTXAsmPrinter::EmitFunctionDeclaration() {
  // The function label could have already been emitted if two symbols end up
  // conflicting due to asm renaming.  Detect this and emit an error.
  if (!CurrentFnSym->isUndefined()) {
    report_fatal_error("'" + Twine(CurrentFnSym->getName()) +
                       "' label emitted multiple times to assembly file");
    return;
  }

  const PTXMachineFunctionInfo *MFI = MF->getInfo<PTXMachineFunctionInfo>();
  const bool isKernel = MFI->isKernel();
  unsigned reg;

  std::string decl = isKernel ? ".entry" : ".func";

  // Print return register
  reg = MFI->retReg();
  if (!isKernel && reg != PTX::NoRegister) {
    decl += " (.reg "; // FIXME: could it return in .param space?
    decl += getRegisterTypeName(reg);
    decl += " ";
    decl += getRegisterName(reg);
    decl += ")";
  }

  // Print function name
  decl += " ";
  decl += CurrentFnSym->getName().str();

  // Print parameter list
  if (!MFI->argRegEmpty()) {
    decl += " (";
    if (isKernel) {
      for (int i = 0, e = MFI->getNumArg(); i != e; ++i) {
        if (i != 0)
          decl += ", ";
        decl += ".param .s32 "; // TODO: param's type
        decl += PARAM_PREFIX;
        decl += utostr(i + 1);
      }
    } else {
      for (PTXMachineFunctionInfo::reg_iterator
           i = MFI->argRegBegin(), e = MFI->argRegEnd(), b = i; i != e; ++i) {
        reg = *i;
        assert(reg != PTX::NoRegister && "Not a valid register!");
        if (i != b)
          decl += ", ";
        decl += ".reg ";
        decl += getRegisterTypeName(reg);
        decl += " ";
        decl += getRegisterName(reg);
      }
    }
    decl += ")";
  }

  OutStreamer.EmitRawText(Twine(decl));
}

#include "PTXGenAsmWriter.inc"

// Force static initialization.
extern "C" void LLVMInitializePTXAsmPrinter() {
  RegisterAsmPrinter<PTXAsmPrinter> X(ThePTXTarget);
}
