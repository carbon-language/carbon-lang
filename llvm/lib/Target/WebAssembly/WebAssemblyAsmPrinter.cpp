//===-- WebAssemblyAsmPrinter.cpp - WebAssembly LLVM assembly writer ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains a printer that converts from our internal
/// representation of machine-dependent LLVM code to the WebAssembly assembly
/// language.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblyRegisterInfo.h"
#include "WebAssemblySubtarget.h"
#include "InstPrinter/WebAssemblyInstPrinter.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

namespace {

class WebAssemblyAsmPrinter final : public AsmPrinter {
  const WebAssemblyInstrInfo *TII;

public:
  WebAssemblyAsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)), TII(nullptr) {}

private:
  const char *getPassName() const override {
    return "WebAssembly Assembly Printer";
  }

  //===------------------------------------------------------------------===//
  // MachineFunctionPass Implementation.
  //===------------------------------------------------------------------===//

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AsmPrinter::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    TII = static_cast<const WebAssemblyInstrInfo *>(
        MF.getSubtarget().getInstrInfo());
    return AsmPrinter::runOnMachineFunction(MF);
  }

  //===------------------------------------------------------------------===//
  // AsmPrinter Implementation.
  //===------------------------------------------------------------------===//

  void EmitInstruction(const MachineInstr *MI) override;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//

void WebAssemblyAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  SmallString<128> Str;
  raw_svector_ostream OS(Str);

  unsigned NumDefs = MI->getDesc().getNumDefs();
  assert(NumDefs <= 1 &&
         "Instructions with multiple result values not implemented");

  if (NumDefs != 0) {
    const MachineOperand &MO = MI->getOperand(0);
    unsigned Reg = MO.getReg();
    OS << "(setlocal @" << TargetRegisterInfo::virtReg2Index(Reg) << ' ';
  }

  OS << '(';

  bool PrintOperands = true;
  switch (MI->getOpcode()) {
  case WebAssembly::ARGUMENT_Int32:
  case WebAssembly::ARGUMENT_Int64:
  case WebAssembly::ARGUMENT_Float32:
  case WebAssembly::ARGUMENT_Float64:
    OS << "argument " << MI->getOperand(1).getImm();
    PrintOperands = false;
    break;
  case WebAssembly::RETURN_Int32:
  case WebAssembly::RETURN_Int64:
  case WebAssembly::RETURN_Float32:
  case WebAssembly::RETURN_Float64:
  case WebAssembly::RETURN_VOID:
    // FIXME This is here only so "return" prints nicely, instead of printing
    //       the isel name. Other operations have the same problem, fix this in
    //       a generic way instead.
    OS << "return";
    break;
  default:
    OS << TII->getName(MI->getOpcode());
    break;
  }

  if (PrintOperands)
    for (const MachineOperand &MO : MI->uses()) {
      if (MO.isReg() && MO.isImplicit())
        continue;
      unsigned Reg = MO.getReg();
      OS << " @" << TargetRegisterInfo::virtReg2Index(Reg);
    }
  OS << ')';

  if (NumDefs != 0)
    OS << ')';

  OS << '\n';

  OutStreamer->EmitRawText(OS.str());
}

// Force static initialization.
extern "C" void LLVMInitializeWebAssemblyAsmPrinter() {
  RegisterAsmPrinter<WebAssemblyAsmPrinter> X(TheWebAssemblyTarget32);
  RegisterAsmPrinter<WebAssemblyAsmPrinter> Y(TheWebAssemblyTarget64);
}
