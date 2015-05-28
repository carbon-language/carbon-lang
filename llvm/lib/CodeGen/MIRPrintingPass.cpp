//===- MIRPrintingPass.cpp - Pass that prints out using the MIR format ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that prints out the LLVM module using the MIR
// serialization format.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MIRYamlMapping.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/YAMLTraits.h"

using namespace llvm;

namespace llvm {
namespace yaml {

/// This struct serializes the LLVM IR module.
template <> struct BlockScalarTraits<Module> {
  static void output(const Module &Mod, void *Ctxt, raw_ostream &OS) {
    Mod.print(OS, nullptr);
  }
  static StringRef input(StringRef Str, void *Ctxt, Module &Mod) {
    llvm_unreachable("LLVM Module is supposed to be parsed separately");
    return "";
  }
};

} // end namespace yaml
} // end namespace llvm

namespace {

/// This class prints out the machine functions using the MIR serialization
/// format.
class MIRPrinter {
  raw_ostream &OS;

public:
  MIRPrinter(raw_ostream &OS) : OS(OS) {}

  void print(const MachineFunction &MF);
};

void MIRPrinter::print(const MachineFunction &MF) {
  yaml::MachineFunction YamlMF;
  YamlMF.Name = MF.getName();
  yaml::Output Out(OS);
  Out << YamlMF;
}

/// This pass prints out the LLVM IR to an output stream using the MIR
/// serialization format.
struct MIRPrintingPass : public MachineFunctionPass {
  static char ID;
  raw_ostream &OS;
  std::string MachineFunctions;

  MIRPrintingPass() : MachineFunctionPass(ID), OS(dbgs()) {}
  MIRPrintingPass(raw_ostream &OS) : MachineFunctionPass(ID), OS(OS) {}

  const char *getPassName() const override { return "MIR Printing Pass"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  virtual bool runOnMachineFunction(MachineFunction &MF) override {
    std::string Str;
    raw_string_ostream StrOS(Str);
    MIRPrinter(StrOS).print(MF);
    MachineFunctions.append(StrOS.str());
    return false;
  }

  virtual bool doFinalization(Module &M) override {
    yaml::Output Out(OS);
    Out << M;
    OS << MachineFunctions;
    return false;
  }
};

char MIRPrintingPass::ID = 0;

} // end anonymous namespace

char &llvm::MIRPrintingPassID = MIRPrintingPass::ID;
INITIALIZE_PASS(MIRPrintingPass, "mir-printer", "MIR Printer", false, false)

namespace llvm {

MachineFunctionPass *createPrintMIRPass(raw_ostream &OS) {
  return new MIRPrintingPass(OS);
}

} // end namespace llvm
