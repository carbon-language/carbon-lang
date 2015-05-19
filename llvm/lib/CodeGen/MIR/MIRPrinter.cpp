//===- MIRPrinter.cpp - MIR serialization format printer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the class that prints out the LLVM IR and machine
// functions using the MIR serialization format.
//
//===----------------------------------------------------------------------===//

#include "MIRPrinter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/YAMLTraits.h"

using namespace llvm;

namespace {

/// This class prints out the LLVM IR using the MIR serialization format and
/// YAML I/O.
class MIRPrinter {
  raw_ostream &OS;

public:
  MIRPrinter(raw_ostream &OS);

  void printModule(const Module &Mod);
};

} // end anonymous namespace

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

MIRPrinter::MIRPrinter(raw_ostream &OS) : OS(OS) {}

void MIRPrinter::printModule(const Module &Mod) {
  yaml::Output Out(OS);
  Out << const_cast<Module &>(Mod);
}

void llvm::printMIR(raw_ostream &OS, const Module &Mod) {
  MIRPrinter Printer(OS);
  Printer.printModule(Mod);
}
