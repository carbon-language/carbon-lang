//===-- ModuleDebugInfoPrinter.cpp - Prints module debug info metadata ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass decodes the debug info metadata in a module and prints in a
// (sufficiently-prepared-) human-readable form.
//
// For example, run this pass from opt along with the -analyze option, and
// it'll print to standard output.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  class ModuleDebugInfoPrinter : public ModulePass {
    DebugInfoFinder Finder;
  public:
    static char ID; // Pass identification, replacement for typeid
    ModuleDebugInfoPrinter() : ModulePass(ID) {
      initializeModuleDebugInfoPrinterPass(*PassRegistry::getPassRegistry());
    }

    bool runOnModule(Module &M) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
    }
    void print(raw_ostream &O, const Module *M) const override;
  };
}

char ModuleDebugInfoPrinter::ID = 0;
INITIALIZE_PASS(ModuleDebugInfoPrinter, "module-debuginfo",
                "Decodes module-level debug info", false, true)

ModulePass *llvm::createModuleDebugInfoPrinterPass() {
  return new ModuleDebugInfoPrinter();
}

bool ModuleDebugInfoPrinter::runOnModule(Module &M) {
  Finder.processModule(M);
  return false;
}

void ModuleDebugInfoPrinter::print(raw_ostream &O, const Module *M) const {
  for (DICompileUnit CU : Finder.compile_units()) {
    O << "Compile Unit: ";
    CU.print(O);
    O << '\n';
  }

  for (DISubprogram S : Finder.subprograms()) {
    O << "Subprogram: ";
    S.print(O);
    O << '\n';
  }

  for (DIGlobalVariable GV : Finder.global_variables()) {
    O << "GlobalVariable: ";
    GV.print(O);
    O << '\n';
  }

  for (DIType T : Finder.types()) {
    O << "Type: ";
    T.print(O);
    O << '\n';
  }
}
