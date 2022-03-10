//===- AnalysisWrappers.cpp - Wrappers around non-pass analyses -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines pass wrappers around LLVM analyses that don't make sense to
// be passes.  It provides a nice standard pass interface to these classes so
// that they can be printed out by analyze.
//
// These classes are separated out of analyze.cpp so that it is more clear which
// code is the integral part of the analyze tool, and which part of the code is
// just making it so more passes are available.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  /// ExternalFunctionsPassedConstants - This pass prints out call sites to
  /// external functions that are called with constant arguments.  This can be
  /// useful when looking for standard library functions we should constant fold
  /// or handle in alias analyses.
  struct ExternalFunctionsPassedConstants : public ModulePass {
    static char ID; // Pass ID, replacement for typeid
    ExternalFunctionsPassedConstants() : ModulePass(ID) {}
    bool runOnModule(Module &M) override {
      for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
        if (!I->isDeclaration()) continue;

        bool PrintedFn = false;
        for (User *U : I->users()) {
          Instruction *UI = dyn_cast<Instruction>(U);
          if (!UI) continue;

          CallBase *CB = dyn_cast<CallBase>(UI);
          if (!CB)
            continue;

          for (auto AI = CB->arg_begin(), E = CB->arg_end(); AI != E; ++AI) {
            if (!isa<Constant>(*AI)) continue;

            if (!PrintedFn) {
              errs() << "Function '" << I->getName() << "':\n";
              PrintedFn = true;
            }
            errs() << *UI;
            break;
          }
        }
      }

      return false;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
    }
  };
}

char ExternalFunctionsPassedConstants::ID = 0;
static RegisterPass<ExternalFunctionsPassedConstants>
  P1("print-externalfnconstants",
     "Print external fn callsites passed constants");
