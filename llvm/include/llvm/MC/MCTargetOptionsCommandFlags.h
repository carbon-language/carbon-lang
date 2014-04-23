//===-- MCTargetOptionsCommandFlags.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains machine code-specific flags that are shared between
// different command line tools.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCTARGETOPTIONSCOMMANDFLAGS_H
#define LLVM_MC_MCTARGETOPTIONSCOMMANDFLAGS_H

#include "llvm/Support/CommandLine.h"
#include "llvm/MC/MCTargetOptions.h"
using namespace llvm;

cl::opt<MCTargetOptions::AsmInstrumentation> AsmInstrumentation(
    "asm-instrumentation",
    cl::desc("Instrumentation of inline assembly and "
             "assembly source files"),
    cl::init(MCTargetOptions::AsmInstrumentationNone),
    cl::values(clEnumValN(MCTargetOptions::AsmInstrumentationNone,
                          "none",
                          "no instrumentation at all"),
               clEnumValN(MCTargetOptions::AsmInstrumentationAddress,
                          "address",
                          "instrument instructions with memory arguments"),
               clEnumValEnd));

static inline MCTargetOptions InitMCTargetOptionsFromFlags() {
  MCTargetOptions Options;
  Options.SanitizeAddress =
      (AsmInstrumentation == MCTargetOptions::AsmInstrumentationAddress);
  return Options;
}

#endif
