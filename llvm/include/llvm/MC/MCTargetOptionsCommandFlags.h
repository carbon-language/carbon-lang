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
    "asm-instrumentation", cl::desc("Instrumentation of inline assembly and "
                                    "assembly source files"),
    cl::init(MCTargetOptions::AsmInstrumentationNone),
    cl::values(clEnumValN(MCTargetOptions::AsmInstrumentationNone, "none",
                          "no instrumentation at all"),
               clEnumValN(MCTargetOptions::AsmInstrumentationAddress, "address",
                          "instrument instructions with memory arguments"),
               clEnumValEnd));

cl::opt<bool> RelaxAll("mc-relax-all",
                       cl::desc("When used with filetype=obj, "
                                "relax all fixups in the emitted object file"));

cl::opt<bool> EnableDwarfDirectory(
    "enable-dwarf-directory", cl::Hidden,
    cl::desc("Use .file directives with an explicit directory."));

cl::opt<bool> NoExecStack("mc-no-exec-stack",
                          cl::desc("File doesn't need an exec stack"));

static cl::opt<bool>
SaveTempLabels("L", cl::desc("Don't discard temporary labels"));

static inline MCTargetOptions InitMCTargetOptionsFromFlags() {
  MCTargetOptions Options;
  Options.SanitizeAddress =
      (AsmInstrumentation == MCTargetOptions::AsmInstrumentationAddress);
  Options.MCRelaxAll = RelaxAll;
  Options.MCUseDwarfDirectory = EnableDwarfDirectory;
  Options.MCNoExecStack = NoExecStack;
  Options.MCSaveTempLabels = SaveTempLabels;
  return Options;
}

#endif
