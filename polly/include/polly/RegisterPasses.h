//===------ polly/RegisterPasses.h - Register the Polly passes *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Functions to register the Polly passes in a LLVM pass manager.
//
//===----------------------------------------------------------------------===//

namespace llvm {
  class PassManagerBase;
}

// Register the Polly preoptimization passes. Preoptimizations are used to
// prepare the LLVM-IR for Polly. They increase the amount of code that can be
// optimized.
// (These passes are automatically included in registerPollyPasses).
void registerPollyPreoptPasses(llvm::PassManagerBase &PM);

// Register the Polly optimizer (including its preoptimizations).
void registerPollyPasses(llvm::PassManagerBase &PM,
                         bool DisableCodegen = false);
