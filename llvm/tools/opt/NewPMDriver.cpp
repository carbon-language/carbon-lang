//===- NewPMDriver.cpp - Driver for opt with new PM -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file is just a split of the code that logically belongs in opt.cpp but
/// that includes the new pass manager headers.
///
//===----------------------------------------------------------------------===//

#include "NewPMDriver.h"
#include "Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;

bool llvm::runPassPipeline(StringRef Arg0, LLVMContext &Context, Module &M,
                           tool_output_file *Out, StringRef PassPipeline,
                           bool NoOutput) {
  // Before executing passes, print the final values of the LLVM options.
  cl::PrintOptionValues();

  ModulePassManager MPM;
  if (!parsePassPipeline(MPM, PassPipeline)) {
    errs() << Arg0 << ": unable to parse pass pipeline description.\n";
    return false;
  }

  // Now that we have all of the passes ready, run them.
  MPM.run(&M);

  // Declare success.
  if (!NoOutput)
    Out->keep();
  return true;
}
