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
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace opt_tool;

bool llvm::runPassPipeline(StringRef Arg0, LLVMContext &Context, Module &M,
                           tool_output_file *Out, StringRef PassPipeline,
                           OutputKind OK) {
  ModulePassManager MPM;
  if (!parsePassPipeline(MPM, PassPipeline)) {
    errs() << Arg0 << ": unable to parse pass pipeline description.\n";
    return false;
  }

  // Add any relevant output pass at the end of the pipeline.
  switch (OK) {
  case OK_NoOutput:
    break; // No output pass needed.
  case OK_OutputAssembly:
    MPM.addPass(PrintModulePass(Out->os()));
    break;
  case OK_OutputBitcode:
    llvm::report_fatal_error("Bitcode output is not yet implemented!");
  }

  // Before executing passes, print the final values of the LLVM options.
  cl::PrintOptionValues();

  // Now that we have all of the passes ready, run them.
  MPM.run(&M);

  // Declare success.
  if (OK != OK_NoOutput)
    Out->keep();
  return true;
}
