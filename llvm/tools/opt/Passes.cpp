//===- Passes.cpp - Parsing, selection, and running of passes -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides the infrastructure to parse and build a custom pass
/// manager based on a commandline flag. It also provides helpers to aid in
/// analyzing, debugging, and testing pass structures.
///
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;

namespace {

  /// \brief No-op module pass which does nothing.
struct NoOpModulePass {
  PreservedAnalyses run(Module *M) { return PreservedAnalyses::all(); }
  static StringRef name() { return "NoOpModulePass"; }
};

} // End anonymous namespace.

// FIXME: Factor all of the parsing logic into a .def file that we include
// under different macros.
static bool isModulePassName(StringRef Name) {
  if (Name == "no-op-module") return true;

  return false;
}

static bool parseModulePassName(ModulePassManager &MPM, StringRef Name) {
  assert(isModulePassName(Name));
  if (Name == "no-op-module") {
    MPM.addPass(NoOpModulePass());
    return true;
  }
  return false;
}

static bool parseModulePassPipeline(ModulePassManager &MPM,
                                    StringRef &PipelineText) {
  for (;;) {
    // Parse nested pass managers by recursing.
    if (PipelineText.startswith("module(")) {
      ModulePassManager NestedMPM;

      // Parse the inner pipeline into the nested manager.
      PipelineText = PipelineText.substr(strlen("module("));
      if (!parseModulePassPipeline(NestedMPM, PipelineText))
        return false;
      assert(!PipelineText.empty() && PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Now add the nested manager as a module pass.
      MPM.addPass(NestedMPM);
    } else {
      // Otherwise try to parse a pass name.
      size_t End = PipelineText.find_first_of(",)");
      if (!parseModulePassName(MPM, PipelineText.substr(0, End)))
        return false;

      PipelineText = PipelineText.substr(End);
    }

    if (PipelineText.empty() || PipelineText[0] == ')')
      return true;

    assert(PipelineText[0] == ',');
    PipelineText = PipelineText.substr(1);
  }
}

// Primary pass pipeline description parsing routine.
// FIXME: Should this routine accept a TargetMachine or require the caller to
// pre-populate the analysis managers with target-specific stuff?
bool llvm::parsePassPipeline(ModulePassManager &MPM, StringRef PipelineText) {
  // Look at the first entry to figure out which layer to start parsing at.
  if (PipelineText.startswith("module("))
    return parseModulePassPipeline(MPM, PipelineText);

  // FIXME: Support parsing function pass manager nests.

  // This isn't a direct pass manager name, look for the end of a pass name.
  StringRef FirstName = PipelineText.substr(0, PipelineText.find_first_of(","));
  if (isModulePassName(FirstName))
    return parseModulePassPipeline(MPM, PipelineText);

  // FIXME: Support parsing function pass names.

  return false;
}
