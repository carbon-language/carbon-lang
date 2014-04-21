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
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

namespace {

/// \brief No-op module pass which does nothing.
struct NoOpModulePass {
  PreservedAnalyses run(Module *M) { return PreservedAnalyses::all(); }
  static StringRef name() { return "NoOpModulePass"; }
};

/// \brief No-op CGSCC pass which does nothing.
struct NoOpCGSCCPass {
  PreservedAnalyses run(LazyCallGraph::SCC *C) {
    return PreservedAnalyses::all();
  }
  static StringRef name() { return "NoOpCGSCCPass"; }
};

/// \brief No-op function pass which does nothing.
struct NoOpFunctionPass {
  PreservedAnalyses run(Function *F) { return PreservedAnalyses::all(); }
  static StringRef name() { return "NoOpFunctionPass"; }
};

} // End anonymous namespace.

static bool isModulePassName(StringRef Name) {
  if (Name == "no-op-module") return true;

#define MODULE_PASS(NAME, CREATE_PASS) if (Name == NAME) return true;
#include "PassRegistry.def"

  return false;
}

static bool isCGSCCPassName(StringRef Name) {
  if (Name == "no-op-cgscc") return true;

#define CGSCC_PASS(NAME, CREATE_PASS) if (Name == NAME) return true;
#include "PassRegistry.def"

  return false;
}

static bool isFunctionPassName(StringRef Name) {
  if (Name == "no-op-function") return true;

#define FUNCTION_PASS(NAME, CREATE_PASS) if (Name == NAME) return true;
#include "PassRegistry.def"

  return false;
}

static bool parseModulePassName(ModulePassManager &MPM, StringRef Name) {
  if (Name == "no-op-module") {
    MPM.addPass(NoOpModulePass());
    return true;
  }

#define MODULE_PASS(NAME, CREATE_PASS)                                         \
  if (Name == NAME) {                                                          \
    MPM.addPass(CREATE_PASS);                                                  \
    return true;                                                               \
  }
#include "PassRegistry.def"

  return false;
}

static bool parseCGSCCPassName(CGSCCPassManager &CGPM, StringRef Name) {
  if (Name == "no-op-cgscc") {
    CGPM.addPass(NoOpCGSCCPass());
    return true;
  }

#define CGSCC_PASS(NAME, CREATE_PASS)                                          \
  if (Name == NAME) {                                                          \
    CGPM.addPass(CREATE_PASS);                                                 \
    return true;                                                               \
  }
#include "PassRegistry.def"

  return false;
}

static bool parseFunctionPassName(FunctionPassManager &FPM, StringRef Name) {
  if (Name == "no-op-function") {
    FPM.addPass(NoOpFunctionPass());
    return true;
  }

#define FUNCTION_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME) {                                                          \
    FPM.addPass(CREATE_PASS);                                                  \
    return true;                                                               \
  }
#include "PassRegistry.def"

  return false;
}

static bool parseFunctionPassPipeline(FunctionPassManager &FPM,
                                      StringRef &PipelineText,
                                      bool VerifyEachPass) {
  for (;;) {
    // Parse nested pass managers by recursing.
    if (PipelineText.startswith("function(")) {
      FunctionPassManager NestedFPM;

      // Parse the inner pipeline inte the nested manager.
      PipelineText = PipelineText.substr(strlen("function("));
      if (!parseFunctionPassPipeline(NestedFPM, PipelineText, VerifyEachPass) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Add the nested pass manager with the appropriate adaptor.
      FPM.addPass(std::move(NestedFPM));
    } else {
      // Otherwise try to parse a pass name.
      size_t End = PipelineText.find_first_of(",)");
      if (!parseFunctionPassName(FPM, PipelineText.substr(0, End)))
        return false;
      if (VerifyEachPass)
        FPM.addPass(VerifierPass());

      PipelineText = PipelineText.substr(End);
    }

    if (PipelineText.empty() || PipelineText[0] == ')')
      return true;

    assert(PipelineText[0] == ',');
    PipelineText = PipelineText.substr(1);
  }
}

static bool parseCGSCCPassPipeline(CGSCCPassManager &CGPM,
                                      StringRef &PipelineText,
                                      bool VerifyEachPass) {
  for (;;) {
    // Parse nested pass managers by recursing.
    if (PipelineText.startswith("cgscc(")) {
      CGSCCPassManager NestedCGPM;

      // Parse the inner pipeline into the nested manager.
      PipelineText = PipelineText.substr(strlen("cgscc("));
      if (!parseCGSCCPassPipeline(NestedCGPM, PipelineText, VerifyEachPass) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Add the nested pass manager with the appropriate adaptor.
      CGPM.addPass(std::move(NestedCGPM));
    } else if (PipelineText.startswith("function(")) {
      FunctionPassManager NestedFPM;

      // Parse the inner pipeline inte the nested manager.
      PipelineText = PipelineText.substr(strlen("function("));
      if (!parseFunctionPassPipeline(NestedFPM, PipelineText, VerifyEachPass) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Add the nested pass manager with the appropriate adaptor.
      CGPM.addPass(createCGSCCToFunctionPassAdaptor(std::move(NestedFPM)));
    } else {
      // Otherwise try to parse a pass name.
      size_t End = PipelineText.find_first_of(",)");
      if (!parseCGSCCPassName(CGPM, PipelineText.substr(0, End)))
        return false;
      // FIXME: No verifier support for CGSCC passes!

      PipelineText = PipelineText.substr(End);
    }

    if (PipelineText.empty() || PipelineText[0] == ')')
      return true;

    assert(PipelineText[0] == ',');
    PipelineText = PipelineText.substr(1);
  }
}

static bool parseModulePassPipeline(ModulePassManager &MPM,
                                    StringRef &PipelineText,
                                    bool VerifyEachPass) {
  for (;;) {
    // Parse nested pass managers by recursing.
    if (PipelineText.startswith("module(")) {
      ModulePassManager NestedMPM;

      // Parse the inner pipeline into the nested manager.
      PipelineText = PipelineText.substr(strlen("module("));
      if (!parseModulePassPipeline(NestedMPM, PipelineText, VerifyEachPass) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Now add the nested manager as a module pass.
      MPM.addPass(std::move(NestedMPM));
    } else if (PipelineText.startswith("cgscc(")) {
      CGSCCPassManager NestedCGPM;

      // Parse the inner pipeline inte the nested manager.
      PipelineText = PipelineText.substr(strlen("cgscc("));
      if (!parseCGSCCPassPipeline(NestedCGPM, PipelineText, VerifyEachPass) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Add the nested pass manager with the appropriate adaptor.
      MPM.addPass(
          createModuleToPostOrderCGSCCPassAdaptor(std::move(NestedCGPM)));
    } else if (PipelineText.startswith("function(")) {
      FunctionPassManager NestedFPM;

      // Parse the inner pipeline inte the nested manager.
      PipelineText = PipelineText.substr(strlen("function("));
      if (!parseFunctionPassPipeline(NestedFPM, PipelineText, VerifyEachPass) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Add the nested pass manager with the appropriate adaptor.
      MPM.addPass(createModuleToFunctionPassAdaptor(std::move(NestedFPM)));
    } else {
      // Otherwise try to parse a pass name.
      size_t End = PipelineText.find_first_of(",)");
      if (!parseModulePassName(MPM, PipelineText.substr(0, End)))
        return false;
      if (VerifyEachPass)
        MPM.addPass(VerifierPass());

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
bool llvm::parsePassPipeline(ModulePassManager &MPM, StringRef PipelineText,
                             bool VerifyEachPass) {
  // Look at the first entry to figure out which layer to start parsing at.
  if (PipelineText.startswith("module("))
    return parseModulePassPipeline(MPM, PipelineText, VerifyEachPass) &&
           PipelineText.empty();
  if (PipelineText.startswith("cgscc(")) {
    CGSCCPassManager CGPM;
    if (!parseCGSCCPassPipeline(CGPM, PipelineText, VerifyEachPass) ||
        !PipelineText.empty())
      return false;
    MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
    return true;
  }
  if (PipelineText.startswith("function(")) {
    FunctionPassManager FPM;
    if (!parseFunctionPassPipeline(FPM, PipelineText, VerifyEachPass) ||
        !PipelineText.empty())
      return false;
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    return true;
  }

  // This isn't a direct pass manager name, look for the end of a pass name.
  StringRef FirstName =
      PipelineText.substr(0, PipelineText.find_first_of(",)"));
  if (isModulePassName(FirstName))
    return parseModulePassPipeline(MPM, PipelineText, VerifyEachPass) &&
           PipelineText.empty();

  if (isCGSCCPassName(FirstName)) {
    CGSCCPassManager CGPM;
    if (!parseCGSCCPassPipeline(CGPM, PipelineText, VerifyEachPass) ||
        !PipelineText.empty())
      return false;
    MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
    return true;
  }

  if (isFunctionPassName(FirstName)) {
    FunctionPassManager FPM;
    if (!parseFunctionPassPipeline(FPM, PipelineText, VerifyEachPass) ||
        !PipelineText.empty())
      return false;
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    return true;
  }

  return false;
}
