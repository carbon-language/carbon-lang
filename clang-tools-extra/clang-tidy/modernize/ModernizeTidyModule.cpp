//===--- ModernizeTidyModule.cpp - clang-tidy -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "LoopConvertCheck.h"
#include "PassByValueCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

class ModernizeModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<LoopConvertCheck>("modernize-loop-convert");
    CheckFactories.registerCheck<PassByValueCheck>("modernize-pass-by-value");
  }

  ClangTidyOptions getModuleOptions() override {
    ClangTidyOptions Options;
    auto &Opts = Options.CheckOptions;
    Opts["modernize-loop-convert.MinConfidence"] = "reasonable";
    Opts["modernize-pass-by-value.IncludeStyle"] = "llvm"; // Also: "google".
    return Options;
  }
};

// Register the ModernizeTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<ModernizeModule> X("modernize-module",
                                                       "Add modernize checks.");

} // namespace modernize

// This anchor is used to force the linker to link in the generated object file
// and thus register the ModernizeModule.
volatile int ModernizeModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
