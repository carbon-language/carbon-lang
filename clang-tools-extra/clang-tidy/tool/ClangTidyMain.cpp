//===--- tools/extra/clang-tidy/ClangTidyMain.cpp - Clang tidy tool -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file This file implements a clang-tidy tool.
///
///  This tool uses the Clang Tooling infrastructure, see
///    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
///  for details on setting it up with LLVM source tree.
///
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include <vector>

using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

cl::OptionCategory ClangTidyCategory("clang-tidy options");

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

static cl::opt<std::string> Checks(
    "checks",
    cl::desc("Regular expression matching the names of the checks to be run."),
    cl::init(".*"), cl::cat(ClangTidyCategory));
static cl::opt<bool> Fix("fix", cl::desc("Fix detected errors if possible."),
                         cl::init(false), cl::cat(ClangTidyCategory));

// FIXME: Add option to list name/description of all checks.

int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, ClangTidyCategory);

  SmallVector<clang::tidy::ClangTidyError, 16> Errors;
  clang::tidy::runClangTidy(Checks, OptionsParser.getCompilations(),
                            OptionsParser.getSourcePathList(), &Errors);
  clang::tidy::handleErrors(Errors, Fix);

  return 0;
}

namespace clang {
namespace tidy {

// This anchor is used to force the linker to link the LLVMModule.
extern volatile int LLVMModuleAnchorSource;
static int LLVMModuleAnchorDestination = LLVMModuleAnchorSource;

// This anchor is used to force the linker to link the GoogleModule.
extern volatile int GoogleModuleAnchorSource;
static int GoogleModuleAnchorDestination = GoogleModuleAnchorSource;

} // namespace tidy
} // namespace clang
