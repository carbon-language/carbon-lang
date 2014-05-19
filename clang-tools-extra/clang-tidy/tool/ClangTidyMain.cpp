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

using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

static cl::OptionCategory ClangTidyCategory("clang-tidy options");

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

const char DefaultChecks[] =
    "*,"                       // Enable all checks, except these:
    "-clang-analyzer-alpha*,"  // Too many false positives.
    "-llvm-include-order,"     // Not implemented yet.
    "-google-*,";              // Doesn't apply to LLVM.
static cl::opt<std::string>
Checks("checks", cl::desc("Comma-separated list of globs with optional '-'\n"
                          "prefix. Globs are processed in order of appearance\n"
                          "in the list. Globs without '-' prefix add checks\n"
                          "with matching names to the set, globs with the '-'\n"
                          "prefix remove checks with matching names from the\n"
                          "set of enabled checks."),
       cl::init(""), cl::cat(ClangTidyCategory));
static cl::opt<std::string>
HeaderFilter("header-filter",
             cl::desc("Regular expression matching the names of the\n"
                      "headers to output diagnostics from.\n"
                      "Diagnostics from the main file of each\n"
                      "translation unit are always displayed."),
             cl::init(""), cl::cat(ClangTidyCategory));
static cl::opt<bool> Fix("fix", cl::desc("Fix detected errors if possible."),
                         cl::init(false), cl::cat(ClangTidyCategory));

static cl::opt<bool>
ListChecks("list-checks",
           cl::desc("List all enabled checks and exit. Use with\n"
                    "-checks='*' to list all available checks."),
           cl::init(false), cl::cat(ClangTidyCategory));

static cl::opt<bool>
AnalyzeTemporaryDtors("analyze-temporary-dtors",
                      cl::desc("Enable temporary destructor-aware analysis in\n"
                               "clang-analyzer- checks."),
                      cl::init(false), cl::cat(ClangTidyCategory));

static void printStats(const clang::tidy::ClangTidyStats &Stats) {
  unsigned ErrorsIgnored = Stats.ErrorsIgnoredNOLINT +
                           Stats.ErrorsIgnoredCheckFilter +
                           Stats.ErrorsIgnoredNonUserCode;
  if (ErrorsIgnored) {
    llvm::errs() << "Suppressed " << ErrorsIgnored << " warnings (";
    StringRef Separator = "";
    if (Stats.ErrorsIgnoredNonUserCode) {
      llvm::errs() << Stats.ErrorsIgnoredNonUserCode << " in non-user code";
      Separator = ", ";
    }
    if (Stats.ErrorsIgnoredNOLINT) {
      llvm::errs() << Separator << Stats.ErrorsIgnoredNOLINT << " NOLINT";
      Separator = ", ";
    }
    if (Stats.ErrorsIgnoredCheckFilter)
      llvm::errs() << Separator << Stats.ErrorsIgnoredCheckFilter
                   << " with check filters";
    llvm::errs() << ").\n";
    if (Stats.ErrorsIgnoredNonUserCode)
      llvm::errs() << "Use -header-filter='.*' to display errors from all "
                      "non-system headers.\n";
  }
}

int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, ClangTidyCategory);

  clang::tidy::ClangTidyOptions Options;
  Options.Checks = DefaultChecks + Checks;
  Options.HeaderFilterRegex = HeaderFilter;
  Options.AnalyzeTemporaryDtors = AnalyzeTemporaryDtors;

  // FIXME: Allow using --list-checks without positional arguments.
  if (ListChecks) {
    llvm::outs() << "Enabled checks:";
    for (auto CheckName : clang::tidy::getCheckNames(Options))
      llvm::outs() << "\n    " << CheckName;
    llvm::outs() << "\n\n";
    return 0;
  }

  std::vector<clang::tidy::ClangTidyError> Errors;
  clang::tidy::ClangTidyStats Stats =
      clang::tidy::runClangTidy(Options, OptionsParser.getCompilations(),
                                OptionsParser.getSourcePathList(), &Errors);
  clang::tidy::handleErrors(Errors, Fix);

  printStats(Stats);
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

// This anchor is used to force the linker to link the MiscModule.
extern volatile int MiscModuleAnchorSource;
static int MiscModuleAnchorDestination = MiscModuleAnchorSource;

} // namespace tidy
} // namespace clang
