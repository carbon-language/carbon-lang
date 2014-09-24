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
#include "llvm/Support/Process.h"

using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

static cl::OptionCategory ClangTidyCategory("clang-tidy options");

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp ClangTidyHelp(
    "Configuration files:\n"
    "  clang-tidy attempts to read configuration for each source file from a\n"
    "  .clang-tidy file located in the closest parent directory of the source\n"
    "  file. If any configuration options have a corresponding command-line\n"
    "  option, command-line option takes precedence. The effective\n"
    "  configuration can be inspected using -dump-config.\n\n");

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
                          "set of enabled checks.\n"
                          "This option's value is appended to the value read\n"
                          "from a .clang-tidy file, if any."),
       cl::init(""), cl::cat(ClangTidyCategory));

static cl::opt<std::string>
HeaderFilter("header-filter",
             cl::desc("Regular expression matching the names of the\n"
                      "headers to output diagnostics from. Diagnostics\n"
                      "from the main file of each translation unit are\n"
                      "always displayed.\n"
                      "Can be used together with -line-filter.\n"
                      "This option overrides the value read from a\n"
                      ".clang-tidy file."),
             cl::init(""), cl::cat(ClangTidyCategory));

static cl::opt<std::string>
LineFilter("line-filter",
           cl::desc("List of files with line ranges to filter the\n"
                    "warnings. Can be used together with\n"
                    "-header-filter. The format of the list is a JSON\n"
                    "array of objects:\n"
                    "  [\n"
                    "    {\"name\":\"file1.cpp\",\"lines\":[[1,3],[5,7]]},\n"
                    "    {\"name\":\"file2.h\"}\n"
                    "  ]"),
           cl::init(""), cl::cat(ClangTidyCategory));

static cl::opt<bool> Fix("fix", cl::desc("Fix detected errors if possible."),
                         cl::init(false), cl::cat(ClangTidyCategory));

static cl::opt<bool>
ListChecks("list-checks",
           cl::desc("List all enabled checks and exit. Use with\n"
                    "-checks='*' to list all available checks."),
           cl::init(false), cl::cat(ClangTidyCategory));

static cl::opt<bool>
DumpConfig("dump-config",
           cl::desc("Dumps configuration in the YAML format to stdout."),
           cl::init(false), cl::cat(ClangTidyCategory));

static cl::opt<bool> AnalyzeTemporaryDtors(
    "analyze-temporary-dtors",
    cl::desc("Enable temporary destructor-aware analysis in\n"
             "clang-analyzer- checks.\n"
             "This option overrides the value read from a\n"
             ".clang-tidy file."),
    cl::init(false), cl::cat(ClangTidyCategory));

static cl::opt<std::string> ExportFixes(
    "export-fixes",
    cl::desc("YAML file to store suggested fixes in. The\n"
             "stored fixes can be applied to the input source\n"
             "code with clang-apply-replacements."),
    cl::value_desc("filename"), cl::cat(ClangTidyCategory));

namespace clang {
namespace tidy {

static void printStats(const ClangTidyStats &Stats) {
  if (Stats.errorsIgnored()) {
    llvm::errs() << "Suppressed " << Stats.errorsIgnored() << " warnings (";
    StringRef Separator = "";
    if (Stats.ErrorsIgnoredNonUserCode) {
      llvm::errs() << Stats.ErrorsIgnoredNonUserCode << " in non-user code";
      Separator = ", ";
    }
    if (Stats.ErrorsIgnoredLineFilter) {
      llvm::errs() << Separator << Stats.ErrorsIgnoredLineFilter
                   << " due to line filter";
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

int clangTidyMain(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, ClangTidyCategory);

  ClangTidyGlobalOptions GlobalOptions;
  if (std::error_code Err = parseLineFilter(LineFilter, GlobalOptions)) {
    llvm::errs() << "Invalid LineFilter: " << Err.message() << "\n\nUsage:\n";
    llvm::cl::PrintHelpMessage(/*Hidden=*/false, /*Categorized=*/true);
    return 1;
  }

  ClangTidyOptions DefaultOptions;
  DefaultOptions.Checks = DefaultChecks;
  DefaultOptions.HeaderFilterRegex = HeaderFilter;
  DefaultOptions.AnalyzeTemporaryDtors = AnalyzeTemporaryDtors;
  DefaultOptions.User = llvm::sys::Process::GetEnv("USER");
  // USERNAME is used on Windows.
  if (!DefaultOptions.User)
    DefaultOptions.User = llvm::sys::Process::GetEnv("USERNAME");

  ClangTidyOptions OverrideOptions;
  if (Checks.getNumOccurrences() > 0)
    OverrideOptions.Checks = Checks;
  if (HeaderFilter.getNumOccurrences() > 0)
    OverrideOptions.HeaderFilterRegex = HeaderFilter;
  if (AnalyzeTemporaryDtors.getNumOccurrences() > 0)
    OverrideOptions.AnalyzeTemporaryDtors = AnalyzeTemporaryDtors;

  auto OptionsProvider = llvm::make_unique<FileOptionsProvider>(
      GlobalOptions, DefaultOptions, OverrideOptions);

  std::string FileName = OptionsParser.getSourcePathList().front();
  ClangTidyOptions EffectiveOptions = OptionsProvider->getOptions(FileName);
  std::vector<std::string> EnabledChecks = getCheckNames(EffectiveOptions);

  // FIXME: Allow using --list-checks without positional arguments.
  if (ListChecks) {
    llvm::outs() << "Enabled checks:";
    for (auto CheckName : EnabledChecks)
      llvm::outs() << "\n    " << CheckName;
    llvm::outs() << "\n\n";
    return 0;
  }

  if (DumpConfig) {
    EffectiveOptions.CheckOptions = getCheckOptions(EffectiveOptions);
    llvm::outs() << configurationAsText(ClangTidyOptions::getDefaults()
                                            .mergeWith(EffectiveOptions))
                 << "\n";
    return 0;
  }

  if (EnabledChecks.empty()) {
    llvm::errs() << "Error: no checks enabled.\n";
    llvm::cl::PrintHelpMessage(/*Hidden=*/false, /*Categorized=*/true);
    return 1;
  }

  std::vector<ClangTidyError> Errors;
  ClangTidyStats Stats =
      runClangTidy(std::move(OptionsProvider), OptionsParser.getCompilations(),
                   OptionsParser.getSourcePathList(), &Errors);
  handleErrors(Errors, Fix);

  if (!ExportFixes.empty() && !Errors.empty()) {
    std::error_code EC;
    llvm::raw_fd_ostream OS(ExportFixes, EC, llvm::sys::fs::F_None);
    if (EC) {
      llvm::errs() << "Error opening output file: " << EC.message() << '\n';
      return 1;
    }
    exportReplacements(Errors, OS);
  }

  printStats(Stats);
  return 0;
}

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

int main(int argc, const char **argv) {
  return clang::tidy::clangTidyMain(argc, argv);
}
