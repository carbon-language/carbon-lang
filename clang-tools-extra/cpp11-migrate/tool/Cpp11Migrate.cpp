//===-- cpp11-migrate/Cpp11Migrate.cpp - Main file C++11 migration tool ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements the C++11 feature migration tool main function
/// and transformation framework.
///
/// See user documentation for usage instructions.
///
//===----------------------------------------------------------------------===//

#include "Core/FileOverrides.h"
#include "Core/PerfSupport.h"
#include "Core/SyntaxCheck.h"
#include "Core/Transform.h"
#include "Core/Transforms.h"
#include "Core/Reformatting.h"
#include "LoopConvert/LoopConvert.h"
#include "UseNullptr/UseNullptr.h"
#include "UseAuto/UseAuto.h"
#include "AddOverride/AddOverride.h"
#include "ReplaceAutoPtr/ReplaceAutoPtr.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"

namespace cl = llvm::cl;
using namespace clang;
using namespace clang::tooling;

TransformOptions GlobalOptions;

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp MoreHelp(
    "EXAMPLES:\n\n"
    "Use 'auto' type specifier, no compilation database:\n\n"
    "  cpp11-migrate -use-auto path/to/file.cpp -- -Ipath/to/include/\n"
    "\n"
    "Convert for loops to the new ranged-based for loops on all files in a "
    "subtree\nand reformat the code automatically using the LLVM style:\n\n"
    "  find path/in/subtree -name '*.cpp' -exec \\\n"
    "    cpp11-migrate -p build/path -format-style=LLVM -loop-convert {} ';'\n"
    "\n"
    "Make use of both nullptr and the override specifier, using git ls-files:\n"
    "\n"
    "  git ls-files '*.cpp' | xargs -I{} cpp11-migrate -p build/path \\\n"
    "    -use-nullptr -add-override -override-macros {}\n");

static cl::opt<RiskLevel, /*ExternalStorage=*/true> MaxRiskLevel(
    "risk", cl::desc("Select a maximum risk level:"),
    cl::values(clEnumValN(RL_Safe, "safe", "Only safe transformations"),
               clEnumValN(RL_Reasonable, "reasonable",
                          "Enable transformations that might change "
                          "semantics (default)"),
               clEnumValN(RL_Risky, "risky",
                          "Enable transformations that are likely to "
                          "change semantics"),
               clEnumValEnd),
    cl::location(GlobalOptions.MaxRiskLevel),
    cl::init(RL_Reasonable));

static cl::opt<bool> FinalSyntaxCheck(
    "final-syntax-check",
    cl::desc("Check for correct syntax after applying transformations"),
    cl::init(false));

static cl::opt<std::string> FormatStyleOpt(
    "format-style",
    cl::desc("Coding style to use on the replacements, either a builtin style\n"
             "or a YAML config file (see: clang-format -dump-config).\n"
             "Currently supports 4 builtins style:\n"
             "  LLVM, Google, Chromium, Mozilla.\n"),
    cl::value_desc("string"));

static cl::opt<bool>
SummaryMode("summary", cl::desc("Print transform summary"),
            cl::init(false));

const char NoTiming[] = "no_timing";
static cl::opt<std::string> TimingDirectoryName(
    "perf", cl::desc("Capture performance data and output to specified "
                     "directory. Default: ./migrate_perf"),
    cl::init(NoTiming), cl::ValueOptional, cl::value_desc("directory name"));

// TODO: Remove cl::Hidden when functionality for acknowledging include/exclude
// options are implemented in the tool.
static cl::opt<std::string>
IncludePaths("include", cl::Hidden,
             cl::desc("Comma seperated list of paths to consider to be "
                      "transformed"));
static cl::opt<std::string>
ExcludePaths("exclude", cl::Hidden,
             cl::desc("Comma seperated list of paths that can not "
                      "be transformed"));
static cl::opt<std::string>
IncludeFromFile("include-from", cl::Hidden, cl::value_desc("filename"),
                cl::desc("File containing a list of paths to consider to "
                         "be transformed"));
static cl::opt<std::string>
ExcludeFromFile("exclude-from", cl::Hidden, cl::value_desc("filename"),
                cl::desc("File containing a list of paths that can not be "
                         "transforms"));

// Header modifications will probably be always on eventually. For now, they
// need to be explicitly enabled.
static cl::opt<bool, /*ExternalStorage=*/true> EnableHeaderModifications(
    "headers",
    cl::Hidden, // Experimental feature for now.
    cl::desc("Enable modifications to headers"),
    cl::location(GlobalOptions.EnableHeaderModifications),
    cl::init(false));

/// \brief Creates the Reformatter if the format style option is provided,
/// return a null pointer otherwise.
///
/// \param ProgName The name of the program, \c argv[0], used to print errors.
/// \param Error If the \c -format-style is provided but with wrong parameters
/// this is parameter is set to \c true, left untouched otherwise. An error
/// message is printed with an explanation.
static Reformatter *handleFormatStyle(const char *ProgName, bool &Error) {
  if (FormatStyleOpt.getNumOccurrences() > 0) {
    format::FormatStyle Style;
    if (!format::getPredefinedStyle(FormatStyleOpt, &Style)) {
      llvm::StringRef ConfigFilePath = FormatStyleOpt;
      llvm::OwningPtr<llvm::MemoryBuffer> Text;
      llvm::error_code ec;

      ec = llvm::MemoryBuffer::getFile(ConfigFilePath, Text);
      if (!ec)
        ec = parseConfiguration(Text->getBuffer(), &Style);

      if (ec) {
        llvm::errs() << ProgName << ": invalid format style " << FormatStyleOpt
                     << ": " << ec.message() << "\n";
        Error = true;
        return 0;
      }
    }

    // force mode to C++11
    Style.Standard = clang::format::FormatStyle::LS_Cpp11;
    return new Reformatter(Style);
  }
  return 0;
}

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  Transforms TransformManager;

  TransformManager.registerTransform(
      "loop-convert", "Make use of range-based for loops where possible",
      &ConstructTransform<LoopConvertTransform>);
  TransformManager.registerTransform(
      "use-nullptr", "Make use of nullptr keyword where possible",
      &ConstructTransform<UseNullptrTransform>);
  TransformManager.registerTransform(
      "use-auto", "Use of 'auto' type specifier",
      &ConstructTransform<UseAutoTransform>);
  TransformManager.registerTransform(
      "add-override", "Make use of override specifier where possible",
      &ConstructTransform<AddOverrideTransform>);
  TransformManager.registerTransform(
      "replace-auto_ptr", "Replace auto_ptr (deprecated) by unique_ptr"
                          " (EXPERIMENTAL)",
      &ConstructTransform<ReplaceAutoPtrTransform>);
  // Add more transform options here.

  // This causes options to be parsed.
  CommonOptionsParser OptionsParser(argc, argv);

  // Since ExecutionTimeDirectoryName could be an empty string we compare
  // against the default value when the command line option is not specified.
  GlobalOptions.EnableTiming = (TimingDirectoryName != NoTiming);

  // Check the reformatting style option
  bool BadStyle = false;
  llvm::OwningPtr<Reformatter> ChangesReformatter(
      handleFormatStyle(argv[0], BadStyle));
  if (BadStyle)
    return 1;

  // Populate the ModifiableHeaders structure if header modifications are
  // enabled.
  if (GlobalOptions.EnableHeaderModifications) {
    GlobalOptions.ModifiableHeaders
        .readListFromString(IncludePaths, ExcludePaths);
    GlobalOptions.ModifiableHeaders
        .readListFromFile(IncludeFromFile, ExcludeFromFile);
  }

  TransformManager.createSelectedTransforms(GlobalOptions);

  if (TransformManager.begin() == TransformManager.end()) {
    llvm::errs() << argv[0] << ": No selected transforms\n";
    return 1;
  }

  // if reformatting is enabled we wants to track file changes so that it's
  // possible to reformat them.
  bool TrackReplacements = static_cast<bool>(ChangesReformatter);
  FileOverrides FileStates(TrackReplacements);
  SourcePerfData PerfData;

  // Apply transforms.
  for (Transforms::const_iterator I = TransformManager.begin(),
                                  E = TransformManager.end();
       I != E; ++I) {
    Transform *T = *I;

    if (T->apply(FileStates, OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList()) !=
        0) {
      // FIXME: Improve ClangTool to not abort if just one file fails.
      return 1;
    }

    if (GlobalOptions.EnableTiming)
      collectSourcePerfData(*T, PerfData);

    if (SummaryMode) {
      llvm::outs() << "Transform: " << T->getName()
                   << " - Accepted: " << T->getAcceptedChanges();
      if (T->getChangesNotMade()) {
        llvm::outs() << " - Rejected: " << T->getRejectedChanges()
                     << " - Deferred: " << T->getDeferredChanges();
      }
      llvm::outs() << "\n";
    }
  }

  // Reformat changes if a reformatter is provided.
  if (ChangesReformatter)
    for (FileOverrides::const_iterator I = FileStates.begin(),
                                       E = FileStates.end();
         I != E; ++I) {
      SourceOverrides &Overrides = *I->second;
      ChangesReformatter->reformatChanges(Overrides);
    }

  if (FinalSyntaxCheck)
    if (!doSyntaxCheck(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList(), FileStates))
      return 1;

  // Write results to file.
  for (FileOverrides::const_iterator I = FileStates.begin(),
                                     E = FileStates.end();
       I != E; ++I) {
    const SourceOverrides &Overrides = *I->second;
    if (Overrides.isSourceOverriden()) {
      std::string ErrorInfo;
      std::string MainFileName = I->getKey();
      llvm::raw_fd_ostream FileStream(MainFileName.c_str(), ErrorInfo,
                                      llvm::sys::fs::F_Binary);
      FileStream << Overrides.getMainFileContent();
    }

    // FIXME: The Migrator shouldn't be responsible for writing headers to disk.
    // Instead, it should write replacement info and another tool should take
    // all replacement info for a header from possibly many other migration
    // processes and merge it into a final form. For now, the updated header is
    // written to disk for testing purposes.
    for (HeaderOverrides::const_iterator HeaderI = Overrides.headers_begin(),
                                         HeaderE = Overrides.headers_end();
         HeaderI != HeaderE; ++HeaderI) {
      assert(!HeaderI->second.FileOverride.empty() &&
             "A header override should not be empty");
      std::string ErrorInfo;
      std::string HeaderFileName = HeaderI->getKey();
      llvm::raw_fd_ostream HeaderStream(HeaderFileName.c_str(), ErrorInfo,
                                        llvm::sys::fs::F_Binary);
      HeaderStream << HeaderI->second.FileOverride;
    }
  }

  // Report execution times.
  if (GlobalOptions.EnableTiming && !PerfData.empty()) {
    std::string DirectoryName = TimingDirectoryName;
    // Use default directory name.
    if (DirectoryName.empty())
      DirectoryName = "./migrate_perf";
    writePerfDataJSON(DirectoryName, PerfData);
  }

  return 0;
}
