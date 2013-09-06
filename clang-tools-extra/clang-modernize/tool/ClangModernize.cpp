//===-- ClangModernize.cpp - Main file for Clang modernization tool -------===//
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
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang-apply-replacements/Tooling/ApplyReplacements.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"

namespace cl = llvm::cl;
using namespace clang;
using namespace clang::tooling;

TransformOptions GlobalOptions;

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::opt<std::string> BuildPath(
    "p", cl::desc("Build Path"), cl::Optional);
static cl::list<std::string> SourcePaths(
    cl::Positional, cl::desc("<source0> [... <sourceN>]"), cl::OneOrMore);
static cl::extrahelp MoreHelp(
    "EXAMPLES:\n\n"
    "Apply all transforms on a given file, no compilation database:\n\n"
    "  clang-modernize path/to/file.cpp -- -Ipath/to/include/\n"
    "\n"
    "Convert for loops to the new ranged-based for loops on all files in a "
    "subtree\nand reformat the code automatically using the LLVM style:\n\n"
    "  find path/in/subtree -name '*.cpp' -exec \\\n"
    "    clang-modernize -p build/path -format-style=LLVM -loop-convert {} ';'\n"
    "\n"
    "Make use of both nullptr and the override specifier, using git ls-files:\n"
    "\n"
    "  git ls-files '*.cpp' | xargs -I{} clang-modernize -p build/path \\\n"
    "    -use-nullptr -add-override -override-macros {}\n"
    "\n"
    "Apply all transforms supported by both clang >= 3.0 and gcc >= 4.7:\n\n"
    "  clang-modernize -for-compilers=clang-3.0,gcc-4.7 foo.cpp -- -Ibar\n");

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

static cl::opt<bool>
SerializeReplacements("serialize-replacements",
                      cl::Hidden,
                      cl::desc("Serialize translation unit replacements to "
                               "disk instead of changing files."),
                      cl::init(false));

cl::opt<std::string> SupportedCompilers(
    "for-compilers", cl::value_desc("string"),
    cl::desc("Select transforms targeting the intersection of\n"
             "language features supported by the given compilers.\n"
             "Takes a comma-seperated list of <compiler>-<version>.\n"
             "\t<compiler> can be any of: clang, gcc, icc, msvc\n"
             "\t<version> is <major>[.<minor>]\n"));

/// \brief Extract the minimum compiler versions as requested on the command
/// line by the switch \c -for-compilers.
///
/// \param ProgName The name of the program, \c argv[0], used to print errors.
/// \param Error If an error occur while parsing the versions this parameter is
/// set to \c true, otherwise it will be left untouched.
static CompilerVersions handleSupportedCompilers(const char *ProgName,
                                                 bool &Error) {
  if (SupportedCompilers.getNumOccurrences() == 0)
    return CompilerVersions();
  CompilerVersions RequiredVersions;
  llvm::SmallVector<llvm::StringRef, 4> Compilers;

  llvm::StringRef(SupportedCompilers).split(Compilers, ",");

  for (llvm::SmallVectorImpl<llvm::StringRef>::iterator I = Compilers.begin(),
                                                        E = Compilers.end();
       I != E; ++I) {
    llvm::StringRef Compiler, VersionStr;
    llvm::tie(Compiler, VersionStr) = I->split('-');
    Version *V = llvm::StringSwitch<Version *>(Compiler)
        .Case("clang", &RequiredVersions.Clang)
        .Case("gcc", &RequiredVersions.Gcc).Case("icc", &RequiredVersions.Icc)
        .Case("msvc", &RequiredVersions.Msvc).Default(NULL);

    if (V == NULL) {
      llvm::errs() << ProgName << ": " << Compiler
                   << ": unsupported platform\n";
      Error = true;
      continue;
    }
    if (VersionStr.empty()) {
      llvm::errs() << ProgName << ": " << *I
                   << ": missing version number in platform\n";
      Error = true;
      continue;
    }

    Version Version = Version::getFromString(VersionStr);
    if (Version.isNull()) {
      llvm::errs()
          << ProgName << ": " << *I
          << ": invalid version, please use \"<major>[.<minor>]\" instead of \""
          << VersionStr << "\"\n";
      Error = true;
      continue;
    }
    // support the lowest version given
    if (V->isNull() || Version < *V)
      *V = Version;
  }
  return RequiredVersions;
}

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

/// \brief Use \c ChangesReformatter to reformat all changed regions of all
/// files stored in \c Overrides and write the result to disk.
///
/// \returns \li true if reformatting replacements were successfully applied
///              without conflicts and all files were successfully written to
///              disk.
///          \li false if reformatting could not be successfully applied or
///              if at least one file failed to write to disk.
void reformat(Reformatter &ChangesReformatter, FileOverrides &Overrides,
              DiagnosticsEngine &Diagnostics) {
  FileManager Files((FileSystemOptions()));
  SourceManager SM(Diagnostics, Files);

  replace::TUReplacements AllReplacements(1);
  ChangesReformatter.reformatChanges(Overrides, SM,
                                     AllReplacements.front().Replacements);

  replace::FileToReplacementsMap GroupedReplacements;
  if (!replace::mergeAndDeduplicate(AllReplacements, GroupedReplacements, SM)) {
    llvm::errs() << "Warning: Reformatting produced conflicts.\n";
    return;
  }

  Rewriter DestRewriter(SM, LangOptions());
  if (!replace::applyReplacements(GroupedReplacements, DestRewriter)) {
    llvm::errs() << "Warning: Failed to apply reformatting conflicts!\n";
    return;
  }

  Overrides.updateState(DestRewriter);
}

bool serializeReplacements(const replace::TUReplacements &Replacements) {
  bool Errors = false;
  for (replace::TUReplacements::const_iterator I = Replacements.begin(),
                                               E = Replacements.end();
       I != E; ++I) {
    llvm::SmallString<128> ReplacementsFileName;
    llvm::SmallString<64> Error;
    bool Result = generateReplacementsFileName(I->MainSourceFile,
                                               ReplacementsFileName, Error);
    if (!Result) {
      llvm::errs() << "Failed to generate replacements filename:" << Error
                   << "\n";
      Errors = true;
      continue;
    }

    std::string ErrorInfo;
    llvm::raw_fd_ostream ReplacementsFile(ReplacementsFileName.c_str(),
                                          ErrorInfo, llvm::sys::fs::F_Binary);
    if (!ErrorInfo.empty()) {
      llvm::errs() << "Error opening file: " << ErrorInfo << "\n";
      Errors = true;
      continue;
    }
    llvm::yaml::Output YAML(ReplacementsFile);
    YAML << const_cast<TranslationUnitReplacements &>(*I);
  }
  return !Errors;
}

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  Transforms TransformManager;

  TransformManager.registerTransforms();

  // Parse options and generate compilations.
  OwningPtr<CompilationDatabase> Compilations(
      FixedCompilationDatabase::loadFromCommandLine(argc, argv));
  cl::ParseCommandLineOptions(argc, argv);

  if (!Compilations) {
    std::string ErrorMessage;
    if (BuildPath.getNumOccurrences() > 0) {
      Compilations.reset(CompilationDatabase::autoDetectFromDirectory(
          BuildPath, ErrorMessage));
    } else {
      Compilations.reset(CompilationDatabase::autoDetectFromSource(
          SourcePaths[0], ErrorMessage));
      // If no compilation database can be detected from source then we create
      // a new FixedCompilationDatabase with c++11 support.
      if (!Compilations) {
        std::string CommandLine[] = {"-std=c++11"};
        Compilations.reset(new FixedCompilationDatabase(".", CommandLine));
      }
    }
    if (!Compilations)
      llvm::report_fatal_error(ErrorMessage);
  }

  // Since ExecutionTimeDirectoryName could be an empty string we compare
  // against the default value when the command line option is not specified.
  GlobalOptions.EnableTiming = (TimingDirectoryName != NoTiming);

  // Check the reformatting style option
  bool CmdSwitchError = false;
  llvm::OwningPtr<Reformatter> ChangesReformatter(
      handleFormatStyle(argv[0], CmdSwitchError));

  CompilerVersions RequiredVersions =
      handleSupportedCompilers(argv[0], CmdSwitchError);
  if (CmdSwitchError)
    return 1;

  // Populate the ModifiableHeaders structure.
  GlobalOptions.ModifiableHeaders
      .readListFromString(IncludePaths, ExcludePaths);
  GlobalOptions.ModifiableHeaders
      .readListFromFile(IncludeFromFile, ExcludeFromFile);

  TransformManager.createSelectedTransforms(GlobalOptions, RequiredVersions);

  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts(
      new DiagnosticOptions());
  DiagnosticsEngine Diagnostics(
      llvm::IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()),
      DiagOpts.getPtr());

  // FIXME: Make this DiagnosticsEngine available to all Transforms probably via
  // GlobalOptions.

  if (TransformManager.begin() == TransformManager.end()) {
    if (SupportedCompilers.empty())
      llvm::errs() << argv[0] << ": no selected transforms\n";
    else
      llvm::errs() << argv[0]
                   << ": no transforms available for specified compilers\n";
    return 1;
  }

  // If SerializeReplacements is requested, then change reformatting must be
  // turned off and only one transform should be requested. Reformatting is
  // basically another transform so even if there's only one other transform,
  // the reformatting pass would make two.
  if (SerializeReplacements &&
      (std::distance(TransformManager.begin(), TransformManager.end()) > 1 ||
       ChangesReformatter)) {
    llvm::errs() << "Serialization of replacements requested for multiple "
                    "transforms.\nChanges from only one transform can be "
                    "serialized.\n";
    return 1;
  }

  SourcePerfData PerfData;
  FileOverrides FileStates;

  for (Transforms::const_iterator I = TransformManager.begin(),
                                  E = TransformManager.end();
       I != E; ++I) {
    Transform *T = *I;

    if (T->apply(FileStates, *Compilations, SourcePaths) != 0) {
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

    // Collect all TranslationUnitReplacements generated from the translation
    // units the transform worked on and store them in AllReplacements.
    replace::TUReplacements AllReplacements;
    const TUReplacementsMap &ReplacementsMap = T->getAllReplacements();
    const TranslationUnitReplacements &(
        TUReplacementsMap::value_type::*getValue)() const =
        &TUReplacementsMap::value_type::getValue;
    std::transform(ReplacementsMap.begin(), ReplacementsMap.end(),
                   std::back_inserter(AllReplacements),
                   std::mem_fun_ref(getValue));

    if (SerializeReplacements)
      serializeReplacements(AllReplacements);

    FileManager Files((FileSystemOptions()));
    SourceManager SM(Diagnostics, Files);

    // Make sure SourceManager is updated to have the same initial state as the
    // transforms.
    FileStates.applyOverrides(SM);

    replace::FileToReplacementsMap GroupedReplacements;
    if (!replace::mergeAndDeduplicate(AllReplacements, GroupedReplacements,
                                      SM)) {
      llvm::outs() << "Transform " << T->getName()
                   << " resulted in conflicts. Discarding all "
                   << "replacements.\n";
      continue;
    }

    // Apply replacements and update FileStates with new state.
    Rewriter DestRewriter(SM, LangOptions());
    if (!replace::applyReplacements(GroupedReplacements, DestRewriter)) {
      llvm::outs() << "Some replacements failed to apply. Discarding "
                      "all replacements.\n";
      continue;
    }

    // Update contents of files in memory to serve as initial state for next
    // transform.
    FileStates.updateState(DestRewriter);

    // Update changed ranges for reformatting
    if (ChangesReformatter)
      FileStates.adjustChangedRanges(GroupedReplacements);
  }

  // Skip writing final file states to disk if we were asked to serialize
  // replacements. Otherwise reformat changes if reformatting is enabled.
  if (!SerializeReplacements) {
    if (ChangesReformatter)
       reformat(*ChangesReformatter, FileStates, Diagnostics);
    FileStates.writeToDisk(Diagnostics);
  }

  if (FinalSyntaxCheck)
    if (!doSyntaxCheck(*Compilations, SourcePaths, FileStates))
      return 1;

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

// These anchors are used to force the linker to link the transforms
extern volatile int AddOverrideTransformAnchorSource;
extern volatile int LoopConvertTransformAnchorSource;
extern volatile int PassByValueTransformAnchorSource;
extern volatile int ReplaceAutoPtrTransformAnchorSource;
extern volatile int UseAutoTransformAnchorSource;
extern volatile int UseNullptrTransformAnchorSource;

static int TransformsAnchorsDestination[] = {
  AddOverrideTransformAnchorSource,
  LoopConvertTransformAnchorSource,
  PassByValueTransformAnchorSource,
  ReplaceAutoPtrTransformAnchorSource,
  UseAutoTransformAnchorSource,
  UseNullptrTransformAnchorSource
};
