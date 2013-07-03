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

#include "Core/SyntaxCheck.h"
#include "Core/Transforms.h"
#include "Core/Transform.h"
#include "Core/PerfSupport.h"
#include "LoopConvert/LoopConvert.h"
#include "UseNullptr/UseNullptr.h"
#include "UseAuto/UseAuto.h"
#include "AddOverride/AddOverride.h"
#include "ReplaceAutoPtr/ReplaceAutoPtr.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Signals.h"

namespace cl = llvm::cl;
using namespace clang::tooling;

TransformOptions GlobalOptions;

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
    llvm::errs() << "No selected transforms\n";
    return 1;
  }

  FileOverrides FileStates;
  SourcePerfData PerfData;

  // Apply transforms.
  for (Transforms::const_iterator I = TransformManager.begin(),
                                  E = TransformManager.end();
       I != E; ++I) {
    if ((*I)->apply(FileStates, OptionsParser.getCompilations(),
                    OptionsParser.getSourcePathList()) !=
        0) {
      // FIXME: Improve ClangTool to not abort if just one file fails.
      return 1;
    }

    if (GlobalOptions.EnableTiming)
      collectSourcePerfData(**I, PerfData);

    if (SummaryMode) {
      llvm::outs() << "Transform: " << (*I)->getName()
                   << " - Accepted: "
                   << (*I)->getAcceptedChanges();
      if ((*I)->getChangesNotMade()) {
         llvm::outs() << " - Rejected: "
                      << (*I)->getRejectedChanges()
                      << " - Deferred: "
                      << (*I)->getDeferredChanges();
      }
      llvm::outs() << "\n";
    }
  }

  if (FinalSyntaxCheck)
    if (!doSyntaxCheck(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList(), FileStates))
      return 1;

  // Write results to file.
  for (FileOverrides::const_iterator I = FileStates.begin(),
                                     E = FileStates.end();
       I != E; ++I) {
    if (I->second.isSourceOverriden()) {
      std::string ErrorInfo;
      llvm::raw_fd_ostream FileStream(I->first.c_str(), ErrorInfo,
                                      llvm::raw_fd_ostream::F_Binary);
      FileStream << I->second.MainFileOverride;
    }

    // FIXME: The Migrator shouldn't be responsible for writing headers
    // to disk. Instead, it should write replacement info and another tool
    // should take all replacement info for a header from possibly many other
    // migration processes and merge it into a final form. For now, the 
    // updated header is written to disk for testing purposes.
    for (HeaderOverrides::const_iterator HeaderI = I->second.Headers.begin(),
                                         HeaderE = I->second.Headers.end();
         HeaderI != HeaderE; ++HeaderI) {
      assert(!HeaderI->second.FileOverride.empty() &&
             "A header override should not be empty");
      std::string ErrorInfo;
      llvm::raw_fd_ostream HeaderStream(HeaderI->first.c_str(), ErrorInfo,
                                        llvm::raw_fd_ostream::F_Binary);
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
