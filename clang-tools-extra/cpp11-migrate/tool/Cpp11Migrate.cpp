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

#include "Core/Transforms.h"
#include "Core/Transform.h"
#include "LoopConvert/LoopConvert.h"
#include "UseNullptr/UseNullptr.h"
#include "UseAuto/UseAuto.h"
#include "AddOverride/AddOverride.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Timer.h"

namespace cl = llvm::cl;
using namespace clang::tooling;

static cl::opt<RiskLevel> MaxRiskLevel(
    "risk", cl::desc("Select a maximum risk level:"),
    cl::values(clEnumValN(RL_Safe, "safe", "Only safe transformations"),
               clEnumValN(RL_Reasonable, "reasonable",
                          "Enable transformations that might change "
                          "semantics (default)"),
               clEnumValN(RL_Risky, "risky",
                          "Enable transformations that are likely to "
                          "change semantics"),
               clEnumValEnd),
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
    "report-times", cl::desc("Capture performance data and output to specified "
                             "directory. Default ./migrate_perf"),
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

class EndSyntaxArgumentsAdjuster : public ArgumentsAdjuster {
  CommandLineArguments Adjust(const CommandLineArguments &Args) {
    CommandLineArguments AdjustedArgs = Args;
    AdjustedArgs.push_back("-fsyntax-only");
    AdjustedArgs.push_back("-std=c++11");
    return AdjustedArgs;
  }
};

struct ExecutionTime {
  std::string TimerId;
  float Time;
  ExecutionTime(const std::string &TimerId, float Time)
      : TimerId(TimerId), Time(Time) {}
};

// Save execution times to a json formatted file.
void reportExecutionTimes(
    const llvm::StringRef DirectoryName,
    const std::map<std::string, std::vector<ExecutionTime> > &TimingResults) {
  // Create directory path if it doesn't exist
  llvm::sys::Path P(DirectoryName);
  P.createDirectoryOnDisk(true);

  // Get PID and current time.
  // FIXME: id_type on Windows is NOT a process id despite the function name.
  // Need to call GetProcessId() providing it what get_id() returns. For now
  // disabling PID-based file names until this is fixed properly.
  //llvm::sys::self_process *SP = llvm::sys::process::get_self();
  //id_type Pid = SP->get_id();
  unsigned Pid = 0;
  llvm::TimeRecord T = llvm::TimeRecord::getCurrentTime();

  std::string FileName;
  llvm::raw_string_ostream SS(FileName);
  SS << P.str() << "/" << static_cast<int>(T.getWallTime()) << Pid << ".json";


  std::string ErrorInfo;
  llvm::raw_fd_ostream FileStream(SS.str().c_str(), ErrorInfo);
  FileStream << "{\n";
  FileStream << "  \"Sources\" : [\n";
  for (std::map<std::string, std::vector<ExecutionTime> >::const_iterator
           I = TimingResults.begin(),
           E = TimingResults.end();
       I != E; ++I) {
    FileStream << "    {\n";
    FileStream << "      \"Source \" : \"" << I->first << "\",\n";
    FileStream << "      \"Data\" : [\n";
    for (std::vector<ExecutionTime>::const_iterator IE = I->second.begin(),
                                                    EE = I->second.end();
         IE != EE; ++IE) {
      FileStream << "        {\n";
      FileStream << "          \"TimerId\" : \"" << (*IE).TimerId << "\",\n";
      FileStream << "          \"Time\" : " << llvm::format("%6.2f", (*IE).Time)
                 << "\n";

      FileStream << "        },\n";

    }
    FileStream << "      ]\n";
    FileStream << "    },\n";
  }
  FileStream << "  ]\n";
  FileStream << "}";
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
  // Add more transform options here.

  // This causes options to be parsed.
  CommonOptionsParser OptionsParser(argc, argv);

  // Since ExecutionTimeDirectoryName could be an empty string we compare
  // against the default value when the command line option is not specified.
  bool EnableTiming = (TimingDirectoryName != NoTiming);
  std::map<std::string, std::vector<ExecutionTime> > TimingResults;

  TransformManager.createSelectedTransforms(EnableTiming);

  if (TransformManager.begin() == TransformManager.end()) {
    llvm::errs() << "No selected transforms\n";
    return 1;
  }

  FileContentsByPath FileStates1, FileStates2,
      *InputFileStates = &FileStates1, *OutputFileStates = &FileStates2;

  // Apply transforms.
  for (Transforms::const_iterator I = TransformManager.begin(),
                                  E = TransformManager.end();
       I != E; ++I) {
    if ((*I)->apply(*InputFileStates, MaxRiskLevel,
                    OptionsParser.getCompilations(),
                    OptionsParser.getSourcePathList(), *OutputFileStates) !=
        0) {
      // FIXME: Improve ClangTool to not abort if just one file fails.
      return 1;
    }
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
    std::swap(InputFileStates, OutputFileStates);
    OutputFileStates->clear();
  }

  // Final state of files is pointed at by InputFileStates.

  if (FinalSyntaxCheck) {
    ClangTool EndSyntaxTool(OptionsParser.getCompilations(),
                            OptionsParser.getSourcePathList());

    // Add c++11 support to clang.
    EndSyntaxTool.setArgumentsAdjuster(new EndSyntaxArgumentsAdjuster);

    for (FileContentsByPath::const_iterator I = InputFileStates->begin(),
                                            E = InputFileStates->end();
         I != E; ++I) {
      EndSyntaxTool.mapVirtualFile(I->first, I->second);
    }

    if (EndSyntaxTool.run(newFrontendActionFactory<clang::SyntaxOnlyAction>())
        != 0) {
      return 1;
    }
  }

  // Write results to file.
  for (FileContentsByPath::const_iterator I = InputFileStates->begin(),
                                          E = InputFileStates->end();
       I != E; ++I) {
    std::string ErrorInfo;
    llvm::raw_fd_ostream FileStream(I->first.c_str(), ErrorInfo,
                                    llvm::raw_fd_ostream::F_Binary);
    FileStream << I->second;
  }

  // Report execution times.
  if (EnableTiming && TimingResults.size() > 0) {
    std::string DirectoryName = TimingDirectoryName;
    // Use default directory name.
    if (DirectoryName == "")
      DirectoryName = "./migrate_perf";
    reportExecutionTimes(DirectoryName, TimingResults);
  }

  return 0;
}
