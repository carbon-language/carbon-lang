//===- llvm-cov.cpp - LLVM coverage tool ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// llvm-cov is a command line tools to analyze and report coverage information.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GCOV.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include <system_error>
using namespace llvm;

static cl::list<std::string> SourceFiles(cl::Positional, cl::OneOrMore,
                                         cl::desc("SOURCEFILE"));

static cl::opt<bool> AllBlocks("a", cl::Grouping, cl::init(false),
                               cl::desc("Display all basic blocks"));
static cl::alias AllBlocksA("all-blocks", cl::aliasopt(AllBlocks));

static cl::opt<bool> BranchProb("b", cl::Grouping, cl::init(false),
                                cl::desc("Display branch probabilities"));
static cl::alias BranchProbA("branch-probabilities", cl::aliasopt(BranchProb));

static cl::opt<bool> BranchCount("c", cl::Grouping, cl::init(false),
                                 cl::desc("Display branch counts instead "
                                           "of percentages (requires -b)"));
static cl::alias BranchCountA("branch-counts", cl::aliasopt(BranchCount));

static cl::opt<bool> LongNames("l", cl::Grouping, cl::init(false),
                               cl::desc("Prefix filenames with the main file"));
static cl::alias LongNamesA("long-file-names", cl::aliasopt(LongNames));

static cl::opt<bool> FuncSummary("f", cl::Grouping, cl::init(false),
                                 cl::desc("Show coverage for each function"));
static cl::alias FuncSummaryA("function-summaries", cl::aliasopt(FuncSummary));

static cl::opt<bool> NoOutput("n", cl::Grouping, cl::init(false),
                              cl::desc("Do not output any .gcov files"));
static cl::alias NoOutputA("no-output", cl::aliasopt(NoOutput));

static cl::opt<std::string>
ObjectDir("o", cl::value_desc("DIR|FILE"), cl::init(""),
          cl::desc("Find objects in DIR or based on FILE's path"));
static cl::alias ObjectDirA("object-directory", cl::aliasopt(ObjectDir));
static cl::alias ObjectDirB("object-file", cl::aliasopt(ObjectDir));

static cl::opt<bool> PreservePaths("p", cl::Grouping, cl::init(false),
                                   cl::desc("Preserve path components"));
static cl::alias PreservePathsA("preserve-paths", cl::aliasopt(PreservePaths));

static cl::opt<bool> UncondBranch("u", cl::Grouping, cl::init(false),
                                  cl::desc("Display unconditional branch info "
                                           "(requires -b)"));
static cl::alias UncondBranchA("unconditional-branches",
                               cl::aliasopt(UncondBranch));

static cl::OptionCategory DebugCat("Internal and debugging options");
static cl::opt<bool> DumpGCOV("dump", cl::init(false), cl::cat(DebugCat),
                              cl::desc("Dump the gcov file to stderr"));
static cl::opt<std::string> InputGCNO("gcno", cl::cat(DebugCat), cl::init(""),
                                      cl::desc("Override inferred gcno file"));
static cl::opt<std::string> InputGCDA("gcda", cl::cat(DebugCat), cl::init(""),
                                      cl::desc("Override inferred gcda file"));

void reportCoverage(StringRef SourceFile) {
  SmallString<128> CoverageFileStem(ObjectDir);
  if (CoverageFileStem.empty()) {
    // If no directory was specified with -o, look next to the source file.
    CoverageFileStem = sys::path::parent_path(SourceFile);
    sys::path::append(CoverageFileStem, sys::path::stem(SourceFile));
  } else if (sys::fs::is_directory(ObjectDir))
    // A directory name was given. Use it and the source file name.
    sys::path::append(CoverageFileStem, sys::path::stem(SourceFile));
  else
    // A file was given. Ignore the source file and look next to this file.
    sys::path::replace_extension(CoverageFileStem, "");

  std::string GCNO = InputGCNO.empty()
                         ? std::string(CoverageFileStem.str()) + ".gcno"
                         : InputGCNO;
  std::string GCDA = InputGCDA.empty()
                         ? std::string(CoverageFileStem.str()) + ".gcda"
                         : InputGCDA;
  GCOVFile GF;

  ErrorOr<std::unique_ptr<MemoryBuffer>> GCNO_Buff =
      MemoryBuffer::getFileOrSTDIN(GCNO);
  if (std::error_code EC = GCNO_Buff.getError()) {
    errs() << GCNO << ": " << EC.message() << "\n";
    return;
  }
  GCOVBuffer GCNO_GB(GCNO_Buff.get().get());
  if (!GF.readGCNO(GCNO_GB)) {
    errs() << "Invalid .gcno File!\n";
    return;
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> GCDA_Buff =
      MemoryBuffer::getFileOrSTDIN(GCDA);
  if (std::error_code EC = GCDA_Buff.getError()) {
    if (EC != errc::no_such_file_or_directory) {
      errs() << GCDA << ": " << EC.message() << "\n";
      return;
    }
    // Clear the filename to make it clear we didn't read anything.
    GCDA = "-";
  } else {
    GCOVBuffer GCDA_GB(GCDA_Buff.get().get());
    if (!GF.readGCDA(GCDA_GB)) {
      errs() << "Invalid .gcda File!\n";
      return;
    }
  }

  if (DumpGCOV)
    GF.dump();

  GCOVOptions Options(AllBlocks, BranchProb, BranchCount, FuncSummary,
                      PreservePaths, UncondBranch, LongNames, NoOutput);
  FileInfo FI(Options);
  GF.collectLineCounts(FI);
  FI.print(SourceFile, GCNO, GCDA);
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "LLVM code coverage tool\n");

  for (const auto &SourceFile : SourceFiles)
    reportCoverage(SourceFile);
  return 0;
}
