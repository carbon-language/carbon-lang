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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GCOV.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/system_error.h"
using namespace llvm;

static cl::opt<std::string> SourceFile(cl::Positional, cl::Required,
                                       cl::desc("SOURCEFILE"));

static cl::opt<bool> AllBlocks("a", cl::init(false),
                               cl::desc("Display all basic blocks"));
static cl::alias AllBlocksA("all-blocks", cl::aliasopt(AllBlocks));

static cl::opt<bool> BranchProb("b", cl::init(false),
                                cl::desc("Display branch probabilities"));
static cl::alias BranchProbA("branch-probabilities", cl::aliasopt(BranchProb));

static cl::opt<bool> BranchCount("c", cl::init(false),
                                 cl::desc("Display branch counts instead "
                                           "of percentages (requires -b)"));
static cl::alias BranchCountA("branch-counts", cl::aliasopt(BranchCount));

static cl::opt<bool> FuncSummary("f", cl::init(false),
                                 cl::desc("Show coverage for each function"));
static cl::alias FuncSummaryA("function-summaries", cl::aliasopt(FuncSummary));

static cl::opt<std::string>
ObjectDir("o", cl::value_desc("DIR|FILE"), cl::init(""),
          cl::desc("Find objects in DIR or based on FILE's path"));
static cl::alias ObjectDirA("object-directory", cl::aliasopt(ObjectDir));
static cl::alias ObjectDirB("object-file", cl::aliasopt(ObjectDir));

static cl::opt<bool> PreservePaths("p", cl::init(false),
                                   cl::desc("Preserve path components"));
static cl::alias PreservePathsA("preserve-paths", cl::aliasopt(PreservePaths));

static cl::opt<bool> UncondBranch("u", cl::init(false),
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

//===----------------------------------------------------------------------===//
int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "LLVM code coverage tool\n");

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

  if (InputGCNO.empty())
    InputGCNO = (CoverageFileStem.str() + ".gcno").str();
  if (InputGCDA.empty())
    InputGCDA = (CoverageFileStem.str() + ".gcda").str();

  GCOVFile GF;

  std::unique_ptr<MemoryBuffer> GCNO_Buff;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputGCNO, GCNO_Buff)) {
    errs() << InputGCNO << ": " << ec.message() << "\n";
    return 1;
  }
  GCOVBuffer GCNO_GB(GCNO_Buff.get());
  if (!GF.readGCNO(GCNO_GB)) {
    errs() << "Invalid .gcno File!\n";
    return 1;
  }

  std::unique_ptr<MemoryBuffer> GCDA_Buff;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputGCDA, GCDA_Buff)) {
    if (ec != errc::no_such_file_or_directory) {
      errs() << InputGCDA << ": " << ec.message() << "\n";
      return 1;
    }
    // Clear the filename to make it clear we didn't read anything.
    InputGCDA = "-";
  } else {
    GCOVBuffer GCDA_GB(GCDA_Buff.get());
    if (!GF.readGCDA(GCDA_GB)) {
      errs() << "Invalid .gcda File!\n";
      return 1;
    }
  }

  if (DumpGCOV)
    GF.dump();

  GCOVOptions Options(AllBlocks, BranchProb, BranchCount, FuncSummary,
                      PreservePaths, UncondBranch);
  FileInfo FI(Options);
  GF.collectLineCounts(FI);
  FI.print(InputGCNO, InputGCDA);
  return 0;
}
