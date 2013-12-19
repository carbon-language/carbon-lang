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

#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/GCOV.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/system_error.h"
using namespace llvm;

static cl::opt<bool>
DumpGCOV("dump", cl::init(false), cl::desc("dump gcov file"));

static cl::opt<std::string>
InputGCNO("gcno", cl::desc("<input gcno file>"), cl::init(""));

static cl::opt<std::string>
InputGCDA("gcda", cl::desc("<input gcda file>"), cl::init(""));

static cl::opt<bool>
AllBlocks("a", cl::init(false), cl::desc("display all block info"));

static cl::opt<bool>
BranchInfo("b", cl::init(false), cl::desc("display branch info"));

static cl::opt<bool>
BranchCount("c", cl::init(false), cl::desc("display branch counts instead of \
                                            probabilities (requires -b)"));

static cl::opt<bool>
FuncCoverage("f", cl::init(false), cl::desc("output function coverage"));

static cl::opt<bool>
UncondBranch("u", cl::init(false), cl::desc("display unconditional branch info \
                                             (requires -b)"));

//===----------------------------------------------------------------------===//
int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm coverage tool\n");

  GCOVFile GF;
  if (InputGCNO.empty())
    errs() << " " << argv[0] << ": No gcov input file!\n";

  OwningPtr<MemoryBuffer> GCNO_Buff;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputGCNO, GCNO_Buff)) {
    errs() << InputGCNO << ": " << ec.message() << "\n";
    return 1;
  }
  GCOVBuffer GCNO_GB(GCNO_Buff.get());
  if (!GF.readGCNO(GCNO_GB)) {
    errs() << "Invalid .gcno File!\n";
    return 1;
  }

  if (!InputGCDA.empty()) {
    OwningPtr<MemoryBuffer> GCDA_Buff;
    if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputGCDA, GCDA_Buff)) {
      errs() << InputGCDA << ": " << ec.message() << "\n";
      return 1;
    }
    GCOVBuffer GCDA_GB(GCDA_Buff.get());
    if (!GF.readGCDA(GCDA_GB)) {
      errs() << "Invalid .gcda File!\n";
      return 1;
    }
  }

  if (DumpGCOV)
    GF.dump();

  GCOVOptions Options(AllBlocks, BranchInfo, BranchCount, FuncCoverage,
                      UncondBranch);
  FileInfo FI(Options);
  GF.collectLineCounts(FI);
  FI.print(InputGCNO, InputGCDA);
  return 0;
}
