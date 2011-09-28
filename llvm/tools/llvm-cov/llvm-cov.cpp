//===- tools/llvm-cov/llvm-cov.cpp - LLVM coverage tool -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
// llvm-cov is a command line tools to analyze and report coverage information.
//
//
//===----------------------------------------------------------------------===//

#include "GCOVReader.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/CommandLine.h"
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


//===----------------------------------------------------------------------===//
int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm cov\n");


  GCOVFile GF;
  if (InputGCNO.empty())
    errs() << " " << argv[0] << ": No gcov input file!\n";

  OwningPtr<MemoryBuffer> Buff;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputGCNO, Buff)) {
    errs() << InputGCNO << ": " << ec.message() << "\n";
    return 1;
  }
  GCOVBuffer GB(Buff.take());

  if (!GF.read(GB)) {
    errs() << "Invalid .gcno File!\n";
    return 1;
  }

  if (!InputGCDA.empty()) {
    OwningPtr<MemoryBuffer> Buff2;
    if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputGCDA, Buff2)) {
      errs() << InputGCDA << ": " << ec.message() << "\n";
      return 1;
    }
    GCOVBuffer GB2(Buff2.take());
    
    if (!GF.read(GB2)) {
      errs() << "Invalid .gcda File!\n";
      return 1;
    }
  }


  if (DumpGCOV)
    GF.dump();

  FileInfo FI;
  GF.collectLineCounts(FI);
  return 0;
}
