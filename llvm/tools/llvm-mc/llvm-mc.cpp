//===-- llvm-dis.cpp - The low-level LLVM disassembler --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility may be invoked in the following manner:
//  llvm-dis [options]      - Read LLVM bitcode from stdin, write asm to stdout
//  llvm-dis [options] x.bc - Read LLVM bitcode from the x.bc file, write asm
//                            to the x.ll file.
//  Options:
//      --help   - Output information about command line switches
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Signals.h"
using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"),
               cl::value_desc("filename"));

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm machine code playground\n");

  std::string ErrorMessage;
  
  MemoryBuffer *Buffer
    = MemoryBuffer::getFileOrSTDIN(InputFilename, &ErrorMessage);

  if (Buffer == 0) {
    errs() << argv[0] << ": ";
    if (ErrorMessage.size())
      errs() << ErrorMessage << "\n";
    else
      errs() << "input file didn't read correctly.\n";
    return 1;
  }
  
  
  
  return 0;
}

