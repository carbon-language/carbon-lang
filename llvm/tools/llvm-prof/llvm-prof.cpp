//===- llvm-prof.cpp - Read in and process llvmprof.out data files --------===//
// 
//                      The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This tools is meant for use with the various LLVM profiling instrumentation
// passes.  It reads in the data file produced by executing an instrumented
// program, and outputs a nice report.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bytecode/Reader.h"
#include "Support/CommandLine.h"

namespace {
  cl::opt<std::string> 
  BytecodeFile(cl::Positional, cl::desc("<program bytecode file>"),
               cl::Required);

  cl::opt<std::string> 
  ProfileDataFile(cl::Positional, cl::desc("<llvmprof.out file>"),
                  cl::Optional, cl::init("llvmprof.out"));
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm profile dump decoder\n");
 



  return 0;
}
