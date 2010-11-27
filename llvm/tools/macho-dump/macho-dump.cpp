//===-- macho-dump.cpp - Mach Object Dumping Tool -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a testing tool for use with the MC/Mach-O LLVM components.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

static cl::opt<std::string>
InputFile(cl::Positional, cl::desc("<input file>"), cl::init("-"));

int main(int argc, char **argv) {
  const char *ProgramName = argv[0];
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm Mach-O dumping tool\n");

  errs() << ProgramName << ": " << "not yet implemented!" << "\n";
  return 1;
}
