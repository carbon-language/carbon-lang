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

#include "llvm/Object/MachOObject.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace llvm::object;

static cl::opt<std::string>
InputFile(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<bool>
DumpSectionData("dump-section-data", cl::desc("Dump the contents of sections"),
                cl::init(false));

int main(int argc, char **argv) {
  const char *ProgramName = argv[0];
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm Mach-O dumping tool\n");

  // Load the input file.
  std::string ErrorStr;
  OwningPtr<MemoryBuffer> InputBuffer(
    MemoryBuffer::getFileOrSTDIN(InputFile, &ErrorStr));
  if (!InputBuffer) {
    errs() << ProgramName << ": " << "unable to read input: '"
           << ErrorStr << "'\n";
    return 1;
  }

  // Construct the Mach-O wrapper object.
  OwningPtr<MachOObject> InputObject(
    MachOObject::LoadFromBuffer(InputBuffer.take(), &ErrorStr));
  if (!InputObject) {
    errs() << ProgramName << ": " << "unable to load object: '"
           << ErrorStr << "'\n";
    return 1;
  }

  errs() << "ok\n";
  return 0;
}
