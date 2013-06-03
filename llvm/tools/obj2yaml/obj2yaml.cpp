//===------ utils/obj2yaml.cpp - obj2yaml conversion tool -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "obj2yaml.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"

using namespace llvm;

namespace {
enum ObjectFileType {
  coff
};
}

cl::opt<ObjectFileType> InputFormat(
    cl::desc("Choose input format"),
    cl::values(clEnumVal(coff, "process COFF object files"), clEnumValEnd));

cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"),
                                   cl::init("-"));

int main(int argc, char *argv[]) {
  cl::ParseCommandLineOptions(argc, argv);
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  // Process the input file
  OwningPtr<MemoryBuffer> buf;

  // TODO: If this is an archive, then burst it and dump each entry
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputFilename, buf)) {
    errs() << "Error: '" << ec.message() << "' opening file '" << InputFilename
           << "'\n";
  } else {
    ec = coff2yaml(outs(), buf.take());
    if (ec)
      errs() << "Error: " << ec.message() << " dumping COFF file\n";
  }

  return 0;
}
