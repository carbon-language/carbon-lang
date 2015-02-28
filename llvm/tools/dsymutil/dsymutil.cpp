//===-- dsymutil.cpp - Debug info dumping utility for llvm ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that aims to be a dropin replacement for
// Darwin's dsymutil.
//
//===----------------------------------------------------------------------===//

#include "DebugMap.h"
#include "dsymutil.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Options.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm::dsymutil;

namespace {
using namespace llvm::cl;

static opt<std::string> InputFile(Positional, desc("<input file>"),
                                  init("a.out"));

static opt<std::string> OutputFileOpt("o", desc("Specify the output file."
                                                " default: <input file>.dwarf"),
                                      value_desc("filename"));

static opt<std::string> OsoPrependPath("oso-prepend-path",
                                       desc("Specify a directory to prepend "
                                            "to the paths of object files."),
                                       value_desc("path"));

static opt<bool> Verbose("v", desc("Verbosity level"), init(false));

static opt<bool>
    ParseOnly("parse-only",
              desc("Only parse the debug map, do not actaully link "
                   "the DWARF."),
              init(false));
}

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram StackPrinter(argc, argv);
  llvm::llvm_shutdown_obj Shutdown;

  llvm::cl::ParseCommandLineOptions(argc, argv, "llvm dsymutil\n");
  auto DebugMapPtrOrErr = parseDebugMap(InputFile, OsoPrependPath, Verbose);

  if (auto EC = DebugMapPtrOrErr.getError()) {
    llvm::errs() << "error: cannot parse the debug map for \"" << InputFile
                 << "\": " << EC.message() << '\n';
    return 1;
  }

  if (Verbose)
    (*DebugMapPtrOrErr)->print(llvm::outs());

  if (ParseOnly)
    return 0;

  std::string OutputFile;
  if (OutputFileOpt.empty()) {
    if (InputFile == "-")
      OutputFile = "a.out.dwarf";
    else
      OutputFile = InputFile + ".dwarf";
  } else {
    OutputFile = OutputFileOpt;
  }

  return !linkDwarf(OutputFile, **DebugMapPtrOrErr, Verbose);
}
