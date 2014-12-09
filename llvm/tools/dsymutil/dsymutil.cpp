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
#include "DwarfLinker.h"
#include "MachODebugMapParser.h"

#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Options.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"

#include <string>

static llvm::cl::opt<std::string> InputFile(llvm::cl::Positional,
                                            llvm::cl::desc("<input file>"),
                                            llvm::cl::init("-"));

static llvm::cl::opt<std::string> OsoPrependPath("oso-prepend-path",
                                                 llvm::cl::desc("<path>"));

static llvm::cl::opt<bool> Verbose("v", llvm::cl::desc("Verbosity level"),
                                   llvm::cl::init(false));

static llvm::cl::opt<bool> ParseOnly("parse-only",
                                     llvm::cl::desc("Only parse the debug map, do "
                                                    "not actaully link the DWARF."),
                                     llvm::cl::init(false));

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram StackPrinter(argc, argv);
  llvm::llvm_shutdown_obj Shutdown;

  llvm::cl::ParseCommandLineOptions(argc, argv, "llvm dsymutil\n");

  llvm::MachODebugMapParser Parser(InputFile);
  Parser.setPreprendPath(OsoPrependPath);
  llvm::ErrorOr<std::unique_ptr<llvm::DebugMap>> DebugMap = Parser.parse();

  if (auto EC = DebugMap.getError()) {
    llvm::errs() << "error: cannot parse the debug map for \"" << InputFile <<
      "\": " << EC.message() << '\n';
    return 1;
  }

  if (Verbose)
    (*DebugMap)->print(llvm::outs());

  if (ParseOnly)
    return 0;

  llvm::DwarfLinker Linker(InputFile + ".dwarf");
  return !Linker.link(*DebugMap.get());
}
