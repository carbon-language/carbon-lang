//===- llvm-pdbdump.cpp - Dump debug info from a PDB file -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Dumps debug information present in PDB files.  This utility makes use of
// the Microsoft Windows SDK, so will not compile or run on non-Windows
// platforms.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/DebugInfo/PDB/PDB.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/IPDBRawSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"

#include <Windows.h>

using namespace llvm;

namespace opts {
cl::list<std::string> InputFilenames(cl::Positional,
                                     cl::desc("<input PDB files>"),
                                     cl::OneOrMore);

cl::opt<bool> Compilands("compilands",
                         cl::desc("Display a list of compilands (e.g. object "
                                  "files) and symbols for each one."));
cl::alias CompilandsShort("c", cl::desc("Alias for --compilands"),
                          cl::aliasopt(Compilands));
}

static void dumpInput(StringRef Path) {
  std::unique_ptr<IPDBSession> Session(
      llvm::createPDBReader(PDB_ReaderType::DIA, Path));
  if (!Session) {
    outs() << "Unable to create PDB reader.  Check that a valid implementation";
    outs() << " is available for your platform.";
    return;
  }

  auto GlobalScope(Session->getGlobalScope());
  GlobalScope->dump(outs(), 0, PDB_DumpLevel::Normal);
  outs().flush();

  if (opts::Compilands) {
    auto Compilands = GlobalScope->findChildren(PDB_SymType::Compiland);
    if (Compilands) {
      while (auto Compiland = Compilands->getNext()) {
        Compiland->dump(outs(), 0, PDB_DumpLevel::Normal);
      }
    }
  }
  outs().flush();
}

int main(int argc_, const char *argv_[]) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc_, argv_);

  SmallVector<const char *, 256> argv;
  llvm::SpecificBumpPtrAllocator<char> ArgAllocator;
  std::error_code EC = llvm::sys::Process::GetArgumentVector(
      argv, llvm::makeArrayRef(argv_, argc_), ArgAllocator);
  if (EC) {
    llvm::errs() << "error: couldn't get arguments: " << EC.message() << '\n';
    return 1;
  }

  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argv.size(), argv.data(), "LLVM PDB Dumper\n");

  CoInitializeEx(nullptr, COINIT_MULTITHREADED);

  std::for_each(opts::InputFilenames.begin(), opts::InputFilenames.end(),
                dumpInput);

  CoUninitialize();
  return 0;
}
