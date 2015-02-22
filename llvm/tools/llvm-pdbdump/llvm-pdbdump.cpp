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

#include "llvm-pdbdump.h"
#include "CompilandDumper.h"
#include "TypeDumper.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/config.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBRawSymbol.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDB.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"

#if defined(HAVE_DIA_SDK)
#include <Windows.h>
#endif

using namespace llvm;

namespace opts {

enum class PDB_DumpType { ByType, ByObjFile, Both };

cl::list<std::string> InputFilenames(cl::Positional,
                                     cl::desc("<input PDB files>"),
                                     cl::OneOrMore);

cl::opt<bool> DumpCompilands("compilands", cl::desc("Display compilands"));
cl::opt<bool> DumpSymbols("symbols",
                          cl::desc("Display symbols (implies --compilands"));
cl::opt<bool> DumpTypes("types", cl::desc("Display types"));
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
  std::string FileName(GlobalScope->getSymbolsFileName());

  outs() << "Summary for " << FileName;
  uint64_t FileSize = 0;
  if (!llvm::sys::fs::file_size(FileName, FileSize))
    outs() << newline(2) << "Size: " << FileSize << " bytes";
  else
    outs() << newline(2) << "Size: (Unable to obtain file size)";

  outs() << newline(2) << "Guid: " << GlobalScope->getGuid();
  outs() << newline(2) << "Age: " << GlobalScope->getAge();
  outs() << newline(2) << "Attributes: ";
  if (GlobalScope->hasCTypes())
    outs() << "HasCTypes ";
  if (GlobalScope->hasPrivateSymbols())
    outs() << "HasPrivateSymbols ";

  PDB_DumpFlags Flags = PDB_DF_None;
  if (opts::DumpTypes) {
    outs() << "\nDumping types";
    TypeDumper Dumper;
    Dumper.start(*GlobalScope, outs(), 2);
  }

  if (opts::DumpSymbols || opts::DumpCompilands) {
    outs() << "\nDumping compilands";
    auto Compilands = GlobalScope->findAllChildren<PDBSymbolCompiland>();
    CompilandDumper Dumper;
    while (auto Compiland = Compilands->getNext())
      Dumper.start(*Compiland, outs(), 2, opts::DumpSymbols);
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

#if defined(HAVE_DIA_SDK)
  CoInitializeEx(nullptr, COINIT_MULTITHREADED);
#endif

  std::for_each(opts::InputFilenames.begin(), opts::InputFilenames.end(),
                dumpInput);

#if defined(HAVE_DIA_SDK)
  CoUninitialize();
#endif

  return 0;
}
