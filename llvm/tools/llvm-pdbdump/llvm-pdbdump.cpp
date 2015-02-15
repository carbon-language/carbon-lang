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
#include "llvm/Config/config.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBRawSymbol.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDB.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

#if defined(HAVE_DIA_SDK)
#include <Windows.h>
#endif

using namespace llvm;

namespace opts {

enum class PDB_DumpType { ByType, ByObjFile, Both };

cl::list<std::string> InputFilenames(cl::Positional,
                                     cl::desc("<input PDB files>"),
                                     cl::OneOrMore);

cl::opt<bool> DumpHidden(
    "hidden",
    cl::desc("Attempt to find hidden symbols.  This can find additional\n"
             "symbols that cannot be found otherwise.  For example, vtables\n"
             "can only be found with an exhaustive search such as this.  Be\n"
             "warned that the performance can be prohibitive on large PDB "
             "files."));

cl::opt<bool> DumpAll(
    "all",
    cl::desc("Specifies all other options except -hidden and -group-by"));
cl::opt<bool> DumpObjFiles("compilands", cl::desc("Display object files"));
cl::opt<bool> DumpFuncs("functions", cl::desc("Display function information"));
cl::opt<bool> DumpData(
    "data",
    cl::desc("Display global, class, and constant variable information."));
cl::opt<bool> DumpLabels("labels", cl::desc("Display labels"));
cl::opt<bool> DumpPublic("public", cl::desc("Display public symbols"));
cl::opt<bool> DumpClasses("classes", cl::desc("Display class type information"));
cl::opt<bool> DumpEnums("enums", cl::desc("Display enum information"));
cl::opt<bool> DumpFuncsigs("funcsigs",
                           cl::desc("Display unique function signatures"));
cl::opt<bool> DumpTypedefs("typedefs", cl::desc("Display typedefs"));
cl::opt<bool> DumpThunks("thunks", cl::desc("Display thunks"));
cl::opt<bool> DumpVtables(
    "vtables",
    cl::desc("Display virtual function tables (only with --exhaustive)"));

static cl::opt<PDB_DumpType> DumpMode(
    "group-by", cl::init(PDB_DumpType::ByType), cl::desc("Dump mode:"),
    cl::values(
        clEnumValN(PDB_DumpType::ByType, "type",
                   "(Default) Display symbols grouped by type"),
        clEnumValN(PDB_DumpType::ByObjFile, "compiland",
                   "Display symbols grouped under their containing object "
                   "file."),
        clEnumValN(PDB_DumpType::Both, "both",
                   "Display symbols grouped by type, and then by object file."),
        clEnumValEnd));
}

#define SET_DUMP_FLAG_FROM_OPT(Var, Flag, Opt) \
  if (opts::Opt) \
    Var |= Flag;

PDB_DumpFlags CalculateDumpFlags() {
  PDB_DumpFlags Flags = PDB_DF_None;

  SET_DUMP_FLAG_FROM_OPT(Flags, PDB_DF_Hidden, DumpHidden)

  if (opts::DumpAll)
    return Flags | PDB_DF_All;

  SET_DUMP_FLAG_FROM_OPT(Flags, PDB_DF_ObjFiles, DumpObjFiles)
  SET_DUMP_FLAG_FROM_OPT(Flags, PDB_DF_Functions, DumpFuncs)
  SET_DUMP_FLAG_FROM_OPT(Flags, PDB_DF_Data, DumpData)
  SET_DUMP_FLAG_FROM_OPT(Flags, PDB_DF_Labels, DumpLabels)
  SET_DUMP_FLAG_FROM_OPT(Flags, PDB_DF_PublicSyms, DumpPublic)
  SET_DUMP_FLAG_FROM_OPT(Flags, PDB_DF_Classes, DumpClasses)
  SET_DUMP_FLAG_FROM_OPT(Flags, PDB_DF_Enums, DumpEnums)
  SET_DUMP_FLAG_FROM_OPT(Flags, PDB_DF_Funcsigs, DumpFuncsigs)
  SET_DUMP_FLAG_FROM_OPT(Flags, PDB_DF_Typedefs, DumpTypedefs)
  SET_DUMP_FLAG_FROM_OPT(Flags, PDB_DF_Thunks, DumpThunks)
  SET_DUMP_FLAG_FROM_OPT(Flags, PDB_DF_VTables, DumpVtables)
  return Flags;
}

static void dumpInput(StringRef Path) {
  std::unique_ptr<IPDBSession> Session(
      llvm::createPDBReader(PDB_ReaderType::DIA, Path));
  if (!Session) {
    outs() << "Unable to create PDB reader.  Check that a valid implementation";
    outs() << " is available for your platform.";
    return;
  }
  PDB_DumpFlags Flags = CalculateDumpFlags();

  if (opts::DumpMode != opts::PDB_DumpType::ByObjFile)
    Flags |= PDB_DF_Children;
  auto GlobalScope(Session->getGlobalScope());
  GlobalScope->dump(outs(), 0, PDB_DumpLevel::Normal, Flags);
  outs() << "\n";

  if (opts::DumpMode != opts::PDB_DumpType::ByType) {
    auto Compilands = GlobalScope->findAllChildren<PDBSymbolCompiland>();
    while (auto Compiland = Compilands->getNext()) {
      Compiland->dump(outs(), 0, PDB_DumpLevel::Detailed,
                      Flags | PDB_DF_Children);
      outs() << "\n";
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
