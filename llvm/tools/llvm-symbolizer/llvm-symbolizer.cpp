//===-- llvm-symbolizer.cpp - Simple addr2line-like symbolizer ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility works much like "addr2line". It is able of transforming
// tuples (module name, module offset) to code locations (function name,
// file, line number, column number). It is targeted for compiler-rt tools
// (especially AddressSanitizer and ThreadSanitizer) that can use it
// to symbolize stack traces in their error reports.
//
//===----------------------------------------------------------------------===//

#include "LLVMSymbolize.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/COM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>
#include <cstring>
#include <string>

using namespace llvm;
using namespace symbolize;

static cl::opt<bool>
ClUseSymbolTable("use-symbol-table", cl::init(true),
                 cl::desc("Prefer names in symbol table to names "
                          "in debug info"));

static cl::opt<FunctionNameKind> ClPrintFunctions(
    "functions", cl::init(FunctionNameKind::LinkageName),
    cl::desc("Print function name for a given address:"),
    cl::values(clEnumValN(FunctionNameKind::None, "none", "omit function name"),
               clEnumValN(FunctionNameKind::ShortName, "short",
                          "print short function name"),
               clEnumValN(FunctionNameKind::LinkageName, "linkage",
                          "print function linkage name"),
               clEnumValEnd));

static cl::opt<bool>
    ClUseRelativeAddress("relative-address", cl::init(false),
                         cl::desc("Interpret addresses as relative addresses"),
                         cl::ReallyHidden);

static cl::opt<bool>
    ClPrintInlining("inlining", cl::init(true),
                    cl::desc("Print all inlined frames for a given address"));

static cl::opt<bool>
ClDemangle("demangle", cl::init(true), cl::desc("Demangle function names"));

static cl::opt<std::string> ClDefaultArch("default-arch", cl::init(""),
                                          cl::desc("Default architecture "
                                                   "(for multi-arch objects)"));

static cl::opt<std::string>
ClBinaryName("obj", cl::init(""),
             cl::desc("Path to object file to be symbolized (if not provided, "
                      "object file should be specified for each input line)"));

static cl::list<std::string>
ClDsymHint("dsym-hint", cl::ZeroOrMore,
           cl::desc("Path to .dSYM bundles to search for debug info for the "
                    "object files"));

static bool parseCommand(bool &IsData, std::string &ModuleName,
                         uint64_t &ModuleOffset) {
  const char *kDataCmd = "DATA ";
  const char *kCodeCmd = "CODE ";
  const int kMaxInputStringLength = 1024;
  const char kDelimiters[] = " \n";
  char InputString[kMaxInputStringLength];
  if (!fgets(InputString, sizeof(InputString), stdin))
    return false;
  IsData = false;
  ModuleName = "";
  char *pos = InputString;
  if (strncmp(pos, kDataCmd, strlen(kDataCmd)) == 0) {
    IsData = true;
    pos += strlen(kDataCmd);
  } else if (strncmp(pos, kCodeCmd, strlen(kCodeCmd)) == 0) {
    IsData = false;
    pos += strlen(kCodeCmd);
  } else {
    // If no cmd, assume it's CODE.
    IsData = false;
  }
  // Skip delimiters and parse input filename (if needed).
  if (ClBinaryName == "") {
    pos += strspn(pos, kDelimiters);
    if (*pos == '"' || *pos == '\'') {
      char quote = *pos;
      pos++;
      char *end = strchr(pos, quote);
      if (!end)
        return false;
      ModuleName = std::string(pos, end - pos);
      pos = end + 1;
    } else {
      int name_length = strcspn(pos, kDelimiters);
      ModuleName = std::string(pos, name_length);
      pos += name_length;
    }
  } else {
    ModuleName = ClBinaryName;
  }
  // Skip delimiters and parse module offset.
  pos += strspn(pos, kDelimiters);
  int offset_length = strcspn(pos, kDelimiters);
  if (StringRef(pos, offset_length).getAsInteger(0, ModuleOffset))
    return false;
  return true;
}

int main(int argc, char **argv) {
  // Print stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  llvm::sys::InitializeCOMRAII COM(llvm::sys::COMThreadingMode::MultiThreaded);

  cl::ParseCommandLineOptions(argc, argv, "llvm-symbolizer\n");
  LLVMSymbolizer::Options Opts(ClPrintFunctions, ClUseSymbolTable,
                               ClPrintInlining, ClDemangle,
                               ClUseRelativeAddress, ClDefaultArch);
  for (const auto &hint : ClDsymHint) {
    if (sys::path::extension(hint) == ".dSYM") {
      Opts.DsymHints.push_back(hint);
    } else {
      errs() << "Warning: invalid dSYM hint: \"" << hint <<
                "\" (must have the '.dSYM' extension).\n";
    }
  }
  LLVMSymbolizer Symbolizer(Opts);

  bool IsData = false;
  std::string ModuleName;
  uint64_t ModuleOffset;
  while (parseCommand(IsData, ModuleName, ModuleOffset)) {
    std::string Result =
        IsData ? Symbolizer.symbolizeData(ModuleName, ModuleOffset)
               : Symbolizer.symbolizeCode(ModuleName, ModuleOffset);
    outs() << Result << "\n";
    outs().flush();
  }

  return 0;
}
