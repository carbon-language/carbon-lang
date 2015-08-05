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
#include "llvm/Support/TargetSelect.h"
#include <string>

using namespace llvm::dsymutil;

namespace {
using namespace llvm::cl;

OptionCategory DsymCategory("Specific Options");
static opt<bool> Help("h", desc("Alias for -help"), Hidden);
static opt<bool> Version("v", desc("Alias for -version"), Hidden);

static list<std::string> InputFiles(Positional, OneOrMore,
                                    desc("<input files>"), cat(DsymCategory));

static opt<std::string>
    OutputFileOpt("o",
                  desc("Specify the output file. default: <input file>.dwarf"),
                  value_desc("filename"), cat(DsymCategory));

static opt<std::string> OsoPrependPath(
    "oso-prepend-path",
    desc("Specify a directory to prepend to the paths of object files."),
    value_desc("path"), cat(DsymCategory));

static opt<bool> Verbose("verbose", desc("Verbosity level"), init(false),
                         cat(DsymCategory));

static opt<bool>
    NoOutput("no-output",
             desc("Do the link in memory, but do not emit the result file."),
             init(false), cat(DsymCategory));

static opt<bool>
    NoODR("no-odr",
          desc("Do not use ODR (One Definition Rule) for type uniquing."),
          init(false), cat(DsymCategory));

static opt<bool> DumpDebugMap(
    "dump-debug-map",
    desc("Parse and dump the debug map to standard output. Not DWARF link "
         "will take place."),
    init(false), cat(DsymCategory));

static opt<bool> InputIsYAMLDebugMap(
    "y", desc("Treat the input file is a YAML debug map rather than a binary."),
    init(false), cat(DsymCategory));
}

static std::string getOutputFileName(llvm::StringRef InputFile) {
  if (OutputFileOpt.empty()) {
    if (InputFile == "-")
      return "a.out.dwarf";
    return (InputFile + ".dwarf").str();
  }
  return OutputFileOpt;
}

void llvm::dsymutil::exitDsymutil(int ExitStatus) {
  exit(ExitStatus);
}

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram StackPrinter(argc, argv);
  llvm::llvm_shutdown_obj Shutdown;
  LinkOptions Options;

  HideUnrelatedOptions(DsymCategory);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "manipulate archived DWARF debug symbol files.\n\n"
      "dsymutil links the DWARF debug information found in the object files\n"
      "for the executable <input file> by using debug symbols information\n"
      "contained in its symbol table.\n");

  if (Help)
    PrintHelpMessage();

  if (Version) {
    llvm::cl::PrintVersionMessage();
    return 0;
  }

  Options.Verbose = Verbose;
  Options.NoOutput = NoOutput;
  Options.NoODR = NoODR;

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();

  if (InputFiles.size() > 1 && !OutputFileOpt.empty()) {
    llvm::errs() << "error: cannot use -o with multiple inputs\n";
    return 1;
  }

  for (auto &InputFile : InputFiles) {
    auto DebugMapPtrOrErr =
        parseDebugMap(InputFile, OsoPrependPath, Verbose, InputIsYAMLDebugMap);

    if (auto EC = DebugMapPtrOrErr.getError()) {
      llvm::errs() << "error: cannot parse the debug map for \"" << InputFile
                   << "\": " << EC.message() << '\n';
      exitDsymutil(1);
    }

    if (Verbose || DumpDebugMap)
      (*DebugMapPtrOrErr)->print(llvm::outs());

    if (DumpDebugMap)
      continue;

    std::string OutputFile = getOutputFileName(InputFile);
    if (!linkDwarf(OutputFile, **DebugMapPtrOrErr, Options))
      exitDsymuti(1);
  }

  exitDsymutil(0);
}
