//===- llvm-profdata.cpp - LLVM profile data tool -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// llvm-profdata merges .profdata files.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ProfileData/InstrProfWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static void exitWithError(const Twine &Message, StringRef Whence = "") {
  errs() << "error: ";
  if (!Whence.empty())
    errs() << Whence << ": ";
  errs() << Message << "\n";
  ::exit(1);
}

int merge_main(int argc, const char *argv[]) {
  cl::list<std::string> Inputs(cl::Positional, cl::Required, cl::OneOrMore,
                               cl::desc("<filenames...>"));

  cl::opt<std::string> OutputFilename("output", cl::value_desc("output"),
                                      cl::init("-"), cl::Required,
                                      cl::desc("Output file"));
  cl::alias OutputFilenameA("o", cl::desc("Alias for --output"),
                            cl::aliasopt(OutputFilename));

  cl::ParseCommandLineOptions(argc, argv, "LLVM profile data merger\n");

  if (OutputFilename.compare("-") == 0)
    exitWithError("Cannot write indexed profdata format to stdout.");

  std::error_code EC;
  raw_fd_ostream Output(OutputFilename.data(), EC, sys::fs::F_None);
  if (EC)
    exitWithError(EC.message(), OutputFilename);

  InstrProfWriter Writer;
  for (const auto &Filename : Inputs) {
    std::unique_ptr<InstrProfReader> Reader;
    if (std::error_code ec = InstrProfReader::create(Filename, Reader))
      exitWithError(ec.message(), Filename);

    for (const auto &I : *Reader)
      if (std::error_code EC =
              Writer.addFunctionCounts(I.Name, I.Hash, I.Counts))
        errs() << Filename << ": " << I.Name << ": " << EC.message() << "\n";
    if (Reader->hasError())
      exitWithError(Reader->getError().message(), Filename);
  }
  Writer.write(Output);

  return 0;
}

int show_main(int argc, const char *argv[]) {
  cl::opt<std::string> Filename(cl::Positional, cl::Required,
                                cl::desc("<profdata-file>"));

  cl::opt<bool> ShowCounts("counts", cl::init(false),
                           cl::desc("Show counter values for shown functions"));
  cl::opt<bool> ShowAllFunctions("all-functions", cl::init(false),
                                 cl::desc("Details for every function"));
  cl::opt<std::string> ShowFunction("function",
                                    cl::desc("Details for matching functions"));

  cl::opt<std::string> OutputFilename("output", cl::value_desc("output"),
                                      cl::init("-"),
                                      cl::desc("Output file"));
  cl::alias OutputFilenameA("o", cl::desc("Alias for --output"),
                            cl::aliasopt(OutputFilename));

  cl::ParseCommandLineOptions(argc, argv, "LLVM profile data summary\n");

  std::unique_ptr<InstrProfReader> Reader;
  if (std::error_code EC = InstrProfReader::create(Filename, Reader))
    exitWithError(EC.message(), Filename);

  if (OutputFilename.empty())
    OutputFilename = "-";

  std::error_code EC;
  raw_fd_ostream OS(OutputFilename.data(), EC, sys::fs::F_Text);
  if (EC)
    exitWithError(EC.message(), OutputFilename);

  if (ShowAllFunctions && !ShowFunction.empty())
    errs() << "warning: -function argument ignored: showing all functions\n";

  uint64_t MaxFunctionCount = 0, MaxBlockCount = 0;
  size_t ShownFunctions = 0, TotalFunctions = 0;
  for (const auto &Func : *Reader) {
    bool Show = ShowAllFunctions ||
                (!ShowFunction.empty() &&
                 Func.Name.find(ShowFunction) != Func.Name.npos);

    ++TotalFunctions;
    assert(Func.Counts.size() > 0 && "function missing entry counter");
    if (Func.Counts[0] > MaxFunctionCount)
      MaxFunctionCount = Func.Counts[0];

    if (Show) {
      if (!ShownFunctions)
        OS << "Counters:\n";
      ++ShownFunctions;

      OS << "  " << Func.Name << ":\n"
         << "    Hash: " << format("0x%016" PRIx64, Func.Hash) << "\n"
         << "    Counters: " << Func.Counts.size() << "\n"
         << "    Function count: " << Func.Counts[0] << "\n";
    }

    if (Show && ShowCounts)
      OS << "    Block counts: [";
    for (size_t I = 1, E = Func.Counts.size(); I < E; ++I) {
      if (Func.Counts[I] > MaxBlockCount)
        MaxBlockCount = Func.Counts[I];
      if (Show && ShowCounts)
        OS << (I == 1 ? "" : ", ") << Func.Counts[I];
    }
    if (Show && ShowCounts)
      OS << "]\n";
  }
  if (Reader->hasError())
    exitWithError(Reader->getError().message(), Filename);

  if (ShowAllFunctions || !ShowFunction.empty())
    OS << "Functions shown: " << ShownFunctions << "\n";
  OS << "Total functions: " << TotalFunctions << "\n";
  OS << "Maximum function count: " << MaxFunctionCount << "\n";
  OS << "Maximum internal block count: " << MaxBlockCount << "\n";
  return 0;
}

int main(int argc, const char *argv[]) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  StringRef ProgName(sys::path::filename(argv[0]));
  if (argc > 1) {
    int (*func)(int, const char *[]) = nullptr;

    if (strcmp(argv[1], "merge") == 0)
      func = merge_main;
    else if (strcmp(argv[1], "show") == 0)
      func = show_main;

    if (func) {
      std::string Invocation(ProgName.str() + " " + argv[1]);
      argv[1] = Invocation.c_str();
      return func(argc - 1, argv + 1);
    }

    if (strcmp(argv[1], "-h") == 0 ||
        strcmp(argv[1], "-help") == 0 ||
        strcmp(argv[1], "--help") == 0) {

      errs() << "OVERVIEW: LLVM profile data tools\n\n"
             << "USAGE: " << ProgName << " <command> [args...]\n"
             << "USAGE: " << ProgName << " <command> -help\n\n"
             << "Available commands: merge, show\n";
      return 0;
    }
  }

  if (argc < 2)
    errs() << ProgName << ": No command specified!\n";
  else
    errs() << ProgName << ": Unknown command!\n";

  errs() << "USAGE: " << ProgName << " <merge|show> [args...]\n";
  return 1;
}
