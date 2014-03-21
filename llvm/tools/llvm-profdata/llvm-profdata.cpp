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
#include "llvm/Support/CommandLine.h"
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
  cl::opt<std::string> Filename1(cl::Positional, cl::Required,
                                 cl::desc("file1"));
  cl::opt<std::string> Filename2(cl::Positional, cl::Required,
                                 cl::desc("file2"));

  cl::opt<std::string> OutputFilename("output", cl::value_desc("output"),
                                      cl::init("-"),
                                      cl::desc("Output file"));
  cl::alias OutputFilenameA("o", cl::desc("Alias for --output"),
                                 cl::aliasopt(OutputFilename));

  cl::ParseCommandLineOptions(argc, argv, "LLVM profile data merger\n");

  std::unique_ptr<InstrProfReader> Reader1, Reader2;
  if (error_code ec = InstrProfReader::create(Filename1, Reader1))
    exitWithError(ec.message(), Filename1);
  if (error_code ec = InstrProfReader::create(Filename2, Reader2))
    exitWithError(ec.message(), Filename2);

  if (OutputFilename.empty())
    OutputFilename = "-";

  std::string ErrorInfo;
  raw_fd_ostream Output(OutputFilename.data(), ErrorInfo, sys::fs::F_Text);
  if (!ErrorInfo.empty())
    exitWithError(ErrorInfo, OutputFilename);

  for (InstrProfIterator I1 = Reader1->begin(), E1 = Reader1->end(),
                         I2 = Reader2->begin(), E2 = Reader2->end();
       I1 != E1 && I2 != E2; ++I1, ++I2) {
    if (I1->Name != I2->Name)
      exitWithError("Function name mismatch, " + I1->Name + " != " + I2->Name);
    if (I1->Hash != I2->Hash)
      exitWithError("Function hash mismatch for " + I1->Name);
    if (I1->Counts.size() != I2->Counts.size())
      exitWithError("Function count mismatch for " + I1->Name);

    Output << I1->Name << "\n" << I1->Hash << "\n" << I1->Counts.size() << "\n";

    for (size_t II = 0, EE = I1->Counts.size(); II < EE; ++II) {
      uint64_t Sum = I1->Counts[II] + I2->Counts[II];
      if (Sum < I1->Counts[II])
        exitWithError("Counter overflow for " + I1->Name);
      Output << Sum << "\n";
    }
    Output << "\n";
  }
  if (Reader1->hasError())
    exitWithError(Reader1->getError().message(), Filename1);
  if (Reader2->hasError())
    exitWithError(Reader2->getError().message(), Filename2);
  if (!Reader1->isEOF() || !Reader2->isEOF())
    exitWithError("Number of instrumented functions differ.");

  return 0;
}

int main(int argc, const char *argv[]) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  StringRef ProgName(sys::path::filename(argv[0]));
  if (argc > 1) {
    int (*func)(int, const char *[]) = 0;

    if (strcmp(argv[1], "merge") == 0)
      func = merge_main;

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
             << "Available commands: merge\n";
      return 0;
    }
  }

  if (argc < 2)
    errs() << ProgName << ": No command specified!\n";
  else
    errs() << ProgName << ": Unknown command!\n";

  errs() << "USAGE: " << ProgName << " <merge> [args...]\n";
  return 1;
}
