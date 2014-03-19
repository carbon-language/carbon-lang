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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static void exitWithError(const std::string &Message,
                          const std::string &Filename, int64_t Line = -1) {
  errs() << "error: " << Filename;
  if (Line >= 0)
    errs() << ":" << Line;
  errs() << ": " << Message << "\n";
  ::exit(1);
}

//===----------------------------------------------------------------------===//
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

  std::unique_ptr<MemoryBuffer> File1;
  std::unique_ptr<MemoryBuffer> File2;
  if (error_code ec = MemoryBuffer::getFile(Filename1, File1))
    exitWithError(ec.message(), Filename1);
  if (error_code ec = MemoryBuffer::getFile(Filename2, File2))
    exitWithError(ec.message(), Filename2);

  if (OutputFilename.empty())
    OutputFilename = "-";

  std::string ErrorInfo;
  raw_fd_ostream Output(OutputFilename.data(), ErrorInfo, sys::fs::F_Text);
  if (!ErrorInfo.empty())
    exitWithError(ErrorInfo, OutputFilename);

  enum {ReadName, ReadHash, ReadCount, ReadCounters} State = ReadName;
  uint64_t N1, N2, NumCounters;
  line_iterator I1(*File1, '#'), I2(*File2, '#');
  for (; !I1.is_at_end() && !I2.is_at_end(); ++I1, ++I2) {
    if (I1->empty()) {
      if (!I2->empty())
        exitWithError("data mismatch", Filename2, I2.line_number());
      Output << "\n";
      continue;
    }
    switch (State) {
    case ReadName:
      if (*I1 != *I2)
        exitWithError("function name mismatch", Filename2, I2.line_number());
      Output << *I1 << "\n";
      State = ReadHash;
      break;
    case ReadHash:
      if (I1->getAsInteger(10, N1))
        exitWithError("bad function hash", Filename1, I1.line_number());
      if (I2->getAsInteger(10, N2))
        exitWithError("bad function hash", Filename2, I2.line_number());
      if (N1 != N2)
        exitWithError("function hash mismatch", Filename2, I2.line_number());
      Output << N1 << "\n";
      State = ReadCount;
      break;
    case ReadCount:
      if (I1->getAsInteger(10, N1))
        exitWithError("bad function count", Filename1, I1.line_number());
      if (I2->getAsInteger(10, N2))
        exitWithError("bad function count", Filename2, I2.line_number());
      if (N1 != N2)
        exitWithError("function count mismatch", Filename2, I2.line_number());
      Output << N1 << "\n";
      NumCounters = N1;
      State = ReadCounters;
      break;
    case ReadCounters:
      if (I1->getAsInteger(10, N1))
        exitWithError("invalid counter", Filename1, I1.line_number());
      if (I2->getAsInteger(10, N2))
        exitWithError("invalid counter", Filename2, I2.line_number());
      uint64_t Sum = N1 + N2;
      if (Sum < N1)
        exitWithError("counter overflow", Filename2, I2.line_number());
      Output << N1 + N2 << "\n";
      if (--NumCounters == 0)
        State = ReadName;
      break;
    }
  }
  if (!I1.is_at_end())
    exitWithError("truncated file", Filename1, I1.line_number());
  if (!I2.is_at_end())
    exitWithError("truncated file", Filename2, I2.line_number());
  if (State != ReadName)
    exitWithError("truncated file", Filename1, I1.line_number());

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

  errs() << "USAGE: " << ProgName << " <merge|show|generate> [args...]\n";
  return 1;
}
