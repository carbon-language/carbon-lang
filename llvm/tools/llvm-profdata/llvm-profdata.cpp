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
#include "llvm/Profile/ProfileDataReader.h"
#include "llvm/Profile/ProfileDataWriter.h"
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

  std::unique_ptr<ProfileDataReader> Reader1, Reader2;
  if (error_code EC = ProfileDataReader::create(Filename1, Reader1))
    exitWithError(EC.message(), Filename1);
  if (error_code EC = ProfileDataReader::create(Filename2, Reader2))
    exitWithError(EC.message(), Filename2);

  if (OutputFilename.empty())
    OutputFilename = "-";

  std::string ErrorInfo;
  raw_fd_ostream Output(OutputFilename.data(), ErrorInfo, sys::fs::F_Text);
  if (!ErrorInfo.empty())
    exitWithError(ErrorInfo, OutputFilename);

  if (Output.is_displayed())
    exitWithError("Refusing to write a binary file to stdout", OutputFilename);

  StringRef Name1, Name2;
  std::vector<uint64_t> Counts1, Counts2, NewCounts;
  uint64_t Hash1, Hash2;
  ProfileDataWriter Writer;
  ProfileDataReader::name_iterator I1 = Reader1->begin(),
                                   E1 = Reader1->end(),
                                   I2 = Reader2->begin(),
                                   E2 = Reader2->end();
  for (; I1 != E1 && I2 != E2; ++I1, ++I2) {
    Name1 = *I1;
    Name2 = *I2;
    if (Name1 != Name2)
      exitWithError("Function name mismatch", Filename2); // ???

    if (error_code EC = Reader1->getFunctionCounts(Name1, Hash1, Counts1))
      exitWithError(EC.message(), Filename1);
    if (error_code EC = Reader2->getFunctionCounts(Name2, Hash2, Counts2))
      exitWithError(EC.message(), Filename2);

    if (Counts1.size() != Counts2.size())
      exitWithError("Function count mismatch", Filename2); // ???
    if (Hash1 != Hash2)
      exitWithError("Function hash mismatch", Filename2); // ???

    for (size_t II = 0, EE = Counts1.size(); II < EE; ++II) {
      uint64_t Sum = Counts1[II] + Counts2[II];
      if (Sum < Counts1[II])
        exitWithError("Counter overflow", Filename2); // ???
      NewCounts.push_back(Sum);
    }

    Writer.addFunctionCounts(Name1, Hash1, NewCounts.size(), NewCounts.data());

    Counts1.clear();
    Counts2.clear();
    NewCounts.clear();
  }
  if (I1 != E1 || I2 != E2)
    exitWithError("Truncated file", Filename2);

  Writer.write(Output);

  return 0;
}

struct HashPrinter {
  uint64_t Hash;
  HashPrinter(uint64_t Hash) : Hash(Hash) {}
  void print(raw_ostream &OS) const {
    char Buf[18], *Cur = Buf;
    *Cur++ = '0'; *Cur++ = 'x';
    for (unsigned I = 16; I;) {
      char Digit = 0xF & (Hash >> (--I * 4));
      *Cur++ = (Digit < 10 ? '0' + Digit : 'A' + Digit - 10);
    }
    OS.write(Buf, 18);
  }
};
static raw_ostream &operator<<(raw_ostream &OS, const HashPrinter &Hash) {
  Hash.print(OS);
  return OS;
}

struct FreqPrinter {
  double Freq;
  FreqPrinter(double Freq) : Freq(Freq) {}
  void print(raw_ostream &OS) const {
    OS << (unsigned)(Freq * 100) << "." << ((unsigned)(Freq * 1000) % 10)
       << ((unsigned)(Freq * 10000) % 10) << "%";
  }
};
static raw_ostream &operator<<(raw_ostream &OS, const FreqPrinter &Freq) {
  Freq.print(OS);
  return OS;
}

int show_main(int argc, const char *argv[]) {
  cl::opt<std::string> Filename(cl::Positional, cl::Required,
                                cl::desc("<profdata-file>"));

  cl::opt<bool> ShowCounts("counts", cl::init(false));
  cl::opt<bool> ShowAllFunctions("all-functions", cl::init(false));
  cl::opt<std::string> ShowFunction("function");

  cl::opt<std::string> OutputFilename("output", cl::value_desc("output"),
                                      cl::init("-"),
                                      cl::desc("Output file"));
  cl::alias OutputFilenameA("o", cl::desc("Alias for --output"),
                            cl::aliasopt(OutputFilename));

  cl::ParseCommandLineOptions(argc, argv, "LLVM profile data summary\n");

  std::unique_ptr<ProfileDataReader> Reader;
  if (error_code EC = ProfileDataReader::create(Filename, Reader))
    exitWithError(EC.message(), Filename);

  if (OutputFilename.empty())
    OutputFilename = "-";

  std::string ErrorInfo;
  raw_fd_ostream OS(OutputFilename.data(), ErrorInfo, sys::fs::F_Text);
  if (!ErrorInfo.empty())
    exitWithError(ErrorInfo, OutputFilename);

  if (ShowAllFunctions && !ShowFunction.empty())
    errs() << "warning: -function argument ignored: showing all functions\n";

  uint64_t MaxFunctionCount = Reader->getMaximumFunctionCount();

  uint64_t MaxBlockCount = 0;
  uint64_t Hash;
  size_t ShownFunctions = false;
  std::vector<uint64_t> Counts;
  for (const auto &Name : *Reader) {
    bool Show = ShowAllFunctions || Name.find(ShowFunction) != Name.npos;
    if (error_code EC = Reader->getFunctionCounts(Name, Hash, Counts))
      exitWithError(EC.message(), Filename);

    if (Show) {
      double CallFreq = Counts[0] / (double)MaxFunctionCount;

      if (!ShownFunctions)
        OS << "Counters:\n";
      ++ShownFunctions;

      OS << "  " << Name << ":\n"
         << "    Hash: " << HashPrinter(Hash) << "\n"
         << "    Relative call frequency: " << FreqPrinter(CallFreq) << "\n"
         << "    Counters: " << Counts.size() << "\n"
         << "    Function count: " << Counts[0] << "\n";
    }

    if (Show && ShowCounts)
      OS << "    Block counts: [";
    for (size_t I = 1, E = Counts.size(); I < E; ++I) {
      if (Counts[I] > MaxBlockCount)
        MaxBlockCount = Counts[I];
      if (Show && ShowCounts)
        OS << (I == 1 ? "" : ", ") << Counts[I];
    }
    if (Show && ShowCounts)
      OS << "]\n";

    Counts.clear();
  }

  if (ShowAllFunctions || !ShowFunction.empty())
    OS << "Functions shown: " << ShownFunctions << "\n";
  OS << "Total functions: " << Reader->numProfiledFunctions() << "\n";
  OS << "Maximum function count: " << MaxFunctionCount << "\n";
  OS << "Maximum internal block count: " << MaxBlockCount << "\n";
  return 0;
}

int generate_main(int argc, const char *argv[]) {
  cl::opt<std::string> InputName(cl::Positional, cl::Required,
                                 cl::desc("<input-file>"));

  cl::opt<std::string> OutputFilename("output", cl::value_desc("output"),
                                      cl::init("-"),
                                      cl::desc("Output file"));
  cl::alias OutputFilenameA("o", cl::desc("Alias for --output"),
                            cl::aliasopt(OutputFilename));

  cl::ParseCommandLineOptions(argc, argv, "LLVM profile data generator\n");

  if (OutputFilename.empty())
    OutputFilename = "-";

  std::string ErrorInfo;
  raw_fd_ostream Output(OutputFilename.data(), ErrorInfo, sys::fs::F_Text);
  if (!ErrorInfo.empty())
    exitWithError(ErrorInfo, OutputFilename);

  if (Output.is_displayed())
    exitWithError("Refusing to write a binary file to stdout", OutputFilename);

  std::unique_ptr<MemoryBuffer> Buffer;
  if (error_code EC = MemoryBuffer::getFile(InputName, Buffer))
    exitWithError(EC.message(), InputName);

  ProfileDataWriter Writer;
  StringRef Name;
  uint64_t Hash, NumCounters;
  std::vector<uint64_t> Counters;
  for (line_iterator I(*Buffer, '#'); !I.is_at_end(); ++I) {
    if (I->empty())
      continue;
    Name = *I;
    if ((++I).is_at_end())
      exitWithError("Truncated file", InputName, I.line_number());
    if (I->getAsInteger(10, Hash))
      exitWithError("Failed to read hash", InputName, I.line_number());
    if ((++I).is_at_end())
      exitWithError("Truncated file", InputName, I.line_number());
    if (I->getAsInteger(10, NumCounters))
      exitWithError("Failed to read num counters", InputName, I.line_number());
    for (uint64_t CurCounter = 0; CurCounter < NumCounters; ++CurCounter) {
      uint64_t Counter;
      if ((++I).is_at_end())
        exitWithError("Truncated file", InputName, I.line_number());
      if (I->getAsInteger(10, Counter))
        exitWithError("Failed to read counter", InputName, I.line_number());
      Counters.push_back(Counter);
    }
    Writer.addFunctionCounts(Name, Hash, NumCounters, Counters.data());
    Counters.clear();
  }

  Writer.write(Output);

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
    else if (strcmp(argv[1], "show") == 0)
      func = show_main;
    else if (strcmp(argv[1], "generate") == 0)
      func = generate_main;

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
             << "Available commands: merge, show, generate\n";
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
