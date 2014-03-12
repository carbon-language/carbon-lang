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
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string> Filename1(cl::Positional, cl::Required,
                                      cl::desc("file1"));
static cl::opt<std::string> Filename2(cl::Positional, cl::Required,
                                      cl::desc("file2"));

static cl::opt<std::string> OutputFilename("output", cl::value_desc("output"),
                                           cl::init("-"),
                                           cl::desc("Output file"));
static cl::alias OutputFilenameA("o", cl::desc("Alias for --output"),
                                 cl::aliasopt(OutputFilename));

static bool readLine(const char *&Start, const char *End, StringRef &S) {
  if (Start == End)
    return false;

  for (const char *I = Start; I != End; ++I) {
    assert(*I && "unexpected binary data");
    if (*I == '\n') {
      S = StringRef(Start, I - Start);
      Start = I + 1;
      return true;
    }
  }

  S = StringRef(Start, End - Start);
  Start = End;
  return true;
}

static StringRef getWord(const char *&Start, const char *End) {
  for (const char *I = Start; I != End; ++I)
    if (*I == ' ') {
      StringRef S(Start, I - Start);
      Start = I + 1;
      return S;
    }
  StringRef S(Start, End - Start);
  Start = End;
  return S;
}

static size_t splitWords(const StringRef &Line, std::vector<StringRef> &Words) {
  const char *Start = Line.data();
  const char *End = Line.data() + Line.size();
  Words.clear();
  while (Start != End)
    Words.push_back(getWord(Start, End));
  return Words.size();
}

static bool getNumber(const StringRef &S, uint64_t &N) {
  N = 0;
  for (StringRef::iterator I = S.begin(), E = S.end(); I != E; ++I)
    if (*I >= '0' && *I <= '9')
      N = N * 10 + (*I - '0');
    else
      return false;

  return true;
}

static void exitWithError(const std::string &Message,
                          const std::string &Filename, int64_t Line = -1) {
  errs() << "error: " << Filename;
  if (Line >= 0)
    errs() << ":" << Line;
  errs() << ": " << Message << "\n";
  ::exit(1);
}

//===----------------------------------------------------------------------===//
int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

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

  const char *Start1 = File1->getBufferStart();
  const char *Start2 = File2->getBufferStart();
  const char *End1 = File1->getBufferEnd();
  const char *End2 = File2->getBufferEnd();
  const char *P1 = Start1;
  const char *P2 = Start2;

  StringRef Line1, Line2;
  int64_t Num = 0;
  while (readLine(P1, End1, Line1)) {
    ++Num;
    if (!readLine(P2, End2, Line2))
      exitWithError("truncated file", Filename2, Num);

    std::vector<StringRef> Words1, Words2;
    if (splitWords(Line1, Words1) != splitWords(Line2, Words2))
      exitWithError("data mismatch", Filename2, Num);

    if (Words1.size() > 2)
      exitWithError("invalid data", Filename1, Num);

    if (Words1.empty()) {
      Output << "\n";
      continue;
    }

    if (Words1.size() == 2) {
      if (Words1[0] != Words2[0])
        exitWithError("function name mismatch", Filename2, Num);

      uint64_t N1, N2;
      if (!getNumber(Words1[1], N1))
        exitWithError("bad function count", Filename1, Num);
      if (!getNumber(Words2[1], N2))
        exitWithError("bad function count", Filename2, Num);

      if (N1 != N2)
        exitWithError("function count mismatch", Filename2, Num);

      Output << Line1 << "\n";
      continue;
    }

    uint64_t N1, N2;
    if (!getNumber(Words1[0], N1))
      exitWithError("invalid counter", Filename1, Num);
    if (!getNumber(Words2[0], N2))
      exitWithError("invalid counter", Filename2, Num);

    uint64_t Sum = N1 + N2;
    if (Sum < N1)
      exitWithError("counter overflow", Filename2, Num);

    Output << N1 + N2 << "\n";
  }
  if (readLine(P2, End2, Line2))
    exitWithError("truncated file", Filename1, Num + 1);

  return 0;
}
