//===- JSONBench - Benchmark the JSONParser implementation ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program executes the JSONParser on differntly sized JSON texts and
// outputs the run time.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSONParser.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<bool>
Verify("verify", llvm::cl::desc(
         "Run a quick verification useful for regression testing"),
       llvm::cl::init(false));

void benchmark(llvm::TimerGroup &Group, llvm::StringRef Name,
               llvm::StringRef JSONText) {
  llvm::Timer BaseLine((Name + ": Loop").str(), Group);
  BaseLine.startTimer();
  char C = 0;
  for (llvm::StringRef::iterator I = JSONText.begin(),
                                 E = JSONText.end();
       I != E; ++I) { C += *I; }
  BaseLine.stopTimer();
  volatile char DontOptimizeOut = C; (void)DontOptimizeOut;

  llvm::Timer Parsing((Name + ": Parsing").str(), Group);
  Parsing.startTimer();
  llvm::JSONParser Parser(JSONText);
  if (!Parser.validate()) {
    llvm::errs() << "Parsing error in JSON parser benchmark.\n";
    exit(1);
  }
  Parsing.stopTimer();
}

std::string createJSONText(int N, int ValueSize) {
  std::string JSONText;
  llvm::raw_string_ostream Stream(JSONText);
  Stream << "[\n";
  for (int I = 0; I < N; ++I) {
    Stream << " {\n"
           << "  \"key1\": \"" << std::string(ValueSize, '*') << "\",\n"
           << "  \"key2\": \"" << std::string(ValueSize, '*') << "\",\n"
           << "  \"key3\": \"" << std::string(ValueSize, '*') << "\"\n"
           << " }";
    if (I + 1 < N) Stream << ",";
    Stream << "\n";
  }
  Stream << "]\n";
  Stream.flush();
  return JSONText;
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  llvm::TimerGroup Group("JSON parser benchmark");
  if (Verify) {
    benchmark(Group, "Fast", createJSONText(1000, 500));
  } else {
    benchmark(Group, "Small Values", createJSONText(1000000, 5));
    benchmark(Group, "Medium Values", createJSONText(1000000, 500));
    benchmark(Group, "Large Values", createJSONText(10000, 50000));
  }
  return 0;
}

