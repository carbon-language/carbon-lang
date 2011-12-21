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

static llvm::cl::opt<unsigned>
MemoryLimitMB("memory-limit", llvm::cl::desc(
                "Do not use more megabytes of memory"),
	          llvm::cl::init(1000));

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
  llvm::SourceMgr SM;
  llvm::JSONParser Parser(JSONText, &SM);
  if (!Parser.validate()) {
    llvm::errs() << "Parsing error in JSON parser benchmark.\n";
    exit(1);
  }
  Parsing.stopTimer();
}

std::string createJSONText(size_t MemoryMB, unsigned ValueSize) {
  std::string JSONText;
  llvm::raw_string_ostream Stream(JSONText);
  Stream << "[\n";
  size_t MemoryBytes = MemoryMB * 1024 * 1024;
  while (JSONText.size() < MemoryBytes) {
    Stream << " {\n"
           << "  \"key1\": \"" << std::string(ValueSize, '*') << "\",\n"
           << "  \"key2\": \"" << std::string(ValueSize, '*') << "\",\n"
           << "  \"key3\": \"" << std::string(ValueSize, '*') << "\"\n"
           << " }";
    Stream.flush();
    if (JSONText.size() < MemoryBytes) Stream << ",";
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
    benchmark(Group, "Fast", createJSONText(10, 500));
  } else {
    benchmark(Group, "Small Values", createJSONText(MemoryLimitMB, 5));
    benchmark(Group, "Medium Values", createJSONText(MemoryLimitMB, 500));
    benchmark(Group, "Large Values", createJSONText(MemoryLimitMB, 50000));
  }
  return 0;
}

