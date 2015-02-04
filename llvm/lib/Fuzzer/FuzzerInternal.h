//===- FuzzerInternal.h - Internal header for the Fuzzer --------*- C++ -* ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Define the main class fuzzer::Fuzzer and most functions.
//===----------------------------------------------------------------------===//
#include <cassert>
#include <climits>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_set>

namespace fuzzer {
typedef std::vector<uint8_t> Unit;
using namespace std::chrono;

Unit ReadFile(const char *Path);
void ReadDirToVectorOfUnits(const char *Path, std::vector<Unit> *V);
void WriteToFile(const Unit &U, const std::string &Path);
void CopyFileToErr(const std::string &Path);
// Returns "Dir/FileName" or equivalent for the current OS.
std::string DirPlusFile(const std::string &DirPath,
                        const std::string &FileName);

void Mutate(Unit *U, size_t MaxLen);

void CrossOver(const Unit &A, const Unit &B, Unit *U, size_t MaxLen);

void Print(const Unit &U, const char *PrintAfter = "");
void PrintASCII(const Unit &U, const char *PrintAfter = "");
std::string Hash(const Unit &U);
void SetTimer(int Seconds);

class Fuzzer {
 public:
  struct FuzzingOptions {
    int Verbosity = 1;
    int MaxLen = 0;
    bool DoCrossOver = true;
    int  MutateDepth = 5;
    bool ExitOnFirst = false;
    bool UseFullCoverageSet  = false;
    int PreferSmallDuringInitialShuffle = -1;
    size_t MaxNumberOfRuns = ULONG_MAX;
    std::string OutputCorpus;
  };
  Fuzzer(FuzzingOptions Options) : Options(Options) {
    SetDeathCallback();
  }
  void AddToCorpus(const Unit &U) { Corpus.push_back(U); }
  size_t Loop(size_t NumIterations);
  void ShuffleAndMinimize();
  size_t CorpusSize() const { return Corpus.size(); }
  void ReadDir(const std::string &Path) {
    ReadDirToVectorOfUnits(Path.c_str(), &Corpus);
  }
  // Save the current corpus to OutputCorpus.
  void SaveCorpus();

  size_t secondsSinceProcessStartUp() {
    return duration_cast<seconds>(system_clock::now() - ProcessStartTime)
        .count();
  }

  size_t getTotalNumberOfRuns() { return TotalNumberOfRuns; }

  static void AlarmCallback();

 private:
  size_t MutateAndTestOne(Unit *U);
  size_t RunOne(const Unit &U);
  size_t RunOneMaximizeTotalCoverage(const Unit &U);
  size_t RunOneMaximizeFullCoverageSet(const Unit &U);
  void WriteToOutputCorpus(const Unit &U);
  static void WriteToCrash(const Unit &U, const char *Prefix);

  void SetDeathCallback();
  static void DeathCallback();
  static Unit CurrentUnit;

  size_t TotalNumberOfRuns = 0;

  std::vector<Unit> Corpus;
  std::unordered_set<uintptr_t> FullCoverageSets;
  FuzzingOptions Options;
  system_clock::time_point ProcessStartTime = system_clock::now();
  static system_clock::time_point UnitStartTime;
};

};  // namespace fuzzer
