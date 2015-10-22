//===- FuzzerLoop.cpp - Fuzzer's main loop --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Fuzzer's main loop.
//===----------------------------------------------------------------------===//

#include "FuzzerInternal.h"
#include <sanitizer/coverage_interface.h>
#include <algorithm>

extern "C" {
__attribute__((weak)) void __sanitizer_print_stack_trace();
}

namespace fuzzer {
static const size_t kMaxUnitSizeToPrint = 256;

// Only one Fuzzer per process.
static Fuzzer *F;

Fuzzer::Fuzzer(UserSuppliedFuzzer &USF, FuzzingOptions Options)
    : USF(USF), Options(Options) {
  SetDeathCallback();
  InitializeTraceState();
  assert(!F);
  F = this;
}

void Fuzzer::SetDeathCallback() {
  __sanitizer_set_death_callback(StaticDeathCallback);
}

void Fuzzer::PrintUnitInASCII(const Unit &U, const char *PrintAfter) {
  PrintASCII(U, PrintAfter);
}

void Fuzzer::StaticDeathCallback() {
  assert(F);
  F->DeathCallback();
}

void Fuzzer::DeathCallback() {
  Printf("DEATH:\n");
  if (CurrentUnit.size() <= kMaxUnitSizeToPrint) {
    Print(CurrentUnit, "\n");
    PrintUnitInASCII(CurrentUnit, "\n");
  }
  WriteUnitToFileWithPrefix(CurrentUnit, "crash-");
}

void Fuzzer::StaticAlarmCallback() {
  assert(F);
  F->AlarmCallback();
}

void Fuzzer::AlarmCallback() {
  assert(Options.UnitTimeoutSec > 0);
  size_t Seconds =
      duration_cast<seconds>(system_clock::now() - UnitStartTime).count();
  if (Seconds == 0) return;
  if (Options.Verbosity >= 2)
    Printf("AlarmCallback %zd\n", Seconds);
  if (Seconds >= (size_t)Options.UnitTimeoutSec) {
    Printf("ALARM: working on the last Unit for %zd seconds\n", Seconds);
    Printf("       and the timeout value is %d (use -timeout=N to change)\n",
           Options.UnitTimeoutSec);
    if (CurrentUnit.size() <= kMaxUnitSizeToPrint) {
      Print(CurrentUnit, "\n");
      PrintUnitInASCII(CurrentUnit, "\n");
    }
    WriteUnitToFileWithPrefix(CurrentUnit, "timeout-");
    Printf("==%d== ERROR: libFuzzer: timeout after %d seconds\n", GetPid(),
           Seconds);
    if (__sanitizer_print_stack_trace)
      __sanitizer_print_stack_trace();
    Printf("SUMMARY: libFuzzer: timeout\n");
    exit(1);
  }
}

void Fuzzer::PrintStats(const char *Where, size_t Cov, const char *End) {
  if (!Options.Verbosity) return;
  size_t Seconds = secondsSinceProcessStartUp();
  size_t ExecPerSec = (Seconds ? TotalNumberOfRuns / Seconds : 0);
  Printf("#%zd\t%s", TotalNumberOfRuns, Where);
  Printf(" cov: %zd", Cov);
  if (auto TB = TotalBits())
    Printf(" bits: %zd", TB);
  Printf(" units: %zd exec/s: %zd", Corpus.size(), ExecPerSec);
  if (TotalNumberOfExecutedTraceBasedMutations)
    Printf(" tbm: %zd", TotalNumberOfExecutedTraceBasedMutations);
  Printf("%s", End);
}

void Fuzzer::RereadOutputCorpus() {
  if (Options.OutputCorpus.empty()) return;
  std::vector<Unit> AdditionalCorpus;
  ReadDirToVectorOfUnits(Options.OutputCorpus.c_str(), &AdditionalCorpus,
                         &EpochOfLastReadOfOutputCorpus);
  if (Corpus.empty()) {
    Corpus = AdditionalCorpus;
    return;
  }
  if (!Options.Reload) return;
  if (Options.Verbosity >= 2)
    Printf("Reload: read %zd new units.\n",  AdditionalCorpus.size());
  for (auto &X : AdditionalCorpus) {
    if (X.size() > (size_t)Options.MaxLen)
      X.resize(Options.MaxLen);
    if (UnitHashesAddedToCorpus.insert(Hash(X)).second) {
      CurrentUnit.clear();
      CurrentUnit.insert(CurrentUnit.begin(), X.begin(), X.end());
      if (RunOne(CurrentUnit)) {
        Corpus.push_back(X);
        if (Options.Verbosity >= 1)
          PrintStats("RELOAD", LastRecordedBlockCoverage);
      }
    }
  }
}

void Fuzzer::ShuffleAndMinimize() {
  bool PreferSmall = (Options.PreferSmallDuringInitialShuffle == 1 ||
                      (Options.PreferSmallDuringInitialShuffle == -1 &&
                       USF.GetRand().RandBool()));
  if (Options.Verbosity)
    Printf("PreferSmall: %d\n", PreferSmall);
  PrintStats("READ  ", 0);
  std::vector<Unit> NewCorpus;
  if (Options.ShuffleAtStartUp) {
    std::random_shuffle(Corpus.begin(), Corpus.end(), USF.GetRand());
    if (PreferSmall)
      std::stable_sort(
          Corpus.begin(), Corpus.end(),
          [](const Unit &A, const Unit &B) { return A.size() < B.size(); });
  }
  Unit &U = CurrentUnit;
  for (const auto &C : Corpus) {
    for (size_t First = 0; First < 1; First++) {
      U.clear();
      size_t Last = std::min(First + Options.MaxLen, C.size());
      U.insert(U.begin(), C.begin() + First, C.begin() + Last);
      if (Options.OnlyASCII)
        ToASCII(U);
      if (RunOne(U)) {
        NewCorpus.push_back(U);
        if (Options.Verbosity >= 2)
          Printf("NEW0: %zd L %zd\n", LastRecordedBlockCoverage, U.size());
      }
    }
  }
  Corpus = NewCorpus;
  for (auto &X : Corpus)
    UnitHashesAddedToCorpus.insert(Hash(X));
  PrintStats("INITED", LastRecordedBlockCoverage);
}

bool Fuzzer::RunOne(const Unit &U) {
  UnitStartTime = system_clock::now();
  TotalNumberOfRuns++;

  PrepareCoverageBeforeRun();
  ExecuteCallback(U);
  bool Res = CheckCoverageAfterRun();

  auto UnitStopTime = system_clock::now();
  auto TimeOfUnit =
      duration_cast<seconds>(UnitStopTime - UnitStartTime).count();
  if (!(TotalNumberOfRuns & (TotalNumberOfRuns - 1)) && Options.Verbosity)
    PrintStats("pulse ", LastRecordedBlockCoverage);
  if (TimeOfUnit > TimeOfLongestUnitInSeconds &&
      TimeOfUnit >= Options.ReportSlowUnits) {
    TimeOfLongestUnitInSeconds = TimeOfUnit;
    Printf("Slowest unit: %zd s:\n", TimeOfLongestUnitInSeconds);
    WriteUnitToFileWithPrefix(U, "slow-unit-");
  }
  return Res;
}

void Fuzzer::RunOneAndUpdateCorpus(Unit &U) {
  if (TotalNumberOfRuns >= Options.MaxNumberOfRuns)
    return;
  if (Options.OnlyASCII)
    ToASCII(U);
  if (RunOne(U))
    ReportNewCoverage(U);
}

void Fuzzer::ExecuteCallback(const Unit &U) {
  int Res = USF.TargetFunction(U.data(), U.size());
  (void)Res;
  assert(Res == 0);
}

size_t Fuzzer::RecordBlockCoverage() {
  return LastRecordedBlockCoverage = __sanitizer_get_total_unique_coverage();
}

void Fuzzer::PrepareCoverageBeforeRun() {
  if (Options.UseCounters) {
    size_t NumCounters = __sanitizer_get_number_of_counters();
    CounterBitmap.resize(NumCounters);
    __sanitizer_update_counter_bitset_and_clear_counters(0);
  }
  RecordBlockCoverage();
}

bool Fuzzer::CheckCoverageAfterRun() {
  size_t OldCoverage = LastRecordedBlockCoverage;
  size_t NewCoverage = RecordBlockCoverage();
  size_t NumNewBits = 0;
  if (Options.UseCounters)
    NumNewBits = __sanitizer_update_counter_bitset_and_clear_counters(
        CounterBitmap.data());
  return NewCoverage > OldCoverage || NumNewBits;
}

void Fuzzer::WriteToOutputCorpus(const Unit &U) {
  if (Options.OutputCorpus.empty()) return;
  std::string Path = DirPlusFile(Options.OutputCorpus, Hash(U));
  WriteToFile(U, Path);
  if (Options.Verbosity >= 2)
    Printf("Written to %s\n", Path.c_str());
  assert(!Options.OnlyASCII || IsASCII(U));
}

void Fuzzer::WriteUnitToFileWithPrefix(const Unit &U, const char *Prefix) {
  if (!Options.SaveArtifacts)
    return;
  std::string Path = Options.ArtifactPrefix + Prefix + Hash(U);
  WriteToFile(U, Path);
  Printf("artifact_prefix='%s'; Test unit written to %s\n",
         Options.ArtifactPrefix.c_str(), Path.c_str());
  if (U.size() <= kMaxUnitSizeToPrint) {
    Printf("Base64: ");
    PrintFileAsBase64(Path);
  }
}

void Fuzzer::SaveCorpus() {
  if (Options.OutputCorpus.empty()) return;
  for (const auto &U : Corpus)
    WriteToFile(U, DirPlusFile(Options.OutputCorpus, Hash(U)));
  if (Options.Verbosity)
    Printf("Written corpus of %zd files to %s\n", Corpus.size(),
           Options.OutputCorpus.c_str());
}

void Fuzzer::ReportNewCoverage(const Unit &U) {
  Corpus.push_back(U);
  UnitHashesAddedToCorpus.insert(Hash(U));
  PrintStats("NEW   ", LastRecordedBlockCoverage, "");
  if (Options.Verbosity) {
    Printf(" L: %zd", U.size());
    if (U.size() < 30) {
      Printf(" ");
      PrintUnitInASCII(U, "\t");
      Print(U);
    }
    Printf("\n");
  }
  WriteToOutputCorpus(U);
  if (Options.ExitOnFirst)
    exit(0);
}

void Fuzzer::MutateAndTestOne(Unit *U) {
  for (int i = 0; i < Options.MutateDepth; i++) {
    StartTraceRecording();
    size_t Size = U->size();
    U->resize(Options.MaxLen);
    size_t NewSize = USF.Mutate(U->data(), Size, U->size());
    assert(NewSize > 0 && "Mutator returned empty unit");
    assert(NewSize <= (size_t)Options.MaxLen &&
           "Mutator return overisized unit");
    U->resize(NewSize);
    RunOneAndUpdateCorpus(*U);
    size_t NumTraceBasedMutations = StopTraceRecording();
    size_t TBMWidth =
        std::min((size_t)Options.TBMWidth, NumTraceBasedMutations);
    size_t TBMDepth =
        std::min((size_t)Options.TBMDepth, NumTraceBasedMutations);
    Unit BackUp = *U;
    for (size_t w = 0; w < TBMWidth; w++) {
      *U = BackUp;
      for (size_t d = 0; d < TBMDepth; d++) {
        TotalNumberOfExecutedTraceBasedMutations++;
        ApplyTraceBasedMutation(USF.GetRand()(NumTraceBasedMutations), U);
        RunOneAndUpdateCorpus(*U);
      }
    }
  }
}

void Fuzzer::Loop() {
  for (auto &U: Options.Dictionary)
    USF.GetMD().AddWordToDictionary(U.data(), U.size());

  while (true) {
    for (size_t J1 = 0; J1 < Corpus.size(); J1++) {
      SyncCorpus();
      RereadOutputCorpus();
      if (TotalNumberOfRuns >= Options.MaxNumberOfRuns)
        return;
      if (Options.MaxTotalTimeSec > 0 &&
          secondsSinceProcessStartUp() >
              static_cast<size_t>(Options.MaxTotalTimeSec))
        return;
      CurrentUnit = Corpus[J1];
      // Optionally, cross with another unit.
      if (Options.DoCrossOver && USF.GetRand().RandBool()) {
        size_t J2 = USF.GetRand()(Corpus.size());
        if (!Corpus[J1].empty() && !Corpus[J2].empty()) {
          assert(!Corpus[J2].empty());
          CurrentUnit.resize(Options.MaxLen);
          size_t NewSize = USF.CrossOver(
              Corpus[J1].data(), Corpus[J1].size(), Corpus[J2].data(),
              Corpus[J2].size(), CurrentUnit.data(), CurrentUnit.size());
          assert(NewSize > 0 && "CrossOver returned empty unit");
          assert(NewSize <= (size_t)Options.MaxLen &&
                 "CrossOver returned overisized unit");
          CurrentUnit.resize(NewSize);
        }
      }
      // Perform several mutations and runs.
      MutateAndTestOne(&CurrentUnit);
    }
  }
}

void Fuzzer::SyncCorpus() {
  if (Options.SyncCommand.empty() || Options.OutputCorpus.empty()) return;
  auto Now = system_clock::now();
  if (duration_cast<seconds>(Now - LastExternalSync).count() <
      Options.SyncTimeout)
    return;
  LastExternalSync = Now;
  ExecuteCommand(Options.SyncCommand + " " + Options.OutputCorpus);
}

}  // namespace fuzzer
