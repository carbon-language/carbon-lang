//=-- InstrProfWriter.cpp - Instrumented profiling writer -------------------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing profiling data for clang's
// instrumentation based PGO and coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProfWriter.h"
#include "llvm/Support/Endian.h"

using namespace llvm;

error_code InstrProfWriter::addFunctionCounts(StringRef FunctionName,
                                              uint64_t FunctionHash,
                                              ArrayRef<uint64_t> Counters) {
  auto Where = FunctionData.find(FunctionName);
  if (Where == FunctionData.end()) {
    // If this is the first time we've seen this function, just add it.
    FunctionData[FunctionName] = {FunctionHash, Counters};
    return instrprof_error::success;;
  }

  auto &Data = Where->getValue();
  // We can only add to existing functions if they match, so we check the hash
  // and number of counters.
  if (Data.Hash != FunctionHash)
    return instrprof_error::hash_mismatch;
  if (Data.Counts.size() != Counters.size())
    return instrprof_error::count_mismatch;
  // These match, add up the counters.
  for (size_t I = 0, E = Counters.size(); I < E; ++I) {
    if (Data.Counts[I] + Counters[I] < Data.Counts[I])
      return instrprof_error::counter_overflow;
    Data.Counts[I] += Counters[I];
  }
  return instrprof_error::success;
}

void InstrProfWriter::write(raw_ostream &OS) {
  // Write out the counts for each function.
  for (const auto &I : FunctionData) {
    StringRef Name = I.getKey();
    uint64_t Hash = I.getValue().Hash;
    const std::vector<uint64_t> &Counts = I.getValue().Counts;

    OS << Name << "\n" << Hash << "\n" << Counts.size() << "\n";
    for (uint64_t Count : Counts)
      OS << Count << "\n";
    OS << "\n";
  }
}
