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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/OnDiskHashTable.h"

#include "InstrProfIndexed.h"

using namespace llvm;

namespace {
class InstrProfRecordTrait {
public:
  typedef StringRef key_type;
  typedef StringRef key_type_ref;

  typedef InstrProfWriter::CounterData data_type;
  typedef const InstrProfWriter::CounterData &data_type_ref;

  typedef uint64_t hash_value_type;
  typedef uint64_t offset_type;

  static hash_value_type ComputeHash(key_type_ref K) {
    return IndexedInstrProf::ComputeHash(IndexedInstrProf::HashType, K);
  }

  static std::pair<offset_type, offset_type>
  EmitKeyDataLength(raw_ostream &Out, key_type_ref K, data_type_ref V) {
    using namespace llvm::support;
    endian::Writer<little> LE(Out);

    offset_type N = K.size();
    LE.write<offset_type>(N);

    offset_type M = (1 + V.Counts.size()) * sizeof(uint64_t);
    LE.write<offset_type>(M);

    return std::make_pair(N, M);
  }

  static void EmitKey(raw_ostream &Out, key_type_ref K, offset_type N){
    Out.write(K.data(), N);
  }

  static void EmitData(raw_ostream &Out, key_type_ref, data_type_ref V,
                       offset_type) {
    using namespace llvm::support;
    endian::Writer<little> LE(Out);
    LE.write<uint64_t>(V.Hash);
    for (uint64_t I : V.Counts)
      LE.write<uint64_t>(I);
  }
};
}

error_code InstrProfWriter::addFunctionCounts(StringRef FunctionName,
                                              uint64_t FunctionHash,
                                              ArrayRef<uint64_t> Counters) {
  auto Where = FunctionData.find(FunctionName);
  if (Where == FunctionData.end()) {
    // If this is the first time we've seen this function, just add it.
    auto &Data = FunctionData[FunctionName];
    Data.Hash = FunctionHash;
    Data.Counts = Counters;
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

void InstrProfWriter::write(raw_fd_ostream &OS) {
  OnDiskChainedHashTableGenerator<InstrProfRecordTrait> Generator;
  uint64_t MaxFunctionCount = 0;

  // Populate the hash table generator.
  for (const auto &I : FunctionData) {
    Generator.insert(I.getKey(), I.getValue());
    if (I.getValue().Counts[0] > MaxFunctionCount)
      MaxFunctionCount = I.getValue().Counts[0];
  }

  using namespace llvm::support;
  endian::Writer<little> LE(OS);

  // Write the header.
  LE.write<uint64_t>(IndexedInstrProf::Magic);
  LE.write<uint64_t>(IndexedInstrProf::Version);
  LE.write<uint64_t>(MaxFunctionCount);
  LE.write<uint64_t>(static_cast<uint64_t>(IndexedInstrProf::HashType));

  // Save a space to write the hash table start location.
  uint64_t HashTableStartLoc = OS.tell();
  LE.write<uint64_t>(0);
  // Write the hash table.
  uint64_t HashTableStart = Generator.Emit(OS);

  // Go back and fill in the hash table start.
  OS.seek(HashTableStartLoc);
  LE.write<uint64_t>(HashTableStart);
}
