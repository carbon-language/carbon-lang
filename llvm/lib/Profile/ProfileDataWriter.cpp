//=-- ProfileDataWriter.cpp - Instrumented profiling writer -----------------=//
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

#include "llvm/Profile/ProfileDataWriter.h"
#include "llvm/Profile/ProfileData.h"
#include "llvm/Support/Endian.h"

using namespace llvm;

template <typename T>
struct LEBytes {
  const T &Data;
  LEBytes(const T &Data) : Data(Data) {}
  void print(raw_ostream &OS) const {
    for (uint32_t Shift = 0; Shift < sizeof(Data); ++Shift)
      OS << (char)((Data >> (8 * Shift)) & 0xFF);
  }
};
template <typename T>
static raw_ostream &operator<<(raw_ostream &OS, const LEBytes<T> &Bytes) {
  Bytes.print(OS);
  return OS;
}

void ProfileDataWriter::addFunctionCounts(StringRef FuncName,
                                          uint64_t FunctionHash,
                                          uint64_t NumCounters,
                                          const uint64_t *Counters) {
  DataStart += 2 * sizeof(uint32_t) + FuncName.size();
  FunctionOffsets[FuncName] = FunctionData.size() * sizeof(uint64_t);
  FunctionData.push_back(FunctionHash);
  FunctionData.push_back(NumCounters);
  assert(NumCounters > 0 && "Function call counter missing!");
  if (Counters[0] > MaxFunctionCount)
    MaxFunctionCount = Counters[0];
  for (uint64_t I = 0; I < NumCounters; ++I)
    FunctionData.push_back(Counters[I]);
}

void ProfileDataWriter::write(raw_ostream &OS) {
  for (char C : PROFILEDATA_MAGIC)
    OS << C;
  OS << LEBytes<uint32_t>(PROFILEDATA_VERSION);
  OS << LEBytes<uint32_t>(DataStart);
  OS << LEBytes<uint32_t>(0);
  OS << LEBytes<uint64_t>(MaxFunctionCount);

  for (const auto &I : FunctionOffsets) {
    StringRef Name = I.getKey();
    OS << LEBytes<uint32_t>(Name.size());
    OS << Name;
    OS << LEBytes<uint32_t>(I.getValue());
  }

  for (unsigned I = 0; I < sizeof(uint64_t) - DataStart % sizeof(uint64_t); ++I)
    OS << '\0';

  for (uint64_t Value : FunctionData)
    OS << LEBytes<uint64_t>(Value);
}
