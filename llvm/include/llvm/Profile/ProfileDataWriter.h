//=-- ProfileDataWriter.h - Instrumented profiling writer ---------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing profiling data for instrumentation
// based PGO and coverage.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILE_PROFILEDATA_WRITER_H__
#define LLVM_PROFILE_PROFILEDATA_WRITER_H__

#include "llvm/ADT/StringMap.h"
#include "llvm/Profile/ProfileData.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

namespace llvm {

/// Writer for instrumentation based profile data
class ProfileDataWriter {
  StringMap<size_t> FunctionOffsets;
  std::vector<uint64_t> FunctionData;
  uint32_t DataStart;
  uint64_t MaxFunctionCount;

  void write32(raw_ostream &OS, uint32_t Value);
  void write64(raw_ostream &OS, uint64_t Value);
public:
  ProfileDataWriter()
      : DataStart(PROFILEDATA_HEADER_SIZE), MaxFunctionCount(0) {}

  void addFunctionCounts(StringRef FuncName, uint64_t FunctionHash,
                         uint64_t NumCounters, const uint64_t *Counters);
  void write(raw_ostream &OS);
};

} // end namespace llvm

#endif // LLVM_PROFILE_PROFILEDATA_WRITER_H__
