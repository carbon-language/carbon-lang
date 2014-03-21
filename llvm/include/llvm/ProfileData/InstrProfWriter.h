//=-- InstrProfWriter.h - Instrumented profiling writer -----------*- C++ -*-=//
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

#ifndef LLVM_PROFILEDATA_INSTRPROF_WRITER_H_
#define LLVM_PROFILEDATA_INSTRPROF_WRITER_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

namespace llvm {

/// Writer for instrumentation based profile data.
class InstrProfWriter {
public:
  struct CounterData {
    uint64_t Hash;
    std::vector<uint64_t> Counts;
  };
private:
  StringMap<CounterData> FunctionData;
public:
  /// Add function counts for the given function. If there are already counts
  /// for this function and the hash and number of counts match, each counter is
  /// summed.
  error_code addFunctionCounts(StringRef FunctionName, uint64_t FunctionHash,
                               ArrayRef<uint64_t> Counters);
  /// Ensure that all data is written to disk.
  void write(raw_ostream &OS);
};

} // end namespace llvm

#endif // LLVM_PROFILE_INSTRPROF_WRITER_H_
