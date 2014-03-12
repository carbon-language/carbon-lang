//=-- ProfileDataReader.h - Instrumented profiling reader ---------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for reading profiling data for instrumentation
// based PGO and coverage.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILE_PROFILEDATA_READER_H__
#define LLVM_PROFILE_PROFILEDATA_READER_H__

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

namespace llvm {

class ProfileDataCursor;

/// Reader for the profile data that is used for instrumentation based PGO.
class ProfileDataReader {
private:
  /// The profile data file contents.
  std::unique_ptr<MemoryBuffer> DataBuffer;
  /// Offsets into DataBuffer for each function's counters.
  StringMap<uint32_t> DataOffsets;
  /// The maximal execution count among all functions.
  uint64_t MaxFunctionCount;

  ProfileDataReader(const ProfileDataReader &) LLVM_DELETED_FUNCTION;
  ProfileDataReader &operator=(const ProfileDataReader &) LLVM_DELETED_FUNCTION;
protected:
  ProfileDataReader(std::unique_ptr<MemoryBuffer> &DataBuffer)
      : DataBuffer(DataBuffer.release()) {}

  /// Populate internal state using the profile data's index
  error_code readIndex();
public:

  class name_iterator {
    typedef StringMap<unsigned>::const_iterator IterTy;
    IterTy Ix;
  public:
    explicit name_iterator(const IterTy &Ix) : Ix(Ix) {}

    StringRef operator*() const { return Ix->getKey(); }

    bool operator==(const name_iterator &RHS) const { return Ix == RHS.Ix; }
    bool operator!=(const name_iterator &RHS) const { return Ix != RHS.Ix; }

    inline name_iterator& operator++() { ++Ix; return *this; }
  };

  /// Iterators over the names of indexed items
  name_iterator begin() const {
    return name_iterator(DataOffsets.begin());
  }
  name_iterator end() const {
    return name_iterator(DataOffsets.end());
  }

private:
  error_code findFunctionCounts(StringRef FuncName, uint64_t &FunctionHash,
                                ProfileDataCursor &Cursor);
public:
  /// The number of profiled functions
  size_t numProfiledFunctions() { return DataOffsets.size(); }
  /// Fill Counts with the profile data for the given function name.
  error_code getFunctionCounts(StringRef FuncName, uint64_t &FunctionHash,
                               std::vector<uint64_t> &Counts);
  /// Return the maximum of all known function counts.
  uint64_t getMaximumFunctionCount() { return MaxFunctionCount; }

  static error_code create(std::string Path,
                           std::unique_ptr<ProfileDataReader> &Result);
};

} // end namespace llvm

#endif // LLVM_PROFILE_PROFILEDATA_READER_H__
