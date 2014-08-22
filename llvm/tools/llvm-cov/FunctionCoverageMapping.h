//===- FunctionCoverageMapping.h - Function coverage mapping record -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A structure that stores the coverage mapping record for a single function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_COV_FUNCTIONCOVERAGEMAPPING_H
#define LLVM_COV_FUNCTIONCOVERAGEMAPPING_H

#include <string>
#include <vector>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ProfileData/CoverageMapping.h"

namespace llvm {

/// \brief Associates a source range with an execution count.
struct MappingRegion : public coverage::CounterMappingRegion {
  uint64_t ExecutionCount;

  MappingRegion(const CounterMappingRegion &R, uint64_t ExecutionCount)
      : CounterMappingRegion(R), ExecutionCount(ExecutionCount) {}
};

/// \brief Stores all the required information
/// about code coverage for a single function.
struct FunctionCoverageMapping {
  /// \brief Raw function name.
  std::string Name;
  /// \brief Demangled function name.
  std::string PrettyName;
  std::vector<std::string> Filenames;
  std::vector<MappingRegion> MappingRegions;

  FunctionCoverageMapping(StringRef Name, ArrayRef<StringRef> Filenames)
      : Name(Name), PrettyName(Name),
        Filenames(Filenames.begin(), Filenames.end()) {}
};

} // namespace llvm

#endif // LLVM_COV_FUNCTIONCOVERAGEMAPPING_H
