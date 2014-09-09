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

/// \brief Stores all the required information
/// about code coverage for a single function.
struct FunctionCoverageMapping {
  /// \brief Raw function name.
  std::string Name;
  std::vector<std::string> Filenames;
  std::vector<coverage::CountedRegion> CountedRegions;

  FunctionCoverageMapping(StringRef Name, ArrayRef<StringRef> Filenames)
      : Name(Name), Filenames(Filenames.begin(), Filenames.end()) {}
};

} // namespace llvm

#endif // LLVM_COV_FUNCTIONCOVERAGEMAPPING_H
