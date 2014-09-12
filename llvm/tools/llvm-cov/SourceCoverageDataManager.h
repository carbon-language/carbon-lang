//===- SourceCoverageDataManager.h - Manager for source file coverage data-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class separates and merges mapping regions for a specific source file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_COV_SOURCECOVERAGEDATAMANAGER_H
#define LLVM_COV_SOURCECOVERAGEDATAMANAGER_H

#include "llvm/ProfileData/CoverageMapping.h"
#include <vector>

namespace llvm {

/// \brief Partions mapping regions by their kind and sums
/// the execution counts of the regions that start at the same location.
class SourceCoverageDataManager {
  std::vector<coverage::CountedRegion> Regions;
  bool Uniqued;

public:
  SourceCoverageDataManager() : Uniqued(false) {}

  void insert(const coverage::CountedRegion &CR);

  /// \brief Return the source regions in order of first to last occurring.
  ArrayRef<coverage::CountedRegion> getSourceRegions();
};

} // namespace llvm

#endif // LLVM_COV_SOURCECOVERAGEDATAMANAGER_H
