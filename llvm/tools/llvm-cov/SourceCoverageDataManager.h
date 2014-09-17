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

struct CoverageSegment {
  unsigned Line;
  unsigned Col;
  bool IsRegionEntry;
  uint64_t Count;
  bool HasCount;

  CoverageSegment(unsigned Line, unsigned Col, bool IsRegionEntry)
      : Line(Line), Col(Col), IsRegionEntry(IsRegionEntry),
        Count(0), HasCount(false) {}
  void setCount(uint64_t NewCount) {
    Count = NewCount;
    HasCount = true;
  }
};

/// \brief Partions mapping regions by their kind and sums
/// the execution counts of the regions that start at the same location.
class SourceCoverageDataManager {
  std::vector<coverage::CountedRegion> Regions;
  std::vector<CoverageSegment> Segments;

public:
  void insert(const coverage::CountedRegion &CR);

  /// \brief Return a sequence of non-overlapping coverage segments.
  ArrayRef<CoverageSegment> getCoverageSegments();
};

} // namespace llvm

#endif // LLVM_COV_SOURCECOVERAGEDATAMANAGER_H
