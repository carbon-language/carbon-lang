//===- SourceCoverageDataManager.cpp - Manager for source file coverage
// data-===//
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

#include "SourceCoverageDataManager.h"

using namespace llvm;
using namespace coverage;

void SourceCoverageDataManager::insert(const CountedRegion &CR) {
  Regions.push_back(CR);
  Segments.clear();
}

namespace {
class SegmentBuilder {
  std::vector<CoverageSegment> Segments;
  SmallVector<const CountedRegion *, 8> ActiveRegions;

  /// Start a segment with no count specified.
  void startSegment(unsigned Line, unsigned Col, bool IsRegionEntry) {
    Segments.emplace_back(Line, Col, IsRegionEntry);
  }

  /// Start a segment with the given Region's count.
  void startSegment(unsigned Line, unsigned Col, bool IsRegionEntry,
                    const CountedRegion &Region) {
    if (Segments.empty())
      Segments.emplace_back(Line, Col, IsRegionEntry);
    CoverageSegment S = Segments.back();
    // Avoid creating empty regions.
    if (S.Line != Line || S.Col != Col) {
      Segments.emplace_back(Line, Col, IsRegionEntry);
      S = Segments.back();
    }
    // Set this region's count.
    if (Region.Kind != coverage::CounterMappingRegion::SkippedRegion)
      Segments.back().setCount(Region.ExecutionCount);
  }

  /// Start a segment for the given region.
  void startSegment(const CountedRegion &Region) {
    startSegment(Region.LineStart, Region.ColumnStart, true, Region);
  }

  /// Pop the top region off of the active stack, starting a new segment with
  /// the containing Region's count.
  void popRegion() {
    const CountedRegion *Active = ActiveRegions.back();
    unsigned Line = Active->LineEnd, Col = Active->ColumnEnd;
    ActiveRegions.pop_back();
    if (ActiveRegions.empty())
      startSegment(Line, Col, /*IsRegionEntry=*/false);
    else
      startSegment(Line, Col, /*IsRegionEntry=*/false, *ActiveRegions.back());
  }

public:
  /// Build a list of CoverageSegments from a sorted list of Regions.
  std::vector<CoverageSegment>
  buildSegments(ArrayRef<CountedRegion> Regions) {
    for (const auto &Region : Regions) {
      // Pop any regions that end before this one starts.
      while (!ActiveRegions.empty() &&
             ActiveRegions.back()->endLoc() <= Region.startLoc())
        popRegion();
      // Add this region to the stack.
      ActiveRegions.push_back(&Region);
      startSegment(Region);
    }
    // Pop any regions that are left in the stack.
    while (!ActiveRegions.empty())
      popRegion();
    return Segments;
  }
};
}

ArrayRef<CoverageSegment> SourceCoverageDataManager::getCoverageSegments() {
  if (Segments.empty()) {
    // Sort the regions given that they're all in the same file at this point.
    std::sort(Regions.begin(), Regions.end(),
              [](const CountedRegion &LHS, const CountedRegion &RHS) {
      if (LHS.startLoc() == RHS.startLoc())
        // When LHS completely contains RHS, we sort LHS first.
        return RHS.endLoc() < LHS.endLoc();
      return LHS.startLoc() < RHS.startLoc();
    });

    Segments = SegmentBuilder().buildSegments(Regions);
  }

  return Segments;
}
