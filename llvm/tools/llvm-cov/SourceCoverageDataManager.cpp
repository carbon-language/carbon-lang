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
  SourceRange Range(CR.LineStart, CR.ColumnStart, CR.LineEnd, CR.ColumnEnd);
  if (CR.Kind == CounterMappingRegion::SkippedRegion) {
    SkippedRegions.push_back(Range);
    return;
  }
  Regions.push_back(std::make_pair(Range, CR.ExecutionCount));
}

ArrayRef<std::pair<SourceCoverageDataManager::SourceRange, uint64_t>>
SourceCoverageDataManager::getSourceRegions() {
  if (Uniqued || Regions.size() <= 1)
    return Regions;

  // Sort.
  std::sort(Regions.begin(), Regions.end(),
            [](const std::pair<SourceRange, uint64_t> &LHS,
               const std::pair<SourceRange, uint64_t> &RHS) {
    return LHS.first < RHS.first;
  });

  // Merge duplicate source ranges and sum their execution counts.
  auto Prev = Regions.begin();
  for (auto I = Prev + 1, E = Regions.end(); I != E; ++I) {
    if (I->first == Prev->first) {
      Prev->second += I->second;
      continue;
    }
    ++Prev;
    *Prev = *I;
  }
  ++Prev;
  Regions.erase(Prev, Regions.end());

  Uniqued = true;
  return Regions;
}
