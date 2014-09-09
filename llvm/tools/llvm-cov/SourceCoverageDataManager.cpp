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
  Uniqued = false;
}

ArrayRef<CountedRegion> SourceCoverageDataManager::getSourceRegions() {
  if (Uniqued || Regions.size() <= 1)
    return Regions;

  // Sort.
  std::sort(Regions.begin(), Regions.end());

  // Merge duplicate source ranges and sum their execution counts.
  auto Prev = Regions.begin();
  for (auto I = Prev + 1, E = Regions.end(); I != E; ++I) {
    if (I->coversSameSource(*Prev)) {
      Prev->ExecutionCount += I->ExecutionCount;
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
