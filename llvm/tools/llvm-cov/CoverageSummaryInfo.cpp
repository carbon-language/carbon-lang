//===- CoverageSummaryInfo.cpp - Coverage summary for function/file -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These structures are used to represent code coverage metrics
// for functions/files.
//
//===----------------------------------------------------------------------===//

#include "CoverageSummaryInfo.h"

using namespace llvm;
using namespace coverage;

FunctionCoverageSummary
FunctionCoverageSummary::get(const FunctionCoverageMapping &Function) {
  // Compute the region coverage
  size_t NumCodeRegions = 0, CoveredRegions = 0;
  for (auto &Region : Function.MappingRegions) {
    if (Region.Kind != CounterMappingRegion::CodeRegion)
      continue;
    ++NumCodeRegions;
    if (Region.ExecutionCount != 0)
      ++CoveredRegions;
  }

  // Compute the line coverage
  size_t NumLines = 0, CoveredLines = 0;
  for (unsigned FileID = 0, E = Function.Filenames.size(); FileID < E;
       ++FileID) {
    // Find the line start and end of the function's source code
    // in that particular file
    unsigned LineStart = std::numeric_limits<unsigned>::max();
    unsigned LineEnd = 0;
    for (auto &Region : Function.MappingRegions) {
      if (Region.FileID != FileID)
        continue;
      LineStart = std::min(LineStart, Region.LineStart);
      LineEnd = std::max(LineEnd, Region.LineEnd);
    }
    unsigned LineCount = LineEnd - LineStart + 1;

    // Get counters
    llvm::SmallVector<uint64_t, 16> ExecutionCounts;
    ExecutionCounts.resize(LineCount, 0);
    for (auto &Region : Function.MappingRegions) {
      if (Region.FileID != FileID)
        continue;
      // Ignore the lines that were skipped by the preprocessor.
      auto ExecutionCount = Region.ExecutionCount;
      if (Region.Kind == MappingRegion::SkippedRegion) {
        LineCount -= Region.LineEnd - Region.LineStart + 1;
        ExecutionCount = 1;
      }
      for (unsigned I = Region.LineStart; I <= Region.LineEnd; ++I)
        ExecutionCounts[I - LineStart] = ExecutionCount;
    }
    CoveredLines += LineCount - std::count(ExecutionCounts.begin(),
                                           ExecutionCounts.end(), 0);
    NumLines += LineCount;
  }
  return FunctionCoverageSummary(
      Function.PrettyName, RegionCoverageInfo(CoveredRegions, NumCodeRegions),
      LineCoverageInfo(CoveredLines, 0, NumLines));
}

FileCoverageSummary
FileCoverageSummary::get(StringRef Name,
                         ArrayRef<FunctionCoverageSummary> FunctionSummaries) {
  size_t NumRegions = 0, CoveredRegions = 0;
  size_t NumLines = 0, NonCodeLines = 0, CoveredLines = 0;
  size_t NumFunctionsCovered = 0;
  for (const auto &Func : FunctionSummaries) {
    CoveredRegions += Func.RegionCoverage.Covered;
    NumRegions += Func.RegionCoverage.NumRegions;

    CoveredLines += Func.LineCoverage.Covered;
    NonCodeLines += Func.LineCoverage.NonCodeLines;
    NumLines += Func.LineCoverage.NumLines;

    if (Func.RegionCoverage.isFullyCovered())
      ++NumFunctionsCovered;
  }

  return FileCoverageSummary(
      Name, RegionCoverageInfo(CoveredRegions, NumRegions),
      LineCoverageInfo(CoveredLines, NonCodeLines, NumLines),
      FunctionCoverageInfo(NumFunctionsCovered, FunctionSummaries.size()),
      FunctionSummaries);
}
