//===- CoverageSummary.cpp - Code coverage summary ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements data management and rendering for the code coverage
// summaries of all files and functions.
//
//===----------------------------------------------------------------------===//

#include "CoverageSummary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"

using namespace llvm;

unsigned CoverageSummary::getFileID(StringRef Filename) {
  for (unsigned I = 0, E = Filenames.size(); I < E; ++I) {
    if (sys::fs::equivalent(Filenames[I], Filename))
      return I;
  }
  Filenames.push_back(Filename);
  return Filenames.size() - 1;
}

void
CoverageSummary::createSummaries(ArrayRef<coverage::FunctionRecord> Functions) {
  std::vector<std::pair<unsigned, size_t>> FunctionFileIDs;

  FunctionFileIDs.resize(Functions.size());
  for (size_t I = 0, E = Functions.size(); I < E; ++I) {
    StringRef Filename = Functions[I].Filenames[0];
    FunctionFileIDs[I] = std::make_pair(getFileID(Filename), I);
  }

  // Sort the function records by file ids
  std::sort(FunctionFileIDs.begin(), FunctionFileIDs.end(),
            [](const std::pair<unsigned, size_t> &lhs,
               const std::pair<unsigned, size_t> &rhs) {
    return lhs.first < rhs.first;
  });

  // Create function summaries in a sorted order (by file ids)
  FunctionSummaries.reserve(Functions.size());
  for (size_t I = 0, E = Functions.size(); I < E; ++I)
    FunctionSummaries.push_back(
        FunctionCoverageSummary::get(Functions[FunctionFileIDs[I].second]));

  // Create file summaries
  size_t CurrentSummary = 0;
  for (unsigned FileID = 0; FileID < Filenames.size(); ++FileID) {
    // Gather the relevant functions summaries
    auto PrevSummary = CurrentSummary;
    while (CurrentSummary < FunctionSummaries.size() &&
           FunctionFileIDs[CurrentSummary].first == FileID)
      ++CurrentSummary;
    ArrayRef<FunctionCoverageSummary> LocalSummaries(
        FunctionSummaries.data() + PrevSummary,
        FunctionSummaries.data() + CurrentSummary);
    if (LocalSummaries.empty())
      continue;

    FileSummaries.push_back(
        FileCoverageSummary::get(Filenames[FileID], LocalSummaries));
  }
}

FileCoverageSummary CoverageSummary::getCombinedFileSummaries() {
  size_t NumRegions = 0, CoveredRegions = 0;
  size_t NumLines = 0, NonCodeLines = 0, CoveredLines = 0;
  size_t NumFunctionsExecuted = 0, NumFunctions = 0;
  for (const auto &File : FileSummaries) {
    NumRegions += File.RegionCoverage.NumRegions;
    CoveredRegions += File.RegionCoverage.Covered;

    NumLines += File.LineCoverage.NumLines;
    NonCodeLines += File.LineCoverage.NonCodeLines;
    CoveredLines += File.LineCoverage.Covered;

    NumFunctionsExecuted += File.FunctionCoverage.Executed;
    NumFunctions += File.FunctionCoverage.NumFunctions;
  }
  return FileCoverageSummary(
      "TOTAL", RegionCoverageInfo(CoveredRegions, NumRegions),
      LineCoverageInfo(CoveredLines, NonCodeLines, NumLines),
      FunctionCoverageInfo(NumFunctionsExecuted, NumFunctions),
      None);
}
