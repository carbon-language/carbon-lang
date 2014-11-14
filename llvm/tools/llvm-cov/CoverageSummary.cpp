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
CoverageSummary::createSummaries(const coverage::CoverageMapping &Coverage) {
  for (StringRef Filename : Coverage.getUniqueSourceFiles()) {
    size_t PrevSize = FunctionSummaries.size();
    for (const auto &F : Coverage.getCoveredFunctions(Filename))
      FunctionSummaries.push_back(FunctionCoverageSummary::get(F));
    size_t Count = FunctionSummaries.size() - PrevSize;
    if (Count == 0)
      continue;
    FileSummaries.push_back(FileCoverageSummary::get(
        Filename, makeArrayRef(FunctionSummaries.data() + PrevSize, Count)));
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
