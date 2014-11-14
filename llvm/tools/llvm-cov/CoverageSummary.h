//===- CoverageSummary.h - Code coverage summary --------------------------===//
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

#ifndef LLVM_COV_COVERAGESUMMARY_H
#define LLVM_COV_COVERAGESUMMARY_H

#include "CoverageSummaryInfo.h"
#include <vector>

namespace llvm {

/// \brief Manager for the function and file code coverage summaries.
class CoverageSummary {
  std::vector<StringRef> Filenames;
  std::vector<FunctionCoverageSummary> FunctionSummaries;
  std::vector<std::pair<unsigned, unsigned>> FunctionSummariesFileIDs;
  std::vector<FileCoverageSummary> FileSummaries;

  unsigned getFileID(StringRef Filename);

public:
  void createSummaries(const coverage::CoverageMapping &Coverage);

  ArrayRef<FileCoverageSummary> getFileSummaries() { return FileSummaries; }

  FileCoverageSummary getCombinedFileSummaries();

  void render(const FunctionCoverageSummary &Summary, raw_ostream &OS);

  void render(raw_ostream &OS);
};
}

#endif // LLVM_COV_COVERAGESUMMARY_H
