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

#include "FunctionCoverageMapping.h"
#include "llvm/ProfileData/CoverageMapping.h"
#include "llvm/ADT/Hashing.h"
#include <vector>
#include <unordered_map>

namespace llvm {

/// \brief Partions mapping regions by their kind and sums
/// the execution counts of the regions that start at the same location.
class SourceCoverageDataManager {
public:
  struct SourceRange {
    unsigned LineStart, ColumnStart, LineEnd, ColumnEnd;

    SourceRange(unsigned LineStart, unsigned ColumnStart, unsigned LineEnd,
                unsigned ColumnEnd)
        : LineStart(LineStart), ColumnStart(ColumnStart), LineEnd(LineEnd),
          ColumnEnd(ColumnEnd) {}

    bool operator==(const SourceRange &Other) const {
      return LineStart == Other.LineStart && ColumnStart == Other.ColumnStart &&
             LineEnd == Other.LineEnd && ColumnEnd == Other.ColumnEnd;
    }

    bool operator<(const SourceRange &Other) const {
      if (LineStart == Other.LineStart)
        return ColumnStart < Other.ColumnStart;
      return LineStart < Other.LineStart;
    }

    bool contains(const SourceRange &Other) {
      if (LineStart > Other.LineStart ||
          (LineStart == Other.LineStart && ColumnStart > Other.ColumnStart))
        return false;
      if (LineEnd < Other.LineEnd ||
          (LineEnd == Other.LineEnd && ColumnEnd < Other.ColumnEnd))
        return false;
      return true;
    }
  };

protected:
  std::vector<std::pair<SourceRange, uint64_t>> Regions;
  std::vector<SourceRange> SkippedRegions;
  bool Uniqued;

public:
  SourceCoverageDataManager() : Uniqued(false) {}

  void insert(const coverage::CountedRegion &CR);

  /// \brief Return the source ranges and execution counts
  /// obtained from the non-skipped mapping regions.
  ArrayRef<std::pair<SourceRange, uint64_t>> getSourceRegions();

  /// \brief Return the source ranges obtained from the skipped mapping regions.
  ArrayRef<SourceRange> getSkippedRegions() const { return SkippedRegions; }
};

} // namespace llvm

#endif // LLVM_COV_SOURCECOVERAGEDATAMANAGER_H
