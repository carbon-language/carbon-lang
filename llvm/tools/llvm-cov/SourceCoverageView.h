//===- SourceCoverageView.h - Code coverage view for source code ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements rendering for code coverage of source code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_COV_SOURCECOVERAGEVIEW_H
#define LLVM_COV_SOURCECOVERAGEVIEW_H

#include "CoverageViewOptions.h"
#include "SourceCoverageDataManager.h"
#include "llvm/ProfileData/CoverageMapping.h"
#include "llvm/Support/MemoryBuffer.h"
#include <vector>

namespace llvm {

/// \brief A code coverage view of a specific source file.
/// It can have embedded coverage views.
class SourceCoverageView {
public:
  enum SubViewKind { View, ExpansionView, InstantiationView };

  /// \brief Coverage information for a single line.
  struct LineCoverageInfo {
    uint64_t ExecutionCount;
    unsigned RegionCount;
    bool Mapped;

    LineCoverageInfo() : ExecutionCount(0), RegionCount(0), Mapped(false) {}

    bool isMapped() const { return Mapped; }

    bool hasMultipleRegions() const { return RegionCount > 1; }

    void addRegionStartCount(uint64_t Count) {
      Mapped = true;
      ExecutionCount = Count;
      ++RegionCount;
    }

    void addRegionCount(uint64_t Count) {
      Mapped = true;
      ExecutionCount = Count;
    }
  };

  /// \brief A marker that points at the start
  /// of a specific mapping region.
  struct RegionMarker {
    unsigned Line, Column;
    uint64_t ExecutionCount;

    RegionMarker(unsigned Line, unsigned Column, uint64_t Value)
        : Line(Line), Column(Column), ExecutionCount(Value) {}
  };

  /// \brief A single line source range used to
  /// render highlighted text.
  struct HighlightRange {
    enum HighlightKind {
      /// The code that wasn't executed.
      NotCovered,

      /// The region of code that was expanded.
      Expanded
    };
    HighlightKind Kind;
    unsigned Line;
    unsigned ColumnStart;
    unsigned ColumnEnd;

    HighlightRange(unsigned Line, unsigned ColumnStart, unsigned ColumnEnd,
                   HighlightKind Kind = NotCovered)
        : Kind(Kind), Line(Line), ColumnStart(ColumnStart),
          ColumnEnd(ColumnEnd) {}

    bool operator<(const HighlightRange &Other) const {
      if (Line == Other.Line)
        return ColumnStart < Other.ColumnStart;
      return Line < Other.Line;
    }

    bool columnStartOverlaps(const HighlightRange &Other) const {
      return ColumnStart <= Other.ColumnStart && ColumnEnd > Other.ColumnStart;
    }
    bool columnEndOverlaps(const HighlightRange &Other) const {
      return ColumnEnd >= Other.ColumnEnd && ColumnStart < Other.ColumnEnd;
    }
    bool contains(const HighlightRange &Other) const {
      if (Line != Other.Line)
        return false;
      return ColumnStart <= Other.ColumnStart && ColumnEnd >= Other.ColumnEnd;
    }

    bool overlaps(const HighlightRange &Other) const {
      if (Line != Other.Line)
        return false;
      return columnStartOverlaps(Other) || columnEndOverlaps(Other);
    }
  };

private:
  const MemoryBuffer &File;
  const CoverageViewOptions &Options;
  unsigned LineOffset;
  SubViewKind Kind;
  coverage::CounterMappingRegion ExpansionRegion;
  std::vector<std::unique_ptr<SourceCoverageView>> Children;
  std::vector<LineCoverageInfo> LineStats;
  std::vector<HighlightRange> HighlightRanges;
  std::vector<RegionMarker> Markers;
  StringRef FunctionName;

  /// \brief Create the line coverage information using the coverage data.
  void createLineCoverageInfo(SourceCoverageDataManager &Data);

  /// \brief Create the line highlighting ranges using the coverage data.
  void createHighlightRanges(SourceCoverageDataManager &Data);

  /// \brief Create the region markers using the coverage data.
  void createRegionMarkers(SourceCoverageDataManager &Data);

  /// \brief Sort children by the starting location.
  void sortChildren();

  /// \brief Return a highlight range for the expansion region of this view.
  HighlightRange getExpansionHighlightRange() const;

  /// \brief Render a source line with highlighting.
  void renderLine(raw_ostream &OS, StringRef Line,
                  ArrayRef<HighlightRange> Ranges);

  void renderOffset(raw_ostream &OS, unsigned I);

  void renderViewDivider(unsigned Offset, unsigned Length, raw_ostream &OS);

  /// \brief Render the line's execution count column.
  void renderLineCoverageColumn(raw_ostream &OS, const LineCoverageInfo &Line);

  /// \brief Render the line number column.
  void renderLineNumberColumn(raw_ostream &OS, unsigned LineNo);

  /// \brief Render all the region's execution counts on a line.
  void renderRegionMarkers(raw_ostream &OS, ArrayRef<RegionMarker> Regions);

  static const unsigned LineCoverageColumnWidth = 7;
  static const unsigned LineNumberColumnWidth = 5;

public:
  SourceCoverageView(const MemoryBuffer &File,
                     const CoverageViewOptions &Options)
      : File(File), Options(Options), LineOffset(0), Kind(View),
        ExpansionRegion(coverage::Counter(), 0, 0, 0, 0, 0) {}

  SourceCoverageView(SourceCoverageView &Parent, StringRef FunctionName)
      : File(Parent.File), Options(Parent.Options), LineOffset(0),
        Kind(InstantiationView),
        ExpansionRegion(coverage::Counter(), 0, 0, 0, 0, 0),
        FunctionName(FunctionName) {}

  SourceCoverageView(const MemoryBuffer &File,
                     const CoverageViewOptions &Options,
                     const coverage::CounterMappingRegion &ExpansionRegion)
      : File(File), Options(Options), LineOffset(0), Kind(ExpansionView),
        ExpansionRegion(ExpansionRegion) {}

  const CoverageViewOptions &getOptions() const { return Options; }

  bool isExpansionSubView() const { return Kind == ExpansionView; }

  bool isInstantiationSubView() const { return Kind == InstantiationView; }

  /// \brief Return the line number after which the subview expansion is shown.
  unsigned getSubViewsExpansionLine() const {
    return ExpansionRegion.LineStart;
  }

  void addChild(std::unique_ptr<SourceCoverageView> View) {
    Children.push_back(std::move(View));
  }

  /// \brief Print the code coverage information for a specific
  /// portion of a source file to the output stream.
  void render(raw_ostream &OS, unsigned Offset = 0);

  /// \brief Load the coverage information required for rendering
  /// from the mapping regions in the data manager.
  void load(SourceCoverageDataManager &Data);
};

} // namespace llvm

#endif // LLVM_COV_SOURCECOVERAGEVIEW_H
