//===- SourceCoverageView.cpp - Code coverage view for source code --------===//
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

#include "SourceCoverageView.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/LineIterator.h"

using namespace llvm;

void SourceCoverageView::renderLine(raw_ostream &OS, StringRef Line,
                                    ArrayRef<HighlightRange> Ranges) {
  if (Ranges.empty()) {
    OS << Line << "\n";
    return;
  }
  if (Line.empty())
    Line = " ";

  unsigned PrevColumnStart = 0;
  unsigned Start = 1;
  for (const auto &Range : Ranges) {
    if (PrevColumnStart == Range.ColumnStart)
      continue;

    // Show the unhighlighted part
    unsigned ColumnStart = PrevColumnStart = Range.ColumnStart;
    OS << Line.substr(Start - 1, ColumnStart - Start);

    // Show the highlighted part
    auto Color = Range.Kind == HighlightRange::NotCovered ? raw_ostream::RED
                                                          : raw_ostream::CYAN;
    OS.changeColor(Color, false, true);
    unsigned ColumnEnd = std::min(Range.ColumnEnd, (unsigned)Line.size() + 1);
    OS << Line.substr(ColumnStart - 1, ColumnEnd - ColumnStart);
    Start = ColumnEnd;
    OS.resetColor();
  }

  // Show the rest of the line
  OS << Line.substr(Start - 1, Line.size() - Start + 1);
  OS << "\n";
}

void SourceCoverageView::renderOffset(raw_ostream &OS, unsigned I) {
  for (unsigned J = 0; J < I; ++J)
    OS << "  |";
}

void SourceCoverageView::renderViewDivider(unsigned Offset, unsigned Length,
                                           raw_ostream &OS) {
  for (unsigned J = 1; J < Offset; ++J)
    OS << "  |";
  if (Offset != 0)
    OS.indent(2);
  for (unsigned I = 0; I < Length; ++I)
    OS << "-";
}

void
SourceCoverageView::renderLineCoverageColumn(raw_ostream &OS,
                                             const LineCoverageInfo &Line) {
  if (!Line.isMapped()) {
    OS.indent(LineCoverageColumnWidth) << '|';
    return;
  }
  SmallString<32> Buffer;
  raw_svector_ostream BufferOS(Buffer);
  BufferOS << Line.ExecutionCount;
  auto Str = BufferOS.str();
  // Trim
  Str = Str.substr(0, std::min(Str.size(), (size_t)LineCoverageColumnWidth));
  // Align to the right
  OS.indent(LineCoverageColumnWidth - Str.size());
  colored_ostream(OS, raw_ostream::MAGENTA,
                  Line.hasMultipleRegions() && Options.Colors)
      << Str;
  OS << '|';
}

void SourceCoverageView::renderLineNumberColumn(raw_ostream &OS,
                                                unsigned LineNo) {
  SmallString<32> Buffer;
  raw_svector_ostream BufferOS(Buffer);
  BufferOS << LineNo;
  auto Str = BufferOS.str();
  // Trim and align to the right
  Str = Str.substr(0, std::min(Str.size(), (size_t)LineNumberColumnWidth));
  OS.indent(LineNumberColumnWidth - Str.size()) << Str << '|';
}

void SourceCoverageView::renderRegionMarkers(raw_ostream &OS,
                                             ArrayRef<RegionMarker> Regions) {
  SmallString<32> Buffer;
  raw_svector_ostream BufferOS(Buffer);

  unsigned PrevColumn = 1;
  for (const auto &Region : Regions) {
    // Skip to the new region
    if (Region.Column > PrevColumn)
      OS.indent(Region.Column - PrevColumn);
    PrevColumn = Region.Column + 1;
    BufferOS << Region.ExecutionCount;
    StringRef Str = BufferOS.str();
    // Trim the execution count
    Str = Str.substr(0, std::min(Str.size(), (size_t)7));
    PrevColumn += Str.size();
    OS << '^' << Str;
    Buffer.clear();
  }
  OS << "\n";
}

/// \brief Insert a new highlighting range into the line's highlighting ranges
/// Return line's new highlighting ranges in result.
static void insertHighlightRange(
    ArrayRef<SourceCoverageView::HighlightRange> Ranges,
    SourceCoverageView::HighlightRange RangeToInsert,
    SmallVectorImpl<SourceCoverageView::HighlightRange> &Result) {
  Result.clear();
  size_t I = 0;
  auto E = Ranges.size();
  for (; I < E; ++I) {
    if (RangeToInsert.ColumnStart < Ranges[I].ColumnEnd) {
      const auto &Range = Ranges[I];
      bool NextRangeContainsInserted = false;
      // If the next range starts before the inserted range, move the end of the
      // next range to the start of the inserted range.
      if (Range.ColumnStart < RangeToInsert.ColumnStart) {
        if (RangeToInsert.ColumnStart != Range.ColumnStart)
          Result.push_back(SourceCoverageView::HighlightRange(
              Range.Line, Range.ColumnStart, RangeToInsert.ColumnStart,
              Range.Kind));
        // If the next range also ends after the inserted range, keep this range
        // and create a new range that starts at the inserted range and ends
        // at the next range later.
        if (Range.ColumnEnd > RangeToInsert.ColumnEnd)
          NextRangeContainsInserted = true;
      }
      if (!NextRangeContainsInserted) {
        ++I;
        // Ignore ranges that are contained in inserted range
        while (I < E && RangeToInsert.contains(Ranges[I]))
          ++I;
      }
      break;
    }
    Result.push_back(Ranges[I]);
  }
  Result.push_back(RangeToInsert);
  // If the next range starts before the inserted range end, move the start
  // of the next range to the end of the inserted range.
  if (I < E && Ranges[I].ColumnStart < RangeToInsert.ColumnEnd) {
    const auto &Range = Ranges[I];
    if (RangeToInsert.ColumnEnd != Range.ColumnEnd)
      Result.push_back(SourceCoverageView::HighlightRange(
          Range.Line, RangeToInsert.ColumnEnd, Range.ColumnEnd, Range.Kind));
    ++I;
  }
  // Add the remaining ranges that are located after the inserted range
  for (; I < E; ++I)
    Result.push_back(Ranges[I]);
}

void SourceCoverageView::sortChildren() {
  for (auto &I : Children)
    I->sortChildren();
  std::sort(Children.begin(), Children.end(),
            [](const std::unique_ptr<SourceCoverageView> &LHS,
               const std::unique_ptr<SourceCoverageView> &RHS) {
    return LHS->ExpansionRegion < RHS->ExpansionRegion;
  });
}

SourceCoverageView::HighlightRange
SourceCoverageView::getExpansionHighlightRange() const {
  return HighlightRange(ExpansionRegion.LineStart, ExpansionRegion.ColumnStart,
                        ExpansionRegion.ColumnEnd, HighlightRange::Expanded);
}

template <typename T>
ArrayRef<T> gatherLineItems(size_t &CurrentIdx, const std::vector<T> &Items,
                            unsigned LineNo) {
  auto PrevIdx = CurrentIdx;
  auto E = Items.size();
  while (CurrentIdx < E && Items[CurrentIdx].Line == LineNo)
    ++CurrentIdx;
  return ArrayRef<T>(Items.data() + PrevIdx, CurrentIdx - PrevIdx);
}

ArrayRef<std::unique_ptr<SourceCoverageView>>
gatherLineSubViews(size_t &CurrentIdx,
                   ArrayRef<std::unique_ptr<SourceCoverageView>> Items,
                   unsigned LineNo) {
  auto PrevIdx = CurrentIdx;
  auto E = Items.size();
  while (CurrentIdx < E &&
         Items[CurrentIdx]->getSubViewsExpansionLine() == LineNo)
    ++CurrentIdx;
  return ArrayRef<std::unique_ptr<SourceCoverageView>>(Items.data() + PrevIdx,
                                                       CurrentIdx - PrevIdx);
}

void SourceCoverageView::render(raw_ostream &OS, unsigned Offset) {
  // Make sure that the children are in sorted order.
  sortChildren();

  SmallVector<HighlightRange, 8> AdjustedLineHighlightRanges;
  size_t CurrentChild = 0;
  size_t CurrentHighlightRange = 0;
  size_t CurrentRegionMarker = 0;

  line_iterator Lines(File);
  // Advance the line iterator to the first line.
  while (Lines.line_number() < LineStart)
    ++Lines;

  // The width of the leading columns
  unsigned CombinedColumnWidth =
      (Options.ShowLineStats ? LineCoverageColumnWidth + 1 : 0) +
      (Options.ShowLineNumbers ? LineNumberColumnWidth + 1 : 0);
  // The width of the line that is used to divide between the view and the
  // subviews.
  unsigned DividerWidth = CombinedColumnWidth + 4;

  for (size_t I = 0; I < LineCount; ++I) {
    unsigned LineNo = I + LineStart;

    // Gather the child subviews that are visible on this line.
    auto LineSubViews = gatherLineSubViews(CurrentChild, Children, LineNo);

    renderOffset(OS, Offset);
    if (Options.ShowLineStats)
      renderLineCoverageColumn(OS, LineStats[I]);
    if (Options.ShowLineNumbers)
      renderLineNumberColumn(OS, LineNo);

    // Gather highlighting ranges.
    auto LineHighlightRanges =
        gatherLineItems(CurrentHighlightRange, HighlightRanges, LineNo);
    auto LineRanges = LineHighlightRanges;
    // Highlight the expansion range if there is an expansion subview on this
    // line.
    if (!LineSubViews.empty() && LineSubViews.front()->isExpansionSubView() &&
        Options.Colors) {
      insertHighlightRange(LineHighlightRanges,
                           LineSubViews.front()->getExpansionHighlightRange(),
                           AdjustedLineHighlightRanges);
      LineRanges = AdjustedLineHighlightRanges;
    }

    // Display the source code for the current line.
    StringRef Line = *Lines;
    // Check if the line is empty, as line_iterator skips blank lines.
    if (LineNo < Lines.line_number())
      Line = "";
    else if (!Lines.is_at_eof())
      ++Lines;
    renderLine(OS, Line, LineRanges);

    // Show the region markers.
    bool ShowMarkers = !Options.ShowLineStatsOrRegionMarkers ||
                       LineStats[I].hasMultipleRegions();
    auto LineMarkers = gatherLineItems(CurrentRegionMarker, Markers, LineNo);
    if (ShowMarkers && !LineMarkers.empty()) {
      renderOffset(OS, Offset);
      OS.indent(CombinedColumnWidth);
      renderRegionMarkers(OS, LineMarkers);
    }

    // Show the line's expanded child subviews.
    bool FirstChildExpansion = true;
    if (LineSubViews.empty())
      continue;
    unsigned NewOffset = Offset + 1;
    renderViewDivider(NewOffset, DividerWidth, OS);
    OS << "\n";
    for (const auto &Child : LineSubViews) {
      // If this subview shows a function instantiation, render the function's
      // name.
      if (Child->isInstantiationSubView()) {
        renderOffset(OS, NewOffset);
        OS << ' ';
        Options.colored_ostream(OS, raw_ostream::CYAN) << Child->FunctionName
                                                       << ":";
        OS << "\n";
      } else {
        if (!FirstChildExpansion) {
          // Re-render the current line and highlight the expansion range for
          // this
          // subview.
          insertHighlightRange(LineHighlightRanges,
                               Child->getExpansionHighlightRange(),
                               AdjustedLineHighlightRanges);
          renderOffset(OS, Offset);
          OS.indent(CombinedColumnWidth + (Offset == 0 ? 0 : 1));
          renderLine(OS, Line, AdjustedLineHighlightRanges);
          renderViewDivider(NewOffset, DividerWidth, OS);
          OS << "\n";
        } else
          FirstChildExpansion = false;
      }
      // Render the child subview
      Child->render(OS, NewOffset);
      renderViewDivider(NewOffset, DividerWidth, OS);
      OS << "\n";
    }
  }
}

void
SourceCoverageView::createLineCoverageInfo(SourceCoverageDataManager &Data) {
  LineStats.resize(LineCount);
  for (const auto &Region : Data.getSourceRegions()) {
    auto Value = Region.second;
    LineStats[Region.first.LineStart - LineStart].addRegionStartCount(Value);
    for (unsigned Line = Region.first.LineStart + 1;
         Line <= Region.first.LineEnd; ++Line)
      LineStats[Line - LineStart].addRegionCount(Value);
  }

  // Reset the line stats for skipped regions.
  for (const auto &Region : Data.getSkippedRegions()) {
    for (unsigned Line = Region.LineStart; Line <= Region.LineEnd; ++Line)
      LineStats[Line - LineStart] = LineCoverageInfo();
  }
}

void
SourceCoverageView::createHighlightRanges(SourceCoverageDataManager &Data) {
  auto Regions = Data.getSourceRegions();
  std::vector<bool> AlreadyHighlighted;
  AlreadyHighlighted.resize(Regions.size(), false);

  for (size_t I = 0, S = Regions.size(); I < S; ++I) {
    const auto &Region = Regions[I];
    auto Value = Region.second;
    auto SrcRange = Region.first;
    if (Value != 0)
      continue;
    if (AlreadyHighlighted[I])
      continue;
    for (size_t J = 0; J < S; ++J) {
      if (SrcRange.contains(Regions[J].first)) {
        AlreadyHighlighted[J] = true;
      }
    }
    if (SrcRange.LineStart == SrcRange.LineEnd) {
      HighlightRanges.push_back(HighlightRange(
          SrcRange.LineStart, SrcRange.ColumnStart, SrcRange.ColumnEnd));
      continue;
    }
    HighlightRanges.push_back(
        HighlightRange(SrcRange.LineStart, SrcRange.ColumnStart,
                       std::numeric_limits<unsigned>::max()));
    HighlightRanges.push_back(
        HighlightRange(SrcRange.LineEnd, 1, SrcRange.ColumnEnd));
    for (unsigned Line = SrcRange.LineStart + 1; Line < SrcRange.LineEnd;
         ++Line) {
      HighlightRanges.push_back(
          HighlightRange(Line, 1, std::numeric_limits<unsigned>::max()));
    }
  }

  std::sort(HighlightRanges.begin(), HighlightRanges.end());

  if (Options.Debug) {
    for (const auto &Range : HighlightRanges) {
      outs() << "Highlighted line " << Range.Line << ", " << Range.ColumnStart
             << " -> ";
      if (Range.ColumnEnd == std::numeric_limits<unsigned>::max()) {
        outs() << "?\n";
      } else {
        outs() << Range.ColumnEnd << "\n";
      }
    }
  }
}

void SourceCoverageView::createRegionMarkers(SourceCoverageDataManager &Data) {
  for (const auto &Region : Data.getSourceRegions()) {
    if (Region.first.LineStart >= LineStart)
      Markers.push_back(RegionMarker(Region.first.LineStart,
                                     Region.first.ColumnStart, Region.second));
  }

  if (Options.Debug) {
    for (const auto &Marker : Markers) {
      outs() << "Marker at " << Marker.Line << ":" << Marker.Column << " = "
             << Marker.ExecutionCount << "\n";
    }
  }
}

void SourceCoverageView::load(SourceCoverageDataManager &Data) {
  if (Options.ShowLineStats)
    createLineCoverageInfo(Data);
  if (Options.Colors)
    createHighlightRanges(Data);
  if (Options.ShowRegionMarkers)
    createRegionMarkers(Data);
}
