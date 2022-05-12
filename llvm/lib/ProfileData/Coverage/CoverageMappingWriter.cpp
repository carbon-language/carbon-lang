//===- CoverageMappingWriter.cpp - Code coverage mapping writer -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing coverage mapping data for
// instrumentation based coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/Coverage/CoverageMappingWriter.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>

using namespace llvm;
using namespace coverage;

CoverageFilenamesSectionWriter::CoverageFilenamesSectionWriter(
    ArrayRef<std::string> Filenames)
    : Filenames(Filenames) {
#ifndef NDEBUG
  StringSet<> NameSet;
  for (StringRef Name : Filenames)
    assert(NameSet.insert(Name).second && "Duplicate filename");
#endif
}

void CoverageFilenamesSectionWriter::write(raw_ostream &OS, bool Compress) {
  std::string FilenamesStr;
  {
    raw_string_ostream FilenamesOS{FilenamesStr};
    for (const auto &Filename : Filenames) {
      encodeULEB128(Filename.size(), FilenamesOS);
      FilenamesOS << Filename;
    }
  }

  SmallString<128> CompressedStr;
  bool doCompression =
      Compress && zlib::isAvailable() && DoInstrProfNameCompression;
  if (doCompression) {
    auto E =
        zlib::compress(FilenamesStr, CompressedStr, zlib::BestSizeCompression);
    if (E)
      report_bad_alloc_error("Failed to zlib compress coverage data");
  }

  // ::= <num-filenames>
  //     <uncompressed-len>
  //     <compressed-len-or-zero>
  //     (<compressed-filenames> | <uncompressed-filenames>)
  encodeULEB128(Filenames.size(), OS);
  encodeULEB128(FilenamesStr.size(), OS);
  encodeULEB128(doCompression ? CompressedStr.size() : 0U, OS);
  OS << (doCompression ? CompressedStr.str() : StringRef(FilenamesStr));
}

namespace {

/// Gather only the expressions that are used by the mapping
/// regions in this function.
class CounterExpressionsMinimizer {
  ArrayRef<CounterExpression> Expressions;
  SmallVector<CounterExpression, 16> UsedExpressions;
  std::vector<unsigned> AdjustedExpressionIDs;

public:
  CounterExpressionsMinimizer(ArrayRef<CounterExpression> Expressions,
                              ArrayRef<CounterMappingRegion> MappingRegions)
      : Expressions(Expressions) {
    AdjustedExpressionIDs.resize(Expressions.size(), 0);
    for (const auto &I : MappingRegions) {
      mark(I.Count);
      mark(I.FalseCount);
    }
    for (const auto &I : MappingRegions) {
      gatherUsed(I.Count);
      gatherUsed(I.FalseCount);
    }
  }

  void mark(Counter C) {
    if (!C.isExpression())
      return;
    unsigned ID = C.getExpressionID();
    AdjustedExpressionIDs[ID] = 1;
    mark(Expressions[ID].LHS);
    mark(Expressions[ID].RHS);
  }

  void gatherUsed(Counter C) {
    if (!C.isExpression() || !AdjustedExpressionIDs[C.getExpressionID()])
      return;
    AdjustedExpressionIDs[C.getExpressionID()] = UsedExpressions.size();
    const auto &E = Expressions[C.getExpressionID()];
    UsedExpressions.push_back(E);
    gatherUsed(E.LHS);
    gatherUsed(E.RHS);
  }

  ArrayRef<CounterExpression> getExpressions() const { return UsedExpressions; }

  /// Adjust the given counter to correctly transition from the old
  /// expression ids to the new expression ids.
  Counter adjust(Counter C) const {
    if (C.isExpression())
      C = Counter::getExpression(AdjustedExpressionIDs[C.getExpressionID()]);
    return C;
  }
};

} // end anonymous namespace

/// Encode the counter.
///
/// The encoding uses the following format:
/// Low 2 bits - Tag:
///   Counter::Zero(0) - A Counter with kind Counter::Zero
///   Counter::CounterValueReference(1) - A counter with kind
///     Counter::CounterValueReference
///   Counter::Expression(2) + CounterExpression::Subtract(0) -
///     A counter with kind Counter::Expression and an expression
///     with kind CounterExpression::Subtract
///   Counter::Expression(2) + CounterExpression::Add(1) -
///     A counter with kind Counter::Expression and an expression
///     with kind CounterExpression::Add
/// Remaining bits - Counter/Expression ID.
static unsigned encodeCounter(ArrayRef<CounterExpression> Expressions,
                              Counter C) {
  unsigned Tag = unsigned(C.getKind());
  if (C.isExpression())
    Tag += Expressions[C.getExpressionID()].Kind;
  unsigned ID = C.getCounterID();
  assert(ID <=
         (std::numeric_limits<unsigned>::max() >> Counter::EncodingTagBits));
  return Tag | (ID << Counter::EncodingTagBits);
}

static void writeCounter(ArrayRef<CounterExpression> Expressions, Counter C,
                         raw_ostream &OS) {
  encodeULEB128(encodeCounter(Expressions, C), OS);
}

void CoverageMappingWriter::write(raw_ostream &OS) {
  // Check that we don't have any bogus regions.
  assert(all_of(MappingRegions,
                [](const CounterMappingRegion &CMR) {
                  return CMR.startLoc() <= CMR.endLoc();
                }) &&
         "Source region does not begin before it ends");

  // Sort the regions in an ascending order by the file id and the starting
  // location. Sort by region kinds to ensure stable order for tests.
  llvm::stable_sort(MappingRegions, [](const CounterMappingRegion &LHS,
                                       const CounterMappingRegion &RHS) {
    if (LHS.FileID != RHS.FileID)
      return LHS.FileID < RHS.FileID;
    if (LHS.startLoc() != RHS.startLoc())
      return LHS.startLoc() < RHS.startLoc();
    return LHS.Kind < RHS.Kind;
  });

  // Write out the fileid -> filename mapping.
  encodeULEB128(VirtualFileMapping.size(), OS);
  for (const auto &FileID : VirtualFileMapping)
    encodeULEB128(FileID, OS);

  // Write out the expressions.
  CounterExpressionsMinimizer Minimizer(Expressions, MappingRegions);
  auto MinExpressions = Minimizer.getExpressions();
  encodeULEB128(MinExpressions.size(), OS);
  for (const auto &E : MinExpressions) {
    writeCounter(MinExpressions, Minimizer.adjust(E.LHS), OS);
    writeCounter(MinExpressions, Minimizer.adjust(E.RHS), OS);
  }

  // Write out the mapping regions.
  // Split the regions into subarrays where each region in a
  // subarray has a fileID which is the index of that subarray.
  unsigned PrevLineStart = 0;
  unsigned CurrentFileID = ~0U;
  for (auto I = MappingRegions.begin(), E = MappingRegions.end(); I != E; ++I) {
    if (I->FileID != CurrentFileID) {
      // Ensure that all file ids have at least one mapping region.
      assert(I->FileID == (CurrentFileID + 1));
      // Find the number of regions with this file id.
      unsigned RegionCount = 1;
      for (auto J = I + 1; J != E && I->FileID == J->FileID; ++J)
        ++RegionCount;
      // Start a new region sub-array.
      encodeULEB128(RegionCount, OS);

      CurrentFileID = I->FileID;
      PrevLineStart = 0;
    }
    Counter Count = Minimizer.adjust(I->Count);
    Counter FalseCount = Minimizer.adjust(I->FalseCount);
    switch (I->Kind) {
    case CounterMappingRegion::CodeRegion:
    case CounterMappingRegion::GapRegion:
      writeCounter(MinExpressions, Count, OS);
      break;
    case CounterMappingRegion::ExpansionRegion: {
      assert(Count.isZero());
      assert(I->ExpandedFileID <=
             (std::numeric_limits<unsigned>::max() >>
              Counter::EncodingCounterTagAndExpansionRegionTagBits));
      // Mark an expansion region with a set bit that follows the counter tag,
      // and pack the expanded file id into the remaining bits.
      unsigned EncodedTagExpandedFileID =
          (1 << Counter::EncodingTagBits) |
          (I->ExpandedFileID
           << Counter::EncodingCounterTagAndExpansionRegionTagBits);
      encodeULEB128(EncodedTagExpandedFileID, OS);
      break;
    }
    case CounterMappingRegion::SkippedRegion:
      assert(Count.isZero());
      encodeULEB128(unsigned(I->Kind)
                        << Counter::EncodingCounterTagAndExpansionRegionTagBits,
                    OS);
      break;
    case CounterMappingRegion::BranchRegion:
      encodeULEB128(unsigned(I->Kind)
                        << Counter::EncodingCounterTagAndExpansionRegionTagBits,
                    OS);
      writeCounter(MinExpressions, Count, OS);
      writeCounter(MinExpressions, FalseCount, OS);
      break;
    }
    assert(I->LineStart >= PrevLineStart);
    encodeULEB128(I->LineStart - PrevLineStart, OS);
    encodeULEB128(I->ColumnStart, OS);
    assert(I->LineEnd >= I->LineStart);
    encodeULEB128(I->LineEnd - I->LineStart, OS);
    encodeULEB128(I->ColumnEnd, OS);
    PrevLineStart = I->LineStart;
  }
  // Ensure that all file ids have at least one mapping region.
  assert(CurrentFileID == (VirtualFileMapping.size() - 1));
}
