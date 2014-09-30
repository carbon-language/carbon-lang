//=-- CoverageMapping.cpp - Code coverage mapping support ---------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for clang's and llvm's instrumentation based
// code coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/CoverageMapping.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ProfileData/CoverageMappingReader.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace coverage;

#define DEBUG_TYPE "coverage-mapping"

CounterExpressionBuilder::CounterExpressionBuilder(unsigned NumCounterValues) {
  Terms.resize(NumCounterValues);
}

Counter CounterExpressionBuilder::get(const CounterExpression &E) {
  for (unsigned I = 0, S = Expressions.size(); I < S; ++I) {
    if (Expressions[I] == E)
      return Counter::getExpression(I);
  }
  Expressions.push_back(E);
  return Counter::getExpression(Expressions.size() - 1);
}

void CounterExpressionBuilder::extractTerms(Counter C, int Sign) {
  switch (C.getKind()) {
  case Counter::Zero:
    break;
  case Counter::CounterValueReference:
    Terms[C.getCounterID()] += Sign;
    break;
  case Counter::Expression:
    const auto &E = Expressions[C.getExpressionID()];
    extractTerms(E.LHS, Sign);
    extractTerms(E.RHS, E.Kind == CounterExpression::Subtract ? -Sign : Sign);
    break;
  }
}

Counter CounterExpressionBuilder::simplify(Counter ExpressionTree) {
  // Gather constant terms.
  for (auto &I : Terms)
    I = 0;
  extractTerms(ExpressionTree);

  Counter C;
  // Create additions.
  // Note: the additions are created first
  // to avoid creation of a tree like ((0 - X) + Y) instead of (Y - X).
  for (unsigned I = 0, S = Terms.size(); I < S; ++I) {
    if (Terms[I] <= 0)
      continue;
    for (int J = 0; J < Terms[I]; ++J) {
      if (C.isZero())
        C = Counter::getCounter(I);
      else
        C = get(CounterExpression(CounterExpression::Add, C,
                                  Counter::getCounter(I)));
    }
  }

  // Create subtractions.
  for (unsigned I = 0, S = Terms.size(); I < S; ++I) {
    if (Terms[I] >= 0)
      continue;
    for (int J = 0; J < (-Terms[I]); ++J)
      C = get(CounterExpression(CounterExpression::Subtract, C,
                                Counter::getCounter(I)));
  }
  return C;
}

Counter CounterExpressionBuilder::add(Counter LHS, Counter RHS) {
  return simplify(get(CounterExpression(CounterExpression::Add, LHS, RHS)));
}

Counter CounterExpressionBuilder::subtract(Counter LHS, Counter RHS) {
  return simplify(
      get(CounterExpression(CounterExpression::Subtract, LHS, RHS)));
}

void CounterMappingContext::dump(const Counter &C,
                                 llvm::raw_ostream &OS) const {
  switch (C.getKind()) {
  case Counter::Zero:
    OS << '0';
    return;
  case Counter::CounterValueReference:
    OS << '#' << C.getCounterID();
    break;
  case Counter::Expression: {
    if (C.getExpressionID() >= Expressions.size())
      return;
    const auto &E = Expressions[C.getExpressionID()];
    OS << '(';
    dump(E.LHS, OS);
    OS << (E.Kind == CounterExpression::Subtract ? " - " : " + ");
    dump(E.RHS, OS);
    OS << ')';
    break;
  }
  }
  if (CounterValues.empty())
    return;
  ErrorOr<int64_t> Value = evaluate(C);
  if (!Value)
    return;
  OS << '[' << *Value << ']';
}

ErrorOr<int64_t> CounterMappingContext::evaluate(const Counter &C) const {
  switch (C.getKind()) {
  case Counter::Zero:
    return 0;
  case Counter::CounterValueReference:
    if (C.getCounterID() >= CounterValues.size())
      return std::make_error_code(std::errc::argument_out_of_domain);
    return CounterValues[C.getCounterID()];
  case Counter::Expression: {
    if (C.getExpressionID() >= Expressions.size())
      return std::make_error_code(std::errc::argument_out_of_domain);
    const auto &E = Expressions[C.getExpressionID()];
    ErrorOr<int64_t> LHS = evaluate(E.LHS);
    if (!LHS)
      return LHS;
    ErrorOr<int64_t> RHS = evaluate(E.RHS);
    if (!RHS)
      return RHS;
    return E.Kind == CounterExpression::Subtract ? *LHS - *RHS : *LHS + *RHS;
  }
  }
  llvm_unreachable("Unhandled CounterKind");
}

ErrorOr<std::unique_ptr<CoverageMapping>>
CoverageMapping::load(ObjectFileCoverageMappingReader &CoverageReader,
                      IndexedInstrProfReader &ProfileReader) {
  auto Coverage = std::unique_ptr<CoverageMapping>(new CoverageMapping());

  std::vector<uint64_t> Counts;
  for (const auto &Record : CoverageReader) {
    Counts.clear();
    if (std::error_code EC = ProfileReader.getFunctionCounts(
            Record.FunctionName, Record.FunctionHash, Counts)) {
      if (EC != instrprof_error::hash_mismatch &&
          EC != instrprof_error::unknown_function)
        return EC;
      Coverage->MismatchedFunctionCount++;
      continue;
    }

    assert(Counts.size() != 0 && "Function's counts are empty");
    FunctionRecord Function(Record.FunctionName, Record.Filenames,
                            Counts.front());
    CounterMappingContext Ctx(Record.Expressions, Counts);
    for (const auto &Region : Record.MappingRegions) {
      ErrorOr<int64_t> ExecutionCount = Ctx.evaluate(Region.Count);
      if (!ExecutionCount)
        break;
      Function.CountedRegions.push_back(CountedRegion(Region, *ExecutionCount));
    }
    if (Function.CountedRegions.size() != Record.MappingRegions.size()) {
      Coverage->MismatchedFunctionCount++;
      continue;
    }

    Coverage->Functions.push_back(Function);
  }

  return std::move(Coverage);
}

ErrorOr<std::unique_ptr<CoverageMapping>>
CoverageMapping::load(StringRef ObjectFilename, StringRef ProfileFilename) {
  auto CounterMappingBuff = MemoryBuffer::getFileOrSTDIN(ObjectFilename);
  if (auto EC = CounterMappingBuff.getError())
    return EC;
  ObjectFileCoverageMappingReader CoverageReader(CounterMappingBuff.get());
  if (auto EC = CoverageReader.readHeader())
    return EC;
  std::unique_ptr<IndexedInstrProfReader> ProfileReader;
  if (auto EC = IndexedInstrProfReader::create(ProfileFilename, ProfileReader))
    return EC;
  return load(CoverageReader, *ProfileReader);
}

namespace {
/// \brief Distributes functions into instantiation sets.
///
/// An instantiation set is a collection of functions that have the same source
/// code, ie, template functions specializations.
class FunctionInstantiationSetCollector {
  typedef DenseMap<std::pair<unsigned, unsigned>,
                   std::vector<const FunctionRecord *>> MapT;
  MapT InstantiatedFunctions;

public:
  void insert(const FunctionRecord &Function, unsigned FileID) {
    auto I = Function.CountedRegions.begin(), E = Function.CountedRegions.end();
    while (I != E && I->FileID != FileID)
      ++I;
    assert(I != E && "function does not cover the given file");
    auto &Functions = InstantiatedFunctions[I->startLoc()];
    Functions.push_back(&Function);
  }

  MapT::iterator begin() { return InstantiatedFunctions.begin(); }

  MapT::iterator end() { return InstantiatedFunctions.end(); }
};

class SegmentBuilder {
  std::vector<CoverageSegment> Segments;
  SmallVector<const CountedRegion *, 8> ActiveRegions;

  /// Start a segment with no count specified.
  void startSegment(unsigned Line, unsigned Col) {
    DEBUG(dbgs() << "Top level segment at " << Line << ":" << Col << "\n");
    Segments.emplace_back(Line, Col, /*IsRegionEntry=*/false);
  }

  /// Start a segment with the given Region's count.
  void startSegment(unsigned Line, unsigned Col, bool IsRegionEntry,
                    const CountedRegion &Region) {
    if (Segments.empty())
      Segments.emplace_back(Line, Col, IsRegionEntry);
    CoverageSegment S = Segments.back();
    // Avoid creating empty regions.
    if (S.Line != Line || S.Col != Col) {
      Segments.emplace_back(Line, Col, IsRegionEntry);
      S = Segments.back();
    }
    DEBUG(dbgs() << "Segment at " << Line << ":" << Col);
    // Set this region's count.
    if (Region.Kind != coverage::CounterMappingRegion::SkippedRegion) {
      DEBUG(dbgs() << " with count " << Region.ExecutionCount);
      Segments.back().setCount(Region.ExecutionCount);
    }
    DEBUG(dbgs() << "\n");
  }

  /// Start a segment for the given region.
  void startSegment(const CountedRegion &Region) {
    startSegment(Region.LineStart, Region.ColumnStart, true, Region);
  }

  /// Pop the top region off of the active stack, starting a new segment with
  /// the containing Region's count.
  void popRegion() {
    const CountedRegion *Active = ActiveRegions.back();
    unsigned Line = Active->LineEnd, Col = Active->ColumnEnd;
    ActiveRegions.pop_back();
    if (ActiveRegions.empty())
      startSegment(Line, Col);
    else
      startSegment(Line, Col, false, *ActiveRegions.back());
  }

public:
  /// Build a list of CoverageSegments from a sorted list of Regions.
  std::vector<CoverageSegment> buildSegments(ArrayRef<CountedRegion> Regions) {
    for (const auto &Region : Regions) {
      // Pop any regions that end before this one starts.
      while (!ActiveRegions.empty() &&
             ActiveRegions.back()->endLoc() <= Region.startLoc())
        popRegion();
      if (Segments.size() && Segments.back().Line == Region.LineStart &&
          Segments.back().Col == Region.ColumnStart) {
        if (Region.Kind != coverage::CounterMappingRegion::SkippedRegion)
          Segments.back().addCount(Region.ExecutionCount);
      } else {
        // Add this region to the stack.
        ActiveRegions.push_back(&Region);
        startSegment(Region);
      }
    }
    // Pop any regions that are left in the stack.
    while (!ActiveRegions.empty())
      popRegion();
    return Segments;
  }
};
}

std::vector<StringRef> CoverageMapping::getUniqueSourceFiles() {
  std::vector<StringRef> Filenames;
  for (const auto &Function : getCoveredFunctions())
    for (const auto &Filename : Function.Filenames)
      Filenames.push_back(Filename);
  std::sort(Filenames.begin(), Filenames.end());
  auto Last = std::unique(Filenames.begin(), Filenames.end());
  Filenames.erase(Last, Filenames.end());
  return Filenames;
}

static Optional<unsigned> findMainViewFileID(StringRef SourceFile,
                                             const FunctionRecord &Function) {
  llvm::SmallVector<bool, 8> IsExpandedFile(Function.Filenames.size(), false);
  llvm::SmallVector<bool, 8> FilenameEquivalence(Function.Filenames.size(),
                                                 false);
  for (unsigned I = 0, E = Function.Filenames.size(); I < E; ++I)
    if (SourceFile == Function.Filenames[I])
      FilenameEquivalence[I] = true;
  for (const auto &CR : Function.CountedRegions)
    if (CR.Kind == CounterMappingRegion::ExpansionRegion &&
        FilenameEquivalence[CR.FileID])
      IsExpandedFile[CR.ExpandedFileID] = true;
  for (unsigned I = 0, E = Function.Filenames.size(); I < E; ++I)
    if (FilenameEquivalence[I] && !IsExpandedFile[I])
      return I;
  return None;
}

static Optional<unsigned> findMainViewFileID(const FunctionRecord &Function) {
  llvm::SmallVector<bool, 8> IsExpandedFile(Function.Filenames.size(), false);
  for (const auto &CR : Function.CountedRegions)
    if (CR.Kind == CounterMappingRegion::ExpansionRegion)
      IsExpandedFile[CR.ExpandedFileID] = true;
  for (unsigned I = 0, E = Function.Filenames.size(); I < E; ++I)
    if (!IsExpandedFile[I])
      return I;
  return None;
}

static SmallSet<unsigned, 8> gatherFileIDs(StringRef SourceFile,
                                           const FunctionRecord &Function) {
  SmallSet<unsigned, 8> IDs;
  for (unsigned I = 0, E = Function.Filenames.size(); I < E; ++I)
    if (SourceFile == Function.Filenames[I])
      IDs.insert(I);
  return IDs;
}

/// Sort a nested sequence of regions from a single file.
template <class It> static void sortNestedRegions(It First, It Last) {
  std::sort(First, Last,
            [](const CountedRegion &LHS, const CountedRegion &RHS) {
    if (LHS.startLoc() == RHS.startLoc())
      // When LHS completely contains RHS, we sort LHS first.
      return RHS.endLoc() < LHS.endLoc();
    return LHS.startLoc() < RHS.startLoc();
  });
}

static bool isExpansion(const CountedRegion &R, unsigned FileID) {
  return R.Kind == CounterMappingRegion::ExpansionRegion && R.FileID == FileID;
}

CoverageData CoverageMapping::getCoverageForFile(StringRef Filename) {
  CoverageData FileCoverage(Filename);
  std::vector<coverage::CountedRegion> Regions;

  for (const auto &Function : Functions) {
    auto MainFileID = findMainViewFileID(Filename, Function);
    if (!MainFileID)
      continue;
    auto FileIDs = gatherFileIDs(Filename, Function);
    for (const auto &CR : Function.CountedRegions)
      if (FileIDs.count(CR.FileID)) {
        Regions.push_back(CR);
        if (isExpansion(CR, *MainFileID))
          FileCoverage.Expansions.emplace_back(CR, Function);
      }
  }

  sortNestedRegions(Regions.begin(), Regions.end());
  FileCoverage.Segments = SegmentBuilder().buildSegments(Regions);

  return FileCoverage;
}

std::vector<const FunctionRecord *>
CoverageMapping::getInstantiations(StringRef Filename) {
  FunctionInstantiationSetCollector InstantiationSetCollector;
  for (const auto &Function : Functions) {
    auto MainFileID = findMainViewFileID(Filename, Function);
    if (!MainFileID)
      continue;
    InstantiationSetCollector.insert(Function, *MainFileID);
  }

  std::vector<const FunctionRecord *> Result;
  for (const auto &InstantiationSet : InstantiationSetCollector) {
    if (InstantiationSet.second.size() < 2)
      continue;
    for (auto Function : InstantiationSet.second)
      Result.push_back(Function);
  }
  return Result;
}

CoverageData
CoverageMapping::getCoverageForFunction(const FunctionRecord &Function) {
  auto MainFileID = findMainViewFileID(Function);
  if (!MainFileID)
    return CoverageData();

  CoverageData FunctionCoverage(Function.Filenames[*MainFileID]);
  std::vector<coverage::CountedRegion> Regions;
  for (const auto &CR : Function.CountedRegions)
    if (CR.FileID == *MainFileID) {
      Regions.push_back(CR);
      if (isExpansion(CR, *MainFileID))
        FunctionCoverage.Expansions.emplace_back(CR, Function);
    }

  sortNestedRegions(Regions.begin(), Regions.end());
  FunctionCoverage.Segments = SegmentBuilder().buildSegments(Regions);

  return FunctionCoverage;
}

CoverageData
CoverageMapping::getCoverageForExpansion(const ExpansionRecord &Expansion) {
  CoverageData ExpansionCoverage(
      Expansion.Function.Filenames[Expansion.FileID]);
  std::vector<coverage::CountedRegion> Regions;
  for (const auto &CR : Expansion.Function.CountedRegions)
    if (CR.FileID == Expansion.FileID) {
      Regions.push_back(CR);
      if (isExpansion(CR, Expansion.FileID))
        ExpansionCoverage.Expansions.emplace_back(CR, Expansion.Function);
    }

  sortNestedRegions(Regions.begin(), Regions.end());
  ExpansionCoverage.Segments = SegmentBuilder().buildSegments(Regions);

  return ExpansionCoverage;
}
