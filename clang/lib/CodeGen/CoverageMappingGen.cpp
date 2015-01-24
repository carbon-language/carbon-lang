//===--- CoverageMappingGen.cpp - Coverage mapping generation ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Instrumentation-based code coverage mapping generator
//
//===----------------------------------------------------------------------===//

#include "CoverageMappingGen.h"
#include "CodeGenFunction.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ProfileData/CoverageMapping.h"
#include "llvm/ProfileData/CoverageMappingReader.h"
#include "llvm/ProfileData/CoverageMappingWriter.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/FileSystem.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm::coverage;

void CoverageSourceInfo::SourceRangeSkipped(SourceRange Range) {
  SkippedRanges.push_back(Range);
}

namespace {

/// \brief A region of source code that can be mapped to a counter.
class SourceMappingRegion {
public:
  enum RegionFlags {
    /// \brief This region won't be emitted if it wasn't extended.
    /// This is useful so that we won't emit source ranges for single tokens
    /// that we don't really care that much about, like:
    ///   the '(' token in #define MACRO (
    IgnoreIfNotExtended = 0x0001,
  };

private:
  FileID File, MacroArgumentFile;

  Counter Count;

  /// \brief A statement that initiated the count of Zero.
  ///
  /// This initiator statement is useful to prevent merging of unreachable
  /// regions with different statements that caused the counter to become
  /// unreachable.
  const Stmt *UnreachableInitiator;

  /// \brief A statement that separates certain mapping regions into groups.
  ///
  /// The group statement is sometimes useful when we are emitting the source
  /// regions not in their correct lexical order, e.g. the regions for the
  /// incrementation expression in the 'for' construct. By marking the regions
  /// in the incrementation expression with the group statement, we avoid the
  /// merging of the regions from the incrementation expression and the loop's
  /// body.
  const Stmt *Group;

  /// \brief The region's starting location.
  SourceLocation LocStart;

  /// \brief The region's ending location.
  SourceLocation LocEnd, AlternativeLocEnd;
  unsigned Flags;

public:
  SourceMappingRegion(FileID File, FileID MacroArgumentFile, Counter Count,
                      const Stmt *UnreachableInitiator, const Stmt *Group,
                      SourceLocation LocStart, SourceLocation LocEnd,
                      unsigned Flags = 0)
      : File(File), MacroArgumentFile(MacroArgumentFile), Count(Count),
        UnreachableInitiator(UnreachableInitiator), Group(Group),
        LocStart(LocStart), LocEnd(LocEnd), AlternativeLocEnd(LocStart),
        Flags(Flags) {}

  const FileID &getFile() const { return File; }

  const Counter &getCounter() const { return Count; }

  const SourceLocation &getStartLoc() const { return LocStart; }

  const SourceLocation &getEndLoc(const SourceManager &SM) const {
    if (SM.getFileID(LocEnd) != File)
      return AlternativeLocEnd;
    return LocEnd;
  }

  bool hasFlag(RegionFlags Flag) const { return (Flags & Flag) != 0; }

  void setFlag(RegionFlags Flag) { Flags |= Flag; }

  void clearFlag(RegionFlags Flag) { Flags &= ~Flag; }

  /// \brief Return true if two regions can be merged together.
  bool isMergeable(SourceMappingRegion &R) {
    // FIXME: We allow merging regions with a gap in between them. Should we?
    return File == R.File && MacroArgumentFile == R.MacroArgumentFile &&
           Count == R.Count && UnreachableInitiator == R.UnreachableInitiator &&
           Group == R.Group;
  }

  /// \brief A comparison that sorts such that mergeable regions are adjacent.
  friend bool operator<(const SourceMappingRegion &LHS,
                        const SourceMappingRegion &RHS) {
    return std::tie(LHS.File, LHS.MacroArgumentFile, LHS.Count,
                    LHS.UnreachableInitiator, LHS.Group) <
           std::tie(RHS.File, RHS.MacroArgumentFile, RHS.Count,
                    RHS.UnreachableInitiator, RHS.Group);
  }
};

/// \brief Provides the common functionality for the different
/// coverage mapping region builders.
class CoverageMappingBuilder {
public:
  CoverageMappingModuleGen &CVM;
  SourceManager &SM;
  const LangOptions &LangOpts;

private:
  struct FileInfo {
    /// \brief The file id that will be used by the coverage mapping system.
    unsigned CovMappingFileID;
    const FileEntry *Entry;

    FileInfo(unsigned CovMappingFileID, const FileEntry *Entry)
        : CovMappingFileID(CovMappingFileID), Entry(Entry) {}
  };

  /// \brief This mapping maps clang's FileIDs to file ids used
  /// by the coverage mapping system and clang's file entries.
  llvm::SmallDenseMap<FileID, FileInfo, 8> FileIDMapping;

public:
  /// \brief The statement that corresponds to the current source group.
  const Stmt *CurrentSourceGroup;

  /// \brief The statement the initiated the current unreachable region.
  const Stmt *CurrentUnreachableRegionInitiator;

  /// \brief The coverage mapping regions for this function
  llvm::SmallVector<CounterMappingRegion, 32> MappingRegions;
  /// \brief The source mapping regions for this function.
  std::vector<SourceMappingRegion> SourceRegions;

  CoverageMappingBuilder(CoverageMappingModuleGen &CVM, SourceManager &SM,
                         const LangOptions &LangOpts)
      : CVM(CVM), SM(SM), LangOpts(LangOpts),
        CurrentSourceGroup(nullptr),
        CurrentUnreachableRegionInitiator(nullptr) {}

  /// \brief Return the precise end location for the given token.
  SourceLocation getPreciseTokenLocEnd(SourceLocation Loc) {
    return Lexer::getLocForEndOfToken(SM.getSpellingLoc(Loc), 0, SM, LangOpts);
  }

  /// \brief Create the mapping that maps from the function's file ids to
  /// the indices for the translation unit's filenames.
  void createFileIDMapping(SmallVectorImpl<unsigned> &Mapping) {
    Mapping.resize(FileIDMapping.size(), 0);
    for (const auto &I : FileIDMapping)
      Mapping[I.second.CovMappingFileID] = CVM.getFileID(I.second.Entry);
  }

  /// \brief Get the coverage mapping file id that corresponds to the given
  /// clang file id. If such file id doesn't exist, it gets added to the
  /// mapping that maps from clang's file ids to coverage mapping file ids.
  /// Returns None if there was an error getting the coverage mapping file id.
  /// An example of an when this function fails is when the region tries
  /// to get a coverage file id for a location in a built-in macro.
  Optional<unsigned> getCoverageFileID(SourceLocation LocStart, FileID File,
                                       FileID SpellingFile) {
    auto Mapping = FileIDMapping.find(File);
    if (Mapping != FileIDMapping.end())
      return Mapping->second.CovMappingFileID;

    auto Entry = SM.getFileEntryForID(SpellingFile);
    if (!Entry)
      return None;

    unsigned Result = FileIDMapping.size();
    FileIDMapping.insert(std::make_pair(File, FileInfo(Result, Entry)));
    createFileExpansionRegion(LocStart, File);
    return Result;
  }

  /// \brief Get the coverage mapping file id that corresponds to the given
  /// clang file id.
  /// Returns None if there was an error getting the coverage mapping file id.
  Optional<unsigned> getExistingCoverageFileID(FileID File) {
    // Make sure that the file is valid.
    if (File.isInvalid())
      return None;
    auto Mapping = FileIDMapping.find(File);
    if (Mapping == FileIDMapping.end())
      return None;
    return Mapping->second.CovMappingFileID;
  }

  /// \brief Return true if the given clang's file id has a corresponding
  /// coverage file id.
  bool hasExistingCoverageFileID(FileID File) const {
    return FileIDMapping.count(File);
  }

  /// \brief Gather all the regions that were skipped by the preprocessor
  /// using the constructs like #if.
  void gatherSkippedRegions() {
    /// An array of the minimum lineStarts and the maximum lineEnds
    /// for mapping regions from the appropriate source files.
    llvm::SmallVector<std::pair<unsigned, unsigned>, 8> FileLineRanges;
    FileLineRanges.resize(
        FileIDMapping.size(),
        std::make_pair(std::numeric_limits<unsigned>::max(), 0));
    for (const auto &R : MappingRegions) {
      FileLineRanges[R.FileID].first =
          std::min(FileLineRanges[R.FileID].first, R.LineStart);
      FileLineRanges[R.FileID].second =
          std::max(FileLineRanges[R.FileID].second, R.LineEnd);
    }

    auto SkippedRanges = CVM.getSourceInfo().getSkippedRanges();
    for (const auto &I : SkippedRanges) {
      auto LocStart = I.getBegin();
      auto LocEnd = I.getEnd();
      auto FileStart = SM.getFileID(LocStart);
      if (!hasExistingCoverageFileID(FileStart))
        continue;
      auto ActualFileStart = SM.getDecomposedSpellingLoc(LocStart).first;
      if (ActualFileStart != SM.getDecomposedSpellingLoc(LocEnd).first)
        // Ignore regions that span across multiple files.
        continue;

      auto CovFileID = getCoverageFileID(LocStart, FileStart, ActualFileStart);
      if (!CovFileID)
        continue;
      unsigned LineStart = SM.getSpellingLineNumber(LocStart);
      unsigned ColumnStart = SM.getSpellingColumnNumber(LocStart);
      unsigned LineEnd = SM.getSpellingLineNumber(LocEnd);
      unsigned ColumnEnd = SM.getSpellingColumnNumber(LocEnd);
      CounterMappingRegion Region(Counter(), *CovFileID, LineStart, ColumnStart,
                                  LineEnd, ColumnEnd, false,
                                  CounterMappingRegion::SkippedRegion);
      // Make sure that we only collect the regions that are inside
      // the souce code of this function.
      if (Region.LineStart >= FileLineRanges[*CovFileID].first &&
          Region.LineEnd <= FileLineRanges[*CovFileID].second)
        MappingRegions.push_back(Region);
    }
  }

  /// \brief Create a mapping region that correponds to an expansion of
  /// a macro or an embedded include.
  void createFileExpansionRegion(SourceLocation Loc, FileID ExpandedFile) {
    SourceLocation LocStart;
    if (Loc.isMacroID())
      LocStart = SM.getImmediateExpansionRange(Loc).first;
    else {
      LocStart = SM.getIncludeLoc(ExpandedFile);
      if (LocStart.isInvalid())
        return; // This file has no expansion region.
    }

    auto File = SM.getFileID(LocStart);
    auto SpellingFile = SM.getDecomposedSpellingLoc(LocStart).first;
    auto CovFileID = getCoverageFileID(LocStart, File, SpellingFile);
    auto ExpandedFileID = getExistingCoverageFileID(ExpandedFile);
    if (!CovFileID || !ExpandedFileID)
      return;
    unsigned LineStart = SM.getSpellingLineNumber(LocStart);
    unsigned ColumnStart = SM.getSpellingColumnNumber(LocStart);
    unsigned LineEnd = LineStart;
    // Compute the end column manually as Lexer::getLocForEndOfToken doesn't
    // give the correct result in all cases.
    unsigned ColumnEnd =
        ColumnStart +
        Lexer::MeasureTokenLength(SM.getSpellingLoc(LocStart), SM, LangOpts);

    MappingRegions.push_back(CounterMappingRegion(
        Counter(), *CovFileID, LineStart, ColumnStart, LineEnd, ColumnEnd,
        false, CounterMappingRegion::ExpansionRegion));
    MappingRegions.back().ExpandedFileID = *ExpandedFileID;
  }

  /// \brief Enter a source region group that is identified by the given
  /// statement.
  /// It's not possible to enter a group when there is already
  /// another group present.
  void beginSourceRegionGroup(const Stmt *Group) {
    assert(!CurrentSourceGroup);
    CurrentSourceGroup = Group;
  }

  /// \brief Exit the current source region group.
  void endSourceRegionGroup() { CurrentSourceGroup = nullptr; }

  /// \brief Associate a counter with a given source code range.
  void mapSourceCodeRange(SourceLocation LocStart, SourceLocation LocEnd,
                          Counter Count, const Stmt *UnreachableInitiator,
                          const Stmt *SourceGroup, unsigned Flags = 0,
                          FileID MacroArgumentFile = FileID()) {
    if (SM.isMacroArgExpansion(LocStart)) {
      // Map the code range with the macro argument's value.
      mapSourceCodeRange(SM.getImmediateSpellingLoc(LocStart),
                         SM.getImmediateSpellingLoc(LocEnd), Count,
                         UnreachableInitiator, SourceGroup, Flags,
                         SM.getFileID(LocStart));
      // Map the code range where the macro argument is referenced.
      SourceLocation RefLocStart(SM.getImmediateExpansionRange(LocStart).first);
      SourceLocation RefLocEnd(RefLocStart);
      if (SM.isMacroArgExpansion(RefLocStart))
        mapSourceCodeRange(RefLocStart, RefLocEnd, Count, UnreachableInitiator,
                           SourceGroup, 0, SM.getFileID(RefLocStart));
      else
        mapSourceCodeRange(RefLocStart, RefLocEnd, Count, UnreachableInitiator,
                           SourceGroup);
      return;
    }
    auto File = SM.getFileID(LocStart);
    // Make sure that the file id is valid.
    if (File.isInvalid())
      return;
    SourceRegions.emplace_back(File, MacroArgumentFile, Count,
                               UnreachableInitiator, SourceGroup, LocStart,
                               LocEnd, Flags);
  }

  void mapSourceCodeRange(SourceLocation LocStart, SourceLocation LocEnd,
                          Counter Count, unsigned Flags = 0) {
    mapSourceCodeRange(LocStart, LocEnd, Count,
                       CurrentUnreachableRegionInitiator, CurrentSourceGroup,
                       Flags);
  }

  /// \brief Generate the coverage counter mapping regions from collected
  /// source regions.
  void emitSourceRegions() {
    std::sort(SourceRegions.begin(), SourceRegions.end());

    for (auto I = SourceRegions.begin(), E = SourceRegions.end(); I != E; ++I) {
      // Keep the original start location of this region.
      SourceLocation LocStart = I->getStartLoc();
      SourceLocation LocEnd = I->getEndLoc(SM);

      bool Ignore = I->hasFlag(SourceMappingRegion::IgnoreIfNotExtended);
      // We need to handle mergeable regions together.
      for (auto Next = I + 1; Next != E && Next->isMergeable(*I); ++Next) {
        ++I;
        LocStart = std::min(LocStart, I->getStartLoc());
        LocEnd = std::max(LocEnd, I->getEndLoc(SM));
        // FIXME: Should we && together the Ignore flag of multiple regions?
        Ignore = false;
      }
      if (Ignore)
        continue;

      // Find the spilling locations for the mapping region.
      LocEnd = getPreciseTokenLocEnd(LocEnd);
      unsigned LineStart = SM.getSpellingLineNumber(LocStart);
      unsigned ColumnStart = SM.getSpellingColumnNumber(LocStart);
      unsigned LineEnd = SM.getSpellingLineNumber(LocEnd);
      unsigned ColumnEnd = SM.getSpellingColumnNumber(LocEnd);

      auto SpellingFile = SM.getDecomposedSpellingLoc(LocStart).first;
      auto CovFileID = getCoverageFileID(LocStart, I->getFile(), SpellingFile);
      if (!CovFileID)
        continue;

      assert(LineStart <= LineEnd);
      MappingRegions.push_back(CounterMappingRegion(
          I->getCounter(), *CovFileID, LineStart, ColumnStart, LineEnd,
          ColumnEnd, false, CounterMappingRegion::CodeRegion));
    }
  }
};

/// \brief Creates unreachable coverage regions for the functions that
/// are not emitted.
struct EmptyCoverageMappingBuilder : public CoverageMappingBuilder {
  EmptyCoverageMappingBuilder(CoverageMappingModuleGen &CVM, SourceManager &SM,
                              const LangOptions &LangOpts)
      : CoverageMappingBuilder(CVM, SM, LangOpts) {}

  void VisitDecl(const Decl *D) {
    if (!D->hasBody())
      return;
    auto Body = D->getBody();
    mapSourceCodeRange(Body->getLocStart(), Body->getLocEnd(), Counter());
  }

  /// \brief Write the mapping data to the output stream
  void write(llvm::raw_ostream &OS) {
    emitSourceRegions();
    SmallVector<unsigned, 16> FileIDMapping;
    createFileIDMapping(FileIDMapping);

    CoverageMappingWriter Writer(FileIDMapping, None, MappingRegions);
    Writer.write(OS);
  }
};

/// \brief A StmtVisitor that creates coverage mapping regions which map
/// from the source code locations to the PGO counters.
struct CounterCoverageMappingBuilder
    : public CoverageMappingBuilder,
      public ConstStmtVisitor<CounterCoverageMappingBuilder> {
  /// \brief The map of statements to count values.
  llvm::DenseMap<const Stmt *, unsigned> &CounterMap;

  Counter CurrentRegionCount;

  CounterExpressionBuilder Builder;

  /// \brief Return a counter that represents the
  /// expression that subracts rhs from lhs.
  Counter subtractCounters(Counter LHS, Counter RHS) {
    return Builder.subtract(LHS, RHS);
  }

  /// \brief Return a counter that represents the
  /// the exression that adds lhs and rhs.
  Counter addCounters(Counter LHS, Counter RHS) {
    return Builder.add(LHS, RHS);
  }

  /// \brief Return the region counter for the given statement.
  /// This should only be called on statements that have a dedicated counter.
  unsigned getRegionCounter(const Stmt *S) { return CounterMap[S]; }

  /// \brief Return the region count for the counter at the given index.
  Counter getRegionCount(unsigned CounterId) {
    return Counter::getCounter(CounterId);
  }

  /// \brief Return the counter value of the current region.
  Counter getCurrentRegionCount() { return CurrentRegionCount; }

  /// \brief Set the counter value for the current region.
  /// This is used to keep track of changes to the most recent counter
  /// from control flow and non-local exits.
  void setCurrentRegionCount(Counter Count) {
    CurrentRegionCount = Count;
    CurrentUnreachableRegionInitiator = nullptr;
  }

  /// \brief Indicate that the current region is never reached,
  /// and thus should have a counter value of zero.
  /// This is important so that subsequent regions can correctly track
  /// their parent counts.
  void setCurrentRegionUnreachable(const Stmt *Initiator) {
    CurrentRegionCount = Counter::getZero();
    CurrentUnreachableRegionInitiator = Initiator;
  }

  /// \brief A counter for a particular region.
  /// This is the primary interface through
  /// which the coverage mapping builder manages counters and their values.
  class RegionMapper {
    CounterCoverageMappingBuilder &Mapping;
    Counter Count;
    Counter ParentCount;
    Counter RegionCount;
    Counter Adjust;

  public:
    RegionMapper(CounterCoverageMappingBuilder *Mapper, const Stmt *S)
        : Mapping(*Mapper),
          Count(Mapper->getRegionCount(Mapper->getRegionCounter(S))),
          ParentCount(Mapper->getCurrentRegionCount()) {}

    /// Get the value of the counter. In most cases this is the number of times
    /// the region of the counter was entered, but for switch labels it's the
    /// number of direct jumps to that label.
    Counter getCount() const { return Count; }

    /// Get the value of the counter with adjustments applied. Adjustments occur
    /// when control enters or leaves the region abnormally; i.e., if there is a
    /// jump to a label within the region, or if the function can return from
    /// within the region. The adjusted count, then, is the value of the counter
    /// at the end of the region.
    Counter getAdjustedCount() const {
      return Mapping.addCounters(Count, Adjust);
    }

    /// Get the value of the counter in this region's parent, i.e., the region
    /// that was active when this region began. This is useful for deriving
    /// counts in implicitly counted regions, like the false case of a condition
    /// or the normal exits of a loop.
    Counter getParentCount() const { return ParentCount; }

    /// Activate the counter by emitting an increment and starting to track
    /// adjustments. If AddIncomingFallThrough is true, the current region count
    /// will be added to the counter for the purposes of tracking the region.
    void beginRegion(bool AddIncomingFallThrough = false) {
      RegionCount = Count;
      if (AddIncomingFallThrough)
        RegionCount =
            Mapping.addCounters(RegionCount, Mapping.getCurrentRegionCount());
      Mapping.setCurrentRegionCount(RegionCount);
    }

    /// For counters on boolean branches, begins tracking adjustments for the
    /// uncounted path.
    void beginElseRegion() {
      RegionCount = Mapping.subtractCounters(ParentCount, Count);
      Mapping.setCurrentRegionCount(RegionCount);
    }

    /// Reset the current region count.
    void setCurrentRegionCount(Counter CurrentCount) {
      RegionCount = CurrentCount;
      Mapping.setCurrentRegionCount(RegionCount);
    }

    /// Adjust for non-local control flow after emitting a subexpression or
    /// substatement. This must be called to account for constructs such as
    /// gotos,
    /// labels, and returns, so that we can ensure that our region's count is
    /// correct in the code that follows.
    void adjustForControlFlow() {
      Adjust = Mapping.addCounters(
          Adjust, Mapping.subtractCounters(Mapping.getCurrentRegionCount(),
                                           RegionCount));
      // Reset the region count in case this is called again later.
      RegionCount = Mapping.getCurrentRegionCount();
    }

    /// Commit all adjustments to the current region. If the region is a loop,
    /// the LoopAdjust value should be the count of all the breaks and continues
    /// from the loop, to compensate for those counts being deducted from the
    /// adjustments for the body of the loop.
    void applyAdjustmentsToRegion() {
      Mapping.setCurrentRegionCount(Mapping.addCounters(ParentCount, Adjust));
    }
    void applyAdjustmentsToRegion(Counter LoopAdjust) {
      Mapping.setCurrentRegionCount(Mapping.addCounters(
          Mapping.addCounters(ParentCount, Adjust), LoopAdjust));
    }
  };

  /// \brief Keep counts of breaks and continues inside loops.
  struct BreakContinue {
    Counter BreakCount;
    Counter ContinueCount;
  };
  SmallVector<BreakContinue, 8> BreakContinueStack;

  CounterCoverageMappingBuilder(
      CoverageMappingModuleGen &CVM,
      llvm::DenseMap<const Stmt *, unsigned> &CounterMap, SourceManager &SM,
      const LangOptions &LangOpts)
      : CoverageMappingBuilder(CVM, SM, LangOpts), CounterMap(CounterMap) {}

  /// \brief Write the mapping data to the output stream
  void write(llvm::raw_ostream &OS) {
    emitSourceRegions();
    llvm::SmallVector<unsigned, 8> VirtualFileMapping;
    createFileIDMapping(VirtualFileMapping);
    gatherSkippedRegions();

    CoverageMappingWriter Writer(
        VirtualFileMapping, Builder.getExpressions(), MappingRegions);
    Writer.write(OS);
  }

  /// \brief Associate the source code range with the current region count.
  void mapSourceCodeRange(SourceLocation LocStart, SourceLocation LocEnd,
                          unsigned Flags = 0) {
    CoverageMappingBuilder::mapSourceCodeRange(LocStart, LocEnd,
                                               CurrentRegionCount, Flags);
  }

  void mapSourceCodeRange(SourceLocation LocStart) {
    CoverageMappingBuilder::mapSourceCodeRange(LocStart, LocStart,
                                               CurrentRegionCount);
  }

  /// \brief Associate the source range of a token with the current region
  /// count.
  /// Ignore the source range for this token if it produces a distinct
  /// mapping region with no other source ranges.
  void mapToken(SourceLocation LocStart) {
    CoverageMappingBuilder::mapSourceCodeRange(
        LocStart, LocStart, CurrentRegionCount,
        SourceMappingRegion::IgnoreIfNotExtended);
  }

  void VisitStmt(const Stmt *S) {
    mapSourceCodeRange(S->getLocStart());
    for (Stmt::const_child_range I = S->children(); I; ++I) {
      if (*I)
        this->Visit(*I);
    }
  }

  void VisitDecl(const Decl *D) {
    if (!D->hasBody())
      return;
    // Counter tracks entry to the function body.
    auto Body = D->getBody();
    RegionMapper Cnt(this, Body);
    Cnt.beginRegion();
    Visit(Body);
  }

  void VisitDeclStmt(const DeclStmt *S) {
    mapSourceCodeRange(S->getLocStart());
    for (Stmt::const_child_range I = static_cast<const Stmt *>(S)->children();
         I; ++I) {
      if (*I)
        this->Visit(*I);
    }
  }

  void VisitCompoundStmt(const CompoundStmt *S) {
    mapSourceCodeRange(S->getLBracLoc());
    mapSourceCodeRange(S->getRBracLoc());
    for (Stmt::const_child_range I = S->children(); I; ++I) {
      if (*I)
        this->Visit(*I);
    }
  }

  void VisitReturnStmt(const ReturnStmt *S) {
    mapSourceCodeRange(S->getLocStart());
    if (S->getRetValue())
      Visit(S->getRetValue());
    setCurrentRegionUnreachable(S);
  }

  void VisitGotoStmt(const GotoStmt *S) {
    mapSourceCodeRange(S->getLocStart());
    mapToken(S->getLabelLoc());
    setCurrentRegionUnreachable(S);
  }

  void VisitLabelStmt(const LabelStmt *S) {
    // Counter tracks the block following the label.
    RegionMapper Cnt(this, S);
    Cnt.beginRegion();
    mapSourceCodeRange(S->getLocStart());
    // Can't map the ':' token as its location isn't known.
    Visit(S->getSubStmt());
  }

  void VisitBreakStmt(const BreakStmt *S) {
    mapSourceCodeRange(S->getLocStart());
    assert(!BreakContinueStack.empty() && "break not in a loop or switch!");
    BreakContinueStack.back().BreakCount = addCounters(
        BreakContinueStack.back().BreakCount, getCurrentRegionCount());
    setCurrentRegionUnreachable(S);
  }

  void VisitContinueStmt(const ContinueStmt *S) {
    mapSourceCodeRange(S->getLocStart());
    assert(!BreakContinueStack.empty() && "continue stmt not in a loop!");
    BreakContinueStack.back().ContinueCount = addCounters(
        BreakContinueStack.back().ContinueCount, getCurrentRegionCount());
    setCurrentRegionUnreachable(S);
  }

  void VisitWhileStmt(const WhileStmt *S) {
    mapSourceCodeRange(S->getLocStart());
    // Counter tracks the body of the loop.
    RegionMapper Cnt(this, S);
    BreakContinueStack.push_back(BreakContinue());
    // Visit the body region first so the break/continue adjustments can be
    // included when visiting the condition.
    Cnt.beginRegion();
    Visit(S->getBody());
    Cnt.adjustForControlFlow();

    // ...then go back and propagate counts through the condition. The count
    // at the start of the condition is the sum of the incoming edges,
    // the backedge from the end of the loop body, and the edges from
    // continue statements.
    BreakContinue BC = BreakContinueStack.pop_back_val();
    Cnt.setCurrentRegionCount(
        addCounters(Cnt.getParentCount(),
                    addCounters(Cnt.getAdjustedCount(), BC.ContinueCount)));
    beginSourceRegionGroup(S->getCond());
    Visit(S->getCond());
    endSourceRegionGroup();
    Cnt.adjustForControlFlow();
    Cnt.applyAdjustmentsToRegion(addCounters(BC.BreakCount, BC.ContinueCount));
  }

  void VisitDoStmt(const DoStmt *S) {
    mapSourceCodeRange(S->getLocStart());
    // Counter tracks the body of the loop.
    RegionMapper Cnt(this, S);
    BreakContinueStack.push_back(BreakContinue());
    Cnt.beginRegion(/*AddIncomingFallThrough=*/true);
    Visit(S->getBody());
    Cnt.adjustForControlFlow();

    BreakContinue BC = BreakContinueStack.pop_back_val();
    // The count at the start of the condition is equal to the count at the
    // end of the body. The adjusted count does not include either the
    // fall-through count coming into the loop or the continue count, so add
    // both of those separately. This is coincidentally the same equation as
    // with while loops but for different reasons.
    Cnt.setCurrentRegionCount(
        addCounters(Cnt.getParentCount(),
                    addCounters(Cnt.getAdjustedCount(), BC.ContinueCount)));
    Visit(S->getCond());
    Cnt.adjustForControlFlow();
    Cnt.applyAdjustmentsToRegion(addCounters(BC.BreakCount, BC.ContinueCount));
  }

  void VisitForStmt(const ForStmt *S) {
    mapSourceCodeRange(S->getLocStart());
    if (S->getInit())
      Visit(S->getInit());

    // Counter tracks the body of the loop.
    RegionMapper Cnt(this, S);
    BreakContinueStack.push_back(BreakContinue());
    // Visit the body region first. (This is basically the same as a while
    // loop; see further comments in VisitWhileStmt.)
    Cnt.beginRegion();
    Visit(S->getBody());
    Cnt.adjustForControlFlow();

    // The increment is essentially part of the body but it needs to include
    // the count for all the continue statements.
    if (S->getInc()) {
      Cnt.setCurrentRegionCount(addCounters(
          getCurrentRegionCount(), BreakContinueStack.back().ContinueCount));
      beginSourceRegionGroup(S->getInc());
      Visit(S->getInc());
      endSourceRegionGroup();
      Cnt.adjustForControlFlow();
    }

    BreakContinue BC = BreakContinueStack.pop_back_val();

    // ...then go back and propagate counts through the condition.
    if (S->getCond()) {
      Cnt.setCurrentRegionCount(
          addCounters(addCounters(Cnt.getParentCount(), Cnt.getAdjustedCount()),
                      BC.ContinueCount));
      beginSourceRegionGroup(S->getCond());
      Visit(S->getCond());
      endSourceRegionGroup();
      Cnt.adjustForControlFlow();
    }
    Cnt.applyAdjustmentsToRegion(addCounters(BC.BreakCount, BC.ContinueCount));
  }

  void VisitCXXForRangeStmt(const CXXForRangeStmt *S) {
    mapSourceCodeRange(S->getLocStart());
    Visit(S->getRangeStmt());
    Visit(S->getBeginEndStmt());
    // Counter tracks the body of the loop.
    RegionMapper Cnt(this, S);
    BreakContinueStack.push_back(BreakContinue());
    // Visit the body region first. (This is basically the same as a while
    // loop; see further comments in VisitWhileStmt.)
    Cnt.beginRegion();
    Visit(S->getBody());
    Cnt.adjustForControlFlow();
    BreakContinue BC = BreakContinueStack.pop_back_val();
    Cnt.applyAdjustmentsToRegion(addCounters(BC.BreakCount, BC.ContinueCount));
  }

  void VisitObjCForCollectionStmt(const ObjCForCollectionStmt *S) {
    mapSourceCodeRange(S->getLocStart());
    Visit(S->getElement());
    // Counter tracks the body of the loop.
    RegionMapper Cnt(this, S);
    BreakContinueStack.push_back(BreakContinue());
    Cnt.beginRegion();
    Visit(S->getBody());
    BreakContinue BC = BreakContinueStack.pop_back_val();
    Cnt.adjustForControlFlow();
    Cnt.applyAdjustmentsToRegion(addCounters(BC.BreakCount, BC.ContinueCount));
  }

  void VisitSwitchStmt(const SwitchStmt *S) {
    mapSourceCodeRange(S->getLocStart());
    Visit(S->getCond());
    BreakContinueStack.push_back(BreakContinue());
    // Map the '}' for the body to have the same count as the regions after
    // the switch.
    SourceLocation RBracLoc;
    if (const auto *CS = dyn_cast<CompoundStmt>(S->getBody())) {
      mapSourceCodeRange(CS->getLBracLoc());
      setCurrentRegionUnreachable(S);
      for (Stmt::const_child_range I = CS->children(); I; ++I) {
        if (*I)
          this->Visit(*I);
      }
      RBracLoc = CS->getRBracLoc();
    } else {
      setCurrentRegionUnreachable(S);
      Visit(S->getBody());
    }
    // If the switch is inside a loop, add the continue counts.
    BreakContinue BC = BreakContinueStack.pop_back_val();
    if (!BreakContinueStack.empty())
      BreakContinueStack.back().ContinueCount = addCounters(
          BreakContinueStack.back().ContinueCount, BC.ContinueCount);
    // Counter tracks the exit block of the switch.
    RegionMapper ExitCnt(this, S);
    ExitCnt.beginRegion();
    if (RBracLoc.isValid())
      mapSourceCodeRange(RBracLoc);
  }

  void VisitCaseStmt(const CaseStmt *S) {
    // Counter for this particular case. This counts only jumps from the
    // switch header and does not include fallthrough from the case before
    // this one.
    RegionMapper Cnt(this, S);
    Cnt.beginRegion(/*AddIncomingFallThrough=*/true);
    mapSourceCodeRange(S->getLocStart());
    mapToken(S->getColonLoc());
    Visit(S->getSubStmt());
  }

  void VisitDefaultStmt(const DefaultStmt *S) {
    // Counter for this default case. This does not include fallthrough from
    // the previous case.
    RegionMapper Cnt(this, S);
    Cnt.beginRegion(/*AddIncomingFallThrough=*/true);
    mapSourceCodeRange(S->getLocStart());
    mapToken(S->getColonLoc());
    Visit(S->getSubStmt());
  }

  void VisitIfStmt(const IfStmt *S) {
    mapSourceCodeRange(S->getLocStart());
    Visit(S->getCond());
    mapToken(S->getElseLoc());

    // Counter tracks the "then" part of an if statement. The count for
    // the "else" part, if it exists, will be calculated from this counter.
    RegionMapper Cnt(this, S);
    Cnt.beginRegion();
    Visit(S->getThen());
    Cnt.adjustForControlFlow();

    if (S->getElse()) {
      Cnt.beginElseRegion();
      Visit(S->getElse());
      Cnt.adjustForControlFlow();
    }
    Cnt.applyAdjustmentsToRegion();
  }

  void VisitCXXTryStmt(const CXXTryStmt *S) {
    mapSourceCodeRange(S->getLocStart());
    Visit(S->getTryBlock());
    for (unsigned I = 0, E = S->getNumHandlers(); I < E; ++I)
      Visit(S->getHandler(I));
    // Counter tracks the continuation block of the try statement.
    RegionMapper Cnt(this, S);
    Cnt.beginRegion();
  }

  void VisitCXXCatchStmt(const CXXCatchStmt *S) {
    mapSourceCodeRange(S->getLocStart());
    // Counter tracks the catch statement's handler block.
    RegionMapper Cnt(this, S);
    Cnt.beginRegion();
    Visit(S->getHandlerBlock());
  }

  void VisitAbstractConditionalOperator(const AbstractConditionalOperator *E) {
    Visit(E->getCond());
    mapToken(E->getQuestionLoc());
    mapToken(E->getColonLoc());

    // Counter tracks the "true" part of a conditional operator. The
    // count in the "false" part will be calculated from this counter.
    RegionMapper Cnt(this, E);
    Cnt.beginRegion();
    Visit(E->getTrueExpr());
    Cnt.adjustForControlFlow();

    Cnt.beginElseRegion();
    Visit(E->getFalseExpr());
    Cnt.adjustForControlFlow();

    Cnt.applyAdjustmentsToRegion();
  }

  void VisitBinLAnd(const BinaryOperator *E) {
    Visit(E->getLHS());
    mapToken(E->getOperatorLoc());
    // Counter tracks the right hand side of a logical and operator.
    RegionMapper Cnt(this, E);
    Cnt.beginRegion();
    Visit(E->getRHS());
    Cnt.adjustForControlFlow();
    Cnt.applyAdjustmentsToRegion();
  }

  void VisitBinLOr(const BinaryOperator *E) {
    Visit(E->getLHS());
    mapToken(E->getOperatorLoc());
    // Counter tracks the right hand side of a logical or operator.
    RegionMapper Cnt(this, E);
    Cnt.beginRegion();
    Visit(E->getRHS());
    Cnt.adjustForControlFlow();
    Cnt.applyAdjustmentsToRegion();
  }

  void VisitParenExpr(const ParenExpr *E) {
    mapToken(E->getLParen());
    Visit(E->getSubExpr());
    mapToken(E->getRParen());
  }

  void VisitBinaryOperator(const BinaryOperator *E) {
    Visit(E->getLHS());
    mapToken(E->getOperatorLoc());
    Visit(E->getRHS());
  }

  void VisitUnaryOperator(const UnaryOperator *E) {
    bool Postfix = E->isPostfix();
    if (!Postfix)
      mapToken(E->getOperatorLoc());
    Visit(E->getSubExpr());
    if (Postfix)
      mapToken(E->getOperatorLoc());
  }

  void VisitMemberExpr(const MemberExpr *E) {
    Visit(E->getBase());
    mapToken(E->getMemberLoc());
  }

  void VisitCallExpr(const CallExpr *E) {
    Visit(E->getCallee());
    for (const auto &Arg : E->arguments())
      Visit(Arg);
    mapToken(E->getRParenLoc());
  }

  void VisitArraySubscriptExpr(const ArraySubscriptExpr *E) {
    Visit(E->getLHS());
    Visit(E->getRHS());
    mapToken(E->getRBracketLoc());
  }

  void VisitCStyleCastExpr(const CStyleCastExpr *E) {
    mapToken(E->getLParenLoc());
    mapToken(E->getRParenLoc());
    Visit(E->getSubExpr());
  }

  // Map literals as tokens so that the macros like #define PI 3.14
  // won't generate coverage mapping regions.

  void VisitIntegerLiteral(const IntegerLiteral *E) {
    mapToken(E->getLocStart());
  }

  void VisitFloatingLiteral(const FloatingLiteral *E) {
    mapToken(E->getLocStart());
  }

  void VisitCharacterLiteral(const CharacterLiteral *E) {
    mapToken(E->getLocStart());
  }

  void VisitStringLiteral(const StringLiteral *E) {
    mapToken(E->getLocStart());
  }

  void VisitImaginaryLiteral(const ImaginaryLiteral *E) {
    mapToken(E->getLocStart());
  }

  void VisitObjCMessageExpr(const ObjCMessageExpr *E) {
    mapToken(E->getLeftLoc());
    for (Stmt::const_child_range I = static_cast<const Stmt*>(E)->children(); I;
         ++I) {
      if (*I)
        this->Visit(*I);
    }
    mapToken(E->getRightLoc());
  }
};
}

static bool isMachO(const CodeGenModule &CGM) {
  return CGM.getTarget().getTriple().isOSBinFormatMachO();
}

static StringRef getCoverageSection(const CodeGenModule &CGM) {
  return isMachO(CGM) ? "__DATA,__llvm_covmap" : "__llvm_covmap";
}

static void dump(llvm::raw_ostream &OS, const CoverageMappingRecord &Function) {
  OS << Function.FunctionName << ":\n";
  CounterMappingContext Ctx(Function.Expressions);
  for (const auto &R : Function.MappingRegions) {
    OS.indent(2);
    switch (R.Kind) {
    case CounterMappingRegion::CodeRegion:
      break;
    case CounterMappingRegion::ExpansionRegion:
      OS << "Expansion,";
      break;
    case CounterMappingRegion::SkippedRegion:
      OS << "Skipped,";
      break;
    }

    OS << "File " << R.FileID << ", " << R.LineStart << ":"
           << R.ColumnStart << " -> " << R.LineEnd << ":" << R.ColumnEnd
           << " = ";
    Ctx.dump(R.Count, OS);
    OS << " (HasCodeBefore = " << R.HasCodeBefore;
    if (R.Kind == CounterMappingRegion::ExpansionRegion)
      OS << ", Expanded file = " << R.ExpandedFileID;

    OS << ")\n";
  }
}

void CoverageMappingModuleGen::addFunctionMappingRecord(
    llvm::GlobalVariable *FunctionName, StringRef FunctionNameValue,
    uint64_t FunctionHash, const std::string &CoverageMapping) {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  auto *Int32Ty = llvm::Type::getInt32Ty(Ctx);
  auto *Int64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *Int8PtrTy = llvm::Type::getInt8PtrTy(Ctx);
  if (!FunctionRecordTy) {
    llvm::Type *FunctionRecordTypes[] = {Int8PtrTy, Int32Ty, Int32Ty, Int64Ty};
    FunctionRecordTy =
        llvm::StructType::get(Ctx, makeArrayRef(FunctionRecordTypes));
  }

  llvm::Constant *FunctionRecordVals[] = {
      llvm::ConstantExpr::getBitCast(FunctionName, Int8PtrTy),
      llvm::ConstantInt::get(Int32Ty, FunctionNameValue.size()),
      llvm::ConstantInt::get(Int32Ty, CoverageMapping.size()),
      llvm::ConstantInt::get(Int64Ty, FunctionHash)};
  FunctionRecords.push_back(llvm::ConstantStruct::get(
      FunctionRecordTy, makeArrayRef(FunctionRecordVals)));
  CoverageMappings += CoverageMapping;

  if (CGM.getCodeGenOpts().DumpCoverageMapping) {
    // Dump the coverage mapping data for this function by decoding the
    // encoded data. This allows us to dump the mapping regions which were
    // also processed by the CoverageMappingWriter which performs
    // additional minimization operations such as reducing the number of
    // expressions.
    std::vector<StringRef> Filenames;
    std::vector<CounterExpression> Expressions;
    std::vector<CounterMappingRegion> Regions;
    llvm::SmallVector<StringRef, 16> FilenameRefs;
    FilenameRefs.resize(FileEntries.size());
    for (const auto &Entry : FileEntries)
      FilenameRefs[Entry.second] = Entry.first->getName();
    RawCoverageMappingReader Reader(FunctionNameValue, CoverageMapping,
                                    FilenameRefs,
                                    Filenames, Expressions, Regions);
    CoverageMappingRecord FunctionRecord;
    if (Reader.read(FunctionRecord))
      return;
    dump(llvm::outs(), FunctionRecord);
  }
}

void CoverageMappingModuleGen::emit() {
  if (FunctionRecords.empty())
    return;
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  auto *Int32Ty = llvm::Type::getInt32Ty(Ctx);

  // Create the filenames and merge them with coverage mappings
  llvm::SmallVector<std::string, 16> FilenameStrs;
  llvm::SmallVector<StringRef, 16> FilenameRefs;
  FilenameStrs.resize(FileEntries.size());
  FilenameRefs.resize(FileEntries.size());
  for (const auto &Entry : FileEntries) {
    llvm::SmallString<256> Path(Entry.first->getName());
    llvm::sys::fs::make_absolute(Path);

    auto I = Entry.second;
    FilenameStrs[I] = std::move(std::string(Path.begin(), Path.end()));
    FilenameRefs[I] = FilenameStrs[I];
  }

  std::string FilenamesAndCoverageMappings;
  llvm::raw_string_ostream OS(FilenamesAndCoverageMappings);
  CoverageFilenamesSectionWriter(FilenameRefs).write(OS);
  OS << CoverageMappings;
  size_t CoverageMappingSize = CoverageMappings.size();
  size_t FilenamesSize = OS.str().size() - CoverageMappingSize;
  // Append extra zeroes if necessary to ensure that the size of the filenames
  // and coverage mappings is a multiple of 8.
  if (size_t Rem = OS.str().size() % 8) {
    CoverageMappingSize += 8 - Rem;
    for (size_t I = 0, S = 8 - Rem; I < S; ++I)
      OS << '\0';
  }
  auto *FilenamesAndMappingsVal =
      llvm::ConstantDataArray::getString(Ctx, OS.str(), false);

  // Create the deferred function records array
  auto RecordsTy =
      llvm::ArrayType::get(FunctionRecordTy, FunctionRecords.size());
  auto RecordsVal = llvm::ConstantArray::get(RecordsTy, FunctionRecords);

  // Create the coverage data record
  llvm::Type *CovDataTypes[] = {Int32Ty,   Int32Ty,
                                Int32Ty,   Int32Ty,
                                RecordsTy, FilenamesAndMappingsVal->getType()};
  auto CovDataTy = llvm::StructType::get(Ctx, makeArrayRef(CovDataTypes));
  llvm::Constant *TUDataVals[] = {
      llvm::ConstantInt::get(Int32Ty, FunctionRecords.size()),
      llvm::ConstantInt::get(Int32Ty, FilenamesSize),
      llvm::ConstantInt::get(Int32Ty, CoverageMappingSize),
      llvm::ConstantInt::get(Int32Ty,
                             /*Version=*/CoverageMappingVersion1),
      RecordsVal, FilenamesAndMappingsVal};
  auto CovDataVal =
      llvm::ConstantStruct::get(CovDataTy, makeArrayRef(TUDataVals));
  auto CovData = new llvm::GlobalVariable(CGM.getModule(), CovDataTy, true,
                                          llvm::GlobalValue::InternalLinkage,
                                          CovDataVal,
                                          "__llvm_coverage_mapping");

  CovData->setSection(getCoverageSection(CGM));
  CovData->setAlignment(8);

  // Make sure the data doesn't get deleted.
  CGM.addUsedGlobal(CovData);
}

unsigned CoverageMappingModuleGen::getFileID(const FileEntry *File) {
  auto It = FileEntries.find(File);
  if (It != FileEntries.end())
    return It->second;
  unsigned FileID = FileEntries.size();
  FileEntries.insert(std::make_pair(File, FileID));
  return FileID;
}

void CoverageMappingGen::emitCounterMapping(const Decl *D,
                                            llvm::raw_ostream &OS) {
  assert(CounterMap);
  CounterCoverageMappingBuilder Walker(CVM, *CounterMap, SM, LangOpts);
  Walker.VisitDecl(D);
  Walker.write(OS);
}

void CoverageMappingGen::emitEmptyMapping(const Decl *D,
                                          llvm::raw_ostream &OS) {
  EmptyCoverageMappingBuilder Walker(CVM, SM, LangOpts);
  Walker.VisitDecl(D);
  Walker.write(OS);
}
