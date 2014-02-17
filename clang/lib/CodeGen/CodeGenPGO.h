//===--- CodeGenPGO.h - PGO Instrumentation for LLVM CodeGen ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Instrumentation-based profile-guided optimization
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CODEGENPGO_H
#define CLANG_CODEGEN_CODEGENPGO_H

#include "CGBuilder.h"
#include "CodeGenModule.h"
#include "CodeGenTypes.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MemoryBuffer.h"

namespace clang {
namespace CodeGen {
class RegionCounter;

/// The raw counter data from an instrumented PGO binary
class PGOProfileData {
private:
  /// The PGO data
  llvm::OwningPtr<llvm::MemoryBuffer> DataBuffer;
  /// Offsets into DataBuffer for each function's counters
  llvm::StringMap<unsigned> DataOffsets;
  /// Execution counts for each function.
  llvm::StringMap<uint64_t> FunctionCounts;
  /// The maximal execution count among all functions.
  uint64_t MaxFunctionCount;
  CodeGenModule &CGM;
public:
  PGOProfileData(CodeGenModule &CGM, std::string Path);
  /// Fill Counts with the profile data for the given function name. Returns
  /// false on success.
  bool getFunctionCounts(StringRef MangledName, std::vector<uint64_t> &Counts);
  /// Return true if a function is hot. If we know nothing about the function,
  /// return false.
  bool isHotFunction(StringRef MangledName);
  /// Return true if a function is cold. If we know nothing about the function,
  /// return false.
  bool isColdFunction(StringRef MangledName);
};

/// Per-function PGO state. This class should generally not be used directly,
/// but instead through the CodeGenFunction and RegionCounter types.
class CodeGenPGO {
private:
  CodeGenModule &CGM;

  unsigned NumRegionCounters;
  llvm::GlobalVariable *RegionCounters;
  llvm::DenseMap<const Stmt*, unsigned> *RegionCounterMap;
  llvm::DenseMap<const Stmt*, uint64_t> *StmtCountMap;
  std::vector<uint64_t> *RegionCounts;
  uint64_t CurrentRegionCount;

public:
  CodeGenPGO(CodeGenModule &CGM)
    : CGM(CGM), NumRegionCounters(0), RegionCounters(0), RegionCounterMap(0),
      StmtCountMap(0), RegionCounts(0), CurrentRegionCount(0) {}
  ~CodeGenPGO() {}

  /// Whether or not we have PGO region data for the current function. This is
  /// false both when we have no data at all and when our data has been
  /// discarded.
  bool haveRegionCounts() const { return RegionCounts != 0; }

  /// Return the counter value of the current region.
  uint64_t getCurrentRegionCount() const { return CurrentRegionCount; }

  /// Set the counter value for the current region. This is used to keep track
  /// of changes to the most recent counter from control flow and non-local
  /// exits.
  void setCurrentRegionCount(uint64_t Count) { CurrentRegionCount = Count; }

  /// Indicate that the current region is never reached, and thus should have a
  /// counter value of zero. This is important so that subsequent regions can
  /// correctly track their parent counts.
  void setCurrentRegionUnreachable() { setCurrentRegionCount(0); }

  /// Check if an execution count is known for a given statement. If so, return
  /// true and put the value in Count; else return false.
  bool getStmtCount(const Stmt *S, uint64_t &Count) {
    if (!StmtCountMap)
      return false;
    llvm::DenseMap<const Stmt*, uint64_t>::const_iterator
      I = StmtCountMap->find(S);
    if (I == StmtCountMap->end())
      return false;
    Count = I->second;
    return true;
  }

  /// If the execution count for the current statement is known, record that
  /// as the current count.
  void setCurrentStmt(const Stmt *S) {
    uint64_t Count;
    if (getStmtCount(S, Count))
      setCurrentRegionCount(Count);
  }

  /// Calculate branch weights appropriate for PGO data
  llvm::MDNode *createBranchWeights(uint64_t TrueCount, uint64_t FalseCount);
  llvm::MDNode *createBranchWeights(ArrayRef<uint64_t> Weights);
  llvm::MDNode *createLoopWeights(const Stmt *Cond, RegionCounter &Cnt);

  /// Assign counters to regions and configure them for PGO of a given
  /// function. Does nothing if instrumentation is not enabled and either
  /// generates global variables or associates PGO data with each of the
  /// counters depending on whether we are generating or using instrumentation.
  void assignRegionCounters(GlobalDecl &GD);
  /// Emit code to write counts for a given function to disk, if necessary.
  void emitWriteoutFunction(GlobalDecl &GD);
  /// Clean up region counter state. Must be called if assignRegionCounters is
  /// used.
  void destroyRegionCounters();
  /// Emit the logic to register region counter write out functions. Returns a
  /// function that implements this logic.
  static llvm::Function *emitInitialization(CodeGenModule &CGM);

private:
  void mapRegionCounters(const Decl *D);
  void computeRegionCounts(const Decl *D);
  void loadRegionCounts(GlobalDecl &GD, PGOProfileData *PGOData);
  void emitCounterVariables();

  /// Emit code to increment the counter at the given index
  void emitCounterIncrement(CGBuilderTy &Builder, unsigned Counter);

  /// Return the region counter for the given statement. This should only be
  /// called on statements that have a dedicated counter.
  unsigned getRegionCounter(const Stmt *S) {
    if (RegionCounterMap == 0)
      return 0;
    return (*RegionCounterMap)[S];
  }

  /// Return the region count for the counter at the given index.
  uint64_t getRegionCount(unsigned Counter) {
    if (!haveRegionCounts())
      return 0;
    return (*RegionCounts)[Counter];
  }

  friend class RegionCounter;
};

/// A counter for a particular region. This is the primary interface through
/// which clients manage PGO counters and their values.
class RegionCounter {
  CodeGenPGO *PGO;
  unsigned Counter;
  uint64_t Count;
  uint64_t ParentCount;
  uint64_t RegionCount;
  int64_t Adjust;

  RegionCounter(CodeGenPGO &PGO, unsigned CounterIndex)
    : PGO(&PGO), Counter(CounterIndex), Count(PGO.getRegionCount(Counter)),
      ParentCount(PGO.getCurrentRegionCount()), Adjust(0) {}

public:
  RegionCounter(CodeGenPGO &PGO, const Stmt *S)
    : PGO(&PGO), Counter(PGO.getRegionCounter(S)),
      Count(PGO.getRegionCount(Counter)),
      ParentCount(PGO.getCurrentRegionCount()), Adjust(0) {}

  /// Get the value of the counter. In most cases this is the number of times
  /// the region of the counter was entered, but for switch labels it's the
  /// number of direct jumps to that label.
  uint64_t getCount() const { return Count; }

  /// Get the value of the counter with adjustments applied. Adjustments occur
  /// when control enters or leaves the region abnormally; i.e., if there is a
  /// jump to a label within the region, or if the function can return from
  /// within the region. The adjusted count, then, is the value of the counter
  /// at the end of the region.
  uint64_t getAdjustedCount() const {
    assert((Adjust > 0 || (uint64_t)(-Adjust) <= Count) && "Negative count");
    return Count + Adjust;
  }

  /// Get the value of the counter in this region's parent, i.e., the region
  /// that was active when this region began. This is useful for deriving
  /// counts in implicitly counted regions, like the false case of a condition
  /// or the normal exits of a loop.
  uint64_t getParentCount() const { return ParentCount; }

  /// Activate the counter by emitting an increment and starting to track
  /// adjustments. If AddIncomingFallThrough is true, the current region count
  /// will be added to the counter for the purposes of tracking the region.
  void beginRegion(CGBuilderTy &Builder, bool AddIncomingFallThrough=false) {
    beginRegion(AddIncomingFallThrough);
    PGO->emitCounterIncrement(Builder, Counter);
  }
  void beginRegion(bool AddIncomingFallThrough=false) {
    RegionCount = Count;
    if (AddIncomingFallThrough)
      RegionCount += PGO->getCurrentRegionCount();
    PGO->setCurrentRegionCount(RegionCount);
  }

  /// For counters on boolean branches, begins tracking adjustments for the
  /// uncounted path.
  void beginElseRegion() {
    RegionCount = ParentCount - Count;
    PGO->setCurrentRegionCount(RegionCount);
  }

  /// Reset the current region count.
  void setCurrentRegionCount(uint64_t CurrentCount) {
    RegionCount = CurrentCount;
    PGO->setCurrentRegionCount(RegionCount);
  }

  /// Adjust for non-local control flow after emitting a subexpression or
  /// substatement. This must be called to account for constructs such as gotos,
  /// labels, and returns, so that we can ensure that our region's count is
  /// correct in the code that follows.
  void adjustForControlFlow() {
    Adjust += PGO->getCurrentRegionCount() - RegionCount;
    // Reset the region count in case this is called again later.
    RegionCount = PGO->getCurrentRegionCount();
  }

  /// Commit all adjustments to the current region. If the region is a loop,
  /// the LoopAdjust value should be the count of all the breaks and continues
  /// from the loop, to compensate for those counts being deducted from the
  /// adjustments for the body of the loop.
  void applyAdjustmentsToRegion(uint64_t LoopAdjust) {
    PGO->setCurrentRegionCount(ParentCount + Adjust + LoopAdjust);
  }
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
