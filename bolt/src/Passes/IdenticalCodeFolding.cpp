//===--- IdenticalCodeFolding.cpp -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//


#include "Passes/IdenticalCodeFolding.h"
#include "llvm/Support/Options.h"
#include <map>
#include <set>
#include <unordered_map>

#define DEBUG_TYPE "bolt-icf"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

static cl::opt<bool>
UseDFS("icf-dfs",
  cl::desc("use DFS ordering when using -icf option"),
  cl::ReallyHidden,
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

} // namespace opts

namespace {

/// Compare two jump tables in 2 functions. The function relies on consistent
/// ordering of basic blocks in both binary functions (e.g. DFS).
bool equalJumpTables(const JumpTable &JumpTableA,
                     const JumpTable &JumpTableB,
                     const BinaryFunction &FunctionA,
                     const BinaryFunction &FunctionB) {
  if (JumpTableA.EntrySize != JumpTableB.EntrySize)
    return false;

  if (JumpTableA.Type != JumpTableB.Type)
    return false;

  if (JumpTableA.getSize() != JumpTableB.getSize())
    return false;

  for (uint64_t Index = 0; Index < JumpTableA.Entries.size(); ++Index) {
    const auto *LabelA = JumpTableA.Entries[Index];
    const auto *LabelB = JumpTableB.Entries[Index];

    const auto *TargetA = FunctionA.getBasicBlockForLabel(LabelA);
    const auto *TargetB = FunctionB.getBasicBlockForLabel(LabelB);

    if (!TargetA || !TargetB) {
      assert((TargetA || LabelA == FunctionA.getFunctionEndLabel()) &&
             "no target basic block found");
      assert((TargetB || LabelB == FunctionB.getFunctionEndLabel()) &&
             "no target basic block found");

      if (TargetA != TargetB)
        return false;

      continue;
    }

    assert(TargetA && TargetB && "cannot locate target block(s)");

    if (TargetA->getLayoutIndex() != TargetB->getLayoutIndex())
      return false;
  }

  return true;
}

/// Helper function that compares an instruction of this function to the
/// given instruction of the given function. The functions should have
/// identical CFG.
template <class Compare>
bool isInstrEquivalentWith(const MCInst &InstA, const BinaryBasicBlock &BBA,
                           const MCInst &InstB, const BinaryBasicBlock &BBB,
                           Compare Comp) {
  if (InstA.getOpcode() != InstB.getOpcode()) {
    return false;
  }

  const auto &BC = BBA.getFunction()->getBinaryContext();

  // In this function we check for special conditions:
  //
  //    * instructions with landing pads
  //
  // Most of the common cases should be handled by MCPlus::equals()
  // that compares regular instruction operands.
  //
  // NB: there's no need to compare jump table indirect jump instructions
  //     separately as jump tables are handled by comparing corresponding
  //     symbols.
  const auto EHInfoA = BC.MIB->getEHInfo(InstA);
  const auto EHInfoB = BC.MIB->getEHInfo(InstB);

  if (EHInfoA || EHInfoB) {
    if (!EHInfoA && (EHInfoB->first || EHInfoB->second))
      return false;

    if (!EHInfoB && (EHInfoA->first || EHInfoA->second))
      return false;

    if (EHInfoA && EHInfoB) {
      // Action indices should match.
      if (EHInfoA->second != EHInfoB->second)
        return false;

      if (!EHInfoA->first != !EHInfoB->first)
        return false;

      if (EHInfoA->first && EHInfoB->first) {
        const auto *LPA = BBA.getLandingPad(EHInfoA->first);
        const auto *LPB = BBB.getLandingPad(EHInfoB->first);
        assert(LPA && LPB && "cannot locate landing pad(s)");

        if (LPA->getLayoutIndex() != LPB->getLayoutIndex())
          return false;
      }
    }
  }

  return BC.MIB->equals(InstA, InstB, Comp);
}


/// Returns true if this function has identical code and CFG with
/// the given function \p BF.
///
/// If \p IgnoreSymbols is set to true, then symbolic operands are ignored
/// during comparison.
///
/// If \p UseDFS is set to true, then compute DFS of each function and use
/// is for CFG equivalency. Potentially it will help to catch more cases,
/// but is slower.
bool isIdenticalWith(const BinaryFunction &A, const BinaryFunction &B,
                     bool IgnoreSymbols, bool UseDFS) {
  assert(A.hasCFG() && B.hasCFG() && "both functions should have CFG");

  // Compare the two functions, one basic block at a time.
  // Currently we require two identical basic blocks to have identical
  // instruction sequences and the same index in their corresponding
  // functions. The latter is important for CFG equality.

  if (A.layout_size() != B.layout_size())
    return false;

  // Comparing multi-entry functions could be non-trivial.
  if (A.isMultiEntry() || B.isMultiEntry())
    return false;

  // Process both functions in either DFS or existing order.
  const auto &OrderA = UseDFS ? A.dfs() : A.getLayout();
  const auto &OrderB = UseDFS ? B.dfs() : B.getLayout();

  const auto &BC = A.getBinaryContext();

  auto BBI = OrderB.begin();
  for (const auto *BB : OrderA) {
    const auto *OtherBB = *BBI;

    if (BB->getLayoutIndex() != OtherBB->getLayoutIndex())
      return false;

    // Compare successor basic blocks.
    // NOTE: the comparison for jump tables is only partially verified here.
    if (BB->succ_size() != OtherBB->succ_size())
      return false;

    auto SuccBBI = OtherBB->succ_begin();
    for (const auto *SuccBB : BB->successors()) {
      const auto *SuccOtherBB = *SuccBBI;
      if (SuccBB->getLayoutIndex() != SuccOtherBB->getLayoutIndex())
        return false;
      ++SuccBBI;
    }

    // Compare all instructions including pseudos.
    auto I = BB->begin(), E = BB->end();
    auto OtherI = OtherBB->begin(), OtherE = OtherBB->end();
    while (I != E && OtherI != OtherE) {

      bool Identical;
      if (IgnoreSymbols) {
        Identical =
          isInstrEquivalentWith(*I, *BB, *OtherI, *OtherBB,
                                [](const MCSymbol *A, const MCSymbol *B) {
                                  return true;
                                });
      } else {
        // Compare symbols.
        auto AreSymbolsIdentical = [&] (const MCSymbol *SymbolA,
                                        const MCSymbol *SymbolB) {
          if (SymbolA == SymbolB)
            return true;

          // All local symbols are considered identical since they affect a
          // control flow and we check the control flow separately.
          // If a local symbol is escaped, then the function (potentially) has
          // multiple entry points and we exclude such functions from
          // comparison.
          if (SymbolA->isTemporary() && SymbolB->isTemporary())
            return true;

          // Compare symbols as functions.
          const auto *FunctionA = BC.getFunctionForSymbol(SymbolA);
          const auto *FunctionB = BC.getFunctionForSymbol(SymbolB);
          if (FunctionA && FunctionB) {
            // Self-referencing functions and recursive calls.
            if (FunctionA == &A && FunctionB == &B)
              return true;
            return FunctionA == FunctionB;
          }

          // Check if symbols are jump tables.
          auto *SIA = BC.getBinaryDataByName(SymbolA->getName());
          if (!SIA)
            return false;
          auto *SIB = BC.getBinaryDataByName(SymbolB->getName());
          if (!SIB)
            return false;

          assert((SIA->getAddress() != SIB->getAddress()) &&
                 "different symbols should not have the same value");

          const auto *JumpTableA =
             A.getJumpTableContainingAddress(SIA->getAddress());
          if (!JumpTableA)
            return false;

          const auto *JumpTableB =
             B.getJumpTableContainingAddress(SIB->getAddress());
          if (!JumpTableB)
            return false;

          if ((SIA->getAddress() - JumpTableA->getAddress()) !=
              (SIB->getAddress() - JumpTableB->getAddress()))
            return false;

          return equalJumpTables(*JumpTableA, *JumpTableB, A, B);
        };

        Identical =
          isInstrEquivalentWith(*I, *BB, *OtherI, *OtherBB,
                                AreSymbolsIdentical);
      }

      if (!Identical) {
        return false;
      }

      ++I; ++OtherI;
    }

    // One of the identical blocks may have a trailing unconditional jump that
    // is ignored for CFG purposes.
    auto *TrailingInstr = (I != E ? &(*I)
                                  : (OtherI != OtherE ? &(*OtherI) : 0));
    if (TrailingInstr && !BC.MIB->isUnconditionalBranch(*TrailingInstr)) {
      return false;
    }

    ++BBI;
  }

  return true;
}
}

namespace llvm {
namespace bolt {

void IdenticalCodeFolding::runOnFunctions(BinaryContext &BC) {
  const auto OriginalFunctionCount = BC.getBinaryFunctions().size();
  uint64_t NumFunctionsFolded = 0;
  uint64_t NumJTFunctionsFolded = 0;
  uint64_t BytesSavedEstimate = 0;
  uint64_t CallsSavedEstimate = 0;

  // This hash table is used to identify identical functions. It maps
  // a function to a bucket of functions identical to it.
  struct KeyHash {
    std::size_t operator()(const BinaryFunction *F) const {
      return F->hash(/*Recompute=*/false);
    }
  };
  struct KeyCongruent {
    bool operator()(const BinaryFunction *A, const BinaryFunction *B) const {
      return isIdenticalWith(*A, *B, /*IgnoreSymbols=*/true, opts::UseDFS);
    }
  };
  struct KeyEqual {
    bool operator()(const BinaryFunction *A, const BinaryFunction *B) const {
      return isIdenticalWith(*A, *B, /*IgnoreSymbols=*/false, opts::UseDFS);
    }
  };

  // Create buckets with congruent functions - functions that potentially could
  // be folded.
  std::unordered_map<BinaryFunction *, std::set<BinaryFunction *>,
                     KeyHash, KeyCongruent> CongruentBuckets;
  for (auto &BFI : BC.getBinaryFunctions()) {
    auto &BF = BFI.second;
    if (!shouldOptimize(BF) || BF.isFolded() || BF.hasSDTMarker())
      continue;

    // Make sure indices are in-order.
    BF.updateLayoutIndices();

    // Pre-compute hash before pushing into hashtable.
    BF.hash(/*Recompute=*/true, opts::UseDFS);

    CongruentBuckets[&BF].emplace(&BF);
  }

  // We repeat the pass until no new modifications happen.
  unsigned Iteration = 1;
  uint64_t NumFoldedLastIteration;
  do {
    NumFoldedLastIteration = 0;

    DEBUG(dbgs() << "BOLT-DEBUG: ICF iteration " << Iteration << "...\n");

    for (auto &CBI : CongruentBuckets) {
      auto &Candidates = CBI.second;
      if (Candidates.size() < 2)
        continue;

      // Identical functions go into the same bucket.
      std::unordered_map<BinaryFunction *, std::vector<BinaryFunction *>,
                         KeyHash, KeyEqual> IdenticalBuckets;
      for (auto *BF : Candidates) {
        IdenticalBuckets[BF].emplace_back(BF);
      }

      for (auto &IBI : IdenticalBuckets) {
        // Functions identified as identical.
        auto &Twins = IBI.second;
        if (Twins.size() < 2)
          continue;

        // Fold functions. Keep the order consistent across invocations with
        // different options.
        std::stable_sort(Twins.begin(), Twins.end(),
            [](const BinaryFunction *A, const BinaryFunction *B) {
              return A->getFunctionNumber() < B->getFunctionNumber();
            });

        BinaryFunction *ParentBF = Twins[0];
        for (unsigned i = 1; i < Twins.size(); ++i) {
          auto *ChildBF = Twins[i];
          DEBUG(dbgs() << "BOLT-DEBUG: folding " << *ChildBF << " into "
                       << *ParentBF << '\n');

          // Remove child function from the list of candidates.
          auto FI = Candidates.find(ChildBF);
          assert(FI != Candidates.end() &&
                 "function expected to be in the set");
          Candidates.erase(FI);

          // Fold the function and remove from the list of processed functions.
          BytesSavedEstimate += ChildBF->getSize();
          CallsSavedEstimate += std::min(ChildBF->getKnownExecutionCount(),
                                         ParentBF->getKnownExecutionCount());
          BC.foldFunction(*ChildBF, *ParentBF);

          ++NumFoldedLastIteration;

          if (ParentBF->hasJumpTables())
            ++NumJTFunctionsFolded;
        }
      }

    }
    NumFunctionsFolded += NumFoldedLastIteration;
    ++Iteration;

  } while (NumFoldedLastIteration > 0);

  DEBUG(
    // Print functions that are congruent but not identical.
    for (auto &CBI : CongruentBuckets) {
      auto &Candidates = CBI.second;
      if (Candidates.size() < 2)
        continue;
      dbgs() << "BOLT-DEBUG: the following " << Candidates.size()
             << " functions (each of size " << (*Candidates.begin())->getSize()
             << " bytes) are congruent but not identical:\n";
      for (auto *BF : Candidates) {
        dbgs() << "  " << *BF;
        if (BF->getKnownExecutionCount()) {
          dbgs() << " (executed " << BF->getKnownExecutionCount() << " times)";
        }
        dbgs() << '\n';
      }
    }
  );

  if (NumFunctionsFolded) {
    outs() << "BOLT-INFO: ICF folded " << NumFunctionsFolded
           << " out of " << OriginalFunctionCount << " functions in "
           << Iteration << " passes. "
           << NumJTFunctionsFolded << " functions had jump tables.\n"
           << "BOLT-INFO: Removing all identical functions will save "
           << format("%.2lf", (double) BytesSavedEstimate / 1024)
           << " KB of code space. Folded functions were called "
           << CallsSavedEstimate << " times based on profile.\n";
  }
}

} // namespace bolt
} // namespace llvm
