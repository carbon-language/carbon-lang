//===------ CacheMetrics.cpp - Calculate metrics for instruction cache ----===//
//
//                     Functions to show metrics of cache lines
//
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "CacheMetrics.h"

using namespace llvm;
using namespace bolt;
using Traversal = std::vector<BinaryBasicBlock *>;

namespace {

/// Initialize and return a position map for binary basic blocks.
std::unordered_map<BinaryBasicBlock *, double>
getPositionMap(const BinaryFunction &Function) {
  std::unordered_map<BinaryBasicBlock *, double> DistMap;
  double CurrAddress = 0;
  for (auto *BB : Function.layout()) {
    uint64_t Size = BB->estimateSize();
    DistMap[BB] = CurrAddress + (double)Size / 2;
    CurrAddress += Size;
  }
  return DistMap;
}

/// Initialize and return a vector of traversals for a given entry block
std::vector<Traversal> getTraversals(BinaryBasicBlock *EntryBB) {
  std::vector<Traversal> AllTraversals;
  std::stack<std::pair<BinaryBasicBlock *, Traversal>> Stack;
  Stack.push(std::make_pair(EntryBB, Traversal()));
  std::unordered_set<BinaryBasicBlock *> BBSet;

  while (!Stack.empty()) {
    BinaryBasicBlock *CurrentBB = Stack.top().first;
    Traversal PrevTraversal(Stack.top().second);
    Stack.pop();

    // Add current basic block into consideration
    BBSet.insert(CurrentBB);
    PrevTraversal.push_back(CurrentBB);

    if (CurrentBB->succ_empty()) {
      AllTraversals.push_back(PrevTraversal);
      continue;
    }

    bool HaveSuccCount = false;
    // Calculate total edges count of successors
    for (auto BI = CurrentBB->branch_info_begin();
         BI != CurrentBB->branch_info_end(); ++BI) {
      if (BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE && BI->Count > 0) {
        HaveSuccCount = true;
        break;
      }
    }
    if (!HaveSuccCount) {
      AllTraversals.push_back(PrevTraversal);
      continue;
    }

    auto BI = CurrentBB->branch_info_begin();
    for (auto *SuccBB : CurrentBB->successors()) {
      // If we have never seen SuccBB, or SuccBB indicates the
      // end of traversal, SuccBB will be added into stack for
      // further exploring.
      if ((BBSet.find(SuccBB) == BBSet.end() && BI->Count != 0 &&
           BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE) ||
          SuccBB->succ_empty()) {
        Stack.push(std::make_pair(SuccBB, PrevTraversal));
      }
      ++BI;
    }
  }

  return AllTraversals;
}

/// Given a traversal, return the sum of block distances along this traversal.
double
getTraversalLength(std::unordered_map<BinaryBasicBlock *, double> &DistMap,
                   Traversal const &Path) {
  double Length = 0.0;
  BinaryBasicBlock *PrevBB = Path.front();
  for (auto BBI = std::next(Path.begin()); BBI != Path.end(); ++BBI) {
    Length += std::abs(DistMap[*BBI] - DistMap[PrevBB]);
    PrevBB = *BBI;
  }

  return Length;
}

/// Calculate average number of call distance for every graph traversal
double calcGraphDistance(const std::vector<BinaryFunction *> &BinaryFunctions) {
  double TotalTraversalLength = 0;
  double NumTraversals = 0;
  for (auto BF : BinaryFunctions) {
    // Only consider functions which are known to be executed
    if (BF->getKnownExecutionCount() == 0)
      continue;

    for (auto BB : BF->layout()) {
      if (BB->isEntryPoint()) {
        auto AllTraversals = getTraversals(BB);
        auto DistMap = getPositionMap(*BF);
        for (auto const &Path : AllTraversals) {
          // Ignore short traversals
          if (Path.size() <= 1)
            continue;
          TotalTraversalLength += getTraversalLength(DistMap, Path);
          NumTraversals++;
        }
      }
    }
  }

  return TotalTraversalLength / NumTraversals;
}

}

void CacheMetrics::printAll(
  const std::vector<BinaryFunction *> &BinaryFunctions) {

  size_t NumFunctions = 0;
  size_t NumHotFunctions = 0;
  size_t NumBlocks = 0;
  size_t NumHotBlocks = 0;

  for (auto BF : BinaryFunctions) {
    NumFunctions++;
    if (BF->getKnownExecutionCount() > 0)
      NumHotFunctions++;
    for (auto BB : BF->layout()) {
      NumBlocks++;
      if (BB->getKnownExecutionCount() > 0)
        NumHotBlocks++;
    }
  }

  outs() << format("  There are %zu functions;", NumFunctions)
         << format(" %zu (%.2lf%%) have non-empty execution count\n",
                   NumHotFunctions, 100.0 * NumHotFunctions / NumFunctions);
  outs() << format("  There are %zu basic blocks;", NumBlocks)
         << format(" %zu (%.2lf%%) have non-empty execution count\n",
                  NumHotBlocks, 100.0 * NumHotBlocks / NumBlocks);

  const auto GraphDistance = calcGraphDistance(BinaryFunctions);
  outs() << "  An average length of graph traversal is "
         << format("%.2lf\n", GraphDistance);
}
