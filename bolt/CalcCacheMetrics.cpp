//===------ CalcCacheMetrics.cpp - Calculate metrics of cache lines -------===//
//
//                     Functions to show metrics of cache lines
//
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//


#include "BinaryBasicBlock.h"
#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "BinaryPassManager.h"
#include "CalcCacheMetrics.h"
#include "Exceptions.h"
#include "RewriteInstance.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSectionELF.h"
#include <fstream>

using namespace llvm;
using namespace object;
using namespace bolt;
using Traversal = std::vector<BinaryBasicBlock *>;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

} // namespace opts


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

/// Initialize and return a vector of traversals for a given function and its
/// entry point
std::vector<Traversal> getTraversals(const BinaryFunction &Function,
                                     BinaryBasicBlock *EntryBB) {
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
  if (Path.size() <= 1) {
    return 0.0;
  }

  double Length = 0.0;
  BinaryBasicBlock *PrevBB = Path.front();
  for (auto BBI = std::next(Path.begin()); BBI != Path.end(); ++BBI) {
    Length += std::abs(DistMap[*BBI] - DistMap[PrevBB]);
    PrevBB = *BBI;
  }

  return Length;
}

/// Helper function of calcGraphDistance to go through the call traversals of
/// certain function and to calculate and record the length of each
/// traversal.
void graphDistHelper(std::vector<Traversal> &AllTraversals,
                     const BinaryFunction &Function,
                     std::unordered_map<uint64_t, double> &TraversalMap,
                     uint64_t &TraversalCount) {
  auto DistMap = getPositionMap(Function);

  for (auto const &Path : AllTraversals) {
    TraversalMap[++TraversalCount] = getTraversalLength(DistMap, Path);
  }
}
}

void CalcCacheMetrics::calcGraphDistance(
    const std::map<uint64_t, BinaryFunction> &BinaryFunctions) {

  double TotalFuncValue = 0;
  uint64_t FuncCount = 0;
  for (auto &BFI : BinaryFunctions) {
    auto &Function = BFI.second;
    // Only consider functions which are known to be executed
    if (Function.getKnownExecutionCount() == 0)
      continue;

    std::unordered_map<uint64_t, double> TraversalMap;
    uint64_t TraversalCount = 0;
    for (auto *BB : Function.layout()) {
      if (BB->isEntryPoint()) {
        auto AllTraversals = getTraversals(Function, BB);
        graphDistHelper(AllTraversals, Function, TraversalMap, TraversalCount);
      }
    }

    double TotalValue = 0;
    for (auto const &Entry : TraversalMap) {
      TotalValue += Entry.second;
    }

    double AverageValue =
        TraversalMap.empty() ? 0 : (TotalValue * 1.0 / TraversalMap.size());
    TotalFuncValue += AverageValue;
    FuncCount += TraversalMap.empty() ? 0 : 1;
  }

  outs() << format("           Sum of averages of traversal distance for all "
                   "functions is: %.2f\n",
                   TotalFuncValue)
         << format("           There are %u functions in total\n", FuncCount)
         << format("           On average, every traversal is %.2f long\n\n",
                   TotalFuncValue / FuncCount);
}
