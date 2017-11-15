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

// The weight of fallthrough jumps for ExtTSP metric
constexpr double FallthroughWeight = 1.0;
// The weight of forward jumps for ExtTSP metric
constexpr double ForwardWeight = 1.0;
// The weight of backward jumps for ExtTSP metric
constexpr double BackwardWeight = 1.0;
// The maximum distance (in bytes) of forward jumps for ExtTSP metric
constexpr uint64_t ForwardDistance = 256;
// The maximum distance (in bytes) of backward jumps for ExtTSP metric
constexpr uint64_t BackwardDistance = 256;

// The size of the i-TLB cache page
constexpr uint64_t ITLBPageSize = 4096;
// Capacity of the i-TLB cache
constexpr uint64_t ITLBEntries = 16;

namespace {

/// Initialize and return a position map for binary basic blocks
void extractBasicBlockInfo(
  const std::vector<BinaryFunction *> &BinaryFunctions,
  std::unordered_map<BinaryBasicBlock *, uint64_t> &BBAddr,
  std::unordered_map<BinaryBasicBlock *, uint64_t> &BBSize) {

  // Use addresses/sizes as in the output binary
  for (auto BF : BinaryFunctions) {
    for (auto BB : BF->layout()) {
      BBAddr[BB] = BB->getOutputAddressRange().first;
      BBSize[BB] = BB->getOutputSize();
    }
  }
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
double getTraversalLength(
  const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBAddr,
  const Traversal &Path) {
  double Length = 0;
  for (size_t I = 0; I + 1 < Path.size(); I++) {
    // Ignore calls between hot and cold parts
    if (Path[I]->isCold() != Path[I + 1]->isCold())
      continue;
    double SrcAddr = BBAddr.at(Path[I]);
    double DstAddr = BBAddr.at(Path[I + 1]);
    Length += std::abs(SrcAddr - DstAddr);
  }
  return Length;
}

/// Calculate average number of call distance for every graph traversal
double calcGraphDistance(
  const std::vector<BinaryFunction *> &BinaryFunctions,
  const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBAddr,
  const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBSize) {

  double TotalTraversalLength = 0;
  double NumTraversals = 0;
  for (auto BF : BinaryFunctions) {
    // Only consider functions which are known to be executed
    if (BF->getKnownExecutionCount() == 0)
      continue;

    for (auto BB : BF->layout()) {
      if (BB->isEntryPoint()) {
        auto AllTraversals = getTraversals(BB);
        for (auto const &Path : AllTraversals) {
          // Ignore short traversals
          if (Path.size() <= 1)
            continue;
          TotalTraversalLength += getTraversalLength(BBAddr, Path);
          NumTraversals++;
        }
      }
    }
  }

  return TotalTraversalLength / NumTraversals;
}

/// Calculate TSP metric, which quantifies the number of fallthrough jumps in
/// the ordering of basic blocks
double calcTSPScore(
  const std::vector<BinaryFunction *> &BinaryFunctions,
  const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBAddr,
  const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBSize) {

  double Score = 0;
  for (auto BF : BinaryFunctions) {
    for (auto SrcBB : BF->layout()) {
      auto BI = SrcBB->branch_info_begin();
      for (auto DstBB : SrcBB->successors()) {
        if (SrcBB != DstBB && BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
            BBAddr.at(SrcBB) + BBSize.at(SrcBB) == BBAddr.at(DstBB))
          Score += BI->Count;
        ++BI;
      }
    }
  }
  return Score;
}

/// Calculate Extended-TSP metric, which quantifies the expected number of
/// i-cache misses for a given ordering of basic blocks. The parameters are:
/// - FallthroughWeight is the impact of fallthrough jumps on the score
/// - ForwardWeight is the impact of forward (but not fallthrough) jumps
/// - BackwardWeight is the impact of backward jumps
/// - ForwardDistance is the max distance of a forward jump affecting the score
/// - BackwardDistance is the max distance of a backward jump affecting the score
double calcExtTSPScore(
  const std::vector<BinaryFunction *> &BinaryFunctions,
  const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBAddr,
  const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBSize,
  double FallthroughWeight,
  double ForwardWeight,
  double BackwardWeight,
  uint64_t ForwardDistance,
  uint64_t BackwardDistance) {

  double Score = 0.0;
  for (auto BF : BinaryFunctions) {
    for (auto SrcBB : BF->layout()) {
      auto BI = SrcBB->branch_info_begin();
      for (auto DstBB : SrcBB->successors()) {
        if (DstBB != SrcBB) {
          double Count = BI->Count == BinaryBasicBlock::COUNT_NO_PROFILE
                         ? 0.0
                         : double(BI->Count);
          uint64_t SrcAddr = BBAddr.at(SrcBB);
          uint64_t SrcSize = BBSize.at(SrcBB);
          uint64_t DstAddr = BBAddr.at(DstBB);

          if (SrcAddr <= DstAddr) {
            if (SrcAddr + SrcSize == DstAddr) {
              // fallthrough jump
              Score += FallthroughWeight * Count;
            } else {
              // the distance of the forward jump
              size_t Dist = DstAddr - (SrcAddr + SrcSize);
              if (Dist <= ForwardDistance) {
                double Prob = double(ForwardDistance - Dist) / ForwardDistance;
                Score += ForwardWeight * Prob * Count;
              }
            }
          } else {
            // the distance of the backward jump
            size_t Dist = SrcAddr + SrcSize - DstAddr;
            if (Dist <= BackwardDistance) {
              double Prob = double(BackwardDistance - Dist) / BackwardDistance;
              Score += BackwardWeight * Prob * Count;
            }
          }
        }
        ++BI;
      }
    }
  }
  return Score;
}

using Predecessors = std::vector<std::pair<BinaryFunction *, uint64_t>>;

/// Build a simplified version of the call graph: For every function, keep
/// its callers and the frequencies of the calls
std::unordered_map<const BinaryFunction *, Predecessors>
extractFunctionCalls(const std::vector<BinaryFunction *> &BinaryFunctions) {
  std::unordered_map<const BinaryFunction *, Predecessors> Calls;

  for (auto SrcFunction : BinaryFunctions) {
    const auto &BC = SrcFunction->getBinaryContext();
    for (auto BB : SrcFunction->layout()) {
      // Find call instructions and extract target symbols from each one
      for (auto &Inst : *BB) {
        if (!BC.MIA->isCall(Inst))
          continue;

        // Call info
        const MCSymbol* DstSym = BC.MIA->getTargetSymbol(Inst);
        auto Count = BB->getKnownExecutionCount();
        // Ignore calls w/o information
        if (DstSym == nullptr || Count == 0)
          continue;

        auto DstFunction = BC.getFunctionForSymbol(DstSym);
        // Ignore recursive calls
        if (DstFunction == nullptr ||
            DstFunction->layout_empty() ||
            DstFunction == SrcFunction)
          continue;

        // Record the call
        Calls[DstFunction].push_back(std::make_pair(SrcFunction, Count));
      }
    }
  }
  return Calls;
}

/// Compute expected hit ratio of the i-TLB cache (optimized by HFSortPlus alg).
/// Given an assignment of functions to the i-TLB pages), we divide all
/// functions calls into two categories:
/// - 'short' ones that have a caller-callee distance less than a page;
/// - 'long' ones where the distance exceeds a page.
/// The short calls are likely to result in a i-TLB cache hit. For the long ones,
/// the hit/miss result depends on the 'hotness' of the page (i.e., how often
/// the page is accessed). Assuming that functions are sent to the i-TLB cache
/// in a random order, the probability that a page is present in the cache is
/// proportional to the number of samples corresponding to the functions on the
/// page. The following procedure detects short and long calls, and estimates
/// the expected number of cache misses for the long ones.
double expectedCacheHitRatio(
  const std::vector<BinaryFunction *> &BinaryFunctions,
  const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBAddr,
  const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBSize,
  double PageSize,
  uint64_t CacheEntries) {

  auto Calls = extractFunctionCalls(BinaryFunctions);
  // Compute 'hotness' of the functions
  double TotalSamples = 0;
  std::unordered_map<BinaryFunction *, double> FunctionSamples;
  for (auto BF : BinaryFunctions) {
    double Samples = 0;
    for (auto Pair : Calls[BF]) {
      Samples += Pair.second;
    }
    Samples = std::max(Samples, (double)BF->getKnownExecutionCount());
    FunctionSamples[BF] = Samples;
    TotalSamples += Samples;
  }

  // Compute 'hotness' of the pages
  std::unordered_map<uint64_t, double> PageSamples;
  for (auto BF : BinaryFunctions) {
    if (BF->layout_empty())
      continue;
    auto Page = BBAddr.at(BF->layout_front()) / PageSize;
    PageSamples[Page] += FunctionSamples.at(BF);
  }

  // Computing the expected number of misses for every function
  double Misses = 0;
  for (auto BF : BinaryFunctions) {
    // Skip the function if it has no samples
    if (BF->layout_empty() || FunctionSamples.at(BF) == 0.0)
      continue;
    double Samples = FunctionSamples.at(BF);
    auto Page = BBAddr.at(BF->layout_front()) / PageSize;
    // The probability that the page is not present in the cache
    double MissProb = pow(1.0 - PageSamples[Page] / TotalSamples, CacheEntries);

    // Processing all callers of the function
    for (auto Pair : Calls[BF]) {
      auto SrcFunction = Pair.first;
      auto SrcPage = BBAddr.at(SrcFunction->layout_front()) / PageSize;
      // Is this a 'long' or a 'short' call?
      if (Page != SrcPage) {
        // This is a miss
        Misses += MissProb * Pair.second;
      }
      Samples -= Pair.second;
    }
    assert(Samples >= 0.0 && "Function samples computed incorrectly");
    // The remaining samples likely come from the jitted code
    Misses += Samples * MissProb;
  }

  return 100.0 * (1.0 - Misses / TotalSamples);
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

  std::unordered_map<BinaryBasicBlock *, uint64_t> BBAddr;
  std::unordered_map<BinaryBasicBlock *, uint64_t> BBSize;
  extractBasicBlockInfo(BinaryFunctions, BBAddr, BBSize);

  size_t TotalCodeSize = 0;
  size_t HotCodeSize = 0;
  for (auto Pair : BBSize) {
    TotalCodeSize += Pair.second;
    auto BB = Pair.first;
    if (!BB->isCold() && BB->getFunction()->hasValidIndex())
      HotCodeSize += Pair.second;
  }
  outs() << format("  Hot code takes %.2lf%% of binary (%zu bytes out of %zu)\n",
                   100.0 * HotCodeSize / TotalCodeSize, HotCodeSize, TotalCodeSize);

  outs() << "  An average length of graph traversal: "
         << format("%.0lf\n", calcGraphDistance(BinaryFunctions,
                                                BBAddr,
                                                BBSize));

  outs() << "  Expected i-TLB cache hit ratio "
         << format("(%zu, %zu): ", ITLBPageSize, ITLBEntries)
         << format("%.2lf%%\n", expectedCacheHitRatio(BinaryFunctions,
                                                      BBAddr,
                                                      BBSize,
                                                      ITLBPageSize,
                                                      ITLBEntries));

  outs() << "  TSP score: "
         << format("%.0lf\n", calcTSPScore(BinaryFunctions, BBAddr, BBSize));

  outs() << "  ExtTSP score "
         << format("(%.2lf, %.2lf, %.2lf, %zu, %zu): ", FallthroughWeight,
                                                        ForwardWeight,
                                                        BackwardWeight,
                                                        ForwardDistance,
                                                        BackwardDistance)
         << format("%.0lf\n", calcExtTSPScore(BinaryFunctions,
                                              BBAddr,
                                              BBSize,
                                              FallthroughWeight,
                                              ForwardWeight,
                                              BackwardWeight,
                                              ForwardDistance,
                                              BackwardDistance));

}
