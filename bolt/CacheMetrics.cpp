//===------ CacheMetrics.cpp - Calculate metrics for instruction cache ----===//
//
//                     Functions to show metrics of cache lines
//
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "CacheMetrics.h"
#include "llvm/Support/Options.h"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

cl::opt<double>
FallthroughWeight("fallthrough-weight",
  cl::desc("The weight of forward jumps for ExtTSP metric"),
  cl::init(1),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<double>
ForwardWeight("forward-weight",
  cl::desc("The weight of forward jumps for ExtTSP metric"),
  cl::init(0.4),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<double>
BackwardWeight("backward-weight",
  cl::desc("The weight of backward jumps for ExtTSP metric"),
  cl::init(0.4),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<unsigned>
ForwardDistance("forward-distance",
  cl::desc("The maximum distance (in bytes) of forward jumps for ExtTSP metric"),
  cl::init(768),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<unsigned>
BackwardDistance("backward-distance",
  cl::desc("The maximum distance (in bytes) of backward jumps for ExtTSP metric"),
  cl::init(192),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<unsigned>
ITLBPageSize("itlb-page-size",
  cl::desc("The size of i-tlb cache page"),
  cl::init(4096),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<unsigned>
ITLBEntries("itlb-entries",
  cl::desc("The number of entries in i-tlb cache"),
  cl::init(16),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

}

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

/// Calculate Ext-TSP metric, which quantifies the expected number of i-cache
/// misses for a given ordering of basic blocks
double calcExtTSPScore(
  const std::vector<BinaryFunction *> &BinaryFunctions,
  const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBAddr,
  const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBSize) {

  double Score = 0.0;
  for (auto BF : BinaryFunctions) {
    for (auto SrcBB : BF->layout()) {
      auto BI = SrcBB->branch_info_begin();
      for (auto DstBB : SrcBB->successors()) {
        if (DstBB != SrcBB) {
          Score += CacheMetrics::extTSPScore(BBAddr.at(SrcBB),
                                             BBSize.at(SrcBB),
                                             BBAddr.at(DstBB),
                                             BI->Count);
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
  const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBSize) {

  const double PageSize = opts::ITLBPageSize;
  const uint64_t CacheEntries = opts::ITLBEntries;
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

} // end namespace anonymous

double CacheMetrics::extTSPScore(uint64_t SrcAddr,
                                 uint64_t SrcSize,
                                 uint64_t DstAddr,
                                 uint64_t Count) {
  assert(Count != BinaryBasicBlock::COUNT_NO_PROFILE);

  // Fallthrough
  if (SrcAddr + SrcSize == DstAddr) {
    return opts::FallthroughWeight * Count;
  }
  // Forward
  if (SrcAddr + SrcSize < DstAddr) {
    const auto Dist = DstAddr - (SrcAddr + SrcSize);
    if (Dist <= opts::ForwardDistance) {
      double Prob = 1.0 - static_cast<double>(Dist) / opts::ForwardDistance;
      return opts::ForwardWeight * Prob * Count;
    }
    return 0;
  }
  // Backward
  const auto Dist = SrcAddr + SrcSize - DstAddr;
  if (Dist <= opts::BackwardDistance) {
    double Prob = 1.0 - static_cast<double>(Dist) / opts::BackwardDistance;
    return opts::BackwardWeight * Prob * Count;
  }
  return 0;
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
         << format(" %zu (%.2lf%%) have positive execution count\n",
                   NumHotFunctions, 100.0 * NumHotFunctions / NumFunctions);
  outs() << format("  There are %zu basic blocks;", NumBlocks)
         << format(" %zu (%.2lf%%) have positive execution count\n",
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

  outs() << "  Expected i-TLB cache hit ratio: "
         << format("%.2lf%%\n", expectedCacheHitRatio(BinaryFunctions,
                                                      BBAddr,
                                                      BBSize));

  outs() << "  TSP score: "
         << format("%.0lf\n", calcTSPScore(BinaryFunctions, BBAddr, BBSize));

  outs() << "  ExtTSP score: "
         << format("%.0lf\n", calcExtTSPScore(BinaryFunctions, BBAddr, BBSize));
}
