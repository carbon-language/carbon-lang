//===- CacheMetrics.h - Interface for instruction cache evaluation       --===//
//
//                     Functions to show metrics of cache lines
//
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_CACHEMETRICS_H
#define LLVM_TOOLS_LLVM_BOLT_CACHEMETRICS_H

#include "BinaryFunction.h"
#include <vector>

namespace llvm {
namespace bolt {
namespace CacheMetrics {

/// Calculate various metrics related to instruction cache performance.
void printAll(const std::vector<BinaryFunction *> &BinaryFunctions);

/// Calculate Extended-TSP metric, which quantifies the expected number of
/// i-cache misses for a given pair of basic blocks. The parameters are:
/// - SrcAddr is the address of the source block;
/// - SrcSize is the size of the source block;
/// - DstAddr is the address of the destination block;
/// - Count is the number of jumps between the pair of blocks.
double extTSPScore(uint64_t SrcAddr,
                   uint64_t SrcSize,
                   uint64_t DstAddr,
                   uint64_t Count);

} // namespace CacheMetrics
} // namespace bolt
} // namespace llvm

#endif //LLVM_CACHEMETRICS_H
