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
#include <map>

namespace llvm {
namespace bolt {
namespace CacheMetrics {

/// Calculate and print various metrics related to instruction cache performance
void printAll(const std::vector<BinaryFunction *> &BinaryFunctions);

} // namespace CacheMetrics
} // namespace bolt
} // namespace llvm

#endif //LLVM_CACHEMETRICS_H
