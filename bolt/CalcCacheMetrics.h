//===- CalcCacheMetrics.h - Interface for metrics printing of cache lines --===//
//
//                     Functions to show metrics of cache lines
//
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_CALCCACHEMETRICS_H
#define LLVM_CALCCACHEMETRICS_H

#include "BinaryFunction.h"
#include <map>

using namespace llvm;
using namespace object;
using namespace bolt;

namespace CalcCacheMetrics {
/// Calculate average number of call distance for every graph traversal.
void calcGraphDistance(
    const std::map<uint64_t, BinaryFunction> &BinaryFunctions);
}

#endif //LLVM_CALCCACHEMETRICS_H
