//===-- Transforms/IPO/PoolAllocate.h - Pool Allocation Pass -----*- C++ -*--=//
//
// This transform changes programs so that disjoint data structures are
// allocated out of different pools of memory, increasing locality and shrinking
// pointer size.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORM_IPO_POOLALLOCATE_H
#define LLVM_TRANSFORM_IPO_POOLALLOCATE_H

class Pass;
Pass *createPoolAllocatePass();

#endif
