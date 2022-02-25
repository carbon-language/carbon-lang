//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors in the loop
// transformation library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_PASSES_H
#define MLIR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/ViewOpGraph.h"
#include "llvm/Support/Debug.h"
#include <limits>

namespace mlir {

class AffineForOp;
class GreedyRewriteConfig;

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// Creates an instance of the BufferDeallocation pass to free all allocated
/// buffers.
std::unique_ptr<Pass> createBufferDeallocationPass();

/// Creates a pass that moves allocations upwards to reduce the number of
/// required copies that are inserted during the BufferDeallocation pass.
std::unique_ptr<Pass> createBufferHoistingPass();

/// Creates a pass that moves allocations upwards out of loops. This avoids
/// reallocations inside of loops.
std::unique_ptr<Pass> createBufferLoopHoistingPass();

/// Creates a pass that promotes heap-based allocations to stack-based ones.
/// Only buffers smaller than the provided size are promoted.
/// Dynamic shaped buffers are promoted up to the given rank.
std::unique_ptr<Pass>
createPromoteBuffersToStackPass(unsigned maxAllocSizeInBytes = 1024,
                                unsigned bitwidthOfIndexType = 64,
                                unsigned maxRankOfAllocatedMemRef = 1);

/// Creates a pass that promotes heap-based allocations to stack-based ones.
/// Only buffers smaller with `isSmallAlloc(alloc) == true` are promoted.
std::unique_ptr<Pass>
createPromoteBuffersToStackPass(std::function<bool(Value)> isSmallAlloc);

/// Creates a pass that finalizes a partial bufferization by removing remaining
/// tensor_load and buffer_cast operations.
std::unique_ptr<FunctionPass> createFinalizingBufferizePass();

/// Creates a pass that converts memref function results to out-params.
std::unique_ptr<Pass> createBufferResultsToOutParamsPass();

/// Creates an instance of the Canonicalizer pass, configured with default
/// settings (which can be overridden by pass options on the command line).
std::unique_ptr<Pass> createCanonicalizerPass();

/// Creates an instance of the Canonicalizer pass with the specified config.
std::unique_ptr<Pass>
createCanonicalizerPass(const GreedyRewriteConfig &config);

/// Creates a pass to perform common sub expression elimination.
std::unique_ptr<Pass> createCSEPass();

/// Creates a loop fusion pass which fuses loops. Buffers of size less than or
/// equal to `localBufSizeThreshold` are promoted to memory space
/// `fastMemorySpace'.
std::unique_ptr<OperationPass<FuncOp>>
createLoopFusionPass(unsigned fastMemorySpace = 0,
                     uint64_t localBufSizeThreshold = 0,
                     bool maximalFusion = false);

/// Creates a loop invariant code motion pass that hoists loop invariant
/// instructions out of the loop.
std::unique_ptr<Pass> createLoopInvariantCodeMotionPass();

/// Creates a pass to pipeline explicit movement of data across levels of the
/// memory hierarchy.
std::unique_ptr<OperationPass<FuncOp>> createPipelineDataTransferPass();

/// Creates a pass that transforms perfectly nested loops with independent
/// bounds into a single loop.
std::unique_ptr<OperationPass<FuncOp>> createLoopCoalescingPass();

/// Creates a pass that transforms a single ParallelLoop over N induction
/// variables into another ParallelLoop over less than N induction variables.
std::unique_ptr<Pass> createParallelLoopCollapsingPass();

/// Creates a pass to strip debug information from a function.
std::unique_ptr<Pass> createStripDebugInfoPass();

/// Creates a pass which prints the list of ops and the number of occurrences in
/// the module.
std::unique_ptr<Pass> createPrintOpStatsPass();

/// Creates a pass which inlines calls and callable operations as defined by
/// the CallGraph.
std::unique_ptr<Pass> createInlinerPass();
/// Creates an instance of the inliner pass, and use the provided pass managers
/// when optimizing callable operations with names matching the key type.
/// Callable operations with a name not within the provided map will use the
/// default inliner pipeline during optimization.
std::unique_ptr<Pass>
createInlinerPass(llvm::StringMap<OpPassManager> opPipelines);
/// Creates an instance of the inliner pass, and use the provided pass managers
/// when optimizing callable operations with names matching the key type.
/// Callable operations with a name not within the provided map will use the
/// provided default pipeline builder.
std::unique_ptr<Pass>
createInlinerPass(llvm::StringMap<OpPassManager> opPipelines,
                  std::function<void(OpPassManager &)> defaultPipelineBuilder);

/// Creates a pass which performs sparse conditional constant propagation over
/// nested operations.
std::unique_ptr<Pass> createSCCPPass();

/// Creates a pass which delete symbol operations that are unreachable. This
/// pass may *only* be scheduled on an operation that defines a SymbolTable.
std::unique_ptr<Pass> createSymbolDCEPass();

/// Creates an interprocedural pass to normalize memrefs to have a trivial
/// (identity) layout map.
std::unique_ptr<OperationPass<ModuleOp>> createNormalizeMemRefsPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Transforms/Passes.h.inc"

} // end namespace mlir

#endif // MLIR_TRANSFORMS_PASSES_H
