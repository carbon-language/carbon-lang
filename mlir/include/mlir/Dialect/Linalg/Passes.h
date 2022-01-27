//===- Passes.h - Linalg pass entry points ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_PASSES_H_
#define MLIR_DIALECT_LINALG_PASSES_H_

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<Pass> createConvertElementwiseToLinalgPass();

std::unique_ptr<Pass> createLinalgFoldUnitExtentDimsPass();

std::unique_ptr<Pass> createLinalgElementwiseOpFusionPass();
std::unique_ptr<Pass> createFoldReshapeOpsByLinearizationPass();

std::unique_ptr<Pass> createLinalgNamedOpConversionPass();

std::unique_ptr<OperationPass<FuncOp>> createLinalgTilingPass(
    ArrayRef<int64_t> tileSizes = {},
    linalg::LinalgTilingLoopType loopType = linalg::LinalgTilingLoopType::Loops,
    ArrayRef<StringRef> distributionTypes = {});

std::unique_ptr<OperationPass<FuncOp>>
createLinalgPromotionPass(bool dynamicBuffers, bool useAlloca);
std::unique_ptr<OperationPass<FuncOp>> createLinalgPromotionPass();

std::unique_ptr<OperationPass<FuncOp>> createLinalgInlineScalarOperandsPass();

/// Create a pass to convert Linalg tiled loops to `scf.for` and `scf.parallel`
/// loops and memref.load/memref.store accesses.
std::unique_ptr<OperationPass<FuncOp>> createConvertLinalgTiledLoopsToSCFPass();

/// Create a pass to convert Linalg operations to scf.for loops and
/// memref.load/memref.store accesses.
std::unique_ptr<OperationPass<FuncOp>> createConvertLinalgToLoopsPass();

/// Create a pass to convert Linalg operations to scf.parallel loops and
/// memref.load/memref.store accesses.
std::unique_ptr<OperationPass<FuncOp>> createConvertLinalgToParallelLoopsPass();

/// Create a pass to convert Linalg operations to affine.for loops and
/// affine_load/affine_store accesses.
/// Placeholder for now, this is NYI.
std::unique_ptr<OperationPass<FuncOp>> createConvertLinalgToAffineLoopsPass();

/// This pass implements a cross-dialect bufferization approach and performs an
/// analysis to determine which op operands and results may be bufferized in the
/// same buffers. The analysis is performed on topologically sorted CallOp and
/// FuncOp within a module. It provides analyses and bufferization across
/// function boundaries. Within a function boundary, the analysis is performed
/// on SSA use-def chains starting from function operands that are annotated
/// with the 'inplaceable' attribute.
std::unique_ptr<Pass> createLinalgComprehensiveModuleBufferizePass();
std::unique_ptr<Pass>
createLinalgComprehensiveModuleBufferizePass(bool useLinalgCopy);

/// Create a pass to convert Linalg operations which work on tensors to use
/// buffers instead.
std::unique_ptr<OperationPass<FuncOp>> createLinalgBufferizePass();

/// Create a pass to convert named Linalg operations to Linalg generic
/// operations.
std::unique_ptr<OperationPass<FuncOp>> createLinalgGeneralizationPass();

/// Create a pass to convert Linalg operations to equivalent operations that
/// work on primitive types, if possible.
std::unique_ptr<Pass> createLinalgDetensorizePass();

//===----------------------------------------------------------------------===//
/// Linalg strategy passes.
//===----------------------------------------------------------------------===//
/// Create a LinalgStrategyTileAndFusePass.
std::unique_ptr<OperationPass<FuncOp>> createLinalgStrategyTileAndFusePass(
    StringRef opName = "", const linalg::LinalgTilingAndFusionOptions &opt = {},
    const linalg::LinalgTransformationFilter &filter =
        linalg::LinalgTransformationFilter());

/// Create a LinalgStrategyTilePass.
std::unique_ptr<OperationPass<FuncOp>> createLinalgStrategyTilePass(
    StringRef opName = "",
    const linalg::LinalgTilingOptions &opt = linalg::LinalgTilingOptions(),
    const linalg::LinalgTransformationFilter &filter =
        linalg::LinalgTransformationFilter());

/// Create a LinalgStrategyPadPass.
std::unique_ptr<OperationPass<FuncOp>> createLinalgStrategyPadPass(
    StringRef opName = "",
    const linalg::LinalgPaddingOptions &opt = linalg::LinalgPaddingOptions(),
    const linalg::LinalgTransformationFilter &filter =
        linalg::LinalgTransformationFilter());

/// Create a LinalgStrategyPromotePass.
std::unique_ptr<OperationPass<FuncOp>> createLinalgStrategyPromotePass(
    StringRef opName = "",
    const linalg::LinalgPromotionOptions &opt =
        linalg::LinalgPromotionOptions(),
    const linalg::LinalgTransformationFilter &filter =
        linalg::LinalgTransformationFilter());

/// Create a LinalgStrategyGeneralizePass.
std::unique_ptr<OperationPass<FuncOp>> createLinalgStrategyGeneralizePass(
    StringRef opName = "", const linalg::LinalgTransformationFilter &filter =
                               linalg::LinalgTransformationFilter());

/// Create a LinalgStrategyDecomposePass.
// TODO: if/when we need finer control add an `opName` parameter.
std::unique_ptr<OperationPass<FuncOp>> createLinalgStrategyDecomposePass(
    const linalg::LinalgTransformationFilter &filter =
        linalg::LinalgTransformationFilter());

/// Create a LinalgStrategyInterchangePass.
std::unique_ptr<OperationPass<FuncOp>> createLinalgStrategyInterchangePass(
    ArrayRef<int64_t> iteratorInterchange = {},
    const linalg::LinalgTransformationFilter &filter =
        linalg::LinalgTransformationFilter());

/// Create a LinalgStrategyVectorizePass.
std::unique_ptr<OperationPass<FuncOp>> createLinalgStrategyVectorizePass(
    StringRef opName = "",
    linalg::LinalgVectorizationOptions opt =
        linalg::LinalgVectorizationOptions(),
    const linalg::LinalgTransformationFilter &filter =
        linalg::LinalgTransformationFilter(),
    bool padVectorize = false);

/// Create a LinalgStrategyEnablePass.
std::unique_ptr<OperationPass<FuncOp>> createLinalgStrategyEnablePass(
    linalg::LinalgEnablingOptions opt = linalg::LinalgEnablingOptions(),
    const linalg::LinalgTransformationFilter &filter =
        linalg::LinalgTransformationFilter());

/// Create a LinalgStrategyLowerVectorsPass.
std::unique_ptr<OperationPass<FuncOp>> createLinalgStrategyLowerVectorsPass(
    linalg::LinalgVectorLoweringOptions opt =
        linalg::LinalgVectorLoweringOptions(),
    const linalg::LinalgTransformationFilter &filter =
        linalg::LinalgTransformationFilter());

/// Create a LinalgStrategyRemoveMarkersPass.
std::unique_ptr<OperationPass<FuncOp>> createLinalgStrategyRemoveMarkersPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Linalg/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_LINALG_PASSES_H_
