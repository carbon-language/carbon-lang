//===- Passes.h - Sparse tensor pipeline entry points -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all sparse tensor pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_PIPELINES_PASSES_H_
#define MLIR_DIALECT_SPARSETENSOR_PIPELINES_PASSES_H_

#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"

using namespace mlir::detail;
using namespace llvm::cl;

namespace mlir {
namespace sparse_tensor {

/// Options for the "sparse-compiler" pipeline.  So far this only contains
/// a subset of the options that can be set for the underlying passes,
/// because it must be manually kept in sync with the tablegen files
/// for those passes.
struct SparseCompilerOptions
    : public PassPipelineOptions<SparseCompilerOptions> {
  // These options must be kept in sync with `SparsificationBase`.
  PassOptions::Option<int32_t> parallelization{
      *this, "parallelization-strategy",
      desc("Set the parallelization strategy"), init(0)};
  PassOptions::Option<int32_t> vectorization{
      *this, "vectorization-strategy", desc("Set the vectorization strategy"),
      init(0)};
  PassOptions::Option<int32_t> vectorLength{
      *this, "vl", desc("Set the vector length"), init(1)};
  PassOptions::Option<bool> enableSIMDIndex32{
      *this, "enable-simd-index32",
      desc("Enable i32 indexing into vectors (for efficiency)"), init(false)};

  /// Projects out the options for `createSparsificationPass`.
  SparsificationOptions sparsificationOptions() const {
    return SparsificationOptions(sparseParallelizationStrategy(parallelization),
                                 sparseVectorizationStrategy(vectorization),
                                 vectorLength, enableSIMDIndex32);
  }

  // These options must be kept in sync with `ConvertVectorToLLVMBase`.
  // TODO(wrengr): does `indexOptimizations` differ from `enableSIMDIndex32`?
  PassOptions::Option<bool> reassociateFPReductions{
      *this, "reassociate-fp-reductions",
      desc("Allows llvm to reassociate floating-point reductions for speed"),
      init(false)};
  PassOptions::Option<bool> indexOptimizations{
      *this, "enable-index-optimizations",
      desc("Allows compiler to assume indices fit in 32-bit if that yields "
           "faster code"),
      init(true)};
  PassOptions::Option<bool> amx{
      *this, "enable-amx",
      desc("Enables the use of AMX dialect while lowering the vector dialect."),
      init(false)};
  PassOptions::Option<bool> armNeon{*this, "enable-arm-neon",
                                    desc("Enables the use of ArmNeon dialect "
                                         "while lowering the vector dialect."),
                                    init(false)};
  PassOptions::Option<bool> armSVE{*this, "enable-arm-sve",
                                   desc("Enables the use of ArmSVE dialect "
                                        "while lowering the vector dialect."),
                                   init(false)};
  PassOptions::Option<bool> x86Vector{
      *this, "enable-x86vector",
      desc("Enables the use of X86Vector dialect while lowering the vector "
           "dialect."),
      init(false)};

  /// Projects out the options for `createConvertVectorToLLVMPass`.
  LowerVectorToLLVMOptions lowerVectorToLLVMOptions() const {
    LowerVectorToLLVMOptions opts{};
    opts.enableReassociateFPReductions(reassociateFPReductions);
    opts.enableIndexOptimizations(indexOptimizations);
    opts.enableArmNeon(armNeon);
    opts.enableArmSVE(armSVE);
    opts.enableAMX(amx);
    opts.enableX86Vector(x86Vector);
    return opts;
  }
};

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the "sparse-compiler" pipeline to the `OpPassManager`.  This
/// is the standard pipeline for taking sparsity-agnostic IR using
/// the sparse-tensor type and lowering it to LLVM IR with concrete
/// representations and algorithms for sparse tensors.
void buildSparseCompiler(OpPassManager &pm,
                         const SparseCompilerOptions &options);

/// Registers all pipelines for the `sparse_tensor` dialect.  At present,
/// this includes only "sparse-compiler".
void registerSparseTensorPipelines();

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_PIPELINES_PASSES_H_
