//===- TestLinalgCodegenStrategy.cpp - Test Linalg codegen strategy -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing the Linalg codegen strategy.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct TestLinalgCodegenStrategy
    : public PassWrapper<TestLinalgCodegenStrategy, FunctionPass> {
  StringRef getArgument() const final { return "test-linalg-codegen-strategy"; }
  StringRef getDescription() const final {
    return "Test Linalg Codegen Strategy.";
  }
  TestLinalgCodegenStrategy() = default;
  TestLinalgCodegenStrategy(const TestLinalgCodegenStrategy &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<AffineDialect,
                    gpu::GPUDialect,
                    linalg::LinalgDialect,
                    memref::MemRefDialect,
                    scf::SCFDialect,
                    StandardOpsDialect,
                    vector::VectorDialect>();
    // clang-format on
  }

  template <typename LinalgNamedOp>
  void applyStrategyToNamedLinalgOp();

  void runOnFunction() override;

  template <typename OpType>
  void runStrategy(LinalgTilingOptions tilingOptions,
                   LinalgTilingOptions registerTilingOptions,
                   vector::VectorContractLowering vectorContractLowering,
                   vector::VectorTransferSplit vectorTransferSplit);

  ListOption<int64_t> tileSizes{*this, "tile-sizes",
                                llvm::cl::MiscFlags::CommaSeparated,
                                llvm::cl::desc("Specifies the tile sizes.")};
  ListOption<unsigned> tileInterchange{
      *this, "tile-interchange", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc("Specifies the tile interchange.")};

  Option<bool> promote{
      *this, "promote",
      llvm::cl::desc("Promote the tile into a small aligned memory buffer."),
      llvm::cl::init(false)};
  Option<bool> promoteFullTile{
      *this, "promote-full-tile-pad",
      llvm::cl::desc("Pad the small aligned memory buffer to the tile sizes."),
      llvm::cl::init(false)};
  ListOption<int64_t> registerTileSizes{
      *this, "register-tile-sizes", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc(
          "Specifies the size of the register tile that will be used "
          " to vectorize")};
  Option<bool> registerPromote{
      *this, "register-promote",
      llvm::cl::desc(
          "Promote the register tile into a small aligned memory buffer."),
      llvm::cl::init(false)};
  Option<bool> registerPromoteFullTile{
      *this, "register-promote-full-tile-pad",
      llvm::cl::desc("Pad the small aligned memory buffer to the tile sizes."),
      llvm::cl::init(false)};
  Option<bool> vectorize{
      *this, "vectorize",
      llvm::cl::desc("Rewrite the linalg op as a vector operation."),
      llvm::cl::init(false)};
  Option<std::string> splitVectorTransfersTo{
      *this, "split-transfers",
      llvm::cl::desc(
          "Split vector transfers between slow (masked) and fast "
          "(unmasked) variants. Possible options are:\n"
          "\tnone: keep unsplit vector.transfer and pay the full price\n"
          "\tlinalg-copy: use linalg.fill + linalg.copy for the slow path\n"
          "\tvector-transfers: use extra small unmasked vector.transfer for"
          " the slow path\n"),
      llvm::cl::init("none")};
  Option<std::string> vectorizeContractionTo{
      *this, "vectorize-contraction-to",
      llvm::cl::desc("the type of vector op to use for linalg contractions"),
      llvm::cl::init("outerproduct")};
  Option<bool> unrollVectorTransfers{
      *this, "unroll-vector-transfers",
      llvm::cl::desc("Enable full unrolling of vector.transfer operations"),
      llvm::cl::init(false)};
  Option<std::string> anchorOpName{
      *this, "anchor-op",
      llvm::cl::desc(
          "Which single linalg op is the anchor for the codegen strategy to "
          "latch on:\n"
          "\tlinalg.matmul: anchor on linalg.matmul\n"
          "\tlinalg.matmul_column_major: anchor on linalg.matmul_column_major\n"
          "\tlinalg.copy: anchor on linalg.copy\n"
          "\tlinalg.fill: anchor on linalg.fill\n"),
      llvm::cl::init("")};
  Option<std::string> anchorFuncOpName{
      *this, "anchor-func",
      llvm::cl::desc(
          "Which single func op is the anchor for the codegen strategy to "
          "latch on."),
      llvm::cl::init("")};
};

template <>
void TestLinalgCodegenStrategy::runStrategy<LinalgOp>(
    LinalgTilingOptions tilingOptions,
    LinalgTilingOptions registerTilingOptions,
    vector::VectorContractLowering vectorContractLowering,
    vector::VectorTransferSplit vectorTransferSplit) {
  assert(!anchorOpName.empty());
  CodegenStrategy strategy;
  strategy.tileIf<LinalgOp>(!tileSizes.empty(), anchorOpName, tilingOptions)
      .promoteIf<LinalgOp>(promote, anchorOpName,
                           LinalgPromotionOptions()
                               .setAlignment(16)
                               .setUseFullTileBuffersByDefault(promoteFullTile))
      .tileIf<LinalgOp>(!registerTileSizes.empty(), anchorOpName,
                        registerTilingOptions)
      .promoteIf<LinalgOp>(
          registerPromote, anchorOpName,
          LinalgPromotionOptions()
              .setAlignment(16)
              .setUseFullTileBuffersByDefault(registerPromoteFullTile))
      .vectorizeIf(vectorize, anchorOpName)
      .setVectorTransformsOptions(
          vector::VectorTransformsOptions()
              .setVectorTransformsOptions(vectorContractLowering)
              .setVectorTransferSplit(vectorTransferSplit))
      .setVectorTransferToSCFOptions(
          VectorTransferToSCFOptions().setUnroll(unrollVectorTransfers));
  strategy.transform(getFunction());
}

template <typename OpType>
void TestLinalgCodegenStrategy::runStrategy(
    LinalgTilingOptions tilingOptions,
    LinalgTilingOptions registerTilingOptions,
    vector::VectorContractLowering vectorContractLowering,
    vector::VectorTransferSplit vectorTransferSplit) {
  CodegenStrategy strategy;
  strategy.tileIf<OpType>(!tileSizes.empty(), tilingOptions)
      .template promoteIf<OpType>(
          promote, LinalgPromotionOptions()
                       .setAlignment(16)
                       .setUseFullTileBuffersByDefault(promoteFullTile))
      .template tileIf<OpType>(!registerTileSizes.empty(),
                               registerTilingOptions)
      .template promoteIf<OpType>(
          registerPromote,
          LinalgPromotionOptions()
              .setAlignment(16)
              .setUseFullTileBuffersByDefault(registerPromoteFullTile))
      .template vectorizeIf<OpType>(vectorize)
      .setVectorTransformsOptions(
          vector::VectorTransformsOptions()
              .setVectorTransformsOptions(vectorContractLowering)
              .setVectorTransferSplit(vectorTransferSplit))
      .setVectorTransferToSCFOptions(
          VectorTransferToSCFOptions().setUnroll(unrollVectorTransfers));
  strategy.transform(getFunction());
}
} // end anonymous namespace

/// Apply transformations specified as patterns.
void TestLinalgCodegenStrategy::runOnFunction() {
  if (!anchorFuncOpName.empty() && anchorFuncOpName != getFunction().getName())
    return;

  LinalgTilingOptions tilingOptions;
  if (!tileSizes.empty())
    tilingOptions = tilingOptions.setTileSizes(tileSizes);
  if (!tileInterchange.empty())
    tilingOptions = tilingOptions.setInterchange(tileInterchange);

  LinalgTilingOptions registerTilingOptions;
  if (!registerTileSizes.empty())
    registerTilingOptions =
        registerTilingOptions.setTileSizes(registerTileSizes);

  vector::VectorContractLowering vectorContractLowering =
      llvm::StringSwitch<vector::VectorContractLowering>(
          vectorizeContractionTo.getValue())
          .Case("matrixintrinsics", vector::VectorContractLowering::Matmul)
          .Case("dot", vector::VectorContractLowering::Dot)
          .Case("outerproduct", vector::VectorContractLowering::OuterProduct)
          .Default(vector::VectorContractLowering::OuterProduct);
  vector::VectorTransferSplit vectorTransferSplit =
      llvm::StringSwitch<vector::VectorTransferSplit>(
          splitVectorTransfersTo.getValue())
          .Case("none", vector::VectorTransferSplit::None)
          .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
          .Case("vector-transfers", vector::VectorTransferSplit::VectorTransfer)
          .Default(vector::VectorTransferSplit::None);

  // If no anchorOpNameis specified, just test that strategy applies properly to
  // linalg::MatmulOp.
  if (anchorOpName.empty())
    runStrategy<linalg::MatmulOp>(tilingOptions, registerTilingOptions,
                                  vectorContractLowering, vectorTransferSplit);
  else
    runStrategy<LinalgOp>(tilingOptions, registerTilingOptions,
                          vectorContractLowering, vectorTransferSplit);
}

namespace mlir {
namespace test {
void registerTestLinalgCodegenStrategy() {
  PassRegistration<TestLinalgCodegenStrategy>();
}
} // namespace test
} // namespace mlir
