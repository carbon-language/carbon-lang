//===- CodegenStrategy.h - Linalg programmable codegen strategy -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_CODEGENSTRATEGY_H_
#define MLIR_DIALECT_LINALG_TRANSFORMS_CODEGENSTRATEGY_H_

#include <utility>

#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {

namespace linalg {

/// Abstract Transformation class applied in a sequence that also handles state
/// through markers.
struct Transformation {
  explicit Transformation(LinalgTransformationFilter::FilterFunction f)
      : filter(std::move(f)) {}
  virtual ~Transformation() = default;
  virtual void addToPassPipeline(OpPassManager &pm,
                                 LinalgTransformationFilter m) const = 0;
  LinalgTransformationFilter::FilterFunction filter = nullptr;
};

/// Represent one application of LinalgStrategyTileAndFusePass.
struct TileAndFuse : public Transformation {
  TileAndFuse(StringRef name, linalg::LinalgTilingAndFusionOptions options,
              LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(std::move(f)), opName(name),
        options(std::move(options)) {}

  void addToPassPipeline(OpPassManager &pm,
                         LinalgTransformationFilter m) const override {
    pm.addPass(createLinalgStrategyTileAndFusePass(opName, options, m));
  }

private:
  std::string opName;
  linalg::LinalgTilingAndFusionOptions options;
};

/// Represent one application of LinalgStrategyTilePass.
struct Tile : public Transformation {
  Tile(StringRef name, linalg::LinalgTilingOptions options,
       LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(std::move(f)), opName(name),
        options(std::move(options)) {}

  void addToPassPipeline(OpPassManager &pm,
                         LinalgTransformationFilter m) const override {
    pm.addPass(createLinalgStrategyTilePass(opName, options, m));
  }

private:
  std::string opName;
  linalg::LinalgTilingOptions options;
};

/// Represent one application of LinalgStrategyPadPass.
struct Pad : public Transformation {
  Pad(StringRef name, linalg::LinalgPaddingOptions options,
      LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(std::move(f)), opName(name),
        options(std::move(options)) {}

  void addToPassPipeline(OpPassManager &pm,
                         LinalgTransformationFilter m) const override {
    pm.addPass(createLinalgStrategyPadPass(opName, options, m));
  }

private:
  std::string opName;
  linalg::LinalgPaddingOptions options;
};

/// Represent one application of createLinalgStrategyPromotePass.
struct Promote : public Transformation {
  Promote(StringRef name, linalg::LinalgPromotionOptions options,
          LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(std::move(f)), opName(name),
        options(std::move(options)) {}

  void addToPassPipeline(OpPassManager &pm,
                         LinalgTransformationFilter m) const override {
    pm.addPass(createLinalgStrategyPromotePass(opName, options, m));
  }

private:
  std::string opName;
  linalg::LinalgPromotionOptions options;
};

/// Represent one application of createLinalgStrategyGeneralizePass.
struct Generalize : public Transformation {
  explicit Generalize(StringRef name,
                      LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(std::move(f)), opName(name) {}

  void addToPassPipeline(OpPassManager &pm,
                         LinalgTransformationFilter m) const override {
    pm.addPass(createLinalgStrategyGeneralizePass(opName, m));
  }

private:
  std::string opName;
};

/// Represent one application of createLinalgStrategyInterchangePass.
struct Interchange : public Transformation {
  explicit Interchange(ArrayRef<int64_t> iteratorInterchange,
                       LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(std::move(f)),
        iteratorInterchange(iteratorInterchange.begin(),
                            iteratorInterchange.end()) {}

  void addToPassPipeline(OpPassManager &pm,
                         LinalgTransformationFilter m) const override {
    pm.addPass(createLinalgStrategyInterchangePass(iteratorInterchange, m));
  }

private:
  SmallVector<int64_t> iteratorInterchange;
};

/// Represent one application of createLinalgStrategyDecomposePass.
struct Decompose : public Transformation {
  explicit Decompose(LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(std::move(f)) {}

  void addToPassPipeline(OpPassManager &pm,
                         LinalgTransformationFilter m) const override {
    pm.addPass(createLinalgStrategyDecomposePass(m));
  }
};

/// Represent one application of createLinalgStrategyPeelPass.
struct Peel : public Transformation {
  explicit Peel(linalg::LinalgPeelOptions options,
                LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(std::move(f)), opName(), options(options) {}

  Peel(StringRef name, linalg::LinalgPeelOptions options,
       LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(std::move(f)), opName(name), options(options) {}

  void addToPassPipeline(OpPassManager &pm,
                         LinalgTransformationFilter m) const override {
    pm.addPass(createLinalgStrategyPeelPass(opName, options, m));
  }

private:
  std::string opName;
  linalg::LinalgPeelOptions options;
};

/// Represent one application of createLinalgStrategyVectorizePass.
struct Vectorize : public Transformation {
  explicit Vectorize(linalg::LinalgVectorizationOptions options,
                     LinalgTransformationFilter::FilterFunction f = nullptr,
                     bool padVectorize = false)
      : Transformation(std::move(f)), opName(), options(options),
        vectorizePadding(padVectorize) {}

  Vectorize(StringRef name, linalg::LinalgVectorizationOptions options,
            LinalgTransformationFilter::FilterFunction f = nullptr,
            bool padVectorize = false)
      : Transformation(std::move(f)), opName(name), options(options),
        vectorizePadding(padVectorize) {}

  void addToPassPipeline(OpPassManager &pm,
                         LinalgTransformationFilter m) const override {
    pm.addPass(createLinalgStrategyVectorizePass(opName, options, m,
                                                 vectorizePadding));
  }

private:
  std::string opName;
  linalg::LinalgVectorizationOptions options;
  bool vectorizePadding;
};

/// Represent one application of createLinalgStrategyLowerVectorsPass.
struct VectorLowering : public Transformation {
  explicit VectorLowering(
      linalg::LinalgVectorLoweringOptions options,
      LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(std::move(f)), options(options) {}

  void addToPassPipeline(OpPassManager &pm,
                         LinalgTransformationFilter m) const override {
    pm.addPass(createLinalgStrategyLowerVectorsPass(options, m));
  }

private:
  linalg::LinalgVectorLoweringOptions options;
};

/// Codegen strategy controls how a Linalg op is progressively lowered.
struct CodegenStrategy {
  /// Append a pattern to tile the Op `opName` and fuse its producers with
  /// tiling and fusion `options`.
  CodegenStrategy &
  tileAndFuse(StringRef opName, const LinalgTilingAndFusionOptions &options,
              const LinalgTransformationFilter::FilterFunction &f = nullptr) {
    transformationSequence.emplace_back(
        std::make_unique<TileAndFuse>(opName, options, f));
    return *this;
  }
  /// Conditionally append a pattern to tile the Op `opName` and fuse its
  /// producers with tiling and fusion `options`.
  CodegenStrategy &
  tileAndFuseIf(bool b, StringRef opName, LinalgTilingAndFusionOptions options,
                LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? tileAndFuse(opName, std::move(options), std::move(f)) : *this;
  }
  /// Append a pattern to add a level of tiling for Op `opName` with tiling
  /// `options`.
  CodegenStrategy &
  tile(StringRef opName, const linalg::LinalgTilingOptions &options,
       const LinalgTransformationFilter::FilterFunction &f = nullptr) {
    transformationSequence.emplace_back(
        std::make_unique<Tile>(opName, options, f));
    return *this;
  }
  /// Conditionally append a pattern to add a level of tiling for
  /// `LinalgOpType` with tiling `options`.
  CodegenStrategy &
  tileIf(bool b, StringRef opName, linalg::LinalgTilingOptions options,
         LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? tile(opName, std::move(options), std::move(f)) : *this;
  }
  /// Append a pattern to pad and hoist the operands of Op `opName` with padding
  /// `options`.
  CodegenStrategy &
  pad(StringRef opName, const linalg::LinalgPaddingOptions &options,
      const LinalgTransformationFilter::FilterFunction &f = nullptr) {
    transformationSequence.emplace_back(
        std::make_unique<Pad>(opName, options, f));
    return *this;
  }
  /// Conditionally append a pattern to pad and hoist the operands of Op
  /// `opName` with padding `options`.
  CodegenStrategy &
  padIf(bool b, StringRef opName, linalg::LinalgPaddingOptions options,
        LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? pad(opName, std::move(options), std::move(f)) : *this;
  }
  /// Append a pattern to add a level of promotion for `LinalgOpType` with
  /// promotion `options`.
  CodegenStrategy &
  promote(StringRef opName, const linalg::LinalgPromotionOptions &options,
          const LinalgTransformationFilter::FilterFunction &f = nullptr) {
    transformationSequence.emplace_back(
        std::make_unique<Promote>(opName, options, f));
    return *this;
  }
  /// Conditionally append a pattern to add a level of promotion for
  /// `LinalgOpType` with promotion `options`.
  CodegenStrategy &
  promoteIf(bool b, StringRef opName, linalg::LinalgPromotionOptions options,
            LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? promote(opName, std::move(options), std::move(f)) : *this;
  }
  /// Append a pattern to generalize named operations.
  CodegenStrategy &
  generalize(StringRef opName,
             const LinalgTransformationFilter::FilterFunction &f = nullptr) {
    transformationSequence.emplace_back(
        std::make_unique<Generalize>(opName, f));
    return *this;
  }
  /// Conditionally append a pattern to generalize named operations.
  CodegenStrategy &
  generalizeIf(bool b, StringRef opName,
               LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? generalize(opName, std::move(f)) : *this;
  }
  /// Append a pattern to interchange iterators.
  CodegenStrategy &
  interchange(ArrayRef<int64_t> iteratorInterchange,
              const LinalgTransformationFilter::FilterFunction &f = nullptr) {
    transformationSequence.emplace_back(
        std::make_unique<Interchange>(iteratorInterchange, f));
    return *this;
  }
  /// Conditionally append a pattern to interchange iterators.
  CodegenStrategy &
  interchangeIf(bool b, ArrayRef<int64_t> iteratorInterchange,
                LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? interchange(iteratorInterchange, std::move(f)) : *this;
  }
  /// Append patterns to decompose convolutions.
  CodegenStrategy &
  decompose(const LinalgTransformationFilter::FilterFunction &f = nullptr) {
    transformationSequence.emplace_back(std::make_unique<Decompose>(f));
    return *this;
  }
  /// Conditionally append patterns to decompose convolutions.
  CodegenStrategy &
  decomposeIf(bool b, LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? decompose(std::move(f)) : *this;
  }
  /// Append a pattern to peel 'LinalgOpType'.
  CodegenStrategy &
  peel(StringRef opName, const LinalgPeelOptions &options,
       const LinalgTransformationFilter::FilterFunction &f = nullptr) {
    transformationSequence.emplace_back(
        std::make_unique<Peel>(opName, options, f));
    return *this;
  }
  /// Conditionally append a pattern to peel 'LinalgOpType'.
  CodegenStrategy &
  peelIf(bool b, StringRef opName, const LinalgPeelOptions &options,
         LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? peel(opName, options, std::move(f)) : *this;
  }
  /// Append a pattern to rewrite `LinalgOpType` as a vector operation.
  CodegenStrategy &
  vectorize(StringRef opName,
            const LinalgTransformationFilter::FilterFunction &f = nullptr,
            bool vectorizePadding = false) {
    transformationSequence.emplace_back(std::make_unique<Vectorize>(
        opName, linalg::LinalgVectorizationOptions(), f, vectorizePadding));
    return *this;
  }
  /// Conditionally append a pattern to rewrite `LinalgOpType` as a vector
  /// operation.
  CodegenStrategy &
  vectorizeIf(bool b, StringRef opName,
              LinalgTransformationFilter::FilterFunction f = nullptr,
              bool vectorizePadding = false) {
    return b ? vectorize(opName, std::move(f), vectorizePadding) : *this;
  }
  /// Append a pattern to lower all vector operations.
  CodegenStrategy &vectorLowering(LinalgVectorLoweringOptions options) {
    transformationSequence.emplace_back(
        std::make_unique<VectorLowering>(options));
    return *this;
  }
  /// Configure the post staged-patterns global enabling passes options.
  CodegenStrategy &
  setVectorTransferToSCFOptions(LinalgEnablingOptions options) {
    linalgEnablingOptions = options;
    return *this;
  }

  /// Apply the transformation patterns in sequence with cleanup
  /// transformations interleaved.
  void configurePassPipeline(OpPassManager &pm, MLIRContext *context,
                             bool addEnablePass = true) const;

private:
  LogicalResult postPatternTransforms(Operation *func) const;

  LinalgEnablingOptions linalgEnablingOptions;
  SmallVector<std::unique_ptr<Transformation>, 4> transformationSequence;
};

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_CODEGENSTRATEGY_H_
