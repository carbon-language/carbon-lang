//===- TestLoopParametricTiling.cpp --- Parametric loop tiling pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to parametrically tile nests of standard loops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

// Extracts fixed-range loops for top-level loop nests with ranges defined in
// the pass constructor.  Assumes loops are permutable.
class SimpleParametricLoopTilingPass
    : public PassWrapper<SimpleParametricLoopTilingPass, FunctionPass> {
public:
  StringRef getArgument() const final {
    return "test-extract-fixed-outer-loops";
  }
  StringRef getDescription() const final {
    return "test application of parametric tiling to the outer loops so that "
           "the "
           "ranges of outer loops become static";
  }
  SimpleParametricLoopTilingPass() = default;
  SimpleParametricLoopTilingPass(const SimpleParametricLoopTilingPass &) {}
  explicit SimpleParametricLoopTilingPass(ArrayRef<int64_t> outerLoopSizes) {
    sizes = outerLoopSizes;
  }

  void runOnFunction() override {
    FuncOp func = getFunction();
    func.walk([this](scf::ForOp op) {
      // Ignore nested loops.
      if (op->getParentRegion()->getParentOfType<scf::ForOp>())
        return;
      extractFixedOuterLoops(op, sizes);
    });
  }

  ListOption<int64_t> sizes{
      *this, "test-outer-loop-sizes", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc(
          "fixed number of iterations that the outer loops should have")};
};
} // namespace

namespace mlir {
namespace test {
void registerSimpleParametricTilingPass() {
  PassRegistration<SimpleParametricLoopTilingPass>();
}
} // namespace test
} // namespace mlir
