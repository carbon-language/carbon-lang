//===- ReconcileUnrealizedCasts.h - Pass entrypoint -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_RECONCILEUNREALIZEDCASTS_RECONCILEUNREALIZEDCASTS_H_
#define MLIR_CONVERSION_RECONCILEUNREALIZEDCASTS_RECONCILEUNREALIZEDCASTS_H_

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;

/// Creates a pass that eliminates noop `unrealized_conversion_cast` operation
/// sequences.
std::unique_ptr<Pass> createReconcileUnrealizedCastsPass();

/// Populates `patterns` with rewrite patterns that eliminate noop
/// `unrealized_conversion_cast` operation sequences.
void populateReconcileUnrealizedCastsPatterns(RewritePatternSet &patterns);
} // namespace mlir

#endif // MLIR_CONVERSION_RECONCILEUNREALIZEDCASTS_RECONCILEUNREALIZEDCASTS_H_
