//===- VPlanTransforms.h - Utility VPlan to VPlan transforms --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides utility VPlan to VPlan transformations.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLANTRANSFORMS_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLANTRANSFORMS_H

#include "VPlan.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Transforms/Vectorize/LoopVectorizationLegality.h"

namespace llvm {

class InductionDescriptor;
class Instruction;
class PHINode;
class ScalarEvolution;

struct VPlanTransforms {
  /// Replaces the VPInstructions in \p Plan with corresponding
  /// widen recipes.
  static void
  VPInstructionsToVPRecipes(Loop *OrigLoop, VPlanPtr &Plan,
                            function_ref<const InductionDescriptor *(PHINode *)>
                                GetIntOrFpInductionDescriptor,
                            SmallPtrSetImpl<Instruction *> &DeadInstructions,
                            ScalarEvolution &SE);

  static bool sinkScalarOperands(VPlan &Plan);

  static bool mergeReplicateRegions(VPlan &Plan);

  /// Remove redundant casts of inductions.
  ///
  /// Such redundant casts are casts of induction variables that can be ignored,
  /// because we already proved that the casted phi is equal to the uncasted phi
  /// in the vectorized loop. There is no need to vectorize the cast - the same
  /// value can be used for both the phi and casts in the vector loop.
  static void removeRedundantInductionCasts(VPlan &Plan);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLANTRANSFORMS_H
