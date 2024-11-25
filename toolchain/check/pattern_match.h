// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_PATTERN_MATCH_H_
#define CARBON_TOOLCHAIN_CHECK_PATTERN_MATCH_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// The outputs of CalleePatternMatch.
// TODO: Rename or remove this struct.
struct ParameterBlocks {
  // The `Call` parameters of the function.
  SemIR::InstBlockId call_params_id;

  // The return slot.
  // TODO: Drop this and just use the last element of above?
  SemIR::InstId return_slot_id;
};

// TODO: Find a better place for this overview, once it has stabilized.
//
// The signature pattern of a function call is matched partially by the caller
// and partially by the callee. `ParamPattern` insts mark the boundary
// between the two: pattern insts that are descendants of a `ParamPattern`
// are matched by the callee, and pattern insts that have a `ParamPattern`
// as a descendant are matched by the caller.

// Emits the pattern-match IR for the declaration of a parameterized entity with
// the given implicit and explicit parameter patterns, and the given return slot
// pattern (any of which may be invalid if not applicable). This IR performs the
// callee side of pattern matching, starting at the `ParamPattern` insts, and
// matching them against the corresponding `Call` parameters (see
// entity_with_params_base.h for the definition of that term).
auto CalleePatternMatch(Context& context,
                        SemIR::InstBlockId implicit_param_patterns_id,
                        SemIR::InstBlockId param_patterns_id,
                        SemIR::InstId return_slot_pattern_id)
    -> ParameterBlocks;

// Emits the pattern-match IR for matching the given arguments with the given
// parameter patterns, and returns an inst block of the arguments that should
// be passed to the `Call` inst.
auto CallerPatternMatch(Context& context, SemIR::SpecificId specific_id,
                        SemIR::InstId self_pattern_id,
                        SemIR::InstBlockId param_patterns_id,
                        SemIR::InstId return_slot_pattern_id,
                        SemIR::InstId self_arg_id,
                        llvm::ArrayRef<SemIR::InstId> arg_refs,
                        SemIR::InstId return_slot_arg_id) -> SemIR::InstBlockId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_PATTERN_MATCH_H_
