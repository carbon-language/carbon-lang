// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_PATTERN_MATCH_H_
#define CARBON_TOOLCHAIN_CHECK_PATTERN_MATCH_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// The pattern-match counterparts of the patterns passed to CalleePatternMatch.
struct ParameterBlocks {
  // The implicit parameter list.
  SemIR::InstBlockId implicit_params_id;

  // The explicit parameter list.
  SemIR::InstBlockId params_id;

  // The return slot.
  SemIR::InstId return_slot_id;
};

// TODO: Find a better place for this overview, once it has stabilized.
//
// The signature pattern of a function call is matched partially by the caller
// and partially by the callee. `ParamPattern` insts mark the boundary
// between the two: pattern insts that are descendants of a `ParamPattern`
// are matched by the callee, and pattern insts that have a `ParamPattern`
// as a descendant are matched by the caller.
//
// "Calling convention arguments" are the values actually passed from caller to
// callee at the semantic IR level, and "calling convention parameters" are
// the corresponding semantic placeholders that they bind to.

// Emits the pattern-match IR for the declaration of a parameterized entity with
// the given implicit and explicit parameter patterns, and the given return slot
// pattern (any of which may be invalid if not applicable). This IR performs the
// callee side of pattern matching, starting at the `ParamPattern` insts, and
// matching them against the corresponding calling-convention parameters.
auto CalleePatternMatch(Context& context,
                        SemIR::InstBlockId implicit_param_patterns_id,
                        SemIR::InstBlockId param_patterns_id,
                        SemIR::InstId return_slot_pattern_id)
    -> ParameterBlocks;

// Emits the pattern-match IR for matching the given arguments with the given
// parameter patterns, and returns an inst block with one inst for each
// calling convention argument. This IR performs the caller side of pattern
// matching.
auto CallerPatternMatch(Context& context, SemIR::SpecificId specific_id,
                        SemIR::InstId self_pattern_id,
                        SemIR::InstBlockId param_patterns_id,
                        SemIR::InstId return_slot_pattern_id,
                        SemIR::InstId self_arg_id,
                        llvm::ArrayRef<SemIR::InstId> arg_refs,
                        SemIR::InstId return_slot_arg_id) -> SemIR::InstBlockId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_PATTERN_MATCH_H_
