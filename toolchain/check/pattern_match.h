// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_PATTERN_MATCH_H_
#define CARBON_TOOLCHAIN_CHECK_PATTERN_MATCH_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// TODO: Find a better place for this overview, once it has stabilized.
//
// The signature pattern of a function call is matched partially by the caller
// and partially by the callee. `ParamPattern` insts mark the boundary
// between the two: pattern insts that are descendants of a `ParamPattern`
// are matched by the callee, and pattern insts that have a `ParamPattern`
// as a descendant are matched by the caller.

// Return type for CalleePatternMatch.
struct CalleePatternMatchResults {
  SemIR::InstBlockId call_param_patterns_id;
  SemIR::InstBlockId call_params_id;

  SemIR::Function::CallParamIndexRanges param_ranges;
};

// Emits the pattern-match IR for the declaration of a parameterized entity with
// the given implicit and explicit parameter patterns, and the given return
// patterns (any of which may be `None` if not applicable). This IR performs the
// callee side of pattern matching, starting at the `ParamPattern` insts, and
// matching them against the corresponding `Call` parameters (see
// entity_with_params_base.h for the definition of that term).
// Returns the IDs of inst blocks consisting of references to the `Call`
// parameter patterns and `Call` parameters of the function, as well as
// the implicit, explicit, and return index ranges of those blocks.
auto CalleePatternMatch(Context& context,
                        SemIR::InstBlockId implicit_param_patterns_id,
                        SemIR::InstBlockId param_patterns_id,
                        SemIR::InstBlockId return_patterns_id)
    -> CalleePatternMatchResults;

// Return type for ThunkPatternMatch.
struct ThunkPatternMatchResults {
  // The syntactic argument list. If `self_pattern_id` is not `None`, the first
  // element will be the corresponding argument.
  llvm::SmallVector<SemIR::InstId> syntactic_args;

  // The trailing elements of `outer_call_args` that were not used in
  // `syntactic_args`. These presumably represent the output arguments for the
  // return.
  llvm::ArrayRef<SemIR::InstId> ignored_call_args;
};

// Given the `Call` arguments for the outer part of a thunked function call,
// computes the corresponding syntactic argument list, suitable for passing to
// the inner part of the thunked function call.
auto ThunkPatternMatch(Context& context, SemIR::InstId self_pattern_id,
                       llvm::ArrayRef<SemIR::InstId> param_pattern_ids,
                       llvm::ArrayRef<SemIR::InstId> outer_call_args)
    -> ThunkPatternMatchResults;

// Emits the pattern-match IR for matching the given arguments with the given
// parameter patterns, and returns an inst block of the arguments that should
// be passed to the `Call` inst. `is_operator_syntax` indicates that this call
// was generated from an operator rather than from function call syntax, so
// arguments to `ref` parameters aren't required to have `ref` tags.
auto CallerPatternMatch(Context& context, SemIR::SpecificId specific_id,
                        SemIR::InstId self_pattern_id,
                        SemIR::InstBlockId param_patterns_id,
                        SemIR::InstBlockId return_patterns_id,
                        SemIR::InstId self_arg_id,
                        llvm::ArrayRef<SemIR::InstId> arg_refs,
                        llvm::ArrayRef<SemIR::InstId> return_arg_ids,
                        bool is_operator_syntax) -> SemIR::InstBlockId;

// Emits the pattern-match IR for a local pattern matching operation with the
// given pattern and scrutinee.
auto LocalPatternMatch(Context& context, SemIR::InstId pattern_id,
                       SemIR::InstId scrutinee_id) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_PATTERN_MATCH_H_
