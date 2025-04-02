// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_EVAL_INST_H_
#define CARBON_TOOLCHAIN_CHECK_EVAL_INST_H_

#include "toolchain/check/eval.h"
#include "toolchain/sem_ir/inst_kind.h"

namespace Carbon::Check {

// The result of constant evaluation of an instruction.
class ConstantEvalResult {
 public:
  // Produce a new constant as the result of an evaluation. The phase of the
  // produced constant must be the same as the greatest phase of the operands in
  // the evaluation. This will typically be the case if the evaluation uses all
  // of its operands.
  static auto NewSamePhase(SemIR::Inst inst) -> ConstantEvalResult {
    return ConstantEvalResult(inst, /*same_phase_as_inst=*/true);
  }

  // Produce a new constant as the result of an evaluation. The constant may
  // have any phase. Use `NewSamePhase` instead where possible, as it avoids a
  // phase recomputation.
  static auto NewAnyPhase(SemIR::Inst inst) -> ConstantEvalResult {
    return ConstantEvalResult(inst, /*same_phase_as_inst=*/false);
  }

  // Produce an existing constant as the result of an evaluation.
  static constexpr auto Existing(SemIR::ConstantId existing_id)
      -> ConstantEvalResult {
    CARBON_CHECK(existing_id.has_value());
    return ConstantEvalResult(existing_id);
  }

  // Indicates that an error was produced by evaluation.
  static const ConstantEvalResult Error;

  // Indicates that we encountered an instruction whose evaluation is
  // non-constant despite having constant operands. This should be rare;
  // usually we want to produce an error in this case.
  static const ConstantEvalResult NotConstant;

  // Indicates that we encountered an instruction for which we've not
  // implemented constant evaluation yet. Instruction is treated as not
  // constant.
  static const ConstantEvalResult TODO;

  // Returns whether the result of evaluation is that we should produce a new
  // constant described by `new_inst()` rather than an existing `ConstantId`
  // described by `existing()`.
  auto is_new() const -> bool { return !result_id_.has_value(); }

  // Returns the existing constant that this the instruction evaluates to, or
  // `None` if this is evaluation produces a new constant.
  auto existing() const -> SemIR::ConstantId { return result_id_; }

  // Returns the new constant instruction that is the result of evaluation.
  auto new_inst() const -> SemIR::Inst {
    CARBON_CHECK(is_new());
    return new_inst_;
  }

  // Whether the new constant instruction is known to have the same phase as the
  // evaluated instruction. Requires `is_new()`.
  auto same_phase_as_inst() const -> bool {
    CARBON_CHECK(is_new());
    return same_phase_as_inst_;
  }

 private:
  constexpr explicit ConstantEvalResult(SemIR::ConstantId raw_id)
      : result_id_(raw_id), same_phase_as_inst_(false) {}

  explicit ConstantEvalResult(SemIR::Inst inst, bool same_phase_as_inst)
      : result_id_(SemIR::ConstantId::None),
        new_inst_(inst),
        same_phase_as_inst_(same_phase_as_inst) {}

  SemIR::ConstantId result_id_;
  union {
    SemIR::Inst new_inst_;
  };
  bool same_phase_as_inst_;
};

constexpr ConstantEvalResult ConstantEvalResult::Error =
    Existing(SemIR::ErrorInst::SingletonConstantId);

constexpr ConstantEvalResult ConstantEvalResult::NotConstant =
    ConstantEvalResult(SemIR::ConstantId::NotConstant);

constexpr ConstantEvalResult ConstantEvalResult::TODO = NotConstant;

// Implementation details to compute the type of the `EvalConstantInst`
// functions.
namespace Internal {

// Returns whether an `EvalConstantInst` overload is expected to exist for this
// instruction constant kind.
constexpr auto ConstantKindHasEvalConstantInst(SemIR::InstConstantKind kind)
    -> bool {
  switch (kind) {
    case SemIR::InstConstantKind::Never:
    case SemIR::InstConstantKind::InstAction:
    case SemIR::InstConstantKind::WheneverPossible:
    case SemIR::InstConstantKind::Always:
    case SemIR::InstConstantKind::Unique:
      return false;

    case SemIR::InstConstantKind::Indirect:
    case SemIR::InstConstantKind::SymbolicOnly:
    case SemIR::InstConstantKind::Conditional:
      return true;
  }
}

// Given an instruction kind, determines the type that should be used to declare
// `EvalConstantInst` for that instruction.
template <typename InstT, bool HasFn, bool HasInstId>
struct FunctionTypeForEvalConstantInstImpl {
  // By default, we want no `EvalConstantInst` function at all. But we can't
  // express that, so use the type `auto () -> voic` as a placaeholder.
  using Type = auto() -> void;
};
template <typename InstT>
struct FunctionTypeForEvalConstantInstImpl<InstT, true, false> {
  // Can be evaluated, evaluation doesn't need InstId.
  using Type = auto(Context& context, InstT inst) -> ConstantEvalResult;
};
template <typename InstT>
struct FunctionTypeForEvalConstantInstImpl<InstT, true, true> {
  // Can be evaluated, evaluation needs InstId.
  using Type = auto(Context& context, SemIR::InstId inst_id, InstT inst)
      -> ConstantEvalResult;
};
template <typename InstT>
using FunctionTypeForEvalConstantInst =
    typename FunctionTypeForEvalConstantInstImpl<
        InstT, ConstantKindHasEvalConstantInst(InstT::Kind.constant_kind()),
        InstT::Kind.constant_needs_inst_id()>::Type;

}  // namespace Internal

// Explicitly delete the overload generated for non-evaluatable instructions.
// These all produce the same signature, so we only need to delete it once.
auto EvalConstantInst() -> void = delete;

// `EvalConstantInst` evaluates an instruction whose operands are all constant,
// in a context unrelated to the enclosing evaluation. The function is given the
// instruction after its operands, including its type, are replaced by their
// evaluated value, and returns a `ConstantEvalResult` describing the result of
// evaluating the instruction.
//
// An overload is defined for each type whose constant kind is one of the
// following:
//
// - InstConstantKind::Indirect
// - InstConstantKind::SymbolicOnly
// - InstConstantKind::Conditional
//
// ... except for cases where the result of evaluation depends on the evaluation
// context itself. Those cases are handled by explicit specialization of
// `TryEvalTypedInst` in `eval.cpp` instead.
//
// The signature of an overload is
//
//   auto EvalConstantInst(Context& context, SemIR::InstId inst_id, InstT inst)
//       -> ConstantEvalResult;
//
// if `InstT::Kind.constant_needs_inst_id()` is true, and
//
//   auto EvalConstantInst(Context& context, InstT inst) -> ConstantEvalResult;
//
// otherwise.
//
// Overloads are *declared* for all types, because there isn't a good way to
// declare only the overloads we want here without duplicating the list of
// types. Missing overloads will be diagnosed when linking. Excess overloads
// map to a deleted signature to prevent accidental calls.
#define CARBON_SEM_IR_INST_KIND(Kind) \
  Internal::FunctionTypeForEvalConstantInst<SemIR::Kind> EvalConstantInst;
#include "toolchain/sem_ir/inst_kind.def"

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_EVAL_INST_H_
