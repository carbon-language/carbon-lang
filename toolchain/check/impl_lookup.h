// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPL_LOOKUP_H_
#define CARBON_TOOLCHAIN_CHECK_IMPL_LOOKUP_H_

#include <variant>

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Looks up the witnesses to use for a type value or facet value, and a facet
// type naming a set of interfaces required to be implemented for that type, as
// well as possible constraints on those interfaces.
//
// N.B. In the future, `TypeType` will become a facet type, at which point type
// values will also be facet values.
//
// The return value is one of:
// - An InstBlockId value, containing an `ImplWitness` instruction for each
//   required interface in the `query_facet_type_const_id`. This verifies the
//   facet type is satisfied for the type in `type_const_id`, and provides a
//   witness for accessing the impl of each interface.
//
// - `InstBlockId::None`, indicating lookup failed for at least one required
//   interface in the `query_facet_type_const_id`. The facet type is not
//   satisfied for the type in `type_const_id`. This represents lookup failure,
//   but is not an error, so no diagnostic is emitted.
//
// - An error value, indicating the program is invalid and a diagonstic has been
//   produced, either in this function or before.
auto LookupImplWitness(Context& context, SemIR::LocId loc_id,
                       SemIR::ConstantId query_self_const_id,
                       SemIR::ConstantId query_facet_type_const_id)
    -> SemIR::InstBlockIdOrError;

// Returns whether the query matches against the given impl. This is like a
// `LookupImplWitness` operation but for a single interface, and against only
// the single impl.
auto LookupMatchesImpl(Context& context, SemIR::LocId loc_id,
                       SemIR::ConstantId query_self_const_id,
                       SemIR::SpecificInterface query_specific_interface,
                       SemIR::ImplId target_impl) -> bool;

// The result of EvalLookupSingleImplWitness(). It can be one of:
// - No value. Lookup failed to find an impl declaration.
// - An effectively final value. Lookup found either a concrete impl or a
//   `final` impl declaration; both can be used definitely. A witness is
//   available.
// - A non-`final` symbolic value. Lookup found an impl, but it is not returned
//   since lookup will need to be done again with a more specific query to look
//   for specializations.
class [[nodiscard]] EvalImplLookupResult {
 public:
  static auto MakeNone() -> EvalImplLookupResult {
    return EvalImplLookupResult(FoundNone());
  }
  static auto MakeFinal(SemIR::InstId inst_id) -> EvalImplLookupResult {
    return EvalImplLookupResult(inst_id);
  }
  static auto MakeNonFinal() -> EvalImplLookupResult {
    return EvalImplLookupResult(FoundNonFinalImpl());
  }

  // True if an impl declaration was found, either effectively final or
  // symbolic.
  auto has_value() const -> bool {
    return !std::holds_alternative<FoundNone>(value_);
  }

  // True if there is an effectively final witness in the result. If false, and
  // `has_value()` is true, it means an impl was found that's not effectively
  // final, and a further more specific query will need to be done.
  auto has_final_value() const -> bool {
    return std::holds_alternative<SemIR::InstId>(value_);
  }

  // Returns the witness id for an effectively final value's impl declaration.
  // Only valid to call if `has_final_value` is true.
  auto final_witness() const -> SemIR::InstId {
    return std::get<SemIR::InstId>(value_);
  }

 private:
  struct FoundNone {};
  struct FoundNonFinalImpl {};
  using Value = std::variant<SemIR::InstId, FoundNone, FoundNonFinalImpl>;

  explicit EvalImplLookupResult(Value value) : value_(value) {}

  Value value_;
};

// Looks for a witness instruction of an impl declaration for a query consisting
// of a type value or facet value, and a single interface. This is for eval to
// execute lookup via the LookupImplWitness instruction. It does not consider
// the self facet value for finding a witness, since LookupImplWitness() would
// have found that and not caused us to defer lookup to here.
//
// `poison_final_results` poisons lookup results which are effectively final,
// preventing overlapping final impls.
auto EvalLookupSingleImplWitness(Context& context, SemIR::LocId loc_id,
                                 SemIR::LookupImplWitness eval_query,
                                 SemIR::InstId self_facet_value_inst_id,
                                 bool poison_final_results)
    -> EvalImplLookupResult;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPL_LOOKUP_H_
