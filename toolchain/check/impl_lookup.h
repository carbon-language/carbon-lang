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
// - A concrete value. Lookup found a concrete impl declaration that can be
//   used definitively.
// - A symbolic value. Lookup found an impl but it is not returned since lookup
//   will need to be done again with a more specific query to look for
//   specializations.
class [[nodiscard]] EvalImplLookupResult {
 public:
  static auto MakeNone() -> EvalImplLookupResult {
    return EvalImplLookupResult(SemIR::InstId::None);
  }
  static auto MakeFinal(SemIR::InstId inst_id) -> EvalImplLookupResult {
    return EvalImplLookupResult(inst_id);
  }
  static auto MakeNonFinalImpl() -> EvalImplLookupResult {
    return EvalImplLookupResult(SemIR::FacetTypeId::None);
  }
  static auto MakeNonFinalFacetValue(SemIR::FacetTypeId facet_type_id)
      -> EvalImplLookupResult {
    CARBON_CHECK(facet_type_id.has_value());
    return EvalImplLookupResult(facet_type_id);
  }

  // True if a concrete impl witness was found or a non-final impl. In the
  // latter case the InstId of the impl's witness is not returned, only the fact
  // that it exists, and possibly a `FacetTypeId` if the witness was obtained
  // from a facet value.
  auto has_value() const -> bool {
    return std::holds_alternative<SemIR::FacetTypeId>(result_) ||
           std::get<SemIR::InstId>(result_).has_value();
  }

  // True if there is a concrete witness in the result. If false, and
  // `has_value()` is true, it means a non-final impl was found and a further
  // more specific query will need to be done.
  auto has_concrete_value() const -> bool {
    const auto* inst_id = std::get_if<SemIR::InstId>(&result_);
    return inst_id && inst_id->has_value();
  }

  // Only valid if `has_concrete_value()` is true. Returns the witness id for
  // the found impl declaration, or None if `has_value()` is false.
  auto concrete_witness() const -> SemIR::InstId {
    return std::get<SemIR::InstId>(result_);
  }

  // Only valid if `has_concrete_value()` is false. Returns the FacetTypeId for
  // the facet from which a non-concrete witness was found, if any.
  auto facet_value_type() const -> SemIR::FacetTypeId {
    return std::get<SemIR::FacetTypeId>(result_);
  }

 private:
  explicit EvalImplLookupResult(SemIR::InstId inst_id) : result_(inst_id) {}
  explicit EvalImplLookupResult(SemIR::FacetTypeId facet_type_id)
      : result_(facet_type_id) {}

  // - If FacetTypeId is active, it means a non-concrete witness was found.
  // - If InstId is active but None, is means no witness was found.
  // - If InstId is active and not None, it means a concrete witness was found.
  std::variant<SemIR::InstId, SemIR::FacetTypeId> result_;
};

// Looks for a witness instruction of an impl declaration for a query consisting
// of a type value or facet value, and a single interface. This is for eval to
// execute lookup via the LookupImplWitness instruction. It does not consider
// the self facet value for finding a witness, since LookupImplWitness() would
// have found that and not caused us to defer lookup to here.
//
// Returns with the lookup result the canonicalized query self instruction,
// which is to be stored in the `LookupImplWitness` instruction to make it
// canonical.
auto EvalLookupSingleImplWitness(
    Context& context, SemIR::LocId loc_id,
    SemIR::InstId non_canonical_query_self_inst_id,
    SemIR::SpecificInterface query_specific_interface,
    bool poison_concrete_results)
    -> std::pair<EvalImplLookupResult, SemIR::InstId>;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPL_LOOKUP_H_
