// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/facet_type.h"

#include <compare>

#include "llvm/ADT/ArrayRef.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/subst.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto FacetTypeFromInterface(Context& context, SemIR::InterfaceId interface_id,
                            SemIR::SpecificId specific_id) -> SemIR::FacetType {
  SemIR::FacetTypeId facet_type_id = context.facet_types().Add(
      SemIR::FacetTypeInfo{.extend_constraints = {{interface_id, specific_id}},
                           .other_requirements = false});
  return {.type_id = SemIR::TypeType::TypeId, .facet_type_id = facet_type_id};
}

// Returns whether the `LookupImplWitness` of `witness_id` matches `interface`.
static auto WitnessQueryMatchesInterface(
    Context& context, SemIR::InstId witness_id,
    const SemIR::SpecificInterface& interface) -> bool {
  auto lookup = context.insts().GetAs<SemIR::LookupImplWitness>(witness_id);
  return interface ==
         context.specific_interfaces().Get(lookup.query_specific_interface_id);
}

static auto IncompleteFacetTypeDiagnosticBuilder(
    Context& context, SemIR::LocId loc_id, SemIR::TypeInstId facet_type_inst_id,
    bool is_definition) -> DiagnosticBuilder {
  if (is_definition) {
    CARBON_DIAGNOSTIC(ImplAsIncompleteFacetTypeDefinition, Error,
                      "definition of impl as incomplete facet type {0}",
                      InstIdAsType);
    return context.emitter().Build(loc_id, ImplAsIncompleteFacetTypeDefinition,
                                   facet_type_inst_id);
  } else {
    CARBON_DIAGNOSTIC(
        ImplAsIncompleteFacetTypeRewrites, Error,
        "declaration of impl as incomplete facet type {0} with rewrites",
        InstIdAsType);
    return context.emitter().Build(loc_id, ImplAsIncompleteFacetTypeRewrites,
                                   facet_type_inst_id);
  }
}

auto InitialFacetTypeImplWitness(
    Context& context, SemIR::LocId witness_loc_id,
    SemIR::TypeInstId facet_type_inst_id, SemIR::TypeInstId self_type_inst_id,
    const SemIR::SpecificInterface& interface_to_witness,
    SemIR::SpecificId self_specific_id, bool is_definition) -> SemIR::InstId {
  // TODO: Finish facet type resolution. This code currently only handles
  // rewrite constraints that set associated constants to a concrete value.
  // Need logic to topologically sort rewrites to respect dependencies, and
  // afterwards reject duplicates that are not identical.

  auto facet_type_id =
      context.types().GetTypeIdForTypeInstId(facet_type_inst_id);
  CARBON_CHECK(facet_type_id != SemIR::ErrorInst::TypeId);
  auto facet_type = context.types().GetAs<SemIR::FacetType>(facet_type_id);
  // TODO: This is currently a copy because I'm not sure whether anything could
  // cause the facet type store to resize before we are done with it.
  auto facet_type_info = context.facet_types().Get(facet_type.facet_type_id);

  if (!is_definition && facet_type_info.rewrite_constraints.empty()) {
    auto witness_table_inst_id = AddInst<SemIR::ImplWitnessTable>(
        context, witness_loc_id,
        {.elements_id = context.inst_blocks().AddPlaceholder(),
         .impl_id = SemIR::ImplId::None});
    return AddInst<SemIR::ImplWitness>(
        context, witness_loc_id,
        {.type_id = GetSingletonType(context, SemIR::WitnessType::TypeInstId),
         .witness_table_id = witness_table_inst_id,
         .specific_id = self_specific_id});
  }

  if (!RequireCompleteType(
          context, facet_type_id, SemIR::LocId(facet_type_inst_id), [&] {
            return IncompleteFacetTypeDiagnosticBuilder(
                context, witness_loc_id, facet_type_inst_id, is_definition);
          })) {
    return SemIR::ErrorInst::InstId;
  }

  const auto& interface =
      context.interfaces().Get(interface_to_witness.interface_id);
  auto assoc_entities =
      context.inst_blocks().Get(interface.associated_entities_id);
  // TODO: When this function is used for things other than just impls, may want
  // to only load the specific associated entities that are mentioned in rewrite
  // rules.
  for (auto decl_id : assoc_entities) {
    LoadImportRef(context, decl_id);
  }

  SemIR::InstId witness_inst_id = SemIR::InstId::None;
  llvm::MutableArrayRef<SemIR::InstId> table;
  {
    auto elements_id =
        context.inst_blocks().AddUninitialized(assoc_entities.size());
    table = context.inst_blocks().GetMutable(elements_id);
    for (auto& uninit : table) {
      uninit = SemIR::ImplWitnessTablePlaceholder::TypeInstId;
    }

    auto witness_table_inst_id = AddInst<SemIR::ImplWitnessTable>(
        context, witness_loc_id,
        {.elements_id = elements_id, .impl_id = SemIR::ImplId::None});

    witness_inst_id = AddInst<SemIR::ImplWitness>(
        context, witness_loc_id,
        {.type_id = GetSingletonType(context, SemIR::WitnessType::TypeInstId),
         .witness_table_id = witness_table_inst_id,
         .specific_id = self_specific_id});
  }

  for (auto rewrite : facet_type_info.rewrite_constraints) {
    auto access =
        context.insts().GetAs<SemIR::ImplWitnessAccess>(rewrite.lhs_id);
    if (!WitnessQueryMatchesInterface(context, access.witness_id,
                                      interface_to_witness)) {
      continue;
    }
    auto& table_entry = table[access.index.index];
    if (table_entry == SemIR::ErrorInst::InstId) {
      // Don't overwrite an error value. This prioritizes not generating
      // multiple errors for one associated constant over picking a value
      // for it to use to attempt recovery.
      continue;
    }
    auto rewrite_inst_id = rewrite.rhs_id;
    if (rewrite_inst_id == SemIR::ErrorInst::InstId) {
      table_entry = SemIR::ErrorInst::InstId;
      continue;
    }

    auto decl_id = context.constant_values().GetConstantInstId(
        assoc_entities[access.index.index]);
    CARBON_CHECK(decl_id.has_value(), "Non-constant associated entity");
    if (decl_id == SemIR::ErrorInst::InstId) {
      table_entry = SemIR::ErrorInst::InstId;
      continue;
    }

    auto assoc_constant_decl =
        context.insts().TryGetAs<SemIR::AssociatedConstantDecl>(decl_id);
    if (!assoc_constant_decl) {
      auto type_id = context.insts().Get(decl_id).type_id();
      auto type_inst = context.types().GetAsInst(type_id);
      auto fn_type = type_inst.As<SemIR::FunctionType>();
      const auto& fn = context.functions().Get(fn_type.function_id);
      CARBON_DIAGNOSTIC(RewriteForAssociatedFunction, Error,
                        "rewrite specified for associated function {0}",
                        SemIR::NameId);
      context.emitter().Emit(facet_type_inst_id, RewriteForAssociatedFunction,
                             fn.name_id);
      table_entry = SemIR::ErrorInst::InstId;
      continue;
    }

    // FacetTypes resolution disallows two rewrites to the same associated
    // constant, so we won't ever have a facet write twice to the same position
    // in the witness table.
    CARBON_CHECK(table_entry == SemIR::ImplWitnessTablePlaceholder::TypeInstId);

    // If the associated constant has a symbolic type, convert the rewrite
    // value to that type now we know the value of `Self`.
    SemIR::TypeId assoc_const_type_id = assoc_constant_decl->type_id;
    if (assoc_const_type_id.is_symbolic()) {
      // Get the type of the associated constant in this interface with this
      // value for `Self`.
      assoc_const_type_id = GetTypeForSpecificAssociatedEntity(
          context, SemIR::LocId(facet_type_inst_id),
          interface_to_witness.specific_id, decl_id,
          context.types().GetTypeIdForTypeInstId(self_type_inst_id),
          witness_inst_id);
      // Perform the conversion of the value to the type. We skipped this when
      // forming the facet type because the type of the associated constant
      // was symbolic.
      auto converted_inst_id =
          ConvertToValueOfType(context, SemIR::LocId(facet_type_inst_id),
                               rewrite_inst_id, assoc_const_type_id);
      // Canonicalize the converted constant value.
      converted_inst_id =
          context.constant_values().GetConstantInstId(converted_inst_id);
      // The result of conversion can be non-constant even if the original
      // value was constant.
      if (converted_inst_id.has_value()) {
        rewrite_inst_id = converted_inst_id;
      } else {
        const auto& assoc_const = context.associated_constants().Get(
            assoc_constant_decl->assoc_const_id);
        CARBON_DIAGNOSTIC(
            AssociatedConstantNotConstantAfterConversion, Error,
            "associated constant {0} given value {1} that is not constant "
            "after conversion to {2}",
            SemIR::NameId, InstIdAsConstant, SemIR::TypeId);
        context.emitter().Emit(
            facet_type_inst_id, AssociatedConstantNotConstantAfterConversion,
            assoc_const.name_id, rewrite_inst_id, assoc_const_type_id);
        rewrite_inst_id = SemIR::ErrorInst::InstId;
      }
    }

    CARBON_CHECK(rewrite_inst_id == context.constant_values().GetConstantInstId(
                                        rewrite_inst_id),
                 "Rewritten value for associated constant is not canonical.");

    table_entry = AddInst<SemIR::ImplWitnessAssociatedConstant>(
        context, witness_loc_id,
        {.type_id = context.insts().Get(rewrite_inst_id).type_id(),
         .inst_id = rewrite_inst_id});
  }
  return witness_inst_id;
}

auto RequireCompleteFacetTypeForImplDefinition(
    Context& context, SemIR::LocId loc_id, SemIR::TypeInstId facet_type_inst_id)
    -> bool {
  auto facet_type_id =
      context.types().GetTypeIdForTypeInstId(facet_type_inst_id);
  return RequireCompleteType(
      context, facet_type_id, SemIR::LocId(facet_type_inst_id), [&] {
        return IncompleteFacetTypeDiagnosticBuilder(context, loc_id,
                                                    facet_type_inst_id,
                                                    /*is_definition=*/true);
      });
}

auto AllocateFacetTypeImplWitness(Context& context,
                                  SemIR::InterfaceId interface_id,
                                  SemIR::InstBlockId witness_id) -> void {
  const auto& interface = context.interfaces().Get(interface_id);
  CARBON_CHECK(interface.is_complete());
  auto assoc_entities =
      context.inst_blocks().Get(interface.associated_entities_id);
  for (auto decl_id : assoc_entities) {
    LoadImportRef(context, decl_id);
  }

  llvm::SmallVector<SemIR::InstId> empty_table(
      assoc_entities.size(), SemIR::ImplWitnessTablePlaceholder::TypeInstId);
  context.inst_blocks().ReplacePlaceholder(witness_id, empty_table);
}

// Returns an ordering between two values in a rewrite constraint. Two
// `ImplWitnessAccess` instructions that refer to the same associated constant
// through the same facet value are treated as equivalent. Otherwise, the
// ordering is somewhat arbitrary with `ImplWitnessAccess` instructions coming
// first.
static auto CompareFacetTypeConstraintValues(Context& context,
                                             SemIR::InstId lhs_id,
                                             SemIR::InstId rhs_id)
    -> std::weak_ordering {
  if (lhs_id == rhs_id) {
    return std::weak_ordering::equivalent;
  }

  auto lhs_access = context.insts().TryGetAs<SemIR::ImplWitnessAccess>(lhs_id);
  auto rhs_access = context.insts().TryGetAs<SemIR::ImplWitnessAccess>(rhs_id);
  if (lhs_access && rhs_access) {
    auto lhs_lookup = context.insts().TryGetAs<SemIR::LookupImplWitness>(
        lhs_access->witness_id);
    auto rhs_lookup = context.insts().TryGetAs<SemIR::LookupImplWitness>(
        rhs_access->witness_id);
    if (lhs_lookup && rhs_lookup) {
      auto lhs_self = context.insts().TryGetAs<SemIR::BindSymbolicName>(
          context.constant_values().GetConstantInstId(
              lhs_lookup->query_self_inst_id));
      auto rhs_self = context.insts().TryGetAs<SemIR::BindSymbolicName>(
          context.constant_values().GetConstantInstId(
              rhs_lookup->query_self_inst_id));
      if (lhs_self && rhs_self) {
        if (lhs_self->entity_name_id != rhs_self->entity_name_id) {
          return lhs_self->entity_name_id.index <=>
                 rhs_self->entity_name_id.index;
        }
        if (lhs_access->index != rhs_access->index) {
          return lhs_access->index <=> rhs_access->index;
        }
        return lhs_lookup->query_specific_interface_id.index <=>
               rhs_lookup->query_specific_interface_id.index;
      }
    }

    // We do *not* want to get the evaluated result of `ImplWitnessAccess` here,
    // we want to keep them as a reference to an associated constant for the
    // resolution phase.
    return lhs_id.index <=> rhs_id.index;
  }

  // ImplWitnessAccess sorts before other instructions.
  if (lhs_access) {
    return std::weak_ordering::less;
  }
  if (rhs_access) {
    return std::weak_ordering::greater;
  }

  return context.constant_values().GetConstantInstId(lhs_id).index <=>
         context.constant_values().GetConstantInstId(rhs_id).index;
}

// Sort and dedupe the rewrite constraints, with accesses to the same associated
// constants through the same facet value being treated as equivalent.
static auto SortAndDedupeRewriteConstraints(
    Context& context,
    llvm::SmallVector<SemIR::FacetTypeInfo::RewriteConstraint>& rewrites) {
  auto ord = [&](const SemIR::FacetTypeInfo::RewriteConstraint& a,
                 const SemIR::FacetTypeInfo::RewriteConstraint& b) {
    auto lhs = CompareFacetTypeConstraintValues(context, a.lhs_id, b.lhs_id);
    if (lhs != std::weak_ordering::equivalent) {
      return lhs;
    }
    auto rhs = CompareFacetTypeConstraintValues(context, a.rhs_id, b.rhs_id);
    return rhs;
  };

  auto less = [&](const SemIR::FacetTypeInfo::RewriteConstraint& a,
                  const SemIR::FacetTypeInfo::RewriteConstraint& b) {
    return ord(a, b) == std::weak_ordering::less;
  };
  llvm::stable_sort(rewrites, less);

  auto eq = [&](const SemIR::FacetTypeInfo::RewriteConstraint& a,
                const SemIR::FacetTypeInfo::RewriteConstraint& b) {
    return ord(a, b) == std::weak_ordering::equivalent;
  };
  rewrites.erase(llvm::unique(rewrites, eq), rewrites.end());
}

// To be used for substituting into the RHS of a rewrite constraint.
//
// It will substitute any `ImplWitnessAccess` into `.Self` (a reference to an
// associated constant) with the RHS of another rewrite constraint that writes
// to the same associated constant. For example:
// ```
// Z where .X = () and .Y = .X
// ```
// Here the second `.X` is an `ImplWitnessAccess` which would be substituted by
// finding the first rewrite constraint, where the LHS is for the same
// associated constant and using its RHS. So the substitution would produce:
// ```
// Z where .X = () and .Y = ()
// ```
//
// This additionally diagnoses cycles when the `ImplWitnessAccess` is reading
// from the same rewrite constraint, and is thus assigning to the associated
// constant a value that refers to the same associated constant, such as with
// `Z where .X = C(.X)`. In the event of a cycle, the `ImplWitnessAccess` is
// replaced with `ErrorInst` so that further evaluation of the
// `ImplWitnessAccess` will not loop infinitely.
class SubstImplWitnessAccessCallbacks : public SubstInstCallbacks {
 public:
  // The `rewrites` is the set of rewrite constraints that are being
  // substituted, and where it looks for rewritten values to substitute from.
  //
  // The `substituting_constraint` is the rewrite constraint for which the RHS
  // is being substituted with the value from another rewrite constraint, if
  // possible. That is, `.Y = .X` in the example in the class docs.
  explicit SubstImplWitnessAccessCallbacks(
      Context* context, SemIR::LocId loc_id,
      llvm::ArrayRef<SemIR::FacetTypeInfo::RewriteConstraint> rewrites,
      const SemIR::FacetTypeInfo::RewriteConstraint* substituting_constraint)
      : SubstInstCallbacks(context),
        loc_id_(loc_id),
        rewrites_(rewrites),
        substituting_constraint_(substituting_constraint) {}

  auto Subst(SemIR::InstId& rhs_inst_id) const -> bool override {
    if (!context().insts().Is<SemIR::ImplWitnessAccess>(rhs_inst_id)) {
      return context().constant_values().Get(rhs_inst_id).is_concrete();
    }

    // TODO: We could consider something better than linear search here, such as
    // a map. However that would probably require heap allocations which may be
    // worse overall since the number of rewrite constraints is generally low.
    for (const auto& search_constraint : rewrites_) {
      if (CompareFacetTypeConstraintValues(context(), search_constraint.lhs_id,
                                           rhs_inst_id) ==
          std::weak_ordering::equivalent) {
        if (&search_constraint == substituting_constraint_) {
          if (search_constraint.rhs_id != SemIR::ErrorInst::InstId) {
            CARBON_DIAGNOSTIC(FacetTypeConstraintCycle, Error,
                              "found cycle in facet type constraint for {0}",
                              InstIdAsConstant);
            // TODO: It would be nice to note the places where the values are
            // assigned but rewrite constraint instructions are from canonical
            // constant values, and have no locations. We'd need to store a
            // location along with them in the rewrite constraints, and track
            // propagation of locations here, which may imply heap allocations.
            context().emitter().Emit(loc_id_, FacetTypeConstraintCycle,
                                     substituting_constraint_->lhs_id);
            rhs_inst_id = SemIR::ErrorInst::InstId;
          }
        } else {
          rhs_inst_id = search_constraint.rhs_id;
        }
      }
    }

    // Never recurse into ImplWitnessAccess, we don't want to substitute into
    // FacetTypes found within. We only substitute ImplWitnessAccesses that
    // appear directly on the RHS.
    return true;
  }

  auto Rebuild(SemIR::InstId /*orig_inst_id*/, SemIR::Inst new_inst) const
      -> SemIR::InstId override {
    return RebuildNewInst(loc_id_, new_inst);
  }

 private:
  SemIR::LocId loc_id_;
  llvm::ArrayRef<SemIR::FacetTypeInfo::RewriteConstraint> rewrites_;
  const SemIR::FacetTypeInfo::RewriteConstraint* substituting_constraint_;
};

auto ResolveFacetTypeRewriteConstraints(
    Context& context, SemIR::LocId loc_id,
    llvm::SmallVector<SemIR::FacetTypeInfo::RewriteConstraint>& rewrites)
    -> void {
  if (rewrites.empty()) {
    return;
  }

  while (true) {
    bool applied_rewrite = false;

    for (auto& constraint : rewrites) {
      if (constraint.lhs_id == SemIR::ErrorInst::InstId ||
          constraint.rhs_id == SemIR::ErrorInst::InstId) {
        continue;
      }

      auto lhs_access =
          context.insts().TryGetAs<SemIR::ImplWitnessAccess>(constraint.lhs_id);
      if (!lhs_access) {
        continue;
      }

      // Replace any `ImplWitnessAccess` in the RHS of this constraint with the
      // RHS of another constraint that sets the value of the associated
      // constant being accessed in the RHS.
      auto subst_inst_id =
          SubstInst(context, constraint.rhs_id,
                    SubstImplWitnessAccessCallbacks(&context, loc_id, rewrites,
                                                    &constraint));
      if (subst_inst_id != constraint.rhs_id) {
        constraint.rhs_id = subst_inst_id;
        if (constraint.rhs_id != SemIR::ErrorInst::InstId) {
          // If the RHS is replaced with a non-error value, we need to do
          // another pass so that the new RHS value can continue to propagate.
          applied_rewrite = true;
        }
      }
    }

    if (!applied_rewrite) {
      break;
    }
  }

  // We sort the constraints so that we can find different values being written
  // to the same LHS by looking at consecutive rewrite constraints.
  //
  // It is important to dedupe so that we don't have redundant rewrite
  // constraints, as these lead to being diagnosed as a cycle. For example:
  // ```
  // (T:! Z where .X = .Y) where .X = .Y
  // ```
  // Here we drop one of the `.X = .Y` in the resulting facet type. If we don't,
  // then the `.X` in the outer facet type can be evaluated to `.Y` from the
  // inner facet type, resulting in `.Y = .Y` which is a cycle. By deduping, we
  // avoid any LHS of a rewrite constraint from being evaluated to the RHS of
  // a duplicate rewrite constraint.
  SortAndDedupeRewriteConstraints(context, rewrites);

  for (size_t i = 0; i < rewrites.size() - 1; ++i) {
    auto& constraint = rewrites[i];
    if (constraint.lhs_id == SemIR::ErrorInst::InstId ||
        constraint.rhs_id == SemIR::ErrorInst::InstId) {
      continue;
    }

    auto lhs_access =
        context.insts().TryGetAs<SemIR::ImplWitnessAccess>(constraint.lhs_id);
    if (!lhs_access) {
      continue;
    }

    // This loop moves `i` to the last position with the same LHS value, so that
    // we don't diagnose more than once within the same contiguous range of
    // assignments to a single LHS value.
    for (; i < rewrites.size() - 1; ++i) {
      auto& next = rewrites[i + 1];
      auto next_lhs_access =
          context.insts().TryGetAs<SemIR::ImplWitnessAccess>(next.lhs_id);
      if (!next_lhs_access) {
        break;
      }

      if (CompareFacetTypeConstraintValues(context, constraint.lhs_id,
                                           next.lhs_id) !=
          std::weak_ordering::equivalent) {
        break;
      }

      if (constraint.rhs_id != SemIR::ErrorInst::InstId &&
          next.rhs_id != SemIR::ErrorInst::InstId) {
        CARBON_DIAGNOSTIC(
            AssociatedConstantWithDifferentValues, Error,
            "associated constant {0} given two different values {1} and {2}",
            InstIdAsConstant, InstIdAsConstant, InstIdAsConstant);
        // Use inst id ordering as a simple proxy for source ordering, to try
        // to name the values in the same order they appear in the facet type.
        auto source_order1 = constraint.rhs_id.index < next.rhs_id.index
                                 ? constraint.rhs_id
                                 : next.rhs_id;
        auto source_order2 = constraint.rhs_id.index >= next.rhs_id.index
                                 ? constraint.rhs_id
                                 : next.rhs_id;
        // TODO: It would be nice to note the places where the values are
        // assigned but rewrite constraint instructions are from canonical
        // constant values, and have no locations. We'd need to store a
        // location along with them in the rewrite constraints.
        context.emitter().Emit(loc_id, AssociatedConstantWithDifferentValues,
                               constraint.lhs_id, source_order1, source_order2);
      }
      constraint.rhs_id = SemIR::ErrorInst::InstId;
      next.rhs_id = SemIR::ErrorInst::InstId;
    }
  }
}

}  // namespace Carbon::Check
