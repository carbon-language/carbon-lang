// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/facet_type.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/interface.h"
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

    if (table_entry != SemIR::ImplWitnessTablePlaceholder::TypeInstId) {
      if (table_entry != rewrite_inst_id) {
        // TODO: Figure out how to print the two different values
        // `const_id` & `rewrite_inst_id` in the diagnostic
        // message.
        CARBON_DIAGNOSTIC(
            AssociatedConstantWithDifferentValues, Error,
            "associated constant {0} given two different values {1} and {2}",
            SemIR::NameId, InstIdAsConstant, InstIdAsConstant);
        auto& assoc_const = context.associated_constants().Get(
            assoc_constant_decl->assoc_const_id);
        context.emitter().Emit(
            facet_type_inst_id, AssociatedConstantWithDifferentValues,
            assoc_const.name_id, table_entry, rewrite_inst_id);
      }
      table_entry = SemIR::ErrorInst::InstId;
      continue;
    }

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

// A rewrite that associates either a value or an associated constant on the
// RHS with the associated constant on the LHS.
//
// Rewrite rules can come from concrete impl witnesses or from facet types,
// and this generalizes them into a common structure that structurally tracks
// if the associated value on the right names a constant.
//
// All ConstantIds must be mapped into the specific of the witness or facet
// that they come from in order to make them comparable in a given context.
struct RewriteRule {
  struct AssociatedConstant {
    // The value of an AssociatedConstantDecl instruction.
    SemIR::ConstantId const_id;
    auto operator==(const AssociatedConstant& rhs) const -> bool = default;
  };
  struct AssignedValue {
    // The final value assigned to an associated constant. May be anything
    // other than an AssociatedConstantDecl.
    SemIR::ConstantId const_id;
    auto operator==(const AssignedValue& rhs) const -> bool = default;
  };

  using Assignment = std::variant<AssociatedConstant, AssignedValue>;

  AssociatedConstant lhs;
  Assignment rhs;
};

// Tracks rewrite rules via their equivalency sets of `AssociatedConstant`s and
// the constant values assigned to each equivalency set.
class AppliedRewriteRules {
 public:
  struct Value {
    auto const_value_id() -> SemIR::ConstantId { return const_value_id_; }

   private:
    friend AppliedRewriteRules;

    Value() = default;

    SemIR::ConstantId const_value_id_ = SemIR::ConstantId::None;
    int index = -1;
  };

  auto Apply(RewriteRule rewrite_rule) -> void {
    auto lhs = rewrite_rule.lhs;
    CARBON_KIND_SWITCH(rewrite_rule.rhs) {
      case CARBON_KIND(RewriteRule::AssociatedConstant rhs): {
        auto* lhs_value = FindValue(lhs);
        auto* rhs_value = FindValue(rhs);
        if (lhs_value && rhs_value) {
          if (lhs_value != rhs_value) {
            Merge(*lhs_value, *rhs_value);
          }
        } else if (lhs_value) {
          // Add the rhs to the lhs equivalency set.
          AddEquivalent(*lhs_value, rhs);
        } else if (rhs_value) {
          // Add the lhs to the rhs equivalency set.
          AddEquivalent(*rhs_value, lhs);
        } else {
          // Create a new equivalency set.
          auto& new_value = AddIndependent(lhs);
          if (lhs != rhs) {
            AddEquivalent(new_value, rhs);
          }
        }
        break;
      }
      case CARBON_KIND(RewriteRule::AssignedValue rhs): {
        auto* lhs_value = FindValue(lhs);
        if (lhs_value) {
          // Assign the constant value to the equivalency group.
          SetConstant(*lhs_value, rhs.const_id);
        } else {
          // Create a new equivalency set, with the assigned constant value.
          auto& new_value = AddIndependent(lhs);
          SetConstant(new_value, rhs.const_id);
        }
        break;
      }
    }
  }

  auto FindValue(RewriteRule::AssociatedConstant ac) -> Value* {
    if (auto e = indexes_.Lookup(ac)) {
      int index = e.value();
      return &values_[index];
    } else {
      return nullptr;
    }
  }

 private:
  auto SetConstant(Value& into, SemIR::ConstantId const_value_id) -> void {
    into.const_value_id_ = const_value_id;
  }

  auto Merge(Value& into, Value& from) -> void {
    if (from.const_value_id_.has_value()) {
      if (into.const_value_id_.has_value()) {
        CARBON_CHECK(from.const_value_id_ == into.const_value_id_);
      }
      into.const_value_id_ = from.const_value_id_;
    }
    indexes_.ForEach([=](RewriteRule::AssociatedConstant& /*key*/, int& index) {
      if (index == from.index) {
        index = into.index;
      }
    });
  }

  auto AddEquivalent(Value& into, RewriteRule::AssociatedConstant ac) -> void {
    auto result = indexes_.Insert(ac, into.index);
    CARBON_CHECK(result.is_inserted());
  }

  auto AddIndependent(RewriteRule::AssociatedConstant ac) -> Value& {
    CARBON_CHECK(values_.size() < std::numeric_limits<int>::max());
    auto new_index = static_cast<int>(values_.size());
    values_.push_back(Value());
    values_.back().index = new_index;
    auto result = indexes_.Insert(ac, new_index);
    CARBON_CHECK(result.is_inserted());
    return values_.back();
  }

  // Maps to an index in the vector below. Constants with the same index value
  // are considered equivalent.
  Map<RewriteRule::AssociatedConstant, int> indexes_;
  llvm::SmallVector<Value> values_;
};

// Returns the value of the `AssociatedConstantDecl` accessed by an
// `ImplWitnessAccess`, or nullopt if the access is not to an associated
// constant (such as for an access to an associated function, or an error).
//
// The AssociatedConstantDecl is mapped into the specific of the witness.
static auto TryGetAssociatedConstantFromImplWitnessAccess(
    Context& context, const SemIR::ImplWitnessAccess& witness_access)
    -> std::optional<SemIR::ConstantId> {
  auto witness = context.insts().GetAs<SemIR::LookupImplWitness>(
      witness_access.witness_id);
  auto specific_interface =
      context.specific_interfaces().Get(witness.query_specific_interface_id);
  const auto& interface =
      context.interfaces().Get(specific_interface.interface_id);
  auto assoc_entities =
      context.inst_blocks().Get(interface.associated_entities_id);
  auto decl_id = assoc_entities[witness_access.index.index];
  if (decl_id == SemIR::ErrorInst::InstId) {
    return std::nullopt;
  }
  if (!context.insts().Is<SemIR::AssociatedConstantDecl>(decl_id)) {
    // Something in the interface other than an associated constant.
    return std::nullopt;
  }
  auto decl_const_id = SemIR::GetConstantValueInSpecific(
      context.sem_ir(), specific_interface.specific_id, decl_id);
  return decl_const_id;
}

// Generates the rewrite rules retrieved from the witness tables found in a
// FacetType.
class RewriteRulesFromFacetType {
 public:
  explicit RewriteRulesFromFacetType(Context& context,
                                     SemIR::FacetTypeId facet_type_id)
      : context_(context) {
    if (facet_type_id.has_value()) {
      const auto& facet_type_info = context.facet_types().Get(facet_type_id);
      it_ = facet_type_info.rewrite_constraints.begin();
      end_ = facet_type_info.rewrite_constraints.end();
    } else {
      it_ = end_ = nullptr;
    }
  }

  // Returns each RewriteRule found in the FacetType, then returns nullopt.
  auto Next() -> std::optional<RewriteRule> {
    while (it_ != end_) {
      auto lhs_id = it_->lhs_id;
      auto rhs_id = it_->rhs_id;
      ++it_;

      while (lhs_id == SemIR::ErrorInst::InstId ||
             rhs_id == SemIR::ErrorInst::InstId) {
        lhs_id = it_->lhs_id;
        rhs_id = it_->rhs_id;
        ++it_;
        if (it_ == end_) {
          return std::nullopt;
        }
      }

      auto lhs_witness_access =
          context_.insts().GetAs<SemIR::ImplWitnessAccess>(lhs_id);
      auto lhs = TryGetAssociatedConstantFromImplWitnessAccess(
          context_, lhs_witness_access);
      if (!lhs) {
        // We found an associated value that is not a constant (like a
        // function), so continue to the next element in the table.
        continue;
      }

      if (auto rhs_witness_access =
              context_.insts().TryGetAs<SemIR::ImplWitnessAccess>(rhs_id)) {
        auto rhs = TryGetAssociatedConstantFromImplWitnessAccess(
            context_, *rhs_witness_access);
        if (!rhs) {
          // The RHS is comes from a witness table but is not an associated
          // constant, such as if we find an error. Continue to the next element
          // in this table.
          continue;
        }
        return {{.lhs = RewriteRule::AssociatedConstant{.const_id = *lhs},
                 .rhs = RewriteRule::AssociatedConstant{.const_id = *rhs}}};
      } else {
        auto rhs = context_.constant_values().Get(rhs_id);
        return {{.lhs = RewriteRule::AssociatedConstant{.const_id = *lhs},
                 .rhs = RewriteRule::AssignedValue{.const_id = rhs}}};
      }
    }

    return std::nullopt;
  }

 private:
  Context& context_;
  const Carbon::SemIR::FacetTypeInfo::RewriteConstraint* it_;
  const Carbon::SemIR::FacetTypeInfo::RewriteConstraint* end_;
};

class RewriteRulesFromImplWitness {
 public:
  RewriteRulesFromImplWitness(Context& context, SemIR::InstId witness_id)
      : context_(context) {
    const auto& witness = context.insts().GetAs<SemIR::ImplWitness>(witness_id);
    auto table = context.insts().GetAs<SemIR::ImplWitnessTable>(
        witness.witness_table_id);

    const auto& impl = context.impls().Get(table.impl_id);
    const auto& interface =
        context.interfaces().Get(impl.interface.interface_id);

    if (!interface.associated_entities_id.has_value()) {
      return;
    }

    impl_interface_specific_id_ = impl.interface.specific_id;
    witness_specific_id_ = witness.specific_id;

    lhs_inst_ids_ =
        context_.inst_blocks().Get(interface.associated_entities_id);
    lhs_inst_ids_it_ = lhs_inst_ids_.begin();
    rhs_inst_ids_ = context.inst_blocks().Get(table.elements_id);
    rhs_inst_ids_it_ = rhs_inst_ids_.begin();
  }

  auto Next() -> std::optional<RewriteRule> {
    while (true) {
      // The witness table (the RHS) may have fewer entries in it (or even zero
      // if there's no impl definition) than the interface (the LHS) has, but
      // they are in the same order for whatever is present.
      if (lhs_inst_ids_it_ == lhs_inst_ids_.end()) {
        return std::nullopt;
      }
      if (rhs_inst_ids_it_ == rhs_inst_ids_.end()) {
        return std::nullopt;
      }

      auto lhs_inst_id = *lhs_inst_ids_it_;
      ++lhs_inst_ids_it_;
      auto rhs_inst_id = *rhs_inst_ids_it_;
      ++rhs_inst_ids_it_;

      auto lhs_assoc_const_decl_id = SemIR::ConstantId::None;
      if (context_.insts().Is<SemIR::AssociatedConstantDecl>(lhs_inst_id)) {
        lhs_assoc_const_decl_id = SemIR::GetConstantValueInSpecific(
            context_.sem_ir(), impl_interface_specific_id_, lhs_inst_id);
      }
      if (!lhs_assoc_const_decl_id.has_value()) {
        // This entry in the witness table is not for an associated constant.
        continue;
      }

      if (rhs_inst_id == SemIR::ErrorInst::InstId) {
        continue;
      }

      auto lhs =
          RewriteRule::AssociatedConstant{.const_id = lhs_assoc_const_decl_id};

      if (context_.insts().Is<SemIR::ImplWitnessTablePlaceholder>(
              rhs_inst_id)) {
        // This witness doesn't specify a value for the LHS associated
        // constant.
        continue;
      }

      auto witness_constant =
          context_.insts().GetAs<SemIR::ImplWitnessAssociatedConstant>(
              rhs_inst_id);
      rhs_inst_id = witness_constant.inst_id;

      if (auto rhs_witness_access =
              context_.insts().TryGetAs<SemIR::ImplWitnessAccess>(
                  rhs_inst_id)) {
        auto rhs = TryGetAssociatedConstantFromImplWitnessAccess(
            context_, *rhs_witness_access);
        if (!rhs) {
          // The RHS is comes from a witness table but is not an associated
          // constant, such as if we find an error. Continue to the next element
          // in this table.
          continue;
        }
        return {{.lhs = lhs,
                 .rhs = RewriteRule::AssociatedConstant{.const_id = *rhs}}};
      } else {
        auto rhs_const_id = SemIR::GetConstantValueInSpecific(
            context_.sem_ir(), witness_specific_id_, rhs_inst_id);
        return {{.lhs = lhs,
                 .rhs = RewriteRule::AssignedValue{.const_id = rhs_const_id}}};
      }
    }
  }

 private:
  Context& context_;
  SemIR::SpecificId impl_interface_specific_id_ = SemIR::SpecificId::None;
  SemIR::SpecificId witness_specific_id_ = SemIR::SpecificId::None;
  llvm::ArrayRef<SemIR::InstId> lhs_inst_ids_;
  const SemIR::InstId* lhs_inst_ids_it_ = lhs_inst_ids_.end();
  llvm::ArrayRef<SemIR::InstId> rhs_inst_ids_;
  const SemIR::InstId* rhs_inst_ids_it_ = rhs_inst_ids_.end();
};

auto CheckRewriteConstraintsMatchRequirements(
    Context& context, SemIR::FacetTypeId requirements_facet_type_id,
    llvm::ArrayRef<RewriteSource> rewrite_sources) -> bool {
  AppliedRewriteRules provided_rewrite_values;

  for (auto rewrite_source : rewrite_sources) {
    CARBON_KIND_SWITCH(rewrite_source) {
      case CARBON_KIND(SemIR::InstId witness_id):
        for (auto rules = RewriteRulesFromImplWitness(context, witness_id);
             auto rule = rules.Next();) {
          provided_rewrite_values.Apply(*rule);
        }
        break;
      case CARBON_KIND(SemIR::FacetTypeId facet_type_id):
        for (auto rules = RewriteRulesFromFacetType(context, facet_type_id);
             auto rule = rules.Next();) {
          provided_rewrite_values.Apply(*rule);
        }
        break;
    }
  }

  for (auto required_rules =
           RewriteRulesFromFacetType(context, requirements_facet_type_id);
       auto required_rule = required_rules.Next();) {
    CARBON_KIND_SWITCH(required_rule->rhs) {
      case CARBON_KIND(RewriteRule::AssociatedConstant rhs): {
        // The lhs and rhs must be in the same equivalency group.
        auto* lhs_value = provided_rewrite_values.FindValue(required_rule->lhs);
        auto* rhs_value = provided_rewrite_values.FindValue(rhs);
        if (!lhs_value) {
          // LHS of a rewrite rule has no value provided by a witness.
          return false;
        }
        if (!rhs_value) {
          // RHS of a rewrite rule has no value provided by a witness.
          return false;
        }
        if (lhs_value != rhs_value) {
          if (lhs_value->const_value_id().has_value() ||
              rhs_value->const_value_id().has_value()) {
            if (lhs_value->const_value_id() != rhs_value->const_value_id()) {
              // LHS and RHS of the rewrite constraint were given different
              // concrete values by a witness.
              return false;
            }
          }
        }
        break;
      }
      case CARBON_KIND(RewriteRule::AssignedValue rhs): {
        auto* lhs_value = provided_rewrite_values.FindValue(required_rule->lhs);
        if (!lhs_value) {
          // LHS of rewrite rule has no value provided by a witness.
          return false;
        }
        if (!lhs_value->const_value_id().has_value()) {
          // LHS of rewrite rule requires a concrete value provided by a witness
          // but not was given one.
          return false;
        }
        if (lhs_value->const_value_id() != rhs.const_id) {
          // LHS of rewrite rule has a different concrete value than the one
          // provided by a witness.
          return false;
        }
        break;
      }
    }
  }

  return true;
}

}  // namespace Carbon::Check
