// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/period_self.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto MakePeriodSelfFacetValue(Context& context, SemIR::LocId loc_id,
                              SemIR::TypeId self_type_id) -> SemIR::InstId {
  CARBON_CHECK(self_type_id == SemIR::ErrorInst::TypeId ||
               context.types().Is<SemIR::FacetType>(self_type_id));
  auto entity_name_id = context.entity_names().AddCanonical({
      .name_id = SemIR::NameId::PeriodSelf,
      .parent_scope_id = context.scope_stack().PeekNameScopeId(),
  });
  auto inst_id = AddInst<SemIR::SymbolicBinding>(
      context, loc_id,
      {
          .type_id = self_type_id,
          .entity_name_id = entity_name_id,
          // `None` because there is no equivalent non-symbolic value.
          .value_id = SemIR::InstId::None,
      });
  auto existing = context.scope_stack().LookupOrAddName(
      SemIR::NameId::PeriodSelf, inst_id, ScopeIndex::None,
      IsCurrentPositionReachable(context));
  // Shouldn't have any names in newly created scope.
  CARBON_CHECK(!existing.has_value());
  return inst_id;
}

SubstPeriodSelfCallbacks::SubstPeriodSelfCallbacks(
    Context* context, SemIR::LocId loc_id,
    SemIR::ConstantId period_self_replacement_id, Behaviour behaviour)
    : SubstInstCallbacks(context),
      loc_id_(loc_id),
      period_self_replacement_id_(period_self_replacement_id),
      behaviour_(behaviour) {}

auto SubstPeriodSelfCallbacks::Subst(SemIR::InstId& inst_id) -> SubstResult {
  // FacetTypes are concrete even if they have `.Self` inside them, but we
  // don't recurse into FacetTypes, so we can use this as a base case. This
  // avoids infinite recursion on TypeType and ErrorInst.
  if (context().constant_values().Get(inst_id).is_concrete()) {
    return FullySubstituted;
  }
  // Don't recurse into nested facet types, even if they are symbolic. Leave
  // their `.Self` as is.
  if (context().insts().Is<SemIR::FacetType>(inst_id)) {
    return FullySubstituted;
  }

  // Look for implicit use of `.Self` in designators: `.X` is really `.Self.X`.
  if (auto access =
          context().insts().TryGetAs<SemIR::ImplWitnessAccess>(inst_id)) {
    if (auto witness = context().insts().TryGetAs<SemIR::LookupImplWitness>(
            access->witness_id)) {
      // Canonicalization not necessary; We are working with the constant
      // value already, and the query self in a witness is already
      // canonicalized.
      if (IsPeriodSelf(context(), witness->query_self_inst_id,
                       /*canonicalize=*/false)) {
        auto replacement_id = GetReplacement(witness->query_self_inst_id);
        auto new_witness =
            Rebuild(access->witness_id,
                    SemIR::LookupImplWitness{
                        .type_id = witness->type_id,
                        .query_self_inst_id = replacement_id,
                        // Don't replace `.Self` in the interface specific
                        // here. That is an explicit `.Self` use. We'll
                        // revisit the instruction for that.
                        .query_specific_interface_id =
                            witness->query_specific_interface_id,
                    });
        auto new_access = Rebuild(inst_id, SemIR::ImplWitnessAccess{
                                               .type_id = access->type_id,
                                               .witness_id = new_witness,
                                               .index = access->index,
                                           });
        inst_id = new_access;
        return SubstAgain;
      }
    }
  }

  // Look for explicit use of `.Self`.
  if (behaviour_ == Behaviour::All) {
    // Canonicalization not necessary; Subst will recurse anyway, so avoid
    // extra work for non-matches.
    if (IsPeriodSelf(context(), inst_id, /*canonicalize=*/false)) {
      inst_id = GetReplacement(inst_id);
      return FullySubstituted;
    }
  }

  return SubstOperands;
}

auto SubstPeriodSelfCallbacks::Rebuild(SemIR::InstId orig_inst_id,
                                       SemIR::Inst new_inst) -> SemIR::InstId {
  return RebuildNewInst(SemIR::LocId(orig_inst_id), new_inst);
}

auto SubstPeriodSelfCallbacks::GetReplacement(SemIR::InstId period_self)
    -> SemIR::InstId {
  auto period_self_type_id = context().insts().Get(period_self).type_id();
  CARBON_CHECK(context().types().Is<SemIR::FacetType>(period_self_type_id));

  auto replacement_self_inst_id =
      context().constant_values().GetInstId(period_self_replacement_id_);
  auto replacement_type_id =
      context().insts().Get(replacement_self_inst_id).type_id();
  CARBON_CHECK(context().types().IsFacetType(replacement_type_id));

  // If the replacement has the same type as `.Self`, use it directly.
  if (replacement_type_id == period_self_type_id) {
    return replacement_self_inst_id;
  }

  // If we have already converted the replacement to the type of `.Self`, use
  // our previous conversion.
  if (period_self_type_id == cached_replacement_type_id_) {
    return cached_replacement_id_;
  }

  // Convert the replacement facet to the type of `.Self`.
  cached_replacement_id_ = ConvertReplacement(
      replacement_self_inst_id, replacement_type_id, period_self_type_id);
  cached_replacement_type_id_ = period_self_type_id;
  return cached_replacement_id_;
}

auto SubstPeriodSelfCallbacks::ConvertReplacement(
    SemIR::InstId replacement_self_inst_id, SemIR::TypeId replacement_type_id,
    SemIR::TypeId period_self_type_id) -> SemIR::InstId {
  // TODO: Replace all empty facet types with TypeType.
  if (period_self_type_id == GetEmptyFacetType(context())) {
    // Convert to an empty facet type (representing TypeType); we don't need
    // any witnesses.
    return ConvertToValueOfType(context(), loc_id_, replacement_self_inst_id,
                                period_self_type_id);
  }

  // We have a facet or a type, but we need more interfaces in the facet type.
  // We will have to synthesize a symbolic witness for each interface.
  //
  // Why is this okay? The type of `.Self` comes from interfaces that are
  // before it (to the left of it) in the facet type. The replacement for
  // `.Self` will have to impl those interfaces in order to match the facet
  // type, so we know that it is valid to construct these witnesses.

  // Make the replacement into a type, which we will need for the FacetValue.
  if (context().types().Is<SemIR::FacetType>(replacement_type_id)) {
    replacement_self_inst_id = context().constant_values().GetInstId(
        EvalOrAddInst<SemIR::FacetAccessType>(
            context(), loc_id_,
            {.type_id = SemIR::TypeType::TypeId,
             .facet_value_inst_id = replacement_self_inst_id}));
  }

  auto period_self_facet_type =
      context().types().GetAs<SemIR::FacetType>(period_self_type_id);
  auto identified_period_self_type_id = RequireIdentifiedFacetType(
      context(), loc_id_,
      context().constant_values().Get(replacement_self_inst_id),
      period_self_facet_type, [&](auto& /*builder*/) {
        // The facet type containing this `.Self` should have already been
        // identified, which would ensure that the type of `.Self` can be
        // identified since it can only depend on things to the left of it
        // inside the same facet type.
        CARBON_FATAL("could not identify type of `.Self`");
      });
  const auto& identified_period_self_type =
      context().identified_facet_types().Get(identified_period_self_type_id);
  auto required_impls = identified_period_self_type.required_impls();
  llvm::SmallVector<SemIR::InstId> witnesses;
  witnesses.reserve(required_impls.size());
  for (const auto& req : required_impls) {
    witnesses.push_back(context().constant_values().GetInstId(
        EvalOrAddInst<SemIR::LookupImplWitness>(
            context(), loc_id_,
            {.type_id =
                 GetSingletonType(context(), SemIR::WitnessType::TypeInstId),
             .query_self_inst_id =
                 context().constant_values().GetInstId(req.self_facet_value),
             .query_specific_interface_id = context().specific_interfaces().Add(
                 req.specific_interface)})));
  }
  return context().constant_values().GetInstId(EvalOrAddInst<SemIR::FacetValue>(
      context(), loc_id_,
      {
          .type_id = period_self_type_id,
          .type_inst_id =
              context().types().GetAsTypeInstId(replacement_self_inst_id),
          .witnesses_block_id = context().inst_blocks().Add(witnesses),
      }));
}

auto SubstPeriodSelf(Context& context, SubstPeriodSelfCallbacks& callbacks,
                     SemIR::ConstantId const_id) -> SemIR::ConstantId {
  // Don't replace `.Self` with itself; that is cyclical.
  //
  // If the types differ, we would try to convert the replacement to a `.Self`
  // of the desired type in `const_id`, which is what we already have, so
  // there's nothing we need to do. But trying to do that conversion recurses
  // when the type of the `.Self` contains a `.Self`.
  if (auto bind_type =
          context.constant_values().TryGetInstAs<SemIR::SymbolicBinding>(
              GetCanonicalFacetOrTypeValue(
                  context, callbacks.period_self_replacement_id()))) {
    if (context.entity_names().Get(bind_type->entity_name_id).name_id ==
        SemIR::NameId::PeriodSelf) {
      return const_id;
    }
  }

  auto subst_id = SubstInst(
      context, context.constant_values().GetInstId(const_id), callbacks);
  return context.constant_values().Get(subst_id);
}

static auto SubstPeriodSelfInSpecific(Context& context,
                                      SubstPeriodSelfCallbacks& callbacks,
                                      SemIR::SpecificId specific_id)
    -> SemIR::SpecificId {
  if (!specific_id.has_value()) {
    return specific_id;
  }

  const auto& specific = context.specifics().Get(specific_id);

  // Substitute into the specific without having to construct a FacetType
  // instruction just to hold the specific interface inside a constant id.
  llvm::SmallVector<SemIR::InstId> args(
      context.inst_blocks().Get(specific.args_id));
  for (auto& arg_id : args) {
    auto const_id = context.constant_values().Get(arg_id);
    const_id = SubstPeriodSelf(context, callbacks, const_id);
    arg_id = context.constant_values().GetInstId(const_id);
  }
  return MakeSpecific(context, callbacks.loc_id(), specific.generic_id, args);
}

auto SubstPeriodSelf(Context& context, SubstPeriodSelfCallbacks& callbacks,
                     SemIR::SpecificInterface interface)
    -> SemIR::SpecificInterface {
  interface.specific_id =
      SubstPeriodSelfInSpecific(context, callbacks, interface.specific_id);
  return interface;
}
auto SubstPeriodSelf(Context& context, SubstPeriodSelfCallbacks& callbacks,
                     SemIR::SpecificNamedConstraint constraint)
    -> SemIR::SpecificNamedConstraint {
  constraint.specific_id =
      SubstPeriodSelfInSpecific(context, callbacks, constraint.specific_id);
  return constraint;
}

auto IsPeriodSelf(Context& context, SemIR::InstId inst_id, bool canonicalize)
    -> bool {
  auto const_inst_id = context.constant_values().GetConstantInstId(inst_id);
  if (!const_inst_id.has_value()) {
    return false;
  }
  auto query_inst_id =
      canonicalize ? GetCanonicalFacetOrTypeValue(context, const_inst_id)
                   : inst_id;
  if (auto bind =
          context.insts().TryGetAs<SemIR::SymbolicBinding>(query_inst_id)) {
    const auto& entity_name = context.entity_names().Get(bind->entity_name_id);
    return entity_name.name_id == SemIR::NameId::PeriodSelf;
  }
  return false;
}

class SearchNonCanonicalForExplicitPeriodSelf : public SubstInstCallbacks {
 public:
  explicit SearchNonCanonicalForExplicitPeriodSelf(Context* context,
                                                   SemIR::LocId* found)
      : SubstInstCallbacks(context), found_(found) {}

  auto Subst(SemIR::InstId& inst_id) -> SubstResult override {
    if (found_->has_value()) {
      return FullySubstituted;
    }

    auto const_inst_id = context().constant_values().GetConstantInstId(inst_id);
    if (const_inst_id == SemIR::TypeType::TypeInstId) {
      // Recursion base case. TypeType has type TypeType.
      return FullySubstituted;
    }
    if (context().insts().Is<SemIR::FacetType>(const_inst_id)) {
      // Don't look for `.Self` in nested facet types, they aren't replaced
      // with a facet value and just remain as abstract. WhereExprs evaluate
      // to a FacetType but are handled outside of Subst.
      return FullySubstituted;
    }

    if (auto name_ref = context().insts().TryGetAs<SemIR::NameRef>(inst_id)) {
      // Canonicalization not necessary; NameRef contains the SymbolicBinding
      // directly, not an `as type` conversion.
      if (IsPeriodSelf(context(), name_ref->value_id,
                       /*canonicalize=*/false)) {
        // `.Self` does not have a location, the NameRef pointing to it does.
        *found_ = SemIR::LocId(inst_id);
        return FullySubstituted;
      }
    }

    return SubstOperands;
  }

  auto Rebuild(SemIR::InstId /*orig_inst_id*/, SemIR::Inst /*new_inst*/)
      -> SemIR::InstId override {
    CARBON_FATAL();
  }

 private:
  SemIR::LocId* found_;
};

class SearchCanonicalForExplicitPeriodSelf : public SubstInstCallbacks {
 public:
  explicit SearchCanonicalForExplicitPeriodSelf(Context* context, bool* found)
      : SubstInstCallbacks(context), found_(found) {}

  auto Subst(SemIR::InstId& inst_id) -> SubstResult override {
    if (*found_) {
      return FullySubstituted;
    }

    auto const_inst_id = context().constant_values().GetConstantInstId(inst_id);
    if (const_inst_id == SemIR::TypeType::TypeInstId) {
      // Recursion base case. TypeType has type TypeType.
      return FullySubstituted;
    }
    if (context().insts().Is<SemIR::FacetType>(const_inst_id)) {
      // Don't look for `.Self` in nested facet types, they aren't replaced
      // with a facet value and just remain as abstract. WhereExprs evaluate
      // to a FacetType but are handled outside of Subst.
      return FullySubstituted;
    }

    if (auto access = context().insts().TryGetAs<SemIR::ImplWitnessAccess>(
            const_inst_id)) {
      if (auto lookup = context().insts().TryGetAs<SemIR::LookupImplWitness>(
              access->witness_id)) {
        // Canonicalization not necessary; We are working with the constant
        // value already, and the query self in a witness is already
        // canonicalized.
        if (IsPeriodSelf(context(), lookup->query_self_inst_id,
                         /*canonicalize=*/false)) {
          // An implicit `.Self` in a member designator is always allowed.
          return FullySubstituted;
        }
      }
    }

    // Canonicalization not necessary; Subst will recurse anyway, so avoid
    // extra work for non-matches.
    if (IsPeriodSelf(context(), const_inst_id, /*canonicalize=*/false)) {
      *found_ = true;
      return FullySubstituted;
    }

    return SubstOperands;
  }

  auto Rebuild(SemIR::InstId /*orig_inst_id*/, SemIR::Inst /*new_inst*/)
      -> SemIR::InstId override {
    CARBON_FATAL();
  }

 private:
  bool* found_;
};

static auto ReportAmbiguousPeriodSelf(Context& context, SemIR::LocId loc_id)
    -> void {
  CARBON_DIAGNOSTIC(AmbiguousPeriodSelf, Error,
                    "`.Self` is ambiguous after nested `where` in `<type> "
                    "impls ...` clause.");
  context.emitter().Emit(loc_id, AmbiguousPeriodSelf);
}

// Searches a type for a reference to `.Self`. Types are canonical, so they
// only contain canonical values/inststructions, which have no location of
// their own.
//
// The search excludes ImplWitnessAccess into `.Self`, which represents a
// designator like `.X`.
//
// The search does not recurse into FacetTypes, as some can include valid
// references to the top level `.Self`, or abstract `.Self` references that
// are not replaced. FacetTypes are handled by the higher level search.
//
// Returns true if found, and diagnosed.
static auto SearchTypeForPeriodSelf(Context& context, SemIR::LocId loc_id,
                                    SemIR::TypeId type_id) -> bool {
  bool found_canonical = false;
  SearchCanonicalForExplicitPeriodSelf callbacks(&context, &found_canonical);

  auto canonical_inst_id = context.types().GetTypeInstId(type_id);
  SubstInst(context, canonical_inst_id, callbacks);

  // The type has no locations internally, as it stores canonical
  // instructions. If we find any `.Self` reference, we report the entire
  // type.
  if (found_canonical) {
    ReportAmbiguousPeriodSelf(context, loc_id);
    return true;
  }
  return false;
}

// Searches a facet type for a reference to `.Self`. FacetTypes are canonical,
// so they only contain canonical values/inststructions, which have no
// location of their own.
//
// The search excludes ImplWitnessAccess into `.Self`, which represents a
// designator like `.X`.
//
// Returns true if found, and diagnosed.
static auto SearchFacetTypeForPeriodSelf(Context& context, SemIR::LocId loc_id,
                                         SemIR::FacetTypeId facet_type_id)
    -> bool {
  bool found_canonical = false;
  SearchCanonicalForExplicitPeriodSelf callbacks(&context, &found_canonical);

  const auto& info = context.facet_types().Get(facet_type_id);
  // The LHS of a `WhereExpr` only has extend constraints.
  for (auto extend : info.extend_constraints) {
    auto block_id = context.specifics().GetArgsOrEmpty(extend.specific_id);
    for (auto inst_id : context.inst_blocks().GetOrEmpty(block_id)) {
      SubstInst(context, inst_id, callbacks);
    }
  }
  for (auto extend : info.extend_named_constraints) {
    auto block_id = context.specifics().GetArgsOrEmpty(extend.specific_id);
    for (auto inst_id : context.inst_blocks().GetOrEmpty(block_id)) {
      SubstInst(context, inst_id, callbacks);
    }
  }
  // The facet type has no locations internally, as it stores canonical
  // instructions. If we find any `.Self` reference, we report the entire
  // facet type.
  if (found_canonical) {
    ReportAmbiguousPeriodSelf(context, loc_id);
    return true;
  }
  return false;
}

// Searches a non-canonical instruction for an explicitly written use of
// `.Self`, which is represented as a NameRef instruction.
//
// Returns true if found, and diagnosed.
static auto SearchNonCanonicalInstForPeriodSelf(Context& context,
                                                SemIR::InstId inst_id) -> bool {
  auto found = SemIR::LocId::None;
  SearchNonCanonicalForExplicitPeriodSelf callbacks(&context, &found);
  SubstInst(context, inst_id, callbacks);
  if (found.has_value()) {
    ReportAmbiguousPeriodSelf(context, found);
    return true;
  }
  return false;
}

auto FindAndDiagnoseAmbiguousPeriodSelf(Context& context,
                                        SemIR::InstId impls_lhs_id,
                                        SemIR::InstId impls_rhs_id) -> bool {
  // Look for errors up front. We don't need to look for them in the rest of
  // the function.
  if (context.constant_values().Get(impls_lhs_id) ==
          SemIR::ErrorInst::ConstantId ||
      context.constant_values().Get(impls_rhs_id) ==
          SemIR::ErrorInst::ConstantId) {
    return false;
  }

  if (IsPeriodSelf(context, impls_lhs_id)) {
    // `.Self impls X where ...` does not restrict any use of `.Self` on the
    // RHS of the `where` since the `.Self` on the LHS of `where` did not
    // introduce any ambiguity. A `.Self` on the RHS of the `where` applies to
    // the same thing as on the LHS of the `impls`.
    return false;
  }

  struct WorkItem {
    SemIR::WhereExpr where_expr;
    bool search_lhs;
  };

  llvm::SmallVector<WorkItem> work;
  if (auto where_expr =
          context.insts().TryGetAs<SemIR::WhereExpr>(impls_rhs_id)) {
    work.push_back({.where_expr = *where_expr, .search_lhs = false});
  }

  while (!work.empty()) {
    auto work_item = work.pop_back_val();

    // Look in the non-canonical WhereExpr for explicit references to `.Self`,
    // which will be considered as ambiguous.
    for (auto inst_id : context.inst_blocks().GetOrEmpty(
             work_item.where_expr.requirements_id)) {
      auto inst = context.insts().Get(inst_id);
      CARBON_KIND_SWITCH(inst) {
        case CARBON_KIND(SemIR::RequirementBaseFacetType base): {
          if (work_item.search_lhs) {
            // If the base type is more than a reference to an interface or
            // constraint, such as having specific arguments, it will be a
            // FacetType instruction.
            if (auto facet_type = context.insts().TryGetAs<SemIR::FacetType>(
                    base.base_type_inst_id)) {
              if (SearchFacetTypeForPeriodSelf(
                      context, SemIR::LocId(base.base_type_inst_id),
                      facet_type->facet_type_id)) {
                return true;
              }
            }
          }
          break;
        }
        case CARBON_KIND(SemIR::RequirementRewrite rewrite): {
          if (SearchNonCanonicalInstForPeriodSelf(context, rewrite.lhs_id)) {
            return true;
          }
          if (SearchNonCanonicalInstForPeriodSelf(context, rewrite.rhs_id)) {
            return true;
          }
          break;
        }
        case CARBON_KIND(SemIR::RequirementEquivalent equiv): {
          if (SearchNonCanonicalInstForPeriodSelf(context, equiv.lhs_id)) {
            return true;
          }
          if (SearchNonCanonicalInstForPeriodSelf(context, equiv.rhs_id)) {
            return true;
          }
          break;
        }
        case CARBON_KIND(SemIR::RequirementImpls impls): {
          if (!IsPeriodSelf(context, impls.lhs_id)) {
            if (SearchTypeForPeriodSelf(
                    context, SemIR::LocId(impls.lhs_id),
                    context.types().GetTypeIdForTypeInstId(impls.lhs_id))) {
              return true;
            }
          }

          CARBON_KIND_SWITCH(context.insts().Get(impls.rhs_id)) {
            case CARBON_KIND(SemIR::FacetType facet_type): {
              // If the RHS of the `impls` is a complex facet type (such as
              // when it has specific arguments) but has no `where`, then it
              // will be a FacetType instruction.
              if (SearchFacetTypeForPeriodSelf(context,
                                               SemIR::LocId(impls.rhs_id),
                                               facet_type.facet_type_id)) {
                return true;
              }
              break;
            }
            case CARBON_KIND(SemIR::WhereExpr rhs_where_expr): {
              // If the RHS of the `impls` contains a `where`, then it will be
              // a WhereExpr instruction.
              work.push_back(
                  {.where_expr = rhs_where_expr, .search_lhs = true});
              break;
            }
            default:
              // Otherwise, it's a simple facet type, which is just a
              // reference to an interface or constraint. There's nowhere to
              // look for a
              // `.Self`.
              break;
          }

          break;
        }
        default:
          CARBON_FATAL("unexpected inst {0} in WhereExpr requirements block",
                       inst);
      }
    }
  }

  return false;
}

}  // namespace Carbon::Check
