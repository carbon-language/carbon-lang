// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/period_self.h"
#include "toolchain/check/subst.h"
#include "toolchain/check/type.h"
#include "toolchain/check/unused.h"
#include "toolchain/sem_ir/facet_type_info.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

static auto GetExtendedOnlyFacetType(Context& context,
                                     const SemIR::FacetType& facet_type)
    -> SemIR::TypeId {
  const auto& info = context.facet_types().Get(facet_type.facet_type_id);
  auto stripped_info = SemIR::FacetTypeInfo::ExtendedOnly(info);
  stripped_info.Canonicalize();
  return GetFacetType(context, stripped_info);
}

static auto GetPeriodSelfType(Context& context,
                              SemIR::TypeId facet_type_type_id)
    -> SemIR::TypeId {
  if (auto facet_type =
          context.types().TryGetAs<SemIR::FacetType>(facet_type_type_id)) {
    return GetExtendedOnlyFacetType(context, *facet_type);
  } else if (facet_type_type_id == SemIR::TypeType::TypeId) {
    // The self may be `TypeType` in `type where X impls Y`, so we use an empty
    // facet type.
    return GetEmptyFacetType(context);
  } else {
    CARBON_CHECK(facet_type_type_id == SemIR::ErrorInst::TypeId,
                 "unexpected .Self type {0}", facet_type_type_id);
    return SemIR::ErrorInst::TypeId;
  }
}

auto HandleParseNode(Context& context, Parse::WhereOperandId node_id) -> bool {
  // The expression at the top of the stack represents a constraint type that
  // is being modified by the `where` operator. It would be `MyInterface` in
  // `MyInterface where .Member = i32`.
  auto [self_node, self_id] = context.node_stack().PopExprWithNodeId();
  auto self_with_constraints_type_id =
      ExprAsType(context, self_node, self_id).type_id;
  // Only facet types may have `where` restrictions.
  if (!context.types().IsFacetTypeOrError(self_with_constraints_type_id)) {
    CARBON_DIAGNOSTIC(WhereOnNonFacetType, Error,
                      "left argument of `where` operator must be a facet type");
    context.emitter().Emit(self_node, WhereOnNonFacetType);
    self_with_constraints_type_id = SemIR::ErrorInst::TypeId;
  }
  if (self_with_constraints_type_id == SemIR::ErrorInst::TypeId) {
    // Keep `self_id` in sync with `self_with_constraints_type_id`, if one is an
    //  error they both are. Note that ExprAsType may have returned ErrorInst,
    //  or we may have set it to ErrorInst in this function.
    self_id = SemIR::ErrorInst::InstId;
  }

  // Strip off any constraints provided by a `WhereExpr` from the `Self` facet
  // type. For a facet type like `I & J where .X = .Y`, this will reduce it down
  // to just `I & J`.
  //
  // Any references to `.Self` in constraints for the current `WhereExpr` will
  // not see constraints in the `Self` facet type, but they will resolve to
  // values through the constraints explicitly when they are combined together.
  auto period_self_type_id =
      GetPeriodSelfType(context, self_with_constraints_type_id);

  // Introduce a name scope so that we can remove the `.Self` entry we are
  // adding to name lookup at the end of the `where` expression.
  context.scope_stack().PushForSameRegion();
  // Introduce `.Self` as a symbolic binding. Its type is the value of the
  // expression to the left of `where`, so `MyInterface` in the example above.
  MakePeriodSelfFacetValue(context, node_id, period_self_type_id);

  // Going to put each requirement on `args_type_info_stack`, so we can have an
  // inst block with the varying number of requirements but keeping other
  // instructions on the current inst block from the `inst_block_stack()`.
  context.args_type_info_stack().Push();

  // Pass along all the constraints from the base facet type to be added to the
  // resulting facet type.
  context.args_type_info_stack().AddInstId(
      AddInstInNoBlock<SemIR::RequirementBaseFacetType>(
          context, SemIR::LocId(node_id),
          {.base_type_inst_id = context.types().GetAsTypeInstId(self_id)}));

  // Add a context stack for tracking constraints, that will be used to allow
  // later constraints to read from them eagerly.
  context.where_stack().push_back({.loc_id = node_id});

  // Make rewrite constraints from the self facet type available immediately to
  // expressions in rewrite constraints for this `where` expression.
  if (auto self_facet_type = context.types().TryGetAs<SemIR::FacetType>(
          self_with_constraints_type_id)) {
    const auto& base_facet_type_info =
        context.facet_types().Get(self_facet_type->facet_type_id);
    for (const auto& rewrite : base_facet_type_info.rewrite_constraints) {
      if (rewrite.lhs_id != SemIR::ErrorInst::InstId) {
        context.where_stack().back().rewrites.Insert(
            context.constant_values().Get(
                GetImplWitnessAccessWithoutSubstitution(context,
                                                        rewrite.lhs_id)),
            rewrite.rhs_id);
      }
    }
  }

  return true;
}

// Returns whether a designator (`.Self` or `.MemberName`) is present in
// `inst_id`.
static auto FindDesignator(Context& context, SemIR::ConstantId const_id)
    -> bool {
  class SubstFindDesignator : public SubstInstCallbacks {
   public:
    explicit SubstFindDesignator(Context* context, bool* found)
        : SubstInstCallbacks(context), found_(found) {}

    auto Subst(SemIR::InstId& inst_id) -> SubstResult override {
      if (*found_) {
        return FullySubstituted;
      }

      // An error was diagnosed for the where clause already.
      if (inst_id == SemIR::ErrorInst::InstId) {
        *found_ = true;
        return FullySubstituted;
      }

      // TypeType has type TypeType, avoid recursing on its type.
      if (context().insts().Is<SemIR::TypeType>(inst_id)) {
        return FullySubstituted;
      }

      // `.MemberName` is represented as an ImplWitnessAccess through `.Self` so
      // we only need to look for `.Self` here.
      if (IsPeriodSelf(context(), inst_id)) {
        *found_ = true;
        return FullySubstituted;
      }

      return SubstOperands;
    }

    auto Rebuild(SemIR::InstId /*orig_inst_id*/, SemIR::Inst /*new_inst*/)
        -> SemIR::InstId override {
      CARBON_FATAL("unexpected rebuild, no insts should change");
    }

    bool* found_;
  };

  // A facet type may contain designators but they do not constrain this where
  // clause's type.
  if (context.constant_values().InstIs<SemIR::FacetType>(const_id)) {
    return false;
  }

  bool found = false;
  SubstFindDesignator callbacks(&context, &found);
  SubstInst(context, context.constant_values().GetInstId(const_id), callbacks);
  return found;
}

static auto DiagnoseMissingDesignator(Context& context, SemIR::LocId loc_id)
    -> void {
  CARBON_DIAGNOSTIC(WhereWithoutDesignator, Error,
                    "constraint in `where` clause without a designator; "
                    "expected `.Self` or a member access like `.M`");
  context.emitter().Emit(loc_id, WhereWithoutDesignator);
}

auto HandleParseNode(Context& context, Parse::RequirementEqualId node_id)
    -> bool {
  auto [rhs_node, rhs_id] = context.node_stack().PopExprWithNodeId();
  auto lhs_id = context.node_stack().PopExpr();

  // Rewrites always contain a designator since the LHS must be one. This is
  // checked elsewhere.

  // Convert rhs to type of lhs.
  auto lhs_type_id = context.insts().Get(lhs_id).type_id();
  if (lhs_type_id.is_symbolic()) {
    // If the type of the associated constant is symbolic, we defer conversion
    // until the constraint is resolved, in case it depends on `Self` (which
    // will now be a reference to `.Self`).
    // For now we convert to a value expression eagerly because otherwise we'll
    // often be unable to constant-evaluate the enclosing `where` expression.
    // TODO: Perform the conversion symbolically and add an implicit constraint
    // that this conversion is valid and produces a constant.
    rhs_id = ConvertToValueExpr(context, rhs_id);
  } else {
    rhs_id = ConvertToValueOfType(context, rhs_node, rhs_id,
                                  context.insts().Get(lhs_id).type_id());
  }

  // Build up the list of arguments for the `WhereExpr` inst.
  context.args_type_info_stack().AddInstId(
      AddInstInNoBlock<SemIR::RequirementRewrite>(
          context, node_id, {.lhs_id = lhs_id, .rhs_id = rhs_id}));

  if (lhs_id != SemIR::ErrorInst::InstId) {
    // Track the value of the rewrite so further constraints can use it
    // immediately, before they are evaluated. This happens directly where the
    // `ImplWitnessAccess` that refers to the rewrite constraint would have been
    // created, and the value of the constraint will be used instead.
    context.where_stack().back().rewrites.Insert(
        context.constant_values().Get(
            GetImplWitnessAccessWithoutSubstitution(context, lhs_id)),
        rhs_id);
  }
  return true;
}

auto HandleParseNode(Context& context, Parse::RequirementEqualEqualId node_id)
    -> bool {
  auto rhs = context.node_stack().PopExpr();
  auto lhs = context.node_stack().PopExpr();
  // TODO: Type check lhs and rhs are comparable.

  auto const_lhs = context.constant_values().Get(lhs);
  auto const_rhs = context.constant_values().Get(rhs);
  if (!FindDesignator(context, const_lhs) &&
      !FindDesignator(context, const_rhs)) {
    if (const_lhs != SemIR::ErrorInst::ConstantId &&
        const_rhs != SemIR::ErrorInst::ConstantId) {
      DiagnoseMissingDesignator(context, node_id);
    }
    lhs = rhs = SemIR::ErrorInst::InstId;
  }

  // Build up the list of arguments for the `WhereExpr` inst.
  context.args_type_info_stack().AddInstId(
      AddInstInNoBlock<SemIR::RequirementEquivalent>(
          context, node_id, {.lhs_id = lhs, .rhs_id = rhs}));
  return true;
}

// Returns whether `inst_id` is `.Self` or an access into `.Self`, possibly
// nested.
static auto IsPeriodSelfAccess(Context& context, SemIR::InstId inst_id)
    -> bool {
  // Walks through nested `ImplWitnessAccess(LookupImplWitness(...))`
  // instructions until it either finds `.Self` and returns true, or finds
  // anything else and returns false.
  while (true) {
    if (IsPeriodSelf(context, inst_id)) {
      return true;
    }
    // Recurse through ImplWitnessAccess into the self type being accessed.
    auto access = context.insts().TryGetAs<SemIR::ImplWitnessAccess>(
        GetImplWitnessAccessWithoutSubstitution(context, inst_id));
    if (!access) {
      return false;
    }
    auto lookup =
        context.insts().TryGetAs<SemIR::LookupImplWitness>(access->witness_id);
    if (!lookup) {
      return false;
    }
    inst_id = lookup->query_self_inst_id;
  }
}

auto HandleParseNode(Context& context, Parse::RequirementImplsId node_id)
    -> bool {
  auto [rhs_node, rhs_id] = context.node_stack().PopExprWithNodeId();
  auto [lhs_node, lhs_id] = context.node_stack().PopExprWithNodeId();

  auto const_lhs = context.constant_values().Get(lhs_id);
  auto const_rhs = context.constant_values().Get(rhs_id);
  if (!FindDesignator(context, const_lhs)) {
    // The RHS of an `impls` may be another `where`. We require a designator to
    // be present in each constraint created from the LHS of that `where`, which
    // equates to requiring a designator in each extend constraint of the facet
    // type.
    //
    // If a designator is part of the LHS of the `impls` or the LHS of the inner
    // `where`, then that implies all constraints nested within the `where`
    // clause will constrain the top level type in some way.
    if (auto facet_type =
            context.constant_values().TryGetInstAs<SemIR::FacetType>(
                const_rhs)) {
      const auto& info = context.facet_types().Get(facet_type->facet_type_id);
      for (auto extend : llvm::concat<SemIR::SpecificId>(
               llvm::map_range(
                   info.extend_constraints,
                   [](SemIR::SpecificInterface i) { return i.specific_id; }),
               llvm::map_range(info.extend_named_constraints,
                               [](SemIR::SpecificNamedConstraint c) {
                                 return c.specific_id;
                               }))) {
        bool found_designator = false;
        for (auto inst_id : context.inst_blocks().Get(
                 context.specifics().GetArgsOrEmpty(extend))) {
          if (FindDesignator(context, context.constant_values().Get(inst_id))) {
            found_designator = true;
            break;
          }
        }
        if (!found_designator) {
          if (const_lhs != SemIR::ErrorInst::ConstantId &&
              const_rhs != SemIR::ErrorInst::ConstantId) {
            DiagnoseMissingDesignator(context, node_id);
          }
          lhs_id = rhs_id = SemIR::ErrorInst::InstId;
          const_lhs = const_rhs = SemIR::ErrorInst::ConstantId;
          break;
        }
      }
    }
  }

  // Check lhs is a facet and rhs is a facet type.
  auto lhs_as_type = ExprAsType(context, lhs_node, lhs_id);
  auto rhs_as_type = ExprAsType(context, rhs_node, rhs_id);
  if (rhs_as_type.type_id != SemIR::ErrorInst::TypeId &&
      !context.types().IsFacetType(rhs_as_type.type_id)) {
    CARBON_DIAGNOSTIC(
        ImplsOnNonFacetType, Error,
        "right argument of `impls` requirement must be a facet type");
    context.emitter().Emit(rhs_node, ImplsOnNonFacetType);
    rhs_as_type.type_id = SemIR::ErrorInst::TypeId;
    rhs_as_type.inst_id = SemIR::ErrorInst::TypeInstId;
  }
  // TODO: For things like `HashSet(.T) as type`, add an implied constraint
  // that `.T impls Hash`.

  // Build up the list of arguments for the `WhereExpr` inst.
  context.args_type_info_stack().AddInstId(
      AddInstInNoBlock<SemIR::RequirementImpls>(
          context, node_id,
          {.lhs_id = lhs_as_type.inst_id, .rhs_id = rhs_as_type.inst_id}));

  if (lhs_as_type.type_id != SemIR::ErrorInst::TypeId &&
      rhs_as_type.type_id != SemIR::ErrorInst::TypeId &&
      rhs_as_type.type_id != SemIR::TypeType::TypeId) {
    // Track the impls relationship so further constraints can use it
    // immediately, before they are evaluated. Impl lookup will search the top
    // of the stack.
    context.where_stack().back().impls.push_back({
        context.constant_values().Get(lhs_as_type.inst_id),
        context.constant_values().Get(rhs_as_type.inst_id),
    });

    // Track any rewrites that are inherited from the impls constraint as the
    // LHS can be referring to `.Self` or a member of it, which makes those
    // rewrites modification of this facet type's self.
    if (IsPeriodSelfAccess(context, lhs_as_type.inst_id)) {
      auto facet_type =
          context.types().GetAs<SemIR::FacetType>(rhs_as_type.type_id);
      const auto& facet_type_info =
          context.facet_types().Get(facet_type.facet_type_id);
      for (const auto& rewrite : facet_type_info.rewrite_constraints) {
        auto lhs = SubstPeriodSelf(
            context, rhs_node, context.constant_values().Get(rewrite.lhs_id),
            context.constant_values().Get(lhs_as_type.inst_id));
        context.where_stack().back().rewrites.Insert(lhs, rewrite.rhs_id);
      }
    }
  }
  return true;
}

auto HandleParseNode(Context& /*context*/, Parse::RequirementAndId /*node_id*/)
    -> bool {
  // Nothing to do.
  return true;
}

// There are two ways to nest `where` expressions, this diagnoses a `where`
// expression inside the RHS of another `where` expression.
//
// Whereas it is valid to nest a `where` expression on the LHS of another
// `where` expression.
static auto DiagnoseNestedWhere(Context& context, SemIR::LocId loc_id,
                                SemIR::LocId outer_loc_id) -> void {
  CARBON_DIAGNOSTIC(
      NestedWhereInsideWhere, Error,
      "found `where` expression nested on the right-hand side of `where`");
  auto builder = context.emitter().Build(loc_id, NestedWhereInsideWhere);
  CARBON_DIAGNOSTIC(NestedWhereInsideWhereOuterNote, Note,
                    "on right-hand side of `where` here");
  builder.Note(outer_loc_id, NestedWhereInsideWhereOuterNote);
  builder.Emit();
}

auto HandleParseNode(Context& context, Parse::WhereExprId node_id) -> bool {
  context.where_stack().pop_back();
  // Remove `PeriodSelf` from name lookup, undoing the `Push` done for the
  // `WhereOperand`.
  context.scope_stack().Pop(/*check_unused=*/true);
  SemIR::InstBlockId requirements_id = context.args_type_info_stack().Pop();

  auto type_id = SemIR::TypeType::TypeId;
  if (!context.where_stack().empty()) {
    DiagnoseNestedWhere(context, node_id, context.where_stack().back().loc_id);
    type_id = SemIR::ErrorInst::TypeId;
  }

  // TODO: Look at the constant value and diagnose NestedWhereInsideWhere if
  // there are any non-extend constraints present.
  AddInstAndPush<SemIR::WhereExpr>(
      context, node_id,
      {.type_id = type_id, .requirements_id = requirements_id});
  return true;
}

}  // namespace Carbon::Check
