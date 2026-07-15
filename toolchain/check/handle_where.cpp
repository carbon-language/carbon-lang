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
    auto extended_id = GetExtendedOnlyFacetType(context, *facet_type);
    auto frozen_const_id =
        FreezePeriodSelf(context, extended_id.AsConstantId());
    return context.types().GetTypeIdForTypeConstantId(frozen_const_id);
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
  auto period_self =
      MakePeriodSelfFacetValue(context, node_id, period_self_type_id);

  // Going to put each requirement on `args_type_info_stack`, so we can have an
  // inst block with the varying number of requirements but keeping other
  // instructions on the current inst block from the `inst_block_stack()`.
  context.args_type_info_stack().Push();

  // Pass along all the constraints from the base facet type to be added to the
  // resulting facet type.
  context.args_type_info_stack().AddInstId(
      AddInst<SemIR::RequirementBaseFacetType>(
          context, SemIR::LocId(node_id),
          {.base_type_inst_id = context.types().GetAsTypeInstId(self_id)}));

  // Add a context stack for tracking constraints, that will be used to allow
  // later constraints to read from them eagerly.
  context.where_stack().push_back({.loc_id = node_id});

  if (auto self_facet_type = context.types().TryGetAs<SemIR::FacetType>(
          self_with_constraints_type_id)) {
    const auto& base_facet_type_info =
        context.facet_types().Get(self_facet_type->facet_type_id);
    // Make rewrite constraints from the self facet type available immediately
    // to expressions in rewrite constraints for this `where` expression.
    //
    // Note that the where_stack rewrites need to be frozen. The rewrites in
    // the base facet type will be thawed since their `WhereExpr` would have
    // already been handled, so we need to freeze them again here.
    for (const auto& rewrite : base_facet_type_info.rewrite_constraints) {
      if (rewrite.lhs_id != SemIR::ErrorInst::InstId) {
        auto const_id = context.constant_values().Get(
            GetImplWitnessAccessWithoutSubstitution(context, rewrite.lhs_id));
        auto frozen_const_id = FreezePeriodSelf(context, const_id);
        context.where_stack().back().rewrites.Insert(frozen_const_id,
                                                     rewrite.rhs_id);
      }
    }

    // Make impls (non-extend) constraints from the self facet type available
    // immediately for this `where` expression, since only extend constraints
    // are preserved in the facet type of `.Self`.
    //
    // Note that the where_stack rewrites need to be frozen. The rewrites in the
    // base facet type will be thawed since their `WhereExpr` would have already
    // been handled, so we need to freeze them again here. Note that
    // `period_self` is already frozen since it is created in that state.
    for (const auto& impls : base_facet_type_info.self_impls_constraints) {
      auto self_frozen_const_id = context.constant_values().Get(period_self);
      auto type_const_id =
          GetInterfaceType(context, impls.interface_id, impls.specific_id)
              .AsConstantId();
      auto type_frozen_const_id = FreezePeriodSelf(context, type_const_id);
      context.where_stack().back().impls.push_back(
          {.self_const_id = self_frozen_const_id,
           .facet_type_const_id = type_frozen_const_id});
    }
    for (const auto& impls :
         base_facet_type_info.self_impls_named_constraints) {
      auto self_frozen_const_id = context.constant_values().Get(period_self);
      auto type_const_id =
          GetNamedConstraintType(context, impls.named_constraint_id,
                                 impls.specific_id)
              .AsConstantId();
      auto type_frozen_const_id = FreezePeriodSelf(context, type_const_id);
      context.where_stack().back().impls.push_back(
          {.self_const_id = self_frozen_const_id,
           .facet_type_const_id = type_frozen_const_id});
    }
    for (const auto& type_impls : base_facet_type_info.type_impls_interfaces) {
      auto self_const_id = context.constant_values().Get(type_impls.self_type);
      auto self_frozen_const_id = FreezePeriodSelf(context, self_const_id);
      auto type_const_id =
          GetInterfaceType(context, type_impls.specific_interface.interface_id,
                           type_impls.specific_interface.specific_id)
              .AsConstantId();
      auto type_frozen_const_id = FreezePeriodSelf(context, type_const_id);
      context.where_stack().back().impls.push_back(
          {.self_const_id = self_frozen_const_id,
           .facet_type_const_id = type_frozen_const_id});
    }
    for (const auto& type_impls :
         base_facet_type_info.type_impls_named_constraints) {
      auto self_const_id = context.constant_values().Get(type_impls.self_type);
      auto self_frozen_const_id = FreezePeriodSelf(context, self_const_id);
      auto type_const_id =
          GetNamedConstraintType(
              context, type_impls.specific_named_constraint.named_constraint_id,
              type_impls.specific_named_constraint.specific_id)
              .AsConstantId();
      auto type_frozen_const_id = FreezePeriodSelf(context, type_const_id);
      context.where_stack().back().impls.push_back(
          {.self_const_id = self_frozen_const_id,
           .facet_type_const_id = type_frozen_const_id});
    }
  }

  return true;
}

// Returns whether a designator (`.Self` or `.MemberName`) is present in
// `inst_id` in a way that will constrain the current `.Self`.
static auto FindDesignator(Context& context, SemIR::InstId inst_id) -> bool {
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
      if (inst_id == SemIR::TypeType::TypeInstId) {
        return FullySubstituted;
      }

      // Arguments to a call do not count, since a call with `.Self` in it will
      // not be evaluated inside the facet type.
      if (context().insts().Is<SemIR::Call>(inst_id)) {
        return FullySubstituted;
      }

      // TODO: When we support parameterized aliases, if an argument has
      // `.Self`, we will need to evaluate the alias here and look for `.Self`
      // in the constant value.

      // `.MemberName` is represented as an ImplWitnessAccess through `.Self` so
      // we only need to look for `.Self` here.
      if (IsPeriodSelf(context(), inst_id, /*canonicalize=*/false)) {
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

  bool found = false;
  SubstFindDesignator callbacks(&context, &found);
  SubstInst(context, inst_id, callbacks);
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
  context.args_type_info_stack().AddInstId(AddInst<SemIR::RequirementRewrite>(
      context, node_id, {.lhs_id = lhs_id, .rhs_id = rhs_id}));

  if (lhs_id != SemIR::ErrorInst::InstId) {
    // Track the value of the rewrite so further constraints can use it
    // immediately, before they are evaluated. This happens directly where the
    // `ImplWitnessAccess` that refers to the rewrite constraint would have been
    // created, and the value of the constraint will be used instead.
    //
    // Note that the where_stack rewrites need to be frozen. Since this
    // expression is inside of facet type construction, it will already be
    // frozen.
    context.where_stack().back().rewrites.Insert(
        context.constant_values().Get(
            GetImplWitnessAccessWithoutSubstitution(context, lhs_id)),
        rhs_id);
  }
  return true;
}

auto HandleParseNode(Context& context, Parse::RequirementEqualEqualId node_id)
    -> bool {
  auto rhs_id = context.node_stack().PopExpr();
  auto lhs_id = context.node_stack().PopExpr();
  // TODO: Type check lhs and rhs are comparable.

  if (!FindDesignator(context, lhs_id) && !FindDesignator(context, rhs_id)) {
    if (context.constant_values().Get(lhs_id) != SemIR::ErrorInst::ConstantId &&
        context.constant_values().Get(rhs_id) != SemIR::ErrorInst::ConstantId) {
      DiagnoseMissingDesignator(context, node_id);
    }
    lhs_id = rhs_id = SemIR::ErrorInst::InstId;
  }

  // Build up the list of arguments for the `WhereExpr` inst.
  context.args_type_info_stack().AddInstId(
      AddInst<SemIR::RequirementEquivalent>(
          context, node_id, {.lhs_id = lhs_id, .rhs_id = rhs_id}));
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

static auto FindDesignatorInSpecific(Context& context,
                                     SemIR::SpecificId specific_id) -> bool {
  for (auto inst_id : context.inst_blocks().Get(
           context.specifics().GetArgsOrEmpty(specific_id))) {
    if (FindDesignator(context, inst_id)) {
      return true;
    }
  }
  return false;
}

static auto FindDesignatorInEveryExtendConstraint(Context& context,
                                                  SemIR::FacetType facet_type)
    -> bool {
  const auto& info = context.facet_types().Get(facet_type.facet_type_id);

  for (const auto& extend : info.extend_constraints) {
    if (!FindDesignatorInSpecific(context, extend.specific_id)) {
      return false;
    }
  }
  for (const auto& extend : info.extend_named_constraints) {
    if (!FindDesignatorInSpecific(context, extend.specific_id)) {
      return false;
    }
  }
  return !info.extend_constraints.empty() ||
         !info.extend_named_constraints.empty();
}

auto HandleParseNode(Context& context, Parse::RequirementImplsId node_id)
    -> bool {
  auto [rhs_node, rhs_id] = context.node_stack().PopExprWithNodeId();
  auto [lhs_node, lhs_id] = context.node_stack().PopExprWithNodeId();

  if (!FindDesignator(context, lhs_id)) {
    bool found_designator = false;
    auto const_rhs_id = context.constant_values().Get(rhs_id);
    if (auto facet_type =
            context.constant_values().TryGetInstAs<SemIR::FacetType>(
                const_rhs_id)) {
      found_designator =
          FindDesignatorInEveryExtendConstraint(context, *facet_type);
    }
    if (!found_designator) {
      auto const_lhs_id = context.constant_values().Get(lhs_id);
      if (const_lhs_id != SemIR::ErrorInst::ConstantId &&
          const_rhs_id != SemIR::ErrorInst::ConstantId) {
        // TODO: Can we diagnose the specific constraint that was missing the
        // `.Self`?
        DiagnoseMissingDesignator(context, node_id);
      }
      lhs_id = rhs_id = SemIR::ErrorInst::InstId;
    }
  }

  // Check lhs is a facet and rhs is a facet type.
  auto lhs_as_type = ExprAsType(context, lhs_node, lhs_id);
  auto rhs_as_type = ExprAsType(context, rhs_node, rhs_id);
  if (rhs_as_type.type_id != SemIR::ErrorInst::TypeId &&
      !context.types().IsFacetType(rhs_as_type.type_id)) {
    DiagnoseImplsOnNonFacetType(context, rhs_node);
    rhs_as_type.type_id = SemIR::ErrorInst::TypeId;
    rhs_as_type.inst_id = SemIR::ErrorInst::TypeInstId;
  }
  // TODO: For things like `HashSet(.T) as type`, add an implied constraint
  // that `.T impls Hash`.

  // Build up the list of arguments for the `WhereExpr` inst.
  context.args_type_info_stack().AddInstId(AddInst<SemIR::RequirementImpls>(
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
    //
    // Note that the where_stack rewrites need to be frozen. Since this
    // expression is inside of facet type construction, it will already be
    // frozen.
    //
    // TODO: Now that we don't allow nested `where`, there should be no rewrites
    // to add here?
    if (IsPeriodSelfAccess(context, lhs_as_type.inst_id)) {
      auto facet_type =
          context.types().GetAs<SemIR::FacetType>(rhs_as_type.type_id);
      const auto& facet_type_info =
          context.facet_types().Get(facet_type.facet_type_id);
      for (const auto& rewrite : facet_type_info.rewrite_constraints) {
        auto lhs_id = SubstPeriodSelf(
            context, rhs_node, context.constant_values().Get(rewrite.lhs_id),
            context.constant_values().Get(lhs_as_type.inst_id));
        context.where_stack().back().rewrites.Insert(lhs_id, rewrite.rhs_id);
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

// Returns whether the constant value `const_id` contains a facet type
// instruction with a `where` expression.
//
// This search ignores the type_id of insts, and just looks at the `const_id`
// and its non-type operands recursively. Any use of a symbolic facet may have a
// type with an arbitrary facet type (including a `where` expression). But we
// are looking for a nested `where` that is part of the current `WhereExpr`
// being checked.
static auto FindWhere(Context& context, SemIR::ConstantId const_id) -> bool {
  class FindWhereCallbacks : public SubstInstCallbacks {
   public:
    FindWhereCallbacks(Context* context, bool* found)
        : SubstInstCallbacks(context), found_(found) {}

    auto Subst(SemIR::InstId& inst_id) -> SubstResult override {
      if (*found_ || inst_id == SemIR::TypeType::TypeInstId ||
          inst_id == SemIR::ErrorInst::InstId) {
        return FullySubstituted;
      }

      // Facet types can contain many references to the same value. Only search
      // a given constant one time to avoid exponential costs.
      if (!searched_.Insert(inst_id).is_inserted()) {
        return FullySubstituted;
      }

      if (auto facet_type =
              context().insts().TryGetAs<SemIR::FacetType>(inst_id)) {
        const auto& info =
            context().facet_types().Get(facet_type->facet_type_id);
        if (!info.IsExtendedOnly()) {
          *found_ = true;
          return FullySubstituted;
        }
      }

      return SubstOperandsSkipType;
    }

    auto Rebuild(SemIR::InstId orig_inst_id, SemIR::Inst /*new_inst*/)
        -> SemIR::InstId override {
      CARBON_FATAL("unexpected rebuild of inst {0}",
                   context().insts().Get(orig_inst_id));
    }

   private:
    bool* found_;
    Set<SemIR::InstId> searched_;
  };

  if (!const_id.is_constant()) {
    return false;
  }

  bool found = false;
  FindWhereCallbacks callbacks(&context, &found);
  SubstInst(context, context.constant_values().GetInstId(const_id), callbacks);
  return found;
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

// Look for nested `where` expressions on the RHS of the current `where` after
// eval. If found, it is diagnosed and replaced with ErrorInst.
static auto CheckForNestedWhereInRequirementsAfterEval(
    Context& context, SemIR::LocId where_loc,
    SemIR::InstBlockId requirements_id) -> SemIR::InstBlockId {
  bool diagnosed = false;

  // The requirements block, but we replace invalid operands with ErrorInst.
  llvm::SmallVector<SemIR::InstId> checked_requirements(
      context.inst_blocks().Get(requirements_id));

  for (auto& inst_id : checked_requirements) {
    // Searches the `lhs_id` and `rhs_id` operands of the requirement inst. If a
    // nested `where` is found and diagnosed, the requirement is rebuilt with an
    // ErrorInst in its place and it replaces the `inst_id` in the requirements
    // block.
    auto find_and_diagnose_nested_where = [&](auto req_inst, bool check_lhs) {
      bool found = false;
      if (check_lhs &&
          FindWhere(context, context.constant_values().Get(req_inst.lhs_id))) {
        DiagnoseNestedWhere(context, SemIR::LocId(req_inst.lhs_id), where_loc);
        req_inst.lhs_id = SemIR::ErrorInst::InstId;
        found = diagnosed = true;
      }
      if (FindWhere(context, context.constant_values().Get(req_inst.rhs_id))) {
        DiagnoseNestedWhere(context, SemIR::LocId(req_inst.rhs_id), where_loc);
        req_inst.rhs_id = SemIR::ErrorInst::InstId;
        found = diagnosed = true;
      }
      if (found) {
        inst_id = AddInst(
            context, SemIR::LocIdAndInst::RuntimeVerified(
                         context.sem_ir(), SemIR::LocId(inst_id), req_inst));
      }
    };

    auto inst = context.insts().Get(inst_id);
    CARBON_KIND_SWITCH(inst) {
      case CARBON_KIND(SemIR::RequirementBaseFacetType _): {
        // Nested `where` is allowed on the LHS of a `where` expression.
        break;
      }
      case CARBON_KIND(SemIR::RequirementImpls impls): {
        find_and_diagnose_nested_where(impls, true);
        break;
      }
      case CARBON_KIND(SemIR::RequirementRewrite rewrite): {
        // The LHS of a rewrite can't have a `where` inside it, so we skip
        // checking it.
        find_and_diagnose_nested_where(rewrite, false);
        break;
      }
      case CARBON_KIND(SemIR::RequirementEquivalent equiv): {
        find_and_diagnose_nested_where(equiv, true);
        break;
      }
      default:
        CARBON_FATAL("unexpected `where` requirement inst {0}", inst);
    }
  }

  if (!diagnosed) {
    return requirements_id;
  }
  return context.inst_blocks().Add(checked_requirements);
}

static auto ThawPeriodSelfInRequirements(Context& context,
                                         SemIR::InstBlockId requirements_id)
    -> SemIR::InstBlockId {
  bool changed = false;
  llvm::SmallVector<SemIR::InstId> ids(
      context.inst_blocks().Get(requirements_id));
  for (SemIR::InstId& inst_id : ids) {
    auto subst_id = ThawPeriodSelf(context, inst_id);
    if (subst_id != inst_id) {
      changed = true;
      inst_id = subst_id;
    }
  }
  if (changed) {
    return context.inst_blocks().Add(ids);
  }
  return requirements_id;
}

auto HandleParseNode(Context& context, Parse::WhereExprId node_id) -> bool {
  auto where_loc = context.where_stack().back().loc_id;
  context.where_stack().pop_back();
  // Remove `PeriodSelf` from name lookup, undoing the `Push` done for the
  // `WhereOperand`.
  context.scope_stack().Pop(/*check_unused=*/true);
  SemIR::InstBlockId requirements_id = context.args_type_info_stack().Pop();

  auto type_id = SemIR::TypeType::TypeId;
  if (!context.where_stack().empty()) {
    // This `where` expression is nested on the RHS of another `where`, which is
    // an error.
    DiagnoseNestedWhere(context, node_id, context.where_stack().back().loc_id);
    type_id = SemIR::ErrorInst::TypeId;
  }
  requirements_id = CheckForNestedWhereInRequirementsAfterEval(
      context, where_loc, requirements_id);
  requirements_id = ThawPeriodSelfInRequirements(context, requirements_id);

  AddInstAndPush<SemIR::WhereExpr>(
      context, node_id,
      {.type_id = type_id, .requirements_id = requirements_id});
  return true;
}

}  // namespace Carbon::Check
