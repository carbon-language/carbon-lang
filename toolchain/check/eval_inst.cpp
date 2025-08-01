// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/eval_inst.h"

#include <variant>

#include "toolchain/check/action.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/impl_lookup.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/parse/typed_nodes.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/expr_info.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/pattern.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Performs an access into an aggregate, retrieving the specified element.
static auto PerformAggregateAccess(Context& context, SemIR::Inst inst)
    -> ConstantEvalResult {
  auto access_inst = inst.As<SemIR::AnyAggregateAccess>();
  if (auto aggregate = context.insts().TryGetAs<SemIR::AnyAggregateValue>(
          access_inst.aggregate_id)) {
    auto elements = context.inst_blocks().Get(aggregate->elements_id);
    auto index = static_cast<size_t>(access_inst.index.index);
    CARBON_CHECK(index < elements.size(), "Access out of bounds.");
    // `Phase` is not used here. If this element is a concrete constant, then
    // so is the result of indexing, even if the aggregate also contains a
    // symbolic context.
    return ConstantEvalResult::Existing(
        context.constant_values().Get(elements[index]));
  }

  return ConstantEvalResult::NewSamePhase(inst);
}

auto EvalConstantInst(Context& /*context*/, SemIR::ArrayInit inst)
    -> ConstantEvalResult {
  // TODO: Add an `ArrayValue` to represent a constant array object
  // representation instead of using a `TupleValue`.
  return ConstantEvalResult::NewSamePhase(
      SemIR::TupleValue{.type_id = inst.type_id, .elements_id = inst.inits_id});
}

auto EvalConstantInst(Context& context, SemIR::InstId inst_id,
                      SemIR::ArrayType inst) -> ConstantEvalResult {
  auto bound_inst = context.insts().Get(inst.bound_id);
  auto int_bound = bound_inst.TryAs<SemIR::IntValue>();
  if (!int_bound) {
    CARBON_CHECK(context.constant_values().Get(inst.bound_id).is_symbolic(),
                 "Unexpected inst {0} for template constant int", bound_inst);
    return ConstantEvalResult::NewSamePhase(inst);
  }
  // TODO: We should check that the size of the resulting array type
  // fits in 64 bits, not just that the bound does. Should we use a
  // 32-bit limit for 32-bit targets?
  const auto& bound_val = context.ints().Get(int_bound->int_id);
  if (context.types().IsSignedInt(int_bound->type_id) &&
      bound_val.isNegative()) {
    CARBON_DIAGNOSTIC(ArrayBoundNegative, Error,
                      "array bound of {0} is negative", TypedInt);
    context.emitter().Emit(
        context.insts().GetAs<SemIR::ArrayType>(inst_id).bound_id,
        ArrayBoundNegative, {.type = int_bound->type_id, .value = bound_val});
    return ConstantEvalResult::Error;
  }
  if (bound_val.getActiveBits() > 64) {
    CARBON_DIAGNOSTIC(ArrayBoundTooLarge, Error,
                      "array bound of {0} is too large", TypedInt);
    context.emitter().Emit(
        context.insts().GetAs<SemIR::ArrayType>(inst_id).bound_id,
        ArrayBoundTooLarge, {.type = int_bound->type_id, .value = bound_val});
    return ConstantEvalResult::Error;
  }
  return ConstantEvalResult::NewSamePhase(inst);
}

auto EvalConstantInst(Context& context, SemIR::AsCompatible inst)
    -> ConstantEvalResult {
  // AsCompatible changes the type of the source instruction; its constant
  // value, if there is one, needs to be modified to be of the same type.
  auto value_id = context.constant_values().Get(inst.source_id);
  CARBON_CHECK(value_id.is_constant());

  auto value_inst =
      context.insts().Get(context.constant_values().GetInstId(value_id));
  value_inst.SetType(inst.type_id);
  return ConstantEvalResult::NewAnyPhase(value_inst);
}

auto EvalConstantInst(Context& context, SemIR::BindAlias inst)
    -> ConstantEvalResult {
  // An alias evaluates to the value it's bound to.
  return ConstantEvalResult::Existing(
      context.constant_values().Get(inst.value_id));
}

auto EvalConstantInst(Context& context, SemIR::BindName inst)
    -> ConstantEvalResult {
  // A reference binding evaluates to the value it's bound to.
  if (inst.value_id.has_value() && SemIR::IsRefCategory(SemIR::GetExprCategory(
                                       context.sem_ir(), inst.value_id))) {
    return ConstantEvalResult::Existing(
        context.constant_values().Get(inst.value_id));
  }

  // Non-`:!` value bindings are not constant.
  return ConstantEvalResult::NotConstant;
}

auto EvalConstantInst(Context& /*context*/, SemIR::BindValue /*inst*/)
    -> ConstantEvalResult {
  // TODO: Handle this once we've decided how to represent constant values of
  // reference expressions.
  return ConstantEvalResult::TODO;
}

auto EvalConstantInst(Context& context, SemIR::ClassElementAccess inst)
    -> ConstantEvalResult {
  return PerformAggregateAccess(context, inst);
}

auto EvalConstantInst(Context& context, SemIR::ClassDecl inst)
    -> ConstantEvalResult {
  const auto& class_info = context.classes().Get(inst.class_id);

  // If the class has generic parameters, we don't produce a class type, but a
  // callable whose return value is a class type.
  if (class_info.has_parameters()) {
    return ConstantEvalResult::NewSamePhase(SemIR::StructValue{
        .type_id = inst.type_id, .elements_id = SemIR::InstBlockId::Empty});
  }

  // A non-generic class declaration evaluates to the class type.
  return ConstantEvalResult::NewAnyPhase(SemIR::ClassType{
      .type_id = SemIR::TypeType::TypeId,
      .class_id = inst.class_id,
      .specific_id =
          context.generics().GetSelfSpecific(class_info.generic_id)});
}

auto EvalConstantInst(Context& /*context*/, SemIR::ClassInit inst)
    -> ConstantEvalResult {
  // TODO: Add a `ClassValue` to represent a constant class object
  // representation instead of using a `StructValue`.
  return ConstantEvalResult::NewSamePhase(SemIR::StructValue{
      .type_id = inst.type_id, .elements_id = inst.elements_id});
}

auto EvalConstantInst(Context& context, SemIR::ConstType inst)
    -> ConstantEvalResult {
  // `const (const T)` evaluates to `const T`.
  if (context.insts().Is<SemIR::ConstType>(inst.inner_id)) {
    return ConstantEvalResult::Existing(
        context.constant_values().Get(inst.inner_id));
  }
  // Otherwise, `const T` evaluates to itself.
  return ConstantEvalResult::NewSamePhase(inst);
}

auto EvalConstantInst(Context& /*context*/, SemIR::PartialType inst)
    -> ConstantEvalResult {
  return ConstantEvalResult::NewSamePhase(inst);
}

auto EvalConstantInst(Context& context, SemIR::Converted inst)
    -> ConstantEvalResult {
  // A conversion evaluates to the result of the conversion.
  return ConstantEvalResult::Existing(
      context.constant_values().Get(inst.result_id));
}

auto EvalConstantInst(Context& /*context*/, SemIR::Deref /*inst*/)
    -> ConstantEvalResult {
  // TODO: Handle this.
  return ConstantEvalResult::TODO;
}

auto EvalConstantInst(Context& context, SemIR::ExportDecl inst)
    -> ConstantEvalResult {
  // An export instruction evaluates to the exported declaration.
  return ConstantEvalResult::Existing(
      context.constant_values().Get(inst.value_id));
}

auto EvalConstantInst(Context& context, SemIR::FacetAccessType inst)
    -> ConstantEvalResult {
  if (auto facet_value = context.insts().TryGetAs<SemIR::FacetValue>(
          inst.facet_value_inst_id)) {
    return ConstantEvalResult::Existing(
        context.constant_values().Get(facet_value->type_inst_id));
  }
  return ConstantEvalResult::NewSamePhase(inst);
}

auto EvalConstantInst(Context& context, SemIR::InstId inst_id,
                      SemIR::FloatType inst) -> ConstantEvalResult {
  return ValidateFloatType(context, SemIR::LocId(inst_id), inst)
             ? ConstantEvalResult::NewSamePhase(inst)
             : ConstantEvalResult::Error;
}

auto EvalConstantInst(Context& /*context*/, SemIR::FunctionDecl inst)
    -> ConstantEvalResult {
  // A function declaration evaluates to a function object, which is an empty
  // object of function type.
  // TODO: Eventually we may need to handle captures here.
  return ConstantEvalResult::NewSamePhase(SemIR::StructValue{
      .type_id = inst.type_id, .elements_id = SemIR::InstBlockId::Empty});
}

auto EvalConstantInst(Context& context, SemIR::InstId inst_id,
                      SemIR::LookupImplWitness inst) -> ConstantEvalResult {
  // The self value is canonicalized in order to produce a canonical
  // LookupImplWitness instruction. We save the non-canonical instruction as it
  // may be a concrete `FacetValue` that contains a concrete witness.
  auto non_canonical_query_self_inst_id = inst.query_self_inst_id;
  inst.query_self_inst_id =
      GetCanonicalizedFacetOrTypeValue(context, inst.query_self_inst_id);

  auto result = EvalLookupSingleImplWitness(
      context, SemIR::LocId(inst_id), inst, non_canonical_query_self_inst_id,
      /*poison_concrete_results=*/true);
  if (!result.has_value()) {
    // We use NotConstant to communicate back to impl lookup that the lookup
    // failed. This can not happen for a deferred symbolic lookup in a generic
    // eval block, since we only add the deferred lookup instruction (being
    // evaluated here) to the SemIR if the lookup succeeds.
    return ConstantEvalResult::NotConstant;
  }
  if (!result.has_concrete_value()) {
    return ConstantEvalResult::NewSamePhase(inst);
  }
  return ConstantEvalResult::Existing(
      context.constant_values().Get(result.concrete_witness()));
}

auto EvalConstantInst(Context& context, SemIR::InstId inst_id,
                      SemIR::ImplWitnessAccess inst) -> ConstantEvalResult {
  if (auto witness =
          context.insts().TryGetAs<SemIR::ImplWitness>(inst.witness_id)) {
    // This is PerformAggregateAccess followed by GetConstantValueInSpecific.
    auto witness_table = context.insts().GetAs<SemIR::ImplWitnessTable>(
        witness->witness_table_id);
    auto elements = context.inst_blocks().Get(witness_table.elements_id);
    // `elements` can be empty if there is only a forward declaration of the
    // impl.
    if (!elements.empty()) {
      auto index = static_cast<size_t>(inst.index.index);
      CARBON_CHECK(index < elements.size(), "Access out of bounds.");
      auto element = elements[index];
      if (element.has_value()) {
        LoadImportRef(context, element);
        return ConstantEvalResult::Existing(GetConstantValueInSpecific(
            context.sem_ir(), witness->specific_id, element));
      }
    }
    CARBON_DIAGNOSTIC(
        ImplAccessMemberBeforeSet, Error,
        "accessing member from impl before it has a defined value");
    // TODO: Add note pointing to the impl declaration.
    context.emitter().Emit(inst_id, ImplAccessMemberBeforeSet);
    return ConstantEvalResult::Error;
  } else if (auto witness = context.insts().TryGetAs<SemIR::LookupImplWitness>(
                 inst.witness_id)) {
    // If the witness is symbolic but has a self type that is a FacetType, it
    // can pull rewrite values from the self type. If the access is for one of
    // those rewrites, evaluate to the RHS of the rewrite.

    auto witness_self_type_id =
        context.insts().Get(witness->query_self_inst_id).type_id();
    if (!context.types().Is<SemIR::FacetType>(witness_self_type_id)) {
      return ConstantEvalResult::NewSamePhase(inst);
    }

    // The `ImplWitnessAccess` is accessing a value, by index, for this
    // interface.
    auto access_interface_id = witness->query_specific_interface_id;

    auto witness_self_facet_type_id =
        context.types()
            .GetAs<SemIR::FacetType>(witness_self_type_id)
            .facet_type_id;
    // TODO: We could consider something better than linear search here, such as
    // a map. However that would probably require heap allocations which may be
    // worse overall since the number of rewrite constraints is generally low.
    // If the `rewrite_constraints` were sorted so that associated constants are
    // grouped together, as in ResolveFacetTypeRewriteConstraints(), and limited
    // to just the `ImplWitnessAccess` entries, then a binary search may work
    // here.
    for (auto witness_rewrite : context.facet_types()
                                    .Get(witness_self_facet_type_id)
                                    .rewrite_constraints) {
      // Look at each rewrite constraint in the self facet value's type. If the
      // LHS is an `ImplWitnessAccess` into the same interface that `inst` is
      // indexing into, then we can use its RHS as the value.
      auto witness_rewrite_lhs_access =
          context.insts().TryGetAs<SemIR::ImplWitnessAccess>(
              witness_rewrite.lhs_id);
      if (!witness_rewrite_lhs_access) {
        continue;
      }
      if (witness_rewrite_lhs_access->index != inst.index) {
        continue;
      }

      auto witness_rewrite_lhs_interface_id =
          context.insts()
              .GetAs<SemIR::LookupImplWitness>(
                  witness_rewrite_lhs_access->witness_id)
              .query_specific_interface_id;
      if (witness_rewrite_lhs_interface_id != access_interface_id) {
        continue;
      }

      // The `ImplWitnessAccess` evaluates to the RHS from the witness self
      // facet value's type.
      return ConstantEvalResult::Existing(
          context.constant_values().Get(witness_rewrite.rhs_id));
    }
  }

  return ConstantEvalResult::NewSamePhase(inst);
}

auto EvalConstantInst(Context& context,
                      SemIR::ImplWitnessAccessSubstituted inst)
    -> ConstantEvalResult {
  return ConstantEvalResult::Existing(
      context.constant_values().Get(inst.value_id));
}

auto EvalConstantInst(Context& context,
                      SemIR::ImplWitnessAssociatedConstant inst)
    -> ConstantEvalResult {
  return ConstantEvalResult::Existing(
      context.constant_values().Get(inst.inst_id));
}

auto EvalConstantInst(Context& /*context*/, SemIR::ImportRefUnloaded inst)
    -> ConstantEvalResult {
  CARBON_FATAL("ImportRefUnloaded should be loaded before TryEvalInst: {0}",
               inst);
}

auto EvalConstantInst(Context& context, SemIR::InitializeFrom inst)
    -> ConstantEvalResult {
  // Initialization is not performed in-place during constant evaluation, so
  // just return the value of the initializer.
  return ConstantEvalResult::Existing(
      context.constant_values().Get(inst.src_id));
}

auto EvalConstantInst(Context& context, SemIR::InstId inst_id,
                      SemIR::IntType inst) -> ConstantEvalResult {
  return ValidateIntType(context, SemIR::LocId(inst_id), inst)
             ? ConstantEvalResult::NewSamePhase(inst)
             : ConstantEvalResult::Error;
}

auto EvalConstantInst(Context& context, SemIR::InterfaceDecl inst)
    -> ConstantEvalResult {
  const auto& interface_info = context.interfaces().Get(inst.interface_id);

  // If the interface has generic parameters, we don't produce an interface
  // type, but a callable whose return value is an interface type.
  if (interface_info.has_parameters()) {
    return ConstantEvalResult::NewSamePhase(SemIR::StructValue{
        .type_id = inst.type_id, .elements_id = SemIR::InstBlockId::Empty});
  }

  // A non-parameterized interface declaration evaluates to a facet type.
  return ConstantEvalResult::NewAnyPhase(FacetTypeFromInterface(
      context, inst.interface_id,
      context.generics().GetSelfSpecific(interface_info.generic_id)));
}

auto EvalConstantInst(Context& context, SemIR::NameRef inst)
    -> ConstantEvalResult {
  // A name reference evaluates to the value the name resolves to.
  return ConstantEvalResult::Existing(
      context.constant_values().Get(inst.value_id));
}

auto EvalConstantInst(Context& context, SemIR::InstId inst_id,
                      SemIR::RequireCompleteType inst) -> ConstantEvalResult {
  auto witness_type_id =
      GetSingletonType(context, SemIR::WitnessType::TypeInstId);

  // If the type is a concrete constant, require it to be complete now.
  auto complete_type_id =
      context.types().GetTypeIdForTypeInstId(inst.complete_type_inst_id);
  if (complete_type_id.is_concrete()) {
    if (!TryToCompleteType(
            context, complete_type_id, SemIR::LocId(inst_id), [&] {
              CARBON_DIAGNOSTIC(IncompleteTypeInMonomorphization, Error,
                                "{0} evaluates to incomplete type {1}",
                                InstIdAsType, InstIdAsType);
              return context.emitter().Build(
                  inst_id, IncompleteTypeInMonomorphization,
                  context.insts()
                      .GetAs<SemIR::RequireCompleteType>(inst_id)
                      .complete_type_inst_id,
                  inst.complete_type_inst_id);
            })) {
      return ConstantEvalResult::Error;
    }
    return ConstantEvalResult::NewSamePhase(SemIR::CompleteTypeWitness{
        .type_id = witness_type_id,
        .object_repr_type_inst_id = context.types().GetInstId(
            context.types().GetObjectRepr(complete_type_id))});
  }

  // If it's not a concrete constant, require it to be complete once it
  // becomes one.
  return ConstantEvalResult::NewSamePhase(inst);
}

auto EvalConstantInst(Context& context, SemIR::SpecificConstant inst)
    -> ConstantEvalResult {
  // Pull the constant value out of the specific.
  return ConstantEvalResult::Existing(SemIR::GetConstantValueInSpecific(
      context.sem_ir(), inst.specific_id, inst.inst_id));
}

auto EvalConstantInst(Context& context, SemIR::InstId inst_id,
                      SemIR::SpecificImplFunction inst) -> ConstantEvalResult {
  auto callee_inst = context.insts().Get(inst.callee_id);
  // If the callee is not a function value, we're not ready to evaluate this
  // yet. Build a symbolic `SpecificImplFunction` constant.
  if (!callee_inst.Is<SemIR::StructValue>()) {
    return ConstantEvalResult::NewSamePhase(inst);
  }
  auto callee_type_id = callee_inst.type_id();
  auto callee_fn_type =
      context.types().TryGetAs<SemIR::FunctionType>(callee_type_id);
  if (!callee_fn_type) {
    return ConstantEvalResult::NewSamePhase(inst);
  }

  // If the callee function found in the impl witness is not generic, the result
  // is simply that function.
  // TODO: We could do this even before the callee is concrete.
  auto generic_id =
      context.functions().Get(callee_fn_type->function_id).generic_id;
  if (!generic_id.has_value()) {
    return ConstantEvalResult::Existing(
        context.constant_values().Get(inst.callee_id));
  }

  // Find the arguments to use.
  auto enclosing_specific_id = callee_fn_type->specific_id;
  auto enclosing_args = context.inst_blocks().Get(
      context.specifics().GetArgsOrEmpty(enclosing_specific_id));
  auto interface_fn_args = context.inst_blocks().Get(
      context.specifics().GetArgsOrEmpty(inst.specific_id));

  // Form new specific for the generic callee function. The arguments for this
  // specific are the enclosing arguments of the callee followed by the
  // remaining arguments from the interface function. Impl checking has ensured
  // that these arguments can also be used for the function in the impl witness.
  auto num_params = context.inst_blocks()
                        .Get(context.generics().Get(generic_id).bindings_id)
                        .size();
  llvm::SmallVector<SemIR::InstId> args;
  args.reserve(num_params);
  args.append(enclosing_args.begin(), enclosing_args.end());
  int remaining_params = num_params - args.size();
  CARBON_CHECK(static_cast<int>(interface_fn_args.size()) >= remaining_params);
  args.append(interface_fn_args.end() - remaining_params,
              interface_fn_args.end());
  auto specific_id =
      MakeSpecific(context, SemIR::LocId(inst_id), generic_id, args);
  context.definitions_required_by_use().push_back(
      {SemIR::LocId(inst_id), specific_id});

  return ConstantEvalResult::NewSamePhase(
      SemIR::SpecificFunction{.type_id = inst.type_id,
                              .callee_id = inst.callee_id,
                              .specific_id = specific_id});
}

auto EvalConstantInst(Context& context, SemIR::InstId inst_id,
                      SemIR::SpecificFunction inst) -> ConstantEvalResult {
  auto callee_function =
      SemIR::GetCalleeFunction(context.sem_ir(), inst.callee_id);
  const auto& fn = context.functions().Get(callee_function.function_id);
  if (!callee_function.self_type_id.has_value() &&
      fn.builtin_function_kind() != SemIR::BuiltinFunctionKind::NoOp &&
      fn.virtual_modifier != SemIR::Function::VirtualModifier::Abstract) {
    // This is not an associated function. Those will be required to be defined
    // as part of checking that the impl is complete.
    context.definitions_required_by_use().push_back(
        {SemIR::LocId(inst_id), inst.specific_id});
  }
  // Create new constant for a specific function.
  return ConstantEvalResult::NewSamePhase(inst);
}

auto EvalConstantInst(Context& context, SemIR::SpliceBlock inst)
    -> ConstantEvalResult {
  // SpliceBlock evaluates to the result value that is (typically) within the
  // block. This can be constant even if the block contains other non-constant
  // instructions.
  return ConstantEvalResult::Existing(
      context.constant_values().Get(inst.result_id));
}

auto EvalConstantInst(Context& context, SemIR::SpliceInst inst)
    -> ConstantEvalResult {
  // The constant value of a SpliceInst is the constant value of the instruction
  // being spliced. Note that `inst.inst_id` is the instruction being spliced,
  // so we need to go through another round of obtaining the constant value in
  // addition to the one performed by the eval infrastructure.
  if (auto inst_value =
          context.insts().TryGetAs<SemIR::InstValue>(inst.inst_id)) {
    return ConstantEvalResult::Existing(
        context.constant_values().Get(inst_value->inst_id));
  }
  // TODO: Consider creating a new `ValueOfInst` instruction analogous to
  // `TypeOfInst` to defer determining the constant value until we know the
  // instruction. Alternatively, produce a symbolic `SpliceInst` constant.
  return ConstantEvalResult::NotConstant;
}

auto EvalConstantInst(Context& context, SemIR::StructAccess inst)
    -> ConstantEvalResult {
  return PerformAggregateAccess(context, inst);
}

auto EvalConstantInst(Context& /*context*/, SemIR::StructInit inst)
    -> ConstantEvalResult {
  return ConstantEvalResult::NewSamePhase(SemIR::StructValue{
      .type_id = inst.type_id, .elements_id = inst.elements_id});
}

auto EvalConstantInst(Context& /*context*/, SemIR::Temporary /*inst*/)
    -> ConstantEvalResult {
  // TODO: Handle this. Can we just return the value of `init_id`?
  return ConstantEvalResult::TODO;
}

auto EvalConstantInst(Context& context, SemIR::TupleAccess inst)
    -> ConstantEvalResult {
  return PerformAggregateAccess(context, inst);
}

auto EvalConstantInst(Context& /*context*/, SemIR::TupleInit inst)
    -> ConstantEvalResult {
  return ConstantEvalResult::NewSamePhase(SemIR::TupleValue{
      .type_id = inst.type_id, .elements_id = inst.elements_id});
}

auto EvalConstantInst(Context& context, SemIR::TypeOfInst inst)
    -> ConstantEvalResult {
  // Grab the type from the instruction produced as our operand.
  if (auto inst_value =
          context.insts().TryGetAs<SemIR::InstValue>(inst.inst_id)) {
    return ConstantEvalResult::Existing(context.types().GetConstantId(
        context.insts().Get(inst_value->inst_id).type_id()));
  }
  return ConstantEvalResult::NewSamePhase(inst);
}

auto EvalConstantInst(Context& context, SemIR::UnaryOperatorNot inst)
    -> ConstantEvalResult {
  // `not true` -> `false`, `not false` -> `true`.
  // All other uses of unary `not` are non-constant.
  auto const_id = context.constant_values().Get(inst.operand_id);
  if (const_id.is_concrete()) {
    auto value = context.insts().GetAs<SemIR::BoolLiteral>(
        context.constant_values().GetInstId(const_id));
    value.value = SemIR::BoolValue::From(!value.value.ToBool());
    return ConstantEvalResult::NewSamePhase(value);
  }
  return ConstantEvalResult::NotConstant;
}

auto EvalConstantInst(Context& context, SemIR::ValueOfInitializer inst)
    -> ConstantEvalResult {
  // Values of value expressions and initializing expressions are represented in
  // the same way during constant evaluation, so just return the value of the
  // operand.
  return ConstantEvalResult::Existing(
      context.constant_values().Get(inst.init_id));
}

auto EvalConstantInst(Context& context, SemIR::InstId inst_id,
                      SemIR::VarStorage inst) -> ConstantEvalResult {
  if (!inst.pattern_id.has_value()) {
    // This variable was not created from a `var` pattern, so isn't a global
    // variable.
    return ConstantEvalResult::NotConstant;
  }

  // A variable is constant if it's global.
  auto entity_name_id = SemIR::GetFirstBindingNameFromPatternId(
      context.sem_ir(), inst.pattern_id);
  if (!entity_name_id.has_value()) {
    // Variable doesn't introduce any bindings, so can only be referenced by its
    // own initializer. We treat such a reference as not being constant.
    return ConstantEvalResult::NotConstant;
  }

  auto scope_id = context.entity_names().Get(entity_name_id).parent_scope_id;
  if (!scope_id.has_value() ||
      !context.insts().Is<SemIR::Namespace>(
          context.name_scopes().Get(scope_id).inst_id())) {
    // Only namespace-scope variables are reference constants.
    return ConstantEvalResult::NotConstant;
  }

  // This is a constant reference expression denoting this global variable.
  return ConstantEvalResult::Existing(
      SemIR::ConstantId::ForConcreteConstant(inst_id));
}

}  // namespace Carbon::Check
