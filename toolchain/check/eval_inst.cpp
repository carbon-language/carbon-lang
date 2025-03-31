// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/eval_inst.h"

#include <variant>

#include "toolchain/check/action.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/impl_lookup.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// When calling from eval to various Check functions, we need the actual LocId.
// This allows us to unwrap the SemIRLoc to do so.
//
// TODO: Decide whether to refactor calls everywhere to accept `SemIRLoc`, or
// fold `SemIRLoc` into `LocId`. Either way, we would like eval to call other
// code without unwrapping `SemIRLoc`.
class UnwrapSemIRLoc {
 public:
  auto operator()(Context& context, SemIRLoc loc) -> SemIR::LocId {
    if (loc.is_inst_id_) {
      if (loc.inst_id_.has_value()) {
        return context.insts().GetLocId(loc.inst_id_);
      } else {
        return SemIR::LocId::None;
      }
    } else {
      return loc.loc_id_;
    }
  }
};

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

auto EvalConstantInst(Context& /*context*/, SemIRLoc /*loc*/,
                      SemIR::ArrayInit inst) -> ConstantEvalResult {
  // TODO: Add an `ArrayValue` to represent a constant array object
  // representation instead of using a `TupleValue`.
  return ConstantEvalResult::NewSamePhase(
      SemIR::TupleValue{.type_id = inst.type_id, .elements_id = inst.inits_id});
}

auto EvalConstantInst(Context& context, SemIRLoc loc, SemIR::ArrayType inst)
    -> ConstantEvalResult {
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
    context.emitter().Emit(loc, ArrayBoundNegative,
                           {.type = int_bound->type_id, .value = bound_val});
    return ConstantEvalResult::Error;
  }
  if (bound_val.getActiveBits() > 64) {
    CARBON_DIAGNOSTIC(ArrayBoundTooLarge, Error,
                      "array bound of {0} is too large", TypedInt);
    context.emitter().Emit(loc, ArrayBoundTooLarge,
                           {.type = int_bound->type_id, .value = bound_val});
    return ConstantEvalResult::Error;
  }
  return ConstantEvalResult::NewSamePhase(inst);
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/,
                      SemIR::AsCompatible inst) -> ConstantEvalResult {
  // AsCompatible changes the type of the source instruction; its constant
  // value, if there is one, needs to be modified to be of the same type.
  auto value_id = context.constant_values().Get(inst.source_id);
  CARBON_CHECK(value_id.is_constant());

  auto value_inst =
      context.insts().Get(context.constant_values().GetInstId(value_id));
  value_inst.SetType(inst.type_id);
  return ConstantEvalResult::NewAnyPhase(value_inst);
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/, SemIR::BindAlias inst)
    -> ConstantEvalResult {
  // An alias evaluates to the value it's bound to.
  return ConstantEvalResult::Existing(
      context.constant_values().Get(inst.value_id));
}

auto EvalConstantInst(Context& /*context*/, SemIRLoc /*loc*/,
                      SemIR::BindValue /*inst*/) -> ConstantEvalResult {
  // TODO: Handle this once we've decided how to represent constant values of
  // reference expressions.
  return ConstantEvalResult::TODO;
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/,
                      SemIR::ClassElementAccess inst) -> ConstantEvalResult {
  return PerformAggregateAccess(context, inst);
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/, SemIR::ClassDecl inst)
    -> ConstantEvalResult {
  // If the class has generic parameters, we don't produce a class type, but a
  // callable whose return value is a class type.
  if (context.classes().Get(inst.class_id).has_parameters()) {
    return ConstantEvalResult::NewSamePhase(SemIR::StructValue{
        .type_id = inst.type_id, .elements_id = SemIR::InstBlockId::Empty});
  }

  // A non-generic class declaration evaluates to the class type.
  return ConstantEvalResult::NewSamePhase(
      SemIR::ClassType{.type_id = SemIR::TypeType::SingletonTypeId,
                       .class_id = inst.class_id,
                       .specific_id = SemIR::SpecificId::None});
}

auto EvalConstantInst(Context& /*context*/, SemIRLoc /*loc*/,
                      SemIR::ClassInit inst) -> ConstantEvalResult {
  // TODO: Add a `ClassValue` to represent a constant class object
  // representation instead of using a `StructValue`.
  return ConstantEvalResult::NewSamePhase(SemIR::StructValue{
      .type_id = inst.type_id, .elements_id = inst.elements_id});
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/, SemIR::ConstType inst)
    -> ConstantEvalResult {
  // `const (const T)` evaluates to `const T`.
  if (context.types().Is<SemIR::ConstType>(inst.inner_id)) {
    return ConstantEvalResult::Existing(
        context.types().GetConstantId(inst.inner_id));
  }
  // Otherwise, `const T` evaluates to itself.
  return ConstantEvalResult::NewSamePhase(inst);
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/, SemIR::Converted inst)
    -> ConstantEvalResult {
  // A conversion evaluates to the result of the conversion.
  return ConstantEvalResult::Existing(
      context.constant_values().Get(inst.result_id));
}

auto EvalConstantInst(Context& /*context*/, SemIRLoc /*loc*/,
                      SemIR::Deref /*inst*/) -> ConstantEvalResult {
  // TODO: Handle this.
  return ConstantEvalResult::TODO;
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/,
                      SemIR::ExportDecl inst) -> ConstantEvalResult {
  // An export instruction evaluates to the exported declaration.
  return ConstantEvalResult::Existing(
      context.constant_values().Get(inst.value_id));
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/,
                      SemIR::FacetAccessType inst) -> ConstantEvalResult {
  if (auto facet_value = context.insts().TryGetAs<SemIR::FacetValue>(
          inst.facet_value_inst_id)) {
    return ConstantEvalResult::Existing(
        context.constant_values().Get(facet_value->type_inst_id));
  }
  return ConstantEvalResult::NewSamePhase(inst);
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/,
                      SemIR::FacetAccessWitness inst) -> ConstantEvalResult {
  if (auto facet_value = context.insts().TryGetAs<SemIR::FacetValue>(
          inst.facet_value_inst_id)) {
    auto impl_witness_inst_id = context.inst_blocks().Get(
        facet_value->witnesses_block_id)[inst.index.index];
    return ConstantEvalResult::Existing(
        context.constant_values().Get(impl_witness_inst_id));
  }
  return ConstantEvalResult::NewSamePhase(inst);
}

auto EvalConstantInst(Context& context, SemIRLoc loc, SemIR::FloatType inst)
    -> ConstantEvalResult {
  return ValidateFloatType(context, loc, inst)
             ? ConstantEvalResult::NewSamePhase(inst)
             : ConstantEvalResult::Error;
}

auto EvalConstantInst(Context& /*context*/, SemIRLoc /*loc*/,
                      SemIR::FunctionDecl inst) -> ConstantEvalResult {
  // A function declaration evaluates to a function object, which is an empty
  // object of function type.
  // TODO: Eventually we may need to handle captures here.
  return ConstantEvalResult::NewSamePhase(SemIR::StructValue{
      .type_id = inst.type_id, .elements_id = SemIR::InstBlockId::Empty});
}

auto EvalConstantInst(Context& context, SemIRLoc loc,
                      SemIR::LookupImplWitness inst) -> ConstantEvalResult {
  auto result = EvalLookupSingleImplWitness(
      context, UnwrapSemIRLoc()(context, loc), inst);
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

auto EvalConstantInst(Context& context, SemIRLoc loc,
                      SemIR::ImplWitnessAccess inst) -> ConstantEvalResult {
  // This is PerformAggregateAccess followed by GetConstantInSpecific.
  if (auto witness =
          context.insts().TryGetAs<SemIR::ImplWitness>(inst.witness_id)) {
    auto elements = context.inst_blocks().Get(witness->elements_id);
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
    context.emitter().Emit(loc, ImplAccessMemberBeforeSet);
    return ConstantEvalResult::Error;
  }

  return ConstantEvalResult::NewSamePhase(inst);
}

auto EvalConstantInst(Context& /*context*/, SemIRLoc /*loc*/,
                      SemIR::ImportRefUnloaded inst) -> ConstantEvalResult {
  CARBON_FATAL("ImportRefUnloaded should be loaded before TryEvalInst: {0}",
               inst);
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/,
                      SemIR::InitializeFrom inst) -> ConstantEvalResult {
  // Initialization is not performed in-place during constant evaluation, so
  // just return the value of the initializer.
  return ConstantEvalResult::Existing(
      context.constant_values().Get(inst.src_id));
}

auto EvalConstantInst(Context& context, SemIRLoc loc, SemIR::IntType inst)
    -> ConstantEvalResult {
  return ValidateIntType(context, loc, inst)
             ? ConstantEvalResult::NewSamePhase(inst)
             : ConstantEvalResult::Error;
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/,
                      SemIR::InterfaceDecl inst) -> ConstantEvalResult {
  // If the interface has generic parameters, we don't produce an interface
  // type, but a callable whose return value is an interface type.
  if (context.interfaces().Get(inst.interface_id).has_parameters()) {
    return ConstantEvalResult::NewSamePhase(SemIR::StructValue{
        .type_id = inst.type_id, .elements_id = SemIR::InstBlockId::Empty});
  }

  // A non-generic interface declaration evaluates to a facet type.
  return ConstantEvalResult::NewSamePhase(FacetTypeFromInterface(
      context, inst.interface_id, SemIR::SpecificId::None));
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/, SemIR::NameRef inst)
    -> ConstantEvalResult {
  // A name reference evaluates to the value the name resolves to.
  return ConstantEvalResult::Existing(
      context.constant_values().Get(inst.value_id));
}

auto EvalConstantInst(Context& context, SemIRLoc loc,
                      SemIR::RequireCompleteType inst) -> ConstantEvalResult {
  auto witness_type_id =
      GetSingletonType(context, SemIR::WitnessType::SingletonInstId);

  // If the type is a concrete constant, require it to be complete now.
  auto complete_type_id = inst.complete_type_id;
  if (context.types().GetConstantId(complete_type_id).is_concrete()) {
    if (!TryToCompleteType(context, complete_type_id, loc, [&] {
          // TODO: It'd be nice to report the original type prior to
          // evaluation here.
          CARBON_DIAGNOSTIC(IncompleteTypeInMonomorphization, Error,
                            "type {0} is incomplete", SemIR::TypeId);
          return context.emitter().Build(loc, IncompleteTypeInMonomorphization,
                                         complete_type_id);
        })) {
      return ConstantEvalResult::Error;
    }
    return ConstantEvalResult::NewSamePhase(SemIR::CompleteTypeWitness{
        .type_id = witness_type_id,
        .object_repr_id = context.types().GetObjectRepr(complete_type_id)});
  }

  // If it's not a concrete constant, require it to be complete once it
  // becomes one.
  return ConstantEvalResult::NewSamePhase(inst);
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/,
                      SemIR::SpecificConstant inst) -> ConstantEvalResult {
  // Pull the constant value out of the specific.
  return ConstantEvalResult::Existing(SemIR::GetConstantValueInSpecific(
      context.sem_ir(), inst.specific_id, inst.inst_id));
}

auto EvalConstantInst(Context& context, SemIRLoc loc,
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
  auto specific_id = MakeSpecific(context, loc, generic_id, args);

  // TODO: Add the new `SpecificFunction` to definitions_required.
  return ConstantEvalResult::NewSamePhase(
      SemIR::SpecificFunction{.type_id = inst.type_id,
                              .callee_id = inst.callee_id,
                              .specific_id = specific_id});
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/,
                      SemIR::SpliceBlock inst) -> ConstantEvalResult {
  // SpliceBlock evaluates to the result value that is (typically) within the
  // block. This can be constant even if the block contains other non-constant
  // instructions.
  return ConstantEvalResult::Existing(
      context.constant_values().Get(inst.result_id));
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/,
                      SemIR::SpliceInst inst) -> ConstantEvalResult {
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

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/,
                      SemIR::StructAccess inst) -> ConstantEvalResult {
  return PerformAggregateAccess(context, inst);
}

auto EvalConstantInst(Context& /*context*/, SemIRLoc /*loc*/,
                      SemIR::StructInit inst) -> ConstantEvalResult {
  return ConstantEvalResult::NewSamePhase(SemIR::StructValue{
      .type_id = inst.type_id, .elements_id = inst.elements_id});
}

auto EvalConstantInst(Context& /*context*/, SemIRLoc /*loc*/,
                      SemIR::Temporary /*inst*/) -> ConstantEvalResult {
  // TODO: Handle this. Can we just return the value of `init_id`?
  return ConstantEvalResult::TODO;
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/,
                      SemIR::TupleAccess inst) -> ConstantEvalResult {
  return PerformAggregateAccess(context, inst);
}

auto EvalConstantInst(Context& /*context*/, SemIRLoc /*loc*/,
                      SemIR::TupleInit inst) -> ConstantEvalResult {
  return ConstantEvalResult::NewSamePhase(SemIR::TupleValue{
      .type_id = inst.type_id, .elements_id = inst.elements_id});
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/,
                      SemIR::TypeOfInst inst) -> ConstantEvalResult {
  // Grab the type from the instruction produced as our operand.
  if (auto inst_value =
          context.insts().TryGetAs<SemIR::InstValue>(inst.inst_id)) {
    return ConstantEvalResult::Existing(context.types().GetConstantId(
        context.insts().Get(inst_value->inst_id).type_id()));
  }
  return ConstantEvalResult::NewSamePhase(inst);
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/,
                      SemIR::UnaryOperatorNot inst) -> ConstantEvalResult {
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

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/,
                      SemIR::ValueOfInitializer inst) -> ConstantEvalResult {
  // Values of value expressions and initializing expressions are represented in
  // the same way during constant evaluation, so just return the value of the
  // operand.
  return ConstantEvalResult::Existing(
      context.constant_values().Get(inst.init_id));
}

auto EvalConstantInst(Context& context, SemIRLoc /*loc*/,
                      SemIR::ValueParamPattern inst) -> ConstantEvalResult {
  // TODO: Treat this as a non-expression (here and in GetExprCategory)
  // once generic deduction doesn't need patterns to have constant values.
  return ConstantEvalResult::Existing(
      context.constant_values().Get(inst.subpattern_id));
}

auto EvalConstantInst(Context& /*context*/, SemIRLoc /*loc*/,
                      SemIR::VtablePtr /*inst*/) -> ConstantEvalResult {
  // TODO: Handle this.
  return ConstantEvalResult::TODO;
}

}  // namespace Carbon::Check
