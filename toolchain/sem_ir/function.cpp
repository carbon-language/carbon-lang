// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/function.h"

#include <optional>
#include <variant>

#include "toolchain/base/canonical_value_store_impl.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/value_store_impl.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

namespace {
struct UnwrappedCallee {
  const File* ir;
  InstId callee_id;
  SpecificId caller_specific_id;
};
}  // namespace

// Returns the callee operand with any splices unwrapped, along with the IR
// containing it and the specific in which it should be evaluated.
static auto UnwrapCalleeSplices(const File& callee_ir, InstId callee_id,
                                const File& caller_specific_ir,
                                SpecificId caller_specific_id)
    -> UnwrappedCallee {
  const File* ir = &callee_ir;
  while (true) {
    auto inst = ir->insts().Get(callee_id);
    if (auto splice_inst = inst.TryAs<SpliceInst>()) {
      auto [inst_ir, inst_value_id] = GetConstantValueInSpecific(
          caller_specific_ir, caller_specific_id, *ir, splice_inst->inst_id);
      if (!inst_value_id.is_concrete()) {
        break;
      }
      ir = inst_ir;
      callee_id = ir->constant_values()
                      .GetInstAs<SemIR::InstValue>(inst_value_id)
                      .inst_id;
      // The spliced instruction has already been refined into the specific.
      caller_specific_id = SpecificId::None;
    } else if (auto splice_block = inst.TryAs<SpliceBlock>()) {
      callee_id = splice_block->result_id;
    } else {
      break;
    }
  }
  return {.ir = ir,
          .callee_id = callee_id,
          .caller_specific_id = caller_specific_id};
}

auto TryGetCalleeAsBoundMethod(const File& sem_ir, InstId callee_id,
                               SpecificId caller_specific_id)
    -> std::optional<BoundMethod> {
  // Step through splices before checking for a bound method. Unfortunately we
  // can't just look at the constant value of callee_id here, because bound
  // methods are non-constant if their `self` is, so we do a mini-eval here.
  auto unwrapped =
      UnwrapCalleeSplices(sem_ir, callee_id, sem_ir, caller_specific_id);
  return unwrapped.ir->insts().TryGetAs<BoundMethod>(unwrapped.callee_id);
}

auto GetCallee(const File& callee_ir, InstId callee_id,
               const File& caller_specific_ir, SpecificId caller_specific_id)
    -> std::pair<const File*, Callee> {
  CalleeFunction fn = {.function_id = FunctionId::None,
                       .enclosing_specific_id = SpecificId::None,
                       .resolved_specific_id = SpecificId::None,
                       .self_type_id = InstId::None,
                       .self_id = InstId::None};
  auto unwrapped = UnwrapCalleeSplices(callee_ir, callee_id, caller_specific_ir,
                                       caller_specific_id);
  if (auto bound_method =
          unwrapped.ir->insts().TryGetAs<BoundMethod>(unwrapped.callee_id)) {
    fn.self_id = bound_method->object_id;
    unwrapped.callee_id = bound_method->function_decl_id;
  }

  if (unwrapped.caller_specific_id.has_value()) {
    auto [const_ir, const_id] = GetConstantValueInSpecific(
        caller_specific_ir, unwrapped.caller_specific_id, *unwrapped.ir,
        unwrapped.callee_id);
    unwrapped.ir = const_ir;
    unwrapped.callee_id =
        unwrapped.ir->constant_values().GetInstIdIfValid(const_id);
    CARBON_CHECK(unwrapped.callee_id.has_value(),
                 "Invalid callee id in a specific context");
  }

  const File* ir = unwrapped.ir;
  if (auto fn_ptr_type = ir->types().TryGetAs<CppFunctionPointerType>(
          ir->insts().Get(unwrapped.callee_id).type_id())) {
    return {ir, CalleeCppFunctionPointer{.function_type_id =
                                             fn_ptr_type->clang_type_id}};
  }

  auto val_id = ir->constant_values().GetConstantInstId(unwrapped.callee_id);
  if (!val_id.has_value()) {
    return {ir, CalleeNonFunction()};
  }

  if (auto specific_function = ir->insts().TryGetAs<SpecificFunction>(val_id)) {
    fn.resolved_specific_id = specific_function->specific_id;
    val_id =
        ir->constant_values().GetConstantInstId(specific_function->callee_id);
  } else if (auto specific_impl_function =
                 ir->insts().TryGetAs<SpecificImplFunction>(val_id)) {
    fn.resolved_specific_id = specific_impl_function->specific_id;
    val_id = ir->constant_values().GetConstantInstId(
        specific_impl_function->callee_id);
  }
  if (!val_id.has_value()) {
    return {ir, CalleeNonFunction()};
  }

  // Identify the function we're calling by its type.
  auto fn_type_inst_id =
      ir->types().GetTypeInstId(ir->insts().Get(val_id).type_id());

  if (auto cpp_overload_set_type =
          ir->insts().TryGetAs<CppOverloadSetType>(fn_type_inst_id)) {
    CARBON_CHECK(!fn.resolved_specific_id.has_value(),
                 "Only `SpecificFunction` will be resolved, not C++ overloads");
    return {ir, CalleeCppOverloadSet{.cpp_overload_set_id =
                                         cpp_overload_set_type->overload_set_id,
                                     .self_id = fn.self_id}};
  }

  if (auto impl_fn_type =
          ir->insts().TryGetAs<FunctionTypeWithSelfType>(fn_type_inst_id)) {
    // Combine the associated function's `Self` with the interface function
    // data.
    fn.self_type_id = impl_fn_type->self_id;
    fn_type_inst_id = impl_fn_type->interface_function_type_id;
  }

  auto fn_type_val_id =
      ir->constant_values().GetConstantInstId(fn_type_inst_id);
  if (fn_type_val_id == ErrorInst::InstId) {
    return {ir, CalleeError()};
  }
  auto fn_type = ir->insts().TryGetAs<FunctionType>(fn_type_val_id);
  if (!fn_type) {
    return {ir, CalleeNonFunction()};
  }

  fn.function_id = fn_type->function_id;
  fn.enclosing_specific_id = fn_type->specific_id;
  return {ir, fn};
}

auto GetCallee(const File& sem_ir, InstId callee_id,
               SpecificId caller_specific_id) -> Callee {
  auto [callee_ir, callee] =
      GetCallee(sem_ir, callee_id, sem_ir, caller_specific_id);
  CARBON_CHECK(callee_ir == &sem_ir);
  return callee;
}

auto GetCalleeAsFunction(const File& callee_ir, InstId callee_id,
                         const File& caller_specific_ir,
                         SpecificId caller_specific_id)
    -> std::pair<const File*, CalleeFunction> {
  auto [result_ir, callee] =
      GetCallee(callee_ir, callee_id, caller_specific_ir, caller_specific_id);
  CARBON_CHECK(std::holds_alternative<CalleeFunction>(callee));
  return {result_ir, std::get<CalleeFunction>(callee)};
}

auto GetCalleeAsFunction(const File& sem_ir, InstId callee_id,
                         SpecificId caller_specific_id) -> CalleeFunction {
  auto [callee_ir, callee] =
      GetCalleeAsFunction(sem_ir, callee_id, sem_ir, caller_specific_id);
  CARBON_CHECK(callee_ir == &sem_ir);
  return callee;
}

auto DecomposeVirtualFunction(const File& sem_ir, InstId fn_decl_id,
                              SpecificId base_class_specific_id)
    -> DecomposedVirtualFunction {
  // Remap the base's vtable entry to the appropriate constant usable in
  // the context of the derived class (for the specific for the base
  // class, for instance).
  auto fn_decl_const_id =
      GetConstantValueInSpecific(sem_ir, base_class_specific_id, fn_decl_id);
  fn_decl_id = sem_ir.constant_values().GetInstId(fn_decl_const_id);
  auto specific_id = SpecificId::None;
  auto callee_id = fn_decl_id;
  if (auto specific_function =
          sem_ir.insts().TryGetAs<SpecificFunction>(fn_decl_id)) {
    specific_id = specific_function->specific_id;
    callee_id = specific_function->callee_id;
  }

  // Identify the function we're calling by its type.
  auto fn_type_inst =
      sem_ir.types().GetAsInst(sem_ir.insts().Get(callee_id).type_id());

  return {.fn_decl_id = fn_decl_id,
          .fn_decl_const_id = fn_decl_const_id,
          .function_id = fn_type_inst.As<FunctionType>().function_id,
          .specific_id = specific_id};
}

auto Function::GetBuiltinFunctionKind(const File& file) const
    -> BuiltinFunctionKind {
  switch (special_function_kind) {
    case SpecialFunctionKind::Builtin:
      return non_generated_builtin_function_kind();
    case SpecialFunctionKind::Generated: {
      auto generated_id = GeneratedFunctionId(special_function_kind_data.index);
      return file.generated_functions().Get(generated_id).builtin_function_kind;
    }
    default:
      return BuiltinFunctionKind::None;
  }
}

auto Function::GetDeclaredReturnType(const File& file,
                                     SpecificId specific_id) const -> TypeId {
  CARBON_CHECK(!has_pending_deduced_return_type(),
               "Querying return type of function before it is deduced");
  if (!return_type_inst_id.has_value()) {
    return TypeId::None;
  }
  return file.types().GetTypeIdForTypeConstantId(
      GetConstantValueInSpecific(file, specific_id, return_type_inst_id));
}

auto Function::GetDeclaredReturnForm(const File& file,
                                     SpecificId specific_id) const -> InstId {
  CARBON_CHECK(!has_pending_deduced_return_type(),
               "Querying return form of function before it is deduced");
  if (!return_form_inst_id.has_value()) {
    // Treat as equivalent to `-> ()`.
    return InstId::None;
  }
  auto return_form_constant_id =
      GetConstantValueInSpecific(file, specific_id, return_form_inst_id);
  return file.constant_values().GetInstId(return_form_constant_id);
}

}  // namespace Carbon::SemIR

namespace Carbon {
template class ValueStore<SemIR::FunctionId, SemIR::Function,
                          Tag<SemIR::CheckIRId>>;
template class CanonicalValueStore<
    SemIR::GeneratedFunctionId, SemIR::GeneratedFunction::CanonicalKey,
    Tag<SemIR::CheckIRId>, SemIR::GeneratedFunction>;
}  // namespace Carbon
