// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/ScopeExit.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleCallExprStart(Context& context, Parse::CallExprStartId parse_node)
    -> bool {
  auto name_id = context.node_stack().PopExpr();
  context.node_stack().Push(parse_node, name_id);
  context.ParamOrArgStart();
  return true;
}

auto HandleCallExprComma(Context& context,
                         Parse::CallExprCommaId /*parse_node*/) -> bool {
  context.ParamOrArgComma();
  return true;
}

auto HandleCallExpr(Context& context, Parse::CallExprId parse_node) -> bool {
  // Process the final explicit call argument now, but leave the arguments
  // block on the stack until the end of this function.
  context.ParamOrArgEndNoPop(Parse::NodeKind::CallExprStart);
  auto discard_args_block = llvm::make_scope_exit(
      [&] { context.params_or_args_stack().PopAndDiscard(); });

  auto [call_expr_parse_node, callee_id] =
      context.node_stack().PopWithParseNode<Parse::NodeKind::CallExprStart>();

  auto diagnose_not_callable = [&, call_expr_parse_node = call_expr_parse_node,
                                callee_id = callee_id] {
    auto callee_type_id = context.insts().Get(callee_id).type_id();
    if (callee_type_id != SemIR::TypeId::Error) {
      CARBON_DIAGNOSTIC(CallToNonCallable, Error,
                        "Value of type `{0}` is not callable.", std::string);
      context.emitter().Emit(call_expr_parse_node, CallToNonCallable,
                             context.sem_ir().StringifyType(callee_type_id));
    }
    context.node_stack().Push(parse_node, SemIR::InstId::BuiltinError);
    return true;
  };

  // For a method call, pick out the `self` value.
  auto function_callee_id = callee_id;
  SemIR::InstId self_id = SemIR::InstId::Invalid;
  if (auto bound_method =
          context.insts().Get(callee_id).TryAs<SemIR::BoundMethod>()) {
    self_id = bound_method->object_id;
    function_callee_id = bound_method->function_id;
  }

  // Identify the function we're calling.
  auto function_decl_id = context.constant_values().Get(function_callee_id);
  if (!function_decl_id.is_constant()) {
    return diagnose_not_callable();
  }
  auto function_decl = context.insts()
                           .Get(function_decl_id.inst_id())
                           .TryAs<SemIR::FunctionDecl>();
  if (!function_decl) {
    return diagnose_not_callable();
  }
  auto function_id = function_decl->function_id;
  const auto& callable = context.functions().Get(function_id);

  // For functions with an implicit return type, the return type is the empty
  // tuple type.
  SemIR::TypeId type_id = callable.return_type_id;
  if (!type_id.is_valid()) {
    type_id = context.GetTupleType({});
  }

  // If there is a return slot, build storage for the result.
  SemIR::InstId return_storage_id = SemIR::InstId::Invalid;
  if (callable.return_slot_id.is_valid()) {
    // Tentatively put storage for a temporary in the function's return slot.
    // This will be replaced if necessary when we perform initialization.
    return_storage_id =
        context.AddInst({call_expr_parse_node,
                         SemIR::TemporaryStorage{callable.return_type_id}});
  }

  // Convert the arguments to match the parameters.
  auto converted_args_id =
      ConvertCallArgs(context, call_expr_parse_node, self_id,
                      context.params_or_args_stack().PeekCurrentBlockContents(),
                      return_storage_id, function_decl_id.inst_id(),
                      callable.implicit_param_refs_id, callable.param_refs_id);
  auto call_inst_id =
      context.AddInst({call_expr_parse_node,
                       SemIR::Call{type_id, callee_id, converted_args_id}});

  context.node_stack().Push(parse_node, call_inst_id);
  return true;
}

}  // namespace Carbon::Check
