// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

// Build a FunctionDeclaration describing the signature of a function. This
// handles the common logic shared by function declaration syntax and function
// definition syntax.
static auto BuildFunctionDeclaration(Context& context)
    -> std::pair<SemIR::FunctionId, SemIR::NodeId> {
  // TODO: This contains the IR block for the parameters and return type. At
  // present, it's just loose, but it's not strictly required for parameter
  // refs; we should either stop constructing it completely or, if it turns out
  // to be needed, store it. Note, the underlying issue is that the LLVM IR has
  // nowhere clear to emit, so changing storage would require addressing that
  // problem. For comparison with function calls, the IR needs to be emitted
  // prior to the call.
  context.node_block_stack().Pop();

  auto return_type_id = SemIR::TypeId::Invalid;
  auto return_slot_id = SemIR::NodeId::Invalid;
  if (context.parse_tree().node_kind(context.node_stack().PeekParseNode()) ==
      Parse::NodeKind::ReturnType) {
    return_slot_id = context.node_stack().Pop<Parse::NodeKind::ReturnType>();
    return_type_id = context.semantics_ir().GetNode(return_slot_id).type_id();

    // The function only has a return slot if it uses in-place initialization.
    if (!SemIR::GetInitializingRepresentation(context.semantics_ir(),
                                              return_type_id)
             .has_return_slot()) {
      return_slot_id = SemIR::NodeId::Invalid;
    }
  }

  SemIR::NodeBlockId param_refs_id =
      context.node_stack().Pop<Parse::NodeKind::ParameterList>();
  auto name_context = context.declaration_name_stack().Pop();
  auto fn_node =
      context.node_stack()
          .PopForSoloParseNode<Parse::NodeKind::FunctionIntroducer>();

  // TODO: Support out-of-line definitions, which will have a resolved
  // name_context. Right now, those become errors in AddNameToLookup.

  // Add the callable.
  auto function_id = context.semantics_ir().AddFunction(
      {.name_id = name_context.state ==
                          DeclarationNameStack::NameContext::State::Unresolved
                      ? name_context.unresolved_name_id
                      : SemIR::StringId(SemIR::StringId::InvalidIndex),
       .param_refs_id = param_refs_id,
       .return_type_id = return_type_id,
       .return_slot_id = return_slot_id,
       .body_block_ids = {}});
  auto decl_id = context.AddNode(
      SemIR::Node::FunctionDeclaration::Make(fn_node, function_id));
  context.declaration_name_stack().AddNameToLookup(name_context, decl_id);
  return {function_id, decl_id};
}

auto HandleFunctionDeclaration(Context& context, Parse::Node /*parse_node*/)
    -> bool {
  BuildFunctionDeclaration(context);
  return true;
}

auto HandleFunctionDefinition(Context& context, Parse::Node parse_node)
    -> bool {
  SemIR::FunctionId function_id =
      context.node_stack().Pop<Parse::NodeKind::FunctionDefinitionStart>();

  // If the `}` of the function is reachable, reject if we need a return value
  // and otherwise add an implicit `return;`.
  if (context.is_current_position_reachable()) {
    if (context.semantics_ir()
            .GetFunction(function_id)
            .return_type_id.is_valid()) {
      CARBON_DIAGNOSTIC(
          MissingReturnStatement, Error,
          "Missing `return` at end of function with declared return type.");
      context.emitter().Emit(parse_node, MissingReturnStatement);
    } else {
      context.AddNode(SemIR::Node::Return::Make(parse_node));
    }
  }

  context.PopScope();
  context.node_block_stack().Pop();
  context.return_scope_stack().pop_back();
  return true;
}

auto HandleFunctionDefinitionStart(Context& context, Parse::Node parse_node)
    -> bool {
  // Process the declaration portion of the function.
  auto [function_id, decl_id] = BuildFunctionDeclaration(context);
  const auto& function = context.semantics_ir().GetFunction(function_id);

  // Create the function scope and the entry block.
  context.return_scope_stack().push_back(decl_id);
  context.node_block_stack().Push();
  context.PushScope();
  context.AddCurrentCodeBlockToFunction();

  // Bring the parameters into scope.
  for (auto ref_id :
       context.semantics_ir().GetNodeBlock(function.param_refs_id)) {
    auto ref = context.semantics_ir().GetNode(ref_id);
    auto name_id = ref.GetAsParameter();
    context.AddNameToLookup(ref.parse_node(), name_id, ref_id);
  }

  context.node_stack().Push(parse_node, function_id);
  return true;
}

auto HandleFunctionIntroducer(Context& context, Parse::Node parse_node)
    -> bool {
  // Create a node block to hold the nodes created as part of the function
  // signature, such as parameter and return types.
  context.node_block_stack().Push();
  // Push the bracketing node.
  context.node_stack().Push(parse_node);
  // A name should always follow.
  context.declaration_name_stack().Push();
  return true;
}

auto HandleReturnType(Context& context, Parse::Node parse_node) -> bool {
  // Propagate the type expression.
  auto [type_parse_node, type_node_id] =
      context.node_stack().PopExpressionWithParseNode();
  auto type_id = context.ExpressionAsType(type_parse_node, type_node_id);
  // TODO: Use a dedicated node rather than VarStorage here.
  context.AddNodeAndPush(
      parse_node,
      SemIR::Node::VarStorage::Make(
          parse_node, type_id, context.semantics_ir().AddString("return")));
  return true;
}

}  // namespace Carbon::Check
