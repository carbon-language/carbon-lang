// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

// Build a FunctionDeclaration describing the signature of a function. This
// handles the common logic shared by function declaration syntax and function
// definition syntax.
static auto BuildFunctionDeclaration(SemanticsContext& context)
    -> std::pair<SemanticsFunctionId, SemanticsNodeId> {
  SemanticsTypeId return_type_id = SemanticsTypeId::Invalid;
  if (context.parse_tree().node_kind(context.node_stack().PeekParseNode()) ==
      ParseNodeKind::ReturnType) {
    return_type_id = context.node_stack().Pop<ParseNodeKind::ReturnType>();
  } else {
    // Canonicalize the empty tuple for the implicit return.
    context.CanonicalizeType(SemanticsNodeId::BuiltinEmptyTupleType);
  }
  SemanticsNodeBlockId param_refs_id =
      context.node_stack().Pop<ParseNodeKind::ParameterList>();
  auto name_context = context.PopDeclarationName();
  auto fn_node = context.node_stack()
                     .PopForSoloParseNode<ParseNodeKind::FunctionIntroducer>();

  // TODO: Support out-of-line definitions, which will have a resolved
  // name_context. Right now, those become errors in AddNameToLookup.

  // Add the callable.
  auto function_id = context.semantics_ir().AddFunction(
      {.name_id = name_context.unresolved_name_id,
       .param_refs_id = param_refs_id,
       .return_type_id = return_type_id,
       .body_block_ids = {}});
  auto decl_id = context.AddNode(
      SemanticsNode::FunctionDeclaration::Make(fn_node, function_id));
  context.AddNameToLookup(name_context, decl_id);
  return {function_id, decl_id};
}

auto SemanticsHandleFunctionDeclaration(SemanticsContext& context,
                                        ParseTree::Node /*parse_node*/)
    -> bool {
  BuildFunctionDeclaration(context);
  return true;
}

auto SemanticsHandleFunctionDefinition(SemanticsContext& context,
                                       ParseTree::Node parse_node) -> bool {
  SemanticsFunctionId function_id =
      context.node_stack().Pop<ParseNodeKind::FunctionDefinitionStart>();

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
      context.AddNode(SemanticsNode::Return::Make(parse_node));
    }
  }

  context.PopScope();
  context.node_block_stack().Pop();
  context.return_scope_stack().pop_back();
  return true;
}

auto SemanticsHandleFunctionDefinitionStart(SemanticsContext& context,
                                            ParseTree::Node parse_node)
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
    auto [name_id, target_id] = ref.GetAsBindName();
    context.AddNameToLookup(ref.parse_node(), name_id, target_id);
  }

  context.node_stack().Push(parse_node, function_id);
  return true;
}

auto SemanticsHandleFunctionIntroducer(SemanticsContext& context,
                                       ParseTree::Node parse_node) -> bool {
  // Push the bracketing node.
  context.node_stack().Push(parse_node);
  // A name should always follow.
  context.PushDeclarationName();
  return true;
}

auto SemanticsHandleReturnType(SemanticsContext& context,
                               ParseTree::Node parse_node) -> bool {
  // Propagate the type expression.
  auto [type_parse_node, type_node_id] =
      context.node_stack().PopExpressionWithParseNode();
  auto cast_node_id = context.ExpressionAsType(type_parse_node, type_node_id);
  context.node_stack().Push(parse_node, cast_node_id);
  return true;
}

}  // namespace Carbon
