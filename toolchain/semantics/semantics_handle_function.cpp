// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandleFunctionDeclaration(SemanticsContext& context,
                                        ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleFunctionDeclaration");
}

auto SemanticsHandleFunctionDefinition(SemanticsContext& context,
                                       ParseTree::Node parse_node) -> bool {
  // Merges code block children up under the FunctionDefinitionStart.
  while (context.parse_tree().node_kind(context.node_stack().PeekParseNode()) !=
         ParseNodeKind::FunctionDefinitionStart) {
    context.node_stack().PopAndIgnore();
  }
  auto decl_id =
      context.node_stack().PopForNodeId(ParseNodeKind::FunctionDefinitionStart);

  context.return_scope_stack().pop_back();
  context.PopScope();
  auto block_id = context.node_block_stack().Pop();
  context.AddNode(
      SemanticsNode::FunctionDefinition::Make(parse_node, decl_id, block_id));
  context.node_stack().Push(parse_node);

  return true;
}

auto SemanticsHandleFunctionDefinitionStart(SemanticsContext& context,
                                            ParseTree::Node parse_node)
    -> bool {
  SemanticsTypeId return_type_id = SemanticsTypeId::Invalid;
  if (context.parse_tree().node_kind(context.node_stack().PeekParseNode()) ==
      ParseNodeKind::ReturnType) {
    return_type_id =
        context.node_stack().PopForTypeId(ParseNodeKind::ReturnType);
  } else {
    // Canonicalize the empty tuple for the implicit return.
    context.CanonicalizeType(SemanticsNodeId::BuiltinEmptyTupleType);
  }
  auto param_refs_id =
      context.node_stack().PopForNodeBlockId(ParseNodeKind::ParameterList);
  auto name_node =
      context.node_stack().PopForSoloParseNode(ParseNodeKind::DeclaredName);
  auto fn_node = context.node_stack().PopForSoloParseNode(
      ParseNodeKind::FunctionIntroducer);

  auto name_str = context.parse_tree().GetNodeText(name_node);
  auto name_id = context.semantics().AddString(name_str);

  auto callable_id = context.semantics().AddCallable(
      {.param_refs_id = param_refs_id, .return_type_id = return_type_id});
  auto decl_id = context.AddNode(
      SemanticsNode::FunctionDeclaration::Make(fn_node, name_id, callable_id));
  context.AddNameToLookup(name_node, name_id, decl_id);

  context.node_block_stack().Push();
  context.PushScope();
  for (auto ref_id : context.semantics().GetNodeBlock(param_refs_id)) {
    auto ref = context.semantics().GetNode(ref_id);
    auto [name_id, target_id] = ref.GetAsBindName();
    context.AddNameToLookupIgnoreConflicts(name_id, target_id);
  }
  context.return_scope_stack().push_back(decl_id);
  context.node_stack().Push(parse_node, decl_id);

  return true;
}

auto SemanticsHandleFunctionIntroducer(SemanticsContext& context,
                                       ParseTree::Node parse_node) -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(parse_node);
  return true;
}

}  // namespace Carbon
