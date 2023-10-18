// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleClassIntroducer(Context& context, Parse::Node parse_node) -> bool {
  // Create a node block to hold the nodes created as part of the class
  // signature, such as generic parameters.
  context.node_block_stack().Push();
  // Push the bracketing node.
  context.node_stack().Push(parse_node);
  // A name should always follow.
  context.declaration_name_stack().Push();
  return true;
}

static auto BuildClassDeclaration(Context& context) -> void {
  auto name_context = context.declaration_name_stack().Pop();

  auto class_keyword =
      context.node_stack()
          .PopForSoloParseNode<Parse::NodeKind::ClassIntroducer>();

  // TODO: Track this somewhere.
  context.node_block_stack().Pop();

  auto class_id = context.semantics_ir().AddClass(
      {.name_id = name_context.state ==
                          DeclarationNameStack::NameContext::State::Unresolved
                      ? name_context.unresolved_name_id
                      : SemIR::StringId(SemIR::StringId::InvalidIndex)});
  auto class_decl_id = context.AddNode(SemIR::ClassDeclaration{
      class_keyword, SemIR::TypeId::TypeType, class_id});
  context.declaration_name_stack().AddNameToLookup(name_context, class_decl_id);
}

auto HandleClassDeclaration(Context& context, Parse::Node /*parse_node*/)
    -> bool {
  BuildClassDeclaration(context);
  return true;
}

auto HandleClassDefinitionStart(Context& context, Parse::Node parse_node)
    -> bool {
  BuildClassDeclaration(context);
  // TODO: Introduce `Self`.
  return context.TODO(parse_node, "HandleClassDefinitionStart");
}

auto HandleClassDefinition(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleClassDefinition");
}

}  // namespace Carbon::Check
