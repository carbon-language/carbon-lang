// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/sem_ir/node.h"

namespace Carbon::Check {

auto HandleMemberAccessExpression(Context& context, Parse::Node parse_node)
    -> bool {
  SemIR::StringId name_id = context.node_stack().Pop<Parse::NodeKind::Name>();

  auto base_id = context.node_stack().PopExpression();

  if (auto base_namespace = context.semantics_ir()
                                .GetNode(context.FollowNameReferences(base_id))
                                .TryAs<SemIR::Namespace>()) {
    // For a namespace, just resolve the name.
    auto node_id =
        context.LookupName(parse_node, name_id, base_namespace->name_scope_id,
                           /*print_diagnostics=*/true);
    auto node = context.semantics_ir().GetNode(node_id);
    // TODO: Track that this node was named within `base_id`.
    context.AddNodeAndPush(
        parse_node,
        SemIR::NameReference(parse_node, node.type_id(), name_id, node_id));
    return true;
  }

  // Materialize a temporary for the base expression if necessary.
  base_id = ConvertToValueOrReferenceExpression(context, base_id);
  auto base_type_id = context.semantics_ir().GetNode(base_id).type_id();

  auto base_type = context.semantics_ir().GetNode(
      context.semantics_ir().GetTypeAllowBuiltinTypes(base_type_id));

  switch (base_type.kind()) {
    case SemIR::StructType::Kind: {
      auto refs = context.semantics_ir().GetNodeBlock(
          base_type.As<SemIR::StructType>().fields_id);
      // TODO: Do we need to optimize this with a lookup table for O(1)?
      for (auto [i, ref_id] : llvm::enumerate(refs)) {
        auto field =
            context.semantics_ir().GetNodeAs<SemIR::StructTypeField>(ref_id);
        if (name_id == field.name_id) {
          context.AddNodeAndPush(
              parse_node, SemIR::StructAccess(parse_node, field.type_id,
                                              base_id, SemIR::MemberIndex(i)));
          return true;
        }
      }
      CARBON_DIAGNOSTIC(QualifiedExpressionNameNotFound, Error,
                        "Type `{0}` does not have a member `{1}`.", std::string,
                        llvm::StringRef);
      context.emitter().Emit(parse_node, QualifiedExpressionNameNotFound,
                             context.semantics_ir().StringifyType(base_type_id),
                             context.semantics_ir().GetString(name_id));
      break;
    }
    default: {
      if (base_type_id != SemIR::TypeId::Error) {
        CARBON_DIAGNOSTIC(QualifiedExpressionUnsupported, Error,
                          "Type `{0}` does not support qualified expressions.",
                          std::string);
        context.emitter().Emit(
            parse_node, QualifiedExpressionUnsupported,
            context.semantics_ir().StringifyType(base_type_id));
      }
      break;
    }
  }

  // Should only be reached on error.
  context.node_stack().Push(parse_node, SemIR::NodeId::BuiltinError);
  return true;
}

auto HandlePointerMemberAccessExpression(Context& context,
                                         Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandlePointerMemberAccessExpression");
}

auto HandleName(Context& context, Parse::Node parse_node) -> bool {
  auto name_str = context.parse_tree().GetNodeText(parse_node);
  auto name_id = context.semantics_ir().AddString(name_str);
  // The parent is responsible for binding the name.
  context.node_stack().Push(parse_node, name_id);
  return true;
}

auto HandleNameExpression(Context& context, Parse::Node parse_node) -> bool {
  auto name_str = context.parse_tree().GetNodeText(parse_node);
  auto name_id = context.semantics_ir().AddString(name_str);
  auto value_id =
      context.LookupName(parse_node, name_id, SemIR::NameScopeId::Invalid,
                         /*print_diagnostics=*/true);
  auto value = context.semantics_ir().GetNode(value_id);
  // This is a reference to a name binding that has a value and a type.
  CARBON_CHECK(value.kind().value_kind() == SemIR::NodeValueKind::Typed);
  context.AddNodeAndPush(
      parse_node,
      SemIR::NameReference(parse_node, value.type_id(), name_id, value_id));
  return true;
}

auto HandleQualifiedDeclaration(Context& context, Parse::Node parse_node)
    -> bool {
  auto pop_and_apply_first_child = [&]() {
    if (context.parse_tree().node_kind(context.node_stack().PeekParseNode()) !=
        Parse::NodeKind::QualifiedDeclaration) {
      // First QualifiedDeclaration in a chain.
      auto [parse_node1, node_id1] =
          context.node_stack().PopExpressionWithParseNode();
      context.declaration_name_stack().ApplyExpressionQualifier(
          parse_node1, context.FollowNameReferences(node_id1));
      // Add the QualifiedDeclaration so that it can be used for bracketing.
      context.node_stack().Push(parse_node);
    } else {
      // Nothing to do: the QualifiedDeclaration remains as a bracketing node
      // for later QualifiedDeclarations.
    }
  };

  Parse::Node parse_node2 = context.node_stack().PeekParseNode();
  if (context.parse_tree().node_kind(parse_node2) == Parse::NodeKind::Name) {
    SemIR::StringId name_id2 =
        context.node_stack().Pop<Parse::NodeKind::Name>();
    pop_and_apply_first_child();
    context.declaration_name_stack().ApplyNameQualifier(parse_node2, name_id2);
  } else {
    SemIR::NodeId node_id2 = context.node_stack().PopExpression();
    pop_and_apply_first_child();
    context.declaration_name_stack().ApplyExpressionQualifier(parse_node2,
                                                              node_id2);
  }

  return true;
}

auto HandleSelfTypeNameExpression(Context& context, Parse::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleSelfTypeNameExpression");
}

auto HandleSelfValueName(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleSelfValueName");
}

}  // namespace Carbon::Check
