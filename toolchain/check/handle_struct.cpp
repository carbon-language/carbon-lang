// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/map.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/type.h"
#include "toolchain/diagnostics/format_providers.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::StructLiteralStartId node_id)
    -> bool {
  context.scope_stack().Push();
  context.node_stack().Push(node_id);
  context.struct_type_fields_stack().PushArray();
  context.param_and_arg_refs_stack().Push();
  return true;
}

auto HandleParseNode(Context& context, Parse::StructTypeLiteralStartId node_id)
    -> bool {
  context.scope_stack().Push();
  context.node_stack().Push(node_id);
  context.struct_type_fields_stack().PushArray();
  return true;
}

auto HandleParseNode(Context& context,
                     Parse::StructFieldDesignatorId /*node_id*/) -> bool {
  // This leaves the designated name on top because the `.` isn't interesting.
  CARBON_CHECK(context.node_stack().PeekIs<SemIR::NameId>());
  return true;
}

auto HandleParseNode(Context& context, Parse::StructLiteralCommaId /*node_id*/)
    -> bool {
  context.param_and_arg_refs_stack().ApplyComma();
  return true;
}

auto HandleParseNode(Context& /*context*/,
                     Parse::StructTypeLiteralCommaId /*node_id*/) -> bool {
  return true;
}

auto HandleParseNode(Context& context, Parse::StructLiteralFieldId node_id)
    -> bool {
  auto value_inst_id = context.node_stack().PopExpr();
  // Get the name while leaving it on the stack.
  auto name_id = context.node_stack().Peek<Parse::NodeCategory::MemberName>();

  // Store the name for the type.
  auto value_type_id = context.insts().Get(value_inst_id).type_id();
  context.struct_type_fields_stack().AppendToTop(
      {.name_id = name_id, .type_id = value_type_id});

  // Push the value back on the stack as an argument.
  context.node_stack().Push(node_id, value_inst_id);
  return true;
}

auto HandleParseNode(Context& context,
                     Parse::StructTypeLiteralFieldId /*node_id*/) -> bool {
  auto [type_node, type_id] = context.node_stack().PopExprWithNodeId();
  SemIR::TypeId cast_type_id = ExprAsType(context, type_node, type_id).type_id;
  // Get the name while leaving it on the stack.
  auto name_id = context.node_stack().Peek<Parse::NodeCategory::MemberName>();

  context.struct_type_fields_stack().AppendToTop(
      {.name_id = name_id, .type_id = cast_type_id});
  return true;
}

// Diagnoses and returns true if there's a duplicate name.
static auto DiagnoseDuplicateNames(
    Context& context, llvm::ArrayRef<Parse::NodeId> field_name_nodes,
    llvm::ArrayRef<SemIR::StructTypeField> fields, bool is_struct_type_literal)
    -> bool {
  Map<SemIR::NameId, Parse::NodeId> names;
  for (auto [field_name_node, field] : llvm::zip(field_name_nodes, fields)) {
    auto result = names.Insert(field.name_id, field_name_node);
    if (!result.is_inserted()) {
      CARBON_DIAGNOSTIC(StructNameDuplicate, Error,
                        "duplicated field name `{1}` in "
                        "{0:struct type literal|struct literal}",
                        Diagnostics::BoolAsSelect, SemIR::NameId);
      CARBON_DIAGNOSTIC(StructNamePrevious, Note,
                        "field with the same name here");
      context.emitter()
          .Build(result.value(), StructNameDuplicate, is_struct_type_literal,
                 field.name_id)
          .Note(field_name_node, StructNamePrevious)
          .Emit();
      return true;
    }
  }
  return false;
}

// Pops the names of each field from the stack. These will have been left while
// handling struct fields.
static auto PopFieldNameNodes(Context& context, size_t field_count)
    -> llvm::SmallVector<Parse::NodeId> {
  llvm::SmallVector<Parse::NodeId> nodes;
  nodes.reserve(field_count);
  while (true) {
    auto [name_node, _] =
        context.node_stack().PopWithNodeIdIf<Parse::NodeCategory::MemberName>();
    if (name_node.has_value()) {
      nodes.push_back(name_node);
    } else {
      break;
    }
  }
  CARBON_CHECK(nodes.size() == field_count, "Found {0} names, expected {1}",
               nodes.size(), field_count);
  return nodes;
}

auto HandleParseNode(Context& context, Parse::StructLiteralId node_id) -> bool {
  if (!context.node_stack().PeekIs(Parse::NodeCategory::MemberName)) {
    // Remove the last parameter from the node stack before collecting names.
    context.param_and_arg_refs_stack().EndNoPop(
        Parse::NodeKind::StructLiteralStart);
  }

  auto fields = context.struct_type_fields_stack().PeekArray();
  llvm::SmallVector<Parse::NodeId> field_name_nodes =
      PopFieldNameNodes(context, fields.size());

  auto elements_id = context.param_and_arg_refs_stack().EndAndPop(
      Parse::NodeKind::StructLiteralStart);

  context.scope_stack().Pop();
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::StructLiteralStart>();

  if (DiagnoseDuplicateNames(context, field_name_nodes, fields,
                             /*is_struct_type_literal=*/false)) {
    context.node_stack().Push(node_id, SemIR::ErrorInst::SingletonInstId);
  } else {
    auto type_id = GetStructType(
        context, context.struct_type_fields().AddCanonical(fields));

    auto value_id = AddInst<SemIR::StructLiteral>(
        context, node_id, {.type_id = type_id, .elements_id = elements_id});
    context.node_stack().Push(node_id, value_id);
  }

  context.struct_type_fields_stack().PopArray();
  return true;
}

auto HandleParseNode(Context& context, Parse::StructTypeLiteralId node_id)
    -> bool {
  auto fields = context.struct_type_fields_stack().PeekArray();
  llvm::SmallVector<Parse::NodeId> field_name_nodes =
      PopFieldNameNodes(context, fields.size());

  context.scope_stack().Pop();
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::StructTypeLiteralStart>();

  if (DiagnoseDuplicateNames(context, field_name_nodes, fields,
                             /*is_struct_type_literal=*/true)) {
    context.node_stack().Push(node_id, SemIR::ErrorInst::SingletonInstId);
  } else {
    auto fields_id = context.struct_type_fields().AddCanonical(fields);
    AddInstAndPush<SemIR::StructType>(
        context, node_id,
        {.type_id = SemIR::TypeType::SingletonTypeId, .fields_id = fields_id});
  }

  context.struct_type_fields_stack().PopArray();
  return true;
}

}  // namespace Carbon::Check
