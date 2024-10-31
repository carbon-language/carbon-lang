// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/map.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
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
  auto [name_node, name_id] = context.node_stack().PopNameWithNodeId();

  // Store the name for the type.
  auto value_type_id = context.insts().Get(value_inst_id).type_id();
  context.struct_type_fields_stack().AppendToTop(
      {.name_id = name_id, .type_id = value_type_id});

  // Push the value back on the stack as an argument.
  context.node_stack().Push(node_id, value_inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::StructTypeLiteralFieldId node_id)
    -> bool {
  auto [type_node, type_id] = context.node_stack().PopExprWithNodeId();
  SemIR::TypeId cast_type_id = ExprAsType(context, type_node, type_id).type_id;

  auto [name_node, name_id] = context.node_stack().PopNameWithNodeId();

  context.struct_type_fields_stack().AppendToTop(
      {.name_id = name_id, .type_id = cast_type_id});
  context.node_stack().Push(node_id);
  return true;
}

// Diagnoses and returns true if there's a duplicate name. `get_field_loc`
// returns the location for a field index, for diagnostics.
static auto DiagnoseDuplicateNames(
    Context& context, llvm::ArrayRef<SemIR::StructTypeField> fields,
    bool is_struct_type_literal,
    llvm::function_ref<SemIRLoc(int)> get_field_loc) -> bool {
  Map<SemIR::NameId, int> names;
  for (auto [index, field] : llvm::enumerate(fields)) {
    auto result = names.Insert(field.name_id, index);
    if (!result.is_inserted()) {
      CARBON_DIAGNOSTIC(StructNameDuplicate, Error,
                        "duplicated field name `{1}` in "
                        "{0:struct type literal|struct literal}",
                        BoolAsSelect, SemIR::NameId);
      CARBON_DIAGNOSTIC(StructNamePrevious, Note,
                        "field with the same name here");
      context.emitter()
          .Build(get_field_loc(index), StructNameDuplicate,
                 is_struct_type_literal, field.name_id)
          .Note(get_field_loc(result.value()), StructNamePrevious)
          .Emit();
      return true;
    }
  }
  return false;
}

auto HandleParseNode(Context& context, Parse::StructLiteralId node_id) -> bool {
  auto elements_id = context.param_and_arg_refs_stack().EndAndPop(
      Parse::NodeKind::StructLiteralStart);
  auto fields = context.struct_type_fields_stack().PeekArray();

  context.scope_stack().Pop();
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::StructLiteralStart>();

  if (DiagnoseDuplicateNames(context, fields, /*is_struct_type_literal=*/false,
                             [&](int index) -> SemIRLoc {
                               return context.inst_blocks().Get(
                                   elements_id)[index];
                             })) {
    context.node_stack().Push(node_id, SemIR::InstId::BuiltinError);
  } else {
    auto type_id = context.GetStructType(
        context.struct_type_fields().AddCanonical(fields));

    auto value_id = context.AddInst<SemIR::StructLiteral>(
        node_id, {.type_id = type_id, .elements_id = elements_id});
    context.node_stack().Push(node_id, value_id);
  }

  context.struct_type_fields_stack().PopArray();
  return true;
}

auto HandleParseNode(Context& context, Parse::StructTypeLiteralId node_id)
    -> bool {
  auto fields = context.struct_type_fields_stack().PeekArray();

  context.scope_stack().Pop();
  llvm::SmallVector<Parse::NodeId> nodes;
  while (
      auto node_id =
          context.node_stack()
              .PopForSoloNodeIdIf<Parse::NodeKind::StructTypeLiteralField>()) {
    nodes.push_back(*node_id);
  }
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::StructTypeLiteralStart>();

  if (DiagnoseDuplicateNames(
          context, fields, /*is_struct_type_literal=*/true,
          [&](int index) -> SemIRLoc { return nodes[index]; })) {
    context.node_stack().Push(node_id, SemIR::InstId::BuiltinError);
  } else {
    auto fields_id = context.struct_type_fields().AddCanonical(fields);
    context.AddInstAndPush<SemIR::StructType>(
        node_id, {.type_id = SemIR::TypeId::TypeType, .fields_id = fields_id});
  }

  context.struct_type_fields_stack().PopArray();
  return true;
}

}  // namespace Carbon::Check
