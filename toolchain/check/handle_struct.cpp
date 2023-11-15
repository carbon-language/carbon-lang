// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"

namespace Carbon::Check {

auto HandleStructComma(Context& context, Parse::Node /*parse_node*/) -> bool {
  context.ParamOrArgComma();
  return true;
}

auto HandleStructFieldDesignator(Context& context, Parse::Node /*parse_node*/)
    -> bool {
  // This leaves the designated name on top because the `.` isn't interesting.
  CARBON_CHECK(
      context.parse_tree().node_kind(context.node_stack().PeekParseNode()) ==
      Parse::NodeKind::Name);
  return true;
}

auto HandleStructFieldType(Context& context, Parse::Node parse_node) -> bool {
  auto [type_node, type_id] = context.node_stack().PopExprWithParseNode();
  SemIR::TypeId cast_type_id = ExprAsType(context, type_node, type_id);

  auto [name_node, name_id] =
      context.node_stack().PopWithParseNode<Parse::NodeKind::Name>();

  context.AddInstAndPush(
      parse_node, SemIR::StructTypeField{name_node, name_id, cast_type_id});
  return true;
}

auto HandleStructFieldUnknown(Context& context, Parse::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleStructFieldUnknown");
}

auto HandleStructFieldValue(Context& context, Parse::Node parse_node) -> bool {
  auto [value_parse_node, value_inst_id] =
      context.node_stack().PopExprWithParseNode();
  SemIR::NameId name_id = context.node_stack().Pop<Parse::NodeKind::Name>();

  // Store the name for the type.
  context.args_type_info_stack().AddInst(SemIR::StructTypeField{
      parse_node, name_id, context.insts().Get(value_inst_id).type_id()});

  // Push the value back on the stack as an argument.
  context.node_stack().Push(parse_node, value_inst_id);
  return true;
}

// Goes from the StructFieldValue node pointing to the `=` in a struct literal
// like `{.a = 1}`, to the name node `a` in its children.
static auto StructFieldValueToName(Context& context,
                                   Parse::Node field_value_node)
    -> Parse::Node {
  // TODO: Will be easier after #3393.
  CARBON_CHECK(context.parse_tree().node_kind(field_value_node) ==
               Parse::NodeKind::StructFieldValue);
  auto children = context.parse_tree().children(field_value_node);
  auto struct_field_designator_iter = children.begin();
  ++struct_field_designator_iter;
  CARBON_CHECK(context.parse_tree().node_kind(*struct_field_designator_iter) ==
               Parse::NodeKind::StructFieldDesignator);
  children = context.parse_tree().children(*struct_field_designator_iter);
  return *children.begin();
}

static auto DiagnoseDuplicateNames(Context& context,
                                   SemIR::InstBlockId type_block_id) -> bool {
  auto& sem_ir = context.sem_ir();
  auto fields = sem_ir.inst_blocks().Get(type_block_id);
  llvm::SmallDenseMap<SemIR::NameId, Parse::Node> names;
  auto& insts = sem_ir.insts();
  for (SemIR::InstId field_inst_id : fields) {
    auto field_inst = insts.GetAs<SemIR::StructTypeField>(field_inst_id);
    auto [it, added] =
        names.insert({field_inst.name_id, field_inst.parse_node});
    if (!added) {
      CARBON_DIAGNOSTIC(StructNameDuplicate, Error,
                        "Duplicated field name `{1}` in {0}.", llvm::StringRef,
                        llvm::StringRef);
      CARBON_DIAGNOSTIC(StructNamePrevious, Note,
                        "Field with the same name here.");
      llvm::StringRef container;
      Parse::Node prev_node = it->second;
      if (context.parse_tree().node_kind(field_inst.parse_node) ==
          Parse::NodeKind::StructFieldValue) {
        container = "struct literal";
        // This avoids using the wrong parse node in the struct literal case.
        prev_node = StructFieldValueToName(context, prev_node);
        field_inst.parse_node =
            StructFieldValueToName(context, field_inst.parse_node);
      } else {
        container = "struct type literal";
      }
      context.emitter()
          .Build(field_inst.parse_node, StructNameDuplicate, container,
                 sem_ir.names().GetFormatted(field_inst.name_id))
          .Note(prev_node, StructNamePrevious)
          .Emit();
      return true;
    }
  }
  return false;
}

auto HandleStructLiteral(Context& context, Parse::Node parse_node) -> bool {
  auto refs_id = context.ParamOrArgEnd(
      Parse::NodeKind::StructLiteralOrStructTypeLiteralStart);

  context.PopScope();
  context.node_stack()
      .PopAndDiscardSoloParseNode<
          Parse::NodeKind::StructLiteralOrStructTypeLiteralStart>();
  auto type_block_id = context.args_type_info_stack().Pop();
  if (DiagnoseDuplicateNames(context, type_block_id)) {
    context.node_stack().Push(parse_node, SemIR::InstId::BuiltinError);
    return true;
  }

  auto type_id = context.CanonicalizeStructType(parse_node, type_block_id);

  auto value_id =
      context.AddInst(SemIR::StructLiteral{parse_node, type_id, refs_id});
  context.node_stack().Push(parse_node, value_id);
  return true;
}

auto HandleStructLiteralOrStructTypeLiteralStart(Context& context,
                                                 Parse::Node parse_node)
    -> bool {
  context.PushScope();
  context.node_stack().Push(parse_node);
  // At this point we aren't sure whether this will be a value or type literal,
  // so we push onto args irrespective. It just won't be used for a type
  // literal.
  context.args_type_info_stack().Push();
  context.ParamOrArgStart();
  return true;
}

auto HandleStructTypeLiteral(Context& context, Parse::Node parse_node) -> bool {
  auto refs_id = context.ParamOrArgEnd(
      Parse::NodeKind::StructLiteralOrStructTypeLiteralStart);

  context.PopScope();
  context.node_stack()
      .PopAndDiscardSoloParseNode<
          Parse::NodeKind::StructLiteralOrStructTypeLiteralStart>();
  // This is only used for value literals.
  context.args_type_info_stack().Pop();

  CARBON_CHECK(refs_id != SemIR::InstBlockId::Empty)
      << "{} is handled by StructLiteral.";

  if (DiagnoseDuplicateNames(context, refs_id)) {
    context.node_stack().Push(parse_node, SemIR::InstId::BuiltinError);
    return true;
  }
  context.AddInstAndPush(
      parse_node,
      SemIR::StructType{parse_node, SemIR::TypeId::TypeType, refs_id});
  return true;
}

}  // namespace Carbon::Check
