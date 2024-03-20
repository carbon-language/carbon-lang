// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"

namespace Carbon::Check {

auto HandleStructLiteralOrStructTypeLiteralStart(
    Context& context, Parse::StructLiteralOrStructTypeLiteralStartId node_id)
    -> bool {
  context.scope_stack().Push();
  context.node_stack().Push(node_id);
  // At this point we aren't sure whether this will be a value or type literal,
  // so we push onto args irrespective. It just won't be used for a type
  // literal.
  context.args_type_info_stack().Push();
  context.param_and_arg_refs_stack().Push();
  return true;
}

auto HandleStructFieldDesignator(Context& context,
                                 Parse::StructFieldDesignatorId /*node_id*/)
    -> bool {
  // This leaves the designated name on top because the `.` isn't interesting.
  CARBON_CHECK(context.node_stack().PeekIsName());
  return true;
}

auto HandleStructComma(Context& context, Parse::StructCommaId /*node_id*/)
    -> bool {
  context.param_and_arg_refs_stack().ApplyComma();
  return true;
}

auto HandleStructFieldValue(Context& context, Parse::StructFieldValueId node_id)
    -> bool {
  auto value_inst_id = context.node_stack().PopExpr();
  auto [name_node, name_id] = context.node_stack().PopNameWithNodeId();

  // Store the name for the type.
  context.args_type_info_stack().AddInstId(context.AddInstInNoBlock(
      {name_node, SemIR::StructTypeField{
                      name_id, context.insts().Get(value_inst_id).type_id()}}));

  // Push the value back on the stack as an argument.
  context.node_stack().Push(node_id, value_inst_id);
  return true;
}

auto HandleStructFieldType(Context& context, Parse::StructFieldTypeId node_id)
    -> bool {
  auto [type_node, type_id] = context.node_stack().PopExprWithNodeId();
  SemIR::TypeId cast_type_id = ExprAsType(context, type_node, type_id);

  auto [name_node, name_id] = context.node_stack().PopNameWithNodeId();

  auto inst_id = context.AddInst(
      {name_node, SemIR::StructTypeField{name_id, cast_type_id}});
  context.node_stack().Push(node_id, inst_id);
  return true;
}

static auto DiagnoseDuplicateNames(Context& context,
                                   SemIR::InstBlockId type_block_id,
                                   llvm::StringRef construct) -> bool {
  auto& sem_ir = context.sem_ir();
  auto fields = sem_ir.inst_blocks().Get(type_block_id);
  llvm::SmallDenseMap<SemIR::NameId, SemIR::InstId> names;
  auto& insts = sem_ir.insts();
  for (SemIR::InstId field_inst_id : fields) {
    auto field_inst = insts.GetAs<SemIR::StructTypeField>(field_inst_id);
    auto [it, added] = names.insert({field_inst.name_id, field_inst_id});
    if (!added) {
      CARBON_DIAGNOSTIC(StructNameDuplicate, Error,
                        "Duplicated field name `{1}` in {0}.", std::string,
                        SemIR::NameId);
      CARBON_DIAGNOSTIC(StructNamePrevious, Note,
                        "Field with the same name here.");
      context.emitter()
          .Build(field_inst_id, StructNameDuplicate, construct.str(),
                 field_inst.name_id)
          .Note(it->second, StructNamePrevious)
          .Emit();
      return true;
    }
  }
  return false;
}

auto HandleStructLiteral(Context& context, Parse::StructLiteralId node_id)
    -> bool {
  auto refs_id = context.param_and_arg_refs_stack().EndAndPop(
      Parse::NodeKind::StructLiteralOrStructTypeLiteralStart);

  context.scope_stack().Pop();
  context.node_stack()
      .PopAndDiscardSoloNodeId<
          Parse::NodeKind::StructLiteralOrStructTypeLiteralStart>();
  auto type_block_id = context.args_type_info_stack().Pop();
  if (DiagnoseDuplicateNames(context, type_block_id, "struct literal")) {
    context.node_stack().Push(node_id, SemIR::InstId::BuiltinError);
    return true;
  }

  auto type_id = context.GetStructType(type_block_id);

  auto value_id =
      context.AddInst({node_id, SemIR::StructLiteral{type_id, refs_id}});
  context.node_stack().Push(node_id, value_id);
  return true;
}

auto HandleStructTypeLiteral(Context& context,
                             Parse::StructTypeLiteralId node_id) -> bool {
  auto refs_id = context.param_and_arg_refs_stack().EndAndPop(
      Parse::NodeKind::StructLiteralOrStructTypeLiteralStart);

  context.scope_stack().Pop();
  context.node_stack()
      .PopAndDiscardSoloNodeId<
          Parse::NodeKind::StructLiteralOrStructTypeLiteralStart>();
  // This is only used for value literals.
  context.args_type_info_stack().Pop();

  CARBON_CHECK(refs_id != SemIR::InstBlockId::Empty)
      << "{} is handled by StructLiteral.";

  if (DiagnoseDuplicateNames(context, refs_id, "struct type literal")) {
    context.node_stack().Push(node_id, SemIR::InstId::BuiltinError);
    return true;
  }
  context.AddInstAndPush(
      {node_id, SemIR::StructType{SemIR::TypeId::TypeType, refs_id}});
  return true;
}

}  // namespace Carbon::Check
