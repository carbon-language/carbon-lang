// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Value.h"
#include "toolchain/lower/aggregate.h"
#include "toolchain/lower/function_context.h"
#include "toolchain/sem_ir/expr_info.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Lower {

auto HandleInst(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                SemIR::ClassDecl /*inst*/) -> void {
  // No action to perform.
}

static auto GetStructFieldName(FunctionContext::TypeInFile struct_type,
                               SemIR::ElementIndex index) -> llvm::StringRef {
  auto struct_type_inst = struct_type.file->types().GetAs<SemIR::AnyStructType>(
      struct_type.type_id);
  auto fields =
      struct_type.file->struct_type_fields().Get(struct_type_inst.fields_id);
  // We intentionally don't add this to the fingerprint because it's only used
  // as an instruction name, and so doesn't affect the semantics of the IR.
  return struct_type.file->names().GetIRBaseName(fields[index.index].name_id);
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::ClassElementAccess inst) -> void {
  // Find the class that we're performing access into.
  auto class_type = context.GetTypeIdOfInst(inst.base_id);
  auto object_repr = FunctionContext::TypeInFile{
      .file = class_type.file,
      .type_id = class_type.file->types().GetObjectRepr(class_type.type_id)};

  // Translate the class field access into a struct access on the object
  // representation.
  context.SetLocal(inst_id, GetAggregateElement(
                                context, inst.base_id, inst.index, inst_id,
                                GetStructFieldName(object_repr, inst.index)));
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::ClassInit inst) -> void {
  context.SetLocal(inst_id,
                   EmitAggregateInitializer(context, inst_id, inst.elements_id,
                                            "class.init"));
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::StructAccess inst) -> void {
  auto struct_type = context.GetTypeIdOfInst(inst.struct_id);
  context.SetLocal(inst_id, GetAggregateElement(
                                context, inst.struct_id, inst.index, inst_id,
                                GetStructFieldName(struct_type, inst.index)));
}

auto HandleInst(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                SemIR::StructLiteral /*inst*/) -> void {
  // A StructLiteral should always be converted to a StructInit or StructValue
  // if its value is needed.
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::StructInit inst) -> void {
  context.SetLocal(inst_id,
                   EmitAggregateInitializer(context, inst_id, inst.elements_id,
                                            "struct.init"));
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::StructValue inst) -> void {
  auto type = context.GetTypeIdOfInst(inst_id);
  if (auto fn_type =
          type.file->types().TryGetAs<SemIR::FunctionType>(type.type_id)) {
    context.SetLocal(inst_id, context.GetFileContext(type.file).GetFunction(
                                  fn_type->function_id));
    return;
  }

  context.SetLocal(inst_id,
                   EmitAggregateValueRepr(context, inst_id, inst.elements_id));
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::TupleAccess inst) -> void {
  context.SetLocal(
      inst_id, GetAggregateElement(context, inst.tuple_id, inst.index, inst_id,
                                   "tuple.elem"));
}

auto HandleInst(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                SemIR::TupleLiteral /*inst*/) -> void {
  // A TupleLiteral should always be converted to a TupleInit or TupleValue if
  // its value is needed.
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::TupleInit inst) -> void {
  context.SetLocal(inst_id,
                   EmitAggregateInitializer(context, inst_id, inst.elements_id,
                                            "tuple.init"));
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::TupleValue inst) -> void {
  context.SetLocal(inst_id,
                   EmitAggregateValueRepr(context, inst_id, inst.elements_id));
}

}  // namespace Carbon::Lower
