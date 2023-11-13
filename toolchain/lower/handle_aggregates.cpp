// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Value.h"
#include "toolchain/lower/function_context.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"

namespace Carbon::Lower {

auto HandleClassDecl(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                     SemIR::ClassDecl /*inst*/) -> void {
  // No action to perform.
}

// Extracts an element of an aggregate, such as a struct, tuple, or class, by
// index. Depending on the expression category and value representation of the
// aggregate input, this will either produce a value or a reference.
static auto GetAggregateElement(FunctionContext& context,
                                SemIR::InstId aggr_inst_id,
                                SemIR::MemberIndex idx,
                                SemIR::TypeId result_type_id, llvm::Twine name)
    -> llvm::Value* {
  auto aggr_inst = context.sem_ir().insts().Get(aggr_inst_id);
  auto* aggr_value = context.GetValue(aggr_inst_id);

  switch (SemIR::GetExprCategory(context.sem_ir(), aggr_inst_id)) {
    case SemIR::ExprCategory::Error:
    case SemIR::ExprCategory::NotExpr:
    case SemIR::ExprCategory::Initializing:
    case SemIR::ExprCategory::Mixed:
      CARBON_FATAL() << "Unexpected expression category for aggregate access";

    case SemIR::ExprCategory::Value: {
      auto value_rep =
          SemIR::GetValueRepresentation(context.sem_ir(), aggr_inst.type_id());
      CARBON_CHECK(value_rep.aggregate_kind !=
                   SemIR::ValueRepresentation::NotAggregate)
          << "aggregate type should have aggregate value representation";
      switch (value_rep.kind) {
        case SemIR::ValueRepresentation::Unknown:
          CARBON_FATAL() << "Lowering access to incomplete aggregate type";
        case SemIR::ValueRepresentation::None:
          return aggr_value;
        case SemIR::ValueRepresentation::Copy:
          // We are holding the values of the aggregate directly, elementwise.
          return context.builder().CreateExtractValue(aggr_value, idx.index,
                                                      name);
        case SemIR::ValueRepresentation::Pointer: {
          // The value representation is a pointer to an aggregate that we want
          // to index into.
          auto pointee_type_id =
              context.sem_ir().GetPointeeType(value_rep.type_id);
          auto* value_type = context.GetType(pointee_type_id);
          auto* elem_ptr = context.builder().CreateStructGEP(
              value_type, aggr_value, idx.index, name);

          if (!value_rep.elements_are_values()) {
            // `elem_ptr` points to an object representation, which is our
            // result.
            return elem_ptr;
          }

          // `elem_ptr` points to a value representation. Load it.
          auto result_value_type_id =
              SemIR::GetValueRepresentation(context.sem_ir(), result_type_id)
                  .type_id;
          return context.builder().CreateLoad(
              context.GetType(result_value_type_id), elem_ptr, name + ".load");
        }
        case SemIR::ValueRepresentation::Custom:
          CARBON_FATAL()
              << "Aggregate should never have custom value representation";
      }
    }

    case SemIR::ExprCategory::DurableReference:
    case SemIR::ExprCategory::EphemeralReference: {
      // Just locate the aggregate element.
      auto* aggr_type = context.GetType(aggr_inst.type_id());
      return context.builder().CreateStructGEP(aggr_type, aggr_value, idx.index,
                                               name);
    }
  }
}

static auto GetStructFieldName(FunctionContext& context,
                               SemIR::TypeId struct_type_id,
                               SemIR::MemberIndex index) -> llvm::StringRef {
  auto fields = context.sem_ir().inst_blocks().Get(
      context.sem_ir()
          .insts()
          .GetAs<SemIR::StructType>(
              context.sem_ir().types().Get(struct_type_id).inst_id)
          .fields_id);
  auto field = context.sem_ir().insts().GetAs<SemIR::StructTypeField>(
      fields[index.index]);
  return context.sem_ir().names().GetIRBaseName(field.name_id);
}

auto HandleClassFieldAccess(FunctionContext& context, SemIR::InstId inst_id,
                            SemIR::ClassFieldAccess inst) -> void {
  // Find the class that we're performing access into.
  auto class_type_id = context.sem_ir().insts().Get(inst.base_id).type_id();
  auto class_id =
      context.sem_ir()
          .insts()
          .GetAs<SemIR::ClassType>(
              context.sem_ir().GetTypeAllowBuiltinTypes(class_type_id))
          .class_id;
  auto& class_info = context.sem_ir().classes().Get(class_id);

  // Translate the class field access into a struct access on the object
  // representation.
  context.SetLocal(
      inst_id,
      GetAggregateElement(
          context, inst.base_id, inst.index, inst.type_id,
          GetStructFieldName(context, class_info.object_representation_id,
                             inst.index)));
}

static auto EmitAggregateInitializer(FunctionContext& context,
                                     SemIR::TypeId type_id,
                                     SemIR::InstBlockId refs_id,
                                     llvm::Twine name) -> llvm::Value* {
  auto* llvm_type = context.GetType(type_id);

  switch (
      SemIR::GetInitializingRepresentation(context.sem_ir(), type_id).kind) {
    case SemIR::InitializingRepresentation::None:
    case SemIR::InitializingRepresentation::InPlace:
      // TODO: Add a helper to poison a value slot.
      return llvm::PoisonValue::get(llvm_type);

    case SemIR::InitializingRepresentation::ByCopy: {
      auto refs = context.sem_ir().inst_blocks().Get(refs_id);
      CARBON_CHECK(refs.size() == 1)
          << "Unexpected size for aggregate with by-copy value representation";
      // TODO: Remove the LLVM StructType wrapper in this case, so we don't
      // need this `insert_value` wrapping.
      return context.builder().CreateInsertValue(
          llvm::PoisonValue::get(llvm_type), context.GetValue(refs[0]), {0},
          name);
    }
  }
}

auto HandleClassInit(FunctionContext& context, SemIR::InstId inst_id,
                     SemIR::ClassInit inst) -> void {
  context.SetLocal(
      inst_id, EmitAggregateInitializer(context, inst.type_id, inst.elements_id,
                                        "class.init"));
}

auto HandleField(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                 SemIR::Field /*inst*/) -> void {
  // No action to perform.
}

auto HandleStructAccess(FunctionContext& context, SemIR::InstId inst_id,
                        SemIR::StructAccess inst) -> void {
  auto struct_type_id = context.sem_ir().insts().Get(inst.struct_id).type_id();
  context.SetLocal(
      inst_id, GetAggregateElement(
                   context, inst.struct_id, inst.index, inst.type_id,
                   GetStructFieldName(context, struct_type_id, inst.index)));
}

auto HandleStructLiteral(FunctionContext& /*context*/,
                         SemIR::InstId /*inst_id*/,
                         SemIR::StructLiteral /*inst*/) -> void {
  // A StructLiteral should always be converted to a StructInit or StructValue
  // if its value is needed.
}

// Emits the value representation for a struct or tuple whose elements are the
// contents of `refs_id`.
auto EmitAggregateValueRepresentation(FunctionContext& context,
                                      SemIR::TypeId type_id,
                                      SemIR::InstBlockId refs_id,
                                      llvm::Twine name) -> llvm::Value* {
  auto value_rep = SemIR::GetValueRepresentation(context.sem_ir(), type_id);
  switch (value_rep.kind) {
    case SemIR::ValueRepresentation::Unknown:
      CARBON_FATAL() << "Incomplete aggregate type in lowering";

    case SemIR::ValueRepresentation::None:
      // TODO: Add a helper to get a "no value representation" value.
      return llvm::PoisonValue::get(context.GetType(value_rep.type_id));

    case SemIR::ValueRepresentation::Copy: {
      auto refs = context.sem_ir().inst_blocks().Get(refs_id);
      CARBON_CHECK(refs.size() == 1)
          << "Unexpected size for aggregate with by-copy value representation";
      // TODO: Remove the LLVM StructType wrapper in this case, so we don't
      // need this `insert_value` wrapping.
      return context.builder().CreateInsertValue(
          llvm::PoisonValue::get(context.GetType(value_rep.type_id)),
          context.GetValue(refs[0]), {0});
    }

    case SemIR::ValueRepresentation::Pointer: {
      auto pointee_type_id = context.sem_ir().GetPointeeType(value_rep.type_id);
      auto* llvm_value_rep_type = context.GetType(pointee_type_id);

      // Write the value representation to a local alloca so we can produce a
      // pointer to it as the value representation of the struct or tuple.
      auto* alloca =
          context.builder().CreateAlloca(llvm_value_rep_type,
                                         /*ArraySize=*/nullptr, name);
      for (auto [i, ref] :
           llvm::enumerate(context.sem_ir().inst_blocks().Get(refs_id))) {
        context.builder().CreateStore(
            context.GetValue(ref),
            context.builder().CreateStructGEP(llvm_value_rep_type, alloca, i));
      }
      return alloca;
    }

    case SemIR::ValueRepresentation::Custom:
      CARBON_FATAL()
          << "Aggregate should never have custom value representation";
  }
}

auto HandleStructInit(FunctionContext& context, SemIR::InstId inst_id,
                      SemIR::StructInit inst) -> void {
  context.SetLocal(
      inst_id, EmitAggregateInitializer(context, inst.type_id, inst.elements_id,
                                        "struct.init"));
}

auto HandleStructValue(FunctionContext& context, SemIR::InstId inst_id,
                       SemIR::StructValue inst) -> void {
  context.SetLocal(
      inst_id, EmitAggregateValueRepresentation(context, inst.type_id,
                                                inst.elements_id, "struct"));
}

auto HandleStructTypeField(FunctionContext& /*context*/,
                           SemIR::InstId /*inst_id*/,
                           SemIR::StructTypeField /*inst*/) -> void {
  // No action to take.
}

auto HandleTupleAccess(FunctionContext& context, SemIR::InstId inst_id,
                       SemIR::TupleAccess inst) -> void {
  context.SetLocal(inst_id,
                   GetAggregateElement(context, inst.tuple_id, inst.index,
                                       inst.type_id, "tuple.elem"));
}

auto HandleTupleIndex(FunctionContext& context, SemIR::InstId inst_id,
                      SemIR::TupleIndex inst) -> void {
  auto index_inst =
      context.sem_ir().insts().GetAs<SemIR::IntegerLiteral>(inst.index_id);
  auto index =
      context.sem_ir().integers().Get(index_inst.integer_id).getZExtValue();
  context.SetLocal(inst_id, GetAggregateElement(context, inst.tuple_id,
                                                SemIR::MemberIndex(index),
                                                inst.type_id, "tuple.index"));
}

auto HandleTupleLiteral(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                        SemIR::TupleLiteral /*inst*/) -> void {
  // A TupleLiteral should always be converted to a TupleInit or TupleValue if
  // its value is needed.
}

auto HandleTupleInit(FunctionContext& context, SemIR::InstId inst_id,
                     SemIR::TupleInit inst) -> void {
  context.SetLocal(
      inst_id, EmitAggregateInitializer(context, inst.type_id, inst.elements_id,
                                        "tuple.init"));
}

auto HandleTupleValue(FunctionContext& context, SemIR::InstId inst_id,
                      SemIR::TupleValue inst) -> void {
  context.SetLocal(inst_id,
                   EmitAggregateValueRepresentation(context, inst.type_id,
                                                    inst.elements_id, "tuple"));
}

}  // namespace Carbon::Lower
