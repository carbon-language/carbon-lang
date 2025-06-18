// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Value.h"
#include "toolchain/lower/function_context.h"
#include "toolchain/sem_ir/expr_info.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Lower {

auto HandleInst(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                SemIR::ClassDecl /*inst*/) -> void {
  // No action to perform.
}

static auto GetPointeeType(FunctionContext::TypeInFile type)
    -> FunctionContext::TypeInFile {
  return {.file = type.file,
          .type_id = type.file->GetPointeeType(type.type_id)};
}

// Extracts an element of an aggregate, such as a struct, tuple, or class, by
// index. Depending on the expression category and value representation of the
// aggregate input, this will either produce a value or a reference.
static auto GetAggregateElement(FunctionContext& context,
                                SemIR::InstId aggr_inst_id,
                                SemIR::ElementIndex idx,
                                SemIR::InstId result_inst_id, llvm::Twine name)
    -> llvm::Value* {
  auto* aggr_value = context.GetValue(aggr_inst_id);

  switch (SemIR::GetExprCategory(context.sem_ir(), aggr_inst_id)) {
    case SemIR::ExprCategory::Error:
    case SemIR::ExprCategory::NotExpr:
    case SemIR::ExprCategory::Initializing:
    case SemIR::ExprCategory::Mixed:
      CARBON_FATAL("Unexpected expression category for aggregate access");

    case SemIR::ExprCategory::Value: {
      auto aggr_type = context.GetTypeIdOfInst(aggr_inst_id);
      auto value_repr = context.GetValueRepr(aggr_type);
      CARBON_CHECK(
          value_repr.repr.aggregate_kind != SemIR::ValueRepr::NotAggregate,
          "aggregate type should have aggregate value representation");
      switch (value_repr.repr.kind) {
        case SemIR::ValueRepr::Unknown:
          CARBON_FATAL("Lowering access to incomplete aggregate type");
        case SemIR::ValueRepr::None:
          return aggr_value;
        case SemIR::ValueRepr::Copy:
          // We are holding the values of the aggregate directly, elementwise.
          return context.builder().CreateExtractValue(aggr_value, idx.index,
                                                      name);
        case SemIR::ValueRepr::Pointer: {
          // The value representation is a pointer to an aggregate that we want
          // to index into.
          auto* value_type = context.GetType(GetPointeeType(value_repr.type()));
          auto* elem_ptr = context.builder().CreateStructGEP(
              value_type, aggr_value, idx.index, name);

          if (!value_repr.repr.elements_are_values()) {
            // `elem_ptr` points to an object representation, which is our
            // result.
            return elem_ptr;
          }

          // `elem_ptr` points to a value representation. Load it.
          auto result_type = context.GetTypeIdOfInst(result_inst_id);
          auto result_value_type = context.GetValueRepr(result_type).type();
          return context.builder().CreateLoad(
              context.GetType(result_value_type), elem_ptr, name + ".load");
        }
        case SemIR::ValueRepr::Custom:
          CARBON_FATAL(
              "Aggregate should never have custom value representation");
      }
    }

    case SemIR::ExprCategory::DurableRef:
    case SemIR::ExprCategory::EphemeralRef: {
      // Just locate the aggregate element.
      auto* aggr_type = context.GetTypeOfInst(aggr_inst_id);
      return context.builder().CreateStructGEP(aggr_type, aggr_value, idx.index,
                                               name);
    }
  }
}

static auto GetStructFieldName(FunctionContext::TypeInFile struct_type,
                               SemIR::ElementIndex index) -> llvm::StringRef {
  auto struct_type_inst =
      struct_type.file->types().GetAs<SemIR::StructType>(struct_type.type_id);
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

static auto EmitAggregateInitializer(FunctionContext& context,
                                     SemIR::InstId init_inst_id,
                                     SemIR::InstBlockId refs_id,
                                     llvm::Twine name) -> llvm::Value* {
  auto type = context.GetTypeIdOfInst(init_inst_id);
  auto* llvm_type = context.GetType(type);
  auto refs = context.sem_ir().inst_blocks().Get(refs_id);

  switch (context.GetInitRepr(type).kind) {
    case SemIR::InitRepr::None: {
      // TODO: Add a helper to poison a value slot.
      return llvm::PoisonValue::get(llvm_type);
    }

    case SemIR::InitRepr::InPlace: {
      // Finish initialization of constant fields. We will have skipped this
      // when emitting the initializers because they have constant values.
      //
      // TODO: This emits the initializers for constant fields after all
      // initialization of non-constant fields. This may be observable in some
      // ways such as under a debugger in a debug build. It would be preferable
      // to initialize the constant portions of the aggregate first, but this
      // will likely need a change to the SemIR representation.
      //
      // TODO: If most of the bytes of the result have known constant values,
      // it'd be nice to emit a memcpy from a constant followed by the
      // non-constant initialization.
      for (auto [i, ref_id] : llvm::enumerate(refs)) {
        if (context.sem_ir().constant_values().Get(ref_id).is_constant()) {
          auto dest_id =
              SemIR::FindReturnSlotArgForInitializer(context.sem_ir(), ref_id);
          auto src_id = ref_id;
          auto storage_type = context.GetTypeIdOfInst(dest_id);
          context.FinishInit(storage_type, dest_id, src_id);
        }
      }
      // TODO: Add a helper to poison a value slot.
      return llvm::PoisonValue::get(llvm_type);
    }

    case SemIR::InitRepr::ByCopy: {
      auto refs = context.sem_ir().inst_blocks().Get(refs_id);
      CARBON_CHECK(
          refs.size() == 1,
          "Unexpected size for aggregate with by-copy value representation");
      // TODO: Remove the LLVM StructType wrapper in this case, so we don't
      // need this `insert_value` wrapping.
      return context.builder().CreateInsertValue(
          llvm::PoisonValue::get(llvm_type), context.GetValue(refs[0]), {0},
          name);
    }

    case SemIR::InitRepr::Incomplete:
      CARBON_FATAL("Lowering aggregate initialization of incomplete type {0}",
                   type.file->types().GetAsInst(type.type_id));
  }
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

// Emits the value representation for a struct or tuple whose elements are the
// contents of `refs_id`.
static auto EmitAggregateValueRepr(FunctionContext& context,
                                   SemIR::InstId value_inst_id,
                                   SemIR::InstBlockId refs_id) -> llvm::Value* {
  auto type = context.GetTypeIdOfInst(value_inst_id);
  auto value_repr = context.GetValueRepr(type);
  auto value_type = value_repr.type();
  switch (value_repr.repr.kind) {
    case SemIR::ValueRepr::Unknown:
      CARBON_FATAL("Incomplete aggregate type in lowering");

    case SemIR::ValueRepr::None:
      // TODO: Add a helper to get a "no value representation" value.
      return llvm::PoisonValue::get(context.GetType(value_type));

    case SemIR::ValueRepr::Copy: {
      auto refs = context.sem_ir().inst_blocks().Get(refs_id);
      CARBON_CHECK(
          refs.size() == 1,
          "Unexpected size for aggregate with by-copy value representation");
      // TODO: Remove the LLVM StructType wrapper in this case, so we don't
      // need this `insert_value` wrapping.
      return context.builder().CreateInsertValue(
          llvm::PoisonValue::get(context.GetType(value_type)),
          context.GetValue(refs[0]), {0});
    }

    case SemIR::ValueRepr::Pointer: {
      auto* llvm_value_rep_type = context.GetType(GetPointeeType(value_type));

      // Write the value representation to a local alloca so we can produce a
      // pointer to it as the value representation of the struct or tuple.
      auto* alloca = context.builder().CreateAlloca(llvm_value_rep_type);
      for (auto [i, ref] :
           llvm::enumerate(context.sem_ir().inst_blocks().Get(refs_id))) {
        context.builder().CreateStore(
            context.GetValue(ref),
            context.builder().CreateStructGEP(llvm_value_rep_type, alloca, i));
      }
      return alloca;
    }

    case SemIR::ValueRepr::Custom:
      CARBON_FATAL("Aggregate should never have custom value representation");
  }
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
