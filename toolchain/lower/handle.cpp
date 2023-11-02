// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "toolchain/lower/function_context.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"

namespace Carbon::Lower {

auto HandleCrossReference(FunctionContext& /*context*/,
                          SemIR::InstId /*inst_id*/, SemIR::CrossReference inst)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << inst;
}

auto HandleAddressOf(FunctionContext& context, SemIR::InstId inst_id,
                     SemIR::AddressOf inst) -> void {
  context.SetLocal(inst_id, context.GetLocal(inst.lvalue_id));
}

auto HandleArrayIndex(FunctionContext& context, SemIR::InstId inst_id,
                      SemIR::ArrayIndex inst) -> void {
  auto* array_value = context.GetLocal(inst.array_id);
  auto* llvm_type =
      context.GetType(context.sem_ir().insts().Get(inst.array_id).type_id());
  llvm::Value* indexes[2] = {
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context.llvm_context()), 0),
      context.GetLocal(inst.index_id)};
  context.SetLocal(inst_id,
                   context.builder().CreateInBoundsGEP(llvm_type, array_value,
                                                       indexes, "array.index"));
}

auto HandleArrayInit(FunctionContext& context, SemIR::InstId inst_id,
                     SemIR::ArrayInit inst) -> void {
  // The result of initialization is the return slot of the initializer.
  context.SetLocal(inst_id,
                   context.GetLocal(context.sem_ir()
                                        .inst_blocks()
                                        .Get(inst.inits_and_return_slot_id)
                                        .back()));
}

auto HandleAssign(FunctionContext& context, SemIR::InstId /*inst_id*/,
                  SemIR::Assign inst) -> void {
  auto storage_type_id = context.sem_ir().insts().Get(inst.lhs_id).type_id();
  context.FinishInitialization(storage_type_id, inst.lhs_id, inst.rhs_id);
}

auto HandleBinaryOperatorAdd(FunctionContext& /*context*/,
                             SemIR::InstId /*inst_id*/,
                             SemIR::BinaryOperatorAdd inst) -> void {
  CARBON_FATAL() << "TODO: Add support: " << inst;
}

auto HandleBindName(FunctionContext& context, SemIR::InstId inst_id,
                    SemIR::BindName inst) -> void {
  context.SetLocal(inst_id, context.GetLocal(inst.value_id));
}

auto HandleBlockArg(FunctionContext& context, SemIR::InstId inst_id,
                    SemIR::BlockArg inst) -> void {
  context.SetLocal(inst_id, context.GetBlockArg(inst.block_id, inst.type_id));
}

auto HandleBoolLiteral(FunctionContext& context, SemIR::InstId inst_id,
                       SemIR::BoolLiteral inst) -> void {
  llvm::Value* v =
      llvm::ConstantInt::get(context.builder().getInt1Ty(), inst.value.index);
  context.SetLocal(inst_id, v);
}

auto HandleBoundMethod(FunctionContext& context, SemIR::InstId inst_id,
                       SemIR::BoundMethod inst) -> void {
  // Propagate just the function; the object is separately provided to the
  // enclosing call as an implicit argument.
  context.SetLocal(inst_id, context.GetLocalOrGlobal(inst.function_id));
}

auto HandleBranch(FunctionContext& context, SemIR::InstId /*inst_id*/,
                  SemIR::Branch inst) -> void {
  // Opportunistically avoid creating a BasicBlock that contains just a branch.
  // TODO: Don't do this if it would remove a loop preheader block.
  llvm::BasicBlock* block = context.builder().GetInsertBlock();
  if (block->empty() && context.TryToReuseBlock(inst.target_id, block)) {
    // Reuse this block as the branch target.
  } else {
    context.builder().CreateBr(context.GetBlock(inst.target_id));
  }

  context.builder().ClearInsertionPoint();
}

auto HandleBranchIf(FunctionContext& context, SemIR::InstId /*inst_id*/,
                    SemIR::BranchIf inst) -> void {
  llvm::Value* cond = context.GetLocal(inst.cond_id);
  llvm::BasicBlock* then_block = context.GetBlock(inst.target_id);
  llvm::BasicBlock* else_block = context.CreateSyntheticBlock();
  context.builder().CreateCondBr(cond, then_block, else_block);
  context.builder().SetInsertPoint(else_block);
}

auto HandleBranchWithArg(FunctionContext& context, SemIR::InstId /*inst_id*/,
                         SemIR::BranchWithArg inst) -> void {
  llvm::Value* arg = context.GetLocal(inst.arg_id);
  SemIR::TypeId arg_type_id =
      context.sem_ir().insts().Get(inst.arg_id).type_id();

  // Opportunistically avoid creating a BasicBlock that contains just a branch.
  // We only do this for a block that we know will only have a single
  // predecessor, so that we can correctly populate the predecessors of the
  // PHINode.
  llvm::BasicBlock* block = context.builder().GetInsertBlock();
  llvm::BasicBlock* phi_predecessor = block;
  if (block->empty() && context.IsCurrentSyntheticBlock(block) &&
      context.TryToReuseBlock(inst.target_id, block)) {
    // Reuse this block as the branch target.
    phi_predecessor = block->getSinglePredecessor();
    CARBON_CHECK(phi_predecessor)
        << "Synthetic block did not have a single predecessor";
  } else {
    context.builder().CreateBr(context.GetBlock(inst.target_id));
  }

  context.GetBlockArg(inst.target_id, arg_type_id)
      ->addIncoming(arg, phi_predecessor);
  context.builder().ClearInsertionPoint();
}

auto HandleBuiltin(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                   SemIR::Builtin inst) -> void {
  CARBON_FATAL() << "TODO: Add support: " << inst;
}

auto HandleCall(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::Call inst) -> void {
  auto* callee = llvm::cast<llvm::Function>(context.GetLocal(inst.callee_id));

  std::vector<llvm::Value*> args;
  llvm::ArrayRef<SemIR::InstId> arg_ids =
      context.sem_ir().inst_blocks().Get(inst.args_id);

  if (SemIR::GetInitializingRepresentation(context.sem_ir(), inst.type_id)
          .has_return_slot()) {
    args.push_back(context.GetLocal(arg_ids.back()));
    arg_ids = arg_ids.drop_back();
  }

  for (auto arg_id : arg_ids) {
    auto arg_type_id = context.sem_ir().insts().Get(arg_id).type_id();
    if (SemIR::GetValueRepresentation(context.sem_ir(), arg_type_id).kind !=
        SemIR::ValueRepresentation::None) {
      args.push_back(context.GetLocal(arg_id));
    }
  }

  auto* call = context.builder().CreateCall(callee, args);
  context.SetLocal(inst_id, call);

  // Name the call's result the same as the callee.
  // TODO: Is this a helpful name?
  if (!call->getType()->isVoidTy()) {
    call->setName(callee->getName());
  }
}

auto HandleClassDeclaration(FunctionContext& /*context*/,
                            SemIR::InstId /*inst_id*/,
                            SemIR::ClassDeclaration /*inst*/) -> void {
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
  auto* aggr_value = context.GetLocal(aggr_inst_id);

  switch (SemIR::GetExpressionCategory(context.sem_ir(), aggr_inst_id)) {
    case SemIR::ExpressionCategory::Error:
    case SemIR::ExpressionCategory::NotExpression:
    case SemIR::ExpressionCategory::Initializing:
    case SemIR::ExpressionCategory::Mixed:
      CARBON_FATAL() << "Unexpected expression category for aggregate access";

    case SemIR::ExpressionCategory::Value: {
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

    case SemIR::ExpressionCategory::DurableReference:
    case SemIR::ExpressionCategory::EphemeralReference: {
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
  return context.sem_ir().identifiers().Get(field.name_id);
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
          llvm::PoisonValue::get(llvm_type), context.GetLocal(refs[0]), {0},
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

auto HandleDereference(FunctionContext& context, SemIR::InstId inst_id,
                       SemIR::Dereference inst) -> void {
  context.SetLocal(inst_id, context.GetLocal(inst.pointer_id));
}

auto HandleField(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                 SemIR::Field /*inst*/) -> void {
  // No action to perform.
}

auto HandleFunctionDeclaration(FunctionContext& /*context*/,
                               SemIR::InstId /*inst_id*/,
                               SemIR::FunctionDeclaration inst) -> void {
  CARBON_FATAL()
      << "Should not be encountered. If that changes, we may want to change "
         "higher-level logic to skip them rather than calling this. "
      << inst;
}

auto HandleInitializeFrom(FunctionContext& context, SemIR::InstId /*inst_id*/,
                          SemIR::InitializeFrom inst) -> void {
  auto storage_type_id = context.sem_ir().insts().Get(inst.dest_id).type_id();
  context.FinishInitialization(storage_type_id, inst.dest_id, inst.src_id);
}

auto HandleIntegerLiteral(FunctionContext& context, SemIR::InstId inst_id,
                          SemIR::IntegerLiteral inst) -> void {
  const llvm::APInt& i = context.sem_ir().integers().Get(inst.integer_id);
  // TODO: This won't offer correct semantics, but seems close enough for now.
  llvm::Value* v =
      llvm::ConstantInt::get(context.builder().getInt32Ty(), i.getZExtValue());
  context.SetLocal(inst_id, v);
}

auto HandleNameReference(FunctionContext& context, SemIR::InstId inst_id,
                         SemIR::NameReference inst) -> void {
  auto type_inst_id = context.sem_ir().GetTypeAllowBuiltinTypes(inst.type_id);
  if (type_inst_id == SemIR::InstId::BuiltinNamespaceType) {
    return;
  }

  context.SetLocal(inst_id, context.GetLocalOrGlobal(inst.value_id));
}

auto HandleNamespace(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                     SemIR::Namespace inst) -> void {
  CARBON_FATAL()
      << "Should not be encountered. If that changes, we may want to change "
         "higher-level logic to skip them rather than calling this. "
      << inst;
}

auto HandleNoOp(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                SemIR::NoOp /*inst*/) -> void {
  // No action to take.
}

auto HandleParameter(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                     SemIR::Parameter /*inst*/) -> void {
  CARBON_FATAL() << "Parameters should be lowered by `BuildFunctionDefinition`";
}

auto HandleRealLiteral(FunctionContext& context, SemIR::InstId inst_id,
                       SemIR::RealLiteral inst) -> void {
  const Real& real = context.sem_ir().reals().Get(inst.real_id);
  // TODO: This will probably have overflow issues, and should be fixed.
  double val =
      real.mantissa.getZExtValue() *
      std::pow((real.is_decimal ? 10 : 2), real.exponent.getSExtValue());
  llvm::APFloat llvm_val(val);
  context.SetLocal(inst_id, llvm::ConstantFP::get(
                                context.builder().getDoubleTy(), llvm_val));
}

auto HandleReturn(FunctionContext& context, SemIR::InstId /*inst_id*/,
                  SemIR::Return /*inst*/) -> void {
  context.builder().CreateRetVoid();
}

auto HandleReturnExpression(FunctionContext& context, SemIR::InstId /*inst_id*/,
                            SemIR::ReturnExpression inst) -> void {
  switch (SemIR::GetInitializingRepresentation(
              context.sem_ir(),
              context.sem_ir().insts().Get(inst.expr_id).type_id())
              .kind) {
    case SemIR::InitializingRepresentation::None:
    case SemIR::InitializingRepresentation::InPlace:
      // Nothing to return.
      context.builder().CreateRetVoid();
      return;
    case SemIR::InitializingRepresentation::ByCopy:
      // The expression produces the value representation for the type.
      context.builder().CreateRet(context.GetLocal(inst.expr_id));
      return;
  }
}

auto HandleSelfParameter(FunctionContext& /*context*/,
                         SemIR::InstId /*inst_id*/,
                         SemIR::SelfParameter /*inst*/) -> void {
  CARBON_FATAL() << "Parameters should be lowered by `BuildFunctionDefinition`";
}

auto HandleSpliceBlock(FunctionContext& context, SemIR::InstId inst_id,
                       SemIR::SpliceBlock inst) -> void {
  context.LowerBlock(inst.block_id);
  context.SetLocal(inst_id, context.GetLocal(inst.result_id));
}

auto HandleStringLiteral(FunctionContext& /*context*/,
                         SemIR::InstId /*inst_id*/, SemIR::StringLiteral inst)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << inst;
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
          context.GetLocal(refs[0]), {0});
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
            context.GetLocal(ref),
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

auto HandleUnaryOperatorNot(FunctionContext& context, SemIR::InstId inst_id,
                            SemIR::UnaryOperatorNot inst) -> void {
  context.SetLocal(
      inst_id, context.builder().CreateNot(context.GetLocal(inst.operand_id)));
}

auto HandleVarStorage(FunctionContext& context, SemIR::InstId inst_id,
                      SemIR::VarStorage inst) -> void {
  // TODO: Eventually this name will be optional, and we'll want to provide
  // something like `var` as a default. However, that's not possible right now
  // so cannot be tested.
  auto name = context.sem_ir().identifiers().Get(inst.name_id);
  auto* alloca = context.builder().CreateAlloca(context.GetType(inst.type_id),
                                                /*ArraySize=*/nullptr, name);
  context.SetLocal(inst_id, alloca);
}

}  // namespace Carbon::Lower
