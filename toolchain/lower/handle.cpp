// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "toolchain/lower/function_context.h"
#include "toolchain/sem_ir/node.h"
#include "toolchain/sem_ir/node_kind.h"

namespace Carbon::Lower {

auto HandleCrossReference(FunctionContext& /*context*/,
                          SemIR::NodeId /*node_id*/, SemIR::CrossReference node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto HandleAddressOf(FunctionContext& context, SemIR::NodeId node_id,
                     SemIR::AddressOf node) -> void {
  context.SetLocal(node_id, context.GetLocal(node.lvalue_id));
}

auto HandleArrayIndex(FunctionContext& context, SemIR::NodeId node_id,
                      SemIR::ArrayIndex node) -> void {
  auto* array_value = context.GetLocal(node.array_id);
  auto* llvm_type =
      context.GetType(context.semantics_ir().GetNode(node.array_id).type_id());
  llvm::Value* indexes[2] = {
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context.llvm_context()), 0),
      context.GetLocal(node.index_id)};
  context.SetLocal(node_id,
                   context.builder().CreateInBoundsGEP(llvm_type, array_value,
                                                       indexes, "array.index"));
}

auto HandleArrayInit(FunctionContext& context, SemIR::NodeId node_id,
                     SemIR::ArrayInit node) -> void {
  // The result of initialization is the return slot of the initializer.
  context.SetLocal(
      node_id, context.GetLocal(context.semantics_ir()
                                    .GetNodeBlock(node.inits_and_return_slot_id)
                                    .back()));
}

auto HandleAssign(FunctionContext& context, SemIR::NodeId /*node_id*/,
                  SemIR::Assign node) -> void {
  auto storage_type_id = context.semantics_ir().GetNode(node.lhs_id).type_id();
  context.FinishInitialization(storage_type_id, node.lhs_id, node.rhs_id);
}

auto HandleBinaryOperatorAdd(FunctionContext& /*context*/,
                             SemIR::NodeId /*node_id*/,
                             SemIR::BinaryOperatorAdd node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto HandleBindName(FunctionContext& context, SemIR::NodeId node_id,
                    SemIR::BindName node) -> void {
  context.SetLocal(node_id, context.GetLocal(node.value_id));
}

auto HandleBlockArg(FunctionContext& context, SemIR::NodeId node_id,
                    SemIR::BlockArg node) -> void {
  context.SetLocal(node_id, context.GetBlockArg(node.block_id, node.type_id));
}

auto HandleBoolLiteral(FunctionContext& context, SemIR::NodeId node_id,
                       SemIR::BoolLiteral node) -> void {
  llvm::Value* v =
      llvm::ConstantInt::get(context.builder().getInt1Ty(), node.value.index);
  context.SetLocal(node_id, v);
}

auto HandleBranch(FunctionContext& context, SemIR::NodeId /*node_id*/,
                  SemIR::Branch node) -> void {
  // Opportunistically avoid creating a BasicBlock that contains just a branch.
  // TODO: Don't do this if it would remove a loop preheader block.
  llvm::BasicBlock* block = context.builder().GetInsertBlock();
  if (block->empty() && context.TryToReuseBlock(node.target_id, block)) {
    // Reuse this block as the branch target.
  } else {
    context.builder().CreateBr(context.GetBlock(node.target_id));
  }

  context.builder().ClearInsertionPoint();
}

auto HandleBranchIf(FunctionContext& context, SemIR::NodeId /*node_id*/,
                    SemIR::BranchIf node) -> void {
  llvm::Value* cond = context.GetLocal(node.cond_id);
  llvm::BasicBlock* then_block = context.GetBlock(node.target_id);
  llvm::BasicBlock* else_block = context.CreateSyntheticBlock();
  context.builder().CreateCondBr(cond, then_block, else_block);
  context.builder().SetInsertPoint(else_block);
}

auto HandleBranchWithArg(FunctionContext& context, SemIR::NodeId /*node_id*/,
                         SemIR::BranchWithArg node) -> void {
  llvm::Value* arg = context.GetLocal(node.arg_id);
  SemIR::TypeId arg_type_id =
      context.semantics_ir().GetNode(node.arg_id).type_id();

  // Opportunistically avoid creating a BasicBlock that contains just a branch.
  // We only do this for a block that we know will only have a single
  // predecessor, so that we can correctly populate the predecessors of the
  // PHINode.
  llvm::BasicBlock* block = context.builder().GetInsertBlock();
  llvm::BasicBlock* phi_predecessor = block;
  if (block->empty() && context.IsCurrentSyntheticBlock(block) &&
      context.TryToReuseBlock(node.target_id, block)) {
    // Reuse this block as the branch target.
    phi_predecessor = block->getSinglePredecessor();
    CARBON_CHECK(phi_predecessor)
        << "Synthetic block did not have a single predecessor";
  } else {
    context.builder().CreateBr(context.GetBlock(node.target_id));
  }

  context.GetBlockArg(node.target_id, arg_type_id)
      ->addIncoming(arg, phi_predecessor);
  context.builder().ClearInsertionPoint();
}

auto HandleBuiltin(FunctionContext& /*context*/, SemIR::NodeId /*node_id*/,
                   SemIR::Builtin node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto HandleCall(FunctionContext& context, SemIR::NodeId node_id,
                SemIR::Call node) -> void {
  auto* callee = llvm::cast<llvm::Function>(context.GetLocal(node.callee_id));

  std::vector<llvm::Value*> args;
  llvm::ArrayRef<SemIR::NodeId> arg_ids =
      context.semantics_ir().GetNodeBlock(node.args_id);

  if (SemIR::GetInitializingRepresentation(context.semantics_ir(), node.type_id)
          .has_return_slot()) {
    args.push_back(context.GetLocal(arg_ids.back()));
    arg_ids = arg_ids.drop_back();
  }

  for (auto arg_id : arg_ids) {
    auto arg_type_id = context.semantics_ir().GetNode(arg_id).type_id();
    if (SemIR::GetValueRepresentation(context.semantics_ir(), arg_type_id)
            .kind != SemIR::ValueRepresentation::None) {
      args.push_back(context.GetLocal(arg_id));
    }
  }

  auto* call = context.builder().CreateCall(callee, args);
  context.SetLocal(node_id, call);

  // Name the call's result the same as the callee.
  // TODO: Is this a helpful name?
  if (!call->getType()->isVoidTy()) {
    call->setName(callee->getName());
  }
}

auto HandleClassDeclaration(FunctionContext& /*context*/,
                            SemIR::NodeId /*node_id*/,
                            SemIR::ClassDeclaration /*node*/) -> void {
  // No action to perform.
}

auto HandleDereference(FunctionContext& context, SemIR::NodeId node_id,
                       SemIR::Dereference node) -> void {
  context.SetLocal(node_id, context.GetLocal(node.pointer_id));
}

auto HandleFunctionDeclaration(FunctionContext& /*context*/,
                               SemIR::NodeId /*node_id*/,
                               SemIR::FunctionDeclaration node) -> void {
  CARBON_FATAL()
      << "Should not be encountered. If that changes, we may want to change "
         "higher-level logic to skip them rather than calling this. "
      << node;
}

auto HandleInitializeFrom(FunctionContext& context, SemIR::NodeId /*node_id*/,
                          SemIR::InitializeFrom node) -> void {
  auto storage_type_id = context.semantics_ir().GetNode(node.dest_id).type_id();
  context.FinishInitialization(storage_type_id, node.dest_id, node.src_id);
}

auto HandleIntegerLiteral(FunctionContext& context, SemIR::NodeId node_id,
                          SemIR::IntegerLiteral node) -> void {
  const llvm::APInt& i = context.semantics_ir().GetInteger(node.integer_id);
  // TODO: This won't offer correct semantics, but seems close enough for now.
  llvm::Value* v =
      llvm::ConstantInt::get(context.builder().getInt32Ty(), i.getZExtValue());
  context.SetLocal(node_id, v);
}

auto HandleNameReference(FunctionContext& context, SemIR::NodeId node_id,
                         SemIR::NameReference node) -> void {
  auto type_node_id =
      context.semantics_ir().GetTypeAllowBuiltinTypes(node.type_id);
  if (type_node_id == SemIR::NodeId::BuiltinNamespaceType) {
    return;
  }

  auto target = context.semantics_ir().GetNode(node.value_id);
  if (auto function_decl = target.TryAs<SemIR::FunctionDeclaration>()) {
    context.SetLocal(node_id, context.GetFunction(function_decl->function_id));
  } else {
    // TODO: Handle other kinds of name references to globals.
    context.SetLocal(node_id, context.GetLocal(node.value_id));
  }
}

auto HandleNamespace(FunctionContext& /*context*/, SemIR::NodeId /*node_id*/,
                     SemIR::Namespace node) -> void {
  CARBON_FATAL()
      << "Should not be encountered. If that changes, we may want to change "
         "higher-level logic to skip them rather than calling this. "
      << node;
}

auto HandleNoOp(FunctionContext& /*context*/, SemIR::NodeId /*node_id*/,
                SemIR::NoOp /*node*/) -> void {
  // No action to take.
}

auto HandleParameter(FunctionContext& /*context*/, SemIR::NodeId /*node_id*/,
                     SemIR::Parameter /*node*/) -> void {
  CARBON_FATAL() << "Parameters should be lowered by `BuildFunctionDefinition`";
}

auto HandleRealLiteral(FunctionContext& context, SemIR::NodeId node_id,
                       SemIR::RealLiteral node) -> void {
  const SemIR::Real& real = context.semantics_ir().GetReal(node.real_id);
  // TODO: This will probably have overflow issues, and should be fixed.
  double val =
      real.mantissa.getZExtValue() *
      std::pow((real.is_decimal ? 10 : 2), real.exponent.getSExtValue());
  llvm::APFloat llvm_val(val);
  context.SetLocal(node_id, llvm::ConstantFP::get(
                                context.builder().getDoubleTy(), llvm_val));
}

auto HandleReturn(FunctionContext& context, SemIR::NodeId /*node_id*/,
                  SemIR::Return /*node*/) -> void {
  context.builder().CreateRetVoid();
}

auto HandleReturnExpression(FunctionContext& context, SemIR::NodeId /*node_id*/,
                            SemIR::ReturnExpression node) -> void {
  switch (SemIR::GetInitializingRepresentation(
              context.semantics_ir(),
              context.semantics_ir().GetNode(node.expr_id).type_id())
              .kind) {
    case SemIR::InitializingRepresentation::None:
    case SemIR::InitializingRepresentation::InPlace:
      // Nothing to return.
      context.builder().CreateRetVoid();
      return;
    case SemIR::InitializingRepresentation::ByCopy:
      // The expression produces the value representation for the type.
      context.builder().CreateRet(context.GetLocal(node.expr_id));
      return;
  }
}

auto HandleSpliceBlock(FunctionContext& context, SemIR::NodeId node_id,
                       SemIR::SpliceBlock node) -> void {
  context.LowerBlock(node.block_id);
  context.SetLocal(node_id, context.GetLocal(node.result_id));
}

auto HandleStringLiteral(FunctionContext& /*context*/,
                         SemIR::NodeId /*node_id*/, SemIR::StringLiteral node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

// Extracts an element of either a struct or a tuple by index. Depending on the
// expression category of the aggregate input, this will either produce a value
// or a reference.
static auto GetStructOrTupleElement(FunctionContext& context,
                                    SemIR::NodeId aggr_node_id, unsigned idx,
                                    SemIR::TypeId result_type_id,
                                    llvm::Twine name) -> llvm::Value* {
  auto aggr_node = context.semantics_ir().GetNode(aggr_node_id);
  auto* aggr_value = context.GetLocal(aggr_node_id);

  switch (SemIR::GetExpressionCategory(context.semantics_ir(), aggr_node_id)) {
    case SemIR::ExpressionCategory::Error:
    case SemIR::ExpressionCategory::NotExpression:
    case SemIR::ExpressionCategory::Initializing:
    case SemIR::ExpressionCategory::Mixed:
      CARBON_FATAL() << "Unexpected expression category for aggregate access";

    case SemIR::ExpressionCategory::Value: {
      auto value_rep = SemIR::GetValueRepresentation(context.semantics_ir(),
                                                     aggr_node.type_id());
      switch (value_rep.kind) {
        case SemIR::ValueRepresentation::Unknown:
          CARBON_FATAL() << "Lowering access to incomplete aggregate type";
        case SemIR::ValueRepresentation::None:
          return aggr_value;
        case SemIR::ValueRepresentation::Copy:
          // We are holding the values of the aggregate directly, elementwise.
          return context.builder().CreateExtractValue(aggr_value, idx, name);
        case SemIR::ValueRepresentation::Pointer: {
          // The value representation is a pointer to an aggregate that we want
          // to index into.
          auto pointee_type_id =
              context.semantics_ir().GetPointeeType(value_rep.type_id);
          auto* value_type = context.GetType(pointee_type_id);
          auto* elem_ptr = context.builder().CreateStructGEP(
              value_type, aggr_value, idx, name);
          auto result_value_type_id =
              SemIR::GetValueRepresentation(context.semantics_ir(),
                                            result_type_id)
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
      auto* aggr_type = context.GetType(aggr_node.type_id());
      return context.builder().CreateStructGEP(aggr_type, aggr_value, idx,
                                               name);
    }
  }
}

auto HandleStructAccess(FunctionContext& context, SemIR::NodeId node_id,
                        SemIR::StructAccess node) -> void {
  auto struct_type_id =
      context.semantics_ir().GetNode(node.struct_id).type_id();

  // Get type information for member names.
  auto fields = context.semantics_ir().GetNodeBlock(
      context.semantics_ir()
          .GetNodeAs<SemIR::StructType>(
              context.semantics_ir().GetType(struct_type_id))
          .fields_id);
  auto field = context.semantics_ir().GetNodeAs<SemIR::StructTypeField>(
      fields[node.index.index]);
  auto member_name = context.semantics_ir().GetString(field.name_id);

  context.SetLocal(node_id, GetStructOrTupleElement(context, node.struct_id,
                                                    node.index.index,
                                                    node.type_id, member_name));
}

auto HandleStructLiteral(FunctionContext& context, SemIR::NodeId node_id,
                         SemIR::StructLiteral node) -> void {
  // A StructLiteral should always be converted to a StructInit or StructValue
  // if its value is needed.
  context.SetLocal(node_id,
                   llvm::PoisonValue::get(context.GetType(node.type_id)));
}

// Emits the value representation for a struct or tuple whose elements are the
// contents of `refs_id`.
auto EmitStructOrTupleValueRepresentation(FunctionContext& context,
                                          SemIR::TypeId type_id,
                                          SemIR::NodeBlockId refs_id,
                                          llvm::Twine name) -> llvm::Value* {
  auto value_rep =
      SemIR::GetValueRepresentation(context.semantics_ir(), type_id);
  switch (value_rep.kind) {
    case SemIR::ValueRepresentation::Unknown:
      CARBON_FATAL() << "Incomplete aggregate type in lowering";

    case SemIR::ValueRepresentation::None:
      // TODO: Add a helper to get a "no value representation" value.
      return llvm::PoisonValue::get(context.GetType(value_rep.type_id));

    case SemIR::ValueRepresentation::Copy: {
      auto refs = context.semantics_ir().GetNodeBlock(refs_id);
      CARBON_CHECK(refs.size() == 1)
          << "Unexpected size for aggregate with by-copy value representation";
      // TODO: Remove the LLVM StructType wrapper in this case, so we don't
      // need this `insert_value` wrapping.
      return context.builder().CreateInsertValue(
          llvm::PoisonValue::get(context.GetType(value_rep.type_id)),
          context.GetLocal(refs[0]), {0});
    }

    case SemIR::ValueRepresentation::Pointer: {
      auto pointee_type_id =
          context.semantics_ir().GetPointeeType(value_rep.type_id);
      auto* llvm_value_rep_type = context.GetType(pointee_type_id);

      // Write the value representation to a local alloca so we can produce a
      // pointer to it as the value representation of the struct or tuple.
      auto* alloca =
          context.builder().CreateAlloca(llvm_value_rep_type,
                                         /*ArraySize=*/nullptr, name);
      for (auto [i, ref] :
           llvm::enumerate(context.semantics_ir().GetNodeBlock(refs_id))) {
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

auto HandleStructInit(FunctionContext& context, SemIR::NodeId node_id,
                      SemIR::StructInit node) -> void {
  auto* llvm_type = context.GetType(node.type_id);

  switch (
      SemIR::GetInitializingRepresentation(context.semantics_ir(), node.type_id)
          .kind) {
    case SemIR::InitializingRepresentation::None:
    case SemIR::InitializingRepresentation::InPlace:
      // TODO: Add a helper to poison a value slot.
      context.SetLocal(node_id, llvm::PoisonValue::get(llvm_type));
      break;

    case SemIR::InitializingRepresentation::ByCopy: {
      context.SetLocal(
          node_id, EmitStructOrTupleValueRepresentation(
                       context, node.type_id, node.elements_id, "struct.init"));
      break;
    }
  }
}

auto HandleStructValue(FunctionContext& context, SemIR::NodeId node_id,
                       SemIR::StructValue node) -> void {
  context.SetLocal(node_id,
                   EmitStructOrTupleValueRepresentation(
                       context, node.type_id, node.elements_id, "struct"));
}

auto HandleStructTypeField(FunctionContext& /*context*/,
                           SemIR::NodeId /*node_id*/,
                           SemIR::StructTypeField /*node*/) -> void {
  // No action to take.
}

auto HandleTupleAccess(FunctionContext& context, SemIR::NodeId node_id,
                       SemIR::TupleAccess node) -> void {
  context.SetLocal(
      node_id, GetStructOrTupleElement(context, node.tuple_id, node.index.index,
                                       node.type_id, "tuple.elem"));
}

auto HandleTupleIndex(FunctionContext& context, SemIR::NodeId node_id,
                      SemIR::TupleIndex node) -> void {
  auto index_node =
      context.semantics_ir().GetNodeAs<SemIR::IntegerLiteral>(node.index_id);
  auto index =
      context.semantics_ir().GetInteger(index_node.integer_id).getZExtValue();
  context.SetLocal(node_id,
                   GetStructOrTupleElement(context, node.tuple_id, index,
                                           node.type_id, "tuple.index"));
}

auto HandleTupleLiteral(FunctionContext& context, SemIR::NodeId node_id,
                        SemIR::TupleLiteral node) -> void {
  // A TupleLiteral should always be converted to a TupleInit or TupleValue if
  // its value is needed.
  context.SetLocal(node_id,
                   llvm::PoisonValue::get(context.GetType(node.type_id)));
}

auto HandleTupleInit(FunctionContext& context, SemIR::NodeId node_id,
                     SemIR::TupleInit node) -> void {
  auto* llvm_type = context.GetType(node.type_id);

  switch (
      SemIR::GetInitializingRepresentation(context.semantics_ir(), node.type_id)
          .kind) {
    case SemIR::InitializingRepresentation::None:
    case SemIR::InitializingRepresentation::InPlace:
      // TODO: Add a helper to poison a value slot.
      context.SetLocal(node_id, llvm::PoisonValue::get(llvm_type));
      break;

    case SemIR::InitializingRepresentation::ByCopy: {
      context.SetLocal(
          node_id, EmitStructOrTupleValueRepresentation(
                       context, node.type_id, node.elements_id, "tuple.init"));
      break;
    }
  }
}

auto HandleTupleValue(FunctionContext& context, SemIR::NodeId node_id,
                      SemIR::TupleValue node) -> void {
  context.SetLocal(
      node_id, EmitStructOrTupleValueRepresentation(context, node.type_id,
                                                    node.elements_id, "tuple"));
}

auto HandleUnaryOperatorNot(FunctionContext& context, SemIR::NodeId node_id,
                            SemIR::UnaryOperatorNot node) -> void {
  context.SetLocal(
      node_id, context.builder().CreateNot(context.GetLocal(node.operand_id)));
}

auto HandleVarStorage(FunctionContext& context, SemIR::NodeId node_id,
                      SemIR::VarStorage node) -> void {
  // TODO: Eventually this name will be optional, and we'll want to provide
  // something like `var` as a default. However, that's not possible right now
  // so cannot be tested.
  auto name = context.semantics_ir().GetString(node.name_id);
  auto* alloca = context.builder().CreateAlloca(context.GetType(node.type_id),
                                                /*ArraySize=*/nullptr, name);
  context.SetLocal(node_id, alloca);
}

}  // namespace Carbon::Lower
