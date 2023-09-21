// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "toolchain/lower/function_context.h"
#include "toolchain/sem_ir/node_kind.h"

namespace Carbon::Lower {

auto HandleInvalid(FunctionContext& /*context*/, SemIR::NodeId /*node_id*/,
                   SemIR::Node /*node*/) -> void {
  llvm_unreachable("never in actual IR");
}

auto HandleCrossReference(FunctionContext& /*context*/,
                          SemIR::NodeId /*node_id*/, SemIR::Node node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto HandleAddressOf(FunctionContext& context, SemIR::NodeId node_id,
                     SemIR::Node node) -> void {
  context.SetLocal(node_id, context.GetLocal(node.GetAsAddressOf()));
}

auto HandleArrayIndex(FunctionContext& context, SemIR::NodeId node_id,
                      SemIR::Node node) -> void {
  auto [array_node_id, index_node_id] = node.GetAsArrayIndex();
  auto* array_value = context.GetLocal(array_node_id);
  auto* llvm_type =
      context.GetType(context.semantics_ir().GetNode(array_node_id).type_id());
  auto index_node = context.semantics_ir().GetNode(index_node_id);
  llvm::Value* array_element_value;

  if (index_node.kind() == SemIR::NodeKind::IntegerLiteral) {
    const auto index = context.semantics_ir()
                           .GetIntegerLiteral(index_node.GetAsIntegerLiteral())
                           .getZExtValue();
    array_element_value = context.GetIndexFromStructOrArray(
        llvm_type, array_value, index, "array.index");
  } else {
    auto* index = context.GetLocalLoaded(index_node_id);
    // TODO: Handle return value or call such as `F()[a]`.
    auto* zero = llvm::ConstantInt::get(
        llvm::Type::getInt32Ty(context.llvm_context()), 0);
    array_element_value = context.builder().CreateInBoundsGEP(
        llvm_type, array_value, {zero, index}, "array.index");
  }
  context.SetLocal(node_id, array_element_value);
}

auto HandleArrayInit(FunctionContext& context, SemIR::NodeId node_id,
                     SemIR::Node node) -> void {
  auto [src_id, refs_id] = node.GetAsArrayInit();
  // The result of initialization is the return slot of the initializer.
  context.SetLocal(
      node_id,
      context.GetLocal(context.semantics_ir().GetNodeBlock(refs_id).back()));
}

auto HandleAssign(FunctionContext& context, SemIR::NodeId /*node_id*/,
                  SemIR::Node node) -> void {
  auto [storage_id, value_id] = node.GetAsAssign();
  auto storage_type_id = context.semantics_ir().GetNode(storage_id).type_id();
  context.FinishInitialization(storage_type_id, storage_id, value_id);
}

auto HandleBinaryOperatorAdd(FunctionContext& /*context*/,
                             SemIR::NodeId /*node_id*/, SemIR::Node node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto HandleBindName(FunctionContext& /*context*/, SemIR::NodeId /*node_id*/,
                    SemIR::Node /*node*/) -> void {
  // Probably need to do something here, but not necessary for now.
}

auto HandleBlockArg(FunctionContext& context, SemIR::NodeId node_id,
                    SemIR::Node node) -> void {
  SemIR::NodeBlockId block_id = node.GetAsBlockArg();
  context.SetLocal(node_id, context.GetBlockArg(block_id, node.type_id()));
}

auto HandleBoolLiteral(FunctionContext& context, SemIR::NodeId node_id,
                       SemIR::Node node) -> void {
  llvm::Value* v = llvm::ConstantInt::get(context.builder().getInt1Ty(),
                                          node.GetAsBoolLiteral().index);
  context.SetLocal(node_id, v);
}

auto HandleBranch(FunctionContext& context, SemIR::NodeId /*node_id*/,
                  SemIR::Node node) -> void {
  SemIR::NodeBlockId target_block_id = node.GetAsBranch();

  // Opportunistically avoid creating a BasicBlock that contains just a branch.
  llvm::BasicBlock* block = context.builder().GetInsertBlock();
  if (block->empty() && context.TryToReuseBlock(target_block_id, block)) {
    // Reuse this block as the branch target.
  } else {
    context.builder().CreateBr(context.GetBlock(target_block_id));
  }

  context.builder().ClearInsertionPoint();
}

auto HandleBranchIf(FunctionContext& context, SemIR::NodeId /*node_id*/,
                    SemIR::Node node) -> void {
  auto [target_block_id, cond_id] = node.GetAsBranchIf();
  llvm::Value* cond = context.GetLocalLoaded(cond_id);
  llvm::BasicBlock* then_block = context.GetBlock(target_block_id);
  llvm::BasicBlock* else_block = context.CreateSyntheticBlock();
  context.builder().CreateCondBr(cond, then_block, else_block);
  context.builder().SetInsertPoint(else_block);
}

auto HandleBranchWithArg(FunctionContext& context, SemIR::NodeId /*node_id*/,
                         SemIR::Node node) -> void {
  auto [target_block_id, arg_id] = node.GetAsBranchWithArg();
  llvm::Value* arg = context.GetLocalLoaded(arg_id);
  SemIR::TypeId arg_type_id = context.semantics_ir().GetNode(arg_id).type_id();

  // Opportunistically avoid creating a BasicBlock that contains just a branch.
  // We only do this for a block that we know will only have a single
  // predecessor, so that we can correctly populate the predecessors of the
  // PHINode.
  llvm::BasicBlock* block = context.builder().GetInsertBlock();
  llvm::BasicBlock* phi_predecessor = block;
  if (block->empty() && context.IsCurrentSyntheticBlock(block) &&
      context.TryToReuseBlock(target_block_id, block)) {
    // Reuse this block as the branch target.
    phi_predecessor = block->getSinglePredecessor();
    CARBON_CHECK(phi_predecessor)
        << "Synthetic block did not have a single predecessor";
  } else {
    context.builder().CreateBr(context.GetBlock(target_block_id));
  }

  context.GetBlockArg(target_block_id, arg_type_id)
      ->addIncoming(arg, phi_predecessor);
  context.builder().ClearInsertionPoint();
}

auto HandleBuiltin(FunctionContext& /*context*/, SemIR::NodeId /*node_id*/,
                   SemIR::Node node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto HandleCall(FunctionContext& context, SemIR::NodeId node_id,
                SemIR::Node node) -> void {
  auto [refs_id, function_id] = node.GetAsCall();
  auto* llvm_function = context.GetFunction(function_id);
  const auto& function = context.semantics_ir().GetFunction(function_id);

  std::vector<llvm::Value*> args;
  llvm::ArrayRef<SemIR::NodeId> arg_ids =
      context.semantics_ir().GetNodeBlock(refs_id);

  if (function.return_slot_id.is_valid()) {
    args.push_back(context.GetLocal(arg_ids.back()));
    arg_ids = arg_ids.drop_back();
  }

  for (auto ref_id : arg_ids) {
    auto arg_type_id = context.semantics_ir().GetNode(ref_id).type_id();
    switch (SemIR::GetValueRepresentation(context.semantics_ir(), arg_type_id)
                .kind) {
      case SemIR::ValueRepresentation::None:
        break;
      case SemIR::ValueRepresentation::Copy:
      case SemIR::ValueRepresentation::Custom:
        args.push_back(context.GetLocalLoaded(ref_id));
        break;
      case SemIR::ValueRepresentation::Pointer:
        args.push_back(context.GetLocal(ref_id));
        break;
    }
  }

  if (llvm_function->getReturnType()->isVoidTy()) {
    context.builder().CreateCall(llvm_function, args);
    // The value of a function call with a void return type shouldn't used, but
    // StubReference needs a value to propagate.
    context.SetLocal(node_id,
                     llvm::PoisonValue::get(context.GetType(node.type_id())));
  } else {
    context.SetLocal(node_id,
                     context.builder().CreateCall(llvm_function, args,
                                                  llvm_function->getName()));
  }
}

auto HandleDereference(FunctionContext& context, SemIR::NodeId node_id,
                       SemIR::Node node) -> void {
  context.SetLocal(node_id, context.GetLocal(node.GetAsDereference()));
}

auto HandleFunctionDeclaration(FunctionContext& /*context*/,
                               SemIR::NodeId /*node_id*/, SemIR::Node node)
    -> void {
  CARBON_FATAL()
      << "Should not be encountered. If that changes, we may want to change "
         "higher-level logic to skip them rather than calling this. "
      << node;
}

auto HandleInitializeFrom(FunctionContext& context, SemIR::NodeId /*node_id*/,
                          SemIR::Node node) -> void {
  auto [init_id, storage_id] = node.GetAsInitializeFrom();
  auto storage_type_id = context.semantics_ir().GetNode(storage_id).type_id();
  context.FinishInitialization(storage_type_id, storage_id, init_id);
}

auto HandleIntegerLiteral(FunctionContext& context, SemIR::NodeId node_id,
                          SemIR::Node node) -> void {
  llvm::APInt i =
      context.semantics_ir().GetIntegerLiteral(node.GetAsIntegerLiteral());
  // TODO: This won't offer correct semantics, but seems close enough for now.
  llvm::Value* v =
      llvm::ConstantInt::get(context.builder().getInt32Ty(), i.getZExtValue());
  context.SetLocal(node_id, v);
}

auto HandleNamespace(FunctionContext& /*context*/, SemIR::NodeId /*node_id*/,
                     SemIR::Node /*node*/) -> void {
  // No action to take.
}

auto HandleNoOp(FunctionContext& /*context*/, SemIR::NodeId /*node_id*/,
                SemIR::Node /*node*/) -> void {
  // No action to take.
}

auto HandleParameter(FunctionContext& /*context*/, SemIR::NodeId /*node_id*/,
                     SemIR::Node /*node*/) -> void {
  CARBON_FATAL() << "Parameters should be lowered by `BuildFunctionDefinition`";
}

auto HandleRealLiteral(FunctionContext& context, SemIR::NodeId node_id,
                       SemIR::Node node) -> void {
  SemIR::RealLiteral real =
      context.semantics_ir().GetRealLiteral(node.GetAsRealLiteral());
  // TODO: This will probably have overflow issues, and should be fixed.
  double val =
      real.mantissa.getZExtValue() *
      std::pow((real.is_decimal ? 10 : 2), real.exponent.getSExtValue());
  llvm::APFloat llvm_val(val);
  context.SetLocal(node_id, llvm::ConstantFP::get(
                                context.builder().getDoubleTy(), llvm_val));
}

auto HandleReturn(FunctionContext& context, SemIR::NodeId /*node_id*/,
                  SemIR::Node /*node*/) -> void {
  context.builder().CreateRetVoid();
}

auto HandleReturnExpression(FunctionContext& context, SemIR::NodeId /*node_id*/,
                            SemIR::Node node) -> void {
  SemIR::NodeId expr_id = node.GetAsReturnExpression();
  switch (SemIR::GetInitializingRepresentation(
              context.semantics_ir(),
              context.semantics_ir().GetNode(expr_id).type_id())
              .kind) {
    case SemIR::InitializingRepresentation::None:
    case SemIR::InitializingRepresentation::InPlace:
      // Nothing to return.
      context.builder().CreateRetVoid();
      return;
    case SemIR::InitializingRepresentation::ByCopy:
      // The expression produces the value representation for the type.
      context.builder().CreateRet(context.GetLocalLoaded(expr_id));
      return;
  }
}

auto HandleStringLiteral(FunctionContext& /*context*/,
                         SemIR::NodeId /*node_id*/, SemIR::Node node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto HandleStructAccess(FunctionContext& context, SemIR::NodeId node_id,
                        SemIR::Node node) -> void {
  auto [struct_id, member_index] = node.GetAsStructAccess();
  auto struct_type_id = context.semantics_ir().GetNode(struct_id).type_id();
  auto* llvm_type = context.GetType(struct_type_id);

  // Get type information for member names.
  auto type_refs = context.semantics_ir().GetNodeBlock(
      context.semantics_ir()
          .GetNode(context.semantics_ir().GetType(struct_type_id))
          .GetAsStructType());
  auto [field_name_id, field_type_id] =
      context.semantics_ir()
          .GetNode(type_refs[member_index.index])
          .GetAsStructTypeField();
  auto member_name = context.semantics_ir().GetString(field_name_id);

  auto* gep = context.builder().CreateStructGEP(
      llvm_type, context.GetLocal(struct_id), member_index.index, member_name);
  context.SetLocal(node_id, gep);
}

auto HandleStructLiteral(FunctionContext& context, SemIR::NodeId node_id,
                         SemIR::Node node) -> void {
  // A StructLiteral should always be converted to a StructInit or StructValue
  // if its value is needed.
  context.SetLocal(node_id,
                   llvm::PoisonValue::get(context.GetType(node.type_id())));
}

// Emits the value representation for a struct or tuple whose elements are the
// contents of `refs_id`.
auto EmitStructOrTupleValueRepresentation(FunctionContext& context,
                                          SemIR::TypeId type_id,
                                          SemIR::NodeBlockId refs_id,
                                          llvm::Twine name) -> llvm::Value* {
  auto* llvm_type = context.GetType(type_id);

  switch (SemIR::GetValueRepresentation(context.semantics_ir(), type_id).kind) {
    case SemIR::ValueRepresentation::None:
      // TODO: Add a helper to get a "no value representation" value.
      return llvm::PoisonValue::get(llvm_type);

    case SemIR::ValueRepresentation::Copy: {
      auto refs = context.semantics_ir().GetNodeBlock(refs_id);
      CARBON_CHECK(refs.size() == 1)
          << "Unexpected size for aggregate with by-copy value representation";
      // TODO: Remove the LLVM StructType wrapper in this case, so we don't
      // need this `insert_value` wrapping.
      return context.builder().CreateInsertValue(
          llvm::PoisonValue::get(llvm_type), context.GetLocal(refs[0]), {0});
    }

    case SemIR::ValueRepresentation::Pointer: {
      // Write the object representation to a local alloca so we can produce a
      // pointer to it as the value representation.
      auto* alloca = context.builder().CreateAlloca(
          llvm_type, /*ArraySize=*/nullptr, name);
      for (auto [i, ref] :
           llvm::enumerate(context.semantics_ir().GetNodeBlock(refs_id))) {
        auto* gep = context.builder().CreateStructGEP(llvm_type, alloca, i);
        // TODO: We are loading a value representation here and storing an
        // object representation!
        context.builder().CreateStore(context.GetLocal(ref), gep);
      }
      return alloca;
    }

    case SemIR::ValueRepresentation::Custom:
      CARBON_FATAL()
          << "Aggregate should never have custom value representation";
  }
}

auto HandleStructInit(FunctionContext& context, SemIR::NodeId node_id,
                      SemIR::Node node) -> void {
  auto* llvm_type = context.GetType(node.type_id());

  switch (SemIR::GetInitializingRepresentation(context.semantics_ir(),
                                               node.type_id())
              .kind) {
    case SemIR::InitializingRepresentation::None:
    case SemIR::InitializingRepresentation::InPlace:
      // TODO: Add a helper to poison a value slot.
      context.SetLocal(node_id, llvm::PoisonValue::get(llvm_type));
      break;

    case SemIR::InitializingRepresentation::ByCopy: {
      auto [struct_literal_id, refs_id] = node.GetAsStructInit();
      context.SetLocal(node_id,
                       EmitStructOrTupleValueRepresentation(
                           context, node.type_id(), refs_id, "struct.init"));
      break;
    }
  }
}

auto HandleStructValue(FunctionContext& context, SemIR::NodeId node_id,
                       SemIR::Node node) -> void {
  auto [struct_literal_id, refs_id] = node.GetAsStructValue();
  context.SetLocal(node_id, EmitStructOrTupleValueRepresentation(
                                context, node.type_id(), refs_id, "struct"));
}

auto HandleStructTypeField(FunctionContext& /*context*/,
                           SemIR::NodeId /*node_id*/, SemIR::Node /*node*/)
    -> void {
  // No action to take.
}

auto HandleStubReference(FunctionContext& context, SemIR::NodeId node_id,
                         SemIR::Node node) -> void {
  context.SetLocal(node_id, context.GetLocal(node.GetAsStubReference()));
}

auto HandleTupleAccess(FunctionContext& context, SemIR::NodeId node_id,
                       SemIR::Node node) -> void {
  auto [tuple_node_id, index] = node.GetAsTupleAccess();
  auto* tuple_value = context.GetLocal(tuple_node_id);
  auto* llvm_type =
      context.GetType(context.semantics_ir().GetNode(tuple_node_id).type_id());
  context.SetLocal(
      node_id, context.GetIndexFromStructOrArray(llvm_type, tuple_value,
                                                 index.index, "tuple.elem"));
}

auto HandleTupleIndex(FunctionContext& context, SemIR::NodeId node_id,
                      SemIR::Node node) -> void {
  auto [tuple_node_id, index_node_id] = node.GetAsTupleIndex();
  auto* tuple_value = context.GetLocal(tuple_node_id);
  auto index_node = context.semantics_ir().GetNode(index_node_id);
  const auto index = context.semantics_ir()
                         .GetIntegerLiteral(index_node.GetAsIntegerLiteral())
                         .getZExtValue();
  auto* llvm_type =
      context.GetType(context.semantics_ir().GetNode(tuple_node_id).type_id());
  context.SetLocal(node_id, context.GetIndexFromStructOrArray(
                                llvm_type, tuple_value, index, "tuple.index"));
}

auto HandleTupleLiteral(FunctionContext& context, SemIR::NodeId node_id,
                        SemIR::Node node) -> void {
  // A TupleLiteral should always be converted to a TupleInit or TupleValue if
  // its value is needed.
  context.SetLocal(node_id,
                   llvm::PoisonValue::get(context.GetType(node.type_id())));
}

auto HandleTupleInit(FunctionContext& context, SemIR::NodeId node_id,
                     SemIR::Node node) -> void {
  auto* llvm_type = context.GetType(node.type_id());

  switch (SemIR::GetInitializingRepresentation(context.semantics_ir(),
                                               node.type_id())
              .kind) {
    case SemIR::InitializingRepresentation::None:
    case SemIR::InitializingRepresentation::InPlace:
      // TODO: Add a helper to poison a value slot.
      context.SetLocal(node_id, llvm::PoisonValue::get(llvm_type));
      break;

    case SemIR::InitializingRepresentation::ByCopy: {
      auto [struct_literal_id, refs_id] = node.GetAsTupleInit();
      context.SetLocal(
          node_id, EmitStructOrTupleValueRepresentation(context, node.type_id(),
                                                        refs_id, "tuple.init"));
      break;
    }
  }
}

auto HandleTupleValue(FunctionContext& context, SemIR::NodeId node_id,
                      SemIR::Node node) -> void {
  auto [struct_literal_id, refs_id] = node.GetAsTupleValue();
  context.SetLocal(node_id, EmitStructOrTupleValueRepresentation(
                                context, node.type_id(), refs_id, "tuple"));
}

auto HandleUnaryOperatorNot(FunctionContext& context, SemIR::NodeId node_id,
                            SemIR::Node node) -> void {
  context.SetLocal(node_id, context.builder().CreateNot(context.GetLocal(
                                node.GetAsUnaryOperatorNot())));
}

auto HandleVarStorage(FunctionContext& context, SemIR::NodeId node_id,
                      SemIR::Node node) -> void {
  // TODO: Eventually this name will be optional, and we'll want to provide
  // something like `var` as a default. However, that's not possible right now
  // so cannot be tested.
  auto name = context.semantics_ir().GetString(node.GetAsVarStorage());
  auto* alloca = context.builder().CreateAlloca(context.GetType(node.type_id()),
                                                /*ArraySize=*/nullptr, name);
  context.SetLocal(node_id, alloca);
}

}  // namespace Carbon::Lower
