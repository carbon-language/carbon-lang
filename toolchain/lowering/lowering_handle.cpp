// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "toolchain/lowering/lowering_function_context.h"
#include "toolchain/semantics/semantics_node_kind.h"

namespace Carbon {

auto LoweringHandleInvalid(LoweringFunctionContext& /*context*/,
                           SemIR::NodeId /*node_id*/, SemIR::Node /*node*/)
    -> void {
  llvm_unreachable("never in actual IR");
}

auto LoweringHandleCrossReference(LoweringFunctionContext& /*context*/,
                                  SemIR::NodeId /*node_id*/, SemIR::Node node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleAddressOf(LoweringFunctionContext& context,
                             SemIR::NodeId node_id, SemIR::Node node) -> void {
  context.SetLocal(node_id, context.GetLocal(node.GetAsAddressOf()));
}

auto LoweringHandleArrayIndex(LoweringFunctionContext& context,
                              SemIR::NodeId node_id, SemIR::Node node) -> void {
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
    auto* index = context.builder().CreateLoad(llvm_type->getArrayElementType(),
                                               context.GetLocal(index_node_id));
    // TODO: Handle return value or call such as `F()[a]`.
    array_element_value = context.builder().CreateInBoundsGEP(
        llvm_type, array_value, index, "array.index");
  }
  context.SetLocal(node_id, array_element_value);
}

auto LoweringHandleArrayValue(LoweringFunctionContext& context,
                              SemIR::NodeId node_id, SemIR::Node node) -> void {
  auto* llvm_type = context.GetType(node.type_id());
  auto* alloca =
      context.builder().CreateAlloca(llvm_type, /*ArraySize=*/nullptr, "array");
  context.SetLocal(node_id, alloca);
  auto tuple_node_id = node.GetAsArrayValue();
  auto* tuple_value = context.GetLocal(tuple_node_id);
  auto* tuple_type =
      context.GetType(context.semantics_ir().GetNode(tuple_node_id).type_id());

  for (auto i : llvm::seq(llvm_type->getArrayNumElements())) {
    llvm::Value* array_element_value = context.GetIndexFromStructOrArray(
        tuple_type, tuple_value, i, "array.element");
    if (tuple_value->getType()->isPointerTy()) {
      array_element_value = context.builder().CreateLoad(
          llvm_type->getArrayElementType(), array_element_value);
    }
    // Initializing the array with values.
    context.builder().CreateStore(
        array_element_value,
        context.builder().CreateStructGEP(llvm_type, alloca, i));
  }
}

auto LoweringHandleAssign(LoweringFunctionContext& context,
                          SemIR::NodeId /*node_id*/, SemIR::Node node) -> void {
  auto [storage_id, value_id] = node.GetAsAssign();
  auto storage_type_id = context.semantics_ir().GetNode(storage_id).type_id();

  // We can assign from either a value expression or a by-copy initializing
  // expression. In either case, the value of the source is the value
  // representation of the target.
  // TODO: Should we use different semantic nodes for those operations?
  switch (auto rep = SemIR::GetValueRepresentation(context.semantics_ir(),
                                                   storage_type_id);
          rep.kind) {
    case SemIR::ValueRepresentation::None:
      break;
    case SemIR::ValueRepresentation::Copy:
      context.builder().CreateStore(context.GetLocalLoaded(value_id),
                                    context.GetLocal(storage_id));
      break;
    case SemIR::ValueRepresentation::Pointer: {
      auto& layout = context.llvm_module().getDataLayout();
      auto* type = context.GetType(storage_type_id);
      // TODO: Compute known alignment of the source and destination, which may
      // be greater than the alignment computed by LLVM.
      auto align = layout.getABITypeAlign(type);

      // TODO: Attach !tbaa.struct metadata indicating which portions of the
      // type we actually need to copy and which are padding.
      context.builder().CreateMemCpy(context.GetLocal(storage_id), align,
                                     context.GetLocal(value_id), align,
                                     layout.getTypeAllocSize(type));
      break;
    }
    case SemIR::ValueRepresentation::Custom:
      CARBON_FATAL() << "TODO: Add support for Assign with custom value rep";
  }
}

auto LoweringHandleBinaryOperatorAdd(LoweringFunctionContext& /*context*/,
                                     SemIR::NodeId /*node_id*/,
                                     SemIR::Node node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleBindName(LoweringFunctionContext& /*context*/,
                            SemIR::NodeId /*node_id*/, SemIR::Node /*node*/)
    -> void {
  // Probably need to do something here, but not necessary for now.
}

auto LoweringHandleBlockArg(LoweringFunctionContext& context,
                            SemIR::NodeId node_id, SemIR::Node node) -> void {
  SemIR::NodeBlockId block_id = node.GetAsBlockArg();
  context.SetLocal(node_id, context.GetBlockArg(block_id, node.type_id()));
}

auto LoweringHandleBoolLiteral(LoweringFunctionContext& context,
                               SemIR::NodeId node_id, SemIR::Node node)
    -> void {
  llvm::Value* v = llvm::ConstantInt::get(context.builder().getInt1Ty(),
                                          node.GetAsBoolLiteral().index);
  context.SetLocal(node_id, v);
}

auto LoweringHandleBranch(LoweringFunctionContext& context,
                          SemIR::NodeId /*node_id*/, SemIR::Node node) -> void {
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

auto LoweringHandleBranchIf(LoweringFunctionContext& context,
                            SemIR::NodeId /*node_id*/, SemIR::Node node)
    -> void {
  auto [target_block_id, cond_id] = node.GetAsBranchIf();
  llvm::Value* cond = context.GetLocalLoaded(cond_id);
  llvm::BasicBlock* then_block = context.GetBlock(target_block_id);
  llvm::BasicBlock* else_block = context.CreateSyntheticBlock();
  context.builder().CreateCondBr(cond, then_block, else_block);
  context.builder().SetInsertPoint(else_block);
}

auto LoweringHandleBranchWithArg(LoweringFunctionContext& context,
                                 SemIR::NodeId /*node_id*/, SemIR::Node node)
    -> void {
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

auto LoweringHandleBuiltin(LoweringFunctionContext& /*context*/,
                           SemIR::NodeId /*node_id*/, SemIR::Node node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleCall(LoweringFunctionContext& context, SemIR::NodeId node_id,
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

auto LoweringHandleDereference(LoweringFunctionContext& context,
                               SemIR::NodeId node_id, SemIR::Node node)
    -> void {
  context.SetLocal(node_id, context.GetLocal(node.GetAsDereference()));
}

auto LoweringHandleFunctionDeclaration(LoweringFunctionContext& /*context*/,
                                       SemIR::NodeId /*node_id*/,
                                       SemIR::Node node) -> void {
  CARBON_FATAL()
      << "Should not be encountered. If that changes, we may want to change "
         "higher-level logic to skip them rather than calling this. "
      << node;
}

auto LoweringHandleIntegerLiteral(LoweringFunctionContext& context,
                                  SemIR::NodeId node_id, SemIR::Node node)
    -> void {
  llvm::APInt i =
      context.semantics_ir().GetIntegerLiteral(node.GetAsIntegerLiteral());
  // TODO: This won't offer correct semantics, but seems close enough for now.
  llvm::Value* v =
      llvm::ConstantInt::get(context.builder().getInt32Ty(), i.getSExtValue());
  context.SetLocal(node_id, v);
}

auto LoweringHandleNamespace(LoweringFunctionContext& /*context*/,
                             SemIR::NodeId /*node_id*/, SemIR::Node /*node*/)
    -> void {
  // No action to take.
}

auto LoweringHandleNoOp(LoweringFunctionContext& /*context*/,
                        SemIR::NodeId /*node_id*/, SemIR::Node /*node*/)
    -> void {
  // No action to take.
}

auto LoweringHandleParameter(LoweringFunctionContext& /*context*/,
                             SemIR::NodeId /*node_id*/, SemIR::Node /*node*/)
    -> void {
  CARBON_FATAL() << "Parameters should be lowered by `BuildFunctionDefinition`";
}

auto LoweringHandleRealLiteral(LoweringFunctionContext& context,
                               SemIR::NodeId node_id, SemIR::Node node)
    -> void {
  SemIR::RealLiteral real =
      context.semantics_ir().GetRealLiteral(node.GetAsRealLiteral());
  // TODO: This will probably have overflow issues, and should be fixed.
  double val =
      real.mantissa.getSExtValue() *
      std::pow((real.is_decimal ? 10 : 2), real.exponent.getSExtValue());
  llvm::APFloat llvm_val(val);
  context.SetLocal(node_id, llvm::ConstantFP::get(
                                context.builder().getDoubleTy(), llvm_val));
}

auto LoweringHandleReturn(LoweringFunctionContext& context,
                          SemIR::NodeId /*node_id*/, SemIR::Node /*node*/)
    -> void {
  context.builder().CreateRetVoid();
}

auto LoweringHandleReturnExpression(LoweringFunctionContext& context,
                                    SemIR::NodeId /*node_id*/, SemIR::Node node)
    -> void {
  SemIR::NodeId expr_id = node.GetAsReturnExpression();
  context.builder().CreateRet(context.GetLocalLoaded(expr_id));
}

auto LoweringHandleStringLiteral(LoweringFunctionContext& /*context*/,
                                 SemIR::NodeId /*node_id*/, SemIR::Node node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleStructAccess(LoweringFunctionContext& context,
                                SemIR::NodeId node_id, SemIR::Node node)
    -> void {
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

auto LoweringHandleTupleIndex(LoweringFunctionContext& context,
                              SemIR::NodeId node_id, SemIR::Node node) -> void {
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

auto LoweringHandleTupleValue(LoweringFunctionContext& context,
                              SemIR::NodeId node_id, SemIR::Node node) -> void {
  auto* llvm_type = context.GetType(node.type_id());
  auto* alloca =
      context.builder().CreateAlloca(llvm_type, /*ArraySize=*/nullptr, "tuple");
  context.SetLocal(node_id, alloca);
  auto refs = context.semantics_ir().GetNodeBlock(node.GetAsTupleValue());
  for (auto [i, ref] : llvm::enumerate(refs)) {
    auto* gep = context.builder().CreateStructGEP(llvm_type, alloca, i);
    context.builder().CreateStore(context.GetLocal(ref), gep);
  }
}

auto LoweringHandleStructTypeField(LoweringFunctionContext& /*context*/,
                                   SemIR::NodeId /*node_id*/,
                                   SemIR::Node /*node*/) -> void {
  // No action to take.
}

auto LoweringHandleStructValue(LoweringFunctionContext& context,
                               SemIR::NodeId node_id, SemIR::Node node)
    -> void {
  auto* llvm_type = context.GetType(node.type_id());
  auto* alloca = context.builder().CreateAlloca(
      llvm_type, /*ArraySize=*/nullptr, "struct");
  context.SetLocal(node_id, alloca);

  auto refs = context.semantics_ir().GetNodeBlock(node.GetAsStructValue());
  // Get type information for member names.
  auto type_refs = context.semantics_ir().GetNodeBlock(
      context.semantics_ir()
          .GetNode(context.semantics_ir().GetType(node.type_id()))
          .GetAsStructType());
  for (auto [i, ref, type_ref] : llvm::enumerate(refs, type_refs)) {
    auto [field_name_id, field_type_id] =
        context.semantics_ir().GetNode(type_ref).GetAsStructTypeField();
    auto member_name = context.semantics_ir().GetString(field_name_id);
    auto* gep =
        context.builder().CreateStructGEP(llvm_type, alloca, i, member_name);
    context.builder().CreateStore(context.GetLocal(ref), gep);
  }
}

auto LoweringHandleStubReference(LoweringFunctionContext& context,
                                 SemIR::NodeId node_id, SemIR::Node node)
    -> void {
  context.SetLocal(node_id, context.GetLocal(node.GetAsStubReference()));
}

auto LoweringHandleUnaryOperatorNot(LoweringFunctionContext& context,
                                    SemIR::NodeId node_id, SemIR::Node node)
    -> void {
  context.SetLocal(node_id, context.builder().CreateNot(context.GetLocal(
                                node.GetAsUnaryOperatorNot())));
}

auto LoweringHandleVarStorage(LoweringFunctionContext& context,
                              SemIR::NodeId node_id, SemIR::Node node) -> void {
  // TODO: Eventually this name will be optional, and we'll want to provide
  // something like `var` as a default. However, that's not possible right now
  // so cannot be tested.
  auto name = context.semantics_ir().GetString(node.GetAsVarStorage());
  auto* alloca = context.builder().CreateAlloca(context.GetType(node.type_id()),
                                                /*ArraySize=*/nullptr, name);
  context.SetLocal(node_id, alloca);
}

}  // namespace Carbon
