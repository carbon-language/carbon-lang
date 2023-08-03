// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lowering/lowering_function_context.h"

namespace Carbon {

auto LoweringHandleInvalid(LoweringFunctionContext& /*context*/,
                           SemanticsNodeId /*node_id*/, SemanticsNode /*node*/)
    -> void {
  llvm_unreachable("never in actual IR");
}

auto LoweringHandleCrossReference(LoweringFunctionContext& /*context*/,
                                  SemanticsNodeId /*node_id*/,
                                  SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleAssign(LoweringFunctionContext& context,
                          SemanticsNodeId /*node_id*/, SemanticsNode node)
    -> void {
  auto [storage_id, value_id] = node.GetAsAssign();
  context.builder().CreateStore(context.GetLocalLoaded(value_id),
                                context.GetLocal(storage_id));
}

auto LoweringHandleBinaryOperatorAdd(LoweringFunctionContext& /*context*/,
                                     SemanticsNodeId /*node_id*/,
                                     SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleBindName(LoweringFunctionContext& /*context*/,
                            SemanticsNodeId /*node_id*/, SemanticsNode /*node*/)
    -> void {
  // Probably need to do something here, but not necessary for now.
}

auto LoweringHandleBlockArg(LoweringFunctionContext& context,
                            SemanticsNodeId node_id, SemanticsNode node)
    -> void {
  SemanticsNodeBlockId block_id = node.GetAsBlockArg();
  context.SetLocal(node_id, context.GetBlockArg(block_id, node.type_id()));
}

auto LoweringHandleBoolLiteral(LoweringFunctionContext& context,
                               SemanticsNodeId node_id, SemanticsNode node)
    -> void {
  llvm::Value* v = llvm::ConstantInt::get(context.builder().getInt1Ty(),
                                          node.GetAsBoolLiteral().index);
  context.SetLocal(node_id, v);
}

auto LoweringHandleBranch(LoweringFunctionContext& context,
                          SemanticsNodeId /*node_id*/, SemanticsNode node)
    -> void {
  SemanticsNodeBlockId target_block_id = node.GetAsBranch();

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
                            SemanticsNodeId /*node_id*/, SemanticsNode node)
    -> void {
  auto [target_block_id, cond_id] = node.GetAsBranchIf();
  llvm::Value* cond = context.GetLocalLoaded(cond_id);
  llvm::BasicBlock* then_block = context.GetBlock(target_block_id);
  llvm::BasicBlock* else_block = context.CreateSyntheticBlock();
  context.builder().CreateCondBr(cond, then_block, else_block);
  context.builder().SetInsertPoint(else_block);
}

auto LoweringHandleBranchWithArg(LoweringFunctionContext& context,
                                 SemanticsNodeId /*node_id*/,
                                 SemanticsNode node) -> void {
  auto [target_block_id, arg_id] = node.GetAsBranchWithArg();
  llvm::Value* arg = context.GetLocalLoaded(arg_id);
  SemanticsTypeId arg_type_id =
      context.semantics_ir().GetNode(arg_id).type_id();

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
                           SemanticsNodeId /*node_id*/, SemanticsNode node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleCall(LoweringFunctionContext& context,
                        SemanticsNodeId node_id, SemanticsNode node) -> void {
  auto [refs_id, function_id] = node.GetAsCall();
  auto* function = context.GetFunction(function_id);
  std::vector<llvm::Value*> args;
  for (auto ref_id : context.semantics_ir().GetNodeBlock(refs_id)) {
    args.push_back(context.GetLocalLoaded(ref_id));
  }
  if (function->getReturnType()->isVoidTy()) {
    context.builder().CreateCall(function, args);
    // TODO: use empty tuple type.
    // TODO: don't create the empty tuple if the call does not get assigned.
    context.SetLocal(node_id, context.builder().CreateAlloca(
                                  llvm::StructType::get(context.llvm_context()),
                                  /*ArraySize=*/nullptr, "TupleLiteralValue"));
  } else {
    context.SetLocal(node_id, context.builder().CreateCall(
                                  function, args, function->getName()));
  }
}

auto LoweringHandleFunctionDeclaration(LoweringFunctionContext& /*context*/,
                                       SemanticsNodeId /*node_id*/,
                                       SemanticsNode node) -> void {
  CARBON_FATAL()
      << "Should not be encountered. If that changes, we may want to change "
         "higher-level logic to skip them rather than calling this. "
      << node;
}

auto LoweringHandleIndex(LoweringFunctionContext& context,
                         SemanticsNodeId node_id, SemanticsNode node) -> void {
  auto [tuple_node_id, index_node_id] = node.GetAsIndex();
  auto* llvm_type =
      context.GetType(context.semantics_ir().GetNode(tuple_node_id).type_id());
  auto index_node = context.semantics_ir().GetNode(index_node_id);
  const auto index = context.semantics_ir()
                         .GetIntegerLiteral(index_node.GetAsIntegerLiteral())
                         .getZExtValue();
  auto* gep = context.builder().CreateStructGEP(
      llvm_type, context.GetLocal(tuple_node_id), index, "Index");
  context.SetLocal(node_id, gep);
}

auto LoweringHandleIntegerLiteral(LoweringFunctionContext& context,
                                  SemanticsNodeId node_id, SemanticsNode node)
    -> void {
  llvm::APInt i =
      context.semantics_ir().GetIntegerLiteral(node.GetAsIntegerLiteral());
  // TODO: This won't offer correct semantics, but seems close enough for now.
  llvm::Value* v =
      llvm::ConstantInt::get(context.builder().getInt32Ty(), i.getSExtValue());
  context.SetLocal(node_id, v);
}

auto LoweringHandleNamespace(LoweringFunctionContext& /*context*/,
                             SemanticsNodeId /*node_id*/,
                             SemanticsNode /*node*/) -> void {
  // No action to take.
}

auto LoweringHandleRealLiteral(LoweringFunctionContext& context,
                               SemanticsNodeId node_id, SemanticsNode node)
    -> void {
  SemanticsRealLiteral real =
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
                          SemanticsNodeId /*node_id*/, SemanticsNode /*node*/)
    -> void {
  context.builder().CreateRetVoid();
}

auto LoweringHandleReturnExpression(LoweringFunctionContext& context,
                                    SemanticsNodeId /*node_id*/,
                                    SemanticsNode node) -> void {
  SemanticsNodeId expr_id = node.GetAsReturnExpression();
  context.builder().CreateRet(context.GetLocalLoaded(expr_id));
}

auto LoweringHandleStringLiteral(LoweringFunctionContext& /*context*/,
                                 SemanticsNodeId /*node_id*/,
                                 SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleStructMemberAccess(LoweringFunctionContext& context,
                                      SemanticsNodeId node_id,
                                      SemanticsNode node) -> void {
  auto [struct_id, member_index] = node.GetAsStructMemberAccess();
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

auto LoweringHandleTupleValue(LoweringFunctionContext& context,
                              SemanticsNodeId node_id, SemanticsNode node)
    -> void {
  auto* llvm_type = context.GetType(node.type_id());
  auto* alloca = context.builder().CreateAlloca(
      llvm_type, /*ArraySize=*/nullptr, "TupleLiteralValue");
  context.SetLocal(node_id, alloca);

  auto refs = context.semantics_ir().GetNodeBlock(node.GetAsTupleValue());
  auto type_refs = context.semantics_ir().GetTypeBlock(
      context.semantics_ir()
          .GetNode(context.semantics_ir().GetType(node.type_id()))
          .GetAsTupleType());

  for (int i = 0; i < static_cast<int>(type_refs.size()); ++i) {
    auto* gep = context.builder().CreateStructGEP(llvm_type, alloca, i);
    context.builder().CreateStore(context.GetLocal(refs[i]), gep);
  }
}

auto LoweringHandleStructTypeField(LoweringFunctionContext& /*context*/,
                                   SemanticsNodeId /*node_id*/,
                                   SemanticsNode /*node*/) -> void {
  // No action to take.
}

auto LoweringHandleStructValue(LoweringFunctionContext& context,
                               SemanticsNodeId node_id, SemanticsNode node)
    -> void {
  auto* llvm_type = context.GetType(node.type_id());
  auto* alloca = context.builder().CreateAlloca(
      llvm_type, /*ArraySize=*/nullptr, "StructLiteralValue");
  context.SetLocal(node_id, alloca);

  auto refs = context.semantics_ir().GetNodeBlock(node.GetAsStructValue());
  // Get type information for member names.
  auto type_refs = context.semantics_ir().GetNodeBlock(
      context.semantics_ir()
          .GetNode(context.semantics_ir().GetType(node.type_id()))
          .GetAsStructType());
  for (int i = 0; i < static_cast<int>(refs.size()); ++i) {
    auto [field_name_id, field_type_id] =
        context.semantics_ir().GetNode(type_refs[i]).GetAsStructTypeField();
    auto member_name = context.semantics_ir().GetString(field_name_id);
    auto* gep =
        context.builder().CreateStructGEP(llvm_type, alloca, i, member_name);
    context.builder().CreateStore(context.GetLocal(refs[i]), gep);
  }
}

auto LoweringHandleStubReference(LoweringFunctionContext& context,
                                 SemanticsNodeId node_id, SemanticsNode node)
    -> void {
  context.SetLocal(node_id, context.GetLocal(node.GetAsStubReference()));
}

auto LoweringHandleUnaryOperatorNot(LoweringFunctionContext& context,
                                    SemanticsNodeId node_id, SemanticsNode node)
    -> void {
  context.SetLocal(node_id, context.builder().CreateNot(context.GetLocal(
                                node.GetAsUnaryOperatorNot())));
}

auto LoweringHandleVarStorage(LoweringFunctionContext& context,
                              SemanticsNodeId node_id, SemanticsNode node)
    -> void {
  // TODO: This should provide a name, not just `var`. Also, LLVM requires
  // globals to have a name. Do we want to generate a name, which would need to
  // be consistent across translation units, or use the given name, which
  // requires either looking ahead for BindName or restructuring semantics,
  // either of which affects the destructuring due to the difference in
  // storage?
  auto* alloca = context.builder().CreateAlloca(context.GetType(node.type_id()),
                                                /*ArraySize=*/nullptr, "var");
  context.SetLocal(node_id, alloca);
}

}  // namespace Carbon
