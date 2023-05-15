// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lowering/lowering_context.h"

namespace Carbon {

auto LoweringHandleInvalid(LoweringContext& /*context*/,
                           SemanticsNodeId /*node_id*/, SemanticsNode /*node*/)
    -> void {
  llvm_unreachable("never in actual IR");
}

auto LoweringHandleCrossReference(LoweringContext& /*context*/,
                                  SemanticsNodeId /*node_id*/,
                                  SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleAssign(LoweringContext& /*context*/,
                          SemanticsNodeId /*node_id*/, SemanticsNode node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleBinaryOperatorAdd(LoweringContext& /*context*/,
                                     SemanticsNodeId /*node_id*/,
                                     SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleBindName(LoweringContext& /*context*/,
                            SemanticsNodeId /*node_id*/, SemanticsNode node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleBuiltin(LoweringContext& /*context*/,
                           SemanticsNodeId /*node_id*/, SemanticsNode node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleCall(LoweringContext& /*context*/,
                        SemanticsNodeId /*node_id*/, SemanticsNode node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleCodeBlock(LoweringContext& /*context*/,
                             SemanticsNodeId /*node_id*/, SemanticsNode node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleFunctionDeclaration(LoweringContext& context,
                                       SemanticsNodeId /*node_id*/,
                                       SemanticsNode node) -> void {
  auto [name_id, callable_id] = node.GetAsFunctionDeclaration();
  auto callable = context.semantics_ir().GetCallable(callable_id);

  // TODO: Lower type information for the arguments prior to building args.
  auto param_refs = context.semantics_ir().GetNodeBlock(callable.param_refs_id);
  llvm::SmallVector<llvm::Type*> args;
  args.resize_for_overwrite(param_refs.size());
  for (int i = 0; i < static_cast<int>(param_refs.size()); ++i) {
    args[i] = context.LowerNodeToType(
        context.semantics_ir().GetNode(param_refs[i]).type_id());
  }

  llvm::Type* return_type = context.LowerNodeToType(
      callable.return_type_id.is_valid() ? callable.return_type_id
                                         : SemanticsNodeId::BuiltinEmptyTuple);
  llvm::FunctionType* function_type =
      llvm::FunctionType::get(return_type, args, /*isVarArg=*/false);
  auto* function = llvm::Function::Create(
      function_type, llvm::Function::ExternalLinkage,
      context.semantics_ir().GetString(name_id), context.llvm_module());

  // Set parameter names.
  for (int i = 0; i < static_cast<int>(param_refs.size()); ++i) {
    auto [param_name_id, _] =
        context.semantics_ir().GetNode(param_refs[i]).GetAsBindName();
    function->getArg(i)->setName(
        context.semantics_ir().GetString(param_name_id));
  }
}

auto LoweringHandleFunctionDefinition(LoweringContext& context,
                                      SemanticsNodeId /*node_id*/,
                                      SemanticsNode node) -> void {
  auto [declaration_id, body_block_id] = node.GetAsFunctionDefinition();
  auto [name_id, callable_id] =
      context.semantics_ir().GetNode(declaration_id).GetAsFunctionDeclaration();

  llvm::Function* function = context.llvm_module().getFunction(
      context.semantics_ir().GetString(name_id));

  // Create a new basic block to start insertion into.
  llvm::BasicBlock* body =
      llvm::BasicBlock::Create(context.llvm_context(), "entry", function);
  context.todo_blocks().push_back({body, body_block_id});
}

auto LoweringHandleIntegerLiteral(LoweringContext& context,
                                  SemanticsNodeId node_id, SemanticsNode node)
    -> void {
  SemanticsIntegerLiteralId int_id = node.GetAsIntegerLiteral();
  llvm::APInt i = context.semantics_ir().GetIntegerLiteral(int_id);
  llvm::Value* v = llvm::ConstantInt::get(context.builder().getInt32Ty(),
                                          i.getLimitedValue());
  context.lowered_nodes()[node_id.index] = v;
}

auto LoweringHandleRealLiteral(LoweringContext& /*context*/,
                               SemanticsNodeId /*node_id*/, SemanticsNode node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleReturn(LoweringContext& context, SemanticsNodeId /*node_id*/,
                          SemanticsNode /*node*/) -> void {
  context.builder().CreateRetVoid();
}

auto LoweringHandleReturnExpression(LoweringContext& context,
                                    SemanticsNodeId /*node_id*/,
                                    SemanticsNode node) -> void {
  SemanticsNodeId expr_id = node.GetAsReturnExpression();
  context.builder().CreateRet(context.lowered_nodes()[expr_id.index]);
}

auto LoweringHandleStringLiteral(LoweringContext& /*context*/,
                                 SemanticsNodeId /*node_id*/,
                                 SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleStructMemberAccess(LoweringContext& /*context*/,
                                      SemanticsNodeId /*node_id*/,
                                      SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleStructType(LoweringContext& /*context*/,
                              SemanticsNodeId /*node_id*/, SemanticsNode node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleStructTypeField(LoweringContext& /*context*/,
                                   SemanticsNodeId /*node_id*/,
                                   SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleStructValue(LoweringContext& /*context*/,
                               SemanticsNodeId /*node_id*/, SemanticsNode node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleStubReference(LoweringContext& /*context*/,
                                 SemanticsNodeId /*node_id*/,
                                 SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleVarStorage(LoweringContext& /*context*/,
                              SemanticsNodeId /*node_id*/, SemanticsNode node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

}  // namespace Carbon
