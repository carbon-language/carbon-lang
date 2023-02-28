// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lowering/lowering.h"

#include "toolchain/semantics/semantics_ir.h"

namespace Carbon {

Lowering::Lowering(llvm::LLVMContext& llvm_context, llvm::StringRef module_name,
                   const SemanticsIR& semantics_ir)
    : llvm_context_(&llvm_context),
      llvm_module_(std::make_unique<llvm::Module>(module_name, llvm_context)),
      builder_(llvm_context),
      semantics_ir_(&semantics_ir) {
  CARBON_CHECK(!semantics_ir.has_errors())
      << "Generating LLVM IR from invalid SemanticsIR is unsupported.";
}

auto Lowering::Run() -> std::unique_ptr<llvm::Module> {
  CARBON_CHECK(llvm_module_) << "Run can only be called once.";

  LowerBlock(semantics_ir_->top_node_block_id());

  while (!todo_blocks_.empty()) {
    auto [llvm_block, block_id] = todo_blocks_.pop_back_val();
    builder_.SetInsertPoint(llvm_block);
    LowerBlock(block_id);
  }

  return std::move(llvm_module_);
}

auto Lowering::LowerBlock(SemanticsNodeBlockId block_id) -> void {
  for (const auto& node_id : semantics_ir_->GetNodeBlock(block_id)) {
    auto node = semantics_ir_->GetNode(node_id);
    switch (node.kind()) {
#define CARBON_SEMANTICS_NODE_KIND(Name) \
  case SemanticsNodeKind::Name:          \
    Handle##Name##Node(node_id, node);   \
    break;
#include "toolchain/semantics/semantics_node_kind.def"
    }
  }
}

auto Lowering::HandleInvalidNode(SemanticsNodeId /*node_id*/,
                                 SemanticsNode /*node*/) -> void {
  llvm_unreachable("never in actual IR");
}

auto Lowering::HandleCrossReferenceNode(SemanticsNodeId /*node_id*/,
                                        SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto Lowering::HandleAssignNode(SemanticsNodeId /*node_id*/, SemanticsNode node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto Lowering::HandleBinaryOperatorAddNode(SemanticsNodeId /*node_id*/,
                                           SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto Lowering::HandleBindNameNode(SemanticsNodeId /*node_id*/,
                                  SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto Lowering::HandleBuiltinNode(SemanticsNodeId /*node_id*/,
                                 SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto Lowering::HandleCallNode(SemanticsNodeId /*node_id*/, SemanticsNode node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto Lowering::HandleCodeBlockNode(SemanticsNodeId /*node_id*/,
                                   SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto Lowering::HandleFunctionDeclarationNode(SemanticsNodeId /*node_id*/,
                                             SemanticsNode node) -> void {
  auto [name_id, callable_id] = node.GetAsFunctionDeclaration();
  auto callable = semantics_ir_->GetCallable(callable_id);
  llvm::SmallVector<llvm::Type*> args;
  // Note, when handling non-empty parameters, we'll also want to set names.
  CARBON_CHECK(callable.param_refs_id == SemanticsNodeBlockId::Empty)
      << "TODO: Handle non-empty parameters.";
  CARBON_CHECK(callable.return_type_id == SemanticsNodeId::BuiltinIntegerType)
      << "TODO: Handle non-i32 return types.";
  llvm::Type* return_type = builder_.getInt32Ty();
  llvm::FunctionType* function_type =
      llvm::FunctionType::get(return_type, args, /*isVarArg=*/false);
  llvm::Function::Create(function_type, llvm::Function::ExternalLinkage,
                         semantics_ir_->GetString(name_id), llvm_module_.get());
  // TODO: Name arguments.
}

auto Lowering::HandleFunctionDefinitionNode(SemanticsNodeId /*node_id*/,
                                            SemanticsNode node) -> void {
  auto [declaration_id, body_block_id] = node.GetAsFunctionDefinition();
  auto [name_id, callable_id] =
      semantics_ir_->GetNode(declaration_id).GetAsFunctionDeclaration();

  llvm::Function* function =
      llvm_module_->getFunction(semantics_ir_->GetString(name_id));

  // Create a new basic block to start insertion into.
  llvm::BasicBlock* body =
      llvm::BasicBlock::Create(*llvm_context_, "entry", function);
  todo_blocks_.push_back({body, body_block_id});
}

auto Lowering::HandleIntegerLiteralNode(SemanticsNodeId node_id,
                                        SemanticsNode node) -> void {
  SemanticsIntegerLiteralId int_id = node.GetAsIntegerLiteral();
  llvm::APInt i = semantics_ir_->GetIntegerLiteral(int_id);
  llvm::Value* v =
      llvm::ConstantInt::get(builder_.getInt32Ty(), i.getLimitedValue());
  node_values_[node_id] = v;
}

auto Lowering::HandleRealLiteralNode(SemanticsNodeId /*node_id*/,
                                     SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto Lowering::HandleReturnNode(SemanticsNodeId /*node_id*/, SemanticsNode node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto Lowering::HandleReturnExpressionNode(SemanticsNodeId /*node_id*/,
                                          SemanticsNode node) -> void {
  SemanticsNodeId expr_id = node.GetAsReturnExpression();
  builder_.CreateRet(node_values_[expr_id]);
}

auto Lowering::HandleStringLiteralNode(SemanticsNodeId /*node_id*/,
                                       SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto Lowering::HandleVarStorageNode(SemanticsNodeId /*node_id*/,
                                    SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

}  // namespace Carbon
