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
      semantics_ir_(&semantics_ir),
      lowered_nodes_(semantics_ir_->nodes_size(), nullptr) {
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

auto Lowering::LowerNodeToType(SemanticsNodeId node_id) -> llvm::Type* {
  CARBON_CHECK(node_id.is_valid());
  switch (node_id.index) {
    case SemanticsBuiltinKind::EmptyTuple.AsInt():
      // TODO: Should probably switch this to an actual empty tuple in the
      // future, but it's implemented as void for now.
      return builder_.getVoidTy();
    case SemanticsBuiltinKind::IntegerType.AsInt():
      // TODO: Handle different sizes.
      return builder_.getInt32Ty();
    default:
      CARBON_FATAL() << "Cannot use node as type: " << node_id;
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

  // TODO: Lower type information for the arguments prior to building args.
  auto param_refs = semantics_ir_->GetNodeBlock(callable.param_refs_id);
  llvm::SmallVector<llvm::Type*> args;
  args.resize_for_overwrite(param_refs.size());
  for (int i = 0; i < static_cast<int>(param_refs.size()); ++i) {
    args[i] = LowerNodeToType(semantics_ir_->GetNode(param_refs[i]).type_id());
  }

  llvm::Type* return_type = LowerNodeToType(
      callable.return_type_id.is_valid() ? callable.return_type_id
                                         : SemanticsNodeId::BuiltinEmptyTuple);
  llvm::FunctionType* function_type =
      llvm::FunctionType::get(return_type, args, /*isVarArg=*/false);
  auto* function = llvm::Function::Create(
      function_type, llvm::Function::ExternalLinkage,
      semantics_ir_->GetString(name_id), llvm_module_.get());

  // Set parameter names.
  for (int i = 0; i < static_cast<int>(param_refs.size()); ++i) {
    auto [param_name_id, _] =
        semantics_ir_->GetNode(param_refs[i]).GetAsBindName();
    function->getArg(i)->setName(semantics_ir_->GetString(param_name_id));
  }
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
  lowered_nodes_[node_id.index] = v;
}

auto Lowering::HandleRealLiteralNode(SemanticsNodeId /*node_id*/,
                                     SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto Lowering::HandleReturnNode(SemanticsNodeId /*node_id*/,
                                SemanticsNode /*node*/) -> void {
  builder_.CreateRetVoid();
}

auto Lowering::HandleReturnExpressionNode(SemanticsNodeId /*node_id*/,
                                          SemanticsNode node) -> void {
  SemanticsNodeId expr_id = node.GetAsReturnExpression();
  builder_.CreateRet(lowered_nodes_[expr_id.index]);
}

auto Lowering::HandleStringLiteralNode(SemanticsNodeId /*node_id*/,
                                       SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto Lowering::HandleStructMemberAccessNode(SemanticsNodeId /*node_id*/,
                                            SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto Lowering::HandleStructTypeNode(SemanticsNodeId /*node_id*/,
                                    SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto Lowering::HandleStructTypeFieldNode(SemanticsNodeId /*node_id*/,
                                         SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto Lowering::HandleStructValueNode(SemanticsNodeId /*node_id*/,
                                     SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto Lowering::HandleStubReferenceNode(SemanticsNodeId /*node_id*/,
                                       SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto Lowering::HandleVarStorageNode(SemanticsNodeId /*node_id*/,
                                    SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

}  // namespace Carbon
