// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lowering/lowering.h"

#include "toolchain/semantics/semantics_ir.h"

/*
semantics_ir: cross_reference_irs_size: 1
calls: [
]
callables: [
  {param_ir: block0, param_refs: block0, return_type: node2},
]
integer_literals: [
  0,
]
real_literals: [
]
strings: [
  Main,
]
nodes: [
 0 {kind: CrossReference, arg0: ir0, arg1: node0, type: node0},
 1 {kind: CrossReference, arg0: ir0, arg1: node1, type: node1},
 2 {kind: CrossReference, arg0: ir0, arg1: node2, type: node0},
 3 {kind: CrossReference, arg0: ir0, arg1: node3, type: node0},
 4 {kind: CrossReference, arg0: ir0, arg1: node4, type: node0},
 5 {kind: CrossReference, arg0: ir0, arg1: node5, type: node0},
 6 {kind: CrossReference, arg0: ir0, arg1: node6, type: node5},
 7 {kind: FunctionDeclaration, arg0: callable0},
 8 {kind: BindName, arg0: str0, arg1: node7},
 9 {kind: IntegerLiteral, arg0: int0, type: node2},
 10 {kind: ReturnExpression, arg0: node9, type: node2},
 11 {kind: FunctionDefinition, arg0: node7, arg1: block2},
]
node_blocks: [
  [
  ],
  [
    node7,
    node8,
    node11,
  ],
  [
    node9,
    node10,
  ],
]
*/

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

  for (const auto& node_id :
       semantics_ir_->GetNodeBlock(semantics_ir_->top_node_block_id())) {
    auto node = semantics_ir_->GetNode(node_id);
    switch (node.kind()) {
#define CARBON_SEMANTICS_NODE_KIND(Name) \
  case SemanticsNodeKind::Name:          \
    Handle##Name##Node(node);            \
    break;
#include "toolchain/semantics/semantics_node_kind.def"
    }
  }

  return std::move(llvm_module_);
}

auto Lowering::HandleInvalidNode(SemanticsNode /*node*/) -> void {
  llvm_unreachable("never in actual IR");
}

auto Lowering::HandleCrossReferenceNode(SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node.kind();
}

auto Lowering::HandleAssignNode(SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node.kind();
}

auto Lowering::HandleBinaryOperatorAddNode(SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node.kind();
}

auto Lowering::HandleBindNameNode(SemanticsNode /*node*/) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node.kind();
}

auto Lowering::HandleBuiltinNode(SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node.kind();
}

auto Lowering::HandleCallNode(SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node.kind();
}

auto Lowering::HandleCodeBlockNode(SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node.kind();
}

auto Lowering::HandleFunctionDeclarationNode(SemanticsNode node) -> void {
  auto [name_id, callable_id] = node.GetAsFunctionDeclaration();
  auto callable = semantics_ir_->GetCallable(callable_id);
  llvm::SmallVector<llvm::Type*> args;
  // Note, when handling non-empty parameters, we'll also want to set names.
  CARBON_CHECK(callable.param_refs_id == SemanticsNodeBlockId::Empty)
      << "TODO: Handle non-empty parameters.";
  CARBON_CHECK(callable.return_type_id == SemanticsNodeId::BuiltinIntegerType)
      << "TODO: Handle non-i32 return types.";
  llvm::Type* return_type = llvm::Type::getInt32Ty(*llvm_context_);
  llvm::FunctionType* function_type =
      llvm::FunctionType::get(return_type, args, /*isVarArg=*/false);
  llvm::Function::Create(function_type, llvm::Function::ExternalLinkage,
                         "TODO_NAME", llvm_module_.get());
}

auto Lowering::HandleFunctionDefinitionNode(SemanticsNode node) -> void {
  auto [declaration_id, body_block_id] = node.GetAsFunctionDefinition();
  auto declaration = semantics_ir_->GetNode(declaration_id);

  // TODO: Currently used.
  // CARBON_FATAL() << "TODO: Add support: " << node.kind();
}

auto Lowering::HandleIntegerLiteralNode(SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node.kind();
}

auto Lowering::HandleRealLiteralNode(SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node.kind();
}

auto Lowering::HandleReturnNode(SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node.kind();
}

auto Lowering::HandleReturnExpressionNode(SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node.kind();
}

auto Lowering::HandleStringLiteralNode(SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node.kind();
}

auto Lowering::HandleVarStorageNode(SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node.kind();
}

}  // namespace Carbon
