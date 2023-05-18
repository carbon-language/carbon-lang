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

auto LoweringHandleAssign(LoweringContext& context, SemanticsNodeId /*node_id*/,
                          SemanticsNode node) -> void {
  auto [storage_id, value_id] = node.GetAsAssign();
  if (value_id == SemanticsNodeId::BuiltinEmptyStruct ||
      value_id == SemanticsNodeId::BuiltinEmptyTuple) {
    // Elide the 0-length store; these have no value assigned and it should have
    // no effect.
    return;
  }
  context.builder().CreateStore(context.GetLoweredNodeAsValue(value_id),
                                context.GetLoweredNodeAsValue(storage_id));
}

auto LoweringHandleBinaryOperatorAdd(LoweringContext& /*context*/,
                                     SemanticsNodeId /*node_id*/,
                                     SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleBindName(LoweringContext& /*context*/,
                            SemanticsNodeId /*node_id*/, SemanticsNode /*node*/)
    -> void {
  // Probably need to do something here, but not necessary for now.
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
    args[i] = context.GetLoweredNodeAsType(
        context.semantics_ir().GetNode(param_refs[i]).type_id());
  }

  llvm::Type* return_type = context.GetLoweredNodeAsType(
      callable.return_type_id.is_valid()
          ? callable.return_type_id
          : SemanticsNodeId::BuiltinEmptyTupleType);
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
  llvm::APInt i =
      context.semantics_ir().GetIntegerLiteral(node.GetAsIntegerLiteral());
  // TODO: This won't offer correct semantics, but seems close enough for now.
  llvm::Value* v =
      llvm::ConstantInt::get(context.builder().getInt32Ty(), i.getSExtValue());
  context.SetLoweredNodeAsValue(node_id, v);
}

auto LoweringHandleRealLiteral(LoweringContext& context,
                               SemanticsNodeId node_id, SemanticsNode node)
    -> void {
  SemanticsRealLiteral real =
      context.semantics_ir().GetRealLiteral(node.GetAsRealLiteral());
  // TODO: This will probably have overflow issues, and should be fixed.
  double val =
      real.mantissa.getSExtValue() *
      std::pow((real.is_decimal ? 10 : 2), real.exponent.getSExtValue());
  llvm::APFloat llvm_val(val);
  context.SetLoweredNodeAsValue(
      node_id,
      llvm::ConstantFP::get(context.builder().getDoubleTy(), llvm_val));
}

auto LoweringHandleReturn(LoweringContext& context, SemanticsNodeId /*node_id*/,
                          SemanticsNode /*node*/) -> void {
  context.builder().CreateRetVoid();
}

auto LoweringHandleReturnExpression(LoweringContext& context,
                                    SemanticsNodeId /*node_id*/,
                                    SemanticsNode node) -> void {
  SemanticsNodeId expr_id = node.GetAsReturnExpression();
  context.builder().CreateRet(context.GetLoweredNodeAsValue(expr_id));
}

auto LoweringHandleStringLiteral(LoweringContext& /*context*/,
                                 SemanticsNodeId /*node_id*/,
                                 SemanticsNode node) -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleStructMemberAccess(LoweringContext& context,
                                      SemanticsNodeId node_id,
                                      SemanticsNode node) -> void {
  auto [struct_id, member_index] = node.GetAsStructMemberAccess();
  auto struct_type_id = context.semantics_ir().GetNode(struct_id).type_id();
  auto* llvm_type = context.GetLoweredNodeAsType(struct_type_id);

  // Get type information for member names.
  auto type_refs = context.semantics_ir().GetNodeBlock(
      context.semantics_ir().GetNode(struct_type_id).GetAsStructType());
  auto member_name = context.semantics_ir().GetString(
      context.semantics_ir()
          .GetNode(type_refs[member_index.index])
          .GetAsStructTypeField());

  auto* gep = context.builder().CreateStructGEP(
      llvm_type, context.GetLoweredNodeAsValue(struct_id), member_index.index,
      member_name);
  context.SetLoweredNodeAsValue(node_id, gep);
}

auto LoweringHandleStructType(LoweringContext& /*context*/,
                              SemanticsNodeId /*node_id*/,
                              SemanticsNode /*node*/) -> void {
  // No action to take.
}

auto LoweringHandleStructTypeField(LoweringContext& /*context*/,
                                   SemanticsNodeId /*node_id*/,
                                   SemanticsNode /*node*/) -> void {
  // No action to take.
}

auto LoweringHandleStructValue(LoweringContext& context,
                               SemanticsNodeId node_id, SemanticsNode node)
    -> void {
  auto* llvm_type = context.GetLoweredNodeAsType(node.type_id());
  auto* alloca = context.builder().CreateAlloca(
      llvm_type, /*ArraySize=*/nullptr, "StructLiteralValue");
  context.SetLoweredNodeAsValue(node_id, alloca);

  auto refs = context.semantics_ir().GetNodeBlock(node.GetAsStructValue());
  // Get type information for member names.
  auto type_refs = context.semantics_ir().GetNodeBlock(
      context.semantics_ir().GetNode(node.type_id()).GetAsStructType());
  for (int i = 0; i < static_cast<int>(refs.size()); ++i) {
    auto member_name = context.semantics_ir().GetString(
        context.semantics_ir().GetNode(type_refs[i]).GetAsStructTypeField());
    auto* gep =
        context.builder().CreateStructGEP(llvm_type, alloca, i, member_name);
    context.builder().CreateStore(context.GetLoweredNodeAsValue(refs[i]), gep);
  }
}

auto LoweringHandleStubReference(LoweringContext& context,
                                 SemanticsNodeId node_id, SemanticsNode node)
    -> void {
  context.SetLoweredNodeAsValue(
      node_id, context.GetLoweredNodeAsValue(node.GetAsStubReference()));
}

auto LoweringHandleVarStorage(LoweringContext& context, SemanticsNodeId node_id,
                              SemanticsNode node) -> void {
  // TODO: This should provide a name, not just `var`. Also, LLVM requires
  // globals to have a name. Do we want to generate a name, which would need to
  // be consistent across translation units, or use the given name, which
  // requires either looking ahead for BindName or restructuring semantics,
  // either of which affects the destructuring due to the difference in
  // storage?
  auto* alloca = context.builder().CreateAlloca(
      context.GetLoweredNodeAsType(node.type_id()), /*ArraySize=*/nullptr,
      "var");
  context.SetLoweredNodeAsValue(node_id, alloca);
}

}  // namespace Carbon
