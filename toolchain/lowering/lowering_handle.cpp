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
  context.builder().CreateStore(context.GetLocalLoaded(value_id),
                                context.GetLocal(storage_id));
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

auto LoweringHandleCall(LoweringContext& context, SemanticsNodeId node_id,
                        SemanticsNode node) -> void {
  auto [refs_id, function_id] = node.GetAsCall();
  auto* function = context.GetFunction(function_id);
  std::vector<llvm::Value*> args;
  for (auto ref_id : context.semantics_ir().GetNodeBlock(refs_id)) {
    args.push_back(context.GetLocalLoaded(ref_id));
  }
  auto* value =
      context.builder().CreateCall(function, args, function->getName());
  context.SetLocal(node_id, value);
}

auto LoweringHandleCodeBlock(LoweringContext& /*context*/,
                             SemanticsNodeId /*node_id*/, SemanticsNode node)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << node;
}

auto LoweringHandleFunctionDeclaration(LoweringContext& /*context*/,
                                       SemanticsNodeId /*node_id*/,
                                       SemanticsNode node) -> void {
  CARBON_FATAL()
      << "Should not be encountered. If that changes, we may want to change "
         "higher-level logic to skip them rather than calling this. "
      << node;
}

auto LoweringHandleIntegerLiteral(LoweringContext& context,
                                  SemanticsNodeId node_id, SemanticsNode node)
    -> void {
  llvm::APInt i =
      context.semantics_ir().GetIntegerLiteral(node.GetAsIntegerLiteral());
  // TODO: This won't offer correct semantics, but seems close enough for now.
  llvm::Value* v =
      llvm::ConstantInt::get(context.builder().getInt32Ty(), i.getSExtValue());
  context.SetLocal(node_id, v);
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
  context.SetLocal(node_id, llvm::ConstantFP::get(
                                context.builder().getDoubleTy(), llvm_val));
}

auto LoweringHandleReturn(LoweringContext& context, SemanticsNodeId /*node_id*/,
                          SemanticsNode /*node*/) -> void {
  context.builder().CreateRetVoid();
}

auto LoweringHandleReturnExpression(LoweringContext& context,
                                    SemanticsNodeId /*node_id*/,
                                    SemanticsNode node) -> void {
  SemanticsNodeId expr_id = node.GetAsReturnExpression();
  context.builder().CreateRet(context.GetLocalLoaded(expr_id));
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
  auto* llvm_type = context.GetType(struct_type_id);

  // Get type information for member names.
  auto type_refs = context.semantics_ir().GetNodeBlock(
      context.semantics_ir()
          .GetNode(context.semantics_ir().GetType(struct_type_id))
          .GetAsStructType());
  auto member_name = context.semantics_ir().GetString(
      context.semantics_ir()
          .GetNode(type_refs[member_index.index])
          .GetAsStructTypeField());

  auto* gep = context.builder().CreateStructGEP(
      llvm_type, context.GetLocal(struct_id), member_index.index, member_name);
  context.SetLocal(node_id, gep);
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
    auto member_name = context.semantics_ir().GetString(
        context.semantics_ir().GetNode(type_refs[i]).GetAsStructTypeField());
    auto* gep =
        context.builder().CreateStructGEP(llvm_type, alloca, i, member_name);
    context.builder().CreateStore(context.GetLocal(refs[i]), gep);
  }
}

auto LoweringHandleStubReference(LoweringContext& context,
                                 SemanticsNodeId node_id, SemanticsNode node)
    -> void {
  context.SetLocal(node_id, context.GetLocal(node.GetAsStubReference()));
}

auto LoweringHandleVarStorage(LoweringContext& context, SemanticsNodeId node_id,
                              SemanticsNode node) -> void {
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
