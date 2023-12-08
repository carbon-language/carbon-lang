// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "toolchain/lower/function_context.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"

namespace Carbon::Lower {

auto HandleCrossRef(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                    SemIR::CrossRef inst) -> void {
  CARBON_FATAL() << "TODO: Add support: " << inst;
}

auto HandleAddressOf(FunctionContext& context, SemIR::InstId inst_id,
                     SemIR::AddressOf inst) -> void {
  context.SetLocal(inst_id, context.GetValue(inst.lvalue_id));
}

auto HandleArrayIndex(FunctionContext& context, SemIR::InstId inst_id,
                      SemIR::ArrayIndex inst) -> void {
  auto* array_value = context.GetValue(inst.array_id);
  auto* llvm_type =
      context.GetType(context.sem_ir().insts().Get(inst.array_id).type_id());
  llvm::Value* indexes[2] = {
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context.llvm_context()), 0),
      context.GetValue(inst.index_id)};
  context.SetLocal(inst_id,
                   context.builder().CreateInBoundsGEP(llvm_type, array_value,
                                                       indexes, "array.index"));
}

auto HandleArrayInit(FunctionContext& context, SemIR::InstId inst_id,
                     SemIR::ArrayInit inst) -> void {
  // The result of initialization is the return slot of the initializer.
  context.SetLocal(inst_id, context.GetValue(inst.dest_id));
}

auto HandleAssign(FunctionContext& context, SemIR::InstId /*inst_id*/,
                  SemIR::Assign inst) -> void {
  auto storage_type_id = context.sem_ir().insts().Get(inst.lhs_id).type_id();
  context.FinishInit(storage_type_id, inst.lhs_id, inst.rhs_id);
}

auto HandleBindName(FunctionContext& context, SemIR::InstId inst_id,
                    SemIR::BindName inst) -> void {
  context.SetLocal(inst_id, context.GetValue(inst.value_id));
}

auto HandleBlockArg(FunctionContext& context, SemIR::InstId inst_id,
                    SemIR::BlockArg inst) -> void {
  context.SetLocal(inst_id, context.GetBlockArg(inst.block_id, inst.type_id));
}

auto HandleBoolLiteral(FunctionContext& context, SemIR::InstId inst_id,
                       SemIR::BoolLiteral inst) -> void {
  llvm::Value* v =
      llvm::ConstantInt::get(context.builder().getInt1Ty(), inst.value.index);
  context.SetLocal(inst_id, v);
}

auto HandleBoundMethod(FunctionContext& context, SemIR::InstId inst_id,
                       SemIR::BoundMethod inst) -> void {
  // Propagate just the function; the object is separately provided to the
  // enclosing call as an implicit argument.
  context.SetLocal(inst_id, context.GetValue(inst.function_id));
}

auto HandleBranch(FunctionContext& context, SemIR::InstId /*inst_id*/,
                  SemIR::Branch inst) -> void {
  // Opportunistically avoid creating a BasicBlock that contains just a branch.
  // TODO: Don't do this if it would remove a loop preheader block.
  llvm::BasicBlock* block = context.builder().GetInsertBlock();
  if (block->empty() && context.TryToReuseBlock(inst.target_id, block)) {
    // Reuse this block as the branch target.
  } else {
    context.builder().CreateBr(context.GetBlock(inst.target_id));
  }

  context.builder().ClearInsertionPoint();
}

auto HandleBranchIf(FunctionContext& context, SemIR::InstId /*inst_id*/,
                    SemIR::BranchIf inst) -> void {
  llvm::Value* cond = context.GetValue(inst.cond_id);
  llvm::BasicBlock* then_block = context.GetBlock(inst.target_id);
  llvm::BasicBlock* else_block = context.CreateSyntheticBlock();
  context.builder().CreateCondBr(cond, then_block, else_block);
  context.builder().SetInsertPoint(else_block);
}

auto HandleBranchWithArg(FunctionContext& context, SemIR::InstId /*inst_id*/,
                         SemIR::BranchWithArg inst) -> void {
  llvm::Value* arg = context.GetValue(inst.arg_id);
  SemIR::TypeId arg_type_id =
      context.sem_ir().insts().Get(inst.arg_id).type_id();

  // Opportunistically avoid creating a BasicBlock that contains just a branch.
  // We only do this for a block that we know will only have a single
  // predecessor, so that we can correctly populate the predecessors of the
  // PHINode.
  llvm::BasicBlock* block = context.builder().GetInsertBlock();
  llvm::BasicBlock* phi_predecessor = block;
  if (block->empty() && context.IsCurrentSyntheticBlock(block) &&
      context.TryToReuseBlock(inst.target_id, block)) {
    // Reuse this block as the branch target.
    phi_predecessor = block->getSinglePredecessor();
    CARBON_CHECK(phi_predecessor)
        << "Synthetic block did not have a single predecessor";
  } else {
    context.builder().CreateBr(context.GetBlock(inst.target_id));
  }

  context.GetBlockArg(inst.target_id, arg_type_id)
      ->addIncoming(arg, phi_predecessor);
  context.builder().ClearInsertionPoint();
}

auto HandleBuiltin(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                   SemIR::Builtin inst) -> void {
  CARBON_FATAL() << "TODO: Add support: " << inst;
}

auto HandleCall(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::Call inst) -> void {
  auto* callee = llvm::cast<llvm::Function>(context.GetValue(inst.callee_id));

  std::vector<llvm::Value*> args;
  llvm::ArrayRef<SemIR::InstId> arg_ids =
      context.sem_ir().inst_blocks().Get(inst.args_id);

  if (SemIR::GetInitRepr(context.sem_ir(), inst.type_id).has_return_slot()) {
    args.push_back(context.GetValue(arg_ids.back()));
    arg_ids = arg_ids.drop_back();
  }

  for (auto arg_id : arg_ids) {
    auto arg_type_id = context.sem_ir().insts().Get(arg_id).type_id();
    if (SemIR::GetValueRepr(context.sem_ir(), arg_type_id).kind !=
        SemIR::ValueRepr::None) {
      args.push_back(context.GetValue(arg_id));
    }
  }

  auto* call = context.builder().CreateCall(callee, args);
  context.SetLocal(inst_id, call);

  // Name the call's result the same as the callee.
  // TODO: Is this a helpful name?
  if (!call->getType()->isVoidTy()) {
    call->setName(callee->getName());
  }
}

auto HandleConverted(FunctionContext& context, SemIR::InstId inst_id,
                     SemIR::Converted inst) -> void {
  context.SetLocal(inst_id, context.GetValue(inst.result_id));
}

auto HandleDeref(FunctionContext& context, SemIR::InstId inst_id,
                 SemIR::Deref inst) -> void {
  context.SetLocal(inst_id, context.GetValue(inst.pointer_id));
}

auto HandleFunctionDecl(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                        SemIR::FunctionDecl inst) -> void {
  CARBON_FATAL()
      << "Should not be encountered. If that changes, we may want to change "
         "higher-level logic to skip them rather than calling this. "
      << inst;
}

auto HandleImport(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                  SemIR::Import inst) -> void {
  CARBON_FATAL()
      << "Should not be encountered. If that changes, we may want to change "
         "higher-level logic to skip them rather than calling this. "
      << inst;
}

auto HandleInitializeFrom(FunctionContext& context, SemIR::InstId /*inst_id*/,
                          SemIR::InitializeFrom inst) -> void {
  auto storage_type_id = context.sem_ir().insts().Get(inst.dest_id).type_id();
  context.FinishInit(storage_type_id, inst.dest_id, inst.src_id);
}

auto HandleInterfaceDecl(FunctionContext& /*context*/,
                         SemIR::InstId /*inst_id*/,
                         SemIR::InterfaceDecl /*inst*/) -> void {
  // No action to perform.
}

auto HandleIntLiteral(FunctionContext& context, SemIR::InstId inst_id,
                      SemIR::IntLiteral inst) -> void {
  const llvm::APInt& i = context.sem_ir().ints().Get(inst.int_id);
  // TODO: This won't offer correct semantics, but seems close enough for now.
  llvm::Value* v =
      llvm::ConstantInt::get(context.builder().getInt32Ty(), i.getZExtValue());
  context.SetLocal(inst_id, v);
}

auto HandleNameRef(FunctionContext& context, SemIR::InstId inst_id,
                   SemIR::NameRef inst) -> void {
  auto type_inst_id = context.sem_ir().types().GetInstId(inst.type_id);
  if (type_inst_id == SemIR::InstId::BuiltinNamespaceType) {
    return;
  }

  context.SetLocal(inst_id, context.GetValue(inst.value_id));
}

auto HandleNamespace(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                     SemIR::Namespace inst) -> void {
  CARBON_FATAL()
      << "Should not be encountered. If that changes, we may want to change "
         "higher-level logic to skip them rather than calling this. "
      << inst;
}

auto HandleNoOp(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                SemIR::NoOp /*inst*/) -> void {
  // No action to take.
}

auto HandleParam(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                 SemIR::Param /*inst*/) -> void {
  CARBON_FATAL() << "Parameters should be lowered by `BuildFunctionDefinition`";
}

auto HandleRealLiteral(FunctionContext& context, SemIR::InstId inst_id,
                       SemIR::RealLiteral inst) -> void {
  const Real& real = context.sem_ir().reals().Get(inst.real_id);
  // TODO: This will probably have overflow issues, and should be fixed.
  double val =
      real.mantissa.getZExtValue() *
      std::pow((real.is_decimal ? 10 : 2), real.exponent.getSExtValue());
  llvm::APFloat llvm_val(val);
  context.SetLocal(inst_id, llvm::ConstantFP::get(
                                context.builder().getDoubleTy(), llvm_val));
}

auto HandleReturn(FunctionContext& context, SemIR::InstId /*inst_id*/,
                  SemIR::Return /*inst*/) -> void {
  context.builder().CreateRetVoid();
}

auto HandleReturnExpr(FunctionContext& context, SemIR::InstId /*inst_id*/,
                      SemIR::ReturnExpr inst) -> void {
  switch (
      SemIR::GetInitRepr(context.sem_ir(),
                         context.sem_ir().insts().Get(inst.expr_id).type_id())
          .kind) {
    case SemIR::InitRepr::None:
    case SemIR::InitRepr::InPlace:
      // Nothing to return.
      context.builder().CreateRetVoid();
      return;
    case SemIR::InitRepr::ByCopy:
      // The expression produces the value representation for the type.
      context.builder().CreateRet(context.GetValue(inst.expr_id));
      return;
  }
}

auto HandleSelfParam(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                     SemIR::SelfParam /*inst*/) -> void {
  CARBON_FATAL() << "Parameters should be lowered by `BuildFunctionDefinition`";
}

auto HandleSpliceBlock(FunctionContext& context, SemIR::InstId inst_id,
                       SemIR::SpliceBlock inst) -> void {
  context.LowerBlock(inst.block_id);
  context.SetLocal(inst_id, context.GetValue(inst.result_id));
}

auto HandleStringLiteral(FunctionContext& /*context*/,
                         SemIR::InstId /*inst_id*/, SemIR::StringLiteral inst)
    -> void {
  CARBON_FATAL() << "TODO: Add support: " << inst;
}

auto HandleUnaryOperatorNot(FunctionContext& context, SemIR::InstId inst_id,
                            SemIR::UnaryOperatorNot inst) -> void {
  context.SetLocal(
      inst_id, context.builder().CreateNot(context.GetValue(inst.operand_id)));
}

auto HandleVarStorage(FunctionContext& context, SemIR::InstId inst_id,
                      SemIR::VarStorage inst) -> void {
  // TODO: Eventually this name will be optional, and we'll want to provide
  // something like `var` as a default. However, that's not possible right now
  // so cannot be tested.
  auto name = context.sem_ir().names().GetIRBaseName(inst.name_id);
  auto* alloca = context.builder().CreateAlloca(context.GetType(inst.type_id),
                                                /*ArraySize=*/nullptr, name);
  context.SetLocal(inst_id, alloca);
}

}  // namespace Carbon::Lower
