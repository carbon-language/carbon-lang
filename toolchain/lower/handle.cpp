// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "toolchain/lower/function_context.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Lower {

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::AddrOf inst) -> void {
  context.SetLocal(inst_id, context.GetValue(inst.lvalue_id));
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
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

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::ArrayInit inst) -> void {
  // The result of initialization is the return slot of the initializer.
  context.SetLocal(inst_id, context.GetValue(inst.dest_id));
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::AsCompatible inst) -> void {
  context.SetLocal(inst_id, context.GetValue(inst.source_id));
}

auto HandleInst(FunctionContext& context, SemIR::InstId /*inst_id*/,
                SemIR::Assign inst) -> void {
  auto storage_type_id = context.sem_ir().insts().Get(inst.lhs_id).type_id();
  context.FinishInit(storage_type_id, inst.lhs_id, inst.rhs_id);
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::BindAlias inst) -> void {
  auto type_inst_id = context.sem_ir().types().GetInstId(inst.type_id);
  if (type_inst_id == SemIR::InstId::BuiltinNamespaceType) {
    return;
  }

  context.SetLocal(inst_id, context.GetValue(inst.value_id));
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::ExportDecl inst) -> void {
  auto type_inst_id = context.sem_ir().types().GetInstId(inst.type_id);
  if (type_inst_id == SemIR::InstId::BuiltinNamespaceType) {
    return;
  }

  context.SetLocal(inst_id, context.GetValue(inst.value_id));
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::BindName inst) -> void {
  context.SetLocal(inst_id, context.GetValue(inst.value_id));
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::BindSymbolicName inst) -> void {
  context.SetLocal(inst_id, context.GetValue(inst.value_id));
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::BlockArg inst) -> void {
  context.SetLocal(inst_id, context.GetBlockArg(inst.block_id, inst.type_id));
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::BoundMethod inst) -> void {
  // Propagate just the function; the object is separately provided to the
  // enclosing call as an implicit argument.
  context.SetLocal(inst_id, context.GetValue(inst.function_id));
}

auto HandleInst(FunctionContext& context, SemIR::InstId /*inst_id*/,
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

auto HandleInst(FunctionContext& context, SemIR::InstId /*inst_id*/,
                SemIR::BranchIf inst) -> void {
  llvm::Value* cond = context.GetValue(inst.cond_id);
  llvm::BasicBlock* then_block = context.GetBlock(inst.target_id);
  llvm::BasicBlock* else_block = context.MakeSyntheticBlock();
  context.builder().CreateCondBr(cond, then_block, else_block);
  context.builder().SetInsertPoint(else_block);
}

auto HandleInst(FunctionContext& context, SemIR::InstId /*inst_id*/,
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
    CARBON_CHECK(phi_predecessor,
                 "Synthetic block did not have a single predecessor");
  } else {
    context.builder().CreateBr(context.GetBlock(inst.target_id));
  }

  context.GetBlockArg(inst.target_id, arg_type_id)
      ->addIncoming(arg, phi_predecessor);
  context.builder().ClearInsertionPoint();
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::Converted inst) -> void {
  context.SetLocal(inst_id, context.GetValue(inst.result_id));
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::Deref inst) -> void {
  context.SetLocal(inst_id, context.GetValue(inst.pointer_id));
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::FacetAccessType /*inst*/) -> void {
  context.SetLocal(inst_id, context.GetTypeAsValue());
}

auto HandleInst(FunctionContext& context, SemIR::InstId /*inst_id*/,
                SemIR::InitializeFrom inst) -> void {
  auto storage_type_id = context.sem_ir().insts().Get(inst.dest_id).type_id();
  context.FinishInit(storage_type_id, inst.dest_id, inst.src_id);
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::NameRef inst) -> void {
  auto type_inst_id = context.sem_ir().types().GetInstId(inst.type_id);
  if (type_inst_id == SemIR::InstId::BuiltinNamespaceType) {
    return;
  }

  context.SetLocal(inst_id, context.GetValue(inst.value_id));
}

auto HandleInst(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                SemIR::OutParam /*inst*/) -> void {
  // Parameters are lowered by `BuildFunctionDefinition`.
}

auto HandleInst(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                SemIR::ValueParam /*inst*/) -> void {
  // Parameters are lowered by `BuildFunctionDefinition`.
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::ReturnSlot inst) -> void {
  if (SemIR::InitRepr::ForType(context.sem_ir(), inst.type_id).kind ==
      SemIR::InitRepr::InPlace) {
    context.SetLocal(inst_id, context.GetValue(inst.storage_id));
  }
}

auto HandleInst(FunctionContext& context, SemIR::InstId /*inst_id*/,
                SemIR::Return /*inst*/) -> void {
  context.builder().CreateRetVoid();
}

auto HandleInst(FunctionContext& context, SemIR::InstId /*inst_id*/,
                SemIR::ReturnExpr inst) -> void {
  auto result_type_id = context.sem_ir().insts().Get(inst.expr_id).type_id();
  switch (SemIR::InitRepr::ForType(context.sem_ir(), result_type_id).kind) {
    case SemIR::InitRepr::None:
      // Nothing to return.
      context.builder().CreateRetVoid();
      return;
    case SemIR::InitRepr::InPlace:
      context.FinishInit(result_type_id, inst.dest_id, inst.expr_id);
      context.builder().CreateRetVoid();
      return;
    case SemIR::InitRepr::ByCopy:
      // The expression produces the value representation for the type.
      context.builder().CreateRet(context.GetValue(inst.expr_id));
      return;
    case SemIR::InitRepr::Incomplete:
      CARBON_FATAL("Lowering return of incomplete type {0}",
                   context.sem_ir().types().GetAsInst(result_type_id));
  }
}

auto HandleInst(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                SemIR::SpecificFunction /*inst*/) -> void {
  // Nothing to do. This value should never be consumed.
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::SpliceBlock inst) -> void {
  context.LowerBlockContents(inst.block_id);
  context.SetLocal(inst_id, context.GetValue(inst.result_id));
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::UnaryOperatorNot inst) -> void {
  context.SetLocal(
      inst_id, context.builder().CreateNot(context.GetValue(inst.operand_id)));
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::VarStorage inst) -> void {
  context.SetLocal(inst_id,
                   context.builder().CreateAlloca(context.GetType(inst.type_id),
                                                  /*ArraySize=*/nullptr));
}

}  // namespace Carbon::Lower
