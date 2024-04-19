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
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Lower {

template <typename InstT>
static auto FatalErrorIfEncountered(InstT inst) -> void {
  CARBON_FATAL()
      << "Encountered an instruction that isn't expected to lower. It's "
         "possible that logic needs to be changed in order to stop "
         "showing this instruction in lowered contexts. Instruction: "
      << inst;
}

auto HandleAdaptDecl(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                     SemIR::AdaptDecl inst) -> void {
  FatalErrorIfEncountered(inst);
}

auto HandleAddrOf(FunctionContext& context, SemIR::InstId inst_id,
                  SemIR::AddrOf inst) -> void {
  context.SetLocal(inst_id, context.GetValue(inst.lvalue_id));
}

auto HandleAddrPattern(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                       SemIR::AddrPattern /*inst*/) -> void {
  CARBON_FATAL() << "`addr` should be lowered by `BuildFunctionDefinition`";
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

auto HandleAsCompatible(FunctionContext& context, SemIR::InstId inst_id,
                        SemIR::AsCompatible inst) -> void {
  context.SetLocal(inst_id, context.GetValue(inst.source_id));
}

auto HandleAssign(FunctionContext& context, SemIR::InstId /*inst_id*/,
                  SemIR::Assign inst) -> void {
  auto storage_type_id = context.sem_ir().insts().Get(inst.lhs_id).type_id();
  context.FinishInit(storage_type_id, inst.lhs_id, inst.rhs_id);
}

auto HandleAssociatedConstantDecl(FunctionContext& /*context*/,
                                  SemIR::InstId /*inst_id*/,
                                  SemIR::AssociatedConstantDecl inst) -> void {
  FatalErrorIfEncountered(inst);
}

auto HandleAssociatedEntity(FunctionContext& /*context*/,
                            SemIR::InstId /*inst_id*/,
                            SemIR::AssociatedEntity inst) -> void {
  FatalErrorIfEncountered(inst);
}

auto HandleBindAlias(FunctionContext& context, SemIR::InstId inst_id,
                     SemIR::BindAlias inst) -> void {
  auto type_inst_id = context.sem_ir().types().GetInstId(inst.type_id);
  if (type_inst_id == SemIR::InstId::BuiltinNamespaceType) {
    return;
  }

  context.SetLocal(inst_id, context.GetValue(inst.value_id));
}

auto HandleBindName(FunctionContext& context, SemIR::InstId inst_id,
                    SemIR::BindName inst) -> void {
  context.SetLocal(inst_id, context.GetValue(inst.value_id));
}

auto HandleBindSymbolicName(FunctionContext& context, SemIR::InstId inst_id,
                            SemIR::BindSymbolicName inst) -> void {
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
  llvm::BasicBlock* else_block = context.MakeSyntheticBlock();
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

// Get the predicate to use for an `icmp` instruction generated for the
// specified builtin.
static auto GetBuiltinICmpPredicate(SemIR::BuiltinFunctionKind builtin_kind,
                                    bool is_signed)
    -> llvm::CmpInst::Predicate {
  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::IntEq:
      return llvm::CmpInst::ICMP_EQ;
    case SemIR::BuiltinFunctionKind::IntNeq:
      return llvm::CmpInst::ICMP_NE;
    case SemIR::BuiltinFunctionKind::IntLess:
      return is_signed ? llvm::CmpInst::ICMP_SLT : llvm::CmpInst::ICMP_ULT;
    case SemIR::BuiltinFunctionKind::IntLessEq:
      return is_signed ? llvm::CmpInst::ICMP_SLE : llvm::CmpInst::ICMP_ULE;
    case SemIR::BuiltinFunctionKind::IntGreater:
      return is_signed ? llvm::CmpInst::ICMP_SGT : llvm::CmpInst::ICMP_UGT;
    case SemIR::BuiltinFunctionKind::IntGreaterEq:
      return is_signed ? llvm::CmpInst::ICMP_SGE : llvm::CmpInst::ICMP_UGE;
    default:
      CARBON_FATAL() << "Unexpected builtin kind " << builtin_kind;
  }
}

// Returns whether the specified instruction has a signed integer type.
static auto IsSignedInt(FunctionContext& context, SemIR::InstId int_id)
    -> bool {
  return context.sem_ir().types().IsSignedInt(
      context.sem_ir().insts().Get(int_id).type_id());
}

// Handles a call to a builtin function.
static auto HandleBuiltinCall(FunctionContext& context, SemIR::InstId inst_id,
                              SemIR::BuiltinFunctionKind builtin_kind,
                              llvm::ArrayRef<SemIR::InstId> arg_ids) -> void {
  // TODO: Consider setting this to true in the performance build mode if the
  // result type is a signed integer type.
  constexpr bool SignedOverflowIsUB = false;

  // TODO: Move the instruction names here into InstNamer.
  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::None:
      CARBON_FATAL() << "No callee in function call.";

    case SemIR::BuiltinFunctionKind::BoolMakeType:
    case SemIR::BuiltinFunctionKind::FloatMakeType:
    case SemIR::BuiltinFunctionKind::IntMakeType32:
    case SemIR::BuiltinFunctionKind::IntMakeTypeSigned:
    case SemIR::BuiltinFunctionKind::IntMakeTypeUnsigned:
      context.SetLocal(inst_id, context.GetTypeAsValue());
      return;

    case SemIR::BuiltinFunctionKind::IntSNegate: {
      // Lower `-x` as `0 - x`.
      auto* operand = context.GetValue(arg_ids[0]);
      context.SetLocal(
          inst_id,
          context.builder().CreateSub(
              llvm::ConstantInt::getNullValue(operand->getType()), operand, "",
              /*HasNUW=*/false,
              /*HasNSW=*/SignedOverflowIsUB));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntUNegate: {
      // Lower `-x` as `0 - x`.
      auto* operand = context.GetValue(arg_ids[0]);
      context.SetLocal(
          inst_id,
          context.builder().CreateSub(
              llvm::ConstantInt::getNullValue(operand->getType()), operand));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntComplement: {
      // Lower `^x` as `-1 ^ x`.
      auto* operand = context.GetValue(arg_ids[0]);
      context.SetLocal(
          inst_id,
          context.builder().CreateXor(
              llvm::ConstantInt::getSigned(operand->getType(), -1), operand));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntSAdd: {
      context.SetLocal(
          inst_id, context.builder().CreateAdd(context.GetValue(arg_ids[0]),
                                               context.GetValue(arg_ids[1]), "",
                                               /*HasNUW=*/false,
                                               /*HasNSW=*/SignedOverflowIsUB));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntSSub: {
      context.SetLocal(
          inst_id, context.builder().CreateSub(context.GetValue(arg_ids[0]),
                                               context.GetValue(arg_ids[1]), "",
                                               /*HasNUW=*/false,
                                               /*HasNSW=*/SignedOverflowIsUB));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntSMul: {
      context.SetLocal(
          inst_id, context.builder().CreateMul(context.GetValue(arg_ids[0]),
                                               context.GetValue(arg_ids[1]), "",
                                               /*HasNUW=*/false,
                                               /*HasNSW=*/SignedOverflowIsUB));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntSDiv: {
      context.SetLocal(
          inst_id, context.builder().CreateSDiv(context.GetValue(arg_ids[0]),
                                                context.GetValue(arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntSMod: {
      context.SetLocal(
          inst_id, context.builder().CreateSRem(context.GetValue(arg_ids[0]),
                                                context.GetValue(arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntUAdd: {
      context.SetLocal(
          inst_id, context.builder().CreateAdd(context.GetValue(arg_ids[0]),
                                               context.GetValue(arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntUSub: {
      context.SetLocal(
          inst_id, context.builder().CreateSub(context.GetValue(arg_ids[0]),
                                               context.GetValue(arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntUMul: {
      context.SetLocal(
          inst_id, context.builder().CreateMul(context.GetValue(arg_ids[0]),
                                               context.GetValue(arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntUDiv: {
      context.SetLocal(
          inst_id, context.builder().CreateUDiv(context.GetValue(arg_ids[0]),
                                                context.GetValue(arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntUMod: {
      context.SetLocal(
          inst_id, context.builder().CreateURem(context.GetValue(arg_ids[0]),
                                                context.GetValue(arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntAnd: {
      context.SetLocal(
          inst_id, context.builder().CreateAnd(context.GetValue(arg_ids[0]),
                                               context.GetValue(arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntOr: {
      context.SetLocal(
          inst_id, context.builder().CreateOr(context.GetValue(arg_ids[0]),
                                              context.GetValue(arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntXor: {
      context.SetLocal(
          inst_id, context.builder().CreateXor(context.GetValue(arg_ids[0]),
                                               context.GetValue(arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntLeftShift: {
      context.SetLocal(
          inst_id, context.builder().CreateShl(context.GetValue(arg_ids[0]),
                                               context.GetValue(arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntRightShift: {
      context.SetLocal(
          inst_id,
          IsSignedInt(context, inst_id)
              ? context.builder().CreateAShr(context.GetValue(arg_ids[0]),
                                             context.GetValue(arg_ids[1]))
              : context.builder().CreateLShr(context.GetValue(arg_ids[0]),
                                             context.GetValue(arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntEq:
    case SemIR::BuiltinFunctionKind::IntNeq:
    case SemIR::BuiltinFunctionKind::IntLess:
    case SemIR::BuiltinFunctionKind::IntLessEq:
    case SemIR::BuiltinFunctionKind::IntGreater:
    case SemIR::BuiltinFunctionKind::IntGreaterEq: {
      context.SetLocal(
          inst_id,
          context.builder().CreateICmp(
              GetBuiltinICmpPredicate(builtin_kind,
                                      IsSignedInt(context, arg_ids[0])),
              context.GetValue(arg_ids[0]), context.GetValue(arg_ids[1])));
      return;
    }
  }

  CARBON_FATAL() << "Unsupported builtin call.";
}

auto HandleCall(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::Call inst) -> void {
  llvm::ArrayRef<SemIR::InstId> arg_ids =
      context.sem_ir().inst_blocks().Get(inst.args_id);

  auto* callee_value = context.GetValue(inst.callee_id);

  // A null callee pointer value indicates this isn't a real function.
  if (!callee_value) {
    auto builtin_kind =
        SemIR::BuiltinFunctionKind::ForCallee(context.sem_ir(), inst.callee_id);
    HandleBuiltinCall(context, inst_id, builtin_kind, arg_ids);
    return;
  }

  auto* callee = llvm::cast<llvm::Function>(callee_value);

  std::vector<llvm::Value*> args;

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

  context.SetLocal(inst_id, context.builder().CreateCall(callee, args));
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
  FatalErrorIfEncountered(inst);
}

auto HandleImplDecl(FunctionContext& /*context*/, SemIR::InstId /*inst_id*/,
                    SemIR::ImplDecl inst) -> void {
  FatalErrorIfEncountered(inst);
}

auto HandleImportRefUnloaded(FunctionContext& /*context*/,
                             SemIR::InstId /*inst_id*/,
                             SemIR::ImportRefUnloaded inst) -> void {
  FatalErrorIfEncountered(inst);
}

auto HandleImportRefLoaded(FunctionContext& /*context*/,
                           SemIR::InstId /*inst_id*/,
                           SemIR::ImportRefLoaded inst) -> void {
  FatalErrorIfEncountered(inst);
}

auto HandleImportRefUsed(FunctionContext& /*context*/,
                         SemIR::InstId /*inst_id*/, SemIR::ImportRefUsed inst)
    -> void {
  FatalErrorIfEncountered(inst);
}

auto HandleInitializeFrom(FunctionContext& context, SemIR::InstId /*inst_id*/,
                          SemIR::InitializeFrom inst) -> void {
  auto storage_type_id = context.sem_ir().insts().Get(inst.dest_id).type_id();
  context.FinishInit(storage_type_id, inst.dest_id, inst.src_id);
}

auto HandleInterfaceDecl(FunctionContext& /*context*/,
                         SemIR::InstId /*inst_id*/, SemIR::InterfaceDecl inst)
    -> void {
  FatalErrorIfEncountered(inst);
}

auto HandleInterfaceWitness(FunctionContext& /*context*/,
                            SemIR::InstId /*inst_id*/,
                            SemIR::InterfaceWitness inst) -> void {
  FatalErrorIfEncountered(inst);
}

auto HandleInterfaceWitnessAccess(FunctionContext& context,
                                  SemIR::InstId inst_id,
                                  SemIR::InterfaceWitnessAccess inst) -> void {
  // TODO: Add general constant lowering.
  auto const_id = context.sem_ir().constant_values().Get(inst_id);
  CARBON_CHECK(const_id.is_constant())
      << "Lowering non-constant witness access " << inst;
  context.SetLocal(inst_id, context.GetValue(const_id.inst_id()));
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
  FatalErrorIfEncountered(inst);
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
  context.SetLocal(inst_id,
                   context.builder().CreateAlloca(context.GetType(inst.type_id),
                                                  /*ArraySize=*/nullptr));
}

}  // namespace Carbon::Lower
