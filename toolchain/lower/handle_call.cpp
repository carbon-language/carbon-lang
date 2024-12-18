// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "toolchain/lower/function_context.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Lower {

// Get the predicate to use for an `icmp` instruction generated for the
// specified builtin.
static auto GetBuiltinICmpPredicate(SemIR::BuiltinFunctionKind builtin_kind,
                                    bool is_signed)
    -> llvm::CmpInst::Predicate {
  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::IntEq:
    case SemIR::BuiltinFunctionKind::BoolEq:
      return llvm::CmpInst::ICMP_EQ;
    case SemIR::BuiltinFunctionKind::IntNeq:
    case SemIR::BuiltinFunctionKind::BoolNeq:
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
      CARBON_FATAL("Unexpected builtin kind {0}", builtin_kind);
  }
}

// Get the predicate to use for an `fcmp` instruction generated for the
// specified builtin.
static auto GetBuiltinFCmpPredicate(SemIR::BuiltinFunctionKind builtin_kind)
    -> llvm::CmpInst::Predicate {
  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::FloatEq:
      return llvm::CmpInst::FCMP_OEQ;
    case SemIR::BuiltinFunctionKind::FloatNeq:
      return llvm::CmpInst::FCMP_ONE;
    case SemIR::BuiltinFunctionKind::FloatLess:
      return llvm::CmpInst::FCMP_OLT;
    case SemIR::BuiltinFunctionKind::FloatLessEq:
      return llvm::CmpInst::FCMP_OLE;
    case SemIR::BuiltinFunctionKind::FloatGreater:
      return llvm::CmpInst::FCMP_OGT;
    case SemIR::BuiltinFunctionKind::FloatGreaterEq:
      return llvm::CmpInst::FCMP_OGE;
    default:
      CARBON_FATAL("Unexpected builtin kind {0}", builtin_kind);
  }
}

// Given an argument that is either a sized integer or integer literal, produces
// a corresponding value.
static auto GetIntOrIntLiteralValue(FunctionContext& context,
                                    SemIR::InstId arg_id) -> llvm::Value* {
  auto* result = context.GetValue(arg_id);

  // An IntLiteral value will have been lowered to an empty struct. Produce a
  // ConstantInt with its constant value instead.
  if (isa<llvm::ConstantStruct>(result)) {
    CARBON_CHECK(result == context.GetIntLiteralAsValue());
    auto int_value = context.sem_ir().insts().GetAs<SemIR::IntValue>(
        context.sem_ir().constant_values().GetConstantInstId(arg_id));
    return llvm::ConstantInt::get(
        context.llvm_context(), context.sem_ir().ints().Get(int_value.int_id));
  }

  return result;
}

// Returns whether the specified instruction has a signed integer type.
static auto IsSignedInt(FunctionContext& context, SemIR::InstId int_id)
    -> bool {
  return context.sem_ir().types().IsSignedInt(
      context.sem_ir().insts().Get(int_id).type_id());
}

static auto HandleIntComparison(FunctionContext& context, SemIR::InstId inst_id,
                                SemIR::BuiltinFunctionKind builtin_kind,
                                SemIR::InstId lhs_id, SemIR::InstId rhs_id)
    -> void {
  llvm::Value* lhs = GetIntOrIntLiteralValue(context, lhs_id);
  llvm::Value* rhs = GetIntOrIntLiteralValue(context, rhs_id);
  const auto* lhs_type = cast<llvm::IntegerType>(lhs->getType());
  const auto* rhs_type = cast<llvm::IntegerType>(rhs->getType());

  // We perform a signed comparison if either operand is signed.
  bool lhs_signed = IsSignedInt(context, lhs_id);
  bool rhs_signed = IsSignedInt(context, rhs_id);
  bool cmp_signed = lhs_signed || rhs_signed;

  // Compute the width for the comparison. This is the smallest width that
  // fits both types, after widening them to include a sign bit if
  // necessary.
  auto width_for_cmp = [&](const llvm::IntegerType* type, bool is_signed) {
    unsigned width = type->getBitWidth();
    if (!is_signed && cmp_signed) {
      // We're performing a signed comparison but this input is unsigned.
      // Widen it by at least one bit to provide a sign bit.
      ++width;
    }
    return width;
  };
  // TODO: This might be an awkward size, such as 33 or 65 bits, for a
  // signed/unsigned comparison. Would it be better to round this up to a
  // "nicer" bit width?
  unsigned cmp_width = std::max(width_for_cmp(lhs_type, lhs_signed),
                                width_for_cmp(rhs_type, rhs_signed));
  auto* cmp_type = llvm::IntegerType::get(context.llvm_context(), cmp_width);

  // Widen the operands as needed.
  auto zext_or_sext = [&](llvm::Value* value, bool is_signed) {
    return is_signed ? context.builder().CreateSExt(value, cmp_type)
                     : context.builder().CreateZExt(value, cmp_type);
  };
  lhs = zext_or_sext(lhs, lhs_signed);
  rhs = zext_or_sext(rhs, rhs_signed);

  context.SetLocal(
      inst_id,
      context.builder().CreateICmp(
          GetBuiltinICmpPredicate(builtin_kind, cmp_signed), lhs, rhs));
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
      CARBON_FATAL("No callee in function call.");

    case SemIR::BuiltinFunctionKind::PrintChar: {
      auto* i32_type = llvm::IntegerType::getInt32Ty(context.llvm_context());
      llvm::Value* arg_value = context.builder().CreateSExtOrTrunc(
          context.GetValue(arg_ids[0]), i32_type);
      auto putchar = context.llvm_module().getOrInsertFunction(
          "putchar", i32_type, i32_type);
      auto* result = context.builder().CreateCall(putchar, {arg_value});
      context.SetLocal(
          inst_id,
          context.builder().CreateSExtOrTrunc(
              result, context.GetType(
                          context.sem_ir().insts().Get(inst_id).type_id())));
      return;
    }

    case SemIR::BuiltinFunctionKind::PrintInt: {
      auto* i32_type = llvm::IntegerType::getInt32Ty(context.llvm_context());
      auto* ptr_type = llvm::PointerType::get(context.llvm_context(), 0);
      auto* printf_type = llvm::FunctionType::get(i32_type, {ptr_type},
                                                  /*isVarArg=*/true);
      llvm::FunctionCallee printf =
          context.llvm_module().getOrInsertFunction("printf", printf_type);

      llvm::Value* format_string =
          context.builder().CreateGlobalString("%d\n", "printf.int.format");
      llvm::Value* arg_value = context.builder().CreateSExtOrTrunc(
          context.GetValue(arg_ids[0]), i32_type);
      context.SetLocal(inst_id, context.builder().CreateCall(
                                    printf, {format_string, arg_value}));
      return;
    }

    case SemIR::BuiltinFunctionKind::ReadChar: {
      auto* i32_type = llvm::IntegerType::getInt32Ty(context.llvm_context());
      auto getchar =
          context.llvm_module().getOrInsertFunction("getchar", i32_type);
      auto* result = context.builder().CreateCall(getchar, {});
      context.SetLocal(
          inst_id,
          context.builder().CreateSExtOrTrunc(
              result, context.GetType(
                          context.sem_ir().insts().Get(inst_id).type_id())));
      return;
    }

    case SemIR::BuiltinFunctionKind::BoolMakeType:
    case SemIR::BuiltinFunctionKind::FloatMakeType:
    case SemIR::BuiltinFunctionKind::IntLiteralMakeType:
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
      context.SetLocal(inst_id,
                       context.builder().CreateShl(
                           context.GetValue(arg_ids[0]),
                           GetIntOrIntLiteralValue(context, arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntRightShift: {
      context.SetLocal(inst_id,
                       IsSignedInt(context, inst_id)
                           ? context.builder().CreateAShr(
                                 context.GetValue(arg_ids[0]),
                                 GetIntOrIntLiteralValue(context, arg_ids[1]))
                           : context.builder().CreateLShr(
                                 context.GetValue(arg_ids[0]),
                                 GetIntOrIntLiteralValue(context, arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntEq:
    case SemIR::BuiltinFunctionKind::IntNeq:
    case SemIR::BuiltinFunctionKind::IntLess:
    case SemIR::BuiltinFunctionKind::IntLessEq:
    case SemIR::BuiltinFunctionKind::IntGreater:
    case SemIR::BuiltinFunctionKind::IntGreaterEq:
    case SemIR::BuiltinFunctionKind::BoolEq:
    case SemIR::BuiltinFunctionKind::BoolNeq: {
      HandleIntComparison(context, inst_id, builtin_kind, arg_ids[0],
                          arg_ids[1]);
      return;
    }
    case SemIR::BuiltinFunctionKind::FloatNegate: {
      context.SetLocal(
          inst_id, context.builder().CreateFNeg(context.GetValue(arg_ids[0])));
      return;
    }
    case SemIR::BuiltinFunctionKind::FloatAdd: {
      context.SetLocal(
          inst_id, context.builder().CreateFAdd(context.GetValue(arg_ids[0]),
                                                context.GetValue(arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::FloatSub: {
      context.SetLocal(
          inst_id, context.builder().CreateFSub(context.GetValue(arg_ids[0]),
                                                context.GetValue(arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::FloatMul: {
      context.SetLocal(
          inst_id, context.builder().CreateFMul(context.GetValue(arg_ids[0]),
                                                context.GetValue(arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::FloatDiv: {
      context.SetLocal(
          inst_id, context.builder().CreateFDiv(context.GetValue(arg_ids[0]),
                                                context.GetValue(arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::FloatEq:
    case SemIR::BuiltinFunctionKind::FloatNeq:
    case SemIR::BuiltinFunctionKind::FloatLess:
    case SemIR::BuiltinFunctionKind::FloatLessEq:
    case SemIR::BuiltinFunctionKind::FloatGreater:
    case SemIR::BuiltinFunctionKind::FloatGreaterEq: {
      context.SetLocal(inst_id, context.builder().CreateFCmp(
                                    GetBuiltinFCmpPredicate(builtin_kind),
                                    context.GetValue(arg_ids[0]),
                                    context.GetValue(arg_ids[1])));
      return;
    }

    case SemIR::BuiltinFunctionKind::IntConvertChecked: {
      // TODO: Check this statically.
      CARBON_CHECK(builtin_kind.IsCompTimeOnly(
          context.sem_ir(), context.sem_ir().insts().Get(inst_id).type_id()));
      CARBON_FATAL("Missing constant value for call to comptime-only function");
    }
  }

  CARBON_FATAL("Unsupported builtin call.");
}

auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::Call inst) -> void {
  llvm::ArrayRef<SemIR::InstId> arg_ids =
      context.sem_ir().inst_blocks().Get(inst.args_id);

  auto callee_function =
      SemIR::GetCalleeFunction(context.sem_ir(), inst.callee_id);
  CARBON_CHECK(callee_function.function_id.is_valid());

  if (auto builtin_kind = context.sem_ir()
                              .functions()
                              .Get(callee_function.function_id)
                              .builtin_function_kind;
      builtin_kind != SemIR::BuiltinFunctionKind::None) {
    HandleBuiltinCall(context, inst_id, builtin_kind, arg_ids);
    return;
  }

  auto* callee = context.GetOrCreateFunction(
      callee_function.function_id, callee_function.resolved_specific_id);

  std::vector<llvm::Value*> args;

  if (SemIR::ReturnTypeInfo::ForType(context.sem_ir(), inst.type_id)
          .has_return_slot()) {
    args.push_back(context.GetValue(arg_ids.back()));
    arg_ids = arg_ids.drop_back();
  }

  for (auto arg_id : arg_ids) {
    auto arg_type_id = context.sem_ir().insts().Get(arg_id).type_id();
    if (SemIR::ValueRepr::ForType(context.sem_ir(), arg_type_id).kind !=
        SemIR::ValueRepr::None) {
      args.push_back(context.GetValue(arg_id));
    }
  }

  context.SetLocal(inst_id, context.builder().CreateCall(callee, args));
}

}  // namespace Carbon::Lower
