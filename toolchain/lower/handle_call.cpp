// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <vector>

#include "common/raw_string_ostream.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "toolchain/lower/function_context.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/function.h"
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

// Returns whether the specified instruction has a signed integer type.
static auto IsSignedInt(FunctionContext& context, SemIR::InstId int_id)
    -> bool {
  auto [file, type_id] = context.GetTypeIdOfInst(int_id);
  return file->types().IsSignedInt(type_id);
}

// Creates a zext or sext instruction depending on the signedness of the
// operand.
static auto CreateExt(FunctionContext& context, llvm::Value* value,
                      llvm::Type* type, bool is_signed,
                      const llvm::Twine& name = "") -> llvm::Value* {
  return is_signed ? context.builder().CreateSExt(value, type, name)
                   : context.builder().CreateZExt(value, type, name);
}

// Creates a zext, sext, or trunc instruction depending on the signedness of the
// operand.
static auto CreateExtOrTrunc(FunctionContext& context, llvm::Value* value,
                             llvm::Type* type, bool is_signed,
                             const llvm::Twine& name = "") -> llvm::Value* {
  return is_signed ? context.builder().CreateSExtOrTrunc(value, type, name)
                   : context.builder().CreateZExtOrTrunc(value, type, name);
}

// Create a integer bit shift for a call to a builtin bit shift function.
static auto CreateIntShift(FunctionContext& context,
                           llvm::Instruction::BinaryOps bin_op,
                           llvm::Value* lhs, llvm::Value* rhs) -> llvm::Value* {
  // Weirdly, LLVM requires the operands of bit shift operators to be of the
  // same type. We can always use the width of the LHS, because if the RHS
  // doesn't fit in that then the cast is out of range anyway. Zero-extending is
  // always fine because it's an error for the RHS to be negative.
  //
  // TODO: In a development build we should trap if the RHS is signed and
  // negative or greater than or equal to the number of bits in the left-hand
  // type.
  rhs = context.builder().CreateZExtOrTrunc(rhs, lhs->getType(), "rhs");
  return context.builder().CreateBinOp(bin_op, lhs, rhs);
}

// Handles a call to a builtin integer comparison operator.
static auto HandleIntComparison(FunctionContext& context, SemIR::InstId inst_id,
                                SemIR::BuiltinFunctionKind builtin_kind,
                                SemIR::InstId lhs_id, SemIR::InstId rhs_id)
    -> void {
  llvm::Value* lhs = context.GetValue(lhs_id);
  llvm::Value* rhs = context.GetValue(rhs_id);
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
  lhs = CreateExt(context, lhs, cmp_type, lhs_signed, "lhs");
  rhs = CreateExt(context, rhs, cmp_type, rhs_signed, "rhs");

  context.SetLocal(
      inst_id,
      context.builder().CreateICmp(
          GetBuiltinICmpPredicate(builtin_kind, cmp_signed), lhs, rhs));
}

// Creates a binary operator for a call to a builtin for either that operation
// or the corresponding compound assignment.
static auto CreateBinaryOperatorForBuiltin(
    FunctionContext& context, SemIR::InstId inst_id,
    SemIR::BuiltinFunctionKind builtin_kind, llvm::Value* lhs, llvm::Value* rhs)
    -> llvm::Value* {
  // TODO: Consider setting this to true in the performance build mode if the
  // result type is a signed integer type.
  constexpr bool SignedOverflowIsUB = false;

  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::IntSAdd:
    case SemIR::BuiltinFunctionKind::IntSAddAssign: {
      return context.builder().CreateAdd(lhs, rhs, "",
                                         /*HasNUW=*/false,
                                         /*HasNSW=*/SignedOverflowIsUB);
    }
    case SemIR::BuiltinFunctionKind::IntSSub:
    case SemIR::BuiltinFunctionKind::IntSSubAssign: {
      return context.builder().CreateSub(lhs, rhs, "",
                                         /*HasNUW=*/false,
                                         /*HasNSW=*/SignedOverflowIsUB);
    }
    case SemIR::BuiltinFunctionKind::IntSMul:
    case SemIR::BuiltinFunctionKind::IntSMulAssign: {
      return context.builder().CreateMul(lhs, rhs, "",
                                         /*HasNUW=*/false,
                                         /*HasNSW=*/SignedOverflowIsUB);
    }
    case SemIR::BuiltinFunctionKind::IntSDiv:
    case SemIR::BuiltinFunctionKind::IntSDivAssign: {
      return context.builder().CreateSDiv(lhs, rhs);
    }
    case SemIR::BuiltinFunctionKind::IntSMod:
    case SemIR::BuiltinFunctionKind::IntSModAssign: {
      return context.builder().CreateSRem(lhs, rhs);
    }
    case SemIR::BuiltinFunctionKind::IntUAdd:
    case SemIR::BuiltinFunctionKind::IntUAddAssign: {
      return context.builder().CreateAdd(lhs, rhs);
    }
    case SemIR::BuiltinFunctionKind::IntUSub:
    case SemIR::BuiltinFunctionKind::IntUSubAssign: {
      return context.builder().CreateSub(lhs, rhs);
    }
    case SemIR::BuiltinFunctionKind::IntUMul:
    case SemIR::BuiltinFunctionKind::IntUMulAssign: {
      return context.builder().CreateMul(lhs, rhs);
    }
    case SemIR::BuiltinFunctionKind::IntUDiv:
    case SemIR::BuiltinFunctionKind::IntUDivAssign: {
      return context.builder().CreateUDiv(lhs, rhs);
    }
    case SemIR::BuiltinFunctionKind::IntUMod:
    case SemIR::BuiltinFunctionKind::IntUModAssign: {
      return context.builder().CreateURem(lhs, rhs);
    }
    case SemIR::BuiltinFunctionKind::IntAnd:
    case SemIR::BuiltinFunctionKind::IntAndAssign: {
      return context.builder().CreateAnd(lhs, rhs);
    }
    case SemIR::BuiltinFunctionKind::IntOr:
    case SemIR::BuiltinFunctionKind::IntOrAssign: {
      return context.builder().CreateOr(lhs, rhs);
    }
    case SemIR::BuiltinFunctionKind::IntXor:
    case SemIR::BuiltinFunctionKind::IntXorAssign: {
      return context.builder().CreateXor(lhs, rhs);
    }
    case SemIR::BuiltinFunctionKind::IntLeftShift:
    case SemIR::BuiltinFunctionKind::IntLeftShiftAssign: {
      return CreateIntShift(context, llvm::Instruction::Shl, lhs, rhs);
    }
    case SemIR::BuiltinFunctionKind::IntRightShift:
    case SemIR::BuiltinFunctionKind::IntRightShiftAssign: {
      // TODO: Split each of these builtins into separate signed and unsigned
      // builtins rather than working out here whether we're performing an
      // arithmetic or logical shift.
      auto lhs_id = context.sem_ir().inst_blocks().Get(
          context.sem_ir().insts().GetAs<SemIR::Call>(inst_id).args_id)[0];
      auto [lhs_type_file, lhs_type_id] = context.GetTypeIdOfInst(lhs_id);
      if (builtin_kind == SemIR::BuiltinFunctionKind::IntRightShiftAssign) {
        lhs_type_id = lhs_type_file->GetPointeeType(lhs_type_id);
      }
      return CreateIntShift(context,
                            lhs_type_file->types().IsSignedInt(lhs_type_id)
                                ? llvm::Instruction::AShr
                                : llvm::Instruction::LShr,
                            lhs, rhs);
    }
    case SemIR::BuiltinFunctionKind::FloatAdd:
    case SemIR::BuiltinFunctionKind::FloatAddAssign: {
      return context.builder().CreateFAdd(lhs, rhs);
    }
    case SemIR::BuiltinFunctionKind::FloatSub:
    case SemIR::BuiltinFunctionKind::FloatSubAssign: {
      return context.builder().CreateFSub(lhs, rhs);
    }
    case SemIR::BuiltinFunctionKind::FloatMul:
    case SemIR::BuiltinFunctionKind::FloatMulAssign: {
      return context.builder().CreateFMul(lhs, rhs);
    }
    case SemIR::BuiltinFunctionKind::FloatDiv:
    case SemIR::BuiltinFunctionKind::FloatDivAssign: {
      return context.builder().CreateFDiv(lhs, rhs);
    }
    default: {
      CARBON_FATAL("Unexpected binary operator {0}", builtin_kind);
    }
  }
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

    case SemIR::BuiltinFunctionKind::NoOp:
      return;

    case SemIR::BuiltinFunctionKind::PrimitiveCopy:
      context.SetLocal(inst_id, context.GetValue(arg_ids[0]));
      return;

    case SemIR::BuiltinFunctionKind::PrintChar: {
      auto* i32_type = llvm::IntegerType::getInt32Ty(context.llvm_context());
      llvm::Value* arg_value = context.builder().CreateSExtOrTrunc(
          context.GetValue(arg_ids[0]), i32_type);
      auto putchar = context.llvm_module().getOrInsertFunction(
          "putchar", i32_type, i32_type);
      auto* result = context.builder().CreateCall(putchar, {arg_value});
      context.SetLocal(inst_id, context.builder().CreateSExtOrTrunc(
                                    result, context.GetTypeOfInst(inst_id)));
      return;
    }

    case SemIR::BuiltinFunctionKind::PrintInt: {
      auto* i32_type = llvm::IntegerType::getInt32Ty(context.llvm_context());
      auto* ptr_type = llvm::PointerType::get(context.llvm_context(), 0);
      auto* printf_type = llvm::FunctionType::get(i32_type, {ptr_type},
                                                  /*isVarArg=*/true);
      llvm::FunctionCallee printf =
          context.llvm_module().getOrInsertFunction("printf", printf_type);

      llvm::Value* format_string = context.printf_int_format_string();
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
      context.SetLocal(inst_id, context.builder().CreateSExtOrTrunc(
                                    result, context.GetTypeOfInst(inst_id)));
      return;
    }

    case SemIR::BuiltinFunctionKind::TypeAnd: {
      context.SetLocal(inst_id, context.GetTypeAsValue());
      return;
    }

    case SemIR::BuiltinFunctionKind::BoolMakeType:
    case SemIR::BuiltinFunctionKind::CharLiteralMakeType:
    case SemIR::BuiltinFunctionKind::FloatLiteralMakeType:
    case SemIR::BuiltinFunctionKind::FloatMakeType:
    case SemIR::BuiltinFunctionKind::IntLiteralMakeType:
    case SemIR::BuiltinFunctionKind::IntMakeTypeSigned:
    case SemIR::BuiltinFunctionKind::IntMakeTypeUnsigned:
    case SemIR::BuiltinFunctionKind::MaybeUnformedMakeType:
      context.SetLocal(inst_id, context.GetTypeAsValue());
      return;

    case SemIR::BuiltinFunctionKind::IntConvert: {
      context.SetLocal(inst_id,
                       CreateExtOrTrunc(context, context.GetValue(arg_ids[0]),
                                        context.GetTypeOfInst(inst_id),
                                        IsSignedInt(context, arg_ids[0])));
      return;
    }

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
    case SemIR::BuiltinFunctionKind::IntSAdd:
    case SemIR::BuiltinFunctionKind::IntSSub:
    case SemIR::BuiltinFunctionKind::IntSMul:
    case SemIR::BuiltinFunctionKind::IntSDiv:
    case SemIR::BuiltinFunctionKind::IntSMod:
    case SemIR::BuiltinFunctionKind::IntUAdd:
    case SemIR::BuiltinFunctionKind::IntUSub:
    case SemIR::BuiltinFunctionKind::IntUMul:
    case SemIR::BuiltinFunctionKind::IntUDiv:
    case SemIR::BuiltinFunctionKind::IntUMod:
    case SemIR::BuiltinFunctionKind::IntAnd:
    case SemIR::BuiltinFunctionKind::IntOr:
    case SemIR::BuiltinFunctionKind::IntXor:
    case SemIR::BuiltinFunctionKind::IntLeftShift:
    case SemIR::BuiltinFunctionKind::IntRightShift:
    case SemIR::BuiltinFunctionKind::FloatAdd:
    case SemIR::BuiltinFunctionKind::FloatSub:
    case SemIR::BuiltinFunctionKind::FloatMul:
    case SemIR::BuiltinFunctionKind::FloatDiv: {
      context.SetLocal(inst_id, CreateBinaryOperatorForBuiltin(
                                    context, inst_id, builtin_kind,
                                    context.GetValue(arg_ids[0]),
                                    context.GetValue(arg_ids[1])));
      return;
    }
    case SemIR::BuiltinFunctionKind::IntSAddAssign:
    case SemIR::BuiltinFunctionKind::IntSSubAssign:
    case SemIR::BuiltinFunctionKind::IntSMulAssign:
    case SemIR::BuiltinFunctionKind::IntSDivAssign:
    case SemIR::BuiltinFunctionKind::IntSModAssign:
    case SemIR::BuiltinFunctionKind::IntUAddAssign:
    case SemIR::BuiltinFunctionKind::IntUSubAssign:
    case SemIR::BuiltinFunctionKind::IntUMulAssign:
    case SemIR::BuiltinFunctionKind::IntUDivAssign:
    case SemIR::BuiltinFunctionKind::IntUModAssign:
    case SemIR::BuiltinFunctionKind::IntAndAssign:
    case SemIR::BuiltinFunctionKind::IntOrAssign:
    case SemIR::BuiltinFunctionKind::IntXorAssign:
    case SemIR::BuiltinFunctionKind::IntLeftShiftAssign:
    case SemIR::BuiltinFunctionKind::IntRightShiftAssign:
    case SemIR::BuiltinFunctionKind::FloatAddAssign:
    case SemIR::BuiltinFunctionKind::FloatSubAssign:
    case SemIR::BuiltinFunctionKind::FloatMulAssign:
    case SemIR::BuiltinFunctionKind::FloatDivAssign: {
      auto* lhs_ptr = context.GetValue(arg_ids[0]);
      auto lhs_type = context.GetTypeIdOfInst(arg_ids[0]);
      auto pointee_type = lhs_type.GetPointeeType();
      auto* lhs_value = context.LoadObject(pointee_type, lhs_ptr);
      auto* result = CreateBinaryOperatorForBuiltin(
          context, inst_id, builtin_kind, lhs_value,
          context.GetValue(arg_ids[1]));
      context.StoreObject(pointee_type, result, lhs_ptr);
      // TODO: Add a helper to get a "no value representation" value.
      context.SetLocal(inst_id,
                       llvm::PoisonValue::get(context.GetTypeOfInst(inst_id)));
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

    case SemIR::BuiltinFunctionKind::CharConvertChecked:
    case SemIR::BuiltinFunctionKind::FloatConvertChecked:
    case SemIR::BuiltinFunctionKind::IntConvertChecked:
    case SemIR::BuiltinFunctionKind::TypeCanDestroy: {
      // TODO: Check this statically.
      CARBON_CHECK(builtin_kind.IsCompTimeOnly(
          context.sem_ir(), arg_ids,
          context.sem_ir().insts().Get(inst_id).type_id()));
      CARBON_FATAL("Missing constant value for call to comptime-only function");
    }

    case SemIR::BuiltinFunctionKind::PointerMakeNull: {
      // MaybeUnformed(T*) has an in-place initializing representation, so an
      // out parameter will be passed.
      context.builder().CreateStore(
          llvm::ConstantPointerNull::get(
              llvm::PointerType::get(context.llvm_context(),
                                     /*AddressSpace=*/0)),
          context.GetValue(arg_ids[0]));
      context.SetLocal(inst_id,
                       llvm::PoisonValue::get(context.GetTypeOfInst(inst_id)));
      return;
    }

    case SemIR::BuiltinFunctionKind::PointerIsNull: {
      auto* ptr = context.builder().CreateLoad(
          context.GetTypeOfInst(arg_ids[0]), context.GetValue(arg_ids[0]));
      context.SetLocal(inst_id, context.builder().CreateIsNull(ptr));
      return;
    }

    case SemIR::BuiltinFunctionKind::TypeDestroy:
      // TODO: Destroy aggregate members.
      return;
  }

  CARBON_FATAL("Unsupported builtin call.");
}

static auto HandleVirtualCall(FunctionContext& context,
                              llvm::ArrayRef<llvm::Value*> args,
                              const SemIR::File* callee_file,
                              const SemIR::Function& function,
                              const SemIR::CalleeFunction& callee_function)
    -> llvm::CallInst* {
  CARBON_CHECK(!args.empty(),
               "Virtual functions must have at least one parameter");
  auto* ptr_type =
      llvm::PointerType::get(context.llvm_context(), /*AddressSpace=*/0);
  // The vtable pointer is always at the start of the object in the Carbon
  // ABI, so a pointer to the object is a pointer to the vtable pointer - load
  // that to get a pointer to the vtable.
  // TODO: Handle the case in C++ interop where the vtable pointer isn't at
  // the start of the object.
  // TODO: Use `context.LoadObject`.
  auto* vtable = context.builder().CreateLoad(ptr_type, args.front(), "vtable");
  auto* i32_type = llvm::IntegerType::getInt32Ty(context.llvm_context());
  auto* pointer_type =
      llvm::PointerType::get(context.llvm_context(), /* address space */ 0);
  auto function_type_info =
      context.GetFileContext(callee_file)
          .BuildFunctionTypeInfo(function,
                                 callee_function.resolved_specific_id);
  llvm::Value* virtual_fn;
  if (function.clang_decl_id.has_value()) {
    // Use absolute vtables for clang interop - the itanium vtable contains
    // function pointers.
    auto* virtual_function_pointer_address = context.builder().CreateGEP(
        pointer_type, vtable,
        {llvm::ConstantInt::get(
            i32_type, static_cast<uint64_t>(function.virtual_index))});
    virtual_fn = context.builder().CreateLoad(
        pointer_type, virtual_function_pointer_address, "memptr.virtualfn");
  } else {
    // For Carbon, use Relative VTables as pioneered by Fuchsia:
    // https://llvm.org/devmtg/2021-11/slides/2021-RelativeVTablesinC.pdf
    // In this case, the vtable contains an offset from the vtable itself to the
    // function in question. This avoids the use of link-time relocations in the
    // vtable (making object files smaller, improving link time) - at the cost
    // of extra instructions to resolve the offset at the call-site.
    // This uses the `llvm.load.relative` intrinsic (
    // https://llvm.org/docs/LangRef.html#llvm-load-relative-intrinsic ) that
    // essentially does the arithmetic in one-shot: ptr + *(ptr + offset)
    virtual_fn = context.builder().CreateCall(
        llvm::Intrinsic::getOrInsertDeclaration(
            &context.llvm_module(), llvm::Intrinsic::load_relative, {i32_type}),
        {vtable,
         llvm::ConstantInt::get(
             i32_type, static_cast<uint64_t>(function.virtual_index) * 4)});
  }
  return context.builder().CreateCall(function_type_info.type, virtual_fn,
                                      args);
}
auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
                SemIR::Call inst) -> void {
  llvm::ArrayRef<SemIR::InstId> arg_ids =
      context.sem_ir().inst_blocks().Get(inst.args_id);

  // TODO: This duplicates the SpecificId handling in `GetCallee`.

  // TODO: Should the `bound_method` be removed when forming the `call`
  // instruction? The `self` parameter is transferred into the call argument
  // list.
  auto callee_id = inst.callee_id;
  if (auto bound_method =
          context.sem_ir().insts().TryGetAs<SemIR::BoundMethod>(callee_id)) {
    callee_id = bound_method->function_decl_id;
  }

  // Map to the callee in the specific. This might be in a different file than
  // the one we're currently lowering.
  const auto* callee_file = &context.sem_ir();
  if (context.specific_id().has_value()) {
    auto [const_file, const_id] = GetConstantValueInSpecific(
        context.specific_sem_ir(), context.specific_id(), context.sem_ir(),
        callee_id);
    callee_file = const_file;
    callee_id = const_file->constant_values().GetInstIdIfValid(const_id);
    CARBON_CHECK(callee_id.has_value());
  }

  auto callee_function = SemIR::GetCalleeAsFunction(*callee_file, callee_id);

  const SemIR::Function& function =
      callee_file->functions().Get(callee_function.function_id);
  context.AddCallToCurrentFingerprint(callee_file->check_ir_id(),
                                      callee_function.function_id,
                                      callee_function.resolved_specific_id);

  if (auto builtin_kind = function.builtin_function_kind();
      builtin_kind != SemIR::BuiltinFunctionKind::None) {
    HandleBuiltinCall(context, inst_id, builtin_kind, arg_ids);
    return;
  }

  std::vector<llvm::Value*> args;

  auto inst_type = context.GetTypeIdOfInst(inst_id);
  bool call_has_return_slot =
      SemIR::ReturnTypeInfo::ForType(context.sem_ir(), inst.type_id)
          .has_return_slot();
  if (context.GetReturnTypeInfo(inst_type).info.has_return_slot()) {
    CARBON_CHECK(call_has_return_slot);
    args.push_back(context.GetValue(arg_ids.consume_back()));
  } else if (call_has_return_slot) {
    // Call instruction has a return slot but this specific callee does not.
    // Just ignore it.
    arg_ids.consume_back();
  }

  for (auto arg_id : arg_ids) {
    auto arg_type = context.GetTypeIdOfInst(arg_id);
    if (context.GetValueRepr(arg_type).repr.kind != SemIR::ValueRepr::None) {
      args.push_back(context.GetValue(arg_id));
    }
  }

  llvm::CallInst* call;
  if (function.virtual_modifier == SemIR::Function::VirtualModifier::None) {
    auto* callee =
        context.GetFileContext(callee_file)
            .GetOrCreateFunction(callee_function.function_id,
                                 callee_function.resolved_specific_id);
    auto describe_call = [&] {
      RawStringOstream out;
      out << "call ";
      callee->printAsOperand(out);
      out << "(";
      llvm::ListSeparator sep;
      for (auto* arg : args) {
        out << sep;
        arg->printAsOperand(out);
      }
      out << ")\n";
      callee->print(out);
      return out.TakeStr();
    };
    CARBON_CHECK(callee->arg_size() == args.size(),
                 "Argument count mismatch: {0}", describe_call());
    call = context.builder().CreateCall(callee, args);
  } else {
    call = HandleVirtualCall(context, args, callee_file, function,
                             callee_function);
  }

  context.SetLocal(inst_id, call);
}

}  // namespace Carbon::Lower
