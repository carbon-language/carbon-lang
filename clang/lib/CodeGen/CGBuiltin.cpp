//===---- CGBuiltin.cpp - Emit LLVM Code for builtins ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Builtin calls as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/Intrinsics.h"
using namespace clang;
using namespace CodeGen;
using namespace llvm;

/// Utility to insert an atomic instruction based on Instrinsic::ID
/// and the expression node.
static RValue EmitBinaryAtomic(CodeGenFunction& CGF, 
                               Intrinsic::ID Id, const CallExpr *E) {
  const llvm::Type *ResType[2];
  ResType[0] = CGF.ConvertType(E->getType());
  ResType[1] = CGF.ConvertType(E->getArg(0)->getType());
  Value *AtomF = CGF.CGM.getIntrinsic(Id, ResType, 2);
  return RValue::get(CGF.Builder.CreateCall2(AtomF, 
                                             CGF.EmitScalarExpr(E->getArg(0)), 
                                             CGF.EmitScalarExpr(E->getArg(1))));
}

/// Utility to insert an atomic instruction based Instrinsic::ID and
// the expression node, where the return value is the result of the
// operation.
static RValue EmitBinaryAtomicPost(CodeGenFunction& CGF, 
                                   Intrinsic::ID Id, const CallExpr *E,
                                   Instruction::BinaryOps Op) {
  const llvm::Type *ResType[2];
  ResType[0] = CGF.ConvertType(E->getType());
  ResType[1] = CGF.ConvertType(E->getArg(0)->getType());
  Value *AtomF = CGF.CGM.getIntrinsic(Id, ResType, 2);
  Value *Ptr = CGF.EmitScalarExpr(E->getArg(0));
  Value *Operand = CGF.EmitScalarExpr(E->getArg(1));
  Value *Result = CGF.Builder.CreateCall2(AtomF, Ptr, Operand);
  
  if (Id == Intrinsic::atomic_load_nand)
    Result = CGF.Builder.CreateNot(Result);
  
  
  return RValue::get(CGF.Builder.CreateBinOp(Op, Result, Operand));
}

RValue CodeGenFunction::EmitBuiltinExpr(const FunctionDecl *FD, 
                                        unsigned BuiltinID, const CallExpr *E) {
  // See if we can constant fold this builtin.  If so, don't emit it at all.
  Expr::EvalResult Result;
  if (E->Evaluate(Result, CGM.getContext())) {
    if (Result.Val.isInt())
      return RValue::get(llvm::ConstantInt::get(VMContext, 
                                                Result.Val.getInt()));
    else if (Result.Val.isFloat())
      return RValue::get(ConstantFP::get(VMContext, Result.Val.getFloat()));
  }
      
  switch (BuiltinID) {
  default: break;  // Handle intrinsics and libm functions below.
  case Builtin::BI__builtin___CFStringMakeConstantString:
    return RValue::get(CGM.EmitConstantExpr(E, E->getType(), 0));
  case Builtin::BI__builtin_stdarg_start:
  case Builtin::BI__builtin_va_start:
  case Builtin::BI__builtin_va_end: {
    Value *ArgValue = EmitVAListRef(E->getArg(0));
    const llvm::Type *DestType = 
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(VMContext));
    if (ArgValue->getType() != DestType)
      ArgValue = Builder.CreateBitCast(ArgValue, DestType, 
                                       ArgValue->getName().data());

    Intrinsic::ID inst = (BuiltinID == Builtin::BI__builtin_va_end) ? 
      Intrinsic::vaend : Intrinsic::vastart;
    return RValue::get(Builder.CreateCall(CGM.getIntrinsic(inst), ArgValue));
  }
  case Builtin::BI__builtin_va_copy: {
    Value *DstPtr = EmitVAListRef(E->getArg(0));
    Value *SrcPtr = EmitVAListRef(E->getArg(1));

    const llvm::Type *Type = 
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(VMContext));

    DstPtr = Builder.CreateBitCast(DstPtr, Type);
    SrcPtr = Builder.CreateBitCast(SrcPtr, Type);
    return RValue::get(Builder.CreateCall2(CGM.getIntrinsic(Intrinsic::vacopy), 
                                           DstPtr, SrcPtr));
  }
  case Builtin::BI__builtin_abs: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));   
    
    Value *NegOp = Builder.CreateNeg(ArgValue, "neg");
    Value *CmpResult = 
    Builder.CreateICmpSGE(ArgValue, 
                          llvm::Constant::getNullValue(ArgValue->getType()),
                                                            "abscond");
    Value *Result = 
      Builder.CreateSelect(CmpResult, ArgValue, NegOp, "abs");
    
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_ctz:
  case Builtin::BI__builtin_ctzl:
  case Builtin::BI__builtin_ctzll: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    
    const llvm::Type *ArgType = ArgValue->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::cttz, &ArgType, 1);

    const llvm::Type *ResultType = ConvertType(E->getType());    
    Value *Result = Builder.CreateCall(F, ArgValue, "tmp");
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_clz:
  case Builtin::BI__builtin_clzl:
  case Builtin::BI__builtin_clzll: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    
    const llvm::Type *ArgType = ArgValue->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::ctlz, &ArgType, 1);

    const llvm::Type *ResultType = ConvertType(E->getType());    
    Value *Result = Builder.CreateCall(F, ArgValue, "tmp");
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_ffs:
  case Builtin::BI__builtin_ffsl:
  case Builtin::BI__builtin_ffsll: {
    // ffs(x) -> x ? cttz(x) + 1 : 0
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    
    const llvm::Type *ArgType = ArgValue->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::cttz, &ArgType, 1);
        
    const llvm::Type *ResultType = ConvertType(E->getType());
    Value *Tmp = Builder.CreateAdd(Builder.CreateCall(F, ArgValue, "tmp"), 
                                   llvm::ConstantInt::get(ArgType, 1), "tmp");
    Value *Zero = llvm::Constant::getNullValue(ArgType);
    Value *IsZero = Builder.CreateICmpEQ(ArgValue, Zero, "iszero");
    Value *Result = Builder.CreateSelect(IsZero, Zero, Tmp, "ffs");
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_parity:
  case Builtin::BI__builtin_parityl:
  case Builtin::BI__builtin_parityll: {
    // parity(x) -> ctpop(x) & 1
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    
    const llvm::Type *ArgType = ArgValue->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::ctpop, &ArgType, 1);
    
    const llvm::Type *ResultType = ConvertType(E->getType());
    Value *Tmp = Builder.CreateCall(F, ArgValue, "tmp");
    Value *Result = Builder.CreateAnd(Tmp, llvm::ConstantInt::get(ArgType, 1), 
                                      "tmp");
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_popcount:
  case Builtin::BI__builtin_popcountl:
  case Builtin::BI__builtin_popcountll: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    
    const llvm::Type *ArgType = ArgValue->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::ctpop, &ArgType, 1);
    
    const llvm::Type *ResultType = ConvertType(E->getType());
    Value *Result = Builder.CreateCall(F, ArgValue, "tmp");
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_expect:
    // FIXME: pass expect through to LLVM
    return RValue::get(EmitScalarExpr(E->getArg(0)));
  case Builtin::BI__builtin_bswap32:
  case Builtin::BI__builtin_bswap64: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    const llvm::Type *ArgType = ArgValue->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::bswap, &ArgType, 1);
    return RValue::get(Builder.CreateCall(F, ArgValue, "tmp"));
  }    
  case Builtin::BI__builtin_object_size: {
    // FIXME: Implement. For now we just always fail and pretend we
    // don't know the object size.
    llvm::APSInt TypeArg = E->getArg(1)->EvaluateAsInt(CGM.getContext());
    const llvm::Type *ResType = ConvertType(E->getType());
    //    bool UseSubObject = TypeArg.getZExtValue() & 1;
    bool UseMinimum = TypeArg.getZExtValue() & 2;
    return RValue::get(
      llvm::ConstantInt::get(ResType, UseMinimum ? 0 : -1LL));
  }
  case Builtin::BI__builtin_prefetch: {
    Value *Locality, *RW, *Address = EmitScalarExpr(E->getArg(0));
    // FIXME: Technically these constants should of type 'int', yes?
    RW = (E->getNumArgs() > 1) ? EmitScalarExpr(E->getArg(1)) : 
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), 0);
    Locality = (E->getNumArgs() > 2) ? EmitScalarExpr(E->getArg(2)) : 
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), 3);
    Value *F = CGM.getIntrinsic(Intrinsic::prefetch, 0, 0);
    return RValue::get(Builder.CreateCall3(F, Address, RW, Locality));
  }
  case Builtin::BI__builtin_trap: {
    Value *F = CGM.getIntrinsic(Intrinsic::trap, 0, 0);
    return RValue::get(Builder.CreateCall(F));
  }

  case Builtin::BI__builtin_powi:
  case Builtin::BI__builtin_powif:
  case Builtin::BI__builtin_powil: {
    Value *Base = EmitScalarExpr(E->getArg(0));
    Value *Exponent = EmitScalarExpr(E->getArg(1));
    const llvm::Type *ArgType = Base->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::powi, &ArgType, 1);
    return RValue::get(Builder.CreateCall2(F, Base, Exponent, "tmp"));
  }

  case Builtin::BI__builtin_isgreater:
  case Builtin::BI__builtin_isgreaterequal:
  case Builtin::BI__builtin_isless:
  case Builtin::BI__builtin_islessequal:
  case Builtin::BI__builtin_islessgreater:
  case Builtin::BI__builtin_isunordered: {
    // Ordered comparisons: we know the arguments to these are matching scalar
    // floating point values.
    Value *LHS = EmitScalarExpr(E->getArg(0));   
    Value *RHS = EmitScalarExpr(E->getArg(1));
    
    switch (BuiltinID) {
    default: assert(0 && "Unknown ordered comparison");
    case Builtin::BI__builtin_isgreater:
      LHS = Builder.CreateFCmpOGT(LHS, RHS, "cmp");
      break;
    case Builtin::BI__builtin_isgreaterequal:
      LHS = Builder.CreateFCmpOGE(LHS, RHS, "cmp");
      break;
    case Builtin::BI__builtin_isless:
      LHS = Builder.CreateFCmpOLT(LHS, RHS, "cmp");
      break;
    case Builtin::BI__builtin_islessequal:
      LHS = Builder.CreateFCmpOLE(LHS, RHS, "cmp");
      break;
    case Builtin::BI__builtin_islessgreater:
      LHS = Builder.CreateFCmpONE(LHS, RHS, "cmp");
      break;
    case Builtin::BI__builtin_isunordered:    
      LHS = Builder.CreateFCmpUNO(LHS, RHS, "cmp");
      break;
    }
    // ZExt bool to int type.
    return RValue::get(Builder.CreateZExt(LHS, ConvertType(E->getType()),
                                          "tmp"));
  }
  case Builtin::BI__builtin_isnan: {
    Value *V = EmitScalarExpr(E->getArg(0));
    V = Builder.CreateFCmpUNO(V, V, "cmp");
    return RValue::get(Builder.CreateZExt(V, ConvertType(E->getType()), "tmp"));
  }
  case Builtin::BIalloca:
  case Builtin::BI__builtin_alloca: {
    // FIXME: LLVM IR Should allow alloca with an i64 size!
    Value *Size = EmitScalarExpr(E->getArg(0));
    Size = Builder.CreateIntCast(Size, llvm::Type::getInt32Ty(VMContext), false, "tmp");
    return RValue::get(Builder.CreateAlloca(llvm::Type::getInt8Ty(VMContext), Size, "tmp"));
  }
  case Builtin::BI__builtin_bzero: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Builder.CreateCall4(CGM.getMemSetFn(), Address,
                        llvm::ConstantInt::get(llvm::Type::getInt8Ty(VMContext), 0),
                        EmitScalarExpr(E->getArg(1)),
                        llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), 1));
    return RValue::get(Address);
  }
  case Builtin::BI__builtin_memcpy: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Builder.CreateCall4(CGM.getMemCpyFn(), Address,
                        EmitScalarExpr(E->getArg(1)),
                        EmitScalarExpr(E->getArg(2)),
                        llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), 1));
    return RValue::get(Address);
  }
  case Builtin::BI__builtin_memmove: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Builder.CreateCall4(CGM.getMemMoveFn(), Address,
                        EmitScalarExpr(E->getArg(1)),
                        EmitScalarExpr(E->getArg(2)),
                        llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), 1));
    return RValue::get(Address);
  }
  case Builtin::BI__builtin_memset: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Builder.CreateCall4(CGM.getMemSetFn(), Address,
                        Builder.CreateTrunc(EmitScalarExpr(E->getArg(1)),
                                            llvm::Type::getInt8Ty(VMContext)),
                        EmitScalarExpr(E->getArg(2)),
                        llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), 1));
    return RValue::get(Address);
  }
  case Builtin::BI__builtin_return_address: {
    Value *F = CGM.getIntrinsic(Intrinsic::returnaddress, 0, 0);
    return RValue::get(Builder.CreateCall(F, EmitScalarExpr(E->getArg(0))));
  }
  case Builtin::BI__builtin_frame_address: {
    Value *F = CGM.getIntrinsic(Intrinsic::frameaddress, 0, 0);
    return RValue::get(Builder.CreateCall(F, EmitScalarExpr(E->getArg(0))));
  }
  case Builtin::BI__builtin_extract_return_addr: {
    // FIXME: There should be a target hook for this
    return RValue::get(EmitScalarExpr(E->getArg(0)));
  }
  case Builtin::BI__builtin_unwind_init: {
    Value *F = CGM.getIntrinsic(Intrinsic::eh_unwind_init, 0, 0);
    return RValue::get(Builder.CreateCall(F));
  }
#if 0
  // FIXME: Finish/enable when LLVM backend support stabilizes
  case Builtin::BI__builtin_setjmp: {
    Value *Buf = EmitScalarExpr(E->getArg(0));
    // Store the frame pointer to the buffer
    Value *FrameAddrF = CGM.getIntrinsic(Intrinsic::frameaddress, 0, 0);
    Value *FrameAddr =
        Builder.CreateCall(FrameAddrF,
                           Constant::getNullValue(llvm::Type::getInt32Ty(VMContext)));
    Builder.CreateStore(FrameAddr, Buf);
    // Call the setjmp intrinsic
    Value *F = CGM.getIntrinsic(Intrinsic::eh_sjlj_setjmp, 0, 0);
    const llvm::Type *DestType =
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(VMContext));
    Buf = Builder.CreateBitCast(Buf, DestType);
    return RValue::get(Builder.CreateCall(F, Buf));
  }
  case Builtin::BI__builtin_longjmp: {
    Value *F = CGM.getIntrinsic(Intrinsic::eh_sjlj_longjmp, 0, 0);
    Value *Buf = EmitScalarExpr(E->getArg(0));
    const llvm::Type *DestType = 
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(VMContext));
    Buf = Builder.CreateBitCast(Buf, DestType);
    return RValue::get(Builder.CreateCall(F, Buf));
  }
#endif
  case Builtin::BI__sync_fetch_and_add:
  case Builtin::BI__sync_fetch_and_sub:
  case Builtin::BI__sync_fetch_and_or:
  case Builtin::BI__sync_fetch_and_and:
  case Builtin::BI__sync_fetch_and_xor:
  case Builtin::BI__sync_add_and_fetch:
  case Builtin::BI__sync_sub_and_fetch:
  case Builtin::BI__sync_and_and_fetch:
  case Builtin::BI__sync_or_and_fetch:
  case Builtin::BI__sync_xor_and_fetch:
  case Builtin::BI__sync_val_compare_and_swap:
  case Builtin::BI__sync_bool_compare_and_swap:
  case Builtin::BI__sync_lock_test_and_set:
  case Builtin::BI__sync_lock_release:
    assert(0 && "Shouldn't make it through sema");
  case Builtin::BI__sync_fetch_and_add_1:
  case Builtin::BI__sync_fetch_and_add_2:
  case Builtin::BI__sync_fetch_and_add_4:
  case Builtin::BI__sync_fetch_and_add_8:
  case Builtin::BI__sync_fetch_and_add_16:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_add, E);
  case Builtin::BI__sync_fetch_and_sub_1:
  case Builtin::BI__sync_fetch_and_sub_2:
  case Builtin::BI__sync_fetch_and_sub_4:
  case Builtin::BI__sync_fetch_and_sub_8:
  case Builtin::BI__sync_fetch_and_sub_16:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_sub, E);
  case Builtin::BI__sync_fetch_and_or_1:
  case Builtin::BI__sync_fetch_and_or_2:
  case Builtin::BI__sync_fetch_and_or_4:
  case Builtin::BI__sync_fetch_and_or_8:
  case Builtin::BI__sync_fetch_and_or_16:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_or, E);
  case Builtin::BI__sync_fetch_and_and_1:
  case Builtin::BI__sync_fetch_and_and_2:
  case Builtin::BI__sync_fetch_and_and_4:
  case Builtin::BI__sync_fetch_and_and_8:
  case Builtin::BI__sync_fetch_and_and_16:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_and, E);
  case Builtin::BI__sync_fetch_and_xor_1:
  case Builtin::BI__sync_fetch_and_xor_2:
  case Builtin::BI__sync_fetch_and_xor_4:
  case Builtin::BI__sync_fetch_and_xor_8:
  case Builtin::BI__sync_fetch_and_xor_16:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_xor, E);
  case Builtin::BI__sync_fetch_and_nand_1:
  case Builtin::BI__sync_fetch_and_nand_2:
  case Builtin::BI__sync_fetch_and_nand_4:
  case Builtin::BI__sync_fetch_and_nand_8:
  case Builtin::BI__sync_fetch_and_nand_16:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_nand, E);
      
  // Clang extensions: not overloaded yet.
  case Builtin::BI__sync_fetch_and_min:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_min, E);
  case Builtin::BI__sync_fetch_and_max:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_max, E);
  case Builtin::BI__sync_fetch_and_umin:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_umin, E);
  case Builtin::BI__sync_fetch_and_umax:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_umax, E);

  case Builtin::BI__sync_add_and_fetch_1:
  case Builtin::BI__sync_add_and_fetch_2:
  case Builtin::BI__sync_add_and_fetch_4:
  case Builtin::BI__sync_add_and_fetch_8:
  case Builtin::BI__sync_add_and_fetch_16:
    return EmitBinaryAtomicPost(*this, Intrinsic::atomic_load_add, E, 
                                llvm::Instruction::Add);
  case Builtin::BI__sync_sub_and_fetch_1:
  case Builtin::BI__sync_sub_and_fetch_2:
  case Builtin::BI__sync_sub_and_fetch_4:
  case Builtin::BI__sync_sub_and_fetch_8:
  case Builtin::BI__sync_sub_and_fetch_16:
    return EmitBinaryAtomicPost(*this, Intrinsic::atomic_load_sub, E,
                                llvm::Instruction::Sub);
  case Builtin::BI__sync_and_and_fetch_1:
  case Builtin::BI__sync_and_and_fetch_2:
  case Builtin::BI__sync_and_and_fetch_4:
  case Builtin::BI__sync_and_and_fetch_8:
  case Builtin::BI__sync_and_and_fetch_16:
    return EmitBinaryAtomicPost(*this, Intrinsic::atomic_load_and, E,
                                llvm::Instruction::And);
  case Builtin::BI__sync_or_and_fetch_1:
  case Builtin::BI__sync_or_and_fetch_2:
  case Builtin::BI__sync_or_and_fetch_4:
  case Builtin::BI__sync_or_and_fetch_8:
  case Builtin::BI__sync_or_and_fetch_16:
    return EmitBinaryAtomicPost(*this, Intrinsic::atomic_load_or, E,
                                llvm::Instruction::Or);
  case Builtin::BI__sync_xor_and_fetch_1:
  case Builtin::BI__sync_xor_and_fetch_2:
  case Builtin::BI__sync_xor_and_fetch_4:
  case Builtin::BI__sync_xor_and_fetch_8:
  case Builtin::BI__sync_xor_and_fetch_16:
    return EmitBinaryAtomicPost(*this, Intrinsic::atomic_load_xor, E,
                                llvm::Instruction::Xor);
  case Builtin::BI__sync_nand_and_fetch_1:
  case Builtin::BI__sync_nand_and_fetch_2:
  case Builtin::BI__sync_nand_and_fetch_4:
  case Builtin::BI__sync_nand_and_fetch_8:
  case Builtin::BI__sync_nand_and_fetch_16:
    return EmitBinaryAtomicPost(*this, Intrinsic::atomic_load_nand, E,
                                llvm::Instruction::And);
      
  case Builtin::BI__sync_val_compare_and_swap_1:
  case Builtin::BI__sync_val_compare_and_swap_2:
  case Builtin::BI__sync_val_compare_and_swap_4:
  case Builtin::BI__sync_val_compare_and_swap_8:
  case Builtin::BI__sync_val_compare_and_swap_16:
  {
    const llvm::Type *ResType[2];
    ResType[0]= ConvertType(E->getType());
    ResType[1] = ConvertType(E->getArg(0)->getType());
    Value *AtomF = CGM.getIntrinsic(Intrinsic::atomic_cmp_swap, ResType, 2);
    return RValue::get(Builder.CreateCall3(AtomF, 
                                           EmitScalarExpr(E->getArg(0)),
                                           EmitScalarExpr(E->getArg(1)),
                                           EmitScalarExpr(E->getArg(2))));
  }

  case Builtin::BI__sync_bool_compare_and_swap_1:
  case Builtin::BI__sync_bool_compare_and_swap_2:
  case Builtin::BI__sync_bool_compare_and_swap_4:
  case Builtin::BI__sync_bool_compare_and_swap_8:
  case Builtin::BI__sync_bool_compare_and_swap_16:
  {
    const llvm::Type *ResType[2];
    ResType[0]= ConvertType(E->getArg(1)->getType());
    ResType[1] = llvm::PointerType::getUnqual(ResType[0]);
    Value *AtomF = CGM.getIntrinsic(Intrinsic::atomic_cmp_swap, ResType, 2);
    Value *OldVal = EmitScalarExpr(E->getArg(1));
    Value *PrevVal = Builder.CreateCall3(AtomF, 
                                        EmitScalarExpr(E->getArg(0)),
                                        OldVal,
                                        EmitScalarExpr(E->getArg(2)));
    Value *Result = Builder.CreateICmpEQ(PrevVal, OldVal);
    // zext bool to int.
    return RValue::get(Builder.CreateZExt(Result, ConvertType(E->getType())));
  }

  case Builtin::BI__sync_lock_test_and_set_1:
  case Builtin::BI__sync_lock_test_and_set_2:
  case Builtin::BI__sync_lock_test_and_set_4:
  case Builtin::BI__sync_lock_test_and_set_8:
  case Builtin::BI__sync_lock_test_and_set_16:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_swap, E);
  case Builtin::BI__sync_lock_release_1:
  case Builtin::BI__sync_lock_release_2:
  case Builtin::BI__sync_lock_release_4:
  case Builtin::BI__sync_lock_release_8:
  case Builtin::BI__sync_lock_release_16: {
    Value *Ptr = EmitScalarExpr(E->getArg(0));
    const llvm::Type *ElTy =
      cast<llvm::PointerType>(Ptr->getType())->getElementType();
    Builder.CreateStore(llvm::Constant::getNullValue(ElTy), Ptr, true);
    return RValue::get(0);
  }

  case Builtin::BI__sync_synchronize: {
    Value *C[5];
    C[0] = C[1] = C[2] = C[3] = llvm::ConstantInt::get(llvm::Type::getInt1Ty(VMContext), 1);
    C[4] = llvm::ConstantInt::get(llvm::Type::getInt1Ty(VMContext), 0);
    Builder.CreateCall(CGM.getIntrinsic(Intrinsic::memory_barrier), C, C + 5);
    return RValue::get(0);
  }
      
    // Library functions with special handling.
  case Builtin::BIsqrt:
  case Builtin::BIsqrtf:
  case Builtin::BIsqrtl: {
    // Rewrite sqrt to intrinsic if allowed.
    if (!FD->hasAttr<ConstAttr>())
      break;
    Value *Arg0 = EmitScalarExpr(E->getArg(0));
    const llvm::Type *ArgType = Arg0->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::sqrt, &ArgType, 1);
    return RValue::get(Builder.CreateCall(F, Arg0, "tmp"));
  }

  case Builtin::BIpow:
  case Builtin::BIpowf:
  case Builtin::BIpowl: {
    // Rewrite sqrt to intrinsic if allowed.
    if (!FD->hasAttr<ConstAttr>())
      break;
    Value *Base = EmitScalarExpr(E->getArg(0));
    Value *Exponent = EmitScalarExpr(E->getArg(1));
    const llvm::Type *ArgType = Base->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::pow, &ArgType, 1);
    return RValue::get(Builder.CreateCall2(F, Base, Exponent, "tmp"));
  }
  }
  
  // If this is an alias for a libm function (e.g. __builtin_sin) turn it into
  // that function.
  if (getContext().BuiltinInfo.isLibFunction(BuiltinID) ||
      getContext().BuiltinInfo.isPredefinedLibFunction(BuiltinID))
    return EmitCall(CGM.getBuiltinLibFunction(BuiltinID), 
                    E->getCallee()->getType(), E->arg_begin(),
                    E->arg_end());
  
  // See if we have a target specific intrinsic.
  const char *Name = getContext().BuiltinInfo.GetName(BuiltinID);
  Intrinsic::ID IntrinsicID = Intrinsic::not_intrinsic;
  if (const char *Prefix =
      llvm::Triple::getArchTypePrefix(Target.getTriple().getArch()))  
    IntrinsicID = Intrinsic::getIntrinsicForGCCBuiltin(Prefix, Name);
  
  if (IntrinsicID != Intrinsic::not_intrinsic) {
    SmallVector<Value*, 16> Args;
    
    Function *F = CGM.getIntrinsic(IntrinsicID);
    const llvm::FunctionType *FTy = F->getFunctionType();
    
    for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
      Value *ArgValue = EmitScalarExpr(E->getArg(i));
      
      // If the intrinsic arg type is different from the builtin arg type
      // we need to do a bit cast.
      const llvm::Type *PTy = FTy->getParamType(i);
      if (PTy != ArgValue->getType()) {
        assert(PTy->canLosslesslyBitCastTo(FTy->getParamType(i)) &&
               "Must be able to losslessly bit cast to param");
        ArgValue = Builder.CreateBitCast(ArgValue, PTy);
      }
      
      Args.push_back(ArgValue);
    }
    
    Value *V = Builder.CreateCall(F, Args.data(), Args.data() + Args.size());
    QualType BuiltinRetType = E->getType();
    
    const llvm::Type *RetTy = llvm::Type::getVoidTy(VMContext);
    if (!BuiltinRetType->isVoidType()) RetTy = ConvertType(BuiltinRetType);
    
    if (RetTy != V->getType()) {
      assert(V->getType()->canLosslesslyBitCastTo(RetTy) &&
             "Must be able to losslessly bit cast result type");
      V = Builder.CreateBitCast(V, RetTy);
    }
    
    return RValue::get(V);
  }
  
  // See if we have a target specific builtin that needs to be lowered.
  if (Value *V = EmitTargetBuiltinExpr(BuiltinID, E))
    return RValue::get(V);
  
  ErrorUnsupported(E, "builtin function");
  
  // Unknown builtin, for now just dump it out and return undef.
  if (hasAggregateLLVMType(E->getType()))
    return RValue::getAggregate(CreateTempAlloca(ConvertType(E->getType())));
  return RValue::get(llvm::UndefValue::get(ConvertType(E->getType())));
}    

Value *CodeGenFunction::EmitTargetBuiltinExpr(unsigned BuiltinID,
                                              const CallExpr *E) {
  switch (Target.getTriple().getArch()) {
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    return EmitX86BuiltinExpr(BuiltinID, E);
  case llvm::Triple::ppc:
  case llvm::Triple::ppc64:
    return EmitPPCBuiltinExpr(BuiltinID, E);
  default:
    return 0;
  }
}

Value *CodeGenFunction::EmitX86BuiltinExpr(unsigned BuiltinID, 
                                           const CallExpr *E) {
  
  llvm::SmallVector<Value*, 4> Ops;

  for (unsigned i = 0, e = E->getNumArgs(); i != e; i++)
    Ops.push_back(EmitScalarExpr(E->getArg(i)));

  switch (BuiltinID) {
  default: return 0;
  case X86::BI__builtin_ia32_pslldi128: 
  case X86::BI__builtin_ia32_psllqi128:
  case X86::BI__builtin_ia32_psllwi128: 
  case X86::BI__builtin_ia32_psradi128:
  case X86::BI__builtin_ia32_psrawi128:
  case X86::BI__builtin_ia32_psrldi128:
  case X86::BI__builtin_ia32_psrlqi128:
  case X86::BI__builtin_ia32_psrlwi128: {
    Ops[1] = Builder.CreateZExt(Ops[1], llvm::Type::getInt64Ty(VMContext), "zext");
    const llvm::Type *Ty = llvm::VectorType::get(llvm::Type::getInt64Ty(VMContext), 2);
    llvm::Value *Zero = llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), 0);
    Ops[1] = Builder.CreateInsertElement(llvm::UndefValue::get(Ty),
                                         Ops[1], Zero, "insert");
    Ops[1] = Builder.CreateBitCast(Ops[1], Ops[0]->getType(), "bitcast");
    const char *name = 0;
    Intrinsic::ID ID = Intrinsic::not_intrinsic;
    
    switch (BuiltinID) {
    default: assert(0 && "Unsupported shift intrinsic!");
    case X86::BI__builtin_ia32_pslldi128:
      name = "pslldi";
      ID = Intrinsic::x86_sse2_psll_d;
      break;
    case X86::BI__builtin_ia32_psllqi128:
      name = "psllqi";
      ID = Intrinsic::x86_sse2_psll_q;
      break;
    case X86::BI__builtin_ia32_psllwi128:
      name = "psllwi";
      ID = Intrinsic::x86_sse2_psll_w;
      break;
    case X86::BI__builtin_ia32_psradi128:
      name = "psradi";
      ID = Intrinsic::x86_sse2_psra_d;
      break;
    case X86::BI__builtin_ia32_psrawi128:
      name = "psrawi";
      ID = Intrinsic::x86_sse2_psra_w;
      break;
    case X86::BI__builtin_ia32_psrldi128:
      name = "psrldi";
      ID = Intrinsic::x86_sse2_psrl_d;
      break;
    case X86::BI__builtin_ia32_psrlqi128:
      name = "psrlqi";
      ID = Intrinsic::x86_sse2_psrl_q;
      break;
    case X86::BI__builtin_ia32_psrlwi128:
      name = "psrlwi";
      ID = Intrinsic::x86_sse2_psrl_w;
      break;
    }
    llvm::Function *F = CGM.getIntrinsic(ID);
    return Builder.CreateCall(F, &Ops[0], &Ops[0] + Ops.size(), name);  
  }
  case X86::BI__builtin_ia32_pslldi: 
  case X86::BI__builtin_ia32_psllqi:
  case X86::BI__builtin_ia32_psllwi: 
  case X86::BI__builtin_ia32_psradi:
  case X86::BI__builtin_ia32_psrawi:
  case X86::BI__builtin_ia32_psrldi:
  case X86::BI__builtin_ia32_psrlqi:
  case X86::BI__builtin_ia32_psrlwi: {
    Ops[1] = Builder.CreateZExt(Ops[1], llvm::Type::getInt64Ty(VMContext), "zext");
    const llvm::Type *Ty = llvm::VectorType::get(llvm::Type::getInt64Ty(VMContext), 1);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty, "bitcast");
    const char *name = 0;
    Intrinsic::ID ID = Intrinsic::not_intrinsic;
    
    switch (BuiltinID) {
    default: assert(0 && "Unsupported shift intrinsic!");
    case X86::BI__builtin_ia32_pslldi:
      name = "pslldi";
      ID = Intrinsic::x86_mmx_psll_d;
      break;
    case X86::BI__builtin_ia32_psllqi:
      name = "psllqi";
      ID = Intrinsic::x86_mmx_psll_q;
      break;
    case X86::BI__builtin_ia32_psllwi:
      name = "psllwi";
      ID = Intrinsic::x86_mmx_psll_w;
      break;
    case X86::BI__builtin_ia32_psradi:
      name = "psradi";
      ID = Intrinsic::x86_mmx_psra_d;
      break;
    case X86::BI__builtin_ia32_psrawi:
      name = "psrawi";
      ID = Intrinsic::x86_mmx_psra_w;
      break;
    case X86::BI__builtin_ia32_psrldi:
      name = "psrldi";
      ID = Intrinsic::x86_mmx_psrl_d;
      break;
    case X86::BI__builtin_ia32_psrlqi:
      name = "psrlqi";
      ID = Intrinsic::x86_mmx_psrl_q;
      break;
    case X86::BI__builtin_ia32_psrlwi:
      name = "psrlwi";
      ID = Intrinsic::x86_mmx_psrl_w;
      break;
    }
    llvm::Function *F = CGM.getIntrinsic(ID);
    return Builder.CreateCall(F, &Ops[0], &Ops[0] + Ops.size(), name);  
  }
  case X86::BI__builtin_ia32_cmpps: {
    llvm::Function *F = CGM.getIntrinsic(Intrinsic::x86_sse_cmp_ps);
    return Builder.CreateCall(F, &Ops[0], &Ops[0] + Ops.size(), "cmpps");
  }
  case X86::BI__builtin_ia32_cmpss: {
    llvm::Function *F = CGM.getIntrinsic(Intrinsic::x86_sse_cmp_ss);
    return Builder.CreateCall(F, &Ops[0], &Ops[0] + Ops.size(), "cmpss");
  }
  case X86::BI__builtin_ia32_ldmxcsr: {
    llvm::Type *PtrTy = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(VMContext));
    Value *One = llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), 1);
    Value *Tmp = Builder.CreateAlloca(llvm::Type::getInt32Ty(VMContext), One, "tmp");
    Builder.CreateStore(Ops[0], Tmp);
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_sse_ldmxcsr),
                              Builder.CreateBitCast(Tmp, PtrTy));
  }
  case X86::BI__builtin_ia32_stmxcsr: {
    llvm::Type *PtrTy = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(VMContext));
    Value *One = llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), 1);
    Value *Tmp = Builder.CreateAlloca(llvm::Type::getInt32Ty(VMContext), One, "tmp");
    One = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_sse_stmxcsr),
                             Builder.CreateBitCast(Tmp, PtrTy));
    return Builder.CreateLoad(Tmp, "stmxcsr");
  }
  case X86::BI__builtin_ia32_cmppd: {
    llvm::Function *F = CGM.getIntrinsic(Intrinsic::x86_sse2_cmp_pd);
    return Builder.CreateCall(F, &Ops[0], &Ops[0] + Ops.size(), "cmppd");
  }
  case X86::BI__builtin_ia32_cmpsd: {
    llvm::Function *F = CGM.getIntrinsic(Intrinsic::x86_sse2_cmp_sd);
    return Builder.CreateCall(F, &Ops[0], &Ops[0] + Ops.size(), "cmpsd");
  }
  case X86::BI__builtin_ia32_storehps:
  case X86::BI__builtin_ia32_storelps: {
    const llvm::Type *EltTy = llvm::Type::getInt64Ty(VMContext);
    llvm::Type *PtrTy = llvm::PointerType::getUnqual(EltTy);
    llvm::Type *VecTy = llvm::VectorType::get(EltTy, 2);
    
    // cast val v2i64
    Ops[1] = Builder.CreateBitCast(Ops[1], VecTy, "cast");
    
    // extract (0, 1)
    unsigned Index = BuiltinID == X86::BI__builtin_ia32_storelps ? 0 : 1;
    llvm::Value *Idx = llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), Index);
    Ops[1] = Builder.CreateExtractElement(Ops[1], Idx, "extract");

    // cast pointer to i64 & store
    Ops[0] = Builder.CreateBitCast(Ops[0], PtrTy);
    return Builder.CreateStore(Ops[1], Ops[0]);
  }
  }
}

Value *CodeGenFunction::EmitPPCBuiltinExpr(unsigned BuiltinID, 
                                           const CallExpr *E) {
  switch (BuiltinID) {
  default: return 0;
  }
}  
