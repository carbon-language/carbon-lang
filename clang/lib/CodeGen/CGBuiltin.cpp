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
#include "clang/AST/ASTContext.h"
#include "clang/AST/Builtins.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TargetBuiltins.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
using namespace clang;
using namespace CodeGen;
using namespace llvm;

/// Utility to insert an atomic instruction based Instrinsic::ID and
// the expression node
static RValue EmitBinaryAtomic(CodeGenFunction& CFG, 
                               Intrinsic::ID Id, const CallExpr *E) {
  const llvm::Type *ResType = CFG.ConvertType(E->getType());
  Value *AtomF = CFG.CGM.getIntrinsic(Id, &ResType, 1);
  return RValue::get(CFG.Builder.CreateCall2(AtomF,
                                             CFG.EmitScalarExpr(E->getArg(0)),
                                             CFG.EmitScalarExpr(E->getArg(1))));
}

RValue CodeGenFunction::EmitBuiltinExpr(unsigned BuiltinID, const CallExpr *E) {
  switch (BuiltinID) {
  default: break;  // Handle intrinsics and libm functions below.
      
  case Builtin::BI__builtin___CFStringMakeConstantString: {
    const Expr *Arg = E->getArg(0);
    
    while (1) {
      if (const ParenExpr *PE = dyn_cast<ParenExpr>(Arg))
        Arg = PE->getSubExpr();
      else if (const ImplicitCastExpr *CE = dyn_cast<ImplicitCastExpr>(Arg))
        Arg = CE->getSubExpr();
      else
        break;
    }
    
    const StringLiteral *Literal = cast<StringLiteral>(Arg);
    std::string S(Literal->getStrData(), Literal->getByteLength());
    
    return RValue::get(CGM.GetAddrOfConstantCFString(S));
  }
  case Builtin::BI__builtin_stdarg_start:
  case Builtin::BI__builtin_va_start:
  case Builtin::BI__builtin_va_end: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    const llvm::Type *DestType = 
      llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
    if (ArgValue->getType() != DestType)
      ArgValue = Builder.CreateBitCast(ArgValue, DestType, 
                                       ArgValue->getNameStart());

    Intrinsic::ID inst = (BuiltinID == Builtin::BI__builtin_va_end) ? 
      Intrinsic::vaend : Intrinsic::vastart;
    return RValue::get(Builder.CreateCall(CGM.getIntrinsic(inst), ArgValue));
  }
  case Builtin::BI__builtin_va_copy: {
    // FIXME: This does not yet handle architectures where va_list is a struct.
    Value *DstPtr = EmitScalarExpr(E->getArg(0));
    Value *SrcValue = EmitScalarExpr(E->getArg(1));
    
    Value *SrcPtr = CreateTempAlloca(SrcValue->getType(), "dst_ptr");
    
    Builder.CreateStore(SrcValue, SrcPtr, false);

    const llvm::Type *Type = 
      llvm::PointerType::getUnqual(llvm::Type::Int8Ty);

    DstPtr = Builder.CreateBitCast(DstPtr, Type);
    SrcPtr = Builder.CreateBitCast(SrcPtr, Type);
    return RValue::get(Builder.CreateCall2(CGM.getIntrinsic(Intrinsic::vacopy), 
                                           DstPtr, SrcPtr));
  }
  case Builtin::BI__builtin_classify_type: {
    APSInt Result(32);
    if (!E->isBuiltinClassifyType(Result))
      assert(0 && "Expr not __builtin_classify_type!");
    return RValue::get(ConstantInt::get(Result));
  }
  case Builtin::BI__builtin_constant_p: {
    APSInt Result(32);
    // FIXME: Analyze the parameter and check if it is a constant.
    Result = 0;
    return RValue::get(ConstantInt::get(Result));
  }
  case Builtin::BI__builtin_abs: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));   
    
    llvm::BinaryOperator *NegOp = 
      Builder.CreateNeg(ArgValue, (ArgValue->getName() + "neg").c_str());
    Value *CmpResult = 
      Builder.CreateICmpSGE(ArgValue, NegOp->getOperand(0), "abscond");
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
                                   ConstantInt::get(ArgType, 1), "tmp");
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
    Value *Result = Builder.CreateAnd(Tmp, ConstantInt::get(ArgType, 1), 
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
  case Builtin::BI__builtin_prefetch: {
    Value *Locality, *RW, *Address = EmitScalarExpr(E->getArg(0));
    // FIXME: Technically these constants should of type 'int', yes?
    RW = (E->getNumArgs() > 1) ? EmitScalarExpr(E->getArg(1)) : 
      ConstantInt::get(llvm::Type::Int32Ty, 0);
    Locality = (E->getNumArgs() > 2) ? EmitScalarExpr(E->getArg(2)) : 
      ConstantInt::get(llvm::Type::Int32Ty, 3);
    Value *F = CGM.getIntrinsic(Intrinsic::prefetch, 0, 0);
    return RValue::get(Builder.CreateCall3(F, Address, RW, Locality));
  }
  case Builtin::BI__builtin_trap: {
    Value *F = CGM.getIntrinsic(Intrinsic::trap, 0, 0);
    return RValue::get(Builder.CreateCall(F));
  }

  case Builtin::BI__builtin_huge_val:
  case Builtin::BI__builtin_huge_valf:
  case Builtin::BI__builtin_huge_vall:
  case Builtin::BI__builtin_inf:
  case Builtin::BI__builtin_inff:
  case Builtin::BI__builtin_infl: {
    const llvm::fltSemantics &Sem =
      CGM.getContext().getFloatTypeSemantics(E->getType());
    return RValue::get(ConstantFP::get(APFloat::getInf(Sem)));
  }
  case Builtin::BI__builtin_nan:
  case Builtin::BI__builtin_nanf:
  case Builtin::BI__builtin_nanl: {
    // If this is __builtin_nan("") turn this into a simple nan, otherwise just
    // call libm nan.
    if (const StringLiteral *S = 
          dyn_cast<StringLiteral>(E->getArg(0)->IgnoreParenCasts())) {
      if (!S->isWide() && S->getByteLength() == 0) { // empty string.
        const llvm::fltSemantics &Sem = 
          CGM.getContext().getFloatTypeSemantics(E->getType());
        return RValue::get(ConstantFP::get(APFloat::getNaN(Sem)));
      }
    }
    // Otherwise, call libm 'nan'.
    break;
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
  case Builtin::BI__builtin_alloca: {
    // FIXME: LLVM IR Should allow alloca with an i64 size!
    Value *Size = EmitScalarExpr(E->getArg(0));
    Size = Builder.CreateIntCast(Size, llvm::Type::Int32Ty, false, "tmp");
    return RValue::get(Builder.CreateAlloca(llvm::Type::Int8Ty, Size, "tmp"));
  }
  case Builtin::BI__builtin_bzero: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Builder.CreateCall4(CGM.getMemSetFn(), Address,
                        llvm::ConstantInt::get(llvm::Type::Int8Ty, 0),
                        EmitScalarExpr(E->getArg(1)),
                        llvm::ConstantInt::get(llvm::Type::Int32Ty, 1));
    return RValue::get(Address);
  }
  case Builtin::BI__builtin_memcpy: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Builder.CreateCall4(CGM.getMemCpyFn(), Address,
                        EmitScalarExpr(E->getArg(1)),
                        EmitScalarExpr(E->getArg(2)),
                        llvm::ConstantInt::get(llvm::Type::Int32Ty, 1));
    return RValue::get(Address);
  }
  case Builtin::BI__builtin_memmove: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Builder.CreateCall4(CGM.getMemMoveFn(), Address,
                        EmitScalarExpr(E->getArg(1)),
                        EmitScalarExpr(E->getArg(2)),
                        llvm::ConstantInt::get(llvm::Type::Int32Ty, 1));
    return RValue::get(Address);
  }
  case Builtin::BI__builtin_memset: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Builder.CreateCall4(CGM.getMemSetFn(), Address,
                        EmitScalarExpr(E->getArg(1)),
                        EmitScalarExpr(E->getArg(2)),
                        llvm::ConstantInt::get(llvm::Type::Int32Ty, 1));
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
  case Builtin::BI__sync_fetch_and_add:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_add, E);
  case Builtin::BI__sync_fetch_and_sub:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_sub, E);
  case Builtin::BI__sync_fetch_and_min:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_min, E);
  case Builtin::BI__sync_fetch_and_max:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_max, E);
  case Builtin::BI__sync_fetch_and_umin:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_umin, E);
  case Builtin::BI__sync_fetch_and_umax:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_umax, E);
  case Builtin::BI__sync_fetch_and_and:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_and, E);
  case Builtin::BI__sync_fetch_and_or:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_or, E);
  case Builtin::BI__sync_fetch_and_xor:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_load_xor, E);
  case Builtin::BI__sync_val_compare_and_swap: {
    Value *Args[3];
    Args[0]= EmitScalarExpr(E->getArg(0));
    Args[1] = EmitScalarExpr(E->getArg(1));
    Args[2] = EmitScalarExpr(E->getArg(2));
    const llvm::Type *ResType = ConvertType(E->getType());
    Value *AtomF = CGM.getIntrinsic(Intrinsic::atomic_cmp_swap, &ResType, 1);
    return RValue::get(Builder.CreateCall(AtomF, &Args[0], &Args[1]+2));
  }
  case Builtin::BI__sync_lock_test_and_set:
    return EmitBinaryAtomic(*this, Intrinsic::atomic_swap, E);
  }
  
  // If this is an alias for a libm function (e.g. __builtin_sin) turn it into
  // that function.
  if (getContext().BuiltinInfo.isLibFunction(BuiltinID))
    return EmitCallExpr(CGM.getBuiltinLibFunction(BuiltinID), 
                        E->getCallee()->getType(), E->arg_begin(),
                        E->arg_end());
  
  // See if we have a target specific intrinsic.
  Intrinsic::ID IntrinsicID;
  const char *TargetPrefix = Target.getTargetPrefix();
  const char *BuiltinName = getContext().BuiltinInfo.GetName(BuiltinID);
#define GET_LLVM_INTRINSIC_FOR_GCC_BUILTIN
#include "llvm/Intrinsics.gen"
#undef GET_LLVM_INTRINSIC_FOR_GCC_BUILTIN
  
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
    
    Value *V = Builder.CreateCall(F, &Args[0], &Args[0] + Args.size());
    QualType BuiltinRetType = E->getType();
    
    const llvm::Type *RetTy = llvm::Type::VoidTy;
    if (!BuiltinRetType->isVoidType()) RetTy = ConvertType(BuiltinRetType);
    
    if (RetTy != V->getType()) {
      assert(V->getType()->canLosslesslyBitCastTo(RetTy) &&
             "Must be able to losslessly bit cast result type");
      V = Builder.CreateBitCast(V, RetTy);
    }
    
    return RValue::get(V);
  }
  
  // See if we have a target specific builtin that needs to be lowered.
  Value *V = 0;
  
  if (strcmp(TargetPrefix, "x86") == 0)
    V = EmitX86BuiltinExpr(BuiltinID, E);
  else if (strcmp(TargetPrefix, "ppc") == 0)
    V = EmitPPCBuiltinExpr(BuiltinID, E);
  
  if (V)
    return RValue::get(V);
  
  WarnUnsupported(E, "builtin function");
  
  // Unknown builtin, for now just dump it out and return undef.
  if (hasAggregateLLVMType(E->getType()))
    return RValue::getAggregate(CreateTempAlloca(ConvertType(E->getType())));
  return RValue::get(UndefValue::get(ConvertType(E->getType())));
}    

Value *CodeGenFunction::EmitX86BuiltinExpr(unsigned BuiltinID, 
                                           const CallExpr *E) {
  
  llvm::SmallVector<Value*, 4> Ops;

  for (unsigned i = 0, e = E->getNumArgs(); i != e; i++)
    Ops.push_back(EmitScalarExpr(E->getArg(i)));

  switch (BuiltinID) {
  default: return 0;
  case X86::BI__builtin_ia32_mulps:
    return Builder.CreateMul(Ops[0], Ops[1], "mulps");
  case X86::BI__builtin_ia32_mulpd:
    return Builder.CreateMul(Ops[0], Ops[1], "mulpd");
  case X86::BI__builtin_ia32_pand:
  case X86::BI__builtin_ia32_pand128:
    return Builder.CreateAnd(Ops[0], Ops[1], "pand");
  case X86::BI__builtin_ia32_por:
  case X86::BI__builtin_ia32_por128:
    return Builder.CreateOr(Ops[0], Ops[1], "por");
  case X86::BI__builtin_ia32_pxor:
  case X86::BI__builtin_ia32_pxor128:
    return Builder.CreateXor(Ops[0], Ops[1], "pxor");
  case X86::BI__builtin_ia32_pandn:
  case X86::BI__builtin_ia32_pandn128:
    Ops[0] = Builder.CreateNot(Ops[0], "tmp");
    return Builder.CreateAnd(Ops[0], Ops[1], "pandn");
  case X86::BI__builtin_ia32_paddb:
  case X86::BI__builtin_ia32_paddb128:
  case X86::BI__builtin_ia32_paddd:
  case X86::BI__builtin_ia32_paddd128:
  case X86::BI__builtin_ia32_paddq:
  case X86::BI__builtin_ia32_paddq128:
  case X86::BI__builtin_ia32_paddw:
  case X86::BI__builtin_ia32_paddw128:
  case X86::BI__builtin_ia32_addps:
  case X86::BI__builtin_ia32_addpd:
    return Builder.CreateAdd(Ops[0], Ops[1], "add");
  case X86::BI__builtin_ia32_psubb:
  case X86::BI__builtin_ia32_psubb128:
  case X86::BI__builtin_ia32_psubd:
  case X86::BI__builtin_ia32_psubd128:
  case X86::BI__builtin_ia32_psubq:
  case X86::BI__builtin_ia32_psubq128:
  case X86::BI__builtin_ia32_psubw:
  case X86::BI__builtin_ia32_psubw128:
  case X86::BI__builtin_ia32_subps:
  case X86::BI__builtin_ia32_subpd:
    return Builder.CreateSub(Ops[0], Ops[1], "sub");
  case X86::BI__builtin_ia32_divps:
    return Builder.CreateFDiv(Ops[0], Ops[1], "divps");
  case X86::BI__builtin_ia32_divpd:
    return Builder.CreateFDiv(Ops[0], Ops[1], "divpd");
  case X86::BI__builtin_ia32_pmullw:
  case X86::BI__builtin_ia32_pmullw128:
    return Builder.CreateMul(Ops[0], Ops[1], "pmul");
  case X86::BI__builtin_ia32_punpckhbw:
    return EmitShuffleVector(Ops[0], Ops[1], 4, 12, 5, 13, 6, 14, 7, 15,
                             "punpckhbw");
  case X86::BI__builtin_ia32_punpckhbw128:
    return EmitShuffleVector(Ops[0], Ops[1],  8, 24,  9, 25, 10, 26, 11, 27,
                                             12, 28, 13, 29, 14, 30, 15, 31,
                             "punpckhbw");
  case X86::BI__builtin_ia32_punpckhwd:
    return EmitShuffleVector(Ops[0], Ops[1], 2, 6, 3, 7, "punpckhwd");
  case X86::BI__builtin_ia32_punpckhwd128:
    return EmitShuffleVector(Ops[0], Ops[1], 4, 12, 5, 13, 6, 14, 7, 15,
                             "punpckhwd");
  case X86::BI__builtin_ia32_punpckhdq:
    return EmitShuffleVector(Ops[0], Ops[1], 1, 3, "punpckhdq");
  case X86::BI__builtin_ia32_punpckhdq128:
    return EmitShuffleVector(Ops[0], Ops[1], 2, 6, 3, 7, "punpckhdq");
  case X86::BI__builtin_ia32_punpcklbw:
    return EmitShuffleVector(Ops[0], Ops[1], 0, 8, 1, 9, 2, 10, 3, 11,
                             "punpcklbw");
  case X86::BI__builtin_ia32_punpcklwd:
    return EmitShuffleVector(Ops[0], Ops[1], 0, 4, 1, 5, "punpcklwd");
  case X86::BI__builtin_ia32_punpckldq:
    return EmitShuffleVector(Ops[0], Ops[1], 0, 2, "punpckldq");
  case X86::BI__builtin_ia32_punpckldq128:
    return EmitShuffleVector(Ops[0], Ops[1], 0, 4, 1, 5, "punpckldq");
  case X86::BI__builtin_ia32_pslldi128: 
  case X86::BI__builtin_ia32_psllqi128:
  case X86::BI__builtin_ia32_psllwi128: 
  case X86::BI__builtin_ia32_psradi128:
  case X86::BI__builtin_ia32_psrawi128:
  case X86::BI__builtin_ia32_psrldi128:
  case X86::BI__builtin_ia32_psrlqi128:
  case X86::BI__builtin_ia32_psrlwi128: {
    Ops[1] = Builder.CreateZExt(Ops[1], llvm::Type::Int64Ty, "zext");
    const llvm::Type *Ty = llvm::VectorType::get(llvm::Type::Int64Ty, 2);
    llvm::Value *Zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
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
    Ops[1] = Builder.CreateZExt(Ops[1], llvm::Type::Int64Ty, "zext");
    const llvm::Type *Ty = llvm::VectorType::get(llvm::Type::Int64Ty, 1);
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
  case X86::BI__builtin_ia32_pshuflw: {
    unsigned i = cast<ConstantInt>(Ops[1])->getZExtValue();
    return EmitShuffleVector(Ops[0], Ops[0], 
                             i & 0x3, (i & 0xc) >> 2,
                             (i & 0x30) >> 4, (i & 0xc0) >> 6, 4, 5, 6, 7,
                             "pshuflw");
  }
  case X86::BI__builtin_ia32_pshufhw: {
    unsigned i = cast<ConstantInt>(Ops[1])->getZExtValue();
    return EmitShuffleVector(Ops[0], Ops[0], 0, 1, 2, 3,
                             4 + (i & 0x3), 4 + ((i & 0xc) >> 2),
                             4 + ((i & 0x30) >> 4), 4 + ((i & 0xc0) >> 6),
                             "pshufhw");
  }
  case X86::BI__builtin_ia32_pshufd: {
    unsigned i = cast<ConstantInt>(Ops[1])->getZExtValue();
    return EmitShuffleVector(Ops[0], Ops[0], 
                             i & 0x3, (i & 0xc) >> 2,
                             (i & 0x30) >> 4, (i & 0xc0) >> 6,
                             "pshufd");
  }
  case X86::BI__builtin_ia32_vec_init_v4hi:
  case X86::BI__builtin_ia32_vec_init_v8qi:
  case X86::BI__builtin_ia32_vec_init_v2si:
    return EmitVector(&Ops[0], Ops.size());
  case X86::BI__builtin_ia32_vec_ext_v2si:
  case X86::BI__builtin_ia32_vec_ext_v2di:
  case X86::BI__builtin_ia32_vec_ext_v4sf:
  case X86::BI__builtin_ia32_vec_ext_v4si:
  case X86::BI__builtin_ia32_vec_ext_v2df:
    return Builder.CreateExtractElement(Ops[0], Ops[1], "result");
  case X86::BI__builtin_ia32_cmpordss:
  case X86::BI__builtin_ia32_cmpordsd:
  case X86::BI__builtin_ia32_cmpunordss:
  case X86::BI__builtin_ia32_cmpunordsd:
  case X86::BI__builtin_ia32_cmpeqss:
  case X86::BI__builtin_ia32_cmpeqsd:
  case X86::BI__builtin_ia32_cmpltss:
  case X86::BI__builtin_ia32_cmpltsd:
  case X86::BI__builtin_ia32_cmpless:
  case X86::BI__builtin_ia32_cmplesd:
  case X86::BI__builtin_ia32_cmpneqss:
  case X86::BI__builtin_ia32_cmpneqsd:
  case X86::BI__builtin_ia32_cmpnltss:
  case X86::BI__builtin_ia32_cmpnltsd:
  case X86::BI__builtin_ia32_cmpnless:
  case X86::BI__builtin_ia32_cmpnlesd: {
    unsigned i = 0;
    const char *name = 0;
    switch (BuiltinID) {
    default: assert(0 && "Unknown compare builtin!");
    case X86::BI__builtin_ia32_cmpeqss:
    case X86::BI__builtin_ia32_cmpeqsd:
      i = 0;
      name = "cmpeq";
      break;
    case X86::BI__builtin_ia32_cmpltss:
    case X86::BI__builtin_ia32_cmpltsd:
      i = 1;
      name = "cmplt";
      break;
    case X86::BI__builtin_ia32_cmpless:
    case X86::BI__builtin_ia32_cmplesd:
      i = 2;
      name = "cmple";
      break;
    case X86::BI__builtin_ia32_cmpunordss:
    case X86::BI__builtin_ia32_cmpunordsd:
      i = 3;
      name = "cmpunord";
      break;
    case X86::BI__builtin_ia32_cmpneqss:
    case X86::BI__builtin_ia32_cmpneqsd:
      i = 4;
      name = "cmpneq";
      break;
    case X86::BI__builtin_ia32_cmpnltss:
    case X86::BI__builtin_ia32_cmpnltsd:
      i = 5;
      name = "cmpntl";
      break;
    case X86::BI__builtin_ia32_cmpnless:
    case X86::BI__builtin_ia32_cmpnlesd:
      i = 6;
      name = "cmpnle";
      break;
    case X86::BI__builtin_ia32_cmpordss:
    case X86::BI__builtin_ia32_cmpordsd:
      i = 7;
      name = "cmpord";
      break;
    }

    llvm::Function *F;
    if (cast<llvm::VectorType>(Ops[0]->getType())->getElementType() ==
        llvm::Type::FloatTy)
      F = CGM.getIntrinsic(Intrinsic::x86_sse_cmp_ss);
    else
      F = CGM.getIntrinsic(Intrinsic::x86_sse2_cmp_sd);

    Ops.push_back(llvm::ConstantInt::get(llvm::Type::Int8Ty, i));
    return Builder.CreateCall(F, &Ops[0], &Ops[0] + Ops.size(), name);
  }
  case X86::BI__builtin_ia32_ldmxcsr: {
    llvm::Type *PtrTy = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
    Value *One = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1);
    Value *Tmp = Builder.CreateAlloca(llvm::Type::Int32Ty, One, "tmp");
    Builder.CreateStore(Ops[0], Tmp);
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_sse_ldmxcsr),
                              Builder.CreateBitCast(Tmp, PtrTy));
  }
  case X86::BI__builtin_ia32_stmxcsr: {
    llvm::Type *PtrTy = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
    Value *One = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1);
    Value *Tmp = Builder.CreateAlloca(llvm::Type::Int32Ty, One, "tmp");
    One = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_sse_stmxcsr),
                             Builder.CreateBitCast(Tmp, PtrTy));
    return Builder.CreateLoad(Tmp, "stmxcsr");
  }
  case X86::BI__builtin_ia32_cmpordps:
  case X86::BI__builtin_ia32_cmpordpd:
  case X86::BI__builtin_ia32_cmpunordps:
  case X86::BI__builtin_ia32_cmpunordpd:
  case X86::BI__builtin_ia32_cmpeqps: 
  case X86::BI__builtin_ia32_cmpeqpd: 
  case X86::BI__builtin_ia32_cmpltps: 
  case X86::BI__builtin_ia32_cmpltpd: 
  case X86::BI__builtin_ia32_cmpleps:
  case X86::BI__builtin_ia32_cmplepd:
  case X86::BI__builtin_ia32_cmpneqps:
  case X86::BI__builtin_ia32_cmpneqpd:
  case X86::BI__builtin_ia32_cmpngtps:
  case X86::BI__builtin_ia32_cmpngtpd:
  case X86::BI__builtin_ia32_cmpnltps: 
  case X86::BI__builtin_ia32_cmpnltpd: 
  case X86::BI__builtin_ia32_cmpgtps:
  case X86::BI__builtin_ia32_cmpgtpd:
  case X86::BI__builtin_ia32_cmpgeps:
  case X86::BI__builtin_ia32_cmpgepd:
  case X86::BI__builtin_ia32_cmpngeps:
  case X86::BI__builtin_ia32_cmpngepd:
  case X86::BI__builtin_ia32_cmpnleps: 
  case X86::BI__builtin_ia32_cmpnlepd: {
    unsigned i = 0;
    const char *name = 0;
    bool ShouldSwap = false;
    switch (BuiltinID) {
    default: assert(0 && "Unknown compare builtin!");
    case X86::BI__builtin_ia32_cmpeqps:
    case X86::BI__builtin_ia32_cmpeqpd:    i = 0; name = "cmpeq"; break;
    case X86::BI__builtin_ia32_cmpltps:
    case X86::BI__builtin_ia32_cmpltpd:    i = 1; name = "cmplt"; break;
    case X86::BI__builtin_ia32_cmpleps:
    case X86::BI__builtin_ia32_cmplepd:    i = 2; name = "cmple"; break;
    case X86::BI__builtin_ia32_cmpunordps:
    case X86::BI__builtin_ia32_cmpunordpd: i = 3; name = "cmpunord"; break;
    case X86::BI__builtin_ia32_cmpneqps:
    case X86::BI__builtin_ia32_cmpneqpd:   i = 4; name = "cmpneq"; break;
    case X86::BI__builtin_ia32_cmpnltps:
    case X86::BI__builtin_ia32_cmpnltpd:   i = 5; name = "cmpntl"; break;
    case X86::BI__builtin_ia32_cmpnleps:
    case X86::BI__builtin_ia32_cmpnlepd:   i = 6; name = "cmpnle"; break;
    case X86::BI__builtin_ia32_cmpordps:
    case X86::BI__builtin_ia32_cmpordpd:   i = 7; name = "cmpord"; break;
    case X86::BI__builtin_ia32_cmpgtps:
    case X86::BI__builtin_ia32_cmpgtpd:
      ShouldSwap = true;
      i = 1;
      name = "cmpgt";
      break;
    case X86::BI__builtin_ia32_cmpgeps:
    case X86::BI__builtin_ia32_cmpgepd:
      i = 2;
      name = "cmpge";
      ShouldSwap = true;
      break;
    case X86::BI__builtin_ia32_cmpngtps:
    case X86::BI__builtin_ia32_cmpngtpd:
      i = 5;
      name = "cmpngt";
      ShouldSwap = true;
      break;
    case X86::BI__builtin_ia32_cmpngeps:
    case X86::BI__builtin_ia32_cmpngepd:
      i = 6;
      name = "cmpnge";
      ShouldSwap = true;
      break;
    }

    if (ShouldSwap)
      std::swap(Ops[0], Ops[1]);

    llvm::Function *F;
    if (cast<llvm::VectorType>(Ops[0]->getType())->getElementType() ==
        llvm::Type::FloatTy)
      F = CGM.getIntrinsic(Intrinsic::x86_sse_cmp_ps);
    else
      F = CGM.getIntrinsic(Intrinsic::x86_sse2_cmp_pd);
    
    Ops.push_back(llvm::ConstantInt::get(llvm::Type::Int8Ty, i));
    return Builder.CreateCall(F, &Ops[0], &Ops[0] + Ops.size(), name);
  }
  case X86::BI__builtin_ia32_movss:
    return EmitShuffleVector(Ops[0], Ops[1], 4, 1, 2, 3, "movss");
  case X86::BI__builtin_ia32_shufps: {
    unsigned i = cast<ConstantInt>(Ops[2])->getZExtValue();
    return EmitShuffleVector(Ops[0], Ops[1], 
                             i & 0x3, (i & 0xc) >> 2, 
                             ((i & 0x30) >> 4) + 4, 
                             ((i & 0xc0) >> 6) + 4, "shufps");
  }
  case X86::BI__builtin_ia32_shufpd: {
    unsigned i = cast<ConstantInt>(Ops[2])->getZExtValue();
    return EmitShuffleVector(Ops[0], Ops[1], i & 1, (i & 2) + 2, "shufpd");
  }
  case X86::BI__builtin_ia32_punpcklbw128:
    return EmitShuffleVector(Ops[0], Ops[1], 0, 16, 1, 17, 2, 18, 3, 19,
                                             4, 20, 5, 21, 6, 22, 7, 23,
                                             "punpcklbw");
  case X86::BI__builtin_ia32_punpcklwd128:
    return EmitShuffleVector(Ops[0], Ops[1], 0, 8, 1, 9, 2, 10, 3, 11,
                             "punpcklwd");
  case X86::BI__builtin_ia32_movlhps:
    return EmitShuffleVector(Ops[0], Ops[1], 0, 1, 4, 5, "movlhps");
  case X86::BI__builtin_ia32_movhlps:
    return EmitShuffleVector(Ops[0], Ops[1], 6, 7, 2, 3, "movhlps");
  case X86::BI__builtin_ia32_unpckhps:
    return EmitShuffleVector(Ops[0], Ops[1], 2, 6, 3, 7, "unpckhps");
  case X86::BI__builtin_ia32_unpcklps:
    return EmitShuffleVector(Ops[0], Ops[1], 0, 4, 1, 5, "unpcklps");
  case X86::BI__builtin_ia32_movqv4si: {
    llvm::Type *Ty = llvm::VectorType::get(llvm::Type::Int64Ty, 2);
    return Builder.CreateBitCast(Ops[0], Ty);
  }
  case X86::BI__builtin_ia32_loadlps:
  case X86::BI__builtin_ia32_loadhps: {
    // FIXME: This should probably be represented as 
    // shuffle (dst, (v4f32 (insert undef, (load i64), 0)), shuf mask hi/lo)
    const llvm::Type *EltTy = llvm::Type::DoubleTy;
    const llvm::Type *VecTy = llvm::VectorType::get(EltTy, 2);
    const llvm::Type *OrigTy = Ops[0]->getType();
    unsigned Index = BuiltinID == X86::BI__builtin_ia32_loadlps ? 0 : 1;
    llvm::Value *Idx = llvm::ConstantInt::get(llvm::Type::Int32Ty, Index);
    Ops[1] = Builder.CreateBitCast(Ops[1], llvm::PointerType::getUnqual(EltTy));
    Ops[1] = Builder.CreateLoad(Ops[1], "tmp");
    Ops[0] = Builder.CreateBitCast(Ops[0], VecTy, "cast");
    Ops[0] = Builder.CreateInsertElement(Ops[0], Ops[1], Idx, "loadps");
    return Builder.CreateBitCast(Ops[0], OrigTy, "loadps");
  }
  case X86::BI__builtin_ia32_storehps:
  case X86::BI__builtin_ia32_storelps: {
    const llvm::Type *EltTy = llvm::Type::Int64Ty;
    llvm::Type *PtrTy = llvm::PointerType::getUnqual(EltTy);
    llvm::Type *VecTy = llvm::VectorType::get(EltTy, 2);
    
    // cast val v2i64
    Ops[1] = Builder.CreateBitCast(Ops[1], VecTy, "cast");
    
    // extract (0, 1)
    unsigned Index = BuiltinID == X86::BI__builtin_ia32_storelps ? 0 : 1;
    llvm::Value *Idx = llvm::ConstantInt::get(llvm::Type::Int32Ty, Index);
    Ops[1] = Builder.CreateExtractElement(Ops[1], Idx, "extract");

    // cast pointer to i64 & store
    Ops[0] = Builder.CreateBitCast(Ops[0], PtrTy);
    return Builder.CreateStore(Ops[1], Ops[0]);
  }
  case X86::BI__builtin_ia32_loadlv4si: {
    // load i64
    const llvm::Type *EltTy = llvm::Type::Int64Ty;
    llvm::Type *PtrTy = llvm::PointerType::getUnqual(EltTy);
    Ops[0] = Builder.CreateBitCast(Ops[0], PtrTy);
    Ops[0] = Builder.CreateLoad(Ops[0], "load");
    
    // scalar to vector: insert i64 into 2 x i64 undef
    llvm::Type *VecTy = llvm::VectorType::get(EltTy, 2);
    llvm::Value *Zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
    Ops[0] = Builder.CreateInsertElement(llvm::UndefValue::get(VecTy),
                                         Ops[0], Zero, "s2v");

    // shuffle into zero vector.
    std::vector<llvm::Constant *>Elts;
    Elts.resize(2, llvm::ConstantInt::get(EltTy, 0));
    llvm::Value *ZV = ConstantVector::get(Elts);
    Ops[0] = EmitShuffleVector(ZV, Ops[0], 2, 1, "loadl");
    
    // bitcast to result.
    return Builder.CreateBitCast(Ops[0], 
                                 llvm::VectorType::get(llvm::Type::Int32Ty, 4));
  }
  case X86::BI__builtin_ia32_vec_set_v4hi:
  case X86::BI__builtin_ia32_vec_set_v8hi:
    return Builder.CreateInsertElement(Ops[0], Ops[1], Ops[2], "pinsrw");
  case X86::BI__builtin_ia32_andps:
  case X86::BI__builtin_ia32_andpd:
  case X86::BI__builtin_ia32_andnps:
  case X86::BI__builtin_ia32_andnpd:
  case X86::BI__builtin_ia32_orps:
  case X86::BI__builtin_ia32_orpd:
  case X86::BI__builtin_ia32_xorpd:
  case X86::BI__builtin_ia32_xorps: {
    const llvm::Type *ITy = llvm::VectorType::get(llvm::Type::Int32Ty, 4);
    const llvm::Type *FTy = Ops[0]->getType();
    Ops[0] = Builder.CreateBitCast(Ops[0], ITy, "bitcast");
    Ops[1] = Builder.CreateBitCast(Ops[1], ITy, "bitcast");
    switch (BuiltinID) {
    case X86::BI__builtin_ia32_andps:
      Ops[0] = Builder.CreateAnd(Ops[0], Ops[1], "andps");
      break;
    case X86::BI__builtin_ia32_andpd:
      Ops[0] = Builder.CreateAnd(Ops[0], Ops[1], "andpd");
      break;
    case X86::BI__builtin_ia32_andnps:
      Ops[0] = Builder.CreateNot(Ops[0], "not");
      Ops[0] = Builder.CreateAnd(Ops[0], Ops[1], "andnps");
      break;
    case X86::BI__builtin_ia32_andnpd:
      Ops[0] = Builder.CreateNot(Ops[0], "not");
      Ops[0] = Builder.CreateAnd(Ops[0], Ops[1], "andnpd");
      break;
    case X86::BI__builtin_ia32_orps:
      Ops[0] = Builder.CreateOr(Ops[0], Ops[1], "orps");
      break;
    case X86::BI__builtin_ia32_orpd:
      Ops[0] = Builder.CreateOr(Ops[0], Ops[1], "orpd");
      break;
    case X86::BI__builtin_ia32_xorps:
      Ops[0] = Builder.CreateXor(Ops[0], Ops[1], "xorps");
      break;
    case X86::BI__builtin_ia32_xorpd:
      Ops[0] = Builder.CreateXor(Ops[0], Ops[1], "xorpd");
      break;
    }
    return Builder.CreateBitCast(Ops[0], FTy, "bitcast");
  }
  }
}

Value *CodeGenFunction::EmitPPCBuiltinExpr(unsigned BuiltinID, 
                                           const CallExpr *E) {
  switch (BuiltinID) {
  default: return 0;
  }
}  
