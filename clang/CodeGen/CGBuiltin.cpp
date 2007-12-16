//===---- CGBuiltin.cpp - Emit LLVM Code for builtins ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Anders Carlsson and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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

RValue CodeGenFunction::EmitBuiltinExpr(unsigned BuiltinID, const CallExpr *E) {
  switch (BuiltinID) {
  default: {
    if (getContext().BuiltinInfo.isLibFunction(BuiltinID))
      return EmitCallExpr(CGM.getBuiltinLibFunction(BuiltinID), E);
  
    // See if we have a target specific intrinsic.
    Intrinsic::ID IntrinsicID;
    const char *TargetPrefix = Target.getTargetPrefix();
    const char *BuiltinName = getContext().BuiltinInfo.GetName(BuiltinID);
#define GET_LLVM_INTRINSIC_FOR_GCC_BUILTIN
#include "llvm/Intrinsics.gen"
#undef GET_LLVM_INTRINSIC_FOR_GCC_BUILTIN
    
    if (IntrinsicID != Intrinsic::not_intrinsic) {
      SmallVector<Value*, 16> Args;
      
      Function *F = Intrinsic::getDeclaration(&CGM.getModule(), IntrinsicID);
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
  case Builtin::BI__builtin_va_start:
  case Builtin::BI__builtin_va_end: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    const llvm::Type *DestType = llvm::PointerType::get(llvm::Type::Int8Ty);
    if (ArgValue->getType() != DestType)
      ArgValue = Builder.CreateBitCast(ArgValue, DestType, 
                                       ArgValue->getNameStart());

    Intrinsic::ID inst = (BuiltinID == Builtin::BI__builtin_va_start) ? 
      Intrinsic::vastart : Intrinsic::vaend;
    Value *F = Intrinsic::getDeclaration(&CGM.getModule(), inst);
    Value *V = Builder.CreateCall(F, ArgValue);

    return RValue::get(V);
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
  case Builtin::BI__builtin_expect:
    return RValue::get(EmitScalarExpr(E->getArg(0)));
  case Builtin::BI__builtin_bswap32:
  case Builtin::BI__builtin_bswap64: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    const llvm::Type *ArgType = ArgValue->getType();
    Value *F = Intrinsic::getDeclaration(&CGM.getModule(), Intrinsic::bswap,
                                         &ArgType, 1);
    return RValue::get(Builder.CreateCall(F, ArgValue, "tmp"));
  }
  case Builtin::BI__builtin_inff: {
    APFloat f(APFloat::IEEEsingle, APFloat::fcInfinity, false);
    return RValue::get(ConstantFP::get(llvm::Type::FloatTy, f));
  }
  case Builtin::BI__builtin_inf:
  // FIXME: mapping long double onto double.      
  case Builtin::BI__builtin_infl: {
    APFloat f(APFloat::IEEEdouble, APFloat::fcInfinity, false);
    return RValue::get(ConstantFP::get(llvm::Type::DoubleTy, f));
  }
  }
  return RValue::get(0);
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
  case X86::BI__builtin_ia32_pand:
    return Builder.CreateAnd(Ops[0], Ops[1], "pand");
  case X86::BI__builtin_ia32_por:
    return Builder.CreateAnd(Ops[0], Ops[1], "por");
  case X86::BI__builtin_ia32_pxor:
    return Builder.CreateAnd(Ops[0], Ops[1], "pxor");
  case X86::BI__builtin_ia32_pandn: {
    Ops[0] = Builder.CreateNot(Ops[0], "tmp");
    return Builder.CreateAnd(Ops[0], Ops[1], "pandn");
  }
  case X86::BI__builtin_ia32_paddb:
  case X86::BI__builtin_ia32_paddd:
  case X86::BI__builtin_ia32_paddq:
  case X86::BI__builtin_ia32_paddw:
  case X86::BI__builtin_ia32_addps:
    return Builder.CreateAdd(Ops[0], Ops[1], "add");
  case X86::BI__builtin_ia32_psubb:
  case X86::BI__builtin_ia32_psubd:
  case X86::BI__builtin_ia32_psubq:
  case X86::BI__builtin_ia32_psubw:
  case X86::BI__builtin_ia32_subps:
    return Builder.CreateSub(Ops[0], Ops[1], "sub");
  case X86::BI__builtin_ia32_divps:
    return Builder.CreateFDiv(Ops[0], Ops[1], "divps");
  case X86::BI__builtin_ia32_pmullw:
    return Builder.CreateMul(Ops[0], Ops[1], "pmul");
  case X86::BI__builtin_ia32_punpckhbw:
    return EmitShuffleVector(Ops[0], Ops[1],
                             4, 12, 5, 13, 6, 14, 7, 15,
                             "punpckhbw");
  case X86::BI__builtin_ia32_punpckhwd:
    return EmitShuffleVector(Ops[0], Ops[1],
                             2, 6, 3, 7,
                             "punpckhwd");
  case X86::BI__builtin_ia32_punpckhdq:
    return EmitShuffleVector(Ops[0], Ops[1],
                             1, 3,
                             "punpckhdq");
  case X86::BI__builtin_ia32_punpcklbw:
    return EmitShuffleVector(Ops[0], Ops[1],
                             0, 8, 1, 9, 2, 10, 3, 11,
                             "punpcklbw");
  case X86::BI__builtin_ia32_punpcklwd:
    return EmitShuffleVector(Ops[0], Ops[1],
                             0, 4, 1, 5,
                             "punpcklwd");
  case X86::BI__builtin_ia32_punpckldq:
    return EmitShuffleVector(Ops[0], Ops[1],
                             0, 2,
                             "punpckldq");
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
    }
    
    llvm::Function *F = Intrinsic::getDeclaration(&CGM.getModule(), ID);
    return Builder.CreateCall(F, &Ops[0], &Ops[0] + Ops.size(), name);  
  }
  case X86::BI__builtin_ia32_pshufd: {
    int i = cast<ConstantInt>(Ops[1])->getZExtValue();
   
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
    return Builder.CreateExtractElement(Ops[0], Ops[1], "result");
  case X86::BI__builtin_ia32_cmpordss:
  case X86::BI__builtin_ia32_cmpunordss:
  case X86::BI__builtin_ia32_cmpeqss: 
  case X86::BI__builtin_ia32_cmpltss: 
  case X86::BI__builtin_ia32_cmpless:
  case X86::BI__builtin_ia32_cmpneqss:
  case X86::BI__builtin_ia32_cmpnltss: 
  case X86::BI__builtin_ia32_cmpnless: {
    int i = 0;
    const char *name = 0;
    switch (BuiltinID) {
    default: assert(0 && "Unknown compare builtin!");
    case X86::BI__builtin_ia32_cmpeqss:
      i = 0;
      name = "cmpeqss";
      break;
    case X86::BI__builtin_ia32_cmpltss:
      i = 1;
      name = "cmpltss";
      break;
    case X86::BI__builtin_ia32_cmpless:
      i = 2;
      name = "cmpless";
      break;
    case X86::BI__builtin_ia32_cmpunordss:
      i = 3;
      name = "cmpunordss";
      break;
    case X86::BI__builtin_ia32_cmpneqss:
      i = 4;
      name = "cmpneqss";
      break;
    case X86::BI__builtin_ia32_cmpnltss:
      i = 5;
      name = "cmpntlss";
      break;
    case X86::BI__builtin_ia32_cmpnless:
      i = 6;
      name = "cmpnless";
      break;
    case X86::BI__builtin_ia32_cmpordss:
      i = 7;
      name = "cmpordss";
      break;
    }

    Ops.push_back(llvm::ConstantInt::get(llvm::Type::Int8Ty, i));
    
    llvm::Function *F = Intrinsic::getDeclaration(&CGM.getModule(),
                                                  Intrinsic::x86_sse_cmp_ss);
    return Builder.CreateCall(F, &Ops[0], &Ops[0] + Ops.size(), name);
  }
  case X86::BI__builtin_ia32_cmpordps:
  case X86::BI__builtin_ia32_cmpunordps:
  case X86::BI__builtin_ia32_cmpeqps: 
  case X86::BI__builtin_ia32_cmpltps: 
  case X86::BI__builtin_ia32_cmpleps:
  case X86::BI__builtin_ia32_cmpneqps:
  case X86::BI__builtin_ia32_cmpngtps:
  case X86::BI__builtin_ia32_cmpnltps: 
  case X86::BI__builtin_ia32_cmpgtps:
  case X86::BI__builtin_ia32_cmpgeps:
  case X86::BI__builtin_ia32_cmpngeps:
  case X86::BI__builtin_ia32_cmpnleps: {
    int i = 0;
    const char *name = 0;
    bool ShouldSwap = false;
    switch (BuiltinID) {
    default: assert(0 && "Unknown compare builtin!");
    case X86::BI__builtin_ia32_cmpeqps:
      i = 0;
      name = "cmpeqps";
      break;
    case X86::BI__builtin_ia32_cmpltps:
      i = 1;
      name = "cmpltps";
      break;
    case X86::BI__builtin_ia32_cmpleps:
      i = 2;
      name = "cmpleps";
      break;
    case X86::BI__builtin_ia32_cmpunordps:
      i = 3;
      name = "cmpunordps";
      break;
    case X86::BI__builtin_ia32_cmpneqps:
      i = 4;
      name = "cmpneqps";
      break;
    case X86::BI__builtin_ia32_cmpnltps:
      i = 5;
      name = "cmpntlps";
      break;
    case X86::BI__builtin_ia32_cmpnleps:
      i = 6;
      name = "cmpnleps";
      break;
    case X86::BI__builtin_ia32_cmpordps:
      i = 7;
      name = "cmpordps";
      break;
    case X86::BI__builtin_ia32_cmpgtps:
      i = 1;
      name = "cmpgtps";
      ShouldSwap = true;
      break;
    case X86::BI__builtin_ia32_cmpgeps:
      i = 2;
      name = "cmpgeps";
      ShouldSwap = true;
      break;
    case X86::BI__builtin_ia32_cmpngtps:
      i = 5;
      name = "cmpngtps";
      ShouldSwap = true;
      break;
    case X86::BI__builtin_ia32_cmpngeps:
      i = 6;
      name = "cmpngeps";
      ShouldSwap = true;
      break;
    }

    if (ShouldSwap)
      std::swap(Ops[0], Ops[1]);
    
    Ops.push_back(llvm::ConstantInt::get(llvm::Type::Int8Ty, i));
    
    llvm::Function *F = Intrinsic::getDeclaration(&CGM.getModule(),
                                                  Intrinsic::x86_sse_cmp_ps);
    return Builder.CreateCall(F, &Ops[0], &Ops[0] + Ops.size(), name);
  }
  case X86::BI__builtin_ia32_movss:
    return EmitShuffleVector(Ops[0], Ops[1], 4, 1, 2, 3, "movss");
  case X86::BI__builtin_ia32_shufps:
    int i = cast<ConstantInt>(Ops[2])->getZExtValue();

    return EmitShuffleVector(Ops[0], Ops[1], 
                             i & 0x3, (i & 0xc) >> 2, 
                             ((i & 0x30) >> 4) + 4, 
                             ((i & 0x60) >> 6) + 4, "shufps");
  }
}

Value *CodeGenFunction::EmitPPCBuiltinExpr(unsigned BuiltinID, 
                                           const CallExpr *E) {
  switch (BuiltinID) {
  default: return 0;
  }
}  
