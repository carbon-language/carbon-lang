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
#include "clang/AST/ASTContext.h"
#include "clang/AST/Builtins.h"
#include "clang/AST/Expr.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"

using namespace clang;
using namespace CodeGen;

RValue CodeGenFunction::EmitBuiltinExpr(unsigned BuiltinID, const CallExpr *E) {
  switch (BuiltinID) {
  default:
    if (getContext().BuiltinInfo.isLibFunction(BuiltinID))
      return EmitCallExpr(CGM.getBuiltinLibFunction(BuiltinID), E);
    
    WarnUnsupported(E, "builtin function");

    // Unknown builtin, for now just dump it out and return undef.
    if (hasAggregateLLVMType(E->getType()))
      return RValue::getAggregate(CreateTempAlloca(ConvertType(E->getType())));
    return RValue::get(llvm::UndefValue::get(ConvertType(E->getType())));
    
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
    llvm::Value *ArgValue = EmitScalarExpr(E->getArg(0));
    const llvm::Type *DestType = llvm::PointerType::get(llvm::Type::Int8Ty);
    if (ArgValue->getType() != DestType)
      ArgValue = Builder.CreateBitCast(ArgValue, DestType, 
                                       ArgValue->getNameStart());

    llvm::Intrinsic::ID inst = (BuiltinID == Builtin::BI__builtin_va_start) ? 
      llvm::Intrinsic::vastart : llvm::Intrinsic::vaend;
    llvm::Value *F = llvm::Intrinsic::getDeclaration(&CGM.getModule(), inst);
    llvm::Value *V = Builder.CreateCall(F, ArgValue);

    return RValue::get(V);
  }
  case Builtin::BI__builtin_classify_type: {
    llvm::APSInt Result(32);
    
    if (!E->isBuiltinClassifyType(Result))
      assert(0 && "Expr not __builtin_classify_type!");
    
    return RValue::get(llvm::ConstantInt::get(Result));
  }
  case Builtin::BI__builtin_constant_p: {
    llvm::APSInt Result(32);

    // FIXME: Analyze the parameter and check if it is a constant.
    Result = 0;
    
    return RValue::get(llvm::ConstantInt::get(Result));
  }
  case Builtin::BI__builtin_abs: {
    llvm::Value *ArgValue = EmitScalarExpr(E->getArg(0));   
    
    llvm::BinaryOperator *NegOp = 
      Builder.CreateNeg(ArgValue, (ArgValue->getName() + "neg").c_str());
    llvm::Value *CmpResult = 
      Builder.CreateICmpSGE(ArgValue, NegOp->getOperand(0), "abscond");
    llvm::Value *Result = 
      Builder.CreateSelect(CmpResult, ArgValue, NegOp, "abs");
    
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_expect: {
    llvm::Value *Condition = EmitScalarExpr(E->getArg(0));   
    return RValue::get(Condition);
  }
  case Builtin::BI__builtin_bswap32:
  case Builtin::BI__builtin_bswap64: {
    llvm::Value *ArgValue = EmitScalarExpr(E->getArg(0));
    const llvm::Type *ArgType = ArgValue->getType();
    llvm::Value *F = 
      llvm::Intrinsic::getDeclaration(&CGM.getModule(), 
                                      llvm::Intrinsic::bswap,
                                      &ArgType, 1);
    llvm::Value *V = Builder.CreateCall(F, ArgValue, "tmp");
      
    return RValue::get(V);      
  }
  case Builtin::BI__builtin_inff: {
    llvm::APFloat f(llvm::APFloat::IEEEsingle,
                    llvm::APFloat::fcInfinity, false);
    
    llvm::Value *V = llvm::ConstantFP::get(llvm::Type::FloatTy, f);
    return RValue::get(V);
  }
  case Builtin::BI__builtin_inf:
  // FIXME: mapping long double onto double.      
  case Builtin::BI__builtin_infl: {
    llvm::APFloat f(llvm::APFloat::IEEEdouble,
                    llvm::APFloat::fcInfinity, false);
    
    llvm::Value *V = llvm::ConstantFP::get(llvm::Type::DoubleTy, f);
    return RValue::get(V);
  }
  }
  
  return RValue::get(0);
}
