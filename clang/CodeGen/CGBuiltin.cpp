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
    
    fprintf(stderr, "Unimplemented builtin!!\n");
    E->dump(getContext().SourceMgr);

    // Unknown builtin, for now just dump it out and return undef.
    if (hasAggregateLLVMType(E->getType()))
      return RValue::getAggregate(CreateTempAlloca(ConvertType(E->getType())));
    return RValue::get(llvm::UndefValue::get(ConvertType(E->getType())));
    
  case Builtin::BI__builtin___CFStringMakeConstantString: {
    const Expr *Arg = E->getArg(0);
    
    while (const ParenExpr *PE = dyn_cast<ParenExpr>(Arg))
      Arg = PE->getSubExpr();
    
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
  }
  
  return RValue::get(0);
}
