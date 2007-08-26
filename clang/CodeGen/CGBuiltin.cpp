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
#include "clang/AST/Builtins.h"
#include "clang/AST/Expr.h"
#include "llvm/Constants.h"
using namespace clang;
using namespace CodeGen;

RValue CodeGenFunction::EmitBuiltinExpr(unsigned BuiltinID, const CallExpr *E) {
  switch (BuiltinID) {
  default:
    fprintf(stderr, "Unimplemented builtin!!\n");
    E->dump();

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
  }
      
  return RValue::get(0);
}
