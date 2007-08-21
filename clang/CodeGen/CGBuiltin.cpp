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
#include "llvm/Constant.h"

using namespace clang;
using namespace CodeGen;

RValue CodeGenFunction::EmitBuiltinExpr(unsigned builtinID, const CallExpr *E)
{
  switch (builtinID) {
    case Builtin::BI__builtin___CFStringMakeConstantString: {
      const Expr *Arg = E->getArg(0);
      
      while (const ParenExpr *PE = dyn_cast<const ParenExpr>(Arg))
        Arg = PE->getSubExpr();
      
      const StringLiteral *Literal = cast<const StringLiteral>(Arg);
      std::string S(Literal->getStrData(), Literal->getByteLength());
      
      return RValue::get(CGM.GetAddrOfConstantCFString(S));
    }      
    default:
      assert(0 && "Unknown builtin id");
  }
      
  return RValue::get(0);
}
