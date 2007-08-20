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

using namespace clang;
using namespace CodeGen;

RValue CodeGenFunction::EmitBuiltinExpr(unsigned builtinID, const CallExpr *E)
{
  switch (builtinID) {
    default:
      assert(0 && "Unknown builtin id");
  }
      
  return RValue::get(0);
}
