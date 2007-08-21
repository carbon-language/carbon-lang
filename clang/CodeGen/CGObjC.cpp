//===---- CGBuiltin.cpp - Emit LLVM Code for builtins ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Anders Carlsson and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Objective-C code as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/Expr.h"
#include "llvm/Constant.h"

using namespace clang;
using namespace CodeGen;

RValue CodeGenFunction::EmitObjCStringLiteral(const ObjCStringLiteral* E)
{
  std::string S(E->getString()->getStrData(), E->getString()->getByteLength());
  
  return RValue::get(CGM.GetAddrOfConstantCFString(S));
}

