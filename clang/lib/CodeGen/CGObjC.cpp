//===---- CGBuiltin.cpp - Emit LLVM Code for builtins ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Objective-C code as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGObjCRuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/ExprObjC.h"
#include "llvm/Constant.h"
using namespace clang;
using namespace CodeGen;

llvm::Value *CodeGenFunction::EmitObjCStringLiteral(const ObjCStringLiteral *E){
  std::string S(E->getString()->getStrData(), E->getString()->getByteLength());
  return CGM.GetAddrOfConstantCFString(S);
}

CGObjCRuntime::~CGObjCRuntime() {}
