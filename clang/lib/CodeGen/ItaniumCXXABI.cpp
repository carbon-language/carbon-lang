//===------- ItaniumCXXABI.cpp - Emit LLVM Code from ASTs for a Module ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides C++ code generation targetting the Itanium C++ ABI.  The class
// in this file generates structures that follow the Itanium C++ ABI, which is
// documented at:
//  http://www.codesourcery.com/public/cxx-abi/abi.html
//  http://www.codesourcery.com/public/cxx-abi/abi-eh.html
//===----------------------------------------------------------------------===//

#include "CGCXXABI.h"
#include "CodeGenModule.h"
#include "Mangle.h"

using namespace clang;

namespace {
class ItaniumCXXABI : public CodeGen::CXXABI {
  CodeGen::MangleContext MangleCtx;
public:
  ItaniumCXXABI(CodeGen::CodeGenModule &CGM) :
    MangleCtx(CGM.getContext(), CGM.getDiags()) { }

  CodeGen::MangleContext &getMangleContext() {
    return MangleCtx;
  }
};
}

CodeGen::CXXABI *CodeGen::CreateItaniumCXXABI(CodeGenModule &CGM) {
  return new ItaniumCXXABI(CGM);
}

