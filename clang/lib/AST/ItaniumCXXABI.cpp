//===------- ItaniumCXXABI.cpp - AST support for the Itanium C++ ABI ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides C++ AST support targetting the Itanium C++ ABI, which is
// documented at:
//  http://www.codesourcery.com/public/cxx-abi/abi.html
//  http://www.codesourcery.com/public/cxx-abi/abi-eh.html
//
// It also supports the closely-related ARM C++ ABI, documented at:
// http://infocenter.arm.com/help/topic/com.arm.doc.ihi0041c/IHI0041C_cppabi.pdf
//
//===----------------------------------------------------------------------===//

#include "CXXABI.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"

using namespace clang;

namespace {
class ItaniumCXXABI : public CXXABI {
protected:
  ASTContext &Context;
public:
  ItaniumCXXABI(ASTContext &Ctx) : Context(Ctx) { }

  unsigned getMemberPointerSize(const MemberPointerType *MPT) const {
    QualType Pointee = MPT->getPointeeType();
    if (Pointee->isFunctionType()) return 2;
    return 1;
  }
};

class ARMCXXABI : public ItaniumCXXABI {
public:
  ARMCXXABI(ASTContext &Ctx) : ItaniumCXXABI(Ctx) { }
};
}

CXXABI *clang::CreateItaniumCXXABI(ASTContext &Ctx) {
  return new ItaniumCXXABI(Ctx);
}

CXXABI *clang::CreateARMCXXABI(ASTContext &Ctx) {
  return new ARMCXXABI(Ctx);
}
