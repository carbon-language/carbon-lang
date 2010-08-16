//===----- CXXABI.h - Interface to C++ ABIs ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for C++ AST support. Concrete
// subclasses of this implement AST support for specific C++ ABIs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_CXXABI_H
#define LLVM_CLANG_AST_CXXABI_H

namespace clang {

class ASTContext;
class MemberPointerType;

/// Implements C++ ABI-specific semantic analysis functions.
class CXXABI {
public:
  virtual ~CXXABI();

  /// Returns the size of a member pointer in multiples of the target
  /// pointer size.
  virtual unsigned getMemberPointerSize(const MemberPointerType *MPT) const = 0;
};

/// Creates an instance of a C++ ABI class.
CXXABI *CreateItaniumCXXABI(ASTContext &Ctx);
CXXABI *CreateMicrosoftCXXABI(ASTContext &Ctx);
}

#endif
