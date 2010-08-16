//===----- CGCXXABI.h - Interface to C++ ABIs -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for C++ code generation. Concrete subclasses
// of this implement code generation for specific C++ ABIs.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CXXABI_H
#define CLANG_CODEGEN_CXXABI_H

namespace clang {
namespace CodeGen {
  class CodeGenModule;
  class MangleContext;

/// Implements C++ ABI-specific code generation functions.
class CGCXXABI {
public:
  virtual ~CGCXXABI();

  /// Gets the mangle context.
  virtual MangleContext &getMangleContext() = 0;
};

/// Creates an instance of a C++ ABI class.
CGCXXABI *CreateItaniumCXXABI(CodeGenModule &CGM);
CGCXXABI *CreateMicrosoftCXXABI(CodeGenModule &CGM);
}
}

#endif
