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

namespace llvm {
  class Constant;
  class Value;
}

namespace clang {
  class CastExpr;
  class CXXMethodDecl;
  class CXXRecordDecl;
  class MemberPointerType;
  class QualType;

namespace CodeGen {
  class CodeGenFunction;
  class CodeGenModule;
  class MangleContext;

/// Implements C++ ABI-specific code generation functions.
class CGCXXABI {
protected:
  CodeGenModule &CGM;

  CGCXXABI(CodeGenModule &CGM) : CGM(CGM) {}

public:

  virtual ~CGCXXABI();

  /// Gets the mangle context.
  virtual MangleContext &getMangleContext() = 0;

  virtual llvm::Value *
  EmitLoadOfMemberFunctionPointer(CodeGenFunction &CGF,
                                  llvm::Value *&This,
                                  llvm::Value *MemPtr,
                                  const MemberPointerType *MPT);

  virtual llvm::Value *
  EmitMemberFunctionPointerConversion(CodeGenFunction &CGF,
                                      const CastExpr *E,
                                      llvm::Value *Src);

  // Manipulations on constant expressions.

  /// \brief Returns true if zero-initializing the given type requires
  /// a constant other than the LLVM null value.
  virtual bool RequiresNonZeroInitializer(QualType T);
  virtual bool RequiresNonZeroInitializer(const CXXRecordDecl *D);

  virtual llvm::Constant *
  EmitMemberFunctionPointerConversion(llvm::Constant *C,
                                      const CastExpr *E);

  virtual llvm::Constant *
  EmitNullMemberFunctionPointer(const MemberPointerType *MPT);

  virtual llvm::Constant *EmitMemberFunctionPointer(const CXXMethodDecl *MD);

  virtual llvm::Value *
  EmitMemberFunctionPointerComparison(CodeGenFunction &CGF,
                                      llvm::Value *L,
                                      llvm::Value *R,
                                      const MemberPointerType *MPT,
                                      bool Inequality);

  virtual llvm::Value *
  EmitMemberFunctionPointerIsNotNull(CodeGenFunction &CGF,
                                     llvm::Value *MemPtr,
                                     const MemberPointerType *MPT);
};

/// Creates an instance of a C++ ABI class.
CGCXXABI *CreateARMCXXABI(CodeGenModule &CGM);
CGCXXABI *CreateItaniumCXXABI(CodeGenModule &CGM);
CGCXXABI *CreateMicrosoftCXXABI(CodeGenModule &CGM);

}
}

#endif
