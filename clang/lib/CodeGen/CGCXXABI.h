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
  class Type;
  class Value;
}

namespace clang {
  class CastExpr;
  class CXXMethodDecl;
  class CXXRecordDecl;
  class FieldDecl;
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

  /// Find the LLVM type used to represent the given member pointer
  /// type.
  virtual const llvm::Type *
  ConvertMemberPointerType(const MemberPointerType *MPT);

  /// Load a member function from an object and a member function
  /// pointer.  Apply the this-adjustment and set 'This' to the
  /// adjusted value.
  virtual llvm::Value *
  EmitLoadOfMemberFunctionPointer(CodeGenFunction &CGF,
                                  llvm::Value *&This,
                                  llvm::Value *MemPtr,
                                  const MemberPointerType *MPT);

  /// Perform a derived-to-base or base-to-derived member pointer
  /// conversion.
  virtual llvm::Value *EmitMemberPointerConversion(CodeGenFunction &CGF,
                                                   const CastExpr *E,
                                                   llvm::Value *Src);

  /// Perform a derived-to-base or base-to-derived member pointer
  /// conversion on a constant member pointer.
  virtual llvm::Constant *EmitMemberPointerConversion(llvm::Constant *C,
                                                      const CastExpr *E);

  /// Return true if the given member pointer can be zero-initialized
  /// (in the C++ sense) with an LLVM zeroinitializer.
  virtual bool isZeroInitializable(const MemberPointerType *MPT);

  /// Create a null member pointer of the given type.
  virtual llvm::Constant *EmitNullMemberPointer(const MemberPointerType *MPT);

  /// Create a member pointer for the given method.
  virtual llvm::Constant *EmitMemberPointer(const CXXMethodDecl *MD);

  /// Create a member pointer for the given field.
  virtual llvm::Constant *EmitMemberPointer(const FieldDecl *FD);

  /// Emit a comparison between two member pointers.  Returns an i1.
  virtual llvm::Value *
  EmitMemberPointerComparison(CodeGenFunction &CGF,
                              llvm::Value *L,
                              llvm::Value *R,
                              const MemberPointerType *MPT,
                              bool Inequality);

  /// Determine if a member pointer is non-null.  Returns an i1.
  virtual llvm::Value *
  EmitMemberPointerIsNotNull(CodeGenFunction &CGF,
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
