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

#include "CodeGenFunction.h"

namespace llvm {
  class Constant;
  class Type;
  class Value;

  template <class T> class SmallVectorImpl;
}

namespace clang {
  class CastExpr;
  class CXXConstructorDecl;
  class CXXDestructorDecl;
  class CXXMethodDecl;
  class CXXRecordDecl;
  class FieldDecl;

namespace CodeGen {
  class CodeGenFunction;
  class CodeGenModule;
  class MangleContext;

/// Implements C++ ABI-specific code generation functions.
class CGCXXABI {
protected:
  CodeGenModule &CGM;

  CGCXXABI(CodeGenModule &CGM) : CGM(CGM) {}

protected:
  ImplicitParamDecl *&getThisDecl(CodeGenFunction &CGF) {
    return CGF.CXXThisDecl;
  }
  llvm::Value *&getThisValue(CodeGenFunction &CGF) {
    return CGF.CXXThisValue;
  }

  ImplicitParamDecl *&getVTTDecl(CodeGenFunction &CGF) {
    return CGF.CXXVTTDecl;
  }
  llvm::Value *&getVTTValue(CodeGenFunction &CGF) {
    return CGF.CXXVTTValue;
  }

  /// Build a parameter variable suitable for 'this'.
  void BuildThisParam(CodeGenFunction &CGF, FunctionArgList &Params);

  /// Perform prolog initialization of the parameter variable suitable
  /// for 'this' emitted by BuildThisParam.
  void EmitThisParam(CodeGenFunction &CGF);

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

  /// Build the signature of the given constructor variant by adding
  /// any required parameters.  For convenience, ResTy has been
  /// initialized to 'void', and ArgTys has been initialized with the
  /// type of 'this' (although this may be changed by the ABI) and
  /// will have the formal parameters added to it afterwards.
  ///
  /// If there are ever any ABIs where the implicit parameters are
  /// intermixed with the formal parameters, we can address those
  /// then.
  virtual void BuildConstructorSignature(const CXXConstructorDecl *Ctor,
                                         CXXCtorType T,
                                         CanQualType &ResTy,
                               llvm::SmallVectorImpl<CanQualType> &ArgTys) = 0;

  /// Build the signature of the given destructor variant by adding
  /// any required parameters.  For convenience, ResTy has been
  /// initialized to 'void' and ArgTys has been initialized with the
  /// type of 'this' (although this may be changed by the ABI).
  virtual void BuildDestructorSignature(const CXXDestructorDecl *Dtor,
                                        CXXDtorType T,
                                        CanQualType &ResTy,
                               llvm::SmallVectorImpl<CanQualType> &ArgTys) = 0;

  /// Build the ABI-specific portion of the parameter list for a
  /// function.  This generally involves a 'this' parameter and
  /// possibly some extra data for constructors and destructors.
  ///
  /// ABIs may also choose to override the return type, which has been
  /// initialized with the formal return type of the function.
  virtual void BuildInstanceFunctionParams(CodeGenFunction &CGF,
                                           QualType &ResTy,
                                           FunctionArgList &Params) = 0;

  /// Emit the ABI-specific prolog for the function.
  virtual void EmitInstanceFunctionProlog(CodeGenFunction &CGF) = 0;

  virtual void EmitReturnFromThunk(CodeGenFunction &CGF,
                                   RValue RV, QualType ResultType);
};

/// Creates an instance of a C++ ABI class.
CGCXXABI *CreateARMCXXABI(CodeGenModule &CGM);
CGCXXABI *CreateItaniumCXXABI(CodeGenModule &CGM);
CGCXXABI *CreateMicrosoftCXXABI(CodeGenModule &CGM);

}
}

#endif
