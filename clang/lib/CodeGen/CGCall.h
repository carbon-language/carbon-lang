//===----- CGCall.h - Encapsulate calling convention details ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGCALL_H
#define CLANG_CODEGEN_CGCALL_H

#include "clang/AST/Type.h"

#include "CGValue.h"

namespace llvm {
  class Function;
  struct ParamAttrsWithIndex;
  class Value;

  template<typename T, unsigned> class SmallVector;
}

namespace clang {
  class ASTContext;
  class Decl;
  class FunctionDecl;
  class ObjCMethodDecl;
  class VarDecl;

namespace CodeGen {
  typedef llvm::SmallVector<llvm::ParamAttrsWithIndex, 8> ParamAttrListType;

  /// CallArgList - Type for representing both the value and type of
  /// arguments in a call.
  typedef llvm::SmallVector<std::pair<RValue, QualType>, 16> CallArgList;

  /// FunctionArgList - Type for representing both the decl and type
  /// of parameters to a function. The decl must be either a
  /// ParmVarDecl or ImplicitParamDecl.
  typedef llvm::SmallVector<std::pair<const VarDecl*, QualType>, 
                            16> FunctionArgList;

  /// CGFunctionInfo - Class to encapsulate the information about a
  /// function definition.
  class CGFunctionInfo {
    /// TheDecl - The decl we are storing information for. This is
    /// either a Function or ObjCMethod Decl.
    const Decl *TheDecl;

    llvm::SmallVector<QualType, 16> ArgTypes;

  public:
    CGFunctionInfo(const FunctionDecl *FD);
    CGFunctionInfo(const ObjCMethodDecl *MD,
                   const ASTContext &Context);

    const Decl* getDecl() const { return TheDecl; }

    void constructParamAttrList(ParamAttrListType &Args) const;
  };

  /// CGCallInfo - Class to encapsulate the arguments and clang types
  /// used in a call.
  class CGCallInfo {
    QualType ResultType;
    const CallArgList &Args;

    llvm::SmallVector<QualType, 16> ArgTypes;

  public:
    CGCallInfo(QualType _ResultType, const CallArgList &Args);
    
    void constructParamAttrList(ParamAttrListType &Args) const;
  };
}  // end namespace CodeGen
}  // end namespace clang

#endif
