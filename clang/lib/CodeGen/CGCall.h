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

#include <llvm/ADT/FoldingSet.h>
#include "clang/AST/Type.h"

#include "CGValue.h"

namespace llvm {
  class Function;
  struct AttributeWithIndex;
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
  typedef llvm::SmallVector<llvm::AttributeWithIndex, 8> AttributeListType;

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
  class CGFunctionInfo : public llvm::FoldingSetNode {
    llvm::SmallVector<QualType, 16> ArgTypes;

  public:
    typedef llvm::SmallVector<QualType, 16>::const_iterator arg_iterator;

    CGFunctionInfo(QualType ResTy, 
                   const llvm::SmallVector<QualType, 16> &ArgTys);

    arg_iterator arg_begin() const;
    arg_iterator arg_end() const;

    QualType getReturnType() const { return ArgTypes[0]; }

    void Profile(llvm::FoldingSetNodeID &ID) {
      Profile(ID, getReturnType(), arg_begin(), arg_end());
    }
    template<class Iterator>
    static void Profile(llvm::FoldingSetNodeID &ID, 
                        QualType ResTy,
                        Iterator begin,
                        Iterator end) {
      ResTy.Profile(ID);
      for (; begin != end; ++begin)
        begin->Profile(ID);
    }
  };
}  // end namespace CodeGen
}  // end namespace clang

#endif
