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

// FIXME: Restructure so we don't have to expose so much stuff.
#include "ABIInfo.h"

namespace llvm {
  struct AttributeWithIndex;
  class Function;
  class Type;
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
    struct ArgInfo {
      QualType type;
      ABIArgInfo info;
    };

    unsigned NumArgs;
    ArgInfo *Args;

  public:
    typedef const ArgInfo *const_arg_iterator;
    typedef ArgInfo *arg_iterator;

    CGFunctionInfo(QualType ResTy, 
                   const llvm::SmallVector<QualType, 16> &ArgTys);
    ~CGFunctionInfo() { delete[] Args; }

    const_arg_iterator arg_begin() const { return Args + 1; }
    const_arg_iterator arg_end() const { return Args + 1 + NumArgs; }
    arg_iterator arg_begin() { return Args + 1; }
    arg_iterator arg_end() { return Args + 1 + NumArgs; }

    unsigned  arg_size() const { return NumArgs; }

    QualType getReturnType() const { return Args[0].type; }

    ABIArgInfo &getReturnInfo() { return Args[0].info; }
    const ABIArgInfo &getReturnInfo() const { return Args[0].info; }

    void Profile(llvm::FoldingSetNodeID &ID) {
      getReturnType().Profile(ID);
      for (arg_iterator it = arg_begin(), ie = arg_end(); it != ie; ++it)
        it->type.Profile(ID);
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
