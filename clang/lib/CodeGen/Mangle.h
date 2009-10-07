//===--- Mangle.h - Mangle C++ Names ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements C++ name mangling according to the Itanium C++ ABI,
// which is used in GCC 3.2 and newer (and many compilers that are
// ABI-compatible with GCC):
//
//   http://www.codesourcery.com/public/cxx-abi/abi.html
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGEN_MANGLE_H
#define LLVM_CLANG_CODEGEN_MANGLE_H

#include "CGCXX.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
  class raw_ostream;
}

namespace clang {
  class ASTContext;
  class CXXConstructorDecl;
  class CXXDestructorDecl;
  class FunctionDecl;
  class NamedDecl;
  class VarDecl;

  class MangleContext {
    ASTContext &Context;
    
    llvm::DenseMap<const TagDecl *, uint64_t> AnonStructIds;

  public:
    explicit MangleContext(ASTContext &Context)
    : Context(Context) { }
    
    ASTContext &getASTContext() const { return Context; }
    
    uint64_t getAnonymousStructId(const TagDecl *TD) {
      std::pair<llvm::DenseMap<const TagDecl *, 
                               uint64_t>::iterator, bool> Result =
      AnonStructIds.insert(std::make_pair(TD, AnonStructIds.size()));
      return Result.first->second;
    }
  };

  bool mangleName(MangleContext &Context, const NamedDecl *D,
                  llvm::raw_ostream &os);
  void mangleThunk(MangleContext &Context, const FunctionDecl *FD, 
                   int64_t n, int64_t vn, llvm::raw_ostream &os);
  void mangleCovariantThunk(MangleContext &Context, const FunctionDecl *FD, 
                            int64_t nv_t, int64_t v_t,
                            int64_t nv_r, int64_t v_r,
                            llvm::raw_ostream &os);
  void mangleGuardVariable(MangleContext &Context, const VarDecl *D,
                           llvm::raw_ostream &os);
  void mangleCXXVtable(MangleContext &Context, QualType T, llvm::raw_ostream &os);
  void mangleCXXRtti(MangleContext &Context, QualType T, llvm::raw_ostream &os);
  void mangleCXXCtor(MangleContext &Context, const CXXConstructorDecl *D, 
                     CXXCtorType Type, llvm::raw_ostream &os);
  void mangleCXXDtor(MangleContext &Context, const CXXDestructorDecl *D, 
                     CXXDtorType Type, llvm::raw_ostream &os);  
}

#endif
