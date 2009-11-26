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
  template<typename T> class SmallVectorImpl;
}

namespace clang {
  class ASTContext;
  class CXXConstructorDecl;
  class CXXDestructorDecl;
  class FunctionDecl;
  class NamedDecl;
  class VarDecl;

namespace CodeGen {
  class CovariantThunkAdjustment;
  class ThunkAdjustment;
   
/// MangleContext - Context for tracking state which persists across multiple
/// calls to the C++ name mangler.
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

  /// @name Mangler Entry Points
  /// @{

  bool shouldMangleDeclName(const NamedDecl *D);

  void mangleName(const NamedDecl *D, llvm::SmallVectorImpl<char> &);
  void mangleThunk(const FunctionDecl *FD, 
                   const ThunkAdjustment &ThisAdjustment,
                   llvm::SmallVectorImpl<char> &);
  void mangleCovariantThunk(const FunctionDecl *FD, 
                            const CovariantThunkAdjustment& Adjustment,
                            llvm::SmallVectorImpl<char> &);
  void mangleGuardVariable(const VarDecl *D, llvm::SmallVectorImpl<char> &);
  void mangleCXXVtable(const CXXRecordDecl *RD, llvm::SmallVectorImpl<char> &);
  void mangleCXXVTT(const CXXRecordDecl *RD, llvm::SmallVectorImpl<char> &);
  void mangleCXXCtorVtable(const CXXRecordDecl *RD, int64_t Offset,
                           const CXXRecordDecl *Type,
                           llvm::SmallVectorImpl<char> &);
  void mangleCXXRtti(QualType T, llvm::SmallVectorImpl<char> &);
  void mangleCXXRttiName(QualType T, llvm::SmallVectorImpl<char> &);
  void mangleCXXCtor(const CXXConstructorDecl *D, CXXCtorType Type,
                     llvm::SmallVectorImpl<char> &);
  void mangleCXXDtor(const CXXDestructorDecl *D, CXXDtorType Type,
                     llvm::SmallVectorImpl<char> &);

  /// @}
};
  
}
}

#endif
