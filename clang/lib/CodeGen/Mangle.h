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
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallString.h"

namespace clang {
  class ASTContext;
  class CXXConstructorDecl;
  class CXXDestructorDecl;
  class CXXMethodDecl;
  class FunctionDecl;
  class NamedDecl;
  class VarDecl;

namespace CodeGen {
  struct ThisAdjustment;
  struct ThunkInfo;

/// MangleBuffer - a convenient class for storing a name which is
/// either the result of a mangling or is a constant string with
/// external memory ownership.
class MangleBuffer {
public:
  void setString(llvm::StringRef Ref) {
    String = Ref;
  }

  llvm::SmallVectorImpl<char> &getBuffer() {
    return Buffer;
  }

  llvm::StringRef getString() const {
    if (!String.empty()) return String;
    return Buffer.str();
  }

  operator llvm::StringRef() const {
    return getString();
  }

private:
  llvm::StringRef String;
  llvm::SmallString<256> Buffer;
};
   
/// MangleContext - Context for tracking state which persists across multiple
/// calls to the C++ name mangler.
class MangleContext {
  ASTContext &Context;
  Diagnostic &Diags;

  llvm::DenseMap<const TagDecl *, uint64_t> AnonStructIds;
  unsigned Discriminator;
  llvm::DenseMap<const NamedDecl*, unsigned> Uniquifier;
  
public:
  explicit MangleContext(ASTContext &Context,
                         Diagnostic &Diags)
    : Context(Context), Diags(Diags) { }

  ASTContext &getASTContext() const { return Context; }

  Diagnostic &getDiags() const { return Diags; }

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
  void mangleThunk(const CXXMethodDecl *MD,
                   const ThunkInfo &Thunk,
                   llvm::SmallVectorImpl<char> &);
  void mangleCXXDtorThunk(const CXXDestructorDecl *DD, CXXDtorType Type,
                          const ThisAdjustment &ThisAdjustment,
                          llvm::SmallVectorImpl<char> &);
  void mangleGuardVariable(const VarDecl *D, llvm::SmallVectorImpl<char> &);
  void mangleCXXVTable(const CXXRecordDecl *RD, llvm::SmallVectorImpl<char> &);
  void mangleCXXVTT(const CXXRecordDecl *RD, llvm::SmallVectorImpl<char> &);
  void mangleCXXCtorVTable(const CXXRecordDecl *RD, int64_t Offset,
                           const CXXRecordDecl *Type,
                           llvm::SmallVectorImpl<char> &);
  void mangleCXXRTTI(QualType T, llvm::SmallVectorImpl<char> &);
  void mangleCXXRTTIName(QualType T, llvm::SmallVectorImpl<char> &);
  void mangleCXXCtor(const CXXConstructorDecl *D, CXXCtorType Type,
                     llvm::SmallVectorImpl<char> &);
  void mangleCXXDtor(const CXXDestructorDecl *D, CXXDtorType Type,
                     llvm::SmallVectorImpl<char> &);
  
  void mangleInitDiscriminator() {
    Discriminator = 0;
  }

  bool getNextDiscriminator(const NamedDecl *ND, unsigned &disc) {
    unsigned &discriminator = Uniquifier[ND];
    if (!discriminator)
      discriminator = ++Discriminator;
    if (discriminator == 1)
      return false;
    disc = discriminator-2;
    return true;
  }
  /// @}
};
  
}
}

#endif
