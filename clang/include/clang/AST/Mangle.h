//===--- Mangle.h - Mangle C++ Names ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the C++ name mangling interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_MANGLE_H
#define LLVM_CLANG_AST_MANGLE_H

#include "clang/AST/Type.h"
#include "clang/Basic/ABI.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
  class ASTContext;
  class BlockDecl;
  class CXXConstructorDecl;
  class CXXDestructorDecl;
  class CXXMethodDecl;
  class FunctionDecl;
  class NamedDecl;
  class ObjCMethodDecl;
  class VarDecl;
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

  llvm::DenseMap<const BlockDecl*, unsigned> GlobalBlockIds;
  llvm::DenseMap<const BlockDecl*, unsigned> LocalBlockIds;
  
public:
  explicit MangleContext(ASTContext &Context,
                         Diagnostic &Diags)
    : Context(Context), Diags(Diags) { }

  virtual ~MangleContext() { }

  ASTContext &getASTContext() const { return Context; }

  Diagnostic &getDiags() const { return Diags; }

  virtual void startNewFunction() { LocalBlockIds.clear(); }
  
  unsigned getBlockId(const BlockDecl *BD, bool Local) {
    llvm::DenseMap<const BlockDecl *, unsigned> &BlockIds
      = Local? LocalBlockIds : GlobalBlockIds;
    std::pair<llvm::DenseMap<const BlockDecl *, unsigned>::iterator, bool>
      Result = BlockIds.insert(std::make_pair(BD, BlockIds.size()));
    return Result.first->second;
  }
  
  /// @name Mangler Entry Points
  /// @{

  virtual bool shouldMangleDeclName(const NamedDecl *D) = 0;
  virtual void mangleName(const NamedDecl *D, llvm::SmallVectorImpl<char> &)=0;
  virtual void mangleThunk(const CXXMethodDecl *MD,
                          const ThunkInfo &Thunk,
                          llvm::SmallVectorImpl<char> &) = 0;
  virtual void mangleCXXDtorThunk(const CXXDestructorDecl *DD, CXXDtorType Type,
                                  const ThisAdjustment &ThisAdjustment,
                                  llvm::SmallVectorImpl<char> &) = 0;
  virtual void mangleReferenceTemporary(const VarDecl *D,
                                        llvm::SmallVectorImpl<char> &) = 0;
  virtual void mangleCXXVTable(const CXXRecordDecl *RD,
                               llvm::SmallVectorImpl<char> &) = 0;
  virtual void mangleCXXVTT(const CXXRecordDecl *RD,
                            llvm::SmallVectorImpl<char> &) = 0;
  virtual void mangleCXXCtorVTable(const CXXRecordDecl *RD, int64_t Offset,
                                   const CXXRecordDecl *Type,
                                   llvm::SmallVectorImpl<char> &) = 0;
  virtual void mangleCXXRTTI(QualType T, llvm::SmallVectorImpl<char> &) = 0;
  virtual void mangleCXXRTTIName(QualType T, llvm::SmallVectorImpl<char> &) = 0;
  virtual void mangleCXXCtor(const CXXConstructorDecl *D, CXXCtorType Type,
                             llvm::SmallVectorImpl<char> &) = 0;
  virtual void mangleCXXDtor(const CXXDestructorDecl *D, CXXDtorType Type,
                             llvm::SmallVectorImpl<char> &) = 0;

  void mangleGlobalBlock(const BlockDecl *BD,
                         llvm::SmallVectorImpl<char> &Res);
  void mangleCtorBlock(const CXXConstructorDecl *CD, CXXCtorType CT,
                       const BlockDecl *BD, llvm::SmallVectorImpl<char> &Res);
  void mangleDtorBlock(const CXXDestructorDecl *CD, CXXDtorType DT,
                       const BlockDecl *BD, llvm::SmallVectorImpl<char> &Res);
  void mangleBlock(const DeclContext *DC, const BlockDecl *BD,
                   llvm::SmallVectorImpl<char> &Res);
  // Do the right thing.
  void mangleBlock(const BlockDecl *BD, llvm::SmallVectorImpl<char> &Res);

  void mangleObjCMethodName(const ObjCMethodDecl *MD,
                            llvm::SmallVectorImpl<char> &);

  // This is pretty lame.
  virtual void mangleItaniumGuardVariable(const VarDecl *D,
                                          llvm::SmallVectorImpl<char> &) {
    assert(0 && "Target does not support mangling guard variables");
  }
  /// @}
};

MangleContext *createItaniumMangleContext(ASTContext &Context,
                                          Diagnostic &Diags);
MangleContext *createMicrosoftMangleContext(ASTContext &Context,
                                            Diagnostic &Diags);

}

#endif
