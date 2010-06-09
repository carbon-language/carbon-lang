//===--- MicrosoftCXXABI.cpp - Emit LLVM Code from ASTs for a Module ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides C++ code generation targetting the Microsoft Visual C++ ABI.
// The class in this file generates structures that follow the Microsoft
// Visual C++ ABI, which is actually not very well documented at all outside
// of Microsoft.
//
//===----------------------------------------------------------------------===//

#include "CGCXXABI.h"
#include "CodeGenModule.h"
#include "Mangle.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "CGVTables.h"

using namespace clang;
using namespace CodeGen;

namespace {

/// MicrosoftMangleContext - Overrides the default MangleContext for the
/// Microsoft Visual C++ ABI.
class MicrosoftMangleContext : public MangleContext {
public:
  MicrosoftMangleContext(ASTContext &Context,
                         Diagnostic &Diags) : MangleContext(Context, Diags) { }
  virtual void mangleName(const NamedDecl *D, llvm::SmallVectorImpl<char> &);
  virtual void mangleThunk(const CXXMethodDecl *MD,
                           const ThunkInfo &Thunk,
                           llvm::SmallVectorImpl<char> &);
  virtual void mangleCXXDtorThunk(const CXXDestructorDecl *DD, CXXDtorType Type,
                                  const ThisAdjustment &ThisAdjustment,
                                  llvm::SmallVectorImpl<char> &);
  virtual void mangleGuardVariable(const VarDecl *D,
                                   llvm::SmallVectorImpl<char> &);
  virtual void mangleCXXVTable(const CXXRecordDecl *RD,
                               llvm::SmallVectorImpl<char> &);
  virtual void mangleCXXVTT(const CXXRecordDecl *RD,
                            llvm::SmallVectorImpl<char> &);
  virtual void mangleCXXCtorVTable(const CXXRecordDecl *RD, int64_t Offset,
                                   const CXXRecordDecl *Type,
                                   llvm::SmallVectorImpl<char> &);
  virtual void mangleCXXRTTI(QualType T, llvm::SmallVectorImpl<char> &);
  virtual void mangleCXXRTTIName(QualType T, llvm::SmallVectorImpl<char> &);
  virtual void mangleCXXCtor(const CXXConstructorDecl *D, CXXCtorType Type,
                             llvm::SmallVectorImpl<char> &);
  virtual void mangleCXXDtor(const CXXDestructorDecl *D, CXXDtorType Type,
                             llvm::SmallVectorImpl<char> &);
};

class MicrosoftCXXABI : public CXXABI {
  MicrosoftMangleContext MangleCtx;
public:
  MicrosoftCXXABI(CodeGenModule &CGM)
   : MangleCtx(CGM.getContext(), CGM.getDiags()) {}

  MicrosoftMangleContext &getMangleContext() {
    return MangleCtx;
  }
};

}

void MicrosoftMangleContext::mangleName(const NamedDecl *D,
                                        llvm::SmallVectorImpl<char> &Name) {
  assert(false && "Can't yet mangle names!");
}
void MicrosoftMangleContext::mangleThunk(const CXXMethodDecl *MD,
                                         const ThunkInfo &Thunk,
                                         llvm::SmallVectorImpl<char> &) {
  assert(false && "Can't yet mangle thunks!");
}
void MicrosoftMangleContext::mangleCXXDtorThunk(const CXXDestructorDecl *DD,
                                                CXXDtorType Type,
                                                const ThisAdjustment &,
                                                llvm::SmallVectorImpl<char> &) {
  assert(false && "Can't yet mangle destructor thunks!");
}
void MicrosoftMangleContext::mangleGuardVariable(const VarDecl *D,
                                                 llvm::SmallVectorImpl<char> &) {
  assert(false && "Can't yet mangle guard variables!");
}
void MicrosoftMangleContext::mangleCXXVTable(const CXXRecordDecl *RD,
                                             llvm::SmallVectorImpl<char> &) {
  assert(false && "Can't yet mangle virtual tables!");
}
void MicrosoftMangleContext::mangleCXXVTT(const CXXRecordDecl *RD,
                                          llvm::SmallVectorImpl<char> &) {
  llvm_unreachable("The MS C++ ABI does not have virtual table tables!");
}
void MicrosoftMangleContext::mangleCXXCtorVTable(const CXXRecordDecl *RD,
                                                 int64_t Offset,
                                                 const CXXRecordDecl *Type,
                                                 llvm::SmallVectorImpl<char> &) {
  llvm_unreachable("The MS C++ ABI does not have constructor vtables!");
}
void MicrosoftMangleContext::mangleCXXRTTI(QualType T,
                                           llvm::SmallVectorImpl<char> &) {
  assert(false && "Can't yet mangle RTTI!");
}
void MicrosoftMangleContext::mangleCXXRTTIName(QualType T,
                                               llvm::SmallVectorImpl<char> &) {
  assert(false && "Can't yet mangle RTTI names!");
}
void MicrosoftMangleContext::mangleCXXCtor(const CXXConstructorDecl *D,
                                           CXXCtorType Type,
                                           llvm::SmallVectorImpl<char> &) {
  assert(false && "Can't yet mangle constructors!");
}
void MicrosoftMangleContext::mangleCXXDtor(const CXXDestructorDecl *D,
                                           CXXDtorType Type,
                                           llvm::SmallVectorImpl<char> &) {
  assert(false && "Can't yet mangle destructors!");
}

CXXABI *CreateMicrosoftCXXABI(CodeGenModule &CGM) {
  return new MicrosoftCXXABI(CGM);
}

