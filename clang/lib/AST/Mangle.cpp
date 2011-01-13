//===--- Mangle.cpp - Mangle C++ Names --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements generic name mangling support for blocks and Objective-C.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/Mangle.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/ABI.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"

#define MANGLE_CHECKER 0

#if MANGLE_CHECKER
#include <cxxabi.h>
#endif

using namespace clang;

// FIXME: For blocks we currently mimic GCC's mangling scheme, which leaves
// much to be desired. Come up with a better mangling scheme.

namespace {

static void mangleFunctionBlock(MangleContext &Context,
                                llvm::StringRef Outer,
                                const BlockDecl *BD,
                                llvm::SmallVectorImpl<char> &Res) {
  llvm::raw_svector_ostream Out(Res);
  Out << "__" << Outer << "_block_invoke_" << Context.getBlockId(BD, true);
}

static void checkMangleDC(const DeclContext *DC, const BlockDecl *BD) {
#ifndef NDEBUG
  const DeclContext *ExpectedDC = BD->getDeclContext();
  while (isa<BlockDecl>(ExpectedDC) || isa<EnumDecl>(ExpectedDC))
    ExpectedDC = ExpectedDC->getParent();
  assert(DC == ExpectedDC && "Given decl context did not match expected!");
#endif
}

}

void MangleContext::mangleGlobalBlock(const BlockDecl *BD,
                                      llvm::SmallVectorImpl<char> &Res) {
  llvm::raw_svector_ostream Out(Res);
  Out << "__block_global_" << getBlockId(BD, false);
}

void MangleContext::mangleCtorBlock(const CXXConstructorDecl *CD,
                                    CXXCtorType CT, const BlockDecl *BD,
                                    llvm::SmallVectorImpl<char> &Res) {
  checkMangleDC(CD, BD);
  llvm::SmallString<64> Buffer;
  mangleCXXCtor(CD, CT, Buffer);
  mangleFunctionBlock(*this, Buffer, BD, Res);
}

void MangleContext::mangleDtorBlock(const CXXDestructorDecl *DD,
                                    CXXDtorType DT, const BlockDecl *BD,
                                    llvm::SmallVectorImpl<char> &Res) {
  checkMangleDC(DD, BD);
  llvm::SmallString<64> Buffer;
  mangleCXXDtor(DD, DT, Buffer);
  mangleFunctionBlock(*this, Buffer, BD, Res);
}

void MangleContext::mangleBlock(const DeclContext *DC, const BlockDecl *BD,
                                llvm::SmallVectorImpl<char> &Res) {
  assert(!isa<CXXConstructorDecl>(DC) && !isa<CXXDestructorDecl>(DC));
  checkMangleDC(DC, BD);

  llvm::SmallString<64> Buffer;
  if (const ObjCMethodDecl *Method = dyn_cast<ObjCMethodDecl>(DC)) {
    mangleObjCMethodName(Method, Buffer);
  } else {
    const NamedDecl *ND = cast<NamedDecl>(DC);
    if (IdentifierInfo *II = ND->getIdentifier())
      Buffer = II->getName();
    else {
      // FIXME: We were doing a mangleUnqualifiedName() before, but that's
      // a private member of a class that will soon itself be private to the
      // Itanium C++ ABI object. What should we do now? Right now, I'm just
      // calling the mangleName() method on the MangleContext; is there a
      // better way?
      mangleName(ND, Buffer);
    }
  }

  mangleFunctionBlock(*this, Buffer, BD, Res);
}

void MangleContext::mangleObjCMethodName(const ObjCMethodDecl *MD,
                                         llvm::SmallVectorImpl<char> &Res) {
  llvm::SmallString<64> Name;
  llvm::raw_svector_ostream OS(Name), Out(Res);
  
  const ObjCContainerDecl *CD =
  dyn_cast<ObjCContainerDecl>(MD->getDeclContext());
  assert (CD && "Missing container decl in GetNameForMethod");
  OS << (MD->isInstanceMethod() ? '-' : '+') << '[' << CD->getName();
  if (const ObjCCategoryImplDecl *CID = dyn_cast<ObjCCategoryImplDecl>(CD))
    OS << '(' << CID << ')';
  OS << ' ' << MD->getSelector().getAsString() << ']';
  
  Out << OS.str().size() << OS.str();
}

void MangleContext::mangleBlock(const BlockDecl *BD,
                                llvm::SmallVectorImpl<char> &Res) {
  const DeclContext *DC = BD->getDeclContext();
  while (isa<BlockDecl>(DC) || isa<EnumDecl>(DC))
    DC = DC->getParent();
  if (DC->isFunctionOrMethod())
    mangleBlock(DC, BD, Res);
  else
    mangleGlobalBlock(BD, Res);
}
