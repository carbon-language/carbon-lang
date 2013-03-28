//===------- MicrosoftCXXABI.cpp - AST support for the Microsoft C++ ABI --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides C++ AST support targeting the Microsoft Visual C++
// ABI.
//
//===----------------------------------------------------------------------===//

#include "CXXABI.h"
#include "clang/AST/Attr.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Basic/TargetInfo.h"

using namespace clang;

namespace {
class MicrosoftCXXABI : public CXXABI {
  ASTContext &Context;
public:
  MicrosoftCXXABI(ASTContext &Ctx) : Context(Ctx) { }

  std::pair<uint64_t, unsigned>
  getMemberPointerWidthAndAlign(const MemberPointerType *MPT) const;

  CallingConv getDefaultMethodCallConv(bool isVariadic) const {
    if (!isVariadic && Context.getTargetInfo().getTriple().getArch() == llvm::Triple::x86)
      return CC_X86ThisCall;
    else
      return CC_C;
  }

  bool isNearlyEmpty(const CXXRecordDecl *RD) const {
    // FIXME: Audit the corners
    if (!RD->isDynamicClass())
      return false;

    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    
    // In the Microsoft ABI, classes can have one or two vtable pointers.
    CharUnits PointerSize = 
      Context.toCharUnitsFromBits(Context.getTargetInfo().getPointerWidth(0));
    return Layout.getNonVirtualSize() == PointerSize ||
      Layout.getNonVirtualSize() == PointerSize * 2;
  }    
};
}

// getNumBases() seems to only give us the number of direct bases, and not the
// total.  This function tells us if we inherit from anybody that uses MI, or if
// we have a non-primary base class, which uses the multiple inheritance model.
static bool usesMultipleInheritanceModel(const CXXRecordDecl *RD) {
  while (RD->getNumBases() > 0) {
    if (RD->getNumBases() > 1)
      return true;
    assert(RD->getNumBases() == 1);
    const CXXRecordDecl *Base = RD->bases_begin()->getType()->getAsCXXRecordDecl();
    if (RD->isPolymorphic() && !Base->isPolymorphic())
      return true;
    RD = Base;
  }
  return false;
}

std::pair<uint64_t, unsigned>
MicrosoftCXXABI::getMemberPointerWidthAndAlign(const MemberPointerType *MPT) const {
  const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  const TargetInfo &Target = Context.getTargetInfo();

  assert(Target.getTriple().getArch() == llvm::Triple::x86 ||
         Target.getTriple().getArch() == llvm::Triple::x86_64);

  Attr *IA = RD->getAttr<MSInheritanceAttr>();
  attr::Kind Inheritance;
  if (IA) {
    Inheritance = IA->getKind();
  } else if (RD->getNumVBases() > 0) {
    Inheritance = attr::VirtualInheritance;
  } else if (MPT->getPointeeType()->isFunctionType() &&
             usesMultipleInheritanceModel(RD)) {
    Inheritance = attr::MultipleInheritance;
  } else {
    Inheritance = attr::SingleInheritance;
  }

  unsigned PtrSize = Target.getPointerWidth(0);
  unsigned IntSize = Target.getIntWidth();
  uint64_t Width;
  unsigned Align;
  if (MPT->getPointeeType()->isFunctionType()) {
    // Member function pointers are a struct of a function pointer followed by a
    // variable number of ints depending on the inheritance model used.  The
    // function pointer is a real function if it is non-virtual and a vftable
    // slot thunk if it is virtual.  The ints select the object base passed for
    // the 'this' pointer.
    Align = Target.getPointerAlign(0);
    switch (Inheritance) {
    case attr::SingleInheritance:      Width = PtrSize;               break;
    case attr::MultipleInheritance:    Width = PtrSize + 1 * IntSize; break;
    case attr::VirtualInheritance:     Width = PtrSize + 2 * IntSize; break;
    case attr::UnspecifiedInheritance: Width = PtrSize + 3 * IntSize; break;
    default: llvm_unreachable("unknown inheritance model");
    }
  } else {
    // Data pointers are an aggregate of ints.  The first int is an offset
    // followed by vbtable-related offsets.
    Align = Target.getIntAlign();
    switch (Inheritance) {
    case attr::SingleInheritance:       // Same as multiple inheritance.
    case attr::MultipleInheritance:     Width = 1 * IntSize; break;
    case attr::VirtualInheritance:      Width = 2 * IntSize; break;
    case attr::UnspecifiedInheritance:  Width = 3 * IntSize; break;
    default: llvm_unreachable("unknown inheritance model");
    }
  }
  Width = llvm::RoundUpToAlignment(Width, Align);

  // FIXME: Verify that our alignment matches MSVC.
  return std::make_pair(Width, Align);
}

CXXABI *clang::CreateMicrosoftCXXABI(ASTContext &Ctx) {
  return new MicrosoftCXXABI(Ctx);
}

