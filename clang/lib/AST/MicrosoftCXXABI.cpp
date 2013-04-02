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
bool usesMultipleInheritanceModel(const CXXRecordDecl *RD) {
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

MSInheritanceModel MSInheritanceAttrToModel(attr::Kind Kind) {
  switch (Kind) {
  default: llvm_unreachable("expected MS inheritance attribute");
  case attr::SingleInheritance:      return MSIM_Single;
  case attr::MultipleInheritance:    return MSIM_Multiple;
  case attr::VirtualInheritance:     return MSIM_Virtual;
  case attr::UnspecifiedInheritance: return MSIM_Unspecified;
  }
}

MSInheritanceModel CXXRecordDecl::getMSInheritanceModel() const {
  Attr *IA = this->getAttr<MSInheritanceAttr>();
  if (IA)
    return MSInheritanceAttrToModel(IA->getKind());
  // If there was no explicit attribute, the record must be defined already, and
  // we can figure out the inheritance model from its other properties.
  if (this->getNumVBases() > 0)
    return MSIM_Virtual;
  if (usesMultipleInheritanceModel(this))
    return MSIM_Multiple;
  return MSIM_Single;
}

// Returns the number of pointer and integer slots used to represent a member
// pointer in the MS C++ ABI.
//
// Member function pointers have the following general form;  however, fields
// are dropped as permitted (under the MSVC interpretation) by the inheritance
// model of the actual class.
//
//   struct {
//     // A pointer to the member function to call.  If the member function is
//     // virtual, this will be a thunk that forwards to the appropriate vftable
//     // slot.
//     void *FunctionPointerOrVirtualThunk;
//
//     // An offset to add to the address of the vbtable pointer after (possibly)
//     // selecting the virtual base but before resolving and calling the function.
//     // Only needed if the class has any virtual bases or bases at a non-zero
//     // offset.
//     int NonVirtualBaseAdjustment;
//
//     // An offset within the vb-table that selects the virtual base containing
//     // the member.  Loading from this offset produces a new offset that is
//     // added to the address of the vb-table pointer to produce the base.
//     int VirtualBaseAdjustmentOffset;
//
//     // The offset of the vb-table pointer within the object.  Only needed for
//     // incomplete types.
//     int VBTableOffset;
//   };
std::pair<unsigned, unsigned> MemberPointerType::getMSMemberPointerSlots() const {
  const CXXRecordDecl *RD = this->getClass()->getAsCXXRecordDecl();
  MSInheritanceModel Inheritance = RD->getMSInheritanceModel();
  unsigned Ptrs;
  unsigned Ints = 0;
  if (this->isMemberFunctionPointer()) {
    // Member function pointers are a struct of a function pointer followed by a
    // variable number of ints depending on the inheritance model used.  The
    // function pointer is a real function if it is non-virtual and a vftable
    // slot thunk if it is virtual.  The ints select the object base passed for
    // the 'this' pointer.
    Ptrs = 1;  // First slot is always a function pointer.
    switch (Inheritance) {
    case MSIM_Unspecified: ++Ints;  // VBTableOffset
    case MSIM_Virtual:     ++Ints;  // VirtualBaseAdjustmentOffset
    case MSIM_Multiple:    ++Ints;  // NonVirtualBaseAdjustment
    case MSIM_Single:      break;   // Nothing
    }
  } else {
    // Data pointers are an aggregate of ints.  The first int is an offset
    // followed by vbtable-related offsets.
    Ptrs = 0;
    switch (Inheritance) {
    case MSIM_Unspecified: ++Ints;  // VBTableOffset
    case MSIM_Virtual:     ++Ints;  // VirtualBaseAdjustmentOffset
    case MSIM_Multiple:             // Nothing
    case MSIM_Single:      ++Ints;  // Field offset
    }
  }
  return std::make_pair(Ptrs, Ints);
}

std::pair<uint64_t, unsigned>
MicrosoftCXXABI::getMemberPointerWidthAndAlign(const MemberPointerType *MPT) const {
  const TargetInfo &Target = Context.getTargetInfo();
  assert(Target.getTriple().getArch() == llvm::Triple::x86 ||
         Target.getTriple().getArch() == llvm::Triple::x86_64);
  unsigned Ptrs, Ints;
  llvm::tie(Ptrs, Ints) = MPT->getMSMemberPointerSlots();
  // The nominal struct is laid out with pointers followed by ints and aligned
  // to a pointer width if any are present and an int width otherwise.
  unsigned PtrSize = Target.getPointerWidth(0);
  unsigned IntSize = Target.getIntWidth();
  uint64_t Width = Ptrs * PtrSize + Ints * IntSize;
  unsigned Align = Ptrs > 0 ? Target.getPointerAlign(0) : Target.getIntAlign();
  Width = llvm::RoundUpToAlignment(Width, Align);
  return std::make_pair(Width, Align);
}

CXXABI *clang::CreateMicrosoftCXXABI(ASTContext &Ctx) {
  return new MicrosoftCXXABI(Ctx);
}

