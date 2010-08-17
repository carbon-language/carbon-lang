//===--- AttrImpl.cpp - Classes for representing attributes -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file contains out-of-line virtual methods for Attr classes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Attr.h"
#include "clang/AST/ASTContext.h"
using namespace clang;

Attr::~Attr() { }

AttrWithString::AttrWithString(attr::Kind AK, ASTContext &C, llvm::StringRef s)
  : Attr(AK) {
  assert(!s.empty());
  StrLen = s.size();
  Str = new (C) char[StrLen];
  memcpy(const_cast<char*>(Str), s.data(), StrLen);
}

void AttrWithString::ReplaceString(ASTContext &C, llvm::StringRef newS) {
  if (newS.size() > StrLen) {
    C.Deallocate(const_cast<char*>(Str));
    Str = new (C) char[newS.size()];
  }
  StrLen = newS.size();
  memcpy(const_cast<char*>(Str), newS.data(), StrLen);
}

void FormatAttr::setType(ASTContext &C, llvm::StringRef type) {
  ReplaceString(C, type);
}

NonNullAttr::NonNullAttr(ASTContext &C, unsigned* arg_nums, unsigned size)
  : Attr(attr::NonNull), ArgNums(0), Size(0) {
  if (size == 0)
    return;
  assert(arg_nums);
  ArgNums = new (C) unsigned[size];
  Size = size;
  memcpy(ArgNums, arg_nums, sizeof(*ArgNums)*size);
}

OwnershipAttr::OwnershipAttr(attr::Kind AK, ASTContext &C, unsigned* arg_nums,
                             unsigned size, llvm::StringRef module)
  : AttrWithString(AK, C, module), ArgNums(0), Size(0) {
  if (size == 0)
    return;
  assert(arg_nums);
  ArgNums = new (C) unsigned[size];
  Size = size;
  memcpy(ArgNums, arg_nums, sizeof(*ArgNums) * size);
}


void OwnershipAttr::Destroy(ASTContext &C) {
  if (ArgNums)
    C.Deallocate(ArgNums);
}

OwnershipTakesAttr::OwnershipTakesAttr(ASTContext &C, unsigned* arg_nums,
                                       unsigned size, llvm::StringRef module)
  : OwnershipAttr(attr::OwnershipTakes, C, arg_nums, size, module) {
}

OwnershipHoldsAttr::OwnershipHoldsAttr(ASTContext &C, unsigned* arg_nums,
                                       unsigned size, llvm::StringRef module)
  : OwnershipAttr(attr::OwnershipHolds, C, arg_nums, size, module) {
}

OwnershipReturnsAttr::OwnershipReturnsAttr(ASTContext &C, unsigned* arg_nums,
                                           unsigned size,
                                           llvm::StringRef module)
  : OwnershipAttr(attr::OwnershipReturns, C, arg_nums, size, module) {
}

#define DEF_SIMPLE_ATTR_CLONE(ATTR)                                     \
  Attr *ATTR##Attr::clone(ASTContext &C) const {                        \
    return ::new (C) ATTR##Attr;                                        \
  }

// FIXME: Can we use variadic macro to define DEF_SIMPLE_ATTR_CLONE for
// "non-simple" classes?

DEF_SIMPLE_ATTR_CLONE(AlignMac68k)
DEF_SIMPLE_ATTR_CLONE(AlwaysInline)
DEF_SIMPLE_ATTR_CLONE(AnalyzerNoReturn)
DEF_SIMPLE_ATTR_CLONE(BaseCheck)
DEF_SIMPLE_ATTR_CLONE(CDecl)
DEF_SIMPLE_ATTR_CLONE(CFReturnsNotRetained)
DEF_SIMPLE_ATTR_CLONE(CFReturnsRetained)
DEF_SIMPLE_ATTR_CLONE(Const)
DEF_SIMPLE_ATTR_CLONE(DLLExport)
DEF_SIMPLE_ATTR_CLONE(DLLImport)
DEF_SIMPLE_ATTR_CLONE(Deprecated)
DEF_SIMPLE_ATTR_CLONE(FastCall)
DEF_SIMPLE_ATTR_CLONE(Final)
DEF_SIMPLE_ATTR_CLONE(Hiding)
DEF_SIMPLE_ATTR_CLONE(Malloc)
DEF_SIMPLE_ATTR_CLONE(NSReturnsNotRetained)
DEF_SIMPLE_ATTR_CLONE(NSReturnsRetained)
DEF_SIMPLE_ATTR_CLONE(NoDebug)
DEF_SIMPLE_ATTR_CLONE(NoInline)
DEF_SIMPLE_ATTR_CLONE(NoInstrumentFunction)
DEF_SIMPLE_ATTR_CLONE(NoReturn)
DEF_SIMPLE_ATTR_CLONE(NoThrow)
DEF_SIMPLE_ATTR_CLONE(ObjCException)
DEF_SIMPLE_ATTR_CLONE(ObjCNSObject)
DEF_SIMPLE_ATTR_CLONE(Override)
DEF_SIMPLE_ATTR_CLONE(Packed)
DEF_SIMPLE_ATTR_CLONE(Pure)
DEF_SIMPLE_ATTR_CLONE(StdCall)
DEF_SIMPLE_ATTR_CLONE(ThisCall)
DEF_SIMPLE_ATTR_CLONE(TransparentUnion)
DEF_SIMPLE_ATTR_CLONE(Unavailable)
DEF_SIMPLE_ATTR_CLONE(Unused)
DEF_SIMPLE_ATTR_CLONE(Used)
DEF_SIMPLE_ATTR_CLONE(VecReturn)
DEF_SIMPLE_ATTR_CLONE(WarnUnusedResult)
DEF_SIMPLE_ATTR_CLONE(Weak)
DEF_SIMPLE_ATTR_CLONE(WeakImport)

DEF_SIMPLE_ATTR_CLONE(WeakRef)
DEF_SIMPLE_ATTR_CLONE(X86ForceAlignArgPointer)

Attr* MaxFieldAlignmentAttr::clone(ASTContext &C) const {
  return ::new (C) MaxFieldAlignmentAttr(Alignment);
}

Attr* AlignedAttr::clone(ASTContext &C) const {
  return ::new (C) AlignedAttr(Alignment);
}

Attr* AnnotateAttr::clone(ASTContext &C) const {
  return ::new (C) AnnotateAttr(C, getAnnotation());
}

Attr *AsmLabelAttr::clone(ASTContext &C) const {
  return ::new (C) AsmLabelAttr(C, getLabel());
}

Attr *AliasAttr::clone(ASTContext &C) const {
  return ::new (C) AliasAttr(C, getAliasee());
}

Attr *ConstructorAttr::clone(ASTContext &C) const {
  return ::new (C) ConstructorAttr(priority);
}

Attr *DestructorAttr::clone(ASTContext &C) const {
  return ::new (C) DestructorAttr(priority);
}

Attr *IBOutletAttr::clone(ASTContext &C) const {
  return ::new (C) IBOutletAttr;
}

Attr *IBOutletCollectionAttr::clone(ASTContext &C) const {
  return ::new (C) IBOutletCollectionAttr(QT);
}

Attr *IBActionAttr::clone(ASTContext &C) const {
  return ::new (C) IBActionAttr;
}

Attr *GNUInlineAttr::clone(ASTContext &C) const {
  return ::new (C) GNUInlineAttr;
}

Attr *SectionAttr::clone(ASTContext &C) const {
  return ::new (C) SectionAttr(C, getName());
}

Attr *NonNullAttr::clone(ASTContext &C) const {
  return ::new (C) NonNullAttr(C, ArgNums, Size);
}

Attr *OwnershipAttr::clone(ASTContext &C) const {
  return ::new (C) OwnershipAttr(AKind, C, ArgNums, Size, getModule());
}

Attr *OwnershipReturnsAttr::clone(ASTContext &C) const {
  return ::new (C) OwnershipReturnsAttr(C, ArgNums, Size, getModule());
}

Attr *OwnershipTakesAttr::clone(ASTContext &C) const {
  return ::new (C) OwnershipTakesAttr(C, ArgNums, Size, getModule());
}

Attr *OwnershipHoldsAttr::clone(ASTContext &C) const {
  return ::new (C) OwnershipHoldsAttr(C, ArgNums, Size, getModule());
}

Attr *FormatAttr::clone(ASTContext &C) const {
  return ::new (C) FormatAttr(C, getType(), formatIdx, firstArg);
}

Attr *FormatArgAttr::clone(ASTContext &C) const {
  return ::new (C) FormatArgAttr(formatIdx);
}

Attr *SentinelAttr::clone(ASTContext &C) const {
  return ::new (C) SentinelAttr(sentinel, NullPos);
}

Attr *VisibilityAttr::clone(ASTContext &C) const {
  return ::new (C) VisibilityAttr(VisibilityType, FromPragma);
}

Attr *OverloadableAttr::clone(ASTContext &C) const {
  return ::new (C) OverloadableAttr;
}

Attr *BlocksAttr::clone(ASTContext &C) const {
  return ::new (C) BlocksAttr(BlocksAttrType);
}

Attr *CleanupAttr::clone(ASTContext &C) const {
  return ::new (C) CleanupAttr(FD);
}

Attr *RegparmAttr::clone(ASTContext &C) const {
  return ::new (C) RegparmAttr(NumParams);
}

Attr *ReqdWorkGroupSizeAttr::clone(ASTContext &C) const {
  return ::new (C) ReqdWorkGroupSizeAttr(X, Y, Z);
}

Attr *InitPriorityAttr::clone(ASTContext &C) const {
  return ::new (C) InitPriorityAttr(Priority);
}

Attr *MSP430InterruptAttr::clone(ASTContext &C) const {
  return ::new (C) MSP430InterruptAttr(Number);
}
