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

#define DEF_SIMPLE_ATTR_CLONE(ATTR)                                     \
  Attr *ATTR##Attr::clone(ASTContext &C) const {                        \
    return ::new (C) ATTR##Attr;                                        \
  }

// FIXME: Can we use variadic macro to define DEF_SIMPLE_ATTR_CLONE for
// "non-simple" classes?

DEF_SIMPLE_ATTR_CLONE(Packed)
DEF_SIMPLE_ATTR_CLONE(AlwaysInline)
DEF_SIMPLE_ATTR_CLONE(Malloc)
DEF_SIMPLE_ATTR_CLONE(NoReturn)
DEF_SIMPLE_ATTR_CLONE(AnalyzerNoReturn)
DEF_SIMPLE_ATTR_CLONE(Deprecated)
DEF_SIMPLE_ATTR_CLONE(Final)
DEF_SIMPLE_ATTR_CLONE(Unavailable)
DEF_SIMPLE_ATTR_CLONE(Unused)
DEF_SIMPLE_ATTR_CLONE(Used)
DEF_SIMPLE_ATTR_CLONE(Weak)
DEF_SIMPLE_ATTR_CLONE(WeakImport)
DEF_SIMPLE_ATTR_CLONE(NoThrow)
DEF_SIMPLE_ATTR_CLONE(Const)
DEF_SIMPLE_ATTR_CLONE(Pure)
DEF_SIMPLE_ATTR_CLONE(FastCall)
DEF_SIMPLE_ATTR_CLONE(StdCall)
DEF_SIMPLE_ATTR_CLONE(CDecl)
DEF_SIMPLE_ATTR_CLONE(TransparentUnion)
DEF_SIMPLE_ATTR_CLONE(ObjCNSObject)
DEF_SIMPLE_ATTR_CLONE(ObjCException)
DEF_SIMPLE_ATTR_CLONE(NoDebug)
DEF_SIMPLE_ATTR_CLONE(WarnUnusedResult)
DEF_SIMPLE_ATTR_CLONE(NoInline)
DEF_SIMPLE_ATTR_CLONE(CFReturnsRetained)
DEF_SIMPLE_ATTR_CLONE(NSReturnsRetained)
DEF_SIMPLE_ATTR_CLONE(BaseCheck)
DEF_SIMPLE_ATTR_CLONE(Hiding)
DEF_SIMPLE_ATTR_CLONE(Override)
DEF_SIMPLE_ATTR_CLONE(DLLImport)
DEF_SIMPLE_ATTR_CLONE(DLLExport)

Attr* PragmaPackAttr::clone(ASTContext &C) const {
  return ::new (C) PragmaPackAttr(Alignment);
}

Attr* AlignedAttr::clone(ASTContext &C) const {
  return ::new (C) AlignedAttr(Alignment);
}

Attr* AnnotateAttr::clone(ASTContext &C) const {
  return ::new (C) AnnotateAttr(Annotation);
}

Attr *AsmLabelAttr::clone(ASTContext &C) const {
  return ::new (C) AsmLabelAttr(Label);
}

Attr *AliasAttr::clone(ASTContext &C) const {
  return ::new (C) AliasAttr(Aliasee);
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

Attr *GNUInlineAttr::clone(ASTContext &C) const {
  return ::new (C) GNUInlineAttr;
}

Attr *SectionAttr::clone(ASTContext &C) const {
  return ::new (C) SectionAttr(Name);
}

Attr *NonNullAttr::clone(ASTContext &C) const {
  return ::new (C) NonNullAttr(ArgNums, Size);
}

Attr *FormatAttr::clone(ASTContext &C) const {
  return ::new (C) FormatAttr(Type, formatIdx, firstArg);
}

Attr *FormatArgAttr::clone(ASTContext &C) const {
  return ::new (C) FormatArgAttr(formatIdx);
}

Attr *SentinelAttr::clone(ASTContext &C) const {
  return ::new (C) SentinelAttr(sentinel, NullPos);
}

Attr *VisibilityAttr::clone(ASTContext &C) const {
  return ::new (C) VisibilityAttr(VisibilityType);
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

Attr *MSP430InterruptAttr::clone(ASTContext &C) const {
  return ::new (C) MSP430InterruptAttr(Number);
}


