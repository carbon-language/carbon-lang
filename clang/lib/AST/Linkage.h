//===----- Linkage.h - Linkage calculation-related utilities ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides AST-internal utilities for linkage and visibility
// calculation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_AST_LINKAGE_H
#define LLVM_CLANG_LIB_AST_LINKAGE_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {
enum : unsigned {
  IgnoreExplicitVisibilityBit = 2,
  IgnoreAllVisibilityBit = 4
};

/// Kinds of LV computation.  The linkage side of the computation is
/// always the same, but different things can change how visibility is
/// computed.
enum LVComputationKind {
  /// Do an LV computation for, ultimately, a type.
  /// Visibility may be restricted by type visibility settings and
  /// the visibility of template arguments.
  LVForType = NamedDecl::VisibilityForType,

  /// Do an LV computation for, ultimately, a non-type declaration.
  /// Visibility may be restricted by value visibility settings and
  /// the visibility of template arguments.
  LVForValue = NamedDecl::VisibilityForValue,

  /// Do an LV computation for, ultimately, a type that already has
  /// some sort of explicit visibility.  Visibility may only be
  /// restricted by the visibility of template arguments.
  LVForExplicitType = (LVForType | IgnoreExplicitVisibilityBit),

  /// Do an LV computation for, ultimately, a non-type declaration
  /// that already has some sort of explicit visibility.  Visibility
  /// may only be restricted by the visibility of template arguments.
  LVForExplicitValue = (LVForValue | IgnoreExplicitVisibilityBit),

  /// Do an LV computation when we only care about the linkage.
  LVForLinkageOnly =
      LVForValue | IgnoreExplicitVisibilityBit | IgnoreAllVisibilityBit
};

class LinkageComputer {
  LinkageInfo getLVForTemplateArgumentList(ArrayRef<TemplateArgument> Args,
                                           LVComputationKind computation);

  LinkageInfo getLVForTemplateArgumentList(const TemplateArgumentList &TArgs,
                                           LVComputationKind computation);

  void mergeTemplateLV(LinkageInfo &LV, const FunctionDecl *fn,
                       const FunctionTemplateSpecializationInfo *specInfo,
                       LVComputationKind computation);

  void mergeTemplateLV(LinkageInfo &LV,
                       const ClassTemplateSpecializationDecl *spec,
                       LVComputationKind computation);

  void mergeTemplateLV(LinkageInfo &LV,
                       const VarTemplateSpecializationDecl *spec,
                       LVComputationKind computation);

  LinkageInfo getLVForNamespaceScopeDecl(const NamedDecl *D,
                                         LVComputationKind computation);

  LinkageInfo getLVForClassMember(const NamedDecl *D,
                                  LVComputationKind computation);

  LinkageInfo getLVForClosure(const DeclContext *DC, Decl *ContextDecl,
                              LVComputationKind computation);

  LinkageInfo getLVForLocalDecl(const NamedDecl *D,
                                LVComputationKind computation);

  LinkageInfo getLVForType(const Type &T, LVComputationKind computation);

  LinkageInfo getLVForTemplateParameterList(const TemplateParameterList *Params,
                                            LVComputationKind computation);

public:
  LinkageInfo computeLVForDecl(const NamedDecl *D,
                               LVComputationKind computation);

  LinkageInfo getLVForDecl(const NamedDecl *D, LVComputationKind computation);

  LinkageInfo computeTypeLinkageInfo(const Type *T);
  LinkageInfo computeTypeLinkageInfo(QualType T) {
    return computeTypeLinkageInfo(T.getTypePtr());
  }

  LinkageInfo getDeclLinkageAndVisibility(const NamedDecl *D);

  LinkageInfo getTypeLinkageAndVisibility(const Type *T);
  LinkageInfo getTypeLinkageAndVisibility(QualType T) {
    return getTypeLinkageAndVisibility(T.getTypePtr());
  }
};
} // namespace clang

#endif
