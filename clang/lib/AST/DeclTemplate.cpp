//===--- DeclCXX.cpp - C++ Declaration AST Node Implementation ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the C++ related Decl classes for templates.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/STLExtras.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// TemplateParameterList Implementation
//===----------------------------------------------------------------------===//

TemplateParameterList::TemplateParameterList(SourceLocation TemplateLoc,
                                             SourceLocation LAngleLoc,
                                             Decl **Params, unsigned NumParams,
                                             SourceLocation RAngleLoc)
  : TemplateLoc(TemplateLoc), LAngleLoc(LAngleLoc), RAngleLoc(RAngleLoc),
    NumParams(NumParams) {
  for (unsigned Idx = 0; Idx < NumParams; ++Idx)
    begin()[Idx] = Params[Idx];
}

TemplateParameterList *
TemplateParameterList::Create(ASTContext &C, SourceLocation TemplateLoc,
                              SourceLocation LAngleLoc, Decl **Params,
                              unsigned NumParams, SourceLocation RAngleLoc) {
  unsigned Size = sizeof(TemplateParameterList) + sizeof(Decl *) * NumParams;
  unsigned Align = llvm::AlignOf<TemplateParameterList>::Alignment;
  void *Mem = C.Allocate(Size, Align);
  return new (Mem) TemplateParameterList(TemplateLoc, LAngleLoc, Params, 
                                         NumParams, RAngleLoc);
}

//===----------------------------------------------------------------------===//
// TemplateDecl Implementation
//===----------------------------------------------------------------------===//

TemplateDecl::~TemplateDecl() {
}

//===----------------------------------------------------------------------===//
// FunctionTemplateDecl Implementation
//===----------------------------------------------------------------------===//

FunctionTemplateDecl *FunctionTemplateDecl::Create(ASTContext &C,
                                                   DeclContext *DC,
                                                   SourceLocation L,
                                                   DeclarationName Name,
                                                   TemplateParameterList *Params,
                                                   NamedDecl *Decl) {
  return new (C) FunctionTemplateDecl(DC, L, Name, Params, Decl);
}

//===----------------------------------------------------------------------===//
// ClassTemplateDecl Implementation
//===----------------------------------------------------------------------===//

ClassTemplateDecl *ClassTemplateDecl::Create(ASTContext &C,
                                             DeclContext *DC,
                                             SourceLocation L,
                                             DeclarationName Name,
                                             TemplateParameterList *Params,
                                             NamedDecl *Decl) {
  return new (C) ClassTemplateDecl(DC, L, Name, Params, Decl);
}

//===----------------------------------------------------------------------===//
// TemplateTypeParm Allocation/Deallocation Method Implementations
//===----------------------------------------------------------------------===//

TemplateTypeParmDecl *
TemplateTypeParmDecl::Create(ASTContext &C, DeclContext *DC,
                             SourceLocation L, unsigned D, unsigned P,
                             IdentifierInfo *Id, bool Typename) {
  QualType Type = C.getTemplateTypeParmType(D, P, Id);
  return new (C) TemplateTypeParmDecl(DC, L, Id, Typename, Type);
}

//===----------------------------------------------------------------------===//
// NonTypeTemplateParmDecl Method Implementations
//===----------------------------------------------------------------------===//

NonTypeTemplateParmDecl *
NonTypeTemplateParmDecl::Create(ASTContext &C, DeclContext *DC,
                                SourceLocation L, unsigned D, unsigned P,
                                IdentifierInfo *Id, QualType T,
                                SourceLocation TypeSpecStartLoc) {
  return new (C) NonTypeTemplateParmDecl(DC, L, D, P, Id, T,
                                         TypeSpecStartLoc);
}

//===----------------------------------------------------------------------===//
// TemplateTemplateParmDecl Method Implementations
//===----------------------------------------------------------------------===//

TemplateTemplateParmDecl *
TemplateTemplateParmDecl::Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation L, unsigned D, unsigned P,
                                 IdentifierInfo *Id,
                                 TemplateParameterList *Params) {
  return new (C) TemplateTemplateParmDecl(DC, L, D, P, Id, Params);
}

