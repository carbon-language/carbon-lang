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
#include "clang/AST/Expr.h"
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

unsigned TemplateParameterList::getMinRequiredArguments() const {
  unsigned NumRequiredArgs = size();
  iterator Param = const_cast<TemplateParameterList *>(this)->end(), 
      ParamBegin = const_cast<TemplateParameterList *>(this)->begin();
  while (Param != ParamBegin) {
    --Param;
    if (!(isa<TemplateTypeParmDecl>(*Param) && 
          cast<TemplateTypeParmDecl>(*Param)->hasDefaultArgument()) &&
        !(isa<NonTypeTemplateParmDecl>(*Param) &&
          cast<NonTypeTemplateParmDecl>(*Param)->hasDefaultArgument()) &&
        !(isa<TemplateTemplateParmDecl>(*Param) &&
          cast<TemplateTemplateParmDecl>(*Param)->hasDefaultArgument()))
      break;
        
    --NumRequiredArgs;
  }

  return NumRequiredArgs;
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
                                             NamedDecl *Decl,
                                             ClassTemplateDecl *PrevDecl) {
  Common *CommonPtr;
  if (PrevDecl)
    CommonPtr = PrevDecl->CommonPtr;
  else
    CommonPtr = new (C) Common;

  return new (C) ClassTemplateDecl(DC, L, Name, Params, Decl, PrevDecl, 
                                   CommonPtr);
}

ClassTemplateDecl::~ClassTemplateDecl() {
  assert(CommonPtr == 0 && "ClassTemplateDecl must be explicitly destroyed");
}

void ClassTemplateDecl::Destroy(ASTContext& C) {
  if (!PreviousDeclaration) {
    CommonPtr->~Common();
    C.Deallocate((void*)CommonPtr);
  }
  CommonPtr = 0;

  this->~ClassTemplateDecl();
  C.Deallocate((void*)this);
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

SourceLocation NonTypeTemplateParmDecl::getDefaultArgumentLoc() const {
  return DefaultArgument? DefaultArgument->getSourceRange().getBegin()
                        : SourceLocation(); 
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

SourceLocation TemplateTemplateParmDecl::getDefaultArgumentLoc() const {
  return DefaultArgument? DefaultArgument->getSourceRange().getBegin()
                        : SourceLocation(); 
}

//===----------------------------------------------------------------------===//
// TemplateArgument Implementation
//===----------------------------------------------------------------------===//

TemplateArgument::TemplateArgument(Expr *E) : Kind(Expression) {
  TypeOrValue = reinterpret_cast<uintptr_t>(E);
  StartLoc = E->getSourceRange().getBegin();
}

//===----------------------------------------------------------------------===//
// ClassTemplateSpecializationDecl Implementation
//===----------------------------------------------------------------------===//
ClassTemplateSpecializationDecl::
ClassTemplateSpecializationDecl(DeclContext *DC, SourceLocation L,
                                ClassTemplateDecl *SpecializedTemplate,
                                TemplateArgument *TemplateArgs, 
                                unsigned NumTemplateArgs)
  : CXXRecordDecl(ClassTemplateSpecialization, 
                  SpecializedTemplate->getTemplatedDecl()->getTagKind(), 
                  DC, L,
                  // FIXME: Should we use DeclarationName for the name of
                  // class template specializations?
                  SpecializedTemplate->getIdentifier()),
    SpecializedTemplate(SpecializedTemplate),
    NumTemplateArgs(NumTemplateArgs), SpecializationKind(TSK_Undeclared) {
  TemplateArgument *Arg = reinterpret_cast<TemplateArgument *>(this + 1);
  for (unsigned ArgIdx = 0; ArgIdx < NumTemplateArgs; ++ArgIdx, ++Arg)
    new (Arg) TemplateArgument(TemplateArgs[ArgIdx]);
}
                  
ClassTemplateSpecializationDecl *
ClassTemplateSpecializationDecl::Create(ASTContext &Context, 
                                        DeclContext *DC, SourceLocation L,
                                        ClassTemplateDecl *SpecializedTemplate,
                                        TemplateArgument *TemplateArgs, 
                                        unsigned NumTemplateArgs,
                                   ClassTemplateSpecializationDecl *PrevDecl) {
  unsigned Size = sizeof(ClassTemplateSpecializationDecl) + 
                  sizeof(TemplateArgument) * NumTemplateArgs;
  unsigned Align = llvm::AlignOf<ClassTemplateSpecializationDecl>::Alignment;
  void *Mem = Context.Allocate(Size, Align);
  ClassTemplateSpecializationDecl *Result
    = new (Mem) ClassTemplateSpecializationDecl(DC, L, SpecializedTemplate,
                                                TemplateArgs, NumTemplateArgs);
  Context.getTypeDeclType(Result, PrevDecl);
  return Result;
}
