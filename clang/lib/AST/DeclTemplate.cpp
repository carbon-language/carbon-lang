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

QualType ClassTemplateDecl::getInjectedClassNameType(ASTContext &Context) {
  if (!CommonPtr->InjectedClassNameType.isNull())
    return CommonPtr->InjectedClassNameType;

  // FIXME: n2800 14.6.1p1 should say how the template arguments
  // corresponding to template parameter packs should be pack
  // expansions. We already say that in 14.6.2.1p2, so it would be
  // better to fix that redundancy.

  TemplateParameterList *Params = getTemplateParameters();

  llvm::SmallVector<TemplateArgument, 16> TemplateArgs;
  llvm::SmallVector<TemplateArgument, 16> CanonTemplateArgs;
  TemplateArgs.reserve(Params->size());
  CanonTemplateArgs.reserve(Params->size());

  for (TemplateParameterList::iterator 
         Param = Params->begin(), ParamEnd = Params->end(); 
       Param != ParamEnd; ++Param) {
    if (isa<TemplateTypeParmDecl>(*Param)) {
      QualType ParamType = Context.getTypeDeclType(cast<TypeDecl>(*Param));
      TemplateArgs.push_back(TemplateArgument((*Param)->getLocation(), 
                                              ParamType));
      CanonTemplateArgs.push_back(
                         TemplateArgument((*Param)->getLocation(),
                                          Context.getCanonicalType(ParamType)));
    } else if (NonTypeTemplateParmDecl *NTTP = 
                 dyn_cast<NonTypeTemplateParmDecl>(*Param)) {
      // FIXME: Build canonical expression, too!
      Expr *E = new (Context) DeclRefExpr(NTTP, NTTP->getType(),
                                          NTTP->getLocation(),
                                          NTTP->getType()->isDependentType(),
                                          /*Value-dependent=*/true);
      TemplateArgs.push_back(TemplateArgument(E));
      CanonTemplateArgs.push_back(TemplateArgument(E));
    } else { 
      TemplateTemplateParmDecl *TTP = cast<TemplateTemplateParmDecl>(*Param);
      TemplateArgs.push_back(TemplateArgument(TTP->getLocation(), TTP));
      CanonTemplateArgs.push_back(TemplateArgument(TTP->getLocation(), 
                                              Context.getCanonicalDecl(TTP)));
    }
  }

  // FIXME: I should really move the "build-the-canonical-type" logic
  // into ASTContext::getTemplateSpecializationType.
  TemplateName Name = TemplateName(this);
  QualType CanonType = Context.getTemplateSpecializationType(
                                       Context.getCanonicalTemplateName(Name),
                                             &CanonTemplateArgs[0],
                                             CanonTemplateArgs.size());

  CommonPtr->InjectedClassNameType
    = Context.getTemplateSpecializationType(Name,
                                            &TemplateArgs[0],
                                            TemplateArgs.size(),
                                            CanonType);
  return CommonPtr->InjectedClassNameType;
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
// TemplateArgumentList Implementation
//===----------------------------------------------------------------------===//
TemplateArgumentList::TemplateArgumentList(ASTContext &Context,
                                           TemplateArgument *TemplateArgs,
                                           unsigned NumTemplateArgs,
                                           bool CopyArgs)
  : NumArguments(NumTemplateArgs) {
  if (!CopyArgs) {
    Arguments.setPointer(TemplateArgs);
    Arguments.setInt(1);
    return;
  }

  unsigned Size = sizeof(TemplateArgument) * NumTemplateArgs;
  unsigned Align = llvm::AlignOf<TemplateArgument>::Alignment;
  void *Mem = Context.Allocate(Size, Align);
  Arguments.setPointer((TemplateArgument *)Mem);
  Arguments.setInt(0);

  TemplateArgument *Args = (TemplateArgument *)Mem;
  for (unsigned I = 0; I != NumTemplateArgs; ++I)
    new (Args + I) TemplateArgument(TemplateArgs[I]);
}

TemplateArgumentList::~TemplateArgumentList() {
  // FIXME: Deallocate template arguments
}

//===----------------------------------------------------------------------===//
// ClassTemplateSpecializationDecl Implementation
//===----------------------------------------------------------------------===//
ClassTemplateSpecializationDecl::
ClassTemplateSpecializationDecl(ASTContext &Context,
                                DeclContext *DC, SourceLocation L,
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
    TemplateArgs(Context, TemplateArgs, NumTemplateArgs, /*CopyArgs=*/true),
    SpecializationKind(TSK_Undeclared) {
}
                  
ClassTemplateSpecializationDecl *
ClassTemplateSpecializationDecl::Create(ASTContext &Context, 
                                        DeclContext *DC, SourceLocation L,
                                        ClassTemplateDecl *SpecializedTemplate,
                                        TemplateArgument *TemplateArgs, 
                                        unsigned NumTemplateArgs,
                                   ClassTemplateSpecializationDecl *PrevDecl) {
  ClassTemplateSpecializationDecl *Result
    = new (Context)ClassTemplateSpecializationDecl(Context, DC, L, 
                                                   SpecializedTemplate,
                                                   TemplateArgs, 
                                                   NumTemplateArgs);
  Context.getTypeDeclType(Result, PrevDecl);
  return Result;
}
