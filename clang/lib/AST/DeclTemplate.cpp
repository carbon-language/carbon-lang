//===--- DeclTemplate.cpp - Template Declaration AST Node Implementation --===//
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
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/STLExtras.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// TemplateParameterList Implementation
//===----------------------------------------------------------------------===//

TemplateParameterList::TemplateParameterList(SourceLocation TemplateLoc,
                                             SourceLocation LAngleLoc,
                                             NamedDecl **Params, unsigned NumParams,
                                             SourceLocation RAngleLoc)
  : TemplateLoc(TemplateLoc), LAngleLoc(LAngleLoc), RAngleLoc(RAngleLoc),
    NumParams(NumParams) {
  for (unsigned Idx = 0; Idx < NumParams; ++Idx)
    begin()[Idx] = Params[Idx];
}

TemplateParameterList *
TemplateParameterList::Create(ASTContext &C, SourceLocation TemplateLoc,
                              SourceLocation LAngleLoc, NamedDecl **Params,
                              unsigned NumParams, SourceLocation RAngleLoc) {
  unsigned Size = sizeof(TemplateParameterList) 
                + sizeof(NamedDecl *) * NumParams;
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

    if (!(*Param)->isTemplateParameterPack() &&
        !(isa<TemplateTypeParmDecl>(*Param) &&
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

unsigned TemplateParameterList::getDepth() const {
  if (size() == 0)
    return 0;
  
  const NamedDecl *FirstParm = getParam(0);
  if (const TemplateTypeParmDecl *TTP
        = dyn_cast<TemplateTypeParmDecl>(FirstParm))
    return TTP->getDepth();
  else if (const NonTypeTemplateParmDecl *NTTP 
             = dyn_cast<NonTypeTemplateParmDecl>(FirstParm))
    return NTTP->getDepth();
  else
    return cast<TemplateTemplateParmDecl>(FirstParm)->getDepth();
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

void FunctionTemplateDecl::Destroy(ASTContext &C) {
  if (Common *CommonPtr = CommonOrPrev.dyn_cast<Common*>()) {
    for (llvm::FoldingSet<FunctionTemplateSpecializationInfo>::iterator
              Spec = CommonPtr->Specializations.begin(),
           SpecEnd = CommonPtr->Specializations.end();
         Spec != SpecEnd; ++Spec)
      C.Deallocate(&*Spec);
  }

  Decl::Destroy(C);
}

FunctionTemplateDecl *FunctionTemplateDecl::getCanonicalDecl() {
  FunctionTemplateDecl *FunTmpl = this;
  while (FunTmpl->getPreviousDeclaration())
    FunTmpl = FunTmpl->getPreviousDeclaration();
  return FunTmpl;
}

FunctionTemplateDecl::Common *FunctionTemplateDecl::getCommonPtr() {
  // Find the first declaration of this function template.
  FunctionTemplateDecl *First = this;
  while (First->getPreviousDeclaration())
    First = First->getPreviousDeclaration();

  if (First->CommonOrPrev.isNull()) {
    // FIXME: Allocate with the ASTContext
    First->CommonOrPrev = new Common;
  }
  return First->CommonOrPrev.get<Common*>();
}

//===----------------------------------------------------------------------===//
// ClassTemplateDecl Implementation
//===----------------------------------------------------------------------===//

ClassTemplateDecl *ClassTemplateDecl::getCanonicalDecl() {
  ClassTemplateDecl *Template = this;
  while (Template->getPreviousDeclaration())
    Template = Template->getPreviousDeclaration();
  return Template;
}

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

void ClassTemplateDecl::getPartialSpecializations(
          llvm::SmallVectorImpl<ClassTemplatePartialSpecializationDecl *> &PS) {
  llvm::FoldingSet<ClassTemplatePartialSpecializationDecl> &PartialSpecs
    = CommonPtr->PartialSpecializations;
  PS.clear();
  PS.resize(PartialSpecs.size());
  for (llvm::FoldingSet<ClassTemplatePartialSpecializationDecl>::iterator
       P = PartialSpecs.begin(), PEnd = PartialSpecs.end();
       P != PEnd; ++P) {
    assert(!PS[P->getSequenceNumber()]);
    PS[P->getSequenceNumber()] = &*P;
  }
}

ClassTemplatePartialSpecializationDecl *
ClassTemplateDecl::findPartialSpecialization(QualType T) {
  ASTContext &Context = getASTContext();
  typedef llvm::FoldingSet<ClassTemplatePartialSpecializationDecl>::iterator
    partial_spec_iterator;
  for (partial_spec_iterator P = getPartialSpecializations().begin(),
                          PEnd = getPartialSpecializations().end();
       P != PEnd; ++P) {
    if (Context.hasSameType(P->getInjectedSpecializationType(), T))
      return &*P;
  }

  return 0;
}

QualType
ClassTemplateDecl::getInjectedClassNameSpecialization(ASTContext &Context) {
  if (!CommonPtr->InjectedClassNameType.isNull())
    return CommonPtr->InjectedClassNameType;

  // FIXME: n2800 14.6.1p1 should say how the template arguments
  // corresponding to template parameter packs should be pack
  // expansions. We already say that in 14.6.2.1p2, so it would be
  // better to fix that redundancy.

  TemplateParameterList *Params = getTemplateParameters();
  llvm::SmallVector<TemplateArgument, 16> TemplateArgs;
  TemplateArgs.reserve(Params->size());
  for (TemplateParameterList::iterator Param = Params->begin(),
                                    ParamEnd = Params->end();
       Param != ParamEnd; ++Param) {
    if (isa<TemplateTypeParmDecl>(*Param)) {
      QualType ParamType = Context.getTypeDeclType(cast<TypeDecl>(*Param));
      TemplateArgs.push_back(TemplateArgument(ParamType));
    } else if (NonTypeTemplateParmDecl *NTTP =
                 dyn_cast<NonTypeTemplateParmDecl>(*Param)) {
      Expr *E = new (Context) DeclRefExpr(NTTP,
                                          NTTP->getType().getNonReferenceType(),
                                          NTTP->getLocation());
      TemplateArgs.push_back(TemplateArgument(E));
    } else {
      TemplateTemplateParmDecl *TTP = cast<TemplateTemplateParmDecl>(*Param);
      TemplateArgs.push_back(TemplateArgument(TemplateName(TTP)));
    }
  }

  CommonPtr->InjectedClassNameType
    = Context.getTemplateSpecializationType(TemplateName(this),
                                            &TemplateArgs[0],
                                            TemplateArgs.size());
  return CommonPtr->InjectedClassNameType;
}

//===----------------------------------------------------------------------===//
// TemplateTypeParm Allocation/Deallocation Method Implementations
//===----------------------------------------------------------------------===//

TemplateTypeParmDecl *
TemplateTypeParmDecl::Create(ASTContext &C, DeclContext *DC,
                             SourceLocation L, unsigned D, unsigned P,
                             IdentifierInfo *Id, bool Typename,
                             bool ParameterPack) {
  QualType Type = C.getTemplateTypeParmType(D, P, ParameterPack, Id);
  return new (C) TemplateTypeParmDecl(DC, L, Id, Typename, Type, ParameterPack);
}

SourceLocation TemplateTypeParmDecl::getDefaultArgumentLoc() const {
  return DefaultArgument->getTypeLoc().getFullSourceRange().getBegin();
}

unsigned TemplateTypeParmDecl::getDepth() const {
  return TypeForDecl->getAs<TemplateTypeParmType>()->getDepth();
}

unsigned TemplateTypeParmDecl::getIndex() const {
  return TypeForDecl->getAs<TemplateTypeParmType>()->getIndex();
}

//===----------------------------------------------------------------------===//
// NonTypeTemplateParmDecl Method Implementations
//===----------------------------------------------------------------------===//

NonTypeTemplateParmDecl *
NonTypeTemplateParmDecl::Create(ASTContext &C, DeclContext *DC,
                                SourceLocation L, unsigned D, unsigned P,
                                IdentifierInfo *Id, QualType T,
                                TypeSourceInfo *TInfo) {
  return new (C) NonTypeTemplateParmDecl(DC, L, D, P, Id, T, TInfo);
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

//===----------------------------------------------------------------------===//
// TemplateArgumentListBuilder Implementation
//===----------------------------------------------------------------------===//

void TemplateArgumentListBuilder::Append(const TemplateArgument& Arg) {
  switch (Arg.getKind()) {
    default: break;
    case TemplateArgument::Type:
      assert(Arg.getAsType().isCanonical() && "Type must be canonical!");
      break;
  }

  assert(NumFlatArgs < MaxFlatArgs && "Argument list builder is full!");
  assert(!StructuredArgs &&
         "Can't append arguments when an argument pack has been added!");

  if (!FlatArgs)
    FlatArgs = new TemplateArgument[MaxFlatArgs];

  FlatArgs[NumFlatArgs++] = Arg;
}

void TemplateArgumentListBuilder::BeginPack() {
  assert(!AddingToPack && "Already adding to pack!");
  assert(!StructuredArgs && "Argument list already contains a pack!");

  AddingToPack = true;
  PackBeginIndex = NumFlatArgs;
}

void TemplateArgumentListBuilder::EndPack() {
  assert(AddingToPack && "Not adding to pack!");
  assert(!StructuredArgs && "Argument list already contains a pack!");

  AddingToPack = false;

  StructuredArgs = new TemplateArgument[MaxStructuredArgs];

  // First copy the flat entries over to the list  (if any)
  for (unsigned I = 0; I != PackBeginIndex; ++I) {
    NumStructuredArgs++;
    StructuredArgs[I] = FlatArgs[I];
  }

  // Next, set the pack.
  TemplateArgument *PackArgs = 0;
  unsigned NumPackArgs = NumFlatArgs - PackBeginIndex;
  if (NumPackArgs)
    PackArgs = &FlatArgs[PackBeginIndex];

  StructuredArgs[NumStructuredArgs++].setArgumentPack(PackArgs, NumPackArgs,
                                                      /*CopyArgs=*/false);
}

void TemplateArgumentListBuilder::ReleaseArgs() {
  FlatArgs = 0;
  NumFlatArgs = 0;
  MaxFlatArgs = 0;
  StructuredArgs = 0;
  NumStructuredArgs = 0;
  MaxStructuredArgs = 0;
}

//===----------------------------------------------------------------------===//
// TemplateArgumentList Implementation
//===----------------------------------------------------------------------===//
TemplateArgumentList::TemplateArgumentList(ASTContext &Context,
                                           TemplateArgumentListBuilder &Builder,
                                           bool TakeArgs)
  : FlatArguments(Builder.getFlatArguments(), TakeArgs),
    NumFlatArguments(Builder.flatSize()),
    StructuredArguments(Builder.getStructuredArguments(), TakeArgs),
    NumStructuredArguments(Builder.structuredSize()) {

  if (!TakeArgs)
    return;

  if (Builder.getStructuredArguments() == Builder.getFlatArguments())
    StructuredArguments.setInt(0);
  Builder.ReleaseArgs();
}

TemplateArgumentList::TemplateArgumentList(const TemplateArgumentList &Other)
  : FlatArguments(Other.FlatArguments.getPointer(), 1),
    NumFlatArguments(Other.flat_size()),
    StructuredArguments(Other.StructuredArguments.getPointer(), 1),
    NumStructuredArguments(Other.NumStructuredArguments) { }

TemplateArgumentList::~TemplateArgumentList() {
  // FIXME: Deallocate template arguments
}

//===----------------------------------------------------------------------===//
// ClassTemplateSpecializationDecl Implementation
//===----------------------------------------------------------------------===//
ClassTemplateSpecializationDecl::
ClassTemplateSpecializationDecl(ASTContext &Context, Kind DK,
                                DeclContext *DC, SourceLocation L,
                                ClassTemplateDecl *SpecializedTemplate,
                                TemplateArgumentListBuilder &Builder,
                                ClassTemplateSpecializationDecl *PrevDecl)
  : CXXRecordDecl(DK,
                  SpecializedTemplate->getTemplatedDecl()->getTagKind(),
                  DC, L,
                  // FIXME: Should we use DeclarationName for the name of
                  // class template specializations?
                  SpecializedTemplate->getIdentifier(),
                  PrevDecl),
    SpecializedTemplate(SpecializedTemplate),
    TypeAsWritten(0),
    TemplateArgs(Context, Builder, /*TakeArgs=*/true),
    SpecializationKind(TSK_Undeclared) {
}

ClassTemplateSpecializationDecl *
ClassTemplateSpecializationDecl::Create(ASTContext &Context,
                                        DeclContext *DC, SourceLocation L,
                                        ClassTemplateDecl *SpecializedTemplate,
                                        TemplateArgumentListBuilder &Builder,
                                   ClassTemplateSpecializationDecl *PrevDecl) {
  ClassTemplateSpecializationDecl *Result
    = new (Context)ClassTemplateSpecializationDecl(Context,
                                                   ClassTemplateSpecialization,
                                                   DC, L,
                                                   SpecializedTemplate,
                                                   Builder,
                                                   PrevDecl);
  Context.getTypeDeclType(Result, PrevDecl);
  return Result;
}

void ClassTemplateSpecializationDecl::Destroy(ASTContext &C) {
  if (SpecializedPartialSpecialization *PartialSpec
        = SpecializedTemplate.dyn_cast<SpecializedPartialSpecialization*>())
    C.Deallocate(PartialSpec);

  CXXRecordDecl::Destroy(C);
}

void
ClassTemplateSpecializationDecl::getNameForDiagnostic(std::string &S,
                                                  const PrintingPolicy &Policy,
                                                      bool Qualified) const {
  NamedDecl::getNameForDiagnostic(S, Policy, Qualified);

  const TemplateArgumentList &TemplateArgs = getTemplateArgs();
  S += TemplateSpecializationType::PrintTemplateArgumentList(
                                       TemplateArgs.getFlatArgumentList(),
                                       TemplateArgs.flat_size(),
                                                             Policy);
}

ClassTemplateDecl *
ClassTemplateSpecializationDecl::getSpecializedTemplate() const {
  if (SpecializedPartialSpecialization *PartialSpec
      = SpecializedTemplate.dyn_cast<SpecializedPartialSpecialization*>())
    return PartialSpec->PartialSpecialization->getSpecializedTemplate();
  return SpecializedTemplate.get<ClassTemplateDecl*>();
}

//===----------------------------------------------------------------------===//
// ClassTemplatePartialSpecializationDecl Implementation
//===----------------------------------------------------------------------===//
ClassTemplatePartialSpecializationDecl *
ClassTemplatePartialSpecializationDecl::
Create(ASTContext &Context, DeclContext *DC, SourceLocation L,
       TemplateParameterList *Params,
       ClassTemplateDecl *SpecializedTemplate,
       TemplateArgumentListBuilder &Builder,
       const TemplateArgumentListInfo &ArgInfos,
       QualType CanonInjectedType,
       ClassTemplatePartialSpecializationDecl *PrevDecl,
       unsigned SequenceNumber) {
  unsigned N = ArgInfos.size();
  TemplateArgumentLoc *ClonedArgs = new (Context) TemplateArgumentLoc[N];
  for (unsigned I = 0; I != N; ++I)
    ClonedArgs[I] = ArgInfos[I];

  ClassTemplatePartialSpecializationDecl *Result
    = new (Context)ClassTemplatePartialSpecializationDecl(Context,
                                                          DC, L, Params,
                                                          SpecializedTemplate,
                                                          Builder,
                                                          ClonedArgs, N,
                                                          PrevDecl,
                                                          SequenceNumber);
  Result->setSpecializationKind(TSK_ExplicitSpecialization);

  Context.getInjectedClassNameType(Result, CanonInjectedType);
  return Result;
}

//===----------------------------------------------------------------------===//
// FriendTemplateDecl Implementation
//===----------------------------------------------------------------------===//

FriendTemplateDecl *FriendTemplateDecl::Create(ASTContext &Context,
                                               DeclContext *DC,
                                               SourceLocation L,
                                               unsigned NParams,
                                               TemplateParameterList **Params,
                                               FriendUnion Friend,
                                               SourceLocation FLoc) {
  FriendTemplateDecl *Result
    = new (Context) FriendTemplateDecl(DC, L, NParams, Params, Friend, FLoc);
  return Result;
}
