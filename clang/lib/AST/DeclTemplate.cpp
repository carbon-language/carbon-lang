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

void FunctionTemplateDecl::DeallocateCommon(void *Ptr) {
  static_cast<Common *>(Ptr)->~Common();
}

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
    Common *CommonPtr = new (getASTContext()) Common;
    getASTContext().AddDeallocation(DeallocateCommon, CommonPtr);
    First->CommonOrPrev = CommonPtr;
  }
  return First->CommonOrPrev.get<Common*>();
}

//===----------------------------------------------------------------------===//
// ClassTemplateDecl Implementation
//===----------------------------------------------------------------------===//

void ClassTemplateDecl::DeallocateCommon(void *Ptr) {
  static_cast<Common *>(Ptr)->~Common();
}

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
  else {
    CommonPtr = new (C) Common;
    C.AddDeallocation(DeallocateCommon, CommonPtr);
  }

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
  return DefaultArgument->getTypeLoc().getSourceRange().getBegin();
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

void TemplateArgumentListBuilder::Append(const TemplateArgument &Arg) {
  assert((Arg.getKind() != TemplateArgument::Type ||
          Arg.getAsType().isCanonical()) && "Type must be canonical!");
  assert(FlatArgs.size() < MaxFlatArgs && "Argument list builder is full!");
  assert(!StructuredArgs &&
         "Can't append arguments when an argument pack has been added!");

  FlatArgs.push_back(Arg);
}

void TemplateArgumentListBuilder::BeginPack() {
  assert(!AddingToPack && "Already adding to pack!");
  assert(!StructuredArgs && "Argument list already contains a pack!");

  AddingToPack = true;
  PackBeginIndex = FlatArgs.size();
}

void TemplateArgumentListBuilder::EndPack() {
  assert(AddingToPack && "Not adding to pack!");
  assert(!StructuredArgs && "Argument list already contains a pack!");

  AddingToPack = false;

  // FIXME: This is a memory leak!
  StructuredArgs = new TemplateArgument[MaxStructuredArgs];

  // First copy the flat entries over to the list  (if any)
  for (unsigned I = 0; I != PackBeginIndex; ++I) {
    NumStructuredArgs++;
    StructuredArgs[I] = FlatArgs[I];
  }

  // Next, set the pack.
  TemplateArgument *PackArgs = 0;
  unsigned NumPackArgs = NumFlatArgs - PackBeginIndex;
  // FIXME: NumPackArgs shouldn't be negative here???
  if (NumPackArgs)
    PackArgs = FlatArgs.data()+PackBeginIndex;

  StructuredArgs[NumStructuredArgs++].setArgumentPack(PackArgs, NumPackArgs,
                                                      /*CopyArgs=*/false);
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

  // If this does take ownership of the arguments, then we have to new them
  // and copy over.
  TemplateArgument *NewArgs =
    new (Context) TemplateArgument[Builder.flatSize()];
  std::copy(Builder.getFlatArguments(),
            Builder.getFlatArguments()+Builder.flatSize(), NewArgs);
  FlatArguments.setPointer(NewArgs);
      
  // Just reuse the structured and flat arguments array if possible.
  if (Builder.getStructuredArguments() == Builder.getFlatArguments()) {
    StructuredArguments.setPointer(NewArgs);    
    StructuredArguments.setInt(0);
  } else {
    TemplateArgument *NewSArgs =
      new (Context) TemplateArgument[Builder.flatSize()];
    std::copy(Builder.getFlatArguments(),
              Builder.getFlatArguments()+Builder.flatSize(), NewSArgs);
    StructuredArguments.setPointer(NewSArgs);
  }
}

/// Produces a shallow copy of the given template argument list.  This
/// assumes that the input argument list outlives it.  This takes the list as
/// a pointer to avoid looking like a copy constructor, since this really
/// really isn't safe to use that way.
TemplateArgumentList::TemplateArgumentList(const TemplateArgumentList *Other)
  : FlatArguments(Other->FlatArguments.getPointer(), false),
    NumFlatArguments(Other->flat_size()),
    StructuredArguments(Other->StructuredArguments.getPointer(), false),
    NumStructuredArguments(Other->NumStructuredArguments) { }

void TemplateArgumentList::Destroy(ASTContext &C) {
  if (FlatArguments.getInt())
    C.Deallocate((void*)FlatArguments.getPointer());
  if (StructuredArguments.getInt())
    C.Deallocate((void*)StructuredArguments.getPointer());
}

TemplateArgumentList::~TemplateArgumentList() {}

//===----------------------------------------------------------------------===//
// ClassTemplateSpecializationDecl Implementation
//===----------------------------------------------------------------------===//
ClassTemplateSpecializationDecl::
ClassTemplateSpecializationDecl(ASTContext &Context, Kind DK, TagKind TK,
                                DeclContext *DC, SourceLocation L,
                                ClassTemplateDecl *SpecializedTemplate,
                                TemplateArgumentListBuilder &Builder,
                                ClassTemplateSpecializationDecl *PrevDecl)
  : CXXRecordDecl(DK, TK, DC, L,
                  SpecializedTemplate->getIdentifier(),
                  PrevDecl),
    SpecializedTemplate(SpecializedTemplate),
    TypeAsWritten(0),
    TemplateArgs(Context, Builder, /*TakeArgs=*/true),
    SpecializationKind(TSK_Undeclared) {
}

ClassTemplateSpecializationDecl *
ClassTemplateSpecializationDecl::Create(ASTContext &Context, TagKind TK,
                                        DeclContext *DC, SourceLocation L,
                                        ClassTemplateDecl *SpecializedTemplate,
                                        TemplateArgumentListBuilder &Builder,
                                   ClassTemplateSpecializationDecl *PrevDecl) {
  ClassTemplateSpecializationDecl *Result
    = new (Context)ClassTemplateSpecializationDecl(Context,
                                                   ClassTemplateSpecialization,
                                                   TK, DC, L,
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
Create(ASTContext &Context, TagKind TK,DeclContext *DC, SourceLocation L,
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
    = new (Context)ClassTemplatePartialSpecializationDecl(Context, TK,
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
