//===- DeclTemplate.cpp - Template Declaration AST Node Implementation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the C++ related Decl classes for templates.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

using namespace clang;

//===----------------------------------------------------------------------===//
// TemplateParameterList Implementation
//===----------------------------------------------------------------------===//


TemplateParameterList::TemplateParameterList(const ASTContext& C,
                                             SourceLocation TemplateLoc,
                                             SourceLocation LAngleLoc,
                                             ArrayRef<NamedDecl *> Params,
                                             SourceLocation RAngleLoc,
                                             Expr *RequiresClause)
    : TemplateLoc(TemplateLoc), LAngleLoc(LAngleLoc), RAngleLoc(RAngleLoc),
      NumParams(Params.size()), ContainsUnexpandedParameterPack(false),
      HasRequiresClause(RequiresClause != nullptr),
      HasConstrainedParameters(false) {
  for (unsigned Idx = 0; Idx < NumParams; ++Idx) {
    NamedDecl *P = Params[Idx];
    begin()[Idx] = P;

    bool IsPack = P->isTemplateParameterPack();
    if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(P)) {
      if (!IsPack && NTTP->getType()->containsUnexpandedParameterPack())
        ContainsUnexpandedParameterPack = true;
      if (NTTP->hasPlaceholderTypeConstraint())
        HasConstrainedParameters = true;
    } else if (const auto *TTP = dyn_cast<TemplateTemplateParmDecl>(P)) {
      if (!IsPack &&
          TTP->getTemplateParameters()->containsUnexpandedParameterPack())
        ContainsUnexpandedParameterPack = true;
    } else if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(P)) {
      if (const TypeConstraint *TC = TTP->getTypeConstraint()) {
        if (TC->getImmediatelyDeclaredConstraint()
            ->containsUnexpandedParameterPack())
          ContainsUnexpandedParameterPack = true;
      }
      if (TTP->hasTypeConstraint())
        HasConstrainedParameters = true;
    } else {
      llvm_unreachable("unexpected template parameter type");
    }
    // FIXME: If a default argument contains an unexpanded parameter pack, the
    // template parameter list does too.
  }

  if (HasRequiresClause) {
    if (RequiresClause->containsUnexpandedParameterPack())
      ContainsUnexpandedParameterPack = true;
    *getTrailingObjects<Expr *>() = RequiresClause;
  }
}

bool TemplateParameterList::containsUnexpandedParameterPack() const {
  if (ContainsUnexpandedParameterPack)
    return true;
  if (!HasConstrainedParameters)
    return false;

  // An implicit constrained parameter might have had a use of an unexpanded
  // pack added to it after the template parameter list was created. All
  // implicit parameters are at the end of the parameter list.
  for (const NamedDecl *Param : llvm::reverse(asArray())) {
    if (!Param->isImplicit())
      break;

    if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(Param)) {
      const auto *TC = TTP->getTypeConstraint();
      if (TC && TC->getImmediatelyDeclaredConstraint()
                    ->containsUnexpandedParameterPack())
        return true;
    }
  }

  return false;
}

TemplateParameterList *
TemplateParameterList::Create(const ASTContext &C, SourceLocation TemplateLoc,
                              SourceLocation LAngleLoc,
                              ArrayRef<NamedDecl *> Params,
                              SourceLocation RAngleLoc, Expr *RequiresClause) {
  void *Mem = C.Allocate(totalSizeToAlloc<NamedDecl *, Expr *>(
                             Params.size(), RequiresClause ? 1u : 0u),
                         alignof(TemplateParameterList));
  return new (Mem) TemplateParameterList(C, TemplateLoc, LAngleLoc, Params,
                                         RAngleLoc, RequiresClause);
}

unsigned TemplateParameterList::getMinRequiredArguments() const {
  unsigned NumRequiredArgs = 0;
  for (const NamedDecl *P : asArray()) {
    if (P->isTemplateParameterPack()) {
      if (Optional<unsigned> Expansions = getExpandedPackSize(P)) {
        NumRequiredArgs += *Expansions;
        continue;
      }
      break;
    }

    if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(P)) {
      if (TTP->hasDefaultArgument())
        break;
    } else if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(P)) {
      if (NTTP->hasDefaultArgument())
        break;
    } else if (cast<TemplateTemplateParmDecl>(P)->hasDefaultArgument())
      break;

    ++NumRequiredArgs;
  }

  return NumRequiredArgs;
}

unsigned TemplateParameterList::getDepth() const {
  if (size() == 0)
    return 0;

  const NamedDecl *FirstParm = getParam(0);
  if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(FirstParm))
    return TTP->getDepth();
  else if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(FirstParm))
    return NTTP->getDepth();
  else
    return cast<TemplateTemplateParmDecl>(FirstParm)->getDepth();
}

static bool AdoptTemplateParameterList(TemplateParameterList *Params,
                                       DeclContext *Owner) {
  bool Invalid = false;
  for (NamedDecl *P : *Params) {
    P->setDeclContext(Owner);

    if (const auto *TTP = dyn_cast<TemplateTemplateParmDecl>(P))
      if (AdoptTemplateParameterList(TTP->getTemplateParameters(), Owner))
        Invalid = true;

    if (P->isInvalidDecl())
      Invalid = true;
  }
  return Invalid;
}

void TemplateParameterList::
getAssociatedConstraints(llvm::SmallVectorImpl<const Expr *> &AC) const {
  if (HasConstrainedParameters)
    for (const NamedDecl *Param : *this) {
      if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(Param)) {
        if (const auto *TC = TTP->getTypeConstraint())
          AC.push_back(TC->getImmediatelyDeclaredConstraint());
      } else if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(Param)) {
        if (const Expr *E = NTTP->getPlaceholderTypeConstraint())
          AC.push_back(E);
      }
    }
  if (HasRequiresClause)
    AC.push_back(getRequiresClause());
}

bool TemplateParameterList::hasAssociatedConstraints() const {
  return HasRequiresClause || HasConstrainedParameters;
}

bool TemplateParameterList::shouldIncludeTypeForArgument(
    const PrintingPolicy &Policy, const TemplateParameterList *TPL,
    unsigned Idx) {
  if (!TPL || Idx >= TPL->size() || Policy.AlwaysIncludeTypeForTemplateArgument)
    return true;
  const NamedDecl *TemplParam = TPL->getParam(Idx);
  if (const auto *ParamValueDecl =
          dyn_cast<NonTypeTemplateParmDecl>(TemplParam))
    if (ParamValueDecl->getType()->getContainedDeducedType())
      return true;
  return false;
}

namespace clang {

void *allocateDefaultArgStorageChain(const ASTContext &C) {
  return new (C) char[sizeof(void*) * 2];
}

} // namespace clang

//===----------------------------------------------------------------------===//
// TemplateDecl Implementation
//===----------------------------------------------------------------------===//

TemplateDecl::TemplateDecl(Kind DK, DeclContext *DC, SourceLocation L,
                           DeclarationName Name, TemplateParameterList *Params,
                           NamedDecl *Decl)
    : NamedDecl(DK, DC, L, Name), TemplatedDecl(Decl), TemplateParams(Params) {}

void TemplateDecl::anchor() {}

void TemplateDecl::
getAssociatedConstraints(llvm::SmallVectorImpl<const Expr *> &AC) const {
  TemplateParams->getAssociatedConstraints(AC);
  if (auto *FD = dyn_cast_or_null<FunctionDecl>(getTemplatedDecl()))
    if (const Expr *TRC = FD->getTrailingRequiresClause())
      AC.push_back(TRC);
}

bool TemplateDecl::hasAssociatedConstraints() const {
  if (TemplateParams->hasAssociatedConstraints())
    return true;
  if (auto *FD = dyn_cast_or_null<FunctionDecl>(getTemplatedDecl()))
    return FD->getTrailingRequiresClause();
  return false;
}

//===----------------------------------------------------------------------===//
// RedeclarableTemplateDecl Implementation
//===----------------------------------------------------------------------===//

void RedeclarableTemplateDecl::anchor() {}

RedeclarableTemplateDecl::CommonBase *RedeclarableTemplateDecl::getCommonPtr() const {
  if (Common)
    return Common;

  // Walk the previous-declaration chain until we either find a declaration
  // with a common pointer or we run out of previous declarations.
  SmallVector<const RedeclarableTemplateDecl *, 2> PrevDecls;
  for (const RedeclarableTemplateDecl *Prev = getPreviousDecl(); Prev;
       Prev = Prev->getPreviousDecl()) {
    if (Prev->Common) {
      Common = Prev->Common;
      break;
    }

    PrevDecls.push_back(Prev);
  }

  // If we never found a common pointer, allocate one now.
  if (!Common) {
    // FIXME: If any of the declarations is from an AST file, we probably
    // need an update record to add the common data.

    Common = newCommon(getASTContext());
  }

  // Update any previous declarations we saw with the common pointer.
  for (const RedeclarableTemplateDecl *Prev : PrevDecls)
    Prev->Common = Common;

  return Common;
}

void RedeclarableTemplateDecl::loadLazySpecializationsImpl() const {
  // Grab the most recent declaration to ensure we've loaded any lazy
  // redeclarations of this template.
  CommonBase *CommonBasePtr = getMostRecentDecl()->getCommonPtr();
  if (CommonBasePtr->LazySpecializations) {
    ASTContext &Context = getASTContext();
    uint32_t *Specs = CommonBasePtr->LazySpecializations;
    CommonBasePtr->LazySpecializations = nullptr;
    for (uint32_t I = 0, N = *Specs++; I != N; ++I)
      (void)Context.getExternalSource()->GetExternalDecl(Specs[I]);
  }
}

template<class EntryType, typename... ProfileArguments>
typename RedeclarableTemplateDecl::SpecEntryTraits<EntryType>::DeclType *
RedeclarableTemplateDecl::findSpecializationImpl(
    llvm::FoldingSetVector<EntryType> &Specs, void *&InsertPos,
    ProfileArguments&&... ProfileArgs) {
  using SETraits = SpecEntryTraits<EntryType>;

  llvm::FoldingSetNodeID ID;
  EntryType::Profile(ID, std::forward<ProfileArguments>(ProfileArgs)...,
                     getASTContext());
  EntryType *Entry = Specs.FindNodeOrInsertPos(ID, InsertPos);
  return Entry ? SETraits::getDecl(Entry)->getMostRecentDecl() : nullptr;
}

template<class Derived, class EntryType>
void RedeclarableTemplateDecl::addSpecializationImpl(
    llvm::FoldingSetVector<EntryType> &Specializations, EntryType *Entry,
    void *InsertPos) {
  using SETraits = SpecEntryTraits<EntryType>;

  if (InsertPos) {
#ifndef NDEBUG
    void *CorrectInsertPos;
    assert(!findSpecializationImpl(Specializations,
                                   CorrectInsertPos,
                                   SETraits::getTemplateArgs(Entry)) &&
           InsertPos == CorrectInsertPos &&
           "given incorrect InsertPos for specialization");
#endif
    Specializations.InsertNode(Entry, InsertPos);
  } else {
    EntryType *Existing = Specializations.GetOrInsertNode(Entry);
    (void)Existing;
    assert(SETraits::getDecl(Existing)->isCanonicalDecl() &&
           "non-canonical specialization?");
  }

  if (ASTMutationListener *L = getASTMutationListener())
    L->AddedCXXTemplateSpecialization(cast<Derived>(this),
                                      SETraits::getDecl(Entry));
}

//===----------------------------------------------------------------------===//
// FunctionTemplateDecl Implementation
//===----------------------------------------------------------------------===//

FunctionTemplateDecl *
FunctionTemplateDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                             DeclarationName Name,
                             TemplateParameterList *Params, NamedDecl *Decl) {
  bool Invalid = AdoptTemplateParameterList(Params, cast<DeclContext>(Decl));
  auto *TD = new (C, DC) FunctionTemplateDecl(C, DC, L, Name, Params, Decl);
  if (Invalid)
    TD->setInvalidDecl();
  return TD;
}

FunctionTemplateDecl *FunctionTemplateDecl::CreateDeserialized(ASTContext &C,
                                                               unsigned ID) {
  return new (C, ID) FunctionTemplateDecl(C, nullptr, SourceLocation(),
                                          DeclarationName(), nullptr, nullptr);
}

RedeclarableTemplateDecl::CommonBase *
FunctionTemplateDecl::newCommon(ASTContext &C) const {
  auto *CommonPtr = new (C) Common;
  C.addDestruction(CommonPtr);
  return CommonPtr;
}

void FunctionTemplateDecl::LoadLazySpecializations() const {
  loadLazySpecializationsImpl();
}

llvm::FoldingSetVector<FunctionTemplateSpecializationInfo> &
FunctionTemplateDecl::getSpecializations() const {
  LoadLazySpecializations();
  return getCommonPtr()->Specializations;
}

FunctionDecl *
FunctionTemplateDecl::findSpecialization(ArrayRef<TemplateArgument> Args,
                                         void *&InsertPos) {
  return findSpecializationImpl(getSpecializations(), InsertPos, Args);
}

void FunctionTemplateDecl::addSpecialization(
      FunctionTemplateSpecializationInfo *Info, void *InsertPos) {
  addSpecializationImpl<FunctionTemplateDecl>(getSpecializations(), Info,
                                              InsertPos);
}

ArrayRef<TemplateArgument> FunctionTemplateDecl::getInjectedTemplateArgs() {
  TemplateParameterList *Params = getTemplateParameters();
  Common *CommonPtr = getCommonPtr();
  if (!CommonPtr->InjectedArgs) {
    auto &Context = getASTContext();
    SmallVector<TemplateArgument, 16> TemplateArgs;
    Context.getInjectedTemplateArgs(Params, TemplateArgs);
    CommonPtr->InjectedArgs =
        new (Context) TemplateArgument[TemplateArgs.size()];
    std::copy(TemplateArgs.begin(), TemplateArgs.end(),
              CommonPtr->InjectedArgs);
  }

  return llvm::makeArrayRef(CommonPtr->InjectedArgs, Params->size());
}

void FunctionTemplateDecl::mergePrevDecl(FunctionTemplateDecl *Prev) {
  using Base = RedeclarableTemplateDecl;

  // If we haven't created a common pointer yet, then it can just be created
  // with the usual method.
  if (!Base::Common)
    return;

  Common *ThisCommon = static_cast<Common *>(Base::Common);
  Common *PrevCommon = nullptr;
  SmallVector<FunctionTemplateDecl *, 8> PreviousDecls;
  for (; Prev; Prev = Prev->getPreviousDecl()) {
    if (Prev->Base::Common) {
      PrevCommon = static_cast<Common *>(Prev->Base::Common);
      break;
    }
    PreviousDecls.push_back(Prev);
  }

  // If the previous redecl chain hasn't created a common pointer yet, then just
  // use this common pointer.
  if (!PrevCommon) {
    for (auto *D : PreviousDecls)
      D->Base::Common = ThisCommon;
    return;
  }

  // Ensure we don't leak any important state.
  assert(ThisCommon->Specializations.size() == 0 &&
         "Can't merge incompatible declarations!");

  Base::Common = PrevCommon;
}

//===----------------------------------------------------------------------===//
// ClassTemplateDecl Implementation
//===----------------------------------------------------------------------===//

ClassTemplateDecl *ClassTemplateDecl::Create(ASTContext &C, DeclContext *DC,
                                             SourceLocation L,
                                             DeclarationName Name,
                                             TemplateParameterList *Params,
                                             NamedDecl *Decl) {
  bool Invalid = AdoptTemplateParameterList(Params, cast<DeclContext>(Decl));
  auto *TD = new (C, DC) ClassTemplateDecl(C, DC, L, Name, Params, Decl);
  if (Invalid)
    TD->setInvalidDecl();
  return TD;
}

ClassTemplateDecl *ClassTemplateDecl::CreateDeserialized(ASTContext &C,
                                                         unsigned ID) {
  return new (C, ID) ClassTemplateDecl(C, nullptr, SourceLocation(),
                                       DeclarationName(), nullptr, nullptr);
}

void ClassTemplateDecl::LoadLazySpecializations() const {
  loadLazySpecializationsImpl();
}

llvm::FoldingSetVector<ClassTemplateSpecializationDecl> &
ClassTemplateDecl::getSpecializations() const {
  LoadLazySpecializations();
  return getCommonPtr()->Specializations;
}

llvm::FoldingSetVector<ClassTemplatePartialSpecializationDecl> &
ClassTemplateDecl::getPartialSpecializations() const {
  LoadLazySpecializations();
  return getCommonPtr()->PartialSpecializations;
}

RedeclarableTemplateDecl::CommonBase *
ClassTemplateDecl::newCommon(ASTContext &C) const {
  auto *CommonPtr = new (C) Common;
  C.addDestruction(CommonPtr);
  return CommonPtr;
}

ClassTemplateSpecializationDecl *
ClassTemplateDecl::findSpecialization(ArrayRef<TemplateArgument> Args,
                                      void *&InsertPos) {
  return findSpecializationImpl(getSpecializations(), InsertPos, Args);
}

void ClassTemplateDecl::AddSpecialization(ClassTemplateSpecializationDecl *D,
                                          void *InsertPos) {
  addSpecializationImpl<ClassTemplateDecl>(getSpecializations(), D, InsertPos);
}

ClassTemplatePartialSpecializationDecl *
ClassTemplateDecl::findPartialSpecialization(
    ArrayRef<TemplateArgument> Args,
    TemplateParameterList *TPL, void *&InsertPos) {
  return findSpecializationImpl(getPartialSpecializations(), InsertPos, Args,
                                TPL);
}

static void ProfileTemplateParameterList(ASTContext &C,
    llvm::FoldingSetNodeID &ID, const TemplateParameterList *TPL) {
  const Expr *RC = TPL->getRequiresClause();
  ID.AddBoolean(RC != nullptr);
  if (RC)
    RC->Profile(ID, C, /*Canonical=*/true);
  ID.AddInteger(TPL->size());
  for (NamedDecl *D : *TPL) {
    if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(D)) {
      ID.AddInteger(0);
      ID.AddBoolean(NTTP->isParameterPack());
      NTTP->getType().getCanonicalType().Profile(ID);
      continue;
    }
    if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(D)) {
      ID.AddInteger(1);
      ID.AddBoolean(TTP->isParameterPack());
      ID.AddBoolean(TTP->hasTypeConstraint());
      if (const TypeConstraint *TC = TTP->getTypeConstraint())
        TC->getImmediatelyDeclaredConstraint()->Profile(ID, C,
                                                        /*Canonical=*/true);
      continue;
    }
    const auto *TTP = cast<TemplateTemplateParmDecl>(D);
    ID.AddInteger(2);
    ID.AddBoolean(TTP->isParameterPack());
    ProfileTemplateParameterList(C, ID, TTP->getTemplateParameters());
  }
}

void
ClassTemplatePartialSpecializationDecl::Profile(llvm::FoldingSetNodeID &ID,
    ArrayRef<TemplateArgument> TemplateArgs, TemplateParameterList *TPL,
    ASTContext &Context) {
  ID.AddInteger(TemplateArgs.size());
  for (const TemplateArgument &TemplateArg : TemplateArgs)
    TemplateArg.Profile(ID, Context);
  ProfileTemplateParameterList(Context, ID, TPL);
}

void ClassTemplateDecl::AddPartialSpecialization(
                                      ClassTemplatePartialSpecializationDecl *D,
                                      void *InsertPos) {
  if (InsertPos)
    getPartialSpecializations().InsertNode(D, InsertPos);
  else {
    ClassTemplatePartialSpecializationDecl *Existing
      = getPartialSpecializations().GetOrInsertNode(D);
    (void)Existing;
    assert(Existing->isCanonicalDecl() && "Non-canonical specialization?");
  }

  if (ASTMutationListener *L = getASTMutationListener())
    L->AddedCXXTemplateSpecialization(this, D);
}

void ClassTemplateDecl::getPartialSpecializations(
    SmallVectorImpl<ClassTemplatePartialSpecializationDecl *> &PS) const {
  llvm::FoldingSetVector<ClassTemplatePartialSpecializationDecl> &PartialSpecs
    = getPartialSpecializations();
  PS.clear();
  PS.reserve(PartialSpecs.size());
  for (ClassTemplatePartialSpecializationDecl &P : PartialSpecs)
    PS.push_back(P.getMostRecentDecl());
}

ClassTemplatePartialSpecializationDecl *
ClassTemplateDecl::findPartialSpecialization(QualType T) {
  ASTContext &Context = getASTContext();
  for (ClassTemplatePartialSpecializationDecl &P :
       getPartialSpecializations()) {
    if (Context.hasSameType(P.getInjectedSpecializationType(), T))
      return P.getMostRecentDecl();
  }

  return nullptr;
}

ClassTemplatePartialSpecializationDecl *
ClassTemplateDecl::findPartialSpecInstantiatedFromMember(
                                    ClassTemplatePartialSpecializationDecl *D) {
  Decl *DCanon = D->getCanonicalDecl();
  for (ClassTemplatePartialSpecializationDecl &P : getPartialSpecializations()) {
    if (P.getInstantiatedFromMember()->getCanonicalDecl() == DCanon)
      return P.getMostRecentDecl();
  }

  return nullptr;
}

QualType
ClassTemplateDecl::getInjectedClassNameSpecialization() {
  Common *CommonPtr = getCommonPtr();
  if (!CommonPtr->InjectedClassNameType.isNull())
    return CommonPtr->InjectedClassNameType;

  // C++0x [temp.dep.type]p2:
  //  The template argument list of a primary template is a template argument
  //  list in which the nth template argument has the value of the nth template
  //  parameter of the class template. If the nth template parameter is a
  //  template parameter pack (14.5.3), the nth template argument is a pack
  //  expansion (14.5.3) whose pattern is the name of the template parameter
  //  pack.
  ASTContext &Context = getASTContext();
  TemplateParameterList *Params = getTemplateParameters();
  SmallVector<TemplateArgument, 16> TemplateArgs;
  Context.getInjectedTemplateArgs(Params, TemplateArgs);
  CommonPtr->InjectedClassNameType
    = Context.getTemplateSpecializationType(TemplateName(this),
                                            TemplateArgs);
  return CommonPtr->InjectedClassNameType;
}

//===----------------------------------------------------------------------===//
// TemplateTypeParm Allocation/Deallocation Method Implementations
//===----------------------------------------------------------------------===//

TemplateTypeParmDecl *
TemplateTypeParmDecl::Create(const ASTContext &C, DeclContext *DC,
                             SourceLocation KeyLoc, SourceLocation NameLoc,
                             unsigned D, unsigned P, IdentifierInfo *Id,
                             bool Typename, bool ParameterPack,
                             bool HasTypeConstraint,
                             Optional<unsigned> NumExpanded) {
  auto *TTPDecl =
      new (C, DC,
           additionalSizeToAlloc<TypeConstraint>(HasTypeConstraint ? 1 : 0))
      TemplateTypeParmDecl(DC, KeyLoc, NameLoc, Id, Typename,
                           HasTypeConstraint, NumExpanded);
  QualType TTPType = C.getTemplateTypeParmType(D, P, ParameterPack, TTPDecl);
  TTPDecl->setTypeForDecl(TTPType.getTypePtr());
  return TTPDecl;
}

TemplateTypeParmDecl *
TemplateTypeParmDecl::CreateDeserialized(const ASTContext &C, unsigned ID) {
  return new (C, ID) TemplateTypeParmDecl(nullptr, SourceLocation(),
                                          SourceLocation(), nullptr, false,
                                          false, None);
}

TemplateTypeParmDecl *
TemplateTypeParmDecl::CreateDeserialized(const ASTContext &C, unsigned ID,
                                         bool HasTypeConstraint) {
  return new (C, ID,
              additionalSizeToAlloc<TypeConstraint>(HasTypeConstraint ? 1 : 0))
         TemplateTypeParmDecl(nullptr, SourceLocation(), SourceLocation(),
                              nullptr, false, HasTypeConstraint, None);
}

SourceLocation TemplateTypeParmDecl::getDefaultArgumentLoc() const {
  return hasDefaultArgument()
             ? getDefaultArgumentInfo()->getTypeLoc().getBeginLoc()
             : SourceLocation();
}

SourceRange TemplateTypeParmDecl::getSourceRange() const {
  if (hasDefaultArgument() && !defaultArgumentWasInherited())
    return SourceRange(getBeginLoc(),
                       getDefaultArgumentInfo()->getTypeLoc().getEndLoc());
  // TypeDecl::getSourceRange returns a range containing name location, which is
  // wrong for unnamed template parameters. e.g:
  // it will return <[[typename>]] instead of <[[typename]]>
  else if (getDeclName().isEmpty())
    return SourceRange(getBeginLoc());
  return TypeDecl::getSourceRange();
}

unsigned TemplateTypeParmDecl::getDepth() const {
  return getTypeForDecl()->castAs<TemplateTypeParmType>()->getDepth();
}

unsigned TemplateTypeParmDecl::getIndex() const {
  return getTypeForDecl()->castAs<TemplateTypeParmType>()->getIndex();
}

bool TemplateTypeParmDecl::isParameterPack() const {
  return getTypeForDecl()->castAs<TemplateTypeParmType>()->isParameterPack();
}

void TemplateTypeParmDecl::setTypeConstraint(NestedNameSpecifierLoc NNS,
    DeclarationNameInfo NameInfo, NamedDecl *FoundDecl, ConceptDecl *CD,
    const ASTTemplateArgumentListInfo *ArgsAsWritten,
    Expr *ImmediatelyDeclaredConstraint) {
  assert(HasTypeConstraint &&
         "HasTypeConstraint=true must be passed at construction in order to "
         "call setTypeConstraint");
  assert(!TypeConstraintInitialized &&
         "TypeConstraint was already initialized!");
  new (getTrailingObjects<TypeConstraint>()) TypeConstraint(NNS, NameInfo,
      FoundDecl, CD, ArgsAsWritten, ImmediatelyDeclaredConstraint);
  TypeConstraintInitialized = true;
}

//===----------------------------------------------------------------------===//
// NonTypeTemplateParmDecl Method Implementations
//===----------------------------------------------------------------------===//

NonTypeTemplateParmDecl::NonTypeTemplateParmDecl(
    DeclContext *DC, SourceLocation StartLoc, SourceLocation IdLoc, unsigned D,
    unsigned P, IdentifierInfo *Id, QualType T, TypeSourceInfo *TInfo,
    ArrayRef<QualType> ExpandedTypes, ArrayRef<TypeSourceInfo *> ExpandedTInfos)
    : DeclaratorDecl(NonTypeTemplateParm, DC, IdLoc, Id, T, TInfo, StartLoc),
      TemplateParmPosition(D, P), ParameterPack(true),
      ExpandedParameterPack(true), NumExpandedTypes(ExpandedTypes.size()) {
  if (!ExpandedTypes.empty() && !ExpandedTInfos.empty()) {
    auto TypesAndInfos =
        getTrailingObjects<std::pair<QualType, TypeSourceInfo *>>();
    for (unsigned I = 0; I != NumExpandedTypes; ++I) {
      new (&TypesAndInfos[I].first) QualType(ExpandedTypes[I]);
      TypesAndInfos[I].second = ExpandedTInfos[I];
    }
  }
}

NonTypeTemplateParmDecl *
NonTypeTemplateParmDecl::Create(const ASTContext &C, DeclContext *DC,
                                SourceLocation StartLoc, SourceLocation IdLoc,
                                unsigned D, unsigned P, IdentifierInfo *Id,
                                QualType T, bool ParameterPack,
                                TypeSourceInfo *TInfo) {
  AutoType *AT =
      C.getLangOpts().CPlusPlus20 ? T->getContainedAutoType() : nullptr;
  return new (C, DC,
              additionalSizeToAlloc<std::pair<QualType, TypeSourceInfo *>,
                                    Expr *>(0,
                                            AT && AT->isConstrained() ? 1 : 0))
      NonTypeTemplateParmDecl(DC, StartLoc, IdLoc, D, P, Id, T, ParameterPack,
                              TInfo);
}

NonTypeTemplateParmDecl *NonTypeTemplateParmDecl::Create(
    const ASTContext &C, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, unsigned D, unsigned P, IdentifierInfo *Id,
    QualType T, TypeSourceInfo *TInfo, ArrayRef<QualType> ExpandedTypes,
    ArrayRef<TypeSourceInfo *> ExpandedTInfos) {
  AutoType *AT = TInfo->getType()->getContainedAutoType();
  return new (C, DC,
              additionalSizeToAlloc<std::pair<QualType, TypeSourceInfo *>,
                                    Expr *>(
                  ExpandedTypes.size(), AT && AT->isConstrained() ? 1 : 0))
      NonTypeTemplateParmDecl(DC, StartLoc, IdLoc, D, P, Id, T, TInfo,
                              ExpandedTypes, ExpandedTInfos);
}

NonTypeTemplateParmDecl *
NonTypeTemplateParmDecl::CreateDeserialized(ASTContext &C, unsigned ID,
                                            bool HasTypeConstraint) {
  return new (C, ID, additionalSizeToAlloc<std::pair<QualType,
                                                     TypeSourceInfo *>,
                                           Expr *>(0,
                                                   HasTypeConstraint ? 1 : 0))
          NonTypeTemplateParmDecl(nullptr, SourceLocation(), SourceLocation(),
                                  0, 0, nullptr, QualType(), false, nullptr);
}

NonTypeTemplateParmDecl *
NonTypeTemplateParmDecl::CreateDeserialized(ASTContext &C, unsigned ID,
                                            unsigned NumExpandedTypes,
                                            bool HasTypeConstraint) {
  auto *NTTP =
      new (C, ID, additionalSizeToAlloc<std::pair<QualType, TypeSourceInfo *>,
                                        Expr *>(
                      NumExpandedTypes, HasTypeConstraint ? 1 : 0))
          NonTypeTemplateParmDecl(nullptr, SourceLocation(), SourceLocation(),
                                  0, 0, nullptr, QualType(), nullptr, None,
                                  None);
  NTTP->NumExpandedTypes = NumExpandedTypes;
  return NTTP;
}

SourceRange NonTypeTemplateParmDecl::getSourceRange() const {
  if (hasDefaultArgument() && !defaultArgumentWasInherited())
    return SourceRange(getOuterLocStart(),
                       getDefaultArgument()->getSourceRange().getEnd());
  return DeclaratorDecl::getSourceRange();
}

SourceLocation NonTypeTemplateParmDecl::getDefaultArgumentLoc() const {
  return hasDefaultArgument()
    ? getDefaultArgument()->getSourceRange().getBegin()
    : SourceLocation();
}

//===----------------------------------------------------------------------===//
// TemplateTemplateParmDecl Method Implementations
//===----------------------------------------------------------------------===//

void TemplateTemplateParmDecl::anchor() {}

TemplateTemplateParmDecl::TemplateTemplateParmDecl(
    DeclContext *DC, SourceLocation L, unsigned D, unsigned P,
    IdentifierInfo *Id, TemplateParameterList *Params,
    ArrayRef<TemplateParameterList *> Expansions)
    : TemplateDecl(TemplateTemplateParm, DC, L, Id, Params),
      TemplateParmPosition(D, P), ParameterPack(true),
      ExpandedParameterPack(true), NumExpandedParams(Expansions.size()) {
  if (!Expansions.empty())
    std::uninitialized_copy(Expansions.begin(), Expansions.end(),
                            getTrailingObjects<TemplateParameterList *>());
}

TemplateTemplateParmDecl *
TemplateTemplateParmDecl::Create(const ASTContext &C, DeclContext *DC,
                                 SourceLocation L, unsigned D, unsigned P,
                                 bool ParameterPack, IdentifierInfo *Id,
                                 TemplateParameterList *Params) {
  return new (C, DC) TemplateTemplateParmDecl(DC, L, D, P, ParameterPack, Id,
                                              Params);
}

TemplateTemplateParmDecl *
TemplateTemplateParmDecl::Create(const ASTContext &C, DeclContext *DC,
                                 SourceLocation L, unsigned D, unsigned P,
                                 IdentifierInfo *Id,
                                 TemplateParameterList *Params,
                                 ArrayRef<TemplateParameterList *> Expansions) {
  return new (C, DC,
              additionalSizeToAlloc<TemplateParameterList *>(Expansions.size()))
      TemplateTemplateParmDecl(DC, L, D, P, Id, Params, Expansions);
}

TemplateTemplateParmDecl *
TemplateTemplateParmDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  return new (C, ID) TemplateTemplateParmDecl(nullptr, SourceLocation(), 0, 0,
                                              false, nullptr, nullptr);
}

TemplateTemplateParmDecl *
TemplateTemplateParmDecl::CreateDeserialized(ASTContext &C, unsigned ID,
                                             unsigned NumExpansions) {
  auto *TTP =
      new (C, ID, additionalSizeToAlloc<TemplateParameterList *>(NumExpansions))
          TemplateTemplateParmDecl(nullptr, SourceLocation(), 0, 0, nullptr,
                                   nullptr, None);
  TTP->NumExpandedParams = NumExpansions;
  return TTP;
}

SourceLocation TemplateTemplateParmDecl::getDefaultArgumentLoc() const {
  return hasDefaultArgument() ? getDefaultArgument().getLocation()
                              : SourceLocation();
}

void TemplateTemplateParmDecl::setDefaultArgument(
    const ASTContext &C, const TemplateArgumentLoc &DefArg) {
  if (DefArg.getArgument().isNull())
    DefaultArgument.set(nullptr);
  else
    DefaultArgument.set(new (C) TemplateArgumentLoc(DefArg));
}

//===----------------------------------------------------------------------===//
// TemplateArgumentList Implementation
//===----------------------------------------------------------------------===//
TemplateArgumentList::TemplateArgumentList(ArrayRef<TemplateArgument> Args)
    : Arguments(getTrailingObjects<TemplateArgument>()),
      NumArguments(Args.size()) {
  std::uninitialized_copy(Args.begin(), Args.end(),
                          getTrailingObjects<TemplateArgument>());
}

TemplateArgumentList *
TemplateArgumentList::CreateCopy(ASTContext &Context,
                                 ArrayRef<TemplateArgument> Args) {
  void *Mem = Context.Allocate(totalSizeToAlloc<TemplateArgument>(Args.size()));
  return new (Mem) TemplateArgumentList(Args);
}

FunctionTemplateSpecializationInfo *FunctionTemplateSpecializationInfo::Create(
    ASTContext &C, FunctionDecl *FD, FunctionTemplateDecl *Template,
    TemplateSpecializationKind TSK, const TemplateArgumentList *TemplateArgs,
    const TemplateArgumentListInfo *TemplateArgsAsWritten, SourceLocation POI,
    MemberSpecializationInfo *MSInfo) {
  const ASTTemplateArgumentListInfo *ArgsAsWritten = nullptr;
  if (TemplateArgsAsWritten)
    ArgsAsWritten = ASTTemplateArgumentListInfo::Create(C,
                                                        *TemplateArgsAsWritten);

  void *Mem =
      C.Allocate(totalSizeToAlloc<MemberSpecializationInfo *>(MSInfo ? 1 : 0));
  return new (Mem) FunctionTemplateSpecializationInfo(
      FD, Template, TSK, TemplateArgs, ArgsAsWritten, POI, MSInfo);
}

//===----------------------------------------------------------------------===//
// ClassTemplateSpecializationDecl Implementation
//===----------------------------------------------------------------------===//

ClassTemplateSpecializationDecl::
ClassTemplateSpecializationDecl(ASTContext &Context, Kind DK, TagKind TK,
                                DeclContext *DC, SourceLocation StartLoc,
                                SourceLocation IdLoc,
                                ClassTemplateDecl *SpecializedTemplate,
                                ArrayRef<TemplateArgument> Args,
                                ClassTemplateSpecializationDecl *PrevDecl)
    : CXXRecordDecl(DK, TK, Context, DC, StartLoc, IdLoc,
                    SpecializedTemplate->getIdentifier(), PrevDecl),
    SpecializedTemplate(SpecializedTemplate),
    TemplateArgs(TemplateArgumentList::CreateCopy(Context, Args)),
    SpecializationKind(TSK_Undeclared) {
}

ClassTemplateSpecializationDecl::ClassTemplateSpecializationDecl(ASTContext &C,
                                                                 Kind DK)
    : CXXRecordDecl(DK, TTK_Struct, C, nullptr, SourceLocation(),
                    SourceLocation(), nullptr, nullptr),
      SpecializationKind(TSK_Undeclared) {}

ClassTemplateSpecializationDecl *
ClassTemplateSpecializationDecl::Create(ASTContext &Context, TagKind TK,
                                        DeclContext *DC,
                                        SourceLocation StartLoc,
                                        SourceLocation IdLoc,
                                        ClassTemplateDecl *SpecializedTemplate,
                                        ArrayRef<TemplateArgument> Args,
                                   ClassTemplateSpecializationDecl *PrevDecl) {
  auto *Result =
      new (Context, DC) ClassTemplateSpecializationDecl(
          Context, ClassTemplateSpecialization, TK, DC, StartLoc, IdLoc,
          SpecializedTemplate, Args, PrevDecl);
  Result->setMayHaveOutOfDateDef(false);

  Context.getTypeDeclType(Result, PrevDecl);
  return Result;
}

ClassTemplateSpecializationDecl *
ClassTemplateSpecializationDecl::CreateDeserialized(ASTContext &C,
                                                    unsigned ID) {
  auto *Result =
    new (C, ID) ClassTemplateSpecializationDecl(C, ClassTemplateSpecialization);
  Result->setMayHaveOutOfDateDef(false);
  return Result;
}

void ClassTemplateSpecializationDecl::getNameForDiagnostic(
    raw_ostream &OS, const PrintingPolicy &Policy, bool Qualified) const {
  NamedDecl::getNameForDiagnostic(OS, Policy, Qualified);

  const auto *PS = dyn_cast<ClassTemplatePartialSpecializationDecl>(this);
  if (const ASTTemplateArgumentListInfo *ArgsAsWritten =
          PS ? PS->getTemplateArgsAsWritten() : nullptr) {
    printTemplateArgumentList(
        OS, ArgsAsWritten->arguments(), Policy,
        getSpecializedTemplate()->getTemplateParameters());
  } else {
    const TemplateArgumentList &TemplateArgs = getTemplateArgs();
    printTemplateArgumentList(
        OS, TemplateArgs.asArray(), Policy,
        getSpecializedTemplate()->getTemplateParameters());
  }
}

ClassTemplateDecl *
ClassTemplateSpecializationDecl::getSpecializedTemplate() const {
  if (const auto *PartialSpec =
          SpecializedTemplate.dyn_cast<SpecializedPartialSpecialization*>())
    return PartialSpec->PartialSpecialization->getSpecializedTemplate();
  return SpecializedTemplate.get<ClassTemplateDecl*>();
}

SourceRange
ClassTemplateSpecializationDecl::getSourceRange() const {
  if (ExplicitInfo) {
    SourceLocation Begin = getTemplateKeywordLoc();
    if (Begin.isValid()) {
      // Here we have an explicit (partial) specialization or instantiation.
      assert(getSpecializationKind() == TSK_ExplicitSpecialization ||
             getSpecializationKind() == TSK_ExplicitInstantiationDeclaration ||
             getSpecializationKind() == TSK_ExplicitInstantiationDefinition);
      if (getExternLoc().isValid())
        Begin = getExternLoc();
      SourceLocation End = getBraceRange().getEnd();
      if (End.isInvalid())
        End = getTypeAsWritten()->getTypeLoc().getEndLoc();
      return SourceRange(Begin, End);
    }
    // An implicit instantiation of a class template partial specialization
    // uses ExplicitInfo to record the TypeAsWritten, but the source
    // locations should be retrieved from the instantiation pattern.
    using CTPSDecl = ClassTemplatePartialSpecializationDecl;
    auto *ctpsd = const_cast<CTPSDecl *>(cast<CTPSDecl>(this));
    CTPSDecl *inst_from = ctpsd->getInstantiatedFromMember();
    assert(inst_from != nullptr);
    return inst_from->getSourceRange();
  }
  else {
    // No explicit info available.
    llvm::PointerUnion<ClassTemplateDecl *,
                       ClassTemplatePartialSpecializationDecl *>
      inst_from = getInstantiatedFrom();
    if (inst_from.isNull())
      return getSpecializedTemplate()->getSourceRange();
    if (const auto *ctd = inst_from.dyn_cast<ClassTemplateDecl *>())
      return ctd->getSourceRange();
    return inst_from.get<ClassTemplatePartialSpecializationDecl *>()
      ->getSourceRange();
  }
}

//===----------------------------------------------------------------------===//
// ConceptDecl Implementation
//===----------------------------------------------------------------------===//
ConceptDecl *ConceptDecl::Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation L, DeclarationName Name,
                                 TemplateParameterList *Params,
                                 Expr *ConstraintExpr) {
  bool Invalid = AdoptTemplateParameterList(Params, DC);
  auto *TD = new (C, DC) ConceptDecl(DC, L, Name, Params, ConstraintExpr);
  if (Invalid)
    TD->setInvalidDecl();
  return TD;
}

ConceptDecl *ConceptDecl::CreateDeserialized(ASTContext &C,
                                             unsigned ID) {
  ConceptDecl *Result = new (C, ID) ConceptDecl(nullptr, SourceLocation(),
                                                DeclarationName(),
                                                nullptr, nullptr);

  return Result;
}

//===----------------------------------------------------------------------===//
// ClassTemplatePartialSpecializationDecl Implementation
//===----------------------------------------------------------------------===//
void ClassTemplatePartialSpecializationDecl::anchor() {}

ClassTemplatePartialSpecializationDecl::
ClassTemplatePartialSpecializationDecl(ASTContext &Context, TagKind TK,
                                       DeclContext *DC,
                                       SourceLocation StartLoc,
                                       SourceLocation IdLoc,
                                       TemplateParameterList *Params,
                                       ClassTemplateDecl *SpecializedTemplate,
                                       ArrayRef<TemplateArgument> Args,
                               const ASTTemplateArgumentListInfo *ArgInfos,
                               ClassTemplatePartialSpecializationDecl *PrevDecl)
    : ClassTemplateSpecializationDecl(Context,
                                      ClassTemplatePartialSpecialization,
                                      TK, DC, StartLoc, IdLoc,
                                      SpecializedTemplate, Args, PrevDecl),
      TemplateParams(Params), ArgsAsWritten(ArgInfos),
      InstantiatedFromMember(nullptr, false) {
  if (AdoptTemplateParameterList(Params, this))
    setInvalidDecl();
}

ClassTemplatePartialSpecializationDecl *
ClassTemplatePartialSpecializationDecl::
Create(ASTContext &Context, TagKind TK,DeclContext *DC,
       SourceLocation StartLoc, SourceLocation IdLoc,
       TemplateParameterList *Params,
       ClassTemplateDecl *SpecializedTemplate,
       ArrayRef<TemplateArgument> Args,
       const TemplateArgumentListInfo &ArgInfos,
       QualType CanonInjectedType,
       ClassTemplatePartialSpecializationDecl *PrevDecl) {
  const ASTTemplateArgumentListInfo *ASTArgInfos =
    ASTTemplateArgumentListInfo::Create(Context, ArgInfos);

  auto *Result = new (Context, DC)
      ClassTemplatePartialSpecializationDecl(Context, TK, DC, StartLoc, IdLoc,
                                             Params, SpecializedTemplate, Args,
                                             ASTArgInfos, PrevDecl);
  Result->setSpecializationKind(TSK_ExplicitSpecialization);
  Result->setMayHaveOutOfDateDef(false);

  Context.getInjectedClassNameType(Result, CanonInjectedType);
  return Result;
}

ClassTemplatePartialSpecializationDecl *
ClassTemplatePartialSpecializationDecl::CreateDeserialized(ASTContext &C,
                                                           unsigned ID) {
  auto *Result = new (C, ID) ClassTemplatePartialSpecializationDecl(C);
  Result->setMayHaveOutOfDateDef(false);
  return Result;
}

//===----------------------------------------------------------------------===//
// FriendTemplateDecl Implementation
//===----------------------------------------------------------------------===//

void FriendTemplateDecl::anchor() {}

FriendTemplateDecl *
FriendTemplateDecl::Create(ASTContext &Context, DeclContext *DC,
                           SourceLocation L,
                           MutableArrayRef<TemplateParameterList *> Params,
                           FriendUnion Friend, SourceLocation FLoc) {
  return new (Context, DC) FriendTemplateDecl(DC, L, Params, Friend, FLoc);
}

FriendTemplateDecl *FriendTemplateDecl::CreateDeserialized(ASTContext &C,
                                                           unsigned ID) {
  return new (C, ID) FriendTemplateDecl(EmptyShell());
}

//===----------------------------------------------------------------------===//
// TypeAliasTemplateDecl Implementation
//===----------------------------------------------------------------------===//

TypeAliasTemplateDecl *
TypeAliasTemplateDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                              DeclarationName Name,
                              TemplateParameterList *Params, NamedDecl *Decl) {
  bool Invalid = AdoptTemplateParameterList(Params, DC);
  auto *TD = new (C, DC) TypeAliasTemplateDecl(C, DC, L, Name, Params, Decl);
  if (Invalid)
    TD->setInvalidDecl();
  return TD;
}

TypeAliasTemplateDecl *TypeAliasTemplateDecl::CreateDeserialized(ASTContext &C,
                                                                 unsigned ID) {
  return new (C, ID) TypeAliasTemplateDecl(C, nullptr, SourceLocation(),
                                           DeclarationName(), nullptr, nullptr);
}

RedeclarableTemplateDecl::CommonBase *
TypeAliasTemplateDecl::newCommon(ASTContext &C) const {
  auto *CommonPtr = new (C) Common;
  C.addDestruction(CommonPtr);
  return CommonPtr;
}

//===----------------------------------------------------------------------===//
// ClassScopeFunctionSpecializationDecl Implementation
//===----------------------------------------------------------------------===//

void ClassScopeFunctionSpecializationDecl::anchor() {}

ClassScopeFunctionSpecializationDecl *
ClassScopeFunctionSpecializationDecl::CreateDeserialized(ASTContext &C,
                                                         unsigned ID) {
  return new (C, ID) ClassScopeFunctionSpecializationDecl(
      nullptr, SourceLocation(), nullptr, nullptr);
}

//===----------------------------------------------------------------------===//
// VarTemplateDecl Implementation
//===----------------------------------------------------------------------===//

VarTemplateDecl *VarTemplateDecl::getDefinition() {
  VarTemplateDecl *CurD = this;
  while (CurD) {
    if (CurD->isThisDeclarationADefinition())
      return CurD;
    CurD = CurD->getPreviousDecl();
  }
  return nullptr;
}

VarTemplateDecl *VarTemplateDecl::Create(ASTContext &C, DeclContext *DC,
                                         SourceLocation L, DeclarationName Name,
                                         TemplateParameterList *Params,
                                         VarDecl *Decl) {
  bool Invalid = AdoptTemplateParameterList(Params, DC);
  auto *TD = new (C, DC) VarTemplateDecl(C, DC, L, Name, Params, Decl);
  if (Invalid)
    TD->setInvalidDecl();
  return TD;
}

VarTemplateDecl *VarTemplateDecl::CreateDeserialized(ASTContext &C,
                                                     unsigned ID) {
  return new (C, ID) VarTemplateDecl(C, nullptr, SourceLocation(),
                                     DeclarationName(), nullptr, nullptr);
}

void VarTemplateDecl::LoadLazySpecializations() const {
  loadLazySpecializationsImpl();
}

llvm::FoldingSetVector<VarTemplateSpecializationDecl> &
VarTemplateDecl::getSpecializations() const {
  LoadLazySpecializations();
  return getCommonPtr()->Specializations;
}

llvm::FoldingSetVector<VarTemplatePartialSpecializationDecl> &
VarTemplateDecl::getPartialSpecializations() const {
  LoadLazySpecializations();
  return getCommonPtr()->PartialSpecializations;
}

RedeclarableTemplateDecl::CommonBase *
VarTemplateDecl::newCommon(ASTContext &C) const {
  auto *CommonPtr = new (C) Common;
  C.addDestruction(CommonPtr);
  return CommonPtr;
}

VarTemplateSpecializationDecl *
VarTemplateDecl::findSpecialization(ArrayRef<TemplateArgument> Args,
                                    void *&InsertPos) {
  return findSpecializationImpl(getSpecializations(), InsertPos, Args);
}

void VarTemplateDecl::AddSpecialization(VarTemplateSpecializationDecl *D,
                                        void *InsertPos) {
  addSpecializationImpl<VarTemplateDecl>(getSpecializations(), D, InsertPos);
}

VarTemplatePartialSpecializationDecl *
VarTemplateDecl::findPartialSpecialization(ArrayRef<TemplateArgument> Args,
     TemplateParameterList *TPL, void *&InsertPos) {
  return findSpecializationImpl(getPartialSpecializations(), InsertPos, Args,
                                TPL);
}

void
VarTemplatePartialSpecializationDecl::Profile(llvm::FoldingSetNodeID &ID,
    ArrayRef<TemplateArgument> TemplateArgs, TemplateParameterList *TPL,
    ASTContext &Context) {
  ID.AddInteger(TemplateArgs.size());
  for (const TemplateArgument &TemplateArg : TemplateArgs)
    TemplateArg.Profile(ID, Context);
  ProfileTemplateParameterList(Context, ID, TPL);
}

void VarTemplateDecl::AddPartialSpecialization(
    VarTemplatePartialSpecializationDecl *D, void *InsertPos) {
  if (InsertPos)
    getPartialSpecializations().InsertNode(D, InsertPos);
  else {
    VarTemplatePartialSpecializationDecl *Existing =
        getPartialSpecializations().GetOrInsertNode(D);
    (void)Existing;
    assert(Existing->isCanonicalDecl() && "Non-canonical specialization?");
  }

  if (ASTMutationListener *L = getASTMutationListener())
    L->AddedCXXTemplateSpecialization(this, D);
}

void VarTemplateDecl::getPartialSpecializations(
    SmallVectorImpl<VarTemplatePartialSpecializationDecl *> &PS) const {
  llvm::FoldingSetVector<VarTemplatePartialSpecializationDecl> &PartialSpecs =
      getPartialSpecializations();
  PS.clear();
  PS.reserve(PartialSpecs.size());
  for (VarTemplatePartialSpecializationDecl &P : PartialSpecs)
    PS.push_back(P.getMostRecentDecl());
}

VarTemplatePartialSpecializationDecl *
VarTemplateDecl::findPartialSpecInstantiatedFromMember(
    VarTemplatePartialSpecializationDecl *D) {
  Decl *DCanon = D->getCanonicalDecl();
  for (VarTemplatePartialSpecializationDecl &P : getPartialSpecializations()) {
    if (P.getInstantiatedFromMember()->getCanonicalDecl() == DCanon)
      return P.getMostRecentDecl();
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// VarTemplateSpecializationDecl Implementation
//===----------------------------------------------------------------------===//

VarTemplateSpecializationDecl::VarTemplateSpecializationDecl(
    Kind DK, ASTContext &Context, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, VarTemplateDecl *SpecializedTemplate, QualType T,
    TypeSourceInfo *TInfo, StorageClass S, ArrayRef<TemplateArgument> Args)
    : VarDecl(DK, Context, DC, StartLoc, IdLoc,
              SpecializedTemplate->getIdentifier(), T, TInfo, S),
      SpecializedTemplate(SpecializedTemplate),
      TemplateArgs(TemplateArgumentList::CreateCopy(Context, Args)),
      SpecializationKind(TSK_Undeclared), IsCompleteDefinition(false) {}

VarTemplateSpecializationDecl::VarTemplateSpecializationDecl(Kind DK,
                                                             ASTContext &C)
    : VarDecl(DK, C, nullptr, SourceLocation(), SourceLocation(), nullptr,
              QualType(), nullptr, SC_None),
      SpecializationKind(TSK_Undeclared), IsCompleteDefinition(false) {}

VarTemplateSpecializationDecl *VarTemplateSpecializationDecl::Create(
    ASTContext &Context, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, VarTemplateDecl *SpecializedTemplate, QualType T,
    TypeSourceInfo *TInfo, StorageClass S, ArrayRef<TemplateArgument> Args) {
  return new (Context, DC) VarTemplateSpecializationDecl(
      VarTemplateSpecialization, Context, DC, StartLoc, IdLoc,
      SpecializedTemplate, T, TInfo, S, Args);
}

VarTemplateSpecializationDecl *
VarTemplateSpecializationDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  return new (C, ID)
      VarTemplateSpecializationDecl(VarTemplateSpecialization, C);
}

void VarTemplateSpecializationDecl::getNameForDiagnostic(
    raw_ostream &OS, const PrintingPolicy &Policy, bool Qualified) const {
  NamedDecl::getNameForDiagnostic(OS, Policy, Qualified);

  const auto *PS = dyn_cast<VarTemplatePartialSpecializationDecl>(this);
  if (const ASTTemplateArgumentListInfo *ArgsAsWritten =
          PS ? PS->getTemplateArgsAsWritten() : nullptr) {
    printTemplateArgumentList(
        OS, ArgsAsWritten->arguments(), Policy,
        getSpecializedTemplate()->getTemplateParameters());
  } else {
    const TemplateArgumentList &TemplateArgs = getTemplateArgs();
    printTemplateArgumentList(
        OS, TemplateArgs.asArray(), Policy,
        getSpecializedTemplate()->getTemplateParameters());
  }
}

VarTemplateDecl *VarTemplateSpecializationDecl::getSpecializedTemplate() const {
  if (const auto *PartialSpec =
          SpecializedTemplate.dyn_cast<SpecializedPartialSpecialization *>())
    return PartialSpec->PartialSpecialization->getSpecializedTemplate();
  return SpecializedTemplate.get<VarTemplateDecl *>();
}

void VarTemplateSpecializationDecl::setTemplateArgsInfo(
    const TemplateArgumentListInfo &ArgsInfo) {
  TemplateArgsInfo.setLAngleLoc(ArgsInfo.getLAngleLoc());
  TemplateArgsInfo.setRAngleLoc(ArgsInfo.getRAngleLoc());
  for (const TemplateArgumentLoc &Loc : ArgsInfo.arguments())
    TemplateArgsInfo.addArgument(Loc);
}

//===----------------------------------------------------------------------===//
// VarTemplatePartialSpecializationDecl Implementation
//===----------------------------------------------------------------------===//

void VarTemplatePartialSpecializationDecl::anchor() {}

VarTemplatePartialSpecializationDecl::VarTemplatePartialSpecializationDecl(
    ASTContext &Context, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, TemplateParameterList *Params,
    VarTemplateDecl *SpecializedTemplate, QualType T, TypeSourceInfo *TInfo,
    StorageClass S, ArrayRef<TemplateArgument> Args,
    const ASTTemplateArgumentListInfo *ArgInfos)
    : VarTemplateSpecializationDecl(VarTemplatePartialSpecialization, Context,
                                    DC, StartLoc, IdLoc, SpecializedTemplate, T,
                                    TInfo, S, Args),
      TemplateParams(Params), ArgsAsWritten(ArgInfos),
      InstantiatedFromMember(nullptr, false) {
  if (AdoptTemplateParameterList(Params, DC))
    setInvalidDecl();
}

VarTemplatePartialSpecializationDecl *
VarTemplatePartialSpecializationDecl::Create(
    ASTContext &Context, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, TemplateParameterList *Params,
    VarTemplateDecl *SpecializedTemplate, QualType T, TypeSourceInfo *TInfo,
    StorageClass S, ArrayRef<TemplateArgument> Args,
    const TemplateArgumentListInfo &ArgInfos) {
  const ASTTemplateArgumentListInfo *ASTArgInfos
    = ASTTemplateArgumentListInfo::Create(Context, ArgInfos);

  auto *Result =
      new (Context, DC) VarTemplatePartialSpecializationDecl(
          Context, DC, StartLoc, IdLoc, Params, SpecializedTemplate, T, TInfo,
          S, Args, ASTArgInfos);
  Result->setSpecializationKind(TSK_ExplicitSpecialization);
  return Result;
}

VarTemplatePartialSpecializationDecl *
VarTemplatePartialSpecializationDecl::CreateDeserialized(ASTContext &C,
                                                         unsigned ID) {
  return new (C, ID) VarTemplatePartialSpecializationDecl(C);
}

static TemplateParameterList *
createMakeIntegerSeqParameterList(const ASTContext &C, DeclContext *DC) {
  // typename T
  auto *T = TemplateTypeParmDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), /*Depth=*/1, /*Position=*/0,
      /*Id=*/nullptr, /*Typename=*/true, /*ParameterPack=*/false,
      /*HasTypeConstraint=*/false);
  T->setImplicit(true);

  // T ...Ints
  TypeSourceInfo *TI =
      C.getTrivialTypeSourceInfo(QualType(T->getTypeForDecl(), 0));
  auto *N = NonTypeTemplateParmDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), /*Depth=*/0, /*Position=*/1,
      /*Id=*/nullptr, TI->getType(), /*ParameterPack=*/true, TI);
  N->setImplicit(true);

  // <typename T, T ...Ints>
  NamedDecl *P[2] = {T, N};
  auto *TPL = TemplateParameterList::Create(
      C, SourceLocation(), SourceLocation(), P, SourceLocation(), nullptr);

  // template <typename T, ...Ints> class IntSeq
  auto *TemplateTemplateParm = TemplateTemplateParmDecl::Create(
      C, DC, SourceLocation(), /*Depth=*/0, /*Position=*/0,
      /*ParameterPack=*/false, /*Id=*/nullptr, TPL);
  TemplateTemplateParm->setImplicit(true);

  // typename T
  auto *TemplateTypeParm = TemplateTypeParmDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), /*Depth=*/0, /*Position=*/1,
      /*Id=*/nullptr, /*Typename=*/true, /*ParameterPack=*/false,
      /*HasTypeConstraint=*/false);
  TemplateTypeParm->setImplicit(true);

  // T N
  TypeSourceInfo *TInfo = C.getTrivialTypeSourceInfo(
      QualType(TemplateTypeParm->getTypeForDecl(), 0));
  auto *NonTypeTemplateParm = NonTypeTemplateParmDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), /*Depth=*/0, /*Position=*/2,
      /*Id=*/nullptr, TInfo->getType(), /*ParameterPack=*/false, TInfo);
  NamedDecl *Params[] = {TemplateTemplateParm, TemplateTypeParm,
                         NonTypeTemplateParm};

  // template <template <typename T, T ...Ints> class IntSeq, typename T, T N>
  return TemplateParameterList::Create(C, SourceLocation(), SourceLocation(),
                                       Params, SourceLocation(), nullptr);
}

static TemplateParameterList *
createTypePackElementParameterList(const ASTContext &C, DeclContext *DC) {
  // std::size_t Index
  TypeSourceInfo *TInfo = C.getTrivialTypeSourceInfo(C.getSizeType());
  auto *Index = NonTypeTemplateParmDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), /*Depth=*/0, /*Position=*/0,
      /*Id=*/nullptr, TInfo->getType(), /*ParameterPack=*/false, TInfo);

  // typename ...T
  auto *Ts = TemplateTypeParmDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), /*Depth=*/0, /*Position=*/1,
      /*Id=*/nullptr, /*Typename=*/true, /*ParameterPack=*/true,
      /*HasTypeConstraint=*/false);
  Ts->setImplicit(true);

  // template <std::size_t Index, typename ...T>
  NamedDecl *Params[] = {Index, Ts};
  return TemplateParameterList::Create(C, SourceLocation(), SourceLocation(),
                                       llvm::makeArrayRef(Params),
                                       SourceLocation(), nullptr);
}

static TemplateParameterList *createBuiltinTemplateParameterList(
    const ASTContext &C, DeclContext *DC, BuiltinTemplateKind BTK) {
  switch (BTK) {
  case BTK__make_integer_seq:
    return createMakeIntegerSeqParameterList(C, DC);
  case BTK__type_pack_element:
    return createTypePackElementParameterList(C, DC);
  }

  llvm_unreachable("unhandled BuiltinTemplateKind!");
}

void BuiltinTemplateDecl::anchor() {}

BuiltinTemplateDecl::BuiltinTemplateDecl(const ASTContext &C, DeclContext *DC,
                                         DeclarationName Name,
                                         BuiltinTemplateKind BTK)
    : TemplateDecl(BuiltinTemplate, DC, SourceLocation(), Name,
                   createBuiltinTemplateParameterList(C, DC, BTK)),
      BTK(BTK) {}

void TypeConstraint::print(llvm::raw_ostream &OS, PrintingPolicy Policy) const {
  if (NestedNameSpec)
    NestedNameSpec.getNestedNameSpecifier()->print(OS, Policy);
  ConceptName.printName(OS, Policy);
  if (hasExplicitTemplateArgs()) {
    OS << "<";
    // FIXME: Find corresponding parameter for argument
    for (auto &ArgLoc : ArgsAsWritten->arguments())
      ArgLoc.getArgument().print(Policy, OS, /*IncludeType*/ false);
    OS << ">";
  }
}

TemplateParamObjectDecl *TemplateParamObjectDecl::Create(const ASTContext &C,
                                                         QualType T,
                                                         const APValue &V) {
  DeclContext *DC = C.getTranslationUnitDecl();
  auto *TPOD = new (C, DC) TemplateParamObjectDecl(DC, T, V);
  C.addDestruction(&TPOD->Value);
  return TPOD;
}

TemplateParamObjectDecl *
TemplateParamObjectDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  auto *TPOD = new (C, ID) TemplateParamObjectDecl(nullptr, QualType(), APValue());
  C.addDestruction(&TPOD->Value);
  return TPOD;
}

void TemplateParamObjectDecl::printName(llvm::raw_ostream &OS) const {
  OS << "<template param ";
  printAsExpr(OS);
  OS << ">";
}

void TemplateParamObjectDecl::printAsExpr(llvm::raw_ostream &OS) const {
  const ASTContext &Ctx = getASTContext();
  getType().getUnqualifiedType().print(OS, Ctx.getPrintingPolicy());
  printAsInit(OS);
}

void TemplateParamObjectDecl::printAsInit(llvm::raw_ostream &OS) const {
  const ASTContext &Ctx = getASTContext();
  getValue().printPretty(OS, Ctx, getType());
}
