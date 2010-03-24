//===---- SemaAccess.cpp - C++ Access Control -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Sema routines for C++ access control semantics.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "Lookup.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DependentDiagnostic.h"
#include "clang/AST/ExprCXX.h"

using namespace clang;

/// SetMemberAccessSpecifier - Set the access specifier of a member.
/// Returns true on error (when the previous member decl access specifier
/// is different from the new member decl access specifier).
bool Sema::SetMemberAccessSpecifier(NamedDecl *MemberDecl,
                                    NamedDecl *PrevMemberDecl,
                                    AccessSpecifier LexicalAS) {
  if (!PrevMemberDecl) {
    // Use the lexical access specifier.
    MemberDecl->setAccess(LexicalAS);
    return false;
  }

  // C++ [class.access.spec]p3: When a member is redeclared its access
  // specifier must be same as its initial declaration.
  if (LexicalAS != AS_none && LexicalAS != PrevMemberDecl->getAccess()) {
    Diag(MemberDecl->getLocation(),
         diag::err_class_redeclared_with_different_access)
      << MemberDecl << LexicalAS;
    Diag(PrevMemberDecl->getLocation(), diag::note_previous_access_declaration)
      << PrevMemberDecl << PrevMemberDecl->getAccess();

    MemberDecl->setAccess(LexicalAS);
    return true;
  }

  MemberDecl->setAccess(PrevMemberDecl->getAccess());
  return false;
}

namespace {
struct EffectiveContext {
  EffectiveContext() : Function(0), Dependent(false) {}

  explicit EffectiveContext(DeclContext *DC) {
    Dependent = DC->isDependentContext();

    if (isa<FunctionDecl>(DC)) {
      Function = cast<FunctionDecl>(DC)->getCanonicalDecl();
      DC = Function->getDeclContext();
    } else
      Function = 0;

    // C++ [class.access.nest]p1:
    //   A nested class is a member and as such has the same access
    //   rights as any other member.
    // C++ [class.access]p2:
    //   A member of a class can also access all the names to which
    //   the class has access.
    // This implies that the privileges of nesting are transitive.
    while (isa<CXXRecordDecl>(DC)) {
      CXXRecordDecl *Record = cast<CXXRecordDecl>(DC)->getCanonicalDecl();
      Records.push_back(Record);
      DC = Record->getDeclContext();
    }
  }

  bool isDependent() const { return Dependent; }

  bool includesClass(const CXXRecordDecl *R) const {
    R = R->getCanonicalDecl();
    return std::find(Records.begin(), Records.end(), R)
             != Records.end();
  }

  DeclContext *getPrimaryContext() const {
    assert((Function || !Records.empty()) && "context has no primary context");
    if (Function) return Function;
    return Records[0];
  }

  typedef llvm::SmallVectorImpl<CXXRecordDecl*>::const_iterator record_iterator;

  llvm::SmallVector<CXXRecordDecl*, 4> Records;
  FunctionDecl *Function;
  bool Dependent;
};
}

static CXXRecordDecl *FindDeclaringClass(NamedDecl *D) {
  CXXRecordDecl *DeclaringClass = cast<CXXRecordDecl>(D->getDeclContext());
  while (DeclaringClass->isAnonymousStructOrUnion())
    DeclaringClass = cast<CXXRecordDecl>(DeclaringClass->getDeclContext());
  return DeclaringClass;
}

static bool MightInstantiateTo(Sema &S, DeclContext *Context,
                               DeclContext *Friend) {
  if (Friend == Context)
    return true;

  assert(!Friend->isDependentContext() &&
         "can't handle friends with dependent contexts here");

  if (!Context->isDependentContext())
    return false;

  if (Friend->isFileContext())
    return false;

  // TODO: this is very conservative
  return true;
}

// Asks whether the type in 'context' can ever instantiate to the type
// in 'friend'.
static bool MightInstantiateTo(Sema &S, CanQualType Context, CanQualType Friend) {
  if (Friend == Context)
    return true;

  if (!Friend->isDependentType() && !Context->isDependentType())
    return false;

  // TODO: this is very conservative.
  return true;
}

static bool MightInstantiateTo(Sema &S,
                               FunctionDecl *Context,
                               FunctionDecl *Friend) {
  if (Context->getDeclName() != Friend->getDeclName())
    return false;

  if (!MightInstantiateTo(S,
                          Context->getDeclContext(),
                          Friend->getDeclContext()))
    return false;

  CanQual<FunctionProtoType> FriendTy
    = S.Context.getCanonicalType(Friend->getType())
         ->getAs<FunctionProtoType>();
  CanQual<FunctionProtoType> ContextTy
    = S.Context.getCanonicalType(Context->getType())
         ->getAs<FunctionProtoType>();

  // There isn't any way that I know of to add qualifiers
  // during instantiation.
  if (FriendTy.getQualifiers() != ContextTy.getQualifiers())
    return false;

  if (FriendTy->getNumArgs() != ContextTy->getNumArgs())
    return false;

  if (!MightInstantiateTo(S,
                          ContextTy->getResultType(),
                          FriendTy->getResultType()))
    return false;

  for (unsigned I = 0, E = FriendTy->getNumArgs(); I != E; ++I)
    if (!MightInstantiateTo(S,
                            ContextTy->getArgType(I),
                            FriendTy->getArgType(I)))
      return false;

  return true;
}

static bool MightInstantiateTo(Sema &S,
                               FunctionTemplateDecl *Context,
                               FunctionTemplateDecl *Friend) {
  return MightInstantiateTo(S,
                            Context->getTemplatedDecl(),
                            Friend->getTemplatedDecl());
}

static Sema::AccessResult MatchesFriend(Sema &S,
                                        const EffectiveContext &EC,
                                        const CXXRecordDecl *Friend) {
  if (EC.includesClass(Friend))
    return Sema::AR_accessible;

  if (EC.isDependent()) {
    CanQualType FriendTy
      = S.Context.getCanonicalType(S.Context.getTypeDeclType(Friend));

    for (EffectiveContext::record_iterator
           I = EC.Records.begin(), E = EC.Records.end(); I != E; ++I) {
      CanQualType ContextTy
        = S.Context.getCanonicalType(S.Context.getTypeDeclType(*I));
      if (MightInstantiateTo(S, ContextTy, FriendTy))
        return Sema::AR_dependent;
    }
  }

  return Sema::AR_inaccessible;
}

static Sema::AccessResult MatchesFriend(Sema &S,
                                        const EffectiveContext &EC,
                                        CanQualType Friend) {
  if (const RecordType *RT = Friend->getAs<RecordType>())
    return MatchesFriend(S, EC, cast<CXXRecordDecl>(RT->getDecl()));

  // TODO: we can do better than this
  if (Friend->isDependentType())
    return Sema::AR_dependent;

  return Sema::AR_inaccessible;
}

/// Determines whether the given friend class template matches
/// anything in the effective context.
static Sema::AccessResult MatchesFriend(Sema &S,
                                        const EffectiveContext &EC,
                                        ClassTemplateDecl *Friend) {
  Sema::AccessResult OnFailure = Sema::AR_inaccessible;

  for (llvm::SmallVectorImpl<CXXRecordDecl*>::const_iterator
         I = EC.Records.begin(), E = EC.Records.end(); I != E; ++I) {
    CXXRecordDecl *Record = *I;

    // Check whether the friend is the template of a class in the
    // context chain.  To do that, we need to figure out whether the
    // current class has a template:
    ClassTemplateDecl *CTD;

    // A specialization of the template...
    if (isa<ClassTemplateSpecializationDecl>(Record)) {
      CTD = cast<ClassTemplateSpecializationDecl>(Record)
        ->getSpecializedTemplate();

    // ... or the template pattern itself.
    } else {
      CTD = Record->getDescribedClassTemplate();
      if (!CTD) continue;
    }

    // It's a match.
    if (Friend == CTD->getCanonicalDecl())
      return Sema::AR_accessible;

    // If the template names don't match, it can't be a dependent
    // match.  This isn't true in C++0x because of template aliases.
    if (!S.LangOpts.CPlusPlus0x && CTD->getDeclName() != Friend->getDeclName())
      continue;

    // If the class's context can't instantiate to the friend's
    // context, it can't be a dependent match.
    if (!MightInstantiateTo(S, CTD->getDeclContext(),
                            Friend->getDeclContext()))
      continue;

    // Otherwise, it's a dependent match.
    OnFailure = Sema::AR_dependent;
  }

  return OnFailure;
}

/// Determines whether the given friend function matches anything in
/// the effective context.
static Sema::AccessResult MatchesFriend(Sema &S,
                                        const EffectiveContext &EC,
                                        FunctionDecl *Friend) {
  if (!EC.Function)
    return Sema::AR_inaccessible;

  if (Friend == EC.Function)
    return Sema::AR_accessible;

  if (EC.isDependent() && MightInstantiateTo(S, EC.Function, Friend))
    return Sema::AR_dependent;

  return Sema::AR_inaccessible;
}

/// Determines whether the given friend function template matches
/// anything in the effective context.
static Sema::AccessResult MatchesFriend(Sema &S,
                                        const EffectiveContext &EC,
                                        FunctionTemplateDecl *Friend) {
  if (!EC.Function) return Sema::AR_inaccessible;

  FunctionTemplateDecl *FTD = EC.Function->getPrimaryTemplate();
  if (!FTD)
    FTD = EC.Function->getDescribedFunctionTemplate();
  if (!FTD)
    return Sema::AR_inaccessible;

  if (Friend == FTD->getCanonicalDecl())
    return Sema::AR_accessible;

  if (MightInstantiateTo(S, FTD, Friend))
    return Sema::AR_dependent;

  return Sema::AR_inaccessible;
}

/// Determines whether the given friend declaration matches anything
/// in the effective context.
static Sema::AccessResult MatchesFriend(Sema &S,
                                        const EffectiveContext &EC,
                                        FriendDecl *FriendD) {
  if (Type *T = FriendD->getFriendType())
    return MatchesFriend(S, EC, T->getCanonicalTypeUnqualified());

  NamedDecl *Friend
    = cast<NamedDecl>(FriendD->getFriendDecl()->getCanonicalDecl());

  // FIXME: declarations with dependent or templated scope.

  if (isa<ClassTemplateDecl>(Friend))
    return MatchesFriend(S, EC, cast<ClassTemplateDecl>(Friend));

  if (isa<FunctionTemplateDecl>(Friend))
    return MatchesFriend(S, EC, cast<FunctionTemplateDecl>(Friend));

  if (isa<CXXRecordDecl>(Friend))
    return MatchesFriend(S, EC, cast<CXXRecordDecl>(Friend));

  assert(isa<FunctionDecl>(Friend) && "unknown friend decl kind");
  return MatchesFriend(S, EC, cast<FunctionDecl>(Friend));
}

static Sema::AccessResult GetFriendKind(Sema &S,
                                        const EffectiveContext &EC,
                                        const CXXRecordDecl *Class) {
  // A class always has access to its own members.
  if (EC.includesClass(Class))
    return Sema::AR_accessible;

  Sema::AccessResult OnFailure = Sema::AR_inaccessible;

  // Okay, check friends.
  for (CXXRecordDecl::friend_iterator I = Class->friend_begin(),
         E = Class->friend_end(); I != E; ++I) {
    FriendDecl *Friend = *I;

    switch (MatchesFriend(S, EC, Friend)) {
    case Sema::AR_accessible:
      return Sema::AR_accessible;

    case Sema::AR_inaccessible:
      break;

    case Sema::AR_dependent:
      OnFailure = Sema::AR_dependent;
      break;

    case Sema::AR_delayed:
      llvm_unreachable("cannot get delayed answer from MatchesFriend");
    }
  }

  // That's it, give up.
  return OnFailure;
}

/// Finds the best path from the naming class to the declaring class,
/// taking friend declarations into account.
///
/// \param FinalAccess the access of the "final step", or AS_none if
///   there is no final step.
/// \return null if friendship is dependent
static CXXBasePath *FindBestPath(Sema &S,
                                 const EffectiveContext &EC,
                                 CXXRecordDecl *Derived,
                                 CXXRecordDecl *Base,
                                 AccessSpecifier FinalAccess,
                                 CXXBasePaths &Paths) {
  // Derive the paths to the desired base.
  bool isDerived = Derived->isDerivedFrom(Base, Paths);
  assert(isDerived && "derived class not actually derived from base");
  (void) isDerived;

  CXXBasePath *BestPath = 0;

  assert(FinalAccess != AS_none && "forbidden access after declaring class");

  bool AnyDependent = false;

  // Derive the friend-modified access along each path.
  for (CXXBasePaths::paths_iterator PI = Paths.begin(), PE = Paths.end();
         PI != PE; ++PI) {

    // Walk through the path backwards.
    AccessSpecifier PathAccess = FinalAccess;
    CXXBasePath::iterator I = PI->end(), E = PI->begin();
    while (I != E) {
      --I;

      assert(PathAccess != AS_none);

      // If the declaration is a private member of a base class, there
      // is no level of friendship in derived classes that can make it
      // accessible.
      if (PathAccess == AS_private) {
        PathAccess = AS_none;
        break;
      }

      AccessSpecifier BaseAccess = I->Base->getAccessSpecifier();
      if (BaseAccess != AS_public) {
        switch (GetFriendKind(S, EC, I->Class)) {
        case Sema::AR_inaccessible:
          PathAccess = CXXRecordDecl::MergeAccess(BaseAccess, PathAccess);
          break;
        case Sema::AR_accessible:
          PathAccess = AS_public;
          break;
        case Sema::AR_dependent:
          AnyDependent = true;
          goto Next;
        case Sema::AR_delayed:
          llvm_unreachable("friend resolution is never delayed"); break;
        }
      }
    }

    // Note that we modify the path's Access field to the
    // friend-modified access.
    if (BestPath == 0 || PathAccess < BestPath->Access) {
      BestPath = &*PI;
      BestPath->Access = PathAccess;

      // Short-circuit if we found a public path.
      if (BestPath->Access == AS_public)
        return BestPath;
    }

  Next: ;
  }

  assert((!BestPath || BestPath->Access != AS_public) &&
         "fell out of loop with public path");

  // We didn't find a public path, but at least one path was subject
  // to dependent friendship, so delay the check.
  if (AnyDependent)
    return 0;

  return BestPath;
}

/// Diagnose the path which caused the given declaration or base class
/// to become inaccessible.
static void DiagnoseAccessPath(Sema &S,
                               const EffectiveContext &EC,
                               CXXRecordDecl *NamingClass,
                               CXXRecordDecl *DeclaringClass,
                               NamedDecl *D, AccessSpecifier Access) {
  // Easy case: the decl's natural access determined its path access.
  // We have to check against AS_private here in case Access is AS_none,
  // indicating a non-public member of a private base class.
  //
  // DependentFriend should be impossible here.
  if (D && (Access == D->getAccess() || D->getAccess() == AS_private)) {
    switch (GetFriendKind(S, EC, DeclaringClass)) {
    case Sema::AR_inaccessible: {
      S.Diag(D->getLocation(), diag::note_access_natural)
        << (unsigned) (Access == AS_protected)
        << /*FIXME: not implicitly*/ 0;
      return;
    }

    case Sema::AR_accessible: break;

    case Sema::AR_dependent:
    case Sema::AR_delayed:
      llvm_unreachable("dependent/delayed not allowed");
      return;
    }
  }

  CXXBasePaths Paths;
  CXXBasePath &Path = *FindBestPath(S, EC, NamingClass, DeclaringClass,
                                    AS_public, Paths);

  CXXBasePath::iterator I = Path.end(), E = Path.begin();
  while (I != E) {
    --I;

    const CXXBaseSpecifier *BS = I->Base;
    AccessSpecifier BaseAccess = BS->getAccessSpecifier();

    // If this is public inheritance, or the derived class is a friend,
    // skip this step.
    if (BaseAccess == AS_public)
      continue;

    switch (GetFriendKind(S, EC, I->Class)) {
    case Sema::AR_accessible: continue;
    case Sema::AR_inaccessible: break;

    case Sema::AR_dependent:
    case Sema::AR_delayed:
      llvm_unreachable("dependent friendship, should not be diagnosing");
    }

    // Check whether this base specifier is the tighest point
    // constraining access.  We have to check against AS_private for
    // the same reasons as above.
    if (BaseAccess == AS_private || BaseAccess >= Access) {

      // We're constrained by inheritance, but we want to say
      // "declared private here" if we're diagnosing a hierarchy
      // conversion and this is the final step.
      unsigned diagnostic;
      if (D) diagnostic = diag::note_access_constrained_by_path;
      else if (I + 1 == Path.end()) diagnostic = diag::note_access_natural;
      else diagnostic = diag::note_access_constrained_by_path;

      S.Diag(BS->getSourceRange().getBegin(), diagnostic)
        << BS->getSourceRange()
        << (BaseAccess == AS_protected)
        << (BS->getAccessSpecifierAsWritten() == AS_none);
      return;
    }
  }

  llvm_unreachable("access not apparently constrained by path");
}

/// Diagnose an inaccessible class member.
static void DiagnoseInaccessibleMember(Sema &S, SourceLocation Loc,
                                       const EffectiveContext &EC,
                                       CXXRecordDecl *NamingClass,
                                       AccessSpecifier Access,
                                       const Sema::AccessedEntity &Entity) {
  NamedDecl *D = Entity.getTargetDecl();
  CXXRecordDecl *DeclaringClass = FindDeclaringClass(D);

  S.Diag(Loc, Entity.getDiag())
    << (Access == AS_protected)
    << D->getDeclName()
    << S.Context.getTypeDeclType(NamingClass)
    << S.Context.getTypeDeclType(DeclaringClass);
  DiagnoseAccessPath(S, EC, NamingClass, DeclaringClass, D, Access);
}

/// Diagnose an inaccessible hierarchy conversion.
static void DiagnoseInaccessibleBase(Sema &S, SourceLocation Loc,
                                     const EffectiveContext &EC,
                                     AccessSpecifier Access,
                                     const Sema::AccessedEntity &Entity) {
  S.Diag(Loc, Entity.getDiag())
    << (Access == AS_protected)
    << DeclarationName()
    << S.Context.getTypeDeclType(Entity.getDerivedClass())
    << S.Context.getTypeDeclType(Entity.getBaseClass());
  DiagnoseAccessPath(S, EC, Entity.getDerivedClass(),
                     Entity.getBaseClass(), 0, Access);
}

static void DiagnoseBadAccess(Sema &S, SourceLocation Loc,
                              const EffectiveContext &EC,
                              CXXRecordDecl *NamingClass,
                              AccessSpecifier Access,
                              const Sema::AccessedEntity &Entity) {
  if (Entity.isMemberAccess())
    DiagnoseInaccessibleMember(S, Loc, EC, NamingClass, Access, Entity);
  else
    DiagnoseInaccessibleBase(S, Loc, EC, Access, Entity);
}


/// Try to elevate access using friend declarations.  This is
/// potentially quite expensive.
///
/// \return true if elevation was dependent
static bool TryElevateAccess(Sema &S,
                             const EffectiveContext &EC,
                             const Sema::AccessedEntity &Entity,
                             AccessSpecifier &Access) {
  CXXRecordDecl *DeclaringClass;
  if (Entity.isMemberAccess()) {
    DeclaringClass = FindDeclaringClass(Entity.getTargetDecl());
  } else {
    DeclaringClass = Entity.getBaseClass();
  }
  CXXRecordDecl *NamingClass = Entity.getNamingClass();

  // Adjust the declaration of the referred entity.
  AccessSpecifier DeclAccess = AS_public;
  if (Entity.isMemberAccess()) {
    NamedDecl *Target = Entity.getTargetDecl();

    DeclAccess = Target->getAccess();
    if (DeclAccess != AS_public) {
      switch (GetFriendKind(S, EC, DeclaringClass)) {
      case Sema::AR_accessible: DeclAccess = AS_public; break;
      case Sema::AR_inaccessible: break;
      case Sema::AR_dependent: return true;
      case Sema::AR_delayed: llvm_unreachable("friend status is never delayed");
      }
    }

    if (DeclaringClass == NamingClass) {
      Access = DeclAccess;
      return false;
    }
  }

  assert(DeclaringClass != NamingClass);

  // Append the declaration's access if applicable.
  CXXBasePaths Paths;
  CXXBasePath *Path = FindBestPath(S, EC, Entity.getNamingClass(),
                                   DeclaringClass, DeclAccess, Paths);
  if (!Path)
    return true;

  // Grab the access along the best path (note that this includes the
  // final-step access).
  AccessSpecifier NewAccess = Path->Access;
  assert(NewAccess <= Access && "access along best path worse than direct?");
  Access = NewAccess;
  return false;
}

static void DelayAccess(Sema &S,
                        const EffectiveContext &EC,
                        SourceLocation Loc,
                        const Sema::AccessedEntity &Entity) {
  assert(EC.isDependent() && "delaying non-dependent access");
  DeclContext *DC = EC.getPrimaryContext();
  assert(DC->isDependentContext() && "delaying non-dependent access");
  DependentDiagnostic::Create(S.Context, DC, DependentDiagnostic::Access,
                              Loc,
                              Entity.isMemberAccess(),
                              Entity.getAccess(),
                              Entity.getTargetDecl(),
                              Entity.getNamingClass(),
                              Entity.getDiag());
}

/// Checks access to an entity from the given effective context.
static Sema::AccessResult CheckEffectiveAccess(Sema &S,
                                               const EffectiveContext &EC,
                                               SourceLocation Loc,
                                         Sema::AccessedEntity const &Entity) {
  AccessSpecifier Access = Entity.getAccess();
  assert(Access != AS_public && "called for public access!");

  // Find a non-anonymous naming class.  For records with access,
  // there should always be one of these.
  CXXRecordDecl *NamingClass = Entity.getNamingClass();
  while (NamingClass->isAnonymousStructOrUnion())
    NamingClass = cast<CXXRecordDecl>(NamingClass->getParent());

  // White-list accesses from classes with privileges equivalent to the
  // naming class --- but only if the access path isn't forbidden
  // (i.e. an access of a private member from a subclass).
  if (Access != AS_none && EC.includesClass(NamingClass))
    return Sema::AR_accessible;

  // Try to elevate access.
  // TODO: on some code, it might be better to do the protected check
  // without trying to elevate first.
  if (TryElevateAccess(S, EC, Entity, Access)) {
    DelayAccess(S, EC, Loc, Entity);
    return Sema::AR_dependent;
  }

  if (Access == AS_public) return Sema::AR_accessible;

  // Protected access.
  if (Access == AS_protected) {
    // FIXME: implement [class.protected]p1
    for (llvm::SmallVectorImpl<CXXRecordDecl*>::const_iterator
           I = EC.Records.begin(), E = EC.Records.end(); I != E; ++I)
      if ((*I)->isDerivedFrom(NamingClass))
        return Sema::AR_accessible;

    // FIXME: delay if we can't decide class derivation yet.
  }

  // Okay, that's it, reject it.
  if (!Entity.isQuiet())
    DiagnoseBadAccess(S, Loc, EC, NamingClass, Access, Entity);
  return Sema::AR_inaccessible;
}

static Sema::AccessResult CheckAccess(Sema &S, SourceLocation Loc,
                                      const Sema::AccessedEntity &Entity) {
  // If the access path is public, it's accessible everywhere.
  if (Entity.getAccess() == AS_public)
    return Sema::AR_accessible;

  // If we're currently parsing a top-level declaration, delay
  // diagnostics.  This is the only case where parsing a declaration
  // can actually change our effective context for the purposes of
  // access control.
  if (S.CurContext->isFileContext() && S.ParsingDeclDepth) {
    S.DelayedDiagnostics.push_back(
        Sema::DelayedDiagnostic::makeAccess(Loc, Entity));
    return Sema::AR_delayed;
  }

  return CheckEffectiveAccess(S, EffectiveContext(S.CurContext),
                              Loc, Entity);
}

void Sema::HandleDelayedAccessCheck(DelayedDiagnostic &DD, Decl *Ctx) {
  // Pretend we did this from the context of the newly-parsed
  // declaration.
  EffectiveContext EC(Ctx->getDeclContext());

  if (CheckEffectiveAccess(*this, EC, DD.Loc, DD.getAccessData()))
    DD.Triggered = true;
}

void Sema::HandleDependentAccessCheck(const DependentDiagnostic &DD,
                        const MultiLevelTemplateArgumentList &TemplateArgs) {
  SourceLocation Loc = DD.getAccessLoc();
  AccessSpecifier Access = DD.getAccess();

  Decl *NamingD = FindInstantiatedDecl(Loc, DD.getAccessNamingClass(),
                                       TemplateArgs);
  if (!NamingD) return;
  Decl *TargetD = FindInstantiatedDecl(Loc, DD.getAccessTarget(),
                                       TemplateArgs);
  if (!TargetD) return;

  if (DD.isAccessToMember()) {
    AccessedEntity Entity(AccessedEntity::Member,
                          cast<CXXRecordDecl>(NamingD),
                          Access,
                          cast<NamedDecl>(TargetD));
    Entity.setDiag(DD.getDiagnostic());
    CheckAccess(*this, Loc, Entity);
  } else {
    AccessedEntity Entity(AccessedEntity::Base,
                          cast<CXXRecordDecl>(TargetD),
                          cast<CXXRecordDecl>(NamingD),
                          Access);
    Entity.setDiag(DD.getDiagnostic());
    CheckAccess(*this, Loc, Entity);
  }
}

Sema::AccessResult Sema::CheckUnresolvedLookupAccess(UnresolvedLookupExpr *E,
                                                     DeclAccessPair Found) {
  if (!getLangOptions().AccessControl ||
      !E->getNamingClass() ||
      Found.getAccess() == AS_public)
    return AR_accessible;

  AccessedEntity Entity(AccessedEntity::Member, E->getNamingClass(), Found);
  Entity.setDiag(diag::err_access) << E->getSourceRange();

  return CheckAccess(*this, E->getNameLoc(), Entity);
}

/// Perform access-control checking on a previously-unresolved member
/// access which has now been resolved to a member.
Sema::AccessResult Sema::CheckUnresolvedMemberAccess(UnresolvedMemberExpr *E,
                                                     DeclAccessPair Found) {
  if (!getLangOptions().AccessControl ||
      Found.getAccess() == AS_public)
    return AR_accessible;

  AccessedEntity Entity(AccessedEntity::Member, E->getNamingClass(), Found);
  Entity.setDiag(diag::err_access) << E->getSourceRange();

  return CheckAccess(*this, E->getMemberLoc(), Entity);
}

Sema::AccessResult Sema::CheckDestructorAccess(SourceLocation Loc,
                                               CXXDestructorDecl *Dtor,
                                               const PartialDiagnostic &PDiag) {
  if (!getLangOptions().AccessControl)
    return AR_accessible;

  // There's never a path involved when checking implicit destructor access.
  AccessSpecifier Access = Dtor->getAccess();
  if (Access == AS_public)
    return AR_accessible;

  CXXRecordDecl *NamingClass = Dtor->getParent();
  AccessedEntity Entity(AccessedEntity::Member, NamingClass,
                        DeclAccessPair::make(Dtor, Access));
  Entity.setDiag(PDiag); // TODO: avoid copy

  return CheckAccess(*this, Loc, Entity);
}

/// Checks access to a constructor.
Sema::AccessResult Sema::CheckConstructorAccess(SourceLocation UseLoc,
                                  CXXConstructorDecl *Constructor,
                                  AccessSpecifier Access) {
  if (!getLangOptions().AccessControl ||
      Access == AS_public)
    return AR_accessible;

  CXXRecordDecl *NamingClass = Constructor->getParent();
  AccessedEntity Entity(AccessedEntity::Member, NamingClass,
                        DeclAccessPair::make(Constructor, Access));
  Entity.setDiag(diag::err_access_ctor);

  return CheckAccess(*this, UseLoc, Entity);
}

/// Checks direct (i.e. non-inherited) access to an arbitrary class
/// member.
Sema::AccessResult Sema::CheckDirectMemberAccess(SourceLocation UseLoc,
                                                 NamedDecl *Target,
                                           const PartialDiagnostic &Diag) {
  AccessSpecifier Access = Target->getAccess();
  if (!getLangOptions().AccessControl ||
      Access == AS_public)
    return AR_accessible;

  CXXRecordDecl *NamingClass = cast<CXXRecordDecl>(Target->getDeclContext());
  AccessedEntity Entity(AccessedEntity::Member, NamingClass,
                        DeclAccessPair::make(Target, Access));
  Entity.setDiag(Diag);
  return CheckAccess(*this, UseLoc, Entity);
}
                                           

/// Checks access to an overloaded operator new or delete.
Sema::AccessResult Sema::CheckAllocationAccess(SourceLocation OpLoc,
                                               SourceRange PlacementRange,
                                               CXXRecordDecl *NamingClass,
                                               DeclAccessPair Found) {
  if (!getLangOptions().AccessControl ||
      !NamingClass ||
      Found.getAccess() == AS_public)
    return AR_accessible;

  AccessedEntity Entity(AccessedEntity::Member, NamingClass, Found);
  Entity.setDiag(diag::err_access)
    << PlacementRange;

  return CheckAccess(*this, OpLoc, Entity);
}

/// Checks access to an overloaded member operator, including
/// conversion operators.
Sema::AccessResult Sema::CheckMemberOperatorAccess(SourceLocation OpLoc,
                                                   Expr *ObjectExpr,
                                                   Expr *ArgExpr,
                                                   DeclAccessPair Found) {
  if (!getLangOptions().AccessControl ||
      Found.getAccess() == AS_public)
    return AR_accessible;

  const RecordType *RT = ObjectExpr->getType()->getAs<RecordType>();
  assert(RT && "found member operator but object expr not of record type");
  CXXRecordDecl *NamingClass = cast<CXXRecordDecl>(RT->getDecl());

  AccessedEntity Entity(AccessedEntity::Member, NamingClass, Found);
  Entity.setDiag(diag::err_access)
    << ObjectExpr->getSourceRange()
    << (ArgExpr ? ArgExpr->getSourceRange() : SourceRange());

  return CheckAccess(*this, OpLoc, Entity);
}

/// Checks access for a hierarchy conversion.
///
/// \param IsBaseToDerived whether this is a base-to-derived conversion (true)
///     or a derived-to-base conversion (false)
/// \param ForceCheck true if this check should be performed even if access
///     control is disabled;  some things rely on this for semantics
/// \param ForceUnprivileged true if this check should proceed as if the
///     context had no special privileges
/// \param ADK controls the kind of diagnostics that are used
Sema::AccessResult Sema::CheckBaseClassAccess(SourceLocation AccessLoc,
                                              QualType Base,
                                              QualType Derived,
                                              const CXXBasePath &Path,
                                              unsigned DiagID,
                                              bool ForceCheck,
                                              bool ForceUnprivileged) {
  if (!ForceCheck && !getLangOptions().AccessControl)
    return AR_accessible;

  if (Path.Access == AS_public)
    return AR_accessible;

  CXXRecordDecl *BaseD, *DerivedD;
  BaseD = cast<CXXRecordDecl>(Base->getAs<RecordType>()->getDecl());
  DerivedD = cast<CXXRecordDecl>(Derived->getAs<RecordType>()->getDecl());

  AccessedEntity Entity(AccessedEntity::Base, BaseD, DerivedD, Path.Access);
  if (DiagID)
    Entity.setDiag(DiagID) << Derived << Base;

  if (ForceUnprivileged)
    return CheckEffectiveAccess(*this, EffectiveContext(), AccessLoc, Entity);
  return CheckAccess(*this, AccessLoc, Entity);
}

/// Checks access to all the declarations in the given result set.
void Sema::CheckLookupAccess(const LookupResult &R) {
  assert(getLangOptions().AccessControl
         && "performing access check without access control");
  assert(R.getNamingClass() && "performing access check without naming class");

  for (LookupResult::iterator I = R.begin(), E = R.end(); I != E; ++I) {
    if (I.getAccess() != AS_public) {
      AccessedEntity Entity(AccessedEntity::Member,
                            R.getNamingClass(),
                            I.getPair());
      Entity.setDiag(diag::err_access);

      CheckAccess(*this, R.getNameLoc(), Entity);
    }
  }
}
