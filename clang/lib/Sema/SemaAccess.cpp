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
  EffectiveContext() : Inner(0), Dependent(false) {}

  explicit EffectiveContext(DeclContext *DC)
    : Inner(DC),
      Dependent(DC->isDependentContext()) {

    // C++ [class.access.nest]p1:
    //   A nested class is a member and as such has the same access
    //   rights as any other member.
    // C++ [class.access]p2:
    //   A member of a class can also access all the names to which
    //   the class has access.  A local class of a member function
    //   may access the same names that the member function itself
    //   may access.
    // This almost implies that the privileges of nesting are transitive.
    // Technically it says nothing about the local classes of non-member
    // functions (which can gain privileges through friendship), but we
    // take that as an oversight.
    while (true) {
      if (isa<CXXRecordDecl>(DC)) {
        CXXRecordDecl *Record = cast<CXXRecordDecl>(DC)->getCanonicalDecl();
        Records.push_back(Record);
        DC = Record->getDeclContext();
      } else if (isa<FunctionDecl>(DC)) {
        FunctionDecl *Function = cast<FunctionDecl>(DC)->getCanonicalDecl();
        Functions.push_back(Function);
        DC = Function->getDeclContext();
      } else if (DC->isFileContext()) {
        break;
      } else {
        DC = DC->getParent();
      }
    }
  }

  bool isDependent() const { return Dependent; }

  bool includesClass(const CXXRecordDecl *R) const {
    R = R->getCanonicalDecl();
    return std::find(Records.begin(), Records.end(), R)
             != Records.end();
  }

  /// Retrieves the innermost "useful" context.  Can be null if we're
  /// doing access-control without privileges.
  DeclContext *getInnerContext() const {
    return Inner;
  }

  typedef llvm::SmallVectorImpl<CXXRecordDecl*>::const_iterator record_iterator;

  DeclContext *Inner;
  llvm::SmallVector<FunctionDecl*, 4> Functions;
  llvm::SmallVector<CXXRecordDecl*, 4> Records;
  bool Dependent;
};
}

static CXXRecordDecl *FindDeclaringClass(NamedDecl *D) {
  DeclContext *DC = D->getDeclContext();

  // This can only happen at top: enum decls only "publish" their
  // immediate members.
  if (isa<EnumDecl>(DC))
    DC = cast<EnumDecl>(DC)->getDeclContext();

  CXXRecordDecl *DeclaringClass = cast<CXXRecordDecl>(DC);
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

  // Check whether the friend is the template of a class in the
  // context chain.
  for (llvm::SmallVectorImpl<CXXRecordDecl*>::const_iterator
         I = EC.Records.begin(), E = EC.Records.end(); I != E; ++I) {
    CXXRecordDecl *Record = *I;

    // Figure out whether the current class has a template:
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

    // If the context isn't dependent, it can't be a dependent match.
    if (!EC.isDependent())
      continue;

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
  Sema::AccessResult OnFailure = Sema::AR_inaccessible;

  for (llvm::SmallVectorImpl<FunctionDecl*>::const_iterator
         I = EC.Functions.begin(), E = EC.Functions.end(); I != E; ++I) {
    if (Friend == *I)
      return Sema::AR_accessible;

    if (EC.isDependent() && MightInstantiateTo(S, *I, Friend))
      OnFailure = Sema::AR_dependent;
  }

  return OnFailure;
}

/// Determines whether the given friend function template matches
/// anything in the effective context.
static Sema::AccessResult MatchesFriend(Sema &S,
                                        const EffectiveContext &EC,
                                        FunctionTemplateDecl *Friend) {
  if (EC.Functions.empty()) return Sema::AR_inaccessible;

  Sema::AccessResult OnFailure = Sema::AR_inaccessible;

  for (llvm::SmallVectorImpl<FunctionDecl*>::const_iterator
         I = EC.Functions.begin(), E = EC.Functions.end(); I != E; ++I) {

    FunctionTemplateDecl *FTD = (*I)->getPrimaryTemplate();
    if (!FTD)
      FTD = (*I)->getDescribedFunctionTemplate();
    if (!FTD)
      continue;

    FTD = FTD->getCanonicalDecl();

    if (Friend == FTD)
      return Sema::AR_accessible;

    if (EC.isDependent() && MightInstantiateTo(S, FTD, Friend))
      OnFailure = Sema::AR_dependent;
  }

  return OnFailure;
}

/// Determines whether the given friend declaration matches anything
/// in the effective context.
static Sema::AccessResult MatchesFriend(Sema &S,
                                        const EffectiveContext &EC,
                                        FriendDecl *FriendD) {
  if (TypeSourceInfo *T = FriendD->getFriendType())
    return MatchesFriend(S, EC, T->getType()->getCanonicalTypeUnqualified());

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

static Sema::AccessResult HasAccess(Sema &S,
                                    const EffectiveContext &EC,
                                    const CXXRecordDecl *NamingClass,
                                    AccessSpecifier Access) {
  assert(NamingClass->getCanonicalDecl() == NamingClass &&
         "declaration should be canonicalized before being passed here");

  if (Access == AS_public) return Sema::AR_accessible;
  assert(Access == AS_private || Access == AS_protected);

  for (EffectiveContext::record_iterator
         I = EC.Records.begin(), E = EC.Records.end(); I != E; ++I) {
    // All the declarations in EC have been canonicalized, so pointer
    // equality from this point on will work fine.
    const CXXRecordDecl *ECRecord = *I;

    // [B2] and [M2]
    if (ECRecord == NamingClass)
      return Sema::AR_accessible;

    // [B3] and [M3]
    if (Access == AS_protected &&
        ECRecord->isDerivedFrom(const_cast<CXXRecordDecl*>(NamingClass)))
      return Sema::AR_accessible;
  }

  return GetFriendKind(S, EC, NamingClass);
}

/// Finds the best path from the naming class to the declaring class,
/// taking friend declarations into account.
///
/// C++0x [class.access.base]p5:
///   A member m is accessible at the point R when named in class N if
///   [M1] m as a member of N is public, or
///   [M2] m as a member of N is private, and R occurs in a member or
///        friend of class N, or
///   [M3] m as a member of N is protected, and R occurs in a member or
///        friend of class N, or in a member or friend of a class P
///        derived from N, where m as a member of P is public, private,
///        or protected, or
///   [M4] there exists a base class B of N that is accessible at R, and
///        m is accessible at R when named in class B.
///
/// C++0x [class.access.base]p4:
///   A base class B of N is accessible at R, if
///   [B1] an invented public member of B would be a public member of N, or
///   [B2] R occurs in a member or friend of class N, and an invented public
///        member of B would be a private or protected member of N, or
///   [B3] R occurs in a member or friend of a class P derived from N, and an
///        invented public member of B would be a private or protected member
///        of P, or
///   [B4] there exists a class S such that B is a base class of S accessible
///        at R and S is a base class of N accessible at R.
///
/// Along a single inheritance path we can restate both of these
/// iteratively:
///
/// First, we note that M1-4 are equivalent to B1-4 if the member is
/// treated as a notional base of its declaring class with inheritance
/// access equivalent to the member's access.  Therefore we need only
/// ask whether a class B is accessible from a class N in context R.
///
/// Let B_1 .. B_n be the inheritance path in question (i.e. where
/// B_1 = N, B_n = B, and for all i, B_{i+1} is a direct base class of
/// B_i).  For i in 1..n, we will calculate ACAB(i), the access to the
/// closest accessible base in the path:
///   Access(a, b) = (* access on the base specifier from a to b *)
///   Merge(a, forbidden) = forbidden
///   Merge(a, private) = forbidden
///   Merge(a, b) = min(a,b)
///   Accessible(c, forbidden) = false
///   Accessible(c, private) = (R is c) || IsFriend(c, R)
///   Accessible(c, protected) = (R derived from c) || IsFriend(c, R)
///   Accessible(c, public) = true
///   ACAB(n) = public
///   ACAB(i) =
///     let AccessToBase = Merge(Access(B_i, B_{i+1}), ACAB(i+1)) in
///     if Accessible(B_i, AccessToBase) then public else AccessToBase
///
/// B is an accessible base of N at R iff ACAB(1) = public.
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
      PathAccess = std::max(PathAccess, BaseAccess);
      switch (HasAccess(S, EC, I->Class, PathAccess)) {
      case Sema::AR_inaccessible: break;
      case Sema::AR_accessible: PathAccess = AS_public; break;
      case Sema::AR_dependent:
        AnyDependent = true;
        goto Next;
      case Sema::AR_delayed:
        llvm_unreachable("friend resolution is never delayed"); break;
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
                               const Sema::AccessedEntity &Entity) {
  AccessSpecifier Access = Entity.getAccess();
  CXXRecordDecl *NamingClass = Entity.getNamingClass();
  NamingClass = NamingClass->getCanonicalDecl();

  NamedDecl *D;
  CXXRecordDecl *DeclaringClass;
  if (Entity.isMemberAccess()) {
    D = Entity.getTargetDecl();
    DeclaringClass = FindDeclaringClass(D);
  } else {
    D = 0;
    DeclaringClass = Entity.getBaseClass();
  }
  DeclaringClass = DeclaringClass->getCanonicalDecl();

  // Easy case: the decl's natural access determined its path access.
  // We have to check against AS_private here in case Access is AS_none,
  // indicating a non-public member of a private base class.
  if (D && (Access == D->getAccess() || D->getAccess() == AS_private)) {
    switch (HasAccess(S, EC, DeclaringClass, D->getAccess())) {
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

static void DiagnoseBadAccess(Sema &S, SourceLocation Loc,
                              const EffectiveContext &EC,
                              const Sema::AccessedEntity &Entity) {
  const CXXRecordDecl *NamingClass = Entity.getNamingClass();
  NamedDecl *D;
  const CXXRecordDecl *DeclaringClass;
  if (Entity.isMemberAccess()) {
    D = Entity.getTargetDecl();
    DeclaringClass = FindDeclaringClass(D);
  } else {
    D = 0;
    DeclaringClass = Entity.getBaseClass();
  }

  S.Diag(Loc, Entity.getDiag())
    << (Entity.getAccess() == AS_protected)
    << (D ? D->getDeclName() : DeclarationName())
    << S.Context.getTypeDeclType(NamingClass)
    << S.Context.getTypeDeclType(DeclaringClass);
  DiagnoseAccessPath(S, EC, Entity);
}

/// Determines whether the accessed entity is accessible.  Public members
/// have been weeded out by this point.
static Sema::AccessResult IsAccessible(Sema &S,
                                       const EffectiveContext &EC,
                                       const Sema::AccessedEntity &Entity) {
  // Determine the actual naming class.
  CXXRecordDecl *NamingClass = Entity.getNamingClass();
  while (NamingClass->isAnonymousStructOrUnion())
    NamingClass = cast<CXXRecordDecl>(NamingClass->getParent());
  NamingClass = NamingClass->getCanonicalDecl();

  AccessSpecifier UnprivilegedAccess = Entity.getAccess();
  assert(UnprivilegedAccess != AS_public && "public access not weeded out");

  // Before we try to recalculate access paths, try to white-list
  // accesses which just trade in on the final step, i.e. accesses
  // which don't require [M4] or [B4]. These are by far the most
  // common forms of access.
  if (UnprivilegedAccess != AS_none) {
    switch (HasAccess(S, EC, NamingClass, UnprivilegedAccess)) {
    case Sema::AR_dependent:
      // This is actually an interesting policy decision.  We don't
      // *have* to delay immediately here: we can do the full access
      // calculation in the hope that friendship on some intermediate
      // class will make the declaration accessible non-dependently.
      // But that's not cheap, and odds are very good (note: assertion
      // made without data) that the friend declaration will determine
      // access.
      return Sema::AR_dependent;

    case Sema::AR_accessible: return Sema::AR_accessible;
    case Sema::AR_inaccessible: break;
    case Sema::AR_delayed:
      llvm_unreachable("friendship never subject to contextual delay");
    }
  }

  // Determine the declaring class.
  CXXRecordDecl *DeclaringClass;
  if (Entity.isMemberAccess()) {
    DeclaringClass = FindDeclaringClass(Entity.getTargetDecl());
  } else {
    DeclaringClass = Entity.getBaseClass();
  }
  DeclaringClass = DeclaringClass->getCanonicalDecl();

  // We lower member accesses to base accesses by pretending that the
  // member is a base class of its declaring class.
  AccessSpecifier FinalAccess;

  if (Entity.isMemberAccess()) {
    // Determine if the declaration is accessible from EC when named
    // in its declaring class.
    NamedDecl *Target = Entity.getTargetDecl();

    FinalAccess = Target->getAccess();
    switch (HasAccess(S, EC, DeclaringClass, FinalAccess)) {
    case Sema::AR_accessible: FinalAccess = AS_public; break;
    case Sema::AR_inaccessible: break;
    case Sema::AR_dependent: return Sema::AR_dependent; // see above
    case Sema::AR_delayed: llvm_unreachable("friend status is never delayed");
    }

    if (DeclaringClass == NamingClass)
      return (FinalAccess == AS_public
              ? Sema::AR_accessible
              : Sema::AR_inaccessible);
  } else {
    FinalAccess = AS_public;
  }

  assert(DeclaringClass != NamingClass);

  // Append the declaration's access if applicable.
  CXXBasePaths Paths;
  CXXBasePath *Path = FindBestPath(S, EC, NamingClass, DeclaringClass,
                                   FinalAccess, Paths);
  if (!Path)
    return Sema::AR_dependent;

  assert(Path->Access <= UnprivilegedAccess &&
         "access along best path worse than direct?");
  if (Path->Access == AS_public)
    return Sema::AR_accessible;
  return Sema::AR_inaccessible;
}

static void DelayAccess(Sema &S,
                        const EffectiveContext &EC,
                        SourceLocation Loc,
                        const Sema::AccessedEntity &Entity) {
  assert(EC.isDependent() && "delaying non-dependent access");
  DeclContext *DC = EC.getInnerContext();
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
                                         const Sema::AccessedEntity &Entity) {
  assert(Entity.getAccess() != AS_public && "called for public access!");

  switch (IsAccessible(S, EC, Entity)) {
  case Sema::AR_dependent:
    DelayAccess(S, EC, Loc, Entity);
    return Sema::AR_dependent;

  case Sema::AR_delayed:
    llvm_unreachable("IsAccessible cannot contextually delay");

  case Sema::AR_inaccessible:
    if (!Entity.isQuiet())
      DiagnoseBadAccess(S, Loc, EC, Entity);
    return Sema::AR_inaccessible;

  case Sema::AR_accessible:
    break;
  }

  // We only consider the natural access of the declaration when
  // deciding whether to do the protected check.
  if (Entity.isMemberAccess() && Entity.getAccess() == AS_protected) {
    NamedDecl *D = Entity.getTargetDecl();
    if (isa<FieldDecl>(D) ||
        (isa<CXXMethodDecl>(D) && cast<CXXMethodDecl>(D)->isInstance())) {
      // FIXME: implement [class.protected]
    }
  }

  return Sema::AR_accessible;
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
    AccessedEntity Entity(Context,
                          AccessedEntity::Member,
                          cast<CXXRecordDecl>(NamingD),
                          Access,
                          cast<NamedDecl>(TargetD));
    Entity.setDiag(DD.getDiagnostic());
    CheckAccess(*this, Loc, Entity);
  } else {
    AccessedEntity Entity(Context,
                          AccessedEntity::Base,
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

  AccessedEntity Entity(Context, AccessedEntity::Member, E->getNamingClass(), 
                        Found);
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

  AccessedEntity Entity(Context, AccessedEntity::Member, E->getNamingClass(), 
                        Found);
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
  AccessedEntity Entity(Context, AccessedEntity::Member, NamingClass,
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
  AccessedEntity Entity(Context, AccessedEntity::Member, NamingClass,
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
  AccessedEntity Entity(Context, AccessedEntity::Member, NamingClass,
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

  AccessedEntity Entity(Context, AccessedEntity::Member, NamingClass, Found);
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

  AccessedEntity Entity(Context, AccessedEntity::Member, NamingClass, Found);
  Entity.setDiag(diag::err_access)
    << ObjectExpr->getSourceRange()
    << (ArgExpr ? ArgExpr->getSourceRange() : SourceRange());

  return CheckAccess(*this, OpLoc, Entity);
}

Sema::AccessResult Sema::CheckAddressOfMemberAccess(Expr *OvlExpr,
                                                    DeclAccessPair Found) {
  if (!getLangOptions().AccessControl ||
      Found.getAccess() == AS_none ||
      Found.getAccess() == AS_public)
    return AR_accessible;

  OverloadExpr *Ovl = OverloadExpr::find(OvlExpr).getPointer();
  NestedNameSpecifier *Qualifier = Ovl->getQualifier();
  assert(Qualifier && "address of overloaded member without qualifier");

  CXXScopeSpec SS;
  SS.setScopeRep(Qualifier);
  SS.setRange(Ovl->getQualifierRange());
  DeclContext *DC = computeDeclContext(SS);
  assert(DC && DC->isRecord() && "scope did not resolve to record");
  CXXRecordDecl *NamingClass = cast<CXXRecordDecl>(DC);

  AccessedEntity Entity(Context, AccessedEntity::Member, NamingClass, Found);
  Entity.setDiag(diag::err_access)
    << Ovl->getSourceRange();

  return CheckAccess(*this, Ovl->getNameLoc(), Entity);
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

  AccessedEntity Entity(Context, AccessedEntity::Base, BaseD, DerivedD, 
                        Path.Access);
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
      AccessedEntity Entity(Context, AccessedEntity::Member,
                            R.getNamingClass(),
                            I.getPair());
      Entity.setDiag(diag::err_access);

      CheckAccess(*this, R.getNameLoc(), Entity);
    }
  }
}
