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
  EffectiveContext() : Record(0), Function(0) {}

  explicit EffectiveContext(DeclContext *DC) {
    if (isa<FunctionDecl>(DC)) {
      Function = cast<FunctionDecl>(DC);
      DC = Function->getDeclContext();
    } else
      Function = 0;
    
    if (isa<CXXRecordDecl>(DC))
      Record = cast<CXXRecordDecl>(DC)->getCanonicalDecl();
    else
      Record = 0;
  }

  bool isClass(const CXXRecordDecl *R) const {
    return R->getCanonicalDecl() == Record;
  }

  CXXRecordDecl *Record;
  FunctionDecl *Function;
};
}

static CXXRecordDecl *FindDeclaringClass(NamedDecl *D) {
  CXXRecordDecl *DeclaringClass = cast<CXXRecordDecl>(D->getDeclContext());
  while (DeclaringClass->isAnonymousStructOrUnion())
    DeclaringClass = cast<CXXRecordDecl>(DeclaringClass->getDeclContext());
  return DeclaringClass;
}

static Sema::AccessResult GetFriendKind(Sema &S,
                                        const EffectiveContext &EC,
                                        const CXXRecordDecl *Class) {
  if (EC.isClass(Class))
    return Sema::AR_accessible;

  // FIXME: implement
  return Sema::AR_inaccessible;
}

/// Finds the best path from the naming class to the declaring class,
/// taking friend declarations into account.
///
/// \return null if friendship is dependent
static CXXBasePath *FindBestPath(Sema &S,
                                 const EffectiveContext &EC,
                                 CXXRecordDecl *Derived,
                                 CXXRecordDecl *Base,
                                 CXXBasePaths &Paths) {
  // Derive the paths to the desired base.
  bool isDerived = Derived->isDerivedFrom(Base, Paths);
  assert(isDerived && "derived class not actually derived from base");
  (void) isDerived;

  CXXBasePath *BestPath = 0;

  // Derive the friend-modified access along each path.
  for (CXXBasePaths::paths_iterator PI = Paths.begin(), PE = Paths.end();
         PI != PE; ++PI) {

    // Walk through the path backwards.
    AccessSpecifier PathAccess = AS_public;
    CXXBasePath::iterator I = PI->end(), E = PI->begin();
    while (I != E) {
      --I;

      AccessSpecifier BaseAccess = I->Base->getAccessSpecifier();
      if (BaseAccess != AS_public) {
        switch (GetFriendKind(S, EC, I->Class)) {
        case Sema::AR_inaccessible: break;
        case Sema::AR_accessible: BaseAccess = AS_public; break;
        case Sema::AR_dependent: return 0;
        case Sema::AR_delayed:
          llvm_unreachable("friend resolution is never delayed"); break;
        }
      }

      PathAccess = CXXRecordDecl::MergeAccess(BaseAccess, PathAccess);
    }

    // Note that we modify the path's Access field to the
    // friend-modified access.
    if (BestPath == 0 || PathAccess < BestPath->Access) {
      BestPath = &*PI;
      BestPath->Access = PathAccess;
    }
  }

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
  CXXBasePath &Path = *FindBestPath(S, EC, NamingClass, DeclaringClass, Paths);

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

  if (isa<CXXConstructorDecl>(D)) {
    unsigned DiagID = (Access == AS_protected ? diag::err_access_ctor_protected
                                              : diag::err_access_ctor_private);
    S.Diag(Loc, DiagID)
      << S.Context.getTypeDeclType(DeclaringClass);
  } else {
    unsigned DiagID = (Access == AS_protected ? diag::err_access_protected
                                              : diag::err_access_private);
    S.Diag(Loc, DiagID)
      << D->getDeclName()
      << S.Context.getTypeDeclType(DeclaringClass);
  }
  DiagnoseAccessPath(S, EC, NamingClass, DeclaringClass, D, Access);
}

/// Diagnose an inaccessible hierarchy conversion.
static void DiagnoseInaccessibleBase(Sema &S, SourceLocation Loc,
                                     const EffectiveContext &EC,
                                     AccessSpecifier Access,
                                     const Sema::AccessedEntity &Entity,
                                     Sema::AccessDiagnosticsKind ADK) {
  if (ADK == Sema::ADK_covariance) {
    S.Diag(Loc, diag::err_covariant_return_inaccessible_base)
      << S.Context.getTypeDeclType(Entity.getDerivedClass())
      << S.Context.getTypeDeclType(Entity.getBaseClass())
      << (Access == AS_protected);
  } else if (Entity.getKind() == Sema::AccessedEntity::BaseToDerivedConversion) {
    S.Diag(Loc, diag::err_downcast_from_inaccessible_base)
      << S.Context.getTypeDeclType(Entity.getDerivedClass())
      << S.Context.getTypeDeclType(Entity.getBaseClass())
      << (Access == AS_protected);
  } else {
    S.Diag(Loc, diag::err_upcast_to_inaccessible_base)
      << S.Context.getTypeDeclType(Entity.getDerivedClass())
      << S.Context.getTypeDeclType(Entity.getBaseClass())
      << (Access == AS_protected);
  }
  DiagnoseAccessPath(S, EC, Entity.getDerivedClass(),
                     Entity.getBaseClass(), 0, Access);
}

static void DiagnoseBadAccess(Sema &S,
                              SourceLocation Loc,
                              const EffectiveContext &EC,
                              CXXRecordDecl *NamingClass,
                              AccessSpecifier Access,
                              const Sema::AccessedEntity &Entity,
                              Sema::AccessDiagnosticsKind ADK) {
  if (Entity.isMemberAccess())
    DiagnoseInaccessibleMember(S, Loc, EC, NamingClass, Access, Entity);
  else
    DiagnoseInaccessibleBase(S, Loc, EC, Access, Entity, ADK);
}


/// Try to elevate access using friend declarations.  This is
/// potentially quite expensive.
static void TryElevateAccess(Sema &S,
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
  AccessSpecifier DeclAccess = AS_none;
  if (Entity.isMemberAccess()) {
    NamedDecl *Target = Entity.getTargetDecl();

    DeclAccess = Target->getAccess();
    if (DeclAccess != AS_public) {
      switch (GetFriendKind(S, EC, DeclaringClass)) {
      case Sema::AR_accessible: DeclAccess = AS_public; break;
      case Sema::AR_inaccessible: break;
      case Sema::AR_dependent: /* FIXME: delay dependent friendship */ return;
      case Sema::AR_delayed: llvm_unreachable("friend status is never delayed");
      }
    }

    if (DeclaringClass == NamingClass) {
      Access = DeclAccess;
      return;
    }
  }

  assert(DeclaringClass != NamingClass);

  // Append the declaration's access if applicable.
  CXXBasePaths Paths;
  CXXBasePath *Path = FindBestPath(S, EC, Entity.getNamingClass(),
                                   DeclaringClass, Paths);
  if (!Path) {
    // FIXME: delay dependent friendship
    return;
  }

  // Grab the access along the best path.
  AccessSpecifier NewAccess = Path->Access;
  if (Entity.isMemberAccess())
    NewAccess = CXXRecordDecl::MergeAccess(NewAccess, DeclAccess);
  
  assert(NewAccess <= Access && "access along best path worse than direct?");
  Access = NewAccess;
}

/// Checks access to an entity from the given effective context.
static Sema::AccessResult CheckEffectiveAccess(Sema &S,
                                               const EffectiveContext &EC,
                                               SourceLocation Loc,
                                         Sema::AccessedEntity const &Entity,
                                         Sema::AccessDiagnosticsKind ADK) {
  AccessSpecifier Access = Entity.getAccess();
  assert(Access != AS_public);

  CXXRecordDecl *NamingClass = Entity.getNamingClass();
  while (NamingClass->isAnonymousStructOrUnion())
    // This should be guaranteed by the fact that the decl has
    // non-public access.  If not, we should make it guaranteed!
    NamingClass = cast<CXXRecordDecl>(NamingClass);

  if (!EC.Record) {
    TryElevateAccess(S, EC, Entity, Access);
    if (Access == AS_public) return Sema::AR_accessible;

    if (ADK != Sema::ADK_quiet)
      DiagnoseBadAccess(S, Loc, EC, NamingClass, Access, Entity, ADK);
    return Sema::AR_inaccessible;
  }

  // White-list accesses from within the declaring class.
  if (Access != AS_none && EC.isClass(NamingClass))
    return Sema::AR_accessible;
  
  // If the access is worse than 'protected', try to promote to it using
  // friend declarations.
  bool TriedElevation = false;
  if (Access != AS_protected) {
    TryElevateAccess(S, EC, Entity, Access);
    if (Access == AS_public) return Sema::AR_accessible;
    TriedElevation = true;
  }

  // Protected access.
  if (Access == AS_protected) {
    // FIXME: implement [class.protected]p1
    if (EC.Record->isDerivedFrom(NamingClass))
      return Sema::AR_accessible;

    // FIXME: delay dependent classes
  }

  // We're about to reject;  one last chance to promote access.
  if (!TriedElevation) {
    TryElevateAccess(S, EC, Entity, Access);
    if (Access == AS_public) return Sema::AR_accessible;
  }
    
  // Okay, that's it, reject it.
  if (ADK != Sema::ADK_quiet)
    DiagnoseBadAccess(S, Loc, EC, NamingClass, Access, Entity, ADK);
  return Sema::AR_inaccessible;
}

static Sema::AccessResult CheckAccess(Sema &S, SourceLocation Loc,
                                      const Sema::AccessedEntity &Entity,
                                      Sema::AccessDiagnosticsKind ADK
                                        = Sema::ADK_normal) {
  // If the access path is public, it's accessible everywhere.
  if (Entity.getAccess() == AS_public)
    return Sema::AR_accessible;

  // If we're currently parsing a top-level declaration, delay
  // diagnostics.  This is the only case where parsing a declaration
  // can actually change our effective context for the purposes of
  // access control.
  if (S.CurContext->isFileContext() && S.ParsingDeclDepth) {
    assert(ADK == Sema::ADK_normal && "delaying abnormal access check");
    S.DelayedDiagnostics.push_back(
        Sema::DelayedDiagnostic::makeAccess(Loc, Entity));
    return Sema::AR_delayed;
  }

  return CheckEffectiveAccess(S, EffectiveContext(S.CurContext),
                              Loc, Entity, ADK);
}

void Sema::HandleDelayedAccessCheck(DelayedDiagnostic &DD, Decl *Ctx) {
  // Pretend we did this from the context of the newly-parsed
  // declaration.
  EffectiveContext EC(Ctx->getDeclContext());

  if (CheckEffectiveAccess(*this, EC, DD.Loc, DD.AccessData, ADK_normal))
    DD.Triggered = true;
}

Sema::AccessResult Sema::CheckUnresolvedLookupAccess(UnresolvedLookupExpr *E,
                                                     NamedDecl *D,
                                                     AccessSpecifier Access) {
  if (!getLangOptions().AccessControl || !E->getNamingClass())
    return AR_accessible;

  return CheckAccess(*this, E->getNameLoc(),
                 AccessedEntity::makeMember(E->getNamingClass(), Access, D));
}

/// Perform access-control checking on a previously-unresolved member
/// access which has now been resolved to a member.
Sema::AccessResult Sema::CheckUnresolvedMemberAccess(UnresolvedMemberExpr *E,
                                                     NamedDecl *D,
                                                     AccessSpecifier Access) {
  if (!getLangOptions().AccessControl)
    return AR_accessible;

  return CheckAccess(*this, E->getMemberLoc(),
                 AccessedEntity::makeMember(E->getNamingClass(), Access, D));
}

Sema::AccessResult Sema::CheckDestructorAccess(SourceLocation Loc,
                                               const RecordType *RT) {
  if (!getLangOptions().AccessControl)
    return AR_accessible;

  CXXRecordDecl *NamingClass = cast<CXXRecordDecl>(RT->getDecl());
  CXXDestructorDecl *Dtor = NamingClass->getDestructor(Context);

  AccessSpecifier Access = Dtor->getAccess();
  if (Access == AS_public)
    return AR_accessible;

  return CheckAccess(*this, Loc,
                 AccessedEntity::makeMember(NamingClass, Access, Dtor));
}

/// Checks access to a constructor.
Sema::AccessResult Sema::CheckConstructorAccess(SourceLocation UseLoc,
                                  CXXConstructorDecl *Constructor,
                                  AccessSpecifier Access) {
  if (!getLangOptions().AccessControl)
    return AR_accessible;

  CXXRecordDecl *NamingClass = Constructor->getParent();
  return CheckAccess(*this, UseLoc,
                 AccessedEntity::makeMember(NamingClass, Access, Constructor));
}

/// Checks access to an overloaded member operator, including
/// conversion operators.
Sema::AccessResult Sema::CheckMemberOperatorAccess(SourceLocation OpLoc,
                                                   Expr *ObjectExpr,
                                                   NamedDecl *MemberOperator,
                                                   AccessSpecifier Access) {
  if (!getLangOptions().AccessControl)
    return AR_accessible;

  const RecordType *RT = ObjectExpr->getType()->getAs<RecordType>();
  assert(RT && "found member operator but object expr not of record type");
  CXXRecordDecl *NamingClass = cast<CXXRecordDecl>(RT->getDecl());

  return CheckAccess(*this, OpLoc,
            AccessedEntity::makeMember(NamingClass, Access, MemberOperator));
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
                                              bool IsBaseToDerived,
                                              QualType Base,
                                              QualType Derived,
                                              const CXXBasePath &Path,
                                              bool ForceCheck,
                                              bool ForceUnprivileged,
                                              AccessDiagnosticsKind ADK) {
  if (!ForceCheck && !getLangOptions().AccessControl)
    return AR_accessible;

  if (Path.Access == AS_public)
    return AR_accessible;

  // TODO: preserve the information about which types exactly were used.
  CXXRecordDecl *BaseD, *DerivedD;
  BaseD = cast<CXXRecordDecl>(Base->getAs<RecordType>()->getDecl());
  DerivedD = cast<CXXRecordDecl>(Derived->getAs<RecordType>()->getDecl());
  AccessedEntity Entity = AccessedEntity::makeBaseClass(IsBaseToDerived,
                                                        BaseD, DerivedD,
                                                        Path.Access);

  if (ForceUnprivileged)
    return CheckEffectiveAccess(*this, EffectiveContext(),
                                AccessLoc, Entity, ADK);
  return CheckAccess(*this, AccessLoc, Entity, ADK);
}

/// Checks access to all the declarations in the given result set.
void Sema::CheckLookupAccess(const LookupResult &R) {
  assert(getLangOptions().AccessControl
         && "performing access check without access control");
  assert(R.getNamingClass() && "performing access check without naming class");

  for (LookupResult::iterator I = R.begin(), E = R.end(); I != E; ++I)
    if (I.getAccess() != AS_public)
      CheckAccess(*this, R.getNameLoc(),
                  AccessedEntity::makeMember(R.getNamingClass(),
                                             I.getAccess(), *I));
}
