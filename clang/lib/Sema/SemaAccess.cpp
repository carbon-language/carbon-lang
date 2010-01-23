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

/// Find a class on the derivation path between Derived and Base that is
/// inaccessible. If @p NoPrivileges is true, special access rights (members
/// and friends) are not considered.
const CXXBaseSpecifier *Sema::FindInaccessibleBase(
    QualType Derived, QualType Base, CXXBasePaths &Paths, bool NoPrivileges) {
  Base = Context.getCanonicalType(Base).getUnqualifiedType();
  assert(!Paths.isAmbiguous(Base) &&
         "Can't check base class access if set of paths is ambiguous");
  assert(Paths.isRecordingPaths() &&
         "Can't check base class access without recorded paths");


  const CXXBaseSpecifier *InaccessibleBase = 0;

  const CXXRecordDecl *CurrentClassDecl = 0;
  if (CXXMethodDecl *MD = dyn_cast_or_null<CXXMethodDecl>(getCurFunctionDecl()))
    CurrentClassDecl = MD->getParent();

  for (CXXBasePaths::paths_iterator Path = Paths.begin(), PathsEnd = Paths.end();
      Path != PathsEnd; ++Path) {

    bool FoundInaccessibleBase = false;

    for (CXXBasePath::const_iterator Element = Path->begin(),
         ElementEnd = Path->end(); Element != ElementEnd; ++Element) {
      const CXXBaseSpecifier *Base = Element->Base;

      switch (Base->getAccessSpecifier()) {
      default:
        assert(0 && "invalid access specifier");
      case AS_public:
        // Nothing to do.
        break;
      case AS_private:
        // FIXME: Check if the current function/class is a friend.
        if (NoPrivileges || CurrentClassDecl != Element->Class)
          FoundInaccessibleBase = true;
        break;
      case AS_protected:
        // FIXME: Implement
        break;
      }

      if (FoundInaccessibleBase) {
        InaccessibleBase = Base;
        break;
      }
    }

    if (!FoundInaccessibleBase) {
      // We found a path to the base, our work here is done.
      return 0;
    }
  }

  assert(InaccessibleBase && "no path found, but no inaccessible base");
  return InaccessibleBase;
}

/// CheckBaseClassAccess - Check that a derived class can access its base class
/// and report an error if it can't. [class.access.base]
bool Sema::CheckBaseClassAccess(QualType Derived, QualType Base,
                                unsigned InaccessibleBaseID,
                                CXXBasePaths &Paths, SourceLocation AccessLoc,
                                DeclarationName Name) {

  if (!getLangOptions().AccessControl)
    return false;
  const CXXBaseSpecifier *InaccessibleBase = FindInaccessibleBase(
                                               Derived, Base, Paths);

  if (InaccessibleBase) {
    Diag(AccessLoc, InaccessibleBaseID)
      << Derived << Base << Name;

    AccessSpecifier AS = InaccessibleBase->getAccessSpecifierAsWritten();

    // If there's no written access specifier, then the inheritance specifier
    // is implicitly private.
    if (AS == AS_none)
      Diag(InaccessibleBase->getSourceRange().getBegin(),
           diag::note_inheritance_implicitly_private_here);
    else
      Diag(InaccessibleBase->getSourceRange().getBegin(),
           diag::note_inheritance_specifier_here) << AS;

    return true;
  }

  return false;
}

/// Diagnose the path which caused the given declaration to become
/// inaccessible.
static void DiagnoseAccessPath(Sema &S, const LookupResult &R, NamedDecl *D,
                               AccessSpecifier Access) {
  // Easy case: the decl's natural access determined its path access.
  if (Access == D->getAccess() || D->getAccess() == AS_private) {
    S.Diag(D->getLocation(), diag::note_access_natural)
      << (unsigned) (Access == AS_protected);
    return;
  }

  // TODO: flesh this out
  S.Diag(D->getLocation(), diag::note_access_constrained_by_path)
    << (unsigned) (Access == AS_protected);
}

/// Checks access to the given declaration in the current context.
///
/// \param R the means via which the access was made; must have a naming
///   class set
/// \param D the declaration accessed
/// \param Access the best access along any inheritance path from the
///   naming class to the declaration.  AS_none means the path is impossible
bool Sema::CheckAccess(const LookupResult &R, NamedDecl *D,
                       AccessSpecifier Access) {
  assert(R.getNamingClass() && "performing access check without naming class");

  // If the access path is public, it's accessible everywhere.
  if (Access == AS_public)
    return false;

  // Otherwise, derive the current class context.
  DeclContext *DC = CurContext;
  while (isa<CXXRecordDecl>(DC) &&
         cast<CXXRecordDecl>(DC)->isAnonymousStructOrUnion())
    DC = DC->getParent();

  CXXRecordDecl *CurRecord;
  if (isa<CXXRecordDecl>(DC))
    CurRecord = cast<CXXRecordDecl>(DC);
  else if (isa<CXXMethodDecl>(DC))
    CurRecord = cast<CXXMethodDecl>(DC)->getParent();
  else {
    Diag(R.getNameLoc(), diag::err_access_outside_class)
      << (Access == AS_protected);
    DiagnoseAccessPath(*this, R, D, Access);
    return true;
  }

  CXXRecordDecl *NamingClass = R.getNamingClass();
  while (NamingClass->isAnonymousStructOrUnion())
    // This should be guaranteed by the fact that the decl has
    // non-public access.  If not, we should make it guaranteed!
    NamingClass = cast<CXXRecordDecl>(NamingClass);

  // White-list accesses from within the declaring class.
  if (Access != AS_none &&
      CurRecord->getCanonicalDecl() == NamingClass->getCanonicalDecl())
    return false;

  // Protected access.
  if (Access == AS_protected) {
    // FIXME: implement [class.protected]p1
    if (CurRecord->isDerivedFrom(NamingClass))
      return false;

    // FIXME: dependent classes
  }

  // FIXME: friends

  // Okay, it's a bad access, reject it.

  
  CXXRecordDecl *DeclaringClass = cast<CXXRecordDecl>(D->getDeclContext());

  if (Access == AS_protected) {
    Diag(R.getNameLoc(), diag::err_access_protected)
      << Context.getTypeDeclType(DeclaringClass)
      << Context.getTypeDeclType(CurRecord);
    DiagnoseAccessPath(*this, R, D, Access);
    return true;
  }

  assert(Access == AS_private || Access == AS_none);
  Diag(R.getNameLoc(), diag::err_access_private)
    << Context.getTypeDeclType(DeclaringClass)
    << Context.getTypeDeclType(CurRecord);
  DiagnoseAccessPath(*this, R, D, Access);
  return true;
}

/// Checks access to all the declarations in the given result set.
void Sema::CheckAccess(const LookupResult &R) {
  for (LookupResult::iterator I = R.begin(), E = R.end(); I != E; ++I)
    CheckAccess(R, *I, I.getAccess());
}
