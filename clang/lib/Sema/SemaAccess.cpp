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

#include "SemaInherit.h"
#include "Sema.h"
#include "clang/AST/ASTContext.h"
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
    return true;
  }

  MemberDecl->setAccess(PrevMemberDecl->getAccess());
  return false;
}

/// Find a class on the derivation path between Derived and Base that is
/// inaccessible. If @p NoPrivileges is true, special access rights (members
/// and friends) are not considered.
const CXXBaseSpecifier *Sema::FindInaccessibleBase(
    QualType Derived, QualType Base, BasePaths &Paths, bool NoPrivileges) {
  Base = Context.getCanonicalType(Base).getUnqualifiedType();
  assert(!Paths.isAmbiguous(Base) &&
         "Can't check base class access if set of paths is ambiguous");
  assert(Paths.isRecordingPaths() &&
         "Can't check base class access without recorded paths");


  const CXXBaseSpecifier *InaccessibleBase = 0;

  const CXXRecordDecl *CurrentClassDecl = 0;
  if (CXXMethodDecl *MD = dyn_cast_or_null<CXXMethodDecl>(getCurFunctionDecl()))
    CurrentClassDecl = MD->getParent();

  for (BasePaths::paths_iterator Path = Paths.begin(), PathsEnd = Paths.end();
      Path != PathsEnd; ++Path) {

    bool FoundInaccessibleBase = false;

    for (BasePath::const_iterator Element = Path->begin(),
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
                                BasePaths &Paths, SourceLocation AccessLoc,
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
