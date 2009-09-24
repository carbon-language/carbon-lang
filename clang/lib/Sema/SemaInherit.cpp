//===---- SemaInherit.cpp - C++ Inheritance ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Sema routines for C++ inheritance semantics,
// including searching the inheritance hierarchy.
//
//===----------------------------------------------------------------------===//

#include "SemaInherit.h"
#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeOrdering.h"
#include <algorithm>
#include <memory>
#include <set>
#include <string>

using namespace clang;

/// \brief Computes the set of declarations referenced by these base
/// paths.
void BasePaths::ComputeDeclsFound() {
  assert(NumDeclsFound == 0 && !DeclsFound &&
         "Already computed the set of declarations");

  std::set<NamedDecl *> Decls;
  for (BasePaths::paths_iterator Path = begin(), PathEnd = end();
       Path != PathEnd; ++Path)
    Decls.insert(*Path->Decls.first);

  NumDeclsFound = Decls.size();
  DeclsFound = new NamedDecl * [NumDeclsFound];
  std::copy(Decls.begin(), Decls.end(), DeclsFound);
}

BasePaths::decl_iterator BasePaths::found_decls_begin() {
  if (NumDeclsFound == 0)
    ComputeDeclsFound();
  return DeclsFound;
}

BasePaths::decl_iterator BasePaths::found_decls_end() {
  if (NumDeclsFound == 0)
    ComputeDeclsFound();
  return DeclsFound + NumDeclsFound;
}

/// isAmbiguous - Determines whether the set of paths provided is
/// ambiguous, i.e., there are two or more paths that refer to
/// different base class subobjects of the same type. BaseType must be
/// an unqualified, canonical class type.
bool BasePaths::isAmbiguous(QualType BaseType) {
  assert(BaseType->isCanonical() && "Base type must be the canonical type");
  assert(BaseType.hasQualifiers() == 0 && "Base type must be unqualified");
  std::pair<bool, unsigned>& Subobjects = ClassSubobjects[BaseType];
  return Subobjects.second + (Subobjects.first? 1 : 0) > 1;
}

/// clear - Clear out all prior path information.
void BasePaths::clear() {
  Paths.clear();
  ClassSubobjects.clear();
  ScratchPath.clear();
  DetectedVirtual = 0;
}

/// @brief Swaps the contents of this BasePaths structure with the
/// contents of Other.
void BasePaths::swap(BasePaths &Other) {
  std::swap(Origin, Other.Origin);
  Paths.swap(Other.Paths);
  ClassSubobjects.swap(Other.ClassSubobjects);
  std::swap(FindAmbiguities, Other.FindAmbiguities);
  std::swap(RecordPaths, Other.RecordPaths);
  std::swap(DetectVirtual, Other.DetectVirtual);
  std::swap(DetectedVirtual, Other.DetectedVirtual);
}

/// IsDerivedFrom - Determine whether the type Derived is derived from
/// the type Base, ignoring qualifiers on Base and Derived. This
/// routine does not assess whether an actual conversion from a
/// Derived* to a Base* is legal, because it does not account for
/// ambiguous conversions or conversions to private/protected bases.
bool Sema::IsDerivedFrom(QualType Derived, QualType Base) {
  BasePaths Paths(/*FindAmbiguities=*/false, /*RecordPaths=*/false,
                  /*DetectVirtual=*/false);
  return IsDerivedFrom(Derived, Base, Paths);
}

/// IsDerivedFrom - Determine whether the type Derived is derived from
/// the type Base, ignoring qualifiers on Base and Derived. This
/// routine does not assess whether an actual conversion from a
/// Derived* to a Base* is legal, because it does not account for
/// ambiguous conversions or conversions to private/protected
/// bases. This routine will use Paths to determine if there are
/// ambiguous paths (if @c Paths.isFindingAmbiguities()) and record
/// information about all of the paths (if @c Paths.isRecordingPaths()).
bool Sema::IsDerivedFrom(QualType Derived, QualType Base, BasePaths &Paths) {
  Derived = Context.getCanonicalType(Derived).getUnqualifiedType();
  Base = Context.getCanonicalType(Base).getUnqualifiedType();

  if (!Derived->isRecordType() || !Base->isRecordType())
    return false;

  if (Derived == Base)
    return false;

  Paths.setOrigin(Derived);
  return LookupInBases(cast<CXXRecordDecl>(Derived->getAs<RecordType>()->getDecl()),
                       MemberLookupCriteria(Base), Paths);
}

/// LookupInBases - Look for something that meets the specified
/// Criteria within the base classes of Class (or any of its base
/// classes, transitively). This routine populates BasePaths with the
/// list of paths that one can take to find the entity that meets the
/// search criteria, and returns true if any such entity is found. The
/// various options passed to the BasePath constructor will affect the
/// behavior of this lookup, e.g., whether it finds ambiguities,
/// records paths, or attempts to detect the use of virtual base
/// classes.
bool Sema::LookupInBases(CXXRecordDecl *Class,
                         const MemberLookupCriteria& Criteria,
                         BasePaths &Paths) {
  bool FoundPath = false;

  for (CXXRecordDecl::base_class_const_iterator BaseSpec = Class->bases_begin(),
                                             BaseSpecEnd = Class->bases_end();
       BaseSpec != BaseSpecEnd; ++BaseSpec) {
    // Find the record of the base class subobjects for this type.
    QualType BaseType = Context.getCanonicalType(BaseSpec->getType());
    BaseType = BaseType.getUnqualifiedType();

    // C++ [temp.dep]p3:
    //   In the definition of a class template or a member of a class template,
    //   if a base class of the class template depends on a template-parameter,
    //   the base class scope is not examined during unqualified name lookup 
    //   either at the point of definition of the class template or member or 
    //   during an instantiation of the class tem- plate or member.
    if (BaseType->isDependentType())
      continue;

    // Determine whether we need to visit this base class at all,
    // updating the count of subobjects appropriately.
    std::pair<bool, unsigned>& Subobjects = Paths.ClassSubobjects[BaseType];
    bool VisitBase = true;
    bool SetVirtual = false;
    if (BaseSpec->isVirtual()) {
      VisitBase = !Subobjects.first;
      Subobjects.first = true;
      if (Paths.isDetectingVirtual() && Paths.DetectedVirtual == 0) {
        // If this is the first virtual we find, remember it. If it turns out
        // there is no base path here, we'll reset it later.
        Paths.DetectedVirtual = BaseType->getAs<RecordType>();
        SetVirtual = true;
      }
    } else
      ++Subobjects.second;

    if (Paths.isRecordingPaths()) {
      // Add this base specifier to the current path.
      BasePathElement Element;
      Element.Base = &*BaseSpec;
      Element.Class = Class;
      if (BaseSpec->isVirtual())
        Element.SubobjectNumber = 0;
      else
        Element.SubobjectNumber = Subobjects.second;
      Paths.ScratchPath.push_back(Element);
    }

    CXXRecordDecl *BaseRecord
      = cast<CXXRecordDecl>(BaseSpec->getType()->getAs<RecordType>()->getDecl());

    // Either look at the base class type or look into the base class
    // type to see if we've found a member that meets the search
    // criteria.
    bool FoundPathToThisBase = false;
    switch (Criteria.Kind) {
    case MemberLookupCriteria::LK_Base:
      FoundPathToThisBase
        = (Context.getCanonicalType(BaseSpec->getType()) == Criteria.Base);
      break;
    case MemberLookupCriteria::LK_NamedMember:
      Paths.ScratchPath.Decls = BaseRecord->lookup(Criteria.Name);
      while (Paths.ScratchPath.Decls.first != Paths.ScratchPath.Decls.second) {
        if (isAcceptableLookupResult(*Paths.ScratchPath.Decls.first,
                                     Criteria.NameKind, Criteria.IDNS)) {
          FoundPathToThisBase = true;
          break;
        }
        ++Paths.ScratchPath.Decls.first;
      }
      break;
    case MemberLookupCriteria::LK_OverriddenMember:
      Paths.ScratchPath.Decls =
        BaseRecord->lookup(Criteria.Method->getDeclName());
      while (Paths.ScratchPath.Decls.first != Paths.ScratchPath.Decls.second) {
        if (CXXMethodDecl *MD =
              dyn_cast<CXXMethodDecl>(*Paths.ScratchPath.Decls.first)) {
          OverloadedFunctionDecl::function_iterator MatchedDecl;
          if (MD->isVirtual() &&
              !IsOverload(Criteria.Method, MD, MatchedDecl)) {
            FoundPathToThisBase = true;
            break;
          }
        }

        ++Paths.ScratchPath.Decls.first;
      }
      break;
    }

    if (FoundPathToThisBase) {
      // We've found a path that terminates that this base.
      FoundPath = true;
      if (Paths.isRecordingPaths()) {
        // We have a path. Make a copy of it before moving on.
        Paths.Paths.push_back(Paths.ScratchPath);
      } else if (!Paths.isFindingAmbiguities()) {
        // We found a path and we don't care about ambiguities;
        // return immediately.
        return FoundPath;
      }
    } else if (VisitBase && LookupInBases(BaseRecord, Criteria, Paths)) {
      // C++ [class.member.lookup]p2:
      //   A member name f in one sub-object B hides a member name f in
      //   a sub-object A if A is a base class sub-object of B. Any
      //   declarations that are so hidden are eliminated from
      //   consideration.

      // There is a path to a base class that meets the criteria. If we're not
      // collecting paths or finding ambiguities, we're done.
      FoundPath = true;
      if (!Paths.isFindingAmbiguities())
        return FoundPath;
    }

    // Pop this base specifier off the current path (if we're
    // collecting paths).
    if (Paths.isRecordingPaths())
      Paths.ScratchPath.pop_back();
    // If we set a virtual earlier, and this isn't a path, forget it again.
    if (SetVirtual && !FoundPath) {
      Paths.DetectedVirtual = 0;
    }
  }

  return FoundPath;
}

/// CheckDerivedToBaseConversion - Check whether the Derived-to-Base
/// conversion (where Derived and Base are class types) is
/// well-formed, meaning that the conversion is unambiguous (and
/// that all of the base classes are accessible). Returns true
/// and emits a diagnostic if the code is ill-formed, returns false
/// otherwise. Loc is the location where this routine should point to
/// if there is an error, and Range is the source range to highlight
/// if there is an error.
bool
Sema::CheckDerivedToBaseConversion(QualType Derived, QualType Base,
                                   unsigned InaccessibleBaseID,
                                   unsigned AmbigiousBaseConvID,
                                   SourceLocation Loc, SourceRange Range,
                                   DeclarationName Name) {
  // First, determine whether the path from Derived to Base is
  // ambiguous. This is slightly more expensive than checking whether
  // the Derived to Base conversion exists, because here we need to
  // explore multiple paths to determine if there is an ambiguity.
  BasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                  /*DetectVirtual=*/false);
  bool DerivationOkay = IsDerivedFrom(Derived, Base, Paths);
  assert(DerivationOkay &&
         "Can only be used with a derived-to-base conversion");
  (void)DerivationOkay;

  if (!Paths.isAmbiguous(Context.getCanonicalType(Base).getUnqualifiedType())) {
    // Check that the base class can be accessed.
    return CheckBaseClassAccess(Derived, Base, InaccessibleBaseID, Paths, Loc,
                                Name);
  }

  // We know that the derived-to-base conversion is ambiguous, and
  // we're going to produce a diagnostic. Perform the derived-to-base
  // search just one more time to compute all of the possible paths so
  // that we can print them out. This is more expensive than any of
  // the previous derived-to-base checks we've done, but at this point
  // performance isn't as much of an issue.
  Paths.clear();
  Paths.setRecordingPaths(true);
  bool StillOkay = IsDerivedFrom(Derived, Base, Paths);
  assert(StillOkay && "Can only be used with a derived-to-base conversion");
  (void)StillOkay;

  // Build up a textual representation of the ambiguous paths, e.g.,
  // D -> B -> A, that will be used to illustrate the ambiguous
  // conversions in the diagnostic. We only print one of the paths
  // to each base class subobject.
  std::string PathDisplayStr = getAmbiguousPathsDisplayString(Paths);

  Diag(Loc, AmbigiousBaseConvID)
    << Derived << Base << PathDisplayStr << Range << Name;
  return true;
}

bool
Sema::CheckDerivedToBaseConversion(QualType Derived, QualType Base,
                                   SourceLocation Loc, SourceRange Range) {
  return CheckDerivedToBaseConversion(Derived, Base,
                                      diag::err_conv_to_inaccessible_base,
                                      diag::err_ambiguous_derived_to_base_conv,
                                      Loc, Range, DeclarationName());
}


/// @brief Builds a string representing ambiguous paths from a
/// specific derived class to different subobjects of the same base
/// class.
///
/// This function builds a string that can be used in error messages
/// to show the different paths that one can take through the
/// inheritance hierarchy to go from the derived class to different
/// subobjects of a base class. The result looks something like this:
/// @code
/// struct D -> struct B -> struct A
/// struct D -> struct C -> struct A
/// @endcode
std::string Sema::getAmbiguousPathsDisplayString(BasePaths &Paths) {
  std::string PathDisplayStr;
  std::set<unsigned> DisplayedPaths;
  for (BasePaths::paths_iterator Path = Paths.begin();
       Path != Paths.end(); ++Path) {
    if (DisplayedPaths.insert(Path->back().SubobjectNumber).second) {
      // We haven't displayed a path to this particular base
      // class subobject yet.
      PathDisplayStr += "\n    ";
      PathDisplayStr += Paths.getOrigin().getAsString();
      for (BasePath::const_iterator Element = Path->begin();
           Element != Path->end(); ++Element)
        PathDisplayStr += " -> " + Element->Base->getType().getAsString();
    }
  }

  return PathDisplayStr;
}
