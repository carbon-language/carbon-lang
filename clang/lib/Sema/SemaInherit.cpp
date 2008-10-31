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
// including searching the inheritance hierarchy and (eventually)
// access checking.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "SemaInherit.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeOrdering.h"
#include "clang/Basic/Diagnostic.h"
#include <memory>
#include <set>
#include <string>

using namespace clang;

/// isAmbiguous - Determines whether the set of paths provided is
/// ambiguous, i.e., there are two or more paths that refer to
/// different base class subobjects of the same type. BaseType must be
/// an unqualified, canonical class type.
bool BasePaths::isAmbiguous(QualType BaseType) {
  assert(BaseType->isCanonical() && "Base type must be the canonical type");
  assert(BaseType.getCVRQualifiers() == 0 && "Base type must be unqualified");
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
  bool FoundPath = false;

  Derived = Context.getCanonicalType(Derived).getUnqualifiedType();
  Base = Context.getCanonicalType(Base).getUnqualifiedType();

  if (!Derived->isRecordType() || !Base->isRecordType())
    return false;

  if (Derived == Base)
    return false;

  if (const RecordType *DerivedType = Derived->getAsRecordType()) {
    const CXXRecordDecl *Decl 
      = static_cast<const CXXRecordDecl *>(DerivedType->getDecl());
    for (CXXRecordDecl::base_class_const_iterator BaseSpec = Decl->bases_begin();
         BaseSpec != Decl->bases_end(); ++BaseSpec) {
      // Find the record of the base class subobjects for this type.
      QualType BaseType = Context.getCanonicalType(BaseSpec->getType());
      BaseType = BaseType.getUnqualifiedType();
      
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
          Paths.DetectedVirtual = static_cast<const CXXRecordType*>(
            BaseType->getAsRecordType());
          SetVirtual = true;
        }
      } else
        ++Subobjects.second;

      if (Paths.isRecordingPaths()) {
        // Add this base specifier to the current path.
        BasePathElement Element;
        Element.Base = &*BaseSpec;
        if (BaseSpec->isVirtual())
          Element.SubobjectNumber = 0;
        else
          Element.SubobjectNumber = Subobjects.second;
        Paths.ScratchPath.push_back(Element);
      }

      if (Context.getCanonicalType(BaseSpec->getType()) == Base) {
        // We've found the base we're looking for.
        FoundPath = true;
        if (Paths.isRecordingPaths()) {
          // We have a path. Make a copy of it before moving on.
          Paths.Paths.push_back(Paths.ScratchPath);
        } else if (!Paths.isFindingAmbiguities()) {
          // We found a path and we don't care about ambiguities;
          // return immediately.
          return FoundPath;
        }
      } else if (VisitBase && IsDerivedFrom(BaseSpec->getType(), Base, Paths)) {
        // There is a path to the base we want. If we're not
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
  }

  return FoundPath;
}

/// CheckDerivedToBaseConversion - Check whether the Derived-to-Base
/// conversion (where Derived and Base are class types) is
/// well-formed, meaning that the conversion is unambiguous (and
/// FIXME: that all of the base classes are accessible). Returns true
/// and emits a diagnostic if the code is ill-formed, returns false
/// otherwise. Loc is the location where this routine should point to
/// if there is an error, and Range is the source range to highlight
/// if there is an error.
bool 
Sema::CheckDerivedToBaseConversion(QualType Derived, QualType Base,
                                   SourceLocation Loc, SourceRange Range) {
  // First, determine whether the path from Derived to Base is
  // ambiguous. This is slightly more expensive than checking whether
  // the Derived to Base conversion exists, because here we need to
  // explore multiple paths to determine if there is an ambiguity.
  BasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/false,
                  /*DetectVirtual=*/false);
  bool DerivationOkay = IsDerivedFrom(Derived, Base, Paths);
  assert(DerivationOkay && "Can only be used with a derived-to-base conversion");
  if (!DerivationOkay)
    return true;

  if (!Paths.isAmbiguous(Context.getCanonicalType(Base).getUnqualifiedType()))
    return false;

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
  if (!StillOkay)
    return true;
  
  // Build up a textual representation of the ambiguous paths, e.g.,
  // D -> B -> A, that will be used to illustrate the ambiguous
  // conversions in the diagnostic. We only print one of the paths
  // to each base class subobject.
  std::string PathDisplayStr;
  std::set<unsigned> DisplayedPaths;
  for (BasePaths::paths_iterator Path = Paths.begin(); 
       Path != Paths.end(); ++Path) {
    if (DisplayedPaths.insert(Path->back().SubobjectNumber).second) {
      // We haven't displayed a path to this particular base
      // class subobject yet.
      PathDisplayStr += "\n    ";
      PathDisplayStr += Derived.getAsString();
      for (BasePath::const_iterator Element = Path->begin(); 
           Element != Path->end(); ++Element)
        PathDisplayStr += " -> " + Element->Base->getType().getAsString(); 
    }
  }
  
  Diag(Loc, diag::err_ambiguous_derived_to_base_conv,
       Derived.getAsString(), Base.getAsString(), PathDisplayStr, Range);
  return true;
}

