//===------ CXXInheritance.cpp - C++ Inheritance ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides routines that help analyzing C++ inheritance hierarchies.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclCXX.h"
#include <algorithm>
#include <set>

using namespace clang;

/// \brief Computes the set of declarations referenced by these base
/// paths.
void CXXBasePaths::ComputeDeclsFound() {
  assert(NumDeclsFound == 0 && !DeclsFound &&
         "Already computed the set of declarations");
  
  std::set<NamedDecl *> Decls;
  for (CXXBasePaths::paths_iterator Path = begin(), PathEnd = end();
       Path != PathEnd; ++Path)
    Decls.insert(*Path->Decls.first);
  
  NumDeclsFound = Decls.size();
  DeclsFound = new NamedDecl * [NumDeclsFound];
  std::copy(Decls.begin(), Decls.end(), DeclsFound);
}

CXXBasePaths::decl_iterator CXXBasePaths::found_decls_begin() {
  if (NumDeclsFound == 0)
    ComputeDeclsFound();
  return DeclsFound;
}

CXXBasePaths::decl_iterator CXXBasePaths::found_decls_end() {
  if (NumDeclsFound == 0)
    ComputeDeclsFound();
  return DeclsFound + NumDeclsFound;
}

/// isAmbiguous - Determines whether the set of paths provided is
/// ambiguous, i.e., there are two or more paths that refer to
/// different base class subobjects of the same type. BaseType must be
/// an unqualified, canonical class type.
bool CXXBasePaths::isAmbiguous(QualType BaseType) {
  assert(BaseType.isCanonical() && "Base type must be the canonical type");
  assert(BaseType.hasQualifiers() == 0 && "Base type must be unqualified");
  std::pair<bool, unsigned>& Subobjects = ClassSubobjects[BaseType];
  return Subobjects.second + (Subobjects.first? 1 : 0) > 1;
}

/// clear - Clear out all prior path information.
void CXXBasePaths::clear() {
  Paths.clear();
  ClassSubobjects.clear();
  ScratchPath.clear();
  DetectedVirtual = 0;
}

/// @brief Swaps the contents of this CXXBasePaths structure with the
/// contents of Other.
void CXXBasePaths::swap(CXXBasePaths &Other) {
  std::swap(Origin, Other.Origin);
  Paths.swap(Other.Paths);
  ClassSubobjects.swap(Other.ClassSubobjects);
  std::swap(FindAmbiguities, Other.FindAmbiguities);
  std::swap(RecordPaths, Other.RecordPaths);
  std::swap(DetectVirtual, Other.DetectVirtual);
  std::swap(DetectedVirtual, Other.DetectedVirtual);
}

bool CXXRecordDecl::isDerivedFrom(CXXRecordDecl *Base) const {
  CXXBasePaths Paths(/*FindAmbiguities=*/false, /*RecordPaths=*/false,
                     /*DetectVirtual=*/false);
  return isDerivedFrom(Base, Paths);
}

bool CXXRecordDecl::isDerivedFrom(CXXRecordDecl *Base, CXXBasePaths &Paths) const {
  if (getCanonicalDecl() == Base->getCanonicalDecl())
    return false;
  
  Paths.setOrigin(const_cast<CXXRecordDecl*>(this));
  return lookupInBases(&FindBaseClass, Base->getCanonicalDecl(), Paths);
}

bool CXXRecordDecl::lookupInBases(BaseMatchesCallback *BaseMatches,
                                  void *UserData,
                                  CXXBasePaths &Paths) const {
  bool FoundPath = false;
  
  ASTContext &Context = getASTContext();
  for (base_class_const_iterator BaseSpec = bases_begin(),
         BaseSpecEnd = bases_end(); BaseSpec != BaseSpecEnd; ++BaseSpec) {
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
      CXXBasePathElement Element;
      Element.Base = &*BaseSpec;
      Element.Class = this;
      if (BaseSpec->isVirtual())
        Element.SubobjectNumber = 0;
      else
        Element.SubobjectNumber = Subobjects.second;
      Paths.ScratchPath.push_back(Element);
    }
        
    if (BaseMatches(BaseSpec, Paths.ScratchPath, UserData)) {
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
    } else if (VisitBase) {
      CXXRecordDecl *BaseRecord
        = cast<CXXRecordDecl>(BaseSpec->getType()->getAs<RecordType>()
                                ->getDecl());
      if (BaseRecord->lookupInBases(BaseMatches, UserData, Paths)) {
        // C++ [class.member.lookup]p2:
        //   A member name f in one sub-object B hides a member name f in
        //   a sub-object A if A is a base class sub-object of B. Any
        //   declarations that are so hidden are eliminated from
        //   consideration.
        
        // There is a path to a base class that meets the criteria. If we're 
        // not collecting paths or finding ambiguities, we're done.
        FoundPath = true;
        if (!Paths.isFindingAmbiguities())
          return FoundPath;
      }
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

bool CXXRecordDecl::FindBaseClass(const CXXBaseSpecifier *Specifier, 
                                  CXXBasePath &Path,
                                  void *BaseRecord) {
  assert(((Decl *)BaseRecord)->getCanonicalDecl() == BaseRecord &&
         "User data for FindBaseClass is not canonical!");
  return Specifier->getType()->getAs<RecordType>()->getDecl()
           ->getCanonicalDecl() == BaseRecord;
}

bool CXXRecordDecl::FindTagMember(const CXXBaseSpecifier *Specifier, 
                                  CXXBasePath &Path,
                                  void *Name) {
  RecordDecl *BaseRecord = Specifier->getType()->getAs<RecordType>()->getDecl();

  DeclarationName N = DeclarationName::getFromOpaquePtr(Name);
  for (Path.Decls = BaseRecord->lookup(N);
       Path.Decls.first != Path.Decls.second;
       ++Path.Decls.first) {
    if ((*Path.Decls.first)->isInIdentifierNamespace(IDNS_Tag))
      return true;
  }

  return false;
}

bool CXXRecordDecl::FindOrdinaryMember(const CXXBaseSpecifier *Specifier, 
                                       CXXBasePath &Path,
                                       void *Name) {
  RecordDecl *BaseRecord = Specifier->getType()->getAs<RecordType>()->getDecl();
  
  const unsigned IDNS = IDNS_Ordinary | IDNS_Tag | IDNS_Member;
  DeclarationName N = DeclarationName::getFromOpaquePtr(Name);
  for (Path.Decls = BaseRecord->lookup(N);
       Path.Decls.first != Path.Decls.second;
       ++Path.Decls.first) {
    if ((*Path.Decls.first)->isInIdentifierNamespace(IDNS))
      return true;
  }
  
  return false;
}

bool CXXRecordDecl::
FindNestedNameSpecifierMember(const CXXBaseSpecifier *Specifier, 
                              CXXBasePath &Path,
                              void *Name) {
  RecordDecl *BaseRecord = Specifier->getType()->getAs<RecordType>()->getDecl();
  
  DeclarationName N = DeclarationName::getFromOpaquePtr(Name);
  for (Path.Decls = BaseRecord->lookup(N);
       Path.Decls.first != Path.Decls.second;
       ++Path.Decls.first) {
    // FIXME: Refactor the "is it a nested-name-specifier?" check
    if (isa<TypedefDecl>(*Path.Decls.first) ||
        (*Path.Decls.first)->isInIdentifierNamespace(IDNS_Tag))
      return true;
  }
  
  return false;
}
