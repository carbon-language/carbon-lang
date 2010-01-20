//===------ CXXInheritance.h - C++ Inheritance ------------------*- C++ -*-===//
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

#ifndef LLVM_CLANG_AST_CXXINHERITANCE_H
#define LLVM_CLANG_AST_CXXINHERITANCE_H

#include "clang/AST/DeclarationName.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeOrdering.h"
#include "llvm/ADT/SmallVector.h"
#include <list>
#include <map>
#include <cassert>

namespace clang {
  
class CXXBaseSpecifier;
class CXXMethodDecl;
class CXXRecordDecl;
class NamedDecl;
  
/// \brief Represents an element in a path from a derived class to a
/// base class. 
/// 
/// Each step in the path references the link from a
/// derived class to one of its direct base classes, along with a
/// base "number" that identifies which base subobject of the
/// original derived class we are referencing.
struct CXXBasePathElement {
  /// \brief The base specifier that states the link from a derived
  /// class to a base class, which will be followed by this base
  /// path element.
  const CXXBaseSpecifier *Base;
  
  /// \brief The record decl of the class that the base is a base of.
  const CXXRecordDecl *Class;
  
  /// \brief Identifies which base class subobject (of type
  /// \c Base->getType()) this base path element refers to. 
  ///
  /// This value is only valid if \c !Base->isVirtual(), because there
  /// is no base numbering for the zero or one virtual bases of a
  /// given type.
  int SubobjectNumber;
};

/// \brief Represents a path from a specific derived class
/// (which is not represented as part of the path) to a particular
/// (direct or indirect) base class subobject.
///
/// Individual elements in the path are described by the \c CXXBasePathElement 
/// structure, which captures both the link from a derived class to one of its
/// direct bases and identification describing which base class
/// subobject is being used.
class CXXBasePath : public llvm::SmallVector<CXXBasePathElement, 4> {
public:
  /// \brief The access along this inheritance path.
  AccessSpecifier Access;

  /// \brief The set of declarations found inside this base class
  /// subobject.
  DeclContext::lookup_result Decls;
};

/// BasePaths - Represents the set of paths from a derived class to
/// one of its (direct or indirect) bases. For example, given the
/// following class hierachy:
///
/// @code
/// class A { };
/// class B : public A { };
/// class C : public A { };
/// class D : public B, public C{ };
/// @endcode
///
/// There are two potential BasePaths to represent paths from D to a
/// base subobject of type A. One path is (D,0) -> (B,0) -> (A,0)
/// and another is (D,0)->(C,0)->(A,1). These two paths actually
/// refer to two different base class subobjects of the same type,
/// so the BasePaths object refers to an ambiguous path. On the
/// other hand, consider the following class hierarchy:
///
/// @code
/// class A { };
/// class B : public virtual A { };
/// class C : public virtual A { };
/// class D : public B, public C{ };
/// @endcode
///
/// Here, there are two potential BasePaths again, (D, 0) -> (B, 0)
/// -> (A,v) and (D, 0) -> (C, 0) -> (A, v), but since both of them
/// refer to the same base class subobject of type A (the virtual
/// one), there is no ambiguity.
class CXXBasePaths {
  /// \brief The type from which this search originated.
  CXXRecordDecl *Origin;
  
  /// Paths - The actual set of paths that can be taken from the
  /// derived class to the same base class.
  std::list<CXXBasePath> Paths;
  
  /// ClassSubobjects - Records the class subobjects for each class
  /// type that we've seen. The first element in the pair says
  /// whether we found a path to a virtual base for that class type,
  /// while the element contains the number of non-virtual base
  /// class subobjects for that class type. The key of the map is
  /// the cv-unqualified canonical type of the base class subobject.
  std::map<QualType, std::pair<bool, unsigned>, QualTypeOrdering>
    ClassSubobjects;
  
  /// FindAmbiguities - Whether Sema::IsDerivedFrom should try find
  /// ambiguous paths while it is looking for a path from a derived
  /// type to a base type.
  bool FindAmbiguities;
  
  /// RecordPaths - Whether Sema::IsDerivedFrom should record paths
  /// while it is determining whether there are paths from a derived
  /// type to a base type.
  bool RecordPaths;
  
  /// DetectVirtual - Whether Sema::IsDerivedFrom should abort the search
  /// if it finds a path that goes across a virtual base. The virtual class
  /// is also recorded.
  bool DetectVirtual;
  
  /// ScratchPath - A BasePath that is used by Sema::lookupInBases
  /// to help build the set of paths.
  CXXBasePath ScratchPath;

  /// ScratchAccess - A stack of accessibility annotations used by
  /// Sema::lookupInBases.
  llvm::SmallVector<AccessSpecifier, 4> ScratchAccess;
  
  /// DetectedVirtual - The base class that is virtual.
  const RecordType *DetectedVirtual;
  
  /// \brief Array of the declarations that have been found. This
  /// array is constructed only if needed, e.g., to iterate over the
  /// results within LookupResult.
  NamedDecl **DeclsFound;
  unsigned NumDeclsFound;
  
  friend class CXXRecordDecl;
  
  void ComputeDeclsFound();
  
public:
  typedef std::list<CXXBasePath>::const_iterator paths_iterator;
  typedef NamedDecl **decl_iterator;
  
  /// BasePaths - Construct a new BasePaths structure to record the
  /// paths for a derived-to-base search.
  explicit CXXBasePaths(bool FindAmbiguities = true,
                        bool RecordPaths = true,
                        bool DetectVirtual = true)
    : FindAmbiguities(FindAmbiguities), RecordPaths(RecordPaths),
      DetectVirtual(DetectVirtual), DetectedVirtual(0), DeclsFound(0),
      NumDeclsFound(0) { }
  
  ~CXXBasePaths() { delete [] DeclsFound; }
  
  paths_iterator begin() const { return Paths.begin(); }
  paths_iterator end()   const { return Paths.end(); }
  
  CXXBasePath&       front()       { return Paths.front(); }
  const CXXBasePath& front() const { return Paths.front(); }
  
  decl_iterator found_decls_begin();
  decl_iterator found_decls_end();
  
  /// \brief Determine whether the path from the most-derived type to the
  /// given base type is ambiguous (i.e., it refers to multiple subobjects of
  /// the same base type).
  bool isAmbiguous(QualType BaseType);
  
  /// \brief Whether we are finding multiple paths to detect ambiguities.
  bool isFindingAmbiguities() const { return FindAmbiguities; }
  
  /// \brief Whether we are recording paths.
  bool isRecordingPaths() const { return RecordPaths; }
  
  /// \brief Specify whether we should be recording paths or not.
  void setRecordingPaths(bool RP) { RecordPaths = RP; }
  
  /// \brief Whether we are detecting virtual bases.
  bool isDetectingVirtual() const { return DetectVirtual; }
  
  /// \brief The virtual base discovered on the path (if we are merely
  /// detecting virtuals).
  const RecordType* getDetectedVirtual() const {
    return DetectedVirtual;
  }
  
  /// \brief Retrieve the type from which this base-paths search
  /// began
  CXXRecordDecl *getOrigin() const { return Origin; }
  void setOrigin(CXXRecordDecl *Rec) { Origin = Rec; }
  
  /// \brief Clear the base-paths results.
  void clear();
  
  /// \brief Swap this data structure's contents with another CXXBasePaths 
  /// object.
  void swap(CXXBasePaths &Other);
};
  
} // end namespace clang

#endif
