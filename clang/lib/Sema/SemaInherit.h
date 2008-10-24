//===------ SemaInherit.h - C++ Inheritance ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Sema data structures that help analyse C++
// inheritance semantics, including searching the inheritance
// hierarchy.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_INHERIT_H
#define LLVM_CLANG_SEMA_INHERIT_H

#include "clang/AST/Type.h"
#include "clang/AST/TypeOrdering.h"
#include "llvm/ADT/SmallVector.h"
#include <list>
#include <map>

namespace clang {
  class Sema;
  class CXXBaseSpecifier;

  /// BasePathElement - An element in a path from a derived class to a
  /// base class. Each step in the path references the link from a
  /// derived class to one of its direct base classes, along with a
  /// base "number" that identifies which base subobject of the
  /// original derived class we are referencing.
  struct BasePathElement {
    /// Base - The base specifier that states the link from a derived
    /// class to a base class, which will be followed by this base
    /// path element.
    const CXXBaseSpecifier *Base;

    /// SubobjectNumber - Identifies which base class subobject (of type
    /// @c Base->getType()) this base path element refers to. This 
    /// value is only valid if @c !Base->isVirtual(), because there
    /// is no base numbering for the zero or one virtual bases of a 
    /// given type.
    int SubobjectNumber;
  };

  /// BasePath - Represents a path from a specific derived class
  /// (which is not represented as part of the path) to a particular
  /// (direct or indirect) base class subobject. Individual elements
  /// in the path are described by the BasePathElement structure,
  /// which captures both the link from a derived class to one of its
  /// direct bases and identification describing which base class
  /// subobject is being used. 
  typedef llvm::SmallVector<BasePathElement, 4> BasePath;

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
  class BasePaths {
    /// Paths - The actual set of paths that can be taken from the
    /// derived class to the same base class.
    std::list<BasePath> Paths;

    /// ClassSubobjects - Records the class subobjects for each class
    /// type that we've seen. The first element in the pair says
    /// whether we found a path to a virtual base for that class type,
    /// while the element contains the number of non-virtual base
    /// class subobjects for that class type. The key of the map is
    /// the cv-unqualified canonical type of the base class subobject.
    std::map<QualType, std::pair<bool, unsigned>, QualTypeOrdering> 
      ClassSubobjects;

    /// FindAmbiguities - Whether Sema::IsDirectedFrom should try find
    /// ambiguous paths while it is looking for a path from a derived
    /// type to a base type.
    bool FindAmbiguities;

    /// RecordPaths - Whether Sema::IsDirectedFrom should record paths
    /// while it is determining whether there are paths from a derived
    /// type to a base type.
    bool RecordPaths;

    /// ScratchPath - A BasePath that is used by Sema::IsDerivedFrom
    /// to help build the set of paths.
    BasePath ScratchPath;

    friend class Sema;

  public:
    typedef std::list<BasePath>::const_iterator paths_iterator;
    
    /// BasePaths - Construct a new BasePaths structure to record the
    /// paths for a derived-to-base search.
    explicit BasePaths(bool FindAmbiguities = true, bool RecordPaths = true) 
      : FindAmbiguities(FindAmbiguities), RecordPaths(RecordPaths) { }

    paths_iterator begin() const { return Paths.begin(); }
    paths_iterator end()   const { return Paths.end(); }

    bool isAmbiguous(QualType BaseType);

    /// isFindingAmbiguities - Whether we are finding multiple paths
    /// to detect ambiguities.
    bool isFindingAmbiguities() const { return FindAmbiguities; }

    /// isRecordingPaths - Whether we are recording paths.
    bool isRecordingPaths() const { return RecordPaths; }

    /// setRecordingPaths - Specify whether we should be recording
    /// paths or not.
    void setRecordingPaths(bool RP) { RecordPaths = RP; }

    void clear();
  };
}

#endif
