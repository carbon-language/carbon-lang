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

#include "Sema.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeOrdering.h"
#include "llvm/ADT/SmallVector.h"
#include <list>
#include <map>

namespace clang {
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

    /// Class - The record decl of the class that the base is a base of.
    const CXXRecordDecl *Class;

    /// SubobjectNumber - Identifies which base class subobject (of type
    /// @c Base->getType()) this base path element refers to. This
    /// value is only valid if @c !Base->isVirtual(), because there
    /// is no base numbering for the zero or one virtual bases of a
    /// given type.
    int SubobjectNumber;
  };

  /// BasePath - Represents a path from a specific derived class
  /// (which is not represented as part of the path) to a particular
  /// (direct or indirect) base class subobject that contains some
  /// number of declarations with the same name. Individual elements
  /// in the path are described by the BasePathElement structure,
  /// which captures both the link from a derived class to one of its
  /// direct bases and identification describing which base class
  /// subobject is being used.
  struct BasePath : public llvm::SmallVector<BasePathElement, 4> {
    /// Decls - The set of declarations found inside this base class
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
  class BasePaths {
    /// Origin - The type from which this search originated.
    QualType Origin;

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

    /// ScratchPath - A BasePath that is used by Sema::IsDerivedFrom
    /// to help build the set of paths.
    BasePath ScratchPath;

    /// DetectedVirtual - The base class that is virtual.
    const RecordType *DetectedVirtual;

    /// \brief Array of the declarations that have been found. This
    /// array is constructed only if needed, e.g., to iterate over the
    /// results within LookupResult.
    NamedDecl **DeclsFound;
    unsigned NumDeclsFound;

    friend class Sema;

    void ComputeDeclsFound();

  public:
    typedef std::list<BasePath>::const_iterator paths_iterator;
    typedef NamedDecl **decl_iterator;

    /// BasePaths - Construct a new BasePaths structure to record the
    /// paths for a derived-to-base search.
    explicit BasePaths(bool FindAmbiguities = true,
                       bool RecordPaths = true,
                       bool DetectVirtual = true)
      : FindAmbiguities(FindAmbiguities), RecordPaths(RecordPaths),
        DetectVirtual(DetectVirtual), DetectedVirtual(0), DeclsFound(0),
        NumDeclsFound(0) { }

    ~BasePaths() { delete [] DeclsFound; }

    paths_iterator begin() const { return Paths.begin(); }
    paths_iterator end()   const { return Paths.end(); }

    BasePath&       front()       { return Paths.front(); }
    const BasePath& front() const { return Paths.front(); }

    decl_iterator found_decls_begin();
    decl_iterator found_decls_end();

    bool isAmbiguous(QualType BaseType);

    /// isFindingAmbiguities - Whether we are finding multiple paths
    /// to detect ambiguities.
    bool isFindingAmbiguities() const { return FindAmbiguities; }

    /// isRecordingPaths - Whether we are recording paths.
    bool isRecordingPaths() const { return RecordPaths; }

    /// setRecordingPaths - Specify whether we should be recording
    /// paths or not.
    void setRecordingPaths(bool RP) { RecordPaths = RP; }

    /// isDetectingVirtual - Whether we are detecting virtual bases.
    bool isDetectingVirtual() const { return DetectVirtual; }

    /// getDetectedVirtual - The virtual base discovered on the path.
    const RecordType* getDetectedVirtual() const {
      return DetectedVirtual;
    }

    /// @brief Retrieve the type from which this base-paths search
    /// began
    QualType getOrigin() const { return Origin; }
    void setOrigin(QualType Type) { Origin = Type; }

    void clear();

    void swap(BasePaths &Other);
  };

  /// MemberLookupCriteria - Criteria for performing lookup of a
  /// member of a C++ class. Objects of this type are used to direct
  /// Sema::LookupCXXClassMember.
  struct MemberLookupCriteria {
    /// LookupKind - the kind of lookup we're doing.
    enum LookupKind {
      LK_Base,
      LK_NamedMember,
      LK_OverriddenMember
    };

    /// MemberLookupCriteria - Constructs member lookup criteria to
    /// search for a base class of type Base.
    explicit MemberLookupCriteria(QualType Base)
      : Kind(LK_Base), Base(Base) { }

    /// MemberLookupCriteria - Constructs member lookup criteria to
    /// search for a class member with the given Name.
    explicit MemberLookupCriteria(DeclarationName Name,
                                  Sema::LookupNameKind NameKind,
                                  unsigned IDNS)
      : Kind(LK_NamedMember), Name(Name), NameKind(NameKind), IDNS(IDNS) { }

    explicit MemberLookupCriteria(CXXMethodDecl *MD)
      : Kind(LK_OverriddenMember), Method(MD) { }

    /// Kind - The kind of lookup we're doing.
    /// LK_Base if we are looking for a base class (whose
    /// type is Base). LK_NamedMember if we are looking for a named member of
    /// the class (with the name Name).
    LookupKind Kind;

    /// Base - The type of the base class we're searching for, if
    /// LookupBase is true.
    QualType Base;

    /// Name - The name of the member we're searching for, if
    /// LookupBase is false.
    DeclarationName Name;

    Sema::LookupNameKind NameKind;
    unsigned IDNS;

    CXXMethodDecl *Method;
  };
}

#endif
