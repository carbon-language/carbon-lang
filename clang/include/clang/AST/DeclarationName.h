//===-- DeclarationName.h - Representation of declaration names -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the DeclarationName and DeclarationNameTable classes.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_DECLARATIONNAME_H
#define LLVM_CLANG_AST_DECLARATIONNAME_H

#include "clang/Basic/IdentifierTable.h"
#include "clang/AST/Type.h"

namespace llvm {
  template <typename T> struct DenseMapInfo;
}

namespace clang {
  class CXXSpecialName;
  class CXXOperatorIdName;
  class DeclarationNameExtra;
  class IdentifierInfo;
  class MultiKeywordSelector;
  class UsingDirectiveDecl;

/// DeclarationName - The name of a declaration. In the common case,
/// this just stores an IdentifierInfo pointer to a normal
/// name. However, it also provides encodings for Objective-C
/// selectors (optimizing zero- and one-argument selectors, which make
/// up 78% percent of all selectors in Cocoa.h) and special C++ names
/// for constructors, destructors, and conversion functions.
class DeclarationName {
public:
  /// NameKind - The kind of name this object contains.
  enum NameKind {
    Identifier,
    ObjCZeroArgSelector,
    ObjCOneArgSelector,
    ObjCMultiArgSelector,
    CXXConstructorName,
    CXXDestructorName,
    CXXConversionFunctionName,
    CXXOperatorName,
    CXXUsingDirective
  };

private:
  /// StoredNameKind - The kind of name that is actually stored in the
  /// upper bits of the Ptr field. This is only used internally.
  enum StoredNameKind {
    StoredIdentifier = 0,
    StoredObjCZeroArgSelector,
    StoredObjCOneArgSelector,
    StoredDeclarationNameExtra,
    PtrMask = 0x03
  };

  /// Ptr - The lowest two bits are used to express what kind of name
  /// we're actually storing, using the values of NameKind. Depending
  /// on the kind of name this is, the upper bits of Ptr may have one
  /// of several different meanings:
  ///
  ///   StoredIdentifier - The name is a normal identifier, and Ptr is
  ///   a normal IdentifierInfo pointer.
  ///
  ///   StoredObjCZeroArgSelector - The name is an Objective-C
  ///   selector with zero arguments, and Ptr is an IdentifierInfo
  ///   pointer pointing to the selector name.
  ///
  ///   StoredObjCOneArgSelector - The name is an Objective-C selector
  ///   with one argument, and Ptr is an IdentifierInfo pointer
  ///   pointing to the selector name.
  ///
  ///   StoredDeclarationNameExtra - Ptr is actually a pointer to a
  ///   DeclarationNameExtra structure, whose first value will tell us
  ///   whether this is an Objective-C selector, C++ operator-id name,
  ///   or special C++ name.
  uintptr_t Ptr;

  /// getStoredNameKind - Return the kind of object that is stored in
  /// Ptr.
  StoredNameKind getStoredNameKind() const {
    return static_cast<StoredNameKind>(Ptr & PtrMask);
  }

  /// getExtra - Get the "extra" information associated with this
  /// multi-argument selector or C++ special name.
  DeclarationNameExtra *getExtra() const {
    assert(getStoredNameKind() == StoredDeclarationNameExtra &&
           "Declaration name does not store an Extra structure");
    return reinterpret_cast<DeclarationNameExtra *>(Ptr & ~PtrMask);
  }

  /// getAsCXXSpecialName - If the stored pointer is actually a
  /// CXXSpecialName, returns a pointer to it. Otherwise, returns
  /// a NULL pointer.
  CXXSpecialName *getAsCXXSpecialName() const {
    if (getNameKind() >= CXXConstructorName && 
        getNameKind() <= CXXConversionFunctionName)
      return reinterpret_cast<CXXSpecialName *>(Ptr & ~PtrMask);
    return 0;
  }

  /// getAsCXXOperatorIdName
  CXXOperatorIdName *getAsCXXOperatorIdName() const {
    if (getNameKind() == CXXOperatorName)
      return reinterpret_cast<CXXOperatorIdName *>(Ptr & ~PtrMask);
    return 0;
  }

  // Construct a declaration name from the name of a C++ constructor,
  // destructor, or conversion function.
  DeclarationName(CXXSpecialName *Name) 
    : Ptr(reinterpret_cast<uintptr_t>(Name)) { 
    assert((Ptr & PtrMask) == 0 && "Improperly aligned CXXSpecialName");
    Ptr |= StoredDeclarationNameExtra;
  }

  // Construct a declaration name from the name of a C++ overloaded
  // operator.
  DeclarationName(CXXOperatorIdName *Name) 
    : Ptr(reinterpret_cast<uintptr_t>(Name)) { 
    assert((Ptr & PtrMask) == 0 && "Improperly aligned CXXOperatorId");
    Ptr |= StoredDeclarationNameExtra;
  }

  /// Construct a declaration name from a raw pointer.
  DeclarationName(uintptr_t Ptr) : Ptr(Ptr) { }

  friend class DeclarationNameTable;
  friend class NamedDecl;

  /// getFETokenInfoAsVoid - Retrieves the front end-specified pointer
  /// for this name as a void pointer.
  void *getFETokenInfoAsVoid() const;

public:
  /// DeclarationName - Used to create an empty selector.
  DeclarationName() : Ptr(0) { }

  // Construct a declaration name from an IdentifierInfo *.
  DeclarationName(const IdentifierInfo *II) 
    : Ptr(reinterpret_cast<uintptr_t>(II)) { 
    assert((Ptr & PtrMask) == 0 && "Improperly aligned IdentifierInfo");
  }

  // Construct a declaration name from an Objective-C selector.
  DeclarationName(Selector Sel);

  /// getUsingDirectiveName - Return name for all using-directives.
  static DeclarationName getUsingDirectiveName();

  // operator bool() - Evaluates true when this declaration name is
  // non-empty.
  operator bool() const { 
    return ((Ptr & PtrMask) != 0) || 
           (reinterpret_cast<IdentifierInfo *>(Ptr & ~PtrMask));
  }

  /// Predicate functions for querying what type of name this is.
  bool isIdentifier() const { return getStoredNameKind() == StoredIdentifier; }
  bool isObjCZeroArgSelector() const {
    return getStoredNameKind() == StoredObjCZeroArgSelector;
  }
  bool isObjCOneArgSelector() const {
    return getStoredNameKind() == StoredObjCOneArgSelector;
  }
  
  /// getNameKind - Determine what kind of name this is.
  NameKind getNameKind() const;
  

  /// getName - Retrieve the human-readable string for this name.
  std::string getAsString() const;

  /// getAsIdentifierInfo - Retrieve the IdentifierInfo * stored in
  /// this declaration name, or NULL if this declaration name isn't a
  /// simple identifier.
  IdentifierInfo *getAsIdentifierInfo() const { 
    if (isIdentifier())
      return reinterpret_cast<IdentifierInfo *>(Ptr);
    return 0;
  }

  /// getAsOpaqueInteger - Get the representation of this declaration
  /// name as an opaque integer.
  uintptr_t getAsOpaqueInteger() const { return Ptr; }

  /// getAsOpaquePtr - Get the representation of this declaration name as
  /// an opaque pointer.
  void *getAsOpaquePtr() const { return reinterpret_cast<void*>(Ptr); }

  static DeclarationName getFromOpaqueInteger(uintptr_t P) {
    DeclarationName N;
    N.Ptr = P;
    return N;
  }
  
  /// getCXXNameType - If this name is one of the C++ names (of a
  /// constructor, destructor, or conversion function), return the
  /// type associated with that name.
  QualType getCXXNameType() const;

  /// getCXXOverloadedOperator - If this name is the name of an
  /// overloadable operator in C++ (e.g., @c operator+), retrieve the
  /// kind of overloaded operator.
  OverloadedOperatorKind getCXXOverloadedOperator() const;

  /// getObjCSelector - Get the Objective-C selector stored in this
  /// declaration name.
  Selector getObjCSelector() const;

  /// getFETokenInfo/setFETokenInfo - The language front-end is
  /// allowed to associate arbitrary metadata with some kinds of
  /// declaration names, including normal identifiers and C++
  /// constructors, destructors, and conversion functions.
  template<typename T>
  T *getFETokenInfo() const { return static_cast<T*>(getFETokenInfoAsVoid()); }

  void setFETokenInfo(void *T);

  /// operator== - Determine whether the specified names are identical..
  friend bool operator==(DeclarationName LHS, DeclarationName RHS) {
    return LHS.Ptr == RHS.Ptr;
  }

  /// operator!= - Determine whether the specified names are different.
  friend bool operator!=(DeclarationName LHS, DeclarationName RHS) {
    return LHS.Ptr != RHS.Ptr;
  }

  static DeclarationName getEmptyMarker() {
    return DeclarationName(uintptr_t(-1));
  }

  static DeclarationName getTombstoneMarker() {
    return DeclarationName(uintptr_t(-2));
  }
};

/// Ordering on two declaration names. If both names are identifiers,
/// this provides a lexicographical ordering.
bool operator<(DeclarationName LHS, DeclarationName RHS);

/// Ordering on two declaration names. If both names are identifiers,
/// this provides a lexicographical ordering.
inline bool operator>(DeclarationName LHS, DeclarationName RHS) {
  return RHS < LHS;
}

/// Ordering on two declaration names. If both names are identifiers,
/// this provides a lexicographical ordering.
inline bool operator<=(DeclarationName LHS, DeclarationName RHS) {
  return !(RHS < LHS);
}

/// Ordering on two declaration names. If both names are identifiers,
/// this provides a lexicographical ordering.
inline bool operator>=(DeclarationName LHS, DeclarationName RHS) {
  return !(LHS < RHS);
}

/// DeclarationNameTable - Used to store and retrieve DeclarationName
/// instances for the various kinds of declaration names, e.g., normal
/// identifiers, C++ constructor names, etc. This class contains
/// uniqued versions of each of the C++ special names, which can be
/// retrieved using its member functions (e.g.,
/// getCXXConstructorName).
class DeclarationNameTable {
  void *CXXSpecialNamesImpl; // Actually a FoldingSet<CXXSpecialName> *
  CXXOperatorIdName *CXXOperatorNames; // Operator names

  DeclarationNameTable(const DeclarationNameTable&);            // NONCOPYABLE
  DeclarationNameTable& operator=(const DeclarationNameTable&); // NONCOPYABLE

public:
  DeclarationNameTable();
  ~DeclarationNameTable();

  /// getIdentifier - Create a declaration name that is a simple
  /// identifier.
  DeclarationName getIdentifier(IdentifierInfo *ID) {
    return DeclarationName(ID);
  }

  /// getCXXConstructorName - Returns the name of a C++ constructor
  /// for the given Type.
  DeclarationName getCXXConstructorName(QualType Ty) {
    return getCXXSpecialName(DeclarationName::CXXConstructorName, Ty);
  }

  /// getCXXDestructorName - Returns the name of a C++ destructor
  /// for the given Type.
  DeclarationName getCXXDestructorName(QualType Ty) {
    return getCXXSpecialName(DeclarationName::CXXDestructorName, Ty);
  }

  /// getCXXConversionFunctionName - Returns the name of a C++
  /// conversion function for the given Type.
  DeclarationName getCXXConversionFunctionName(QualType Ty) {
    return getCXXSpecialName(DeclarationName::CXXConversionFunctionName, Ty);
  }

  /// getCXXSpecialName - Returns a declaration name for special kind
  /// of C++ name, e.g., for a constructor, destructor, or conversion
  /// function.
  DeclarationName getCXXSpecialName(DeclarationName::NameKind Kind, 
                                    QualType Ty);

  /// getCXXOperatorName - Get the name of the overloadable C++
  /// operator corresponding to Op.
  DeclarationName getCXXOperatorName(OverloadedOperatorKind Op);
};  

/// Insertion operator for diagnostics.  This allows sending DeclarationName's
/// into a diagnostic with <<.
inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           DeclarationName N) {
  DB.AddTaggedVal(N.getAsOpaqueInteger(),
                  Diagnostic::ak_declarationname);
  return DB;
}
  
  
}  // end namespace clang

namespace llvm {
/// Define DenseMapInfo so that DeclarationNames can be used as keys
/// in DenseMap and DenseSets.
template<>
struct DenseMapInfo<clang::DeclarationName> {
  static inline clang::DeclarationName getEmptyKey() {
    return clang::DeclarationName::getEmptyMarker();
  }

  static inline clang::DeclarationName getTombstoneKey() {
    return clang::DeclarationName::getTombstoneMarker();
  }

  static unsigned getHashValue(clang::DeclarationName);

  static inline bool 
  isEqual(clang::DeclarationName LHS, clang::DeclarationName RHS) {
    return LHS == RHS;
  }

  static inline bool isPod() { return true; }
};

}  // end namespace llvm

#endif
