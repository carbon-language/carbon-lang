//===-- DeclCXX.h - Classes for representing C++ declarations -*- C++ -*-=====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the C++ Decl subclasses, other than those for
//  templates (in DeclTemplate.h) and friends (in DeclFriend.h).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLCXX_H
#define LLVM_CLANG_AST_DECLCXX_H

#include "clang/AST/Expr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/UnresolvedSet.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace clang {

class ClassTemplateDecl;
class ClassTemplateSpecializationDecl;
class CXXBasePath;
class CXXBasePaths;
class CXXConstructorDecl;
class CXXConversionDecl;
class CXXDestructorDecl;
class CXXMethodDecl;
class CXXRecordDecl;
class CXXMemberLookupCriteria;
class CXXFinalOverriderMap;
class CXXIndirectPrimaryBaseSet;
class FriendDecl;
  
/// \brief Represents any kind of function declaration, whether it is a
/// concrete function or a function template.
class AnyFunctionDecl {
  NamedDecl *Function;

  AnyFunctionDecl(NamedDecl *ND) : Function(ND) { }

public:
  AnyFunctionDecl(FunctionDecl *FD) : Function(FD) { }
  AnyFunctionDecl(FunctionTemplateDecl *FTD);

  /// \brief Implicily converts any function or function template into a
  /// named declaration.
  operator NamedDecl *() const { return Function; }

  /// \brief Retrieve the underlying function or function template.
  NamedDecl *get() const { return Function; }

  static AnyFunctionDecl getFromNamedDecl(NamedDecl *ND) {
    return AnyFunctionDecl(ND);
  }
};

} // end namespace clang

namespace llvm {
  /// Implement simplify_type for AnyFunctionDecl, so that we can dyn_cast from
  /// AnyFunctionDecl to any function or function template declaration.
  template<> struct simplify_type<const ::clang::AnyFunctionDecl> {
    typedef ::clang::NamedDecl* SimpleType;
    static SimpleType getSimplifiedValue(const ::clang::AnyFunctionDecl &Val) {
      return Val;
    }
  };
  template<> struct simplify_type< ::clang::AnyFunctionDecl>
  : public simplify_type<const ::clang::AnyFunctionDecl> {};

  // Provide PointerLikeTypeTraits for non-cvr pointers.
  template<>
  class PointerLikeTypeTraits< ::clang::AnyFunctionDecl> {
  public:
    static inline void *getAsVoidPointer(::clang::AnyFunctionDecl F) {
      return F.get();
    }
    static inline ::clang::AnyFunctionDecl getFromVoidPointer(void *P) {
      return ::clang::AnyFunctionDecl::getFromNamedDecl(
                                      static_cast< ::clang::NamedDecl*>(P));
    }

    enum { NumLowBitsAvailable = 2 };
  };

} // end namespace llvm

namespace clang {

/// AccessSpecDecl - An access specifier followed by colon ':'.
///
/// An objects of this class represents sugar for the syntactic occurrence
/// of an access specifier followed by a colon in the list of member
/// specifiers of a C++ class definition.
///
/// Note that they do not represent other uses of access specifiers,
/// such as those occurring in a list of base specifiers.
/// Also note that this class has nothing to do with so-called
/// "access declarations" (C++98 11.3 [class.access.dcl]).
class AccessSpecDecl : public Decl {
  /// ColonLoc - The location of the ':'.
  SourceLocation ColonLoc;

  AccessSpecDecl(AccessSpecifier AS, DeclContext *DC,
                 SourceLocation ASLoc, SourceLocation ColonLoc)
    : Decl(AccessSpec, DC, ASLoc), ColonLoc(ColonLoc) {
    setAccess(AS);
  }
  AccessSpecDecl(EmptyShell Empty)
    : Decl(AccessSpec, Empty) { }
public:
  /// getAccessSpecifierLoc - The location of the access specifier.
  SourceLocation getAccessSpecifierLoc() const { return getLocation(); }
  /// setAccessSpecifierLoc - Sets the location of the access specifier.
  void setAccessSpecifierLoc(SourceLocation ASLoc) { setLocation(ASLoc); }

  /// getColonLoc - The location of the colon following the access specifier.
  SourceLocation getColonLoc() const { return ColonLoc; }
  /// setColonLoc - Sets the location of the colon.
  void setColonLoc(SourceLocation CLoc) { ColonLoc = CLoc; }

  SourceRange getSourceRange() const {
    return SourceRange(getAccessSpecifierLoc(), getColonLoc());
  }

  static AccessSpecDecl *Create(ASTContext &C, AccessSpecifier AS,
                                DeclContext *DC, SourceLocation ASLoc,
                                SourceLocation ColonLoc) {
    return new (C) AccessSpecDecl(AS, DC, ASLoc, ColonLoc);
  }
  static AccessSpecDecl *Create(ASTContext &C, EmptyShell Empty) {
    return new (C) AccessSpecDecl(Empty);
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const AccessSpecDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == AccessSpec; }
};


/// CXXBaseSpecifier - A base class of a C++ class.
///
/// Each CXXBaseSpecifier represents a single, direct base class (or
/// struct) of a C++ class (or struct). It specifies the type of that
/// base class, whether it is a virtual or non-virtual base, and what
/// level of access (public, protected, private) is used for the
/// derivation. For example:
///
/// @code
///   class A { };
///   class B { };
///   class C : public virtual A, protected B { };
/// @endcode
///
/// In this code, C will have two CXXBaseSpecifiers, one for "public
/// virtual A" and the other for "protected B".
class CXXBaseSpecifier {
  /// Range - The source code range that covers the full base
  /// specifier, including the "virtual" (if present) and access
  /// specifier (if present).
  SourceRange Range;

  /// \brief The source location of the ellipsis, if this is a pack
  /// expansion.
  SourceLocation EllipsisLoc;
  
  /// Virtual - Whether this is a virtual base class or not.
  bool Virtual : 1;

  /// BaseOfClass - Whether this is the base of a class (true) or of a
  /// struct (false). This determines the mapping from the access
  /// specifier as written in the source code to the access specifier
  /// used for semantic analysis.
  bool BaseOfClass : 1;

  /// Access - Access specifier as written in the source code (which
  /// may be AS_none). The actual type of data stored here is an
  /// AccessSpecifier, but we use "unsigned" here to work around a
  /// VC++ bug.
  unsigned Access : 2;

  /// InheritConstructors - Whether the class contains a using declaration
  /// to inherit the named class's constructors.
  bool InheritConstructors : 1;

  /// BaseTypeInfo - The type of the base class. This will be a class or struct
  /// (or a typedef of such). The source code range does not include the
  /// "virtual" or access specifier.
  TypeSourceInfo *BaseTypeInfo;

public:
  CXXBaseSpecifier() { }

  CXXBaseSpecifier(SourceRange R, bool V, bool BC, AccessSpecifier A,
                   TypeSourceInfo *TInfo, SourceLocation EllipsisLoc)
    : Range(R), EllipsisLoc(EllipsisLoc), Virtual(V), BaseOfClass(BC), 
      Access(A), InheritConstructors(false), BaseTypeInfo(TInfo) { }

  /// getSourceRange - Retrieves the source range that contains the
  /// entire base specifier.
  SourceRange getSourceRange() const { return Range; }

  /// isVirtual - Determines whether the base class is a virtual base
  /// class (or not).
  bool isVirtual() const { return Virtual; }

  /// \brief Determine whether this base class is a base of a class declared
  /// with the 'class' keyword (vs. one declared with the 'struct' keyword).
  bool isBaseOfClass() const { return BaseOfClass; }
  
  /// \brief Determine whether this base specifier is a pack expansion.
  bool isPackExpansion() const { return EllipsisLoc.isValid(); }

  /// \brief Determine whether this base class's constructors get inherited.
  bool getInheritConstructors() const { return InheritConstructors; }

  /// \brief Set that this base class's constructors should be inherited.
  void setInheritConstructors(bool Inherit = true) {
    InheritConstructors = Inherit;
  }

  /// \brief For a pack expansion, determine the location of the ellipsis.
  SourceLocation getEllipsisLoc() const {
    return EllipsisLoc;
  }

  /// getAccessSpecifier - Returns the access specifier for this base
  /// specifier. This is the actual base specifier as used for
  /// semantic analysis, so the result can never be AS_none. To
  /// retrieve the access specifier as written in the source code, use
  /// getAccessSpecifierAsWritten().
  AccessSpecifier getAccessSpecifier() const {
    if ((AccessSpecifier)Access == AS_none)
      return BaseOfClass? AS_private : AS_public;
    else
      return (AccessSpecifier)Access;
  }

  /// getAccessSpecifierAsWritten - Retrieves the access specifier as
  /// written in the source code (which may mean that no access
  /// specifier was explicitly written). Use getAccessSpecifier() to
  /// retrieve the access specifier for use in semantic analysis.
  AccessSpecifier getAccessSpecifierAsWritten() const {
    return (AccessSpecifier)Access;
  }

  /// getType - Retrieves the type of the base class. This type will
  /// always be an unqualified class type.
  QualType getType() const { return BaseTypeInfo->getType(); }

  /// getTypeLoc - Retrieves the type and source location of the base class.
  TypeSourceInfo *getTypeSourceInfo() const { return BaseTypeInfo; }
};

/// CXXRecordDecl - Represents a C++ struct/union/class.
/// FIXME: This class will disappear once we've properly taught RecordDecl
/// to deal with C++-specific things.
class CXXRecordDecl : public RecordDecl {

  friend void TagDecl::startDefinition();

  struct DefinitionData {
    DefinitionData(CXXRecordDecl *D);

    /// UserDeclaredConstructor - True when this class has a
    /// user-declared constructor.
    bool UserDeclaredConstructor : 1;

    /// UserDeclaredCopyConstructor - True when this class has a
    /// user-declared copy constructor.
    bool UserDeclaredCopyConstructor : 1;

    /// UserDeclareMoveConstructor - True when this class has a
    /// user-declared move constructor.
    bool UserDeclaredMoveConstructor : 1;

    /// UserDeclaredCopyAssignment - True when this class has a
    /// user-declared copy assignment operator.
    bool UserDeclaredCopyAssignment : 1;

    /// UserDeclareMoveAssignment - True when this class has a
    /// user-declared move assignment.
    bool UserDeclaredMoveAssignment : 1;

    /// UserDeclaredDestructor - True when this class has a
    /// user-declared destructor.
    bool UserDeclaredDestructor : 1;

    /// Aggregate - True when this class is an aggregate.
    bool Aggregate : 1;

    /// PlainOldData - True when this class is a POD-type.
    bool PlainOldData : 1;

    /// Empty - true when this class is empty for traits purposes,
    /// i.e. has no data members other than 0-width bit-fields, has no
    /// virtual function/base, and doesn't inherit from a non-empty
    /// class. Doesn't take union-ness into account.
    bool Empty : 1;

    /// Polymorphic - True when this class is polymorphic, i.e. has at
    /// least one virtual member or derives from a polymorphic class.
    bool Polymorphic : 1;

    /// Abstract - True when this class is abstract, i.e. has at least
    /// one pure virtual function, (that can come from a base class).
    bool Abstract : 1;

    /// IsStandardLayout - True when this class has standard layout.
    ///
    /// C++0x [class]p7.  A standard-layout class is a class that:
    /// * has no non-static data members of type non-standard-layout class (or
    ///   array of such types) or reference,
    /// * has no virtual functions (10.3) and no virtual base classes (10.1),
    /// * has the same access control (Clause 11) for all non-static data members
    /// * has no non-standard-layout base classes,
    /// * either has no non-static data members in the most derived class and at
    ///   most one base class with non-static data members, or has no base
    ///   classes with non-static data members, and
    /// * has no base classes of the same type as the first non-static data
    ///   member.
    bool IsStandardLayout : 1;

    /// HasNoNonEmptyBases - True when there are no non-empty base classes.
    ///
    /// This is a helper bit of state used to implement IsStandardLayout more
    /// efficiently.
    bool HasNoNonEmptyBases : 1;

    /// HasPrivateFields - True when there are private non-static data members.
    bool HasPrivateFields : 1;

    /// HasProtectedFields - True when there are protected non-static data
    /// members.
    bool HasProtectedFields : 1;

    /// HasPublicFields - True when there are private non-static data members.
    bool HasPublicFields : 1;

    /// \brief True if this class (or any subobject) has mutable fields.
    bool HasMutableFields : 1;
    
    /// HasTrivialDefaultConstructor - True when, if this class has a default
    /// constructor, this default constructor is trivial.
    ///
    /// C++0x [class.ctor]p5
    ///    A default constructor is trivial if it is not user-provided and if
    ///     -- its class has no virtual functions and no virtual base classes,
    ///        and
    ///     -- no non-static data member of its class has a
    ///        brace-or-equal-initializer, and
    ///     -- all the direct base classes of its class have trivial
    ///        default constructors, and
    ///     -- for all the nonstatic data members of its class that are of class
    ///        type (or array thereof), each such class has a trivial
    ///        default constructor.
    bool HasTrivialDefaultConstructor : 1;

    /// HasConstexprNonCopyMoveConstructor - True when this class has at least
    /// one constexpr constructor which is neither the copy nor move
    /// constructor.
    bool HasConstexprNonCopyMoveConstructor : 1;

    /// HasTrivialCopyConstructor - True when this class has a trivial copy
    /// constructor.
    ///
    /// C++0x [class.copy]p13:
    ///   A copy/move constructor for class X is trivial if it is neither
    ///   user-provided and if
    ///    -- class X has no virtual functions and no virtual base classes, and
    ///    -- the constructor selected to copy/move each direct base class
    ///       subobject is trivial, and
    ///    -- for each non-static data member of X that is of class type (or an
    ///       array thereof), the constructor selected to copy/move that member
    ///       is trivial;
    ///   otherwise the copy/move constructor is non-trivial.
    bool HasTrivialCopyConstructor : 1;

    /// HasTrivialMoveConstructor - True when this class has a trivial move
    /// constructor.
    ///
    /// C++0x [class.copy]p13:
    ///   A copy/move constructor for class X is trivial if it is neither
    ///   user-provided and if
    ///    -- class X has no virtual functions and no virtual base classes, and
    ///    -- the constructor selected to copy/move each direct base class
    ///       subobject is trivial, and
    ///    -- for each non-static data member of X that is of class type (or an
    ///       array thereof), the constructor selected to copy/move that member
    ///       is trivial;
    ///   otherwise the copy/move constructor is non-trivial.
    bool HasTrivialMoveConstructor : 1;

    /// HasTrivialCopyAssignment - True when this class has a trivial copy
    /// assignment operator.
    ///
    /// C++0x [class.copy]p27:
    ///   A copy/move assignment operator for class X is trivial if it is
    ///   neither user-provided nor deleted and if
    ///    -- class X has no virtual functions and no virtual base classes, and
    ///    -- the assignment operator selected to copy/move each direct base
    ///       class subobject is trivial, and
    ///    -- for each non-static data member of X that is of class type (or an
    ///       array thereof), the assignment operator selected to copy/move
    ///       that member is trivial;
    ///   otherwise the copy/move assignment operator is non-trivial.
    bool HasTrivialCopyAssignment : 1;

    /// HasTrivialMoveAssignment - True when this class has a trivial move
    /// assignment operator.
    ///
    /// C++0x [class.copy]p27:
    ///   A copy/move assignment operator for class X is trivial if it is
    ///   neither user-provided nor deleted and if
    ///    -- class X has no virtual functions and no virtual base classes, and
    ///    -- the assignment operator selected to copy/move each direct base
    ///       class subobject is trivial, and
    ///    -- for each non-static data member of X that is of class type (or an
    ///       array thereof), the assignment operator selected to copy/move
    ///       that member is trivial;
    ///   otherwise the copy/move assignment operator is non-trivial.
    bool HasTrivialMoveAssignment : 1;

    /// HasTrivialDestructor - True when this class has a trivial destructor.
    ///
    /// C++ [class.dtor]p3.  A destructor is trivial if it is an
    /// implicitly-declared destructor and if:
    /// * all of the direct base classes of its class have trivial destructors
    ///   and
    /// * for all of the non-static data members of its class that are of class
    ///   type (or array thereof), each such class has a trivial destructor.
    bool HasTrivialDestructor : 1;

    /// HasNonLiteralTypeFieldsOrBases - True when this class contains at least
    /// one non-static data member or base class of non literal type.
    bool HasNonLiteralTypeFieldsOrBases : 1;

    /// ComputedVisibleConversions - True when visible conversion functions are
    /// already computed and are available.
    bool ComputedVisibleConversions : 1;

    /// \brief Whether we have a C++0x user-provided default constructor (not
    /// explicitly deleted or defaulted).
    bool UserProvidedDefaultConstructor : 1;

    /// \brief Whether we have already declared the default constructor.
    bool DeclaredDefaultConstructor : 1;

    /// \brief Whether we have already declared the copy constructor.
    bool DeclaredCopyConstructor : 1;

    /// \brief Whether we have already declared the move constructor.
    bool DeclaredMoveConstructor : 1;
    
    /// \brief Whether we have already declared the copy-assignment operator.
    bool DeclaredCopyAssignment : 1;

    /// \brief Whether we have already declared the move-assignment operator.
    bool DeclaredMoveAssignment : 1;
    
    /// \brief Whether we have already declared a destructor within the class.
    bool DeclaredDestructor : 1;

    /// NumBases - The number of base class specifiers in Bases.
    unsigned NumBases;
    
    /// NumVBases - The number of virtual base class specifiers in VBases.
    unsigned NumVBases;

    /// Bases - Base classes of this class.
    /// FIXME: This is wasted space for a union.
    LazyCXXBaseSpecifiersPtr Bases;

    /// VBases - direct and indirect virtual base classes of this class.
    LazyCXXBaseSpecifiersPtr VBases;

    /// Conversions - Overload set containing the conversion functions
    /// of this C++ class (but not its inherited conversion
    /// functions). Each of the entries in this overload set is a
    /// CXXConversionDecl.
    UnresolvedSet<4> Conversions;

    /// VisibleConversions - Overload set containing the conversion
    /// functions of this C++ class and all those inherited conversion
    /// functions that are visible in this class. Each of the entries
    /// in this overload set is a CXXConversionDecl or a
    /// FunctionTemplateDecl.
    UnresolvedSet<4> VisibleConversions;

    /// Definition - The declaration which defines this record.
    CXXRecordDecl *Definition;

    /// FirstFriend - The first friend declaration in this class, or
    /// null if there aren't any.  This is actually currently stored
    /// in reverse order.
    FriendDecl *FirstFriend;

    /// \brief Retrieve the set of direct base classes.    
    CXXBaseSpecifier *getBases() const {
      return Bases.get(Definition->getASTContext().getExternalSource());
    }

    /// \brief Retrieve the set of virtual base classes.    
    CXXBaseSpecifier *getVBases() const {
      return VBases.get(Definition->getASTContext().getExternalSource());
    }
  } *DefinitionData;

  struct DefinitionData &data() {
    assert(DefinitionData && "queried property of class with no definition");
    return *DefinitionData;
  }

  const struct DefinitionData &data() const {
    assert(DefinitionData && "queried property of class with no definition");
    return *DefinitionData;
  }
  
  /// \brief The template or declaration that this declaration
  /// describes or was instantiated from, respectively.
  ///
  /// For non-templates, this value will be NULL. For record
  /// declarations that describe a class template, this will be a
  /// pointer to a ClassTemplateDecl. For member
  /// classes of class template specializations, this will be the
  /// MemberSpecializationInfo referring to the member class that was 
  /// instantiated or specialized.
  llvm::PointerUnion<ClassTemplateDecl*, MemberSpecializationInfo*>
    TemplateOrInstantiation;

  friend class DeclContext;
  
  /// \brief Notify the class that member has been added.
  ///
  /// This routine helps maintain information about the class based on which 
  /// members have been added. It will be invoked by DeclContext::addDecl()
  /// whenever a member is added to this record.
  void addedMember(Decl *D);

  void markedVirtualFunctionPure();
  friend void FunctionDecl::setPure(bool);
  
protected:
  CXXRecordDecl(Kind K, TagKind TK, DeclContext *DC,
                SourceLocation StartLoc, SourceLocation IdLoc,
                IdentifierInfo *Id, CXXRecordDecl *PrevDecl);

public:
  /// base_class_iterator - Iterator that traverses the base classes
  /// of a class.
  typedef CXXBaseSpecifier*       base_class_iterator;

  /// base_class_const_iterator - Iterator that traverses the base
  /// classes of a class.
  typedef const CXXBaseSpecifier* base_class_const_iterator;

  /// reverse_base_class_iterator = Iterator that traverses the base classes
  /// of a class in reverse order.
  typedef std::reverse_iterator<base_class_iterator>
    reverse_base_class_iterator;

  /// reverse_base_class_iterator = Iterator that traverses the base classes
  /// of a class in reverse order.
  typedef std::reverse_iterator<base_class_const_iterator>
    reverse_base_class_const_iterator;

  virtual CXXRecordDecl *getCanonicalDecl() {
    return cast<CXXRecordDecl>(RecordDecl::getCanonicalDecl());
  }
  virtual const CXXRecordDecl *getCanonicalDecl() const {
    return cast<CXXRecordDecl>(RecordDecl::getCanonicalDecl());
  }
  
  const CXXRecordDecl *getPreviousDeclaration() const {
    return cast_or_null<CXXRecordDecl>(RecordDecl::getPreviousDeclaration());
  }
  CXXRecordDecl *getPreviousDeclaration() {
    return cast_or_null<CXXRecordDecl>(RecordDecl::getPreviousDeclaration());
  }

  CXXRecordDecl *getDefinition() const {
    if (!DefinitionData) return 0;
    return data().Definition;
  }

  bool hasDefinition() const { return DefinitionData != 0; }

  static CXXRecordDecl *Create(const ASTContext &C, TagKind TK, DeclContext *DC,
                               SourceLocation StartLoc, SourceLocation IdLoc,
                               IdentifierInfo *Id, CXXRecordDecl* PrevDecl=0,
                               bool DelayTypeCreation = false);
  static CXXRecordDecl *Create(const ASTContext &C, EmptyShell Empty);

  bool isDynamicClass() const {
    return data().Polymorphic || data().NumVBases != 0;
  }

  /// setBases - Sets the base classes of this struct or class.
  void setBases(CXXBaseSpecifier const * const *Bases, unsigned NumBases);

  /// getNumBases - Retrieves the number of base classes of this
  /// class.
  unsigned getNumBases() const { return data().NumBases; }

  base_class_iterator bases_begin() { return data().getBases(); }
  base_class_const_iterator bases_begin() const { return data().getBases(); }
  base_class_iterator bases_end() { return bases_begin() + data().NumBases; }
  base_class_const_iterator bases_end() const {
    return bases_begin() + data().NumBases;
  }
  reverse_base_class_iterator       bases_rbegin() {
    return reverse_base_class_iterator(bases_end());
  }
  reverse_base_class_const_iterator bases_rbegin() const {
    return reverse_base_class_const_iterator(bases_end());
  }
  reverse_base_class_iterator bases_rend() {
    return reverse_base_class_iterator(bases_begin());
  }
  reverse_base_class_const_iterator bases_rend() const {
    return reverse_base_class_const_iterator(bases_begin());
  }

  /// getNumVBases - Retrieves the number of virtual base classes of this
  /// class.
  unsigned getNumVBases() const { return data().NumVBases; }

  base_class_iterator vbases_begin() { return data().getVBases(); }
  base_class_const_iterator vbases_begin() const { return data().getVBases(); }
  base_class_iterator vbases_end() { return vbases_begin() + data().NumVBases; }
  base_class_const_iterator vbases_end() const {
    return vbases_begin() + data().NumVBases;
  }
  reverse_base_class_iterator vbases_rbegin() {
    return reverse_base_class_iterator(vbases_end());
  }
  reverse_base_class_const_iterator vbases_rbegin() const {
    return reverse_base_class_const_iterator(vbases_end());
  }
  reverse_base_class_iterator vbases_rend() {
    return reverse_base_class_iterator(vbases_begin());
  }
  reverse_base_class_const_iterator vbases_rend() const {
    return reverse_base_class_const_iterator(vbases_begin());
 }

  /// \brief Determine whether this class has any dependent base classes.
  bool hasAnyDependentBases() const;

  /// Iterator access to method members.  The method iterator visits
  /// all method members of the class, including non-instance methods,
  /// special methods, etc.
  typedef specific_decl_iterator<CXXMethodDecl> method_iterator;

  /// method_begin - Method begin iterator.  Iterates in the order the methods
  /// were declared.
  method_iterator method_begin() const {
    return method_iterator(decls_begin());
  }
  /// method_end - Method end iterator.
  method_iterator method_end() const {
    return method_iterator(decls_end());
  }

  /// Iterator access to constructor members.
  typedef specific_decl_iterator<CXXConstructorDecl> ctor_iterator;

  ctor_iterator ctor_begin() const {
    return ctor_iterator(decls_begin());
  }
  ctor_iterator ctor_end() const {
    return ctor_iterator(decls_end());
  }

  /// An iterator over friend declarations.  All of these are defined
  /// in DeclFriend.h.
  class friend_iterator;
  friend_iterator friend_begin() const;
  friend_iterator friend_end() const;
  void pushFriendDecl(FriendDecl *FD);

  /// Determines whether this record has any friends.
  bool hasFriends() const {
    return data().FirstFriend != 0;
  }

  /// \brief Determine if we need to declare a default constructor for
  /// this class.
  ///
  /// This value is used for lazy creation of default constructors.
  bool needsImplicitDefaultConstructor() const {
    return !data().UserDeclaredConstructor && 
           !data().DeclaredDefaultConstructor;
  }

  /// hasDeclaredDefaultConstructor - Whether this class's default constructor
  /// has been declared (either explicitly or implicitly).
  bool hasDeclaredDefaultConstructor() const {
    return data().DeclaredDefaultConstructor;
  }

  /// hasConstCopyConstructor - Determines whether this class has a
  /// copy constructor that accepts a const-qualified argument.
  bool hasConstCopyConstructor() const;

  /// getCopyConstructor - Returns the copy constructor for this class
  CXXConstructorDecl *getCopyConstructor(unsigned TypeQuals) const;

  /// getMoveConstructor - Returns the move constructor for this class
  CXXConstructorDecl *getMoveConstructor() const; 

  /// \brief Retrieve the copy-assignment operator for this class, if available.
  ///
  /// This routine attempts to find the copy-assignment operator for this 
  /// class, using a simplistic form of overload resolution.
  ///
  /// \param ArgIsConst Whether the argument to the copy-assignment operator
  /// is const-qualified.
  ///
  /// \returns The copy-assignment operator that can be invoked, or NULL if
  /// a unique copy-assignment operator could not be found.
  CXXMethodDecl *getCopyAssignmentOperator(bool ArgIsConst) const;

  /// getMoveAssignmentOperator - Returns the move assignment operator for this
  /// class
  CXXMethodDecl *getMoveAssignmentOperator() const;
  
  /// hasUserDeclaredConstructor - Whether this class has any
  /// user-declared constructors. When true, a default constructor
  /// will not be implicitly declared.
  bool hasUserDeclaredConstructor() const {
    return data().UserDeclaredConstructor;
  }

  /// hasUserProvidedDefaultconstructor - Whether this class has a
  /// user-provided default constructor per C++0x.
  bool hasUserProvidedDefaultConstructor() const {
    return data().UserProvidedDefaultConstructor;
  }

  /// hasUserDeclaredCopyConstructor - Whether this class has a
  /// user-declared copy constructor. When false, a copy constructor
  /// will be implicitly declared.
  bool hasUserDeclaredCopyConstructor() const {
    return data().UserDeclaredCopyConstructor;
  }

  /// \brief Determine whether this class has had its copy constructor 
  /// declared, either via the user or via an implicit declaration.
  ///
  /// This value is used for lazy creation of copy constructors.
  bool hasDeclaredCopyConstructor() const {
    return data().DeclaredCopyConstructor;
  }

  /// hasUserDeclaredMoveOperation - Whether this class has a user-
  /// declared move constructor or assignment operator. When false, a
  /// move constructor and assignment operator may be implicitly declared.
  bool hasUserDeclaredMoveOperation() const {
    return data().UserDeclaredMoveConstructor ||
           data().UserDeclaredMoveAssignment;
  }

  /// \brief Determine whether this class has had a move constructor
  /// declared by the user.
  bool hasUserDeclaredMoveConstructor() const {
    return data().UserDeclaredMoveConstructor;
  }

  /// \brief Determine whether this class has had a move constructor
  /// declared.
  bool hasDeclaredMoveConstructor() const {
    return data().DeclaredMoveConstructor;
  }

  /// hasUserDeclaredCopyAssignment - Whether this class has a
  /// user-declared copy assignment operator. When false, a copy
  /// assigment operator will be implicitly declared.
  bool hasUserDeclaredCopyAssignment() const {
    return data().UserDeclaredCopyAssignment;
  }

  /// \brief Determine whether this class has had its copy assignment operator 
  /// declared, either via the user or via an implicit declaration.
  ///
  /// This value is used for lazy creation of copy assignment operators.
  bool hasDeclaredCopyAssignment() const {
    return data().DeclaredCopyAssignment;
  }

  /// \brief Determine whether this class has had a move assignment
  /// declared by the user.
  bool hasUserDeclaredMoveAssignment() const {
    return data().UserDeclaredMoveAssignment;
  }

  /// hasDeclaredMoveAssignment - Whether this class has a
  /// declared move assignment operator.
  bool hasDeclaredMoveAssignment() const {
    return data().DeclaredMoveAssignment;
  }

  /// hasUserDeclaredDestructor - Whether this class has a
  /// user-declared destructor. When false, a destructor will be
  /// implicitly declared.
  bool hasUserDeclaredDestructor() const {
    return data().UserDeclaredDestructor;
  }

  /// \brief Determine whether this class has had its destructor declared,
  /// either via the user or via an implicit declaration.
  ///
  /// This value is used for lazy creation of destructors.
  bool hasDeclaredDestructor() const { return data().DeclaredDestructor; }

  /// getConversions - Retrieve the overload set containing all of the
  /// conversion functions in this class.
  UnresolvedSetImpl *getConversionFunctions() {
    return &data().Conversions;
  }
  const UnresolvedSetImpl *getConversionFunctions() const {
    return &data().Conversions;
  }

  typedef UnresolvedSetImpl::iterator conversion_iterator;
  conversion_iterator conversion_begin() const {
    return getConversionFunctions()->begin();
  }
  conversion_iterator conversion_end() const {
    return getConversionFunctions()->end();
  }

  /// Removes a conversion function from this class.  The conversion
  /// function must currently be a member of this class.  Furthermore,
  /// this class must currently be in the process of being defined.
  void removeConversion(const NamedDecl *Old);

  /// getVisibleConversionFunctions - get all conversion functions visible
  /// in current class; including conversion function templates.
  const UnresolvedSetImpl *getVisibleConversionFunctions();

  /// isAggregate - Whether this class is an aggregate (C++
  /// [dcl.init.aggr]), which is a class with no user-declared
  /// constructors, no private or protected non-static data members,
  /// no base classes, and no virtual functions (C++ [dcl.init.aggr]p1).
  bool isAggregate() const { return data().Aggregate; }

  /// isPOD - Whether this class is a POD-type (C++ [class]p4), which is a class
  /// that is an aggregate that has no non-static non-POD data members, no
  /// reference data members, no user-defined copy assignment operator and no
  /// user-defined destructor.
  bool isPOD() const { return data().PlainOldData; }

  /// isEmpty - Whether this class is empty (C++0x [meta.unary.prop]), which
  /// means it has a virtual function, virtual base, data member (other than
  /// 0-width bit-field) or inherits from a non-empty class. Does NOT include
  /// a check for union-ness.
  bool isEmpty() const { return data().Empty; }

  /// isPolymorphic - Whether this class is polymorphic (C++ [class.virtual]),
  /// which means that the class contains or inherits a virtual function.
  bool isPolymorphic() const { return data().Polymorphic; }

  /// isAbstract - Whether this class is abstract (C++ [class.abstract]),
  /// which means that the class contains or inherits a pure virtual function.
  bool isAbstract() const { return data().Abstract; }

  /// isStandardLayout - Whether this class has standard layout
  /// (C++ [class]p7)
  bool isStandardLayout() const { return data().IsStandardLayout; }

  /// \brief Whether this class, or any of its class subobjects, contains a
  /// mutable field.
  bool hasMutableFields() const { return data().HasMutableFields; }
  
  // hasTrivialDefaultConstructor - Whether this class has a trivial default
  // constructor
  // (C++0x [class.ctor]p5)
  bool hasTrivialDefaultConstructor() const {
    return data().HasTrivialDefaultConstructor &&
           (!data().UserDeclaredConstructor ||
             data().DeclaredDefaultConstructor);
  }

  // hasConstexprNonCopyMoveConstructor - Whether this class has at least one
  // constexpr constructor other than the copy or move constructors
  bool hasConstexprNonCopyMoveConstructor() const {
    return data().HasConstexprNonCopyMoveConstructor;
  }

  // hasTrivialCopyConstructor - Whether this class has a trivial copy
  // constructor (C++ [class.copy]p6, C++0x [class.copy]p13)
  bool hasTrivialCopyConstructor() const {
    return data().HasTrivialCopyConstructor;
  }

  // hasTrivialMoveConstructor - Whether this class has a trivial move
  // constructor (C++0x [class.copy]p13)
  bool hasTrivialMoveConstructor() const {
    return data().HasTrivialMoveConstructor;
  }

  // hasTrivialCopyAssignment - Whether this class has a trivial copy
  // assignment operator (C++ [class.copy]p11, C++0x [class.copy]p27)
  bool hasTrivialCopyAssignment() const {
    return data().HasTrivialCopyAssignment;
  }

  // hasTrivialMoveAssignment - Whether this class has a trivial move
  // assignment operator (C++0x [class.copy]p27)
  bool hasTrivialMoveAssignment() const {
    return data().HasTrivialMoveAssignment;
  }

  // hasTrivialDestructor - Whether this class has a trivial destructor
  // (C++ [class.dtor]p3)
  bool hasTrivialDestructor() const { return data().HasTrivialDestructor; }

  // hasNonLiteralTypeFieldsOrBases - Whether this class has a non-literal type
  // non-static data member or base class.
  bool hasNonLiteralTypeFieldsOrBases() const {
    return data().HasNonLiteralTypeFieldsOrBases;
  }

  // isTriviallyCopyable - Whether this class is considered trivially copyable
  // (C++0x [class]p6).
  bool isTriviallyCopyable() const;

  // isTrivial - Whether this class is considered trivial
  //
  // C++0x [class]p6
  //    A trivial class is a class that has a trivial default constructor and
  //    is trivially copiable.
  bool isTrivial() const {
    return isTriviallyCopyable() && hasTrivialDefaultConstructor();
  }

  /// \brief If this record is an instantiation of a member class,
  /// retrieves the member class from which it was instantiated.
  ///
  /// This routine will return non-NULL for (non-templated) member
  /// classes of class templates. For example, given:
  ///
  /// \code
  /// template<typename T>
  /// struct X {
  ///   struct A { };
  /// };
  /// \endcode
  ///
  /// The declaration for X<int>::A is a (non-templated) CXXRecordDecl
  /// whose parent is the class template specialization X<int>. For
  /// this declaration, getInstantiatedFromMemberClass() will return
  /// the CXXRecordDecl X<T>::A. When a complete definition of
  /// X<int>::A is required, it will be instantiated from the
  /// declaration returned by getInstantiatedFromMemberClass().
  CXXRecordDecl *getInstantiatedFromMemberClass() const;
  
  /// \brief If this class is an instantiation of a member class of a
  /// class template specialization, retrieves the member specialization
  /// information.
  MemberSpecializationInfo *getMemberSpecializationInfo() const;
  
  /// \brief Specify that this record is an instantiation of the
  /// member class RD.
  void setInstantiationOfMemberClass(CXXRecordDecl *RD,
                                     TemplateSpecializationKind TSK);

  /// \brief Retrieves the class template that is described by this
  /// class declaration.
  ///
  /// Every class template is represented as a ClassTemplateDecl and a
  /// CXXRecordDecl. The former contains template properties (such as
  /// the template parameter lists) while the latter contains the
  /// actual description of the template's
  /// contents. ClassTemplateDecl::getTemplatedDecl() retrieves the
  /// CXXRecordDecl that from a ClassTemplateDecl, while
  /// getDescribedClassTemplate() retrieves the ClassTemplateDecl from
  /// a CXXRecordDecl.
  ClassTemplateDecl *getDescribedClassTemplate() const {
    return TemplateOrInstantiation.dyn_cast<ClassTemplateDecl*>();
  }

  void setDescribedClassTemplate(ClassTemplateDecl *Template) {
    TemplateOrInstantiation = Template;
  }

  /// \brief Determine whether this particular class is a specialization or
  /// instantiation of a class template or member class of a class template,
  /// and how it was instantiated or specialized.
  TemplateSpecializationKind getTemplateSpecializationKind() const;
  
  /// \brief Set the kind of specialization or template instantiation this is.
  void setTemplateSpecializationKind(TemplateSpecializationKind TSK);

  /// getDestructor - Returns the destructor decl for this class.
  CXXDestructorDecl *getDestructor() const;

  /// isLocalClass - If the class is a local class [class.local], returns
  /// the enclosing function declaration.
  const FunctionDecl *isLocalClass() const {
    if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(getDeclContext()))
      return RD->isLocalClass();

    return dyn_cast<FunctionDecl>(getDeclContext());
  }

  /// \brief Determine whether this class is derived from the class \p Base.
  ///
  /// This routine only determines whether this class is derived from \p Base,
  /// but does not account for factors that may make a Derived -> Base class
  /// ill-formed, such as private/protected inheritance or multiple, ambiguous
  /// base class subobjects.
  ///
  /// \param Base the base class we are searching for.
  ///
  /// \returns true if this class is derived from Base, false otherwise.
  bool isDerivedFrom(const CXXRecordDecl *Base) const;
  
  /// \brief Determine whether this class is derived from the type \p Base.
  ///
  /// This routine only determines whether this class is derived from \p Base,
  /// but does not account for factors that may make a Derived -> Base class
  /// ill-formed, such as private/protected inheritance or multiple, ambiguous
  /// base class subobjects.
  ///
  /// \param Base the base class we are searching for.
  ///
  /// \param Paths will contain the paths taken from the current class to the
  /// given \p Base class.
  ///
  /// \returns true if this class is derived from Base, false otherwise.
  ///
  /// \todo add a separate paramaeter to configure IsDerivedFrom, rather than 
  /// tangling input and output in \p Paths  
  bool isDerivedFrom(const CXXRecordDecl *Base, CXXBasePaths &Paths) const;

  /// \brief Determine whether this class is virtually derived from
  /// the class \p Base.
  ///
  /// This routine only determines whether this class is virtually
  /// derived from \p Base, but does not account for factors that may
  /// make a Derived -> Base class ill-formed, such as
  /// private/protected inheritance or multiple, ambiguous base class
  /// subobjects.
  ///
  /// \param Base the base class we are searching for.
  ///
  /// \returns true if this class is virtually derived from Base,
  /// false otherwise.
  bool isVirtuallyDerivedFrom(CXXRecordDecl *Base) const;

  /// \brief Determine whether this class is provably not derived from
  /// the type \p Base.
  bool isProvablyNotDerivedFrom(const CXXRecordDecl *Base) const;

  /// \brief Function type used by forallBases() as a callback.
  ///
  /// \param Base the definition of the base class
  ///
  /// \returns true if this base matched the search criteria
  typedef bool ForallBasesCallback(const CXXRecordDecl *BaseDefinition,
                                   void *UserData);

  /// \brief Determines if the given callback holds for all the direct
  /// or indirect base classes of this type.
  ///
  /// The class itself does not count as a base class.  This routine
  /// returns false if the class has non-computable base classes.
  /// 
  /// \param AllowShortCircuit if false, forces the callback to be called
  /// for every base class, even if a dependent or non-matching base was
  /// found.
  bool forallBases(ForallBasesCallback *BaseMatches, void *UserData,
                   bool AllowShortCircuit = true) const;
  
  /// \brief Function type used by lookupInBases() to determine whether a 
  /// specific base class subobject matches the lookup criteria.
  ///
  /// \param Specifier the base-class specifier that describes the inheritance 
  /// from the base class we are trying to match.
  ///
  /// \param Path the current path, from the most-derived class down to the 
  /// base named by the \p Specifier.
  ///
  /// \param UserData a single pointer to user-specified data, provided to
  /// lookupInBases().
  ///
  /// \returns true if this base matched the search criteria, false otherwise.
  typedef bool BaseMatchesCallback(const CXXBaseSpecifier *Specifier,
                                   CXXBasePath &Path,
                                   void *UserData);
  
  /// \brief Look for entities within the base classes of this C++ class,
  /// transitively searching all base class subobjects.
  ///
  /// This routine uses the callback function \p BaseMatches to find base 
  /// classes meeting some search criteria, walking all base class subobjects
  /// and populating the given \p Paths structure with the paths through the 
  /// inheritance hierarchy that resulted in a match. On a successful search,
  /// the \p Paths structure can be queried to retrieve the matching paths and
  /// to determine if there were any ambiguities.
  ///
  /// \param BaseMatches callback function used to determine whether a given
  /// base matches the user-defined search criteria.
  ///
  /// \param UserData user data pointer that will be provided to \p BaseMatches.
  ///
  /// \param Paths used to record the paths from this class to its base class
  /// subobjects that match the search criteria.
  ///
  /// \returns true if there exists any path from this class to a base class
  /// subobject that matches the search criteria.
  bool lookupInBases(BaseMatchesCallback *BaseMatches, void *UserData,
                     CXXBasePaths &Paths) const;
  
  /// \brief Base-class lookup callback that determines whether the given
  /// base class specifier refers to a specific class declaration.
  ///
  /// This callback can be used with \c lookupInBases() to determine whether
  /// a given derived class has is a base class subobject of a particular type.
  /// The user data pointer should refer to the canonical CXXRecordDecl of the
  /// base class that we are searching for.
  static bool FindBaseClass(const CXXBaseSpecifier *Specifier,
                            CXXBasePath &Path, void *BaseRecord);

  /// \brief Base-class lookup callback that determines whether the
  /// given base class specifier refers to a specific class
  /// declaration and describes virtual derivation.
  ///
  /// This callback can be used with \c lookupInBases() to determine
  /// whether a given derived class has is a virtual base class
  /// subobject of a particular type.  The user data pointer should
  /// refer to the canonical CXXRecordDecl of the base class that we
  /// are searching for.
  static bool FindVirtualBaseClass(const CXXBaseSpecifier *Specifier,
                                   CXXBasePath &Path, void *BaseRecord);
  
  /// \brief Base-class lookup callback that determines whether there exists
  /// a tag with the given name.
  ///
  /// This callback can be used with \c lookupInBases() to find tag members
  /// of the given name within a C++ class hierarchy. The user data pointer
  /// is an opaque \c DeclarationName pointer.
  static bool FindTagMember(const CXXBaseSpecifier *Specifier,
                            CXXBasePath &Path, void *Name);

  /// \brief Base-class lookup callback that determines whether there exists
  /// a member with the given name.
  ///
  /// This callback can be used with \c lookupInBases() to find members
  /// of the given name within a C++ class hierarchy. The user data pointer
  /// is an opaque \c DeclarationName pointer.
  static bool FindOrdinaryMember(const CXXBaseSpecifier *Specifier,
                                 CXXBasePath &Path, void *Name);
  
  /// \brief Base-class lookup callback that determines whether there exists
  /// a member with the given name that can be used in a nested-name-specifier.
  ///
  /// This callback can be used with \c lookupInBases() to find membes of
  /// the given name within a C++ class hierarchy that can occur within
  /// nested-name-specifiers.
  static bool FindNestedNameSpecifierMember(const CXXBaseSpecifier *Specifier,
                                            CXXBasePath &Path,
                                            void *UserData);

  /// \brief Retrieve the final overriders for each virtual member
  /// function in the class hierarchy where this class is the
  /// most-derived class in the class hierarchy.
  void getFinalOverriders(CXXFinalOverriderMap &FinaOverriders) const;

  /// \brief Get the indirect primary bases for this class.
  void getIndirectPrimaryBases(CXXIndirectPrimaryBaseSet& Bases) const;

  /// viewInheritance - Renders and displays an inheritance diagram
  /// for this C++ class and all of its base classes (transitively) using
  /// GraphViz.
  void viewInheritance(ASTContext& Context) const;

  /// MergeAccess - Calculates the access of a decl that is reached
  /// along a path.
  static AccessSpecifier MergeAccess(AccessSpecifier PathAccess,
                                     AccessSpecifier DeclAccess) {
    assert(DeclAccess != AS_none);
    if (DeclAccess == AS_private) return AS_none;
    return (PathAccess > DeclAccess ? PathAccess : DeclAccess);
  }

  /// \brief Indicates that the definition of this class is now complete.
  virtual void completeDefinition();

  /// \brief Indicates that the definition of this class is now complete, 
  /// and provides a final overrider map to help determine
  /// 
  /// \param FinalOverriders The final overrider map for this class, which can
  /// be provided as an optimization for abstract-class checking. If NULL,
  /// final overriders will be computed if they are needed to complete the
  /// definition.
  void completeDefinition(CXXFinalOverriderMap *FinalOverriders);
  
  /// \brief Determine whether this class may end up being abstract, even though
  /// it is not yet known to be abstract.
  ///
  /// \returns true if this class is not known to be abstract but has any
  /// base classes that are abstract. In this case, \c completeDefinition()
  /// will need to compute final overriders to determine whether the class is
  /// actually abstract.
  bool mayBeAbstract() const;
  
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) {
    return K >= firstCXXRecord && K <= lastCXXRecord;
  }
  static bool classof(const CXXRecordDecl *D) { return true; }
  static bool classof(const ClassTemplateSpecializationDecl *D) {
    return true;
  }

  friend class ASTDeclReader;
  friend class ASTDeclWriter;
  friend class ASTReader;
  friend class ASTWriter;
};

/// CXXMethodDecl - Represents a static or instance method of a
/// struct/union/class.
class CXXMethodDecl : public FunctionDecl {
protected:
  CXXMethodDecl(Kind DK, CXXRecordDecl *RD, SourceLocation StartLoc,
                const DeclarationNameInfo &NameInfo,
                QualType T, TypeSourceInfo *TInfo,
                bool isStatic, StorageClass SCAsWritten, bool isInline,
                SourceLocation EndLocation)
    : FunctionDecl(DK, RD, StartLoc, NameInfo, T, TInfo,
                   (isStatic ? SC_Static : SC_None),
                   SCAsWritten, isInline) {
      if (EndLocation.isValid())
        setRangeEnd(EndLocation);
    }

public:
  static CXXMethodDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                               SourceLocation StartLoc,
                               const DeclarationNameInfo &NameInfo,
                               QualType T, TypeSourceInfo *TInfo,
                               bool isStatic,
                               StorageClass SCAsWritten,
                               bool isInline,
                               SourceLocation EndLocation);

  bool isStatic() const { return getStorageClass() == SC_Static; }
  bool isInstance() const { return !isStatic(); }

  bool isVirtual() const {
    CXXMethodDecl *CD = 
      cast<CXXMethodDecl>(const_cast<CXXMethodDecl*>(this)->getCanonicalDecl());

    if (CD->isVirtualAsWritten())
      return true;
    
    return (CD->begin_overridden_methods() != CD->end_overridden_methods());
  }

  /// \brief Determine whether this is a usual deallocation function
  /// (C++ [basic.stc.dynamic.deallocation]p2), which is an overloaded
  /// delete or delete[] operator with a particular signature.
  bool isUsualDeallocationFunction() const;
  
  /// \brief Determine whether this is a copy-assignment operator, regardless
  /// of whether it was declared implicitly or explicitly.
  bool isCopyAssignmentOperator() const;

  /// \brief Determine whether this is a move assignment operator.
  bool isMoveAssignmentOperator() const;
  
  const CXXMethodDecl *getCanonicalDecl() const {
    return cast<CXXMethodDecl>(FunctionDecl::getCanonicalDecl());
  }
  CXXMethodDecl *getCanonicalDecl() {
    return cast<CXXMethodDecl>(FunctionDecl::getCanonicalDecl());
  }

  /// isUserProvided - True if it is either an implicit constructor or
  /// if it was defaulted or deleted on first declaration.
  bool isUserProvided() const {
    return !(isDeleted() || getCanonicalDecl()->isDefaulted());
  }
  
  ///
  void addOverriddenMethod(const CXXMethodDecl *MD);

  typedef const CXXMethodDecl ** method_iterator;

  method_iterator begin_overridden_methods() const;
  method_iterator end_overridden_methods() const;
  unsigned size_overridden_methods() const;

  /// getParent - Returns the parent of this method declaration, which
  /// is the class in which this method is defined.
  const CXXRecordDecl *getParent() const {
    return cast<CXXRecordDecl>(FunctionDecl::getParent());
  }

  /// getParent - Returns the parent of this method declaration, which
  /// is the class in which this method is defined.
  CXXRecordDecl *getParent() {
    return const_cast<CXXRecordDecl *>(
             cast<CXXRecordDecl>(FunctionDecl::getParent()));
  }

  /// getThisType - Returns the type of 'this' pointer.
  /// Should only be called for instance methods.
  QualType getThisType(ASTContext &C) const;

  unsigned getTypeQualifiers() const {
    return getType()->getAs<FunctionProtoType>()->getTypeQuals();
  }

  /// \brief Retrieve the ref-qualifier associated with this method.
  ///
  /// In the following example, \c f() has an lvalue ref-qualifier, \c g()
  /// has an rvalue ref-qualifier, and \c h() has no ref-qualifier.
  /// \code
  /// struct X {
  ///   void f() &;
  ///   void g() &&;
  ///   void h();
  /// };
  /// \endcode
  RefQualifierKind getRefQualifier() const {
    return getType()->getAs<FunctionProtoType>()->getRefQualifier();
  }
  
  bool hasInlineBody() const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const CXXMethodDecl *D) { return true; }
  static bool classofKind(Kind K) {
    return K >= firstCXXMethod && K <= lastCXXMethod;
  }
};

/// CXXCtorInitializer - Represents a C++ base or member
/// initializer, which is part of a constructor initializer that
/// initializes one non-static member variable or one base class. For
/// example, in the following, both 'A(a)' and 'f(3.14159)' are member
/// initializers:
///
/// @code
/// class A { };
/// class B : public A {
///   float f;
/// public:
///   B(A& a) : A(a), f(3.14159) { }
/// };
/// @endcode
class CXXCtorInitializer {
  /// \brief Either the base class name (stored as a TypeSourceInfo*), an normal
  /// field (FieldDecl), anonymous field (IndirectFieldDecl*), or target
  /// constructor (CXXConstructorDecl*) being initialized.
  llvm::PointerUnion4<TypeSourceInfo *, FieldDecl *, IndirectFieldDecl *,
                      CXXConstructorDecl *>
    Initializee;
  
  /// \brief The source location for the field name or, for a base initializer
  /// pack expansion, the location of the ellipsis. In the case of a delegating
  /// constructor, it will still include the type's source location as the
  /// Initializee points to the CXXConstructorDecl (to allow loop detection).
  SourceLocation MemberOrEllipsisLocation;
  
  /// \brief The argument used to initialize the base or member, which may
  /// end up constructing an object (when multiple arguments are involved).
  /// If 0, this is a field initializer, and the in-class member initializer 
  /// will be used.
  Stmt *Init;
  
  /// LParenLoc - Location of the left paren of the ctor-initializer.
  SourceLocation LParenLoc;

  /// RParenLoc - Location of the right paren of the ctor-initializer.
  SourceLocation RParenLoc;

  /// IsVirtual - If the initializer is a base initializer, this keeps track
  /// of whether the base is virtual or not.
  bool IsVirtual : 1;

  /// IsWritten - Whether or not the initializer is explicitly written
  /// in the sources.
  bool IsWritten : 1;

  /// SourceOrderOrNumArrayIndices - If IsWritten is true, then this
  /// number keeps track of the textual order of this initializer in the
  /// original sources, counting from 0; otherwise, if IsWritten is false,
  /// it stores the number of array index variables stored after this
  /// object in memory.
  unsigned SourceOrderOrNumArrayIndices : 14;

  CXXCtorInitializer(ASTContext &Context, FieldDecl *Member,
                     SourceLocation MemberLoc, SourceLocation L, Expr *Init,
                     SourceLocation R, VarDecl **Indices, unsigned NumIndices);
  
public:
  /// CXXCtorInitializer - Creates a new base-class initializer.
  explicit
  CXXCtorInitializer(ASTContext &Context, TypeSourceInfo *TInfo, bool IsVirtual,
                     SourceLocation L, Expr *Init, SourceLocation R,
                     SourceLocation EllipsisLoc);

  /// CXXCtorInitializer - Creates a new member initializer.
  explicit
  CXXCtorInitializer(ASTContext &Context, FieldDecl *Member,
                     SourceLocation MemberLoc, SourceLocation L, Expr *Init,
                     SourceLocation R);

  /// CXXCtorInitializer - Creates a new anonymous field initializer.
  explicit
  CXXCtorInitializer(ASTContext &Context, IndirectFieldDecl *Member,
                     SourceLocation MemberLoc, SourceLocation L, Expr *Init,
                     SourceLocation R);

  /// CXXCtorInitializer - Creates a new delegating Initializer.
  explicit
  CXXCtorInitializer(ASTContext &Context, SourceLocation D, SourceLocation L,
                     CXXConstructorDecl *Target, Expr *Init, SourceLocation R);

  /// \brief Creates a new member initializer that optionally contains 
  /// array indices used to describe an elementwise initialization.
  static CXXCtorInitializer *Create(ASTContext &Context, FieldDecl *Member,
                                    SourceLocation MemberLoc, SourceLocation L,
                                    Expr *Init, SourceLocation R,
                                    VarDecl **Indices, unsigned NumIndices);
  
  /// isBaseInitializer - Returns true when this initializer is
  /// initializing a base class.
  bool isBaseInitializer() const { return Initializee.is<TypeSourceInfo*>(); }

  /// isMemberInitializer - Returns true when this initializer is
  /// initializing a non-static data member.
  bool isMemberInitializer() const { return Initializee.is<FieldDecl*>(); }

  bool isAnyMemberInitializer() const { 
    return isMemberInitializer() || isIndirectMemberInitializer();
  }

  bool isIndirectMemberInitializer() const {
    return Initializee.is<IndirectFieldDecl*>();
  }

  /// isInClassMemberInitializer - Returns true when this initializer is an
  /// implicit ctor initializer generated for a field with an initializer
  /// defined on the member declaration.
  bool isInClassMemberInitializer() const {
    return !Init;
  }

  /// isDelegatingInitializer - Returns true when this initializer is creating
  /// a delegating constructor.
  bool isDelegatingInitializer() const {
    return Initializee.is<CXXConstructorDecl *>();
  }

  /// \brief Determine whether this initializer is a pack expansion.
  bool isPackExpansion() const { 
    return isBaseInitializer() && MemberOrEllipsisLocation.isValid(); 
  }
  
  // \brief For a pack expansion, returns the location of the ellipsis.
  SourceLocation getEllipsisLoc() const {
    assert(isPackExpansion() && "Initializer is not a pack expansion");
    return MemberOrEllipsisLocation;
  }
           
  /// If this is a base class initializer, returns the type of the 
  /// base class with location information. Otherwise, returns an NULL
  /// type location.
  TypeLoc getBaseClassLoc() const;

  /// If this is a base class initializer, returns the type of the base class.
  /// Otherwise, returns NULL.
  const Type *getBaseClass() const;

  /// Returns whether the base is virtual or not.
  bool isBaseVirtual() const {
    assert(isBaseInitializer() && "Must call this on base initializer!");
    
    return IsVirtual;
  }

  /// \brief Returns the declarator information for a base class initializer.
  TypeSourceInfo *getBaseClassInfo() const {
    return Initializee.dyn_cast<TypeSourceInfo *>();
  }
  
  /// getMember - If this is a member initializer, returns the
  /// declaration of the non-static data member being
  /// initialized. Otherwise, returns NULL.
  FieldDecl *getMember() const {
    if (isMemberInitializer())
      return Initializee.get<FieldDecl*>();
    else
      return 0;
  }
  FieldDecl *getAnyMember() const {
    if (isMemberInitializer())
      return Initializee.get<FieldDecl*>();
    else if (isIndirectMemberInitializer())
      return Initializee.get<IndirectFieldDecl*>()->getAnonField();
    else
      return 0;
  }

  IndirectFieldDecl *getIndirectMember() const {
    if (isIndirectMemberInitializer())
      return Initializee.get<IndirectFieldDecl*>();
    else
      return 0;
  }

  CXXConstructorDecl *getTargetConstructor() const {
    if (isDelegatingInitializer())
      return Initializee.get<CXXConstructorDecl*>();
    else
      return 0;
  }

  SourceLocation getMemberLocation() const { 
    return MemberOrEllipsisLocation;
  }
  
  /// \brief Determine the source location of the initializer.
  SourceLocation getSourceLocation() const;
  
  /// \brief Determine the source range covering the entire initializer.
  SourceRange getSourceRange() const;

  /// isWritten - Returns true if this initializer is explicitly written
  /// in the source code.
  bool isWritten() const { return IsWritten; }

  /// \brief Return the source position of the initializer, counting from 0.
  /// If the initializer was implicit, -1 is returned.
  int getSourceOrder() const {
    return IsWritten ? static_cast<int>(SourceOrderOrNumArrayIndices) : -1;
  }

  /// \brief Set the source order of this initializer. This method can only
  /// be called once for each initializer; it cannot be called on an
  /// initializer having a positive number of (implicit) array indices.
  void setSourceOrder(int pos) {
    assert(!IsWritten &&
           "calling twice setSourceOrder() on the same initializer");
    assert(SourceOrderOrNumArrayIndices == 0 &&
           "setSourceOrder() used when there are implicit array indices");
    assert(pos >= 0 &&
           "setSourceOrder() used to make an initializer implicit");
    IsWritten = true;
    SourceOrderOrNumArrayIndices = static_cast<unsigned>(pos);
  }

  SourceLocation getLParenLoc() const { return LParenLoc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }

  /// \brief Determine the number of implicit array indices used while
  /// described an array member initialization.
  unsigned getNumArrayIndices() const {
    return IsWritten ? 0 : SourceOrderOrNumArrayIndices;
  }

  /// \brief Retrieve a particular array index variable used to 
  /// describe an array member initialization.
  VarDecl *getArrayIndex(unsigned I) {
    assert(I < getNumArrayIndices() && "Out of bounds member array index");
    return reinterpret_cast<VarDecl **>(this + 1)[I];
  }
  const VarDecl *getArrayIndex(unsigned I) const {
    assert(I < getNumArrayIndices() && "Out of bounds member array index");
    return reinterpret_cast<const VarDecl * const *>(this + 1)[I];
  }
  void setArrayIndex(unsigned I, VarDecl *Index) {
    assert(I < getNumArrayIndices() && "Out of bounds member array index");
    reinterpret_cast<VarDecl **>(this + 1)[I] = Index;
  }
  
  /// \brief Get the initializer. This is 0 if this is an in-class initializer
  /// for a non-static data member which has not yet been parsed.
  Expr *getInit() const {
    if (!Init)
      return getAnyMember()->getInClassInitializer();

    return static_cast<Expr*>(Init);
  }
};

/// CXXConstructorDecl - Represents a C++ constructor within a
/// class. For example:
///
/// @code
/// class X {
/// public:
///   explicit X(int); // represented by a CXXConstructorDecl.
/// };
/// @endcode
class CXXConstructorDecl : public CXXMethodDecl {
  /// IsExplicitSpecified - Whether this constructor declaration has the
  /// 'explicit' keyword specified.
  bool IsExplicitSpecified : 1;

  /// ImplicitlyDefined - Whether this constructor was implicitly
  /// defined by the compiler. When false, the constructor was defined
  /// by the user. In C++03, this flag will have the same value as
  /// Implicit. In C++0x, however, a constructor that is
  /// explicitly defaulted (i.e., defined with " = default") will have
  /// @c !Implicit && ImplicitlyDefined.
  bool ImplicitlyDefined : 1;

  /// Support for base and member initializers.
  /// CtorInitializers - The arguments used to initialize the base
  /// or member.
  CXXCtorInitializer **CtorInitializers;
  unsigned NumCtorInitializers;

  CXXConstructorDecl(CXXRecordDecl *RD, SourceLocation StartLoc,
                     const DeclarationNameInfo &NameInfo,
                     QualType T, TypeSourceInfo *TInfo,
                     bool isExplicitSpecified, bool isInline, 
                     bool isImplicitlyDeclared)
    : CXXMethodDecl(CXXConstructor, RD, StartLoc, NameInfo, T, TInfo, false,
                    SC_None, isInline, SourceLocation()),
      IsExplicitSpecified(isExplicitSpecified), ImplicitlyDefined(false),
      CtorInitializers(0), NumCtorInitializers(0) {
    setImplicit(isImplicitlyDeclared);
  }

public:
  static CXXConstructorDecl *Create(ASTContext &C, EmptyShell Empty);
  static CXXConstructorDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                                    SourceLocation StartLoc,
                                    const DeclarationNameInfo &NameInfo,
                                    QualType T, TypeSourceInfo *TInfo,
                                    bool isExplicit,
                                    bool isInline, bool isImplicitlyDeclared);

  /// isExplicitSpecified - Whether this constructor declaration has the
  /// 'explicit' keyword specified.
  bool isExplicitSpecified() const { return IsExplicitSpecified; }
  
  /// isExplicit - Whether this constructor was marked "explicit" or not.
  bool isExplicit() const {
    return cast<CXXConstructorDecl>(getFirstDeclaration())
      ->isExplicitSpecified();
  }

  /// isImplicitlyDefined - Whether this constructor was implicitly
  /// defined. If false, then this constructor was defined by the
  /// user. This operation can only be invoked if the constructor has
  /// already been defined.
  bool isImplicitlyDefined() const {
    assert(isThisDeclarationADefinition() &&
           "Can only get the implicit-definition flag once the "
           "constructor has been defined");
    return ImplicitlyDefined;
  }

  /// setImplicitlyDefined - Set whether this constructor was
  /// implicitly defined or not.
  void setImplicitlyDefined(bool ID) {
    assert(isThisDeclarationADefinition() &&
           "Can only set the implicit-definition flag once the constructor "
           "has been defined");
    ImplicitlyDefined = ID;
  }

  /// init_iterator - Iterates through the member/base initializer list.
  typedef CXXCtorInitializer **init_iterator;

  /// init_const_iterator - Iterates through the memberbase initializer list.
  typedef CXXCtorInitializer * const * init_const_iterator;

  /// init_begin() - Retrieve an iterator to the first initializer.
  init_iterator       init_begin()       { return CtorInitializers; }
  /// begin() - Retrieve an iterator to the first initializer.
  init_const_iterator init_begin() const { return CtorInitializers; }

  /// init_end() - Retrieve an iterator past the last initializer.
  init_iterator       init_end()       {
    return CtorInitializers + NumCtorInitializers;
  }
  /// end() - Retrieve an iterator past the last initializer.
  init_const_iterator init_end() const {
    return CtorInitializers + NumCtorInitializers;
  }

  typedef std::reverse_iterator<init_iterator> init_reverse_iterator;
  typedef std::reverse_iterator<init_const_iterator> init_const_reverse_iterator;

  init_reverse_iterator init_rbegin() {
    return init_reverse_iterator(init_end());
  }
  init_const_reverse_iterator init_rbegin() const {
    return init_const_reverse_iterator(init_end());
  }

  init_reverse_iterator init_rend() {
    return init_reverse_iterator(init_begin());
  }
  init_const_reverse_iterator init_rend() const {
    return init_const_reverse_iterator(init_begin());
  }

  /// getNumArgs - Determine the number of arguments used to
  /// initialize the member or base.
  unsigned getNumCtorInitializers() const {
      return NumCtorInitializers;
  }

  void setNumCtorInitializers(unsigned numCtorInitializers) {
    NumCtorInitializers = numCtorInitializers;
  }

  void setCtorInitializers(CXXCtorInitializer ** initializers) {
    CtorInitializers = initializers;
  }

  /// isDelegatingConstructor - Whether this constructor is a
  /// delegating constructor
  bool isDelegatingConstructor() const {
    return (getNumCtorInitializers() == 1) &&
      CtorInitializers[0]->isDelegatingInitializer();
  }

  /// getTargetConstructor - When this constructor delegates to
  /// another, retrieve the target
  CXXConstructorDecl *getTargetConstructor() const {
    assert(isDelegatingConstructor() &&
           "A non-delegating constructor has no target");
    return CtorInitializers[0]->getTargetConstructor();
  }

  /// isDefaultConstructor - Whether this constructor is a default
  /// constructor (C++ [class.ctor]p5), which can be used to
  /// default-initialize a class of this type.
  bool isDefaultConstructor() const;

  /// isCopyConstructor - Whether this constructor is a copy
  /// constructor (C++ [class.copy]p2, which can be used to copy the
  /// class. @p TypeQuals will be set to the qualifiers on the
  /// argument type. For example, @p TypeQuals would be set to @c
  /// QualType::Const for the following copy constructor:
  ///
  /// @code
  /// class X {
  /// public:
  ///   X(const X&);
  /// };
  /// @endcode
  bool isCopyConstructor(unsigned &TypeQuals) const;

  /// isCopyConstructor - Whether this constructor is a copy
  /// constructor (C++ [class.copy]p2, which can be used to copy the
  /// class.
  bool isCopyConstructor() const {
    unsigned TypeQuals = 0;
    return isCopyConstructor(TypeQuals);
  }

  /// \brief Determine whether this constructor is a move constructor
  /// (C++0x [class.copy]p3), which can be used to move values of the class.
  ///
  /// \param TypeQuals If this constructor is a move constructor, will be set
  /// to the type qualifiers on the referent of the first parameter's type.
  bool isMoveConstructor(unsigned &TypeQuals) const;

  /// \brief Determine whether this constructor is a move constructor
  /// (C++0x [class.copy]p3), which can be used to move values of the class.
  bool isMoveConstructor() const {
    unsigned TypeQuals = 0;
    return isMoveConstructor(TypeQuals);
  }

  /// \brief Determine whether this is a copy or move constructor.
  ///
  /// \param TypeQuals Will be set to the type qualifiers on the reference
  /// parameter, if in fact this is a copy or move constructor.
  bool isCopyOrMoveConstructor(unsigned &TypeQuals) const;

  /// \brief Determine whether this a copy or move constructor.
  bool isCopyOrMoveConstructor() const {
    unsigned Quals;
    return isCopyOrMoveConstructor(Quals);
  }

  /// isConvertingConstructor - Whether this constructor is a
  /// converting constructor (C++ [class.conv.ctor]), which can be
  /// used for user-defined conversions.
  bool isConvertingConstructor(bool AllowExplicit) const;

  /// \brief Determine whether this is a member template specialization that
  /// would copy the object to itself. Such constructors are never used to copy
  /// an object.
  bool isSpecializationCopyingObject() const;

  /// \brief Get the constructor that this inheriting constructor is based on.
  const CXXConstructorDecl *getInheritedConstructor() const;

  /// \brief Set the constructor that this inheriting constructor is based on.
  void setInheritedConstructor(const CXXConstructorDecl *BaseCtor);

  const CXXConstructorDecl *getCanonicalDecl() const {
    return cast<CXXConstructorDecl>(FunctionDecl::getCanonicalDecl());
  }
  CXXConstructorDecl *getCanonicalDecl() {
    return cast<CXXConstructorDecl>(FunctionDecl::getCanonicalDecl());
  }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const CXXConstructorDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == CXXConstructor; }
  
  friend class ASTDeclReader;
  friend class ASTDeclWriter;
};

/// CXXDestructorDecl - Represents a C++ destructor within a
/// class. For example:
///
/// @code
/// class X {
/// public:
///   ~X(); // represented by a CXXDestructorDecl.
/// };
/// @endcode
class CXXDestructorDecl : public CXXMethodDecl {
  /// ImplicitlyDefined - Whether this destructor was implicitly
  /// defined by the compiler. When false, the destructor was defined
  /// by the user. In C++03, this flag will have the same value as
  /// Implicit. In C++0x, however, a destructor that is
  /// explicitly defaulted (i.e., defined with " = default") will have
  /// @c !Implicit && ImplicitlyDefined.
  bool ImplicitlyDefined : 1;

  FunctionDecl *OperatorDelete;
  
  CXXDestructorDecl(CXXRecordDecl *RD, SourceLocation StartLoc,
                    const DeclarationNameInfo &NameInfo,
                    QualType T, TypeSourceInfo *TInfo,
                    bool isInline, bool isImplicitlyDeclared)
    : CXXMethodDecl(CXXDestructor, RD, StartLoc, NameInfo, T, TInfo, false,
                    SC_None, isInline, SourceLocation()),
      ImplicitlyDefined(false), OperatorDelete(0) {
    setImplicit(isImplicitlyDeclared);
  }

public:
  static CXXDestructorDecl *Create(ASTContext& C, EmptyShell Empty);
  static CXXDestructorDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                                   SourceLocation StartLoc,
                                   const DeclarationNameInfo &NameInfo,
                                   QualType T, TypeSourceInfo* TInfo,
                                   bool isInline,
                                   bool isImplicitlyDeclared);

  /// isImplicitlyDefined - Whether this destructor was implicitly
  /// defined. If false, then this destructor was defined by the
  /// user. This operation can only be invoked if the destructor has
  /// already been defined.
  bool isImplicitlyDefined() const {
    assert(isThisDeclarationADefinition() &&
           "Can only get the implicit-definition flag once the destructor has been defined");
    return ImplicitlyDefined;
  }

  /// setImplicitlyDefined - Set whether this destructor was
  /// implicitly defined or not.
  void setImplicitlyDefined(bool ID) {
    assert(isThisDeclarationADefinition() &&
           "Can only set the implicit-definition flag once the destructor has been defined");
    ImplicitlyDefined = ID;
  }

  void setOperatorDelete(FunctionDecl *OD) { OperatorDelete = OD; }
  const FunctionDecl *getOperatorDelete() const { return OperatorDelete; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const CXXDestructorDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == CXXDestructor; }
  
  friend class ASTDeclReader;
  friend class ASTDeclWriter;
};

/// CXXConversionDecl - Represents a C++ conversion function within a
/// class. For example:
///
/// @code
/// class X {
/// public:
///   operator bool();
/// };
/// @endcode
class CXXConversionDecl : public CXXMethodDecl {
  /// IsExplicitSpecified - Whether this conversion function declaration is 
  /// marked "explicit", meaning that it can only be applied when the user
  /// explicitly wrote a cast. This is a C++0x feature.
  bool IsExplicitSpecified : 1;

  CXXConversionDecl(CXXRecordDecl *RD, SourceLocation StartLoc,
                    const DeclarationNameInfo &NameInfo,
                    QualType T, TypeSourceInfo *TInfo,
                    bool isInline, bool isExplicitSpecified,
                    SourceLocation EndLocation)
    : CXXMethodDecl(CXXConversion, RD, StartLoc, NameInfo, T, TInfo, false,
                    SC_None, isInline, EndLocation),
      IsExplicitSpecified(isExplicitSpecified) { }

public:
  static CXXConversionDecl *Create(ASTContext &C, EmptyShell Empty);
  static CXXConversionDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                                   SourceLocation StartLoc,
                                   const DeclarationNameInfo &NameInfo,
                                   QualType T, TypeSourceInfo *TInfo,
                                   bool isInline, bool isExplicit,
                                   SourceLocation EndLocation);

  /// IsExplicitSpecified - Whether this conversion function declaration is 
  /// marked "explicit", meaning that it can only be applied when the user
  /// explicitly wrote a cast. This is a C++0x feature.
  bool isExplicitSpecified() const { return IsExplicitSpecified; }

  /// isExplicit - Whether this is an explicit conversion operator
  /// (C++0x only). Explicit conversion operators are only considered
  /// when the user has explicitly written a cast.
  bool isExplicit() const {
    return cast<CXXConversionDecl>(getFirstDeclaration())
      ->isExplicitSpecified();
  }

  /// getConversionType - Returns the type that this conversion
  /// function is converting to.
  QualType getConversionType() const {
    return getType()->getAs<FunctionType>()->getResultType();
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const CXXConversionDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == CXXConversion; }
  
  friend class ASTDeclReader;
  friend class ASTDeclWriter;
};

/// LinkageSpecDecl - This represents a linkage specification.  For example:
///   extern "C" void foo();
///
class LinkageSpecDecl : public Decl, public DeclContext {
public:
  /// LanguageIDs - Used to represent the language in a linkage
  /// specification.  The values are part of the serialization abi for
  /// ASTs and cannot be changed without altering that abi.  To help
  /// ensure a stable abi for this, we choose the DW_LANG_ encodings
  /// from the dwarf standard.
  enum LanguageIDs {
    lang_c = /* DW_LANG_C */ 0x0002,
    lang_cxx = /* DW_LANG_C_plus_plus */ 0x0004
  };
private:
  /// Language - The language for this linkage specification.
  LanguageIDs Language;
  /// ExternLoc - The source location for the extern keyword.
  SourceLocation ExternLoc;
  /// RBraceLoc - The source location for the right brace (if valid).
  SourceLocation RBraceLoc;

  LinkageSpecDecl(DeclContext *DC, SourceLocation ExternLoc,
                  SourceLocation LangLoc, LanguageIDs lang,
                  SourceLocation RBLoc)
    : Decl(LinkageSpec, DC, LangLoc), DeclContext(LinkageSpec),
      Language(lang), ExternLoc(ExternLoc), RBraceLoc(RBLoc) { }

public:
  static LinkageSpecDecl *Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation ExternLoc,
                                 SourceLocation LangLoc, LanguageIDs Lang,
                                 SourceLocation RBraceLoc = SourceLocation());

  /// \brief Return the language specified by this linkage specification.
  LanguageIDs getLanguage() const { return Language; }
  /// \brief Set the language specified by this linkage specification.
  void setLanguage(LanguageIDs L) { Language = L; }

  /// \brief Determines whether this linkage specification had braces in
  /// its syntactic form.
  bool hasBraces() const { return RBraceLoc.isValid(); }

  SourceLocation getExternLoc() const { return ExternLoc; }
  SourceLocation getRBraceLoc() const { return RBraceLoc; }
  void setExternLoc(SourceLocation L) { ExternLoc = L; }
  void setRBraceLoc(SourceLocation L) { RBraceLoc = L; }

  SourceLocation getLocEnd() const {
    if (hasBraces())
      return getRBraceLoc();
    // No braces: get the end location of the (only) declaration in context
    // (if present).
    return decls_empty() ? getLocation() : decls_begin()->getLocEnd();
  }

  SourceRange getSourceRange() const {
    return SourceRange(ExternLoc, getLocEnd());
  }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const LinkageSpecDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == LinkageSpec; }
  static DeclContext *castToDeclContext(const LinkageSpecDecl *D) {
    return static_cast<DeclContext *>(const_cast<LinkageSpecDecl*>(D));
  }
  static LinkageSpecDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<LinkageSpecDecl *>(const_cast<DeclContext*>(DC));
  }
};

/// UsingDirectiveDecl - Represents C++ using-directive. For example:
///
///    using namespace std;
///
// NB: UsingDirectiveDecl should be Decl not NamedDecl, but we provide
// artificial name, for all using-directives in order to store
// them in DeclContext effectively.
class UsingDirectiveDecl : public NamedDecl {
  /// \brief The location of the "using" keyword.
  SourceLocation UsingLoc;
  
  /// SourceLocation - Location of 'namespace' token.
  SourceLocation NamespaceLoc;

  /// \brief The nested-name-specifier that precedes the namespace.
  NestedNameSpecifierLoc QualifierLoc;

  /// NominatedNamespace - Namespace nominated by using-directive.
  NamedDecl *NominatedNamespace;

  /// Enclosing context containing both using-directive and nominated
  /// namespace.
  DeclContext *CommonAncestor;

  /// getUsingDirectiveName - Returns special DeclarationName used by
  /// using-directives. This is only used by DeclContext for storing
  /// UsingDirectiveDecls in its lookup structure.
  static DeclarationName getName() {
    return DeclarationName::getUsingDirectiveName();
  }

  UsingDirectiveDecl(DeclContext *DC, SourceLocation UsingLoc,
                     SourceLocation NamespcLoc,
                     NestedNameSpecifierLoc QualifierLoc,
                     SourceLocation IdentLoc,
                     NamedDecl *Nominated,
                     DeclContext *CommonAncestor)
    : NamedDecl(UsingDirective, DC, IdentLoc, getName()), UsingLoc(UsingLoc),
      NamespaceLoc(NamespcLoc), QualifierLoc(QualifierLoc),
      NominatedNamespace(Nominated), CommonAncestor(CommonAncestor) { }

public:
  /// \brief Retrieve the nested-name-specifier that qualifies the
  /// name of the namespace, with source-location information.
  NestedNameSpecifierLoc getQualifierLoc() const { return QualifierLoc; }
  
  /// \brief Retrieve the nested-name-specifier that qualifies the
  /// name of the namespace.
  NestedNameSpecifier *getQualifier() const { 
    return QualifierLoc.getNestedNameSpecifier(); 
  }

  NamedDecl *getNominatedNamespaceAsWritten() { return NominatedNamespace; }
  const NamedDecl *getNominatedNamespaceAsWritten() const {
    return NominatedNamespace;
  }

  /// getNominatedNamespace - Returns namespace nominated by using-directive.
  NamespaceDecl *getNominatedNamespace();

  const NamespaceDecl *getNominatedNamespace() const {
    return const_cast<UsingDirectiveDecl*>(this)->getNominatedNamespace();
  }

  /// \brief Returns the common ancestor context of this using-directive and
  /// its nominated namespace.
  DeclContext *getCommonAncestor() { return CommonAncestor; }
  const DeclContext *getCommonAncestor() const { return CommonAncestor; }

  /// \brief Return the location of the "using" keyword.
  SourceLocation getUsingLoc() const { return UsingLoc; }
  
  // FIXME: Could omit 'Key' in name.
  /// getNamespaceKeyLocation - Returns location of namespace keyword.
  SourceLocation getNamespaceKeyLocation() const { return NamespaceLoc; }

  /// getIdentLocation - Returns location of identifier.
  SourceLocation getIdentLocation() const { return getLocation(); }

  static UsingDirectiveDecl *Create(ASTContext &C, DeclContext *DC,
                                    SourceLocation UsingLoc,
                                    SourceLocation NamespaceLoc,
                                    NestedNameSpecifierLoc QualifierLoc,
                                    SourceLocation IdentLoc,
                                    NamedDecl *Nominated,
                                    DeclContext *CommonAncestor);

  SourceRange getSourceRange() const {
    return SourceRange(UsingLoc, getLocation());
  }
  
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const UsingDirectiveDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == UsingDirective; }

  // Friend for getUsingDirectiveName.
  friend class DeclContext;
  
  friend class ASTDeclReader;
};

/// NamespaceAliasDecl - Represents a C++ namespace alias. For example:
///
/// @code
/// namespace Foo = Bar;
/// @endcode
class NamespaceAliasDecl : public NamedDecl {
  /// \brief The location of the "namespace" keyword.
  SourceLocation NamespaceLoc;

  /// IdentLoc - Location of namespace identifier. Accessed by TargetNameLoc.
  SourceLocation IdentLoc;
  
  /// \brief The nested-name-specifier that precedes the namespace.
  NestedNameSpecifierLoc QualifierLoc;
  
  /// Namespace - The Decl that this alias points to. Can either be a
  /// NamespaceDecl or a NamespaceAliasDecl.
  NamedDecl *Namespace;

  NamespaceAliasDecl(DeclContext *DC, SourceLocation NamespaceLoc,
                     SourceLocation AliasLoc, IdentifierInfo *Alias,
                     NestedNameSpecifierLoc QualifierLoc,
                     SourceLocation IdentLoc, NamedDecl *Namespace)
    : NamedDecl(NamespaceAlias, DC, AliasLoc, Alias), 
      NamespaceLoc(NamespaceLoc), IdentLoc(IdentLoc),
      QualifierLoc(QualifierLoc), Namespace(Namespace) { }

  friend class ASTDeclReader;
  
public:
  /// \brief Retrieve the nested-name-specifier that qualifies the
  /// name of the namespace, with source-location information.
  NestedNameSpecifierLoc getQualifierLoc() const { return QualifierLoc; }
  
  /// \brief Retrieve the nested-name-specifier that qualifies the
  /// name of the namespace.
  NestedNameSpecifier *getQualifier() const { 
    return QualifierLoc.getNestedNameSpecifier(); 
  }
  
  /// \brief Retrieve the namespace declaration aliased by this directive.
  NamespaceDecl *getNamespace() {
    if (NamespaceAliasDecl *AD = dyn_cast<NamespaceAliasDecl>(Namespace))
      return AD->getNamespace();

    return cast<NamespaceDecl>(Namespace);
  }

  const NamespaceDecl *getNamespace() const {
    return const_cast<NamespaceAliasDecl*>(this)->getNamespace();
  }

  /// Returns the location of the alias name, i.e. 'foo' in
  /// "namespace foo = ns::bar;".
  SourceLocation getAliasLoc() const { return getLocation(); }

  /// Returns the location of the 'namespace' keyword.
  SourceLocation getNamespaceLoc() const { return NamespaceLoc; }

  /// Returns the location of the identifier in the named namespace.
  SourceLocation getTargetNameLoc() const { return IdentLoc; }

  /// \brief Retrieve the namespace that this alias refers to, which
  /// may either be a NamespaceDecl or a NamespaceAliasDecl.
  NamedDecl *getAliasedNamespace() const { return Namespace; }

  static NamespaceAliasDecl *Create(ASTContext &C, DeclContext *DC,
                                    SourceLocation NamespaceLoc, 
                                    SourceLocation AliasLoc,
                                    IdentifierInfo *Alias,
                                    NestedNameSpecifierLoc QualifierLoc,
                                    SourceLocation IdentLoc,
                                    NamedDecl *Namespace);

  virtual SourceRange getSourceRange() const {
    return SourceRange(NamespaceLoc, IdentLoc);
  }
  
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const NamespaceAliasDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == NamespaceAlias; }
};

/// UsingShadowDecl - Represents a shadow declaration introduced into
/// a scope by a (resolved) using declaration.  For example,
///
/// namespace A {
///   void foo();
/// }
/// namespace B {
///   using A::foo(); // <- a UsingDecl
///                   // Also creates a UsingShadowDecl for A::foo in B
/// }
///
class UsingShadowDecl : public NamedDecl {
  /// The referenced declaration.
  NamedDecl *Underlying;

  /// \brief The using declaration which introduced this decl or the next using
  /// shadow declaration contained in the aforementioned using declaration.
  NamedDecl *UsingOrNextShadow;
  friend class UsingDecl;

  UsingShadowDecl(DeclContext *DC, SourceLocation Loc, UsingDecl *Using,
                  NamedDecl *Target)
    : NamedDecl(UsingShadow, DC, Loc, DeclarationName()),
      Underlying(Target),
      UsingOrNextShadow(reinterpret_cast<NamedDecl *>(Using)) {
    if (Target) {
      setDeclName(Target->getDeclName());
      IdentifierNamespace = Target->getIdentifierNamespace();
    }
    setImplicit();
  }

public:
  static UsingShadowDecl *Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation Loc, UsingDecl *Using,
                                 NamedDecl *Target) {
    return new (C) UsingShadowDecl(DC, Loc, Using, Target);
  }

  /// \brief Gets the underlying declaration which has been brought into the
  /// local scope.
  NamedDecl *getTargetDecl() const { return Underlying; }

  /// \brief Sets the underlying declaration which has been brought into the
  /// local scope.
  void setTargetDecl(NamedDecl* ND) {
    assert(ND && "Target decl is null!");
    Underlying = ND;
    IdentifierNamespace = ND->getIdentifierNamespace();
  }

  /// \brief Gets the using declaration to which this declaration is tied.
  UsingDecl *getUsingDecl() const;

  /// \brief The next using shadow declaration contained in the shadow decl
  /// chain of the using declaration which introduced this decl.
  UsingShadowDecl *getNextUsingShadowDecl() const {
    return dyn_cast_or_null<UsingShadowDecl>(UsingOrNextShadow);
  }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const UsingShadowDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == Decl::UsingShadow; }

  friend class ASTDeclReader;
  friend class ASTDeclWriter;
};

/// UsingDecl - Represents a C++ using-declaration. For example:
///    using someNameSpace::someIdentifier;
class UsingDecl : public NamedDecl {
  /// \brief The source location of the "using" location itself.
  SourceLocation UsingLocation;

  /// \brief The nested-name-specifier that precedes the name.
  NestedNameSpecifierLoc QualifierLoc;

  /// DNLoc - Provides source/type location info for the
  /// declaration name embedded in the ValueDecl base class.
  DeclarationNameLoc DNLoc;

  /// \brief The first shadow declaration of the shadow decl chain associated
  /// with this using declaration.
  UsingShadowDecl *FirstUsingShadow;

  // \brief Has 'typename' keyword.
  bool IsTypeName;

  UsingDecl(DeclContext *DC, SourceLocation UL, 
            NestedNameSpecifierLoc QualifierLoc,
            const DeclarationNameInfo &NameInfo, bool IsTypeNameArg)
    : NamedDecl(Using, DC, NameInfo.getLoc(), NameInfo.getName()),
      UsingLocation(UL), QualifierLoc(QualifierLoc),
      DNLoc(NameInfo.getInfo()), FirstUsingShadow(0),IsTypeName(IsTypeNameArg) {
  }

public:
  /// \brief Returns the source location of the "using" keyword.
  SourceLocation getUsingLocation() const { return UsingLocation; }

  /// \brief Set the source location of the 'using' keyword.
  void setUsingLocation(SourceLocation L) { UsingLocation = L; }

  /// \brief Retrieve the nested-name-specifier that qualifies the name,
  /// with source-location information.
  NestedNameSpecifierLoc getQualifierLoc() const { return QualifierLoc; }

  /// \brief Retrieve the nested-name-specifier that qualifies the name.
  NestedNameSpecifier *getQualifier() const { 
    return QualifierLoc.getNestedNameSpecifier(); 
  }

  DeclarationNameInfo getNameInfo() const {
    return DeclarationNameInfo(getDeclName(), getLocation(), DNLoc);
  }

  /// \brief Return true if the using declaration has 'typename'.
  bool isTypeName() const { return IsTypeName; }

  /// \brief Sets whether the using declaration has 'typename'.
  void setTypeName(bool TN) { IsTypeName = TN; }

  /// \brief Iterates through the using shadow declarations assosiated with
  /// this using declaration.
  class shadow_iterator {
    /// \brief The current using shadow declaration.
    UsingShadowDecl *Current;

  public:
    typedef UsingShadowDecl*          value_type;
    typedef UsingShadowDecl*          reference;
    typedef UsingShadowDecl*          pointer;
    typedef std::forward_iterator_tag iterator_category;
    typedef std::ptrdiff_t            difference_type;

    shadow_iterator() : Current(0) { }
    explicit shadow_iterator(UsingShadowDecl *C) : Current(C) { }

    reference operator*() const { return Current; }
    pointer operator->() const { return Current; }

    shadow_iterator& operator++() {
      Current = Current->getNextUsingShadowDecl();
      return *this;
    }

    shadow_iterator operator++(int) {
      shadow_iterator tmp(*this);
      ++(*this);
      return tmp;
    }

    friend bool operator==(shadow_iterator x, shadow_iterator y) {
      return x.Current == y.Current;
    }
    friend bool operator!=(shadow_iterator x, shadow_iterator y) {
      return x.Current != y.Current;
    }
  };

  shadow_iterator shadow_begin() const {
    return shadow_iterator(FirstUsingShadow);
  }
  shadow_iterator shadow_end() const { return shadow_iterator(); }

  /// \brief Return the number of shadowed declarations associated with this
  /// using declaration.
  unsigned shadow_size() const {
    return std::distance(shadow_begin(), shadow_end());
  }

  void addShadowDecl(UsingShadowDecl *S);
  void removeShadowDecl(UsingShadowDecl *S);

  static UsingDecl *Create(ASTContext &C, DeclContext *DC,
                           SourceLocation UsingL,
                           NestedNameSpecifierLoc QualifierLoc,
                           const DeclarationNameInfo &NameInfo,
                           bool IsTypeNameArg);

  SourceRange getSourceRange() const {
    return SourceRange(UsingLocation, getNameInfo().getEndLoc());
  }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const UsingDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == Using; }

  friend class ASTDeclReader;
  friend class ASTDeclWriter;
};

/// UnresolvedUsingValueDecl - Represents a dependent using
/// declaration which was not marked with 'typename'.  Unlike
/// non-dependent using declarations, these *only* bring through
/// non-types; otherwise they would break two-phase lookup.
///
/// template <class T> class A : public Base<T> {
///   using Base<T>::foo;
/// };
class UnresolvedUsingValueDecl : public ValueDecl {
  /// \brief The source location of the 'using' keyword
  SourceLocation UsingLocation;

  /// \brief The nested-name-specifier that precedes the name.
  NestedNameSpecifierLoc QualifierLoc;

  /// DNLoc - Provides source/type location info for the
  /// declaration name embedded in the ValueDecl base class.
  DeclarationNameLoc DNLoc;

  UnresolvedUsingValueDecl(DeclContext *DC, QualType Ty,
                           SourceLocation UsingLoc, 
                           NestedNameSpecifierLoc QualifierLoc,
                           const DeclarationNameInfo &NameInfo)
    : ValueDecl(UnresolvedUsingValue, DC,
                NameInfo.getLoc(), NameInfo.getName(), Ty),
      UsingLocation(UsingLoc), QualifierLoc(QualifierLoc),
      DNLoc(NameInfo.getInfo())
  { }

public:
  /// \brief Returns the source location of the 'using' keyword.
  SourceLocation getUsingLoc() const { return UsingLocation; }

  /// \brief Set the source location of the 'using' keyword.
  void setUsingLoc(SourceLocation L) { UsingLocation = L; }

  /// \brief Retrieve the nested-name-specifier that qualifies the name,
  /// with source-location information.
  NestedNameSpecifierLoc getQualifierLoc() const { return QualifierLoc; }

  /// \brief Retrieve the nested-name-specifier that qualifies the name.
  NestedNameSpecifier *getQualifier() const { 
    return QualifierLoc.getNestedNameSpecifier(); 
  }
  
  DeclarationNameInfo getNameInfo() const {
    return DeclarationNameInfo(getDeclName(), getLocation(), DNLoc);
  }

  static UnresolvedUsingValueDecl *
    Create(ASTContext &C, DeclContext *DC, SourceLocation UsingLoc,
           NestedNameSpecifierLoc QualifierLoc, 
           const DeclarationNameInfo &NameInfo);

  SourceRange getSourceRange() const {
    return SourceRange(UsingLocation, getNameInfo().getEndLoc());
  }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const UnresolvedUsingValueDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == UnresolvedUsingValue; }

  friend class ASTDeclReader;
  friend class ASTDeclWriter;
};

/// UnresolvedUsingTypenameDecl - Represents a dependent using
/// declaration which was marked with 'typename'.
///
/// template <class T> class A : public Base<T> {
///   using typename Base<T>::foo;
/// };
///
/// The type associated with a unresolved using typename decl is
/// currently always a typename type.
class UnresolvedUsingTypenameDecl : public TypeDecl {
  /// \brief The source location of the 'using' keyword
  SourceLocation UsingLocation;

  /// \brief The source location of the 'typename' keyword
  SourceLocation TypenameLocation;

  /// \brief The nested-name-specifier that precedes the name.
  NestedNameSpecifierLoc QualifierLoc;

  UnresolvedUsingTypenameDecl(DeclContext *DC, SourceLocation UsingLoc,
                              SourceLocation TypenameLoc,
                              NestedNameSpecifierLoc QualifierLoc,
                              SourceLocation TargetNameLoc, 
                              IdentifierInfo *TargetName)
    : TypeDecl(UnresolvedUsingTypename, DC, TargetNameLoc, TargetName,
               UsingLoc),
      TypenameLocation(TypenameLoc), QualifierLoc(QualifierLoc) { }

  friend class ASTDeclReader;
  
public:
  /// \brief Returns the source location of the 'using' keyword.
  SourceLocation getUsingLoc() const { return getLocStart(); }

  /// \brief Returns the source location of the 'typename' keyword.
  SourceLocation getTypenameLoc() const { return TypenameLocation; }

  /// \brief Retrieve the nested-name-specifier that qualifies the name,
  /// with source-location information.
  NestedNameSpecifierLoc getQualifierLoc() const { return QualifierLoc; }

  /// \brief Retrieve the nested-name-specifier that qualifies the name.
  NestedNameSpecifier *getQualifier() const { 
    return QualifierLoc.getNestedNameSpecifier(); 
  }

  static UnresolvedUsingTypenameDecl *
    Create(ASTContext &C, DeclContext *DC, SourceLocation UsingLoc,
           SourceLocation TypenameLoc, NestedNameSpecifierLoc QualifierLoc,
           SourceLocation TargetNameLoc, DeclarationName TargetName);

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const UnresolvedUsingTypenameDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == UnresolvedUsingTypename; }
};

/// StaticAssertDecl - Represents a C++0x static_assert declaration.
class StaticAssertDecl : public Decl {
  Expr *AssertExpr;
  StringLiteral *Message;
  SourceLocation RParenLoc;

  StaticAssertDecl(DeclContext *DC, SourceLocation StaticAssertLoc,
                   Expr *assertexpr, StringLiteral *message,
                   SourceLocation RParenLoc)
  : Decl(StaticAssert, DC, StaticAssertLoc), AssertExpr(assertexpr),
    Message(message), RParenLoc(RParenLoc) { }

public:
  static StaticAssertDecl *Create(ASTContext &C, DeclContext *DC,
                                  SourceLocation StaticAssertLoc,
                                  Expr *AssertExpr, StringLiteral *Message,
                                  SourceLocation RParenLoc);

  Expr *getAssertExpr() { return AssertExpr; }
  const Expr *getAssertExpr() const { return AssertExpr; }

  StringLiteral *getMessage() { return Message; }
  const StringLiteral *getMessage() const { return Message; }

  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  SourceRange getSourceRange() const {
    return SourceRange(getLocation(), getRParenLoc());
  }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(StaticAssertDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == StaticAssert; }

  friend class ASTDeclReader;
};

/// Insertion operator for diagnostics.  This allows sending AccessSpecifier's
/// into a diagnostic with <<.
const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                    AccessSpecifier AS);

} // end namespace clang

#endif
