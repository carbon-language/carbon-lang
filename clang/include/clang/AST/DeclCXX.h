//===-- DeclCXX.h - Classes for representing C++ declarations -*- C++ -*-=====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the C++ Decl subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLCXX_H
#define LLVM_CLANG_AST_DECLCXX_H

#include "clang/AST/Expr.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/SmallVector.h"

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

/// OverloadedFunctionDecl - An instance of this class represents a
/// set of overloaded functions. All of the functions have the same
/// name and occur within the same scope.
///
/// An OverloadedFunctionDecl has no ownership over the FunctionDecl
/// nodes it contains. Rather, the FunctionDecls are owned by the
/// enclosing scope (which also owns the OverloadedFunctionDecl
/// node). OverloadedFunctionDecl is used primarily to store a set of
/// overloaded functions for name lookup.
class OverloadedFunctionDecl : public NamedDecl {
protected:
  OverloadedFunctionDecl(DeclContext *DC, DeclarationName N)
    : NamedDecl(OverloadedFunction, DC, SourceLocation(), N) { }

  /// Functions - the set of overloaded functions contained in this
  /// overload set.
  llvm::SmallVector<AnyFunctionDecl, 4> Functions;

  // FIXME: This should go away when we stop using
  // OverloadedFunctionDecl to store conversions in CXXRecordDecl.
  friend class CXXRecordDecl;

public:
  typedef llvm::SmallVector<AnyFunctionDecl, 4>::iterator function_iterator;
  typedef llvm::SmallVector<AnyFunctionDecl, 4>::const_iterator
    function_const_iterator;

  static OverloadedFunctionDecl *Create(ASTContext &C, DeclContext *DC,
                                        DeclarationName N);

  /// \brief Add a new overloaded function or function template to the set
  /// of overloaded function templates.
  void addOverload(AnyFunctionDecl F);

  function_iterator function_begin() { return Functions.begin(); }
  function_iterator function_end() { return Functions.end(); }
  function_const_iterator function_begin() const { return Functions.begin(); }
  function_const_iterator function_end() const { return Functions.end(); }

  /// \brief Returns the number of overloaded functions stored in
  /// this set.
  unsigned size() const { return Functions.size(); }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() == OverloadedFunction;
  }
  static bool classof(const OverloadedFunctionDecl *D) { return true; }
};

/// \brief Provides uniform iteration syntax for an overload set, function,
/// or function template.
class OverloadIterator {
  /// \brief An overloaded function set, function declaration, or
  /// function template declaration.
  NamedDecl *D;

  /// \brief If the declaration is an overloaded function set, this is the
  /// iterator pointing to the current position within that overloaded
  /// function set.
  OverloadedFunctionDecl::function_iterator Iter;

public:
  typedef AnyFunctionDecl value_type;
  typedef value_type      reference;
  typedef NamedDecl      *pointer;
  typedef int             difference_type;
  typedef std::forward_iterator_tag iterator_category;

  OverloadIterator() : D(0) { }

  OverloadIterator(FunctionDecl *FD) : D(FD) { }
  OverloadIterator(FunctionTemplateDecl *FTD)
    : D(reinterpret_cast<NamedDecl*>(FTD)) { }
  OverloadIterator(OverloadedFunctionDecl *Ovl)
    : D(Ovl), Iter(Ovl->function_begin()) { }

  OverloadIterator(NamedDecl *ND);

  reference operator*() const;

  pointer operator->() const { return (**this).get(); }

  OverloadIterator &operator++();

  OverloadIterator operator++(int) {
    OverloadIterator Temp(*this);
    ++(*this);
    return Temp;
  }

  bool Equals(const OverloadIterator &Other) const;
};

inline bool operator==(const OverloadIterator &X, const OverloadIterator &Y) {
  return X.Equals(Y);
}

inline bool operator!=(const OverloadIterator &X, const OverloadIterator &Y) {
  return !(X == Y);
}

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

  /// BaseType - The type of the base class. This will be a class or
  /// struct (or a typedef of such).
  QualType BaseType;

public:
  CXXBaseSpecifier() { }

  CXXBaseSpecifier(SourceRange R, bool V, bool BC, AccessSpecifier A, QualType T)
    : Range(R), Virtual(V), BaseOfClass(BC), Access(A), BaseType(T) { }

  /// getSourceRange - Retrieves the source range that contains the
  /// entire base specifier.
  SourceRange getSourceRange() const { return Range; }

  /// isVirtual - Determines whether the base class is a virtual base
  /// class (or not).
  bool isVirtual() const { return Virtual; }

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
  QualType getType() const { return BaseType; }
};

/// CXXRecordDecl - Represents a C++ struct/union/class.
/// FIXME: This class will disappear once we've properly taught RecordDecl
/// to deal with C++-specific things.
class CXXRecordDecl : public RecordDecl {
  /// UserDeclaredConstructor - True when this class has a
  /// user-declared constructor.
  bool UserDeclaredConstructor : 1;

  /// UserDeclaredCopyConstructor - True when this class has a
  /// user-declared copy constructor.
  bool UserDeclaredCopyConstructor : 1;

  /// UserDeclaredCopyAssignment - True when this class has a
  /// user-declared copy assignment operator.
  bool UserDeclaredCopyAssignment : 1;

  /// UserDeclaredDestructor - True when this class has a
  /// user-declared destructor.
  bool UserDeclaredDestructor : 1;

  /// Aggregate - True when this class is an aggregate.
  bool Aggregate : 1;

  /// PlainOldData - True when this class is a POD-type.
  bool PlainOldData : 1;

  /// Empty - true when this class is empty for traits purposes, i.e. has no
  /// data members other than 0-width bit-fields, has no virtual function/base,
  /// and doesn't inherit from a non-empty class. Doesn't take union-ness into
  /// account.
  bool Empty : 1;

  /// Polymorphic - True when this class is polymorphic, i.e. has at least one
  /// virtual member or derives from a polymorphic class.
  bool Polymorphic : 1;

  /// Abstract - True when this class is abstract, i.e. has at least one
  /// pure virtual function, (that can come from a base class).
  bool Abstract : 1;

  /// HasTrivialConstructor - True when this class has a trivial constructor.
  ///
  /// C++ [class.ctor]p5.  A constructor is trivial if it is an
  /// implicitly-declared default constructor and if:
  /// * its class has no virtual functions and no virtual base classes, and
  /// * all the direct base classes of its class have trivial constructors, and
  /// * for all the nonstatic data members of its class that are of class type
  ///   (or array thereof), each such class has a trivial constructor.
  bool HasTrivialConstructor : 1;

  /// HasTrivialCopyConstructor - True when this class has a trivial copy
  /// constructor.
  ///
  /// C++ [class.copy]p6.  A copy constructor for class X is trivial
  /// if it is implicitly declared and if
  /// * class X has no virtual functions and no virtual base classes, and
  /// * each direct base class of X has a trivial copy constructor, and
  /// * for all the nonstatic data members of X that are of class type (or
  ///   array thereof), each such class type has a trivial copy constructor;
  /// otherwise the copy constructor is non-trivial.
  bool HasTrivialCopyConstructor : 1;

  /// HasTrivialCopyAssignment - True when this class has a trivial copy
  /// assignment operator.
  ///
  /// C++ [class.copy]p11.  A copy assignment operator for class X is
  /// trivial if it is implicitly declared and if
  /// * class X has no virtual functions and no virtual base classes, and
  /// * each direct base class of X has a trivial copy assignment operator, and
  /// * for all the nonstatic data members of X that are of class type (or
  ///   array thereof), each such class type has a trivial copy assignment
  ///   operator;
  /// otherwise the copy assignment operator is non-trivial.
  bool HasTrivialCopyAssignment : 1;

  /// HasTrivialDestructor - True when this class has a trivial destructor.
  ///
  /// C++ [class.dtor]p3.  A destructor is trivial if it is an
  /// implicitly-declared destructor and if:
  /// * all of the direct base classes of its class have trivial destructors
  ///   and
  /// * for all of the non-static data members of its class that are of class
  ///   type (or array thereof), each such class has a trivial destructor.
  bool HasTrivialDestructor : 1;

  /// ComputedVisibleConversions - True when visible conversion functions are
  /// already computed and are available.
  bool ComputedVisibleConversions : 1;
  
  /// Bases - Base classes of this class.
  /// FIXME: This is wasted space for a union.
  CXXBaseSpecifier *Bases;

  /// NumBases - The number of base class specifiers in Bases.
  unsigned NumBases;

  /// VBases - direct and indirect virtual base classes of this class.
  CXXBaseSpecifier *VBases;

  /// NumVBases - The number of virtual base class specifiers in VBases.
  unsigned NumVBases;

  /// Conversions - Overload set containing the conversion functions
  /// of this C++ class (but not its inherited conversion
  /// functions). Each of the entries in this overload set is a
  /// CXXConversionDecl. 
  OverloadedFunctionDecl Conversions;

  /// VisibleConversions - Overload set containing the conversion functions
  /// of this C++ class and all those inherited conversion functions that
  /// are visible in this class. Each of the entries in this overload set is
  /// a CXXConversionDecl or a FunctionTemplateDecl.
  OverloadedFunctionDecl VisibleConversions;
  
  /// \brief The template or declaration that this declaration
  /// describes or was instantiated from, respectively.
  ///
  /// For non-templates, this value will be NULL. For record
  /// declarations that describe a class template, this will be a
  /// pointer to a ClassTemplateDecl. For member
  /// classes of class template specializations, this will be the
  /// RecordDecl from which the member class was instantiated.
  llvm::PointerUnion<ClassTemplateDecl*, CXXRecordDecl*>
    TemplateOrInstantiation;
  
  void getNestedVisibleConversionFunctions(CXXRecordDecl *RD);
  
protected:
  CXXRecordDecl(Kind K, TagKind TK, DeclContext *DC,
                SourceLocation L, IdentifierInfo *Id,
                CXXRecordDecl *PrevDecl,
                SourceLocation TKL = SourceLocation());

  ~CXXRecordDecl();

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

  static CXXRecordDecl *Create(ASTContext &C, TagKind TK, DeclContext *DC,
                               SourceLocation L, IdentifierInfo *Id,
                               SourceLocation TKL = SourceLocation(),
                               CXXRecordDecl* PrevDecl=0,
                               bool DelayTypeCreation = false);

  virtual void Destroy(ASTContext& C);

  bool isDynamicClass() const {
    return Polymorphic || NumVBases != 0;
  }

  /// setBases - Sets the base classes of this struct or class.
  void setBases(ASTContext &C,
                CXXBaseSpecifier const * const *Bases, unsigned NumBases);

  /// getNumBases - Retrieves the number of base classes of this
  /// class.
  unsigned getNumBases() const { return NumBases; }

  base_class_iterator       bases_begin()       { return Bases; }
  base_class_const_iterator bases_begin() const { return Bases; }
  base_class_iterator       bases_end()         { return Bases + NumBases; }
  base_class_const_iterator bases_end()   const { return Bases + NumBases; }
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
  unsigned getNumVBases() const { return NumVBases; }

  base_class_iterator       vbases_begin()       { return VBases; }
  base_class_const_iterator vbases_begin() const { return VBases; }
  base_class_iterator       vbases_end()         { return VBases + NumVBases; }
  base_class_const_iterator vbases_end()   const { return VBases + NumVBases; }
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

  /// hasConstCopyConstructor - Determines whether this class has a
  /// copy constructor that accepts a const-qualified argument.
  bool hasConstCopyConstructor(ASTContext &Context) const;

  /// getCopyConstructor - Returns the copy constructor for this class
  CXXConstructorDecl *getCopyConstructor(ASTContext &Context,
                                         unsigned TypeQuals) const;

  /// hasConstCopyAssignment - Determines whether this class has a
  /// copy assignment operator that accepts a const-qualified argument.
  /// It returns its decl in MD if found.
  bool hasConstCopyAssignment(ASTContext &Context,
                              const CXXMethodDecl *&MD) const;

  /// addedConstructor - Notify the class that another constructor has
  /// been added. This routine helps maintain information about the
  /// class based on which constructors have been added.
  void addedConstructor(ASTContext &Context, CXXConstructorDecl *ConDecl);

  /// hasUserDeclaredConstructor - Whether this class has any
  /// user-declared constructors. When true, a default constructor
  /// will not be implicitly declared.
  bool hasUserDeclaredConstructor() const {
    assert((isDefinition() ||
            cast<RecordType>(getTypeForDecl())->isBeingDefined()) &&
           "Incomplete record decl!");
    return UserDeclaredConstructor;
  }

  /// hasUserDeclaredCopyConstructor - Whether this class has a
  /// user-declared copy constructor. When false, a copy constructor
  /// will be implicitly declared.
  bool hasUserDeclaredCopyConstructor() const {
    return UserDeclaredCopyConstructor;
  }

  /// addedAssignmentOperator - Notify the class that another assignment
  /// operator has been added. This routine helps maintain information about the
   /// class based on which operators have been added.
  void addedAssignmentOperator(ASTContext &Context, CXXMethodDecl *OpDecl);

  /// hasUserDeclaredCopyAssignment - Whether this class has a
  /// user-declared copy assignment operator. When false, a copy
  /// assigment operator will be implicitly declared.
  bool hasUserDeclaredCopyAssignment() const {
    return UserDeclaredCopyAssignment;
  }

  /// hasUserDeclaredDestructor - Whether this class has a
  /// user-declared destructor. When false, a destructor will be
  /// implicitly declared.
  bool hasUserDeclaredDestructor() const { return UserDeclaredDestructor; }

  /// setUserDeclaredDestructor - Set whether this class has a
  /// user-declared destructor. If not set by the time the class is
  /// fully defined, a destructor will be implicitly declared.
  void setUserDeclaredDestructor(bool UCD) {
    UserDeclaredDestructor = UCD;
  }

  /// getConversions - Retrieve the overload set containing all of the
  /// conversion functions in this class.
  OverloadedFunctionDecl *getConversionFunctions() {
    assert((this->isDefinition() ||
            cast<RecordType>(getTypeForDecl())->isBeingDefined()) &&
           "getConversionFunctions() called on incomplete type");
    return &Conversions;
  }
  const OverloadedFunctionDecl *getConversionFunctions() const {
    assert((this->isDefinition() ||
            cast<RecordType>(getTypeForDecl())->isBeingDefined()) &&
           "getConversionFunctions() called on incomplete type");
    return &Conversions;
  }

  /// getVisibleConversionFunctions - get all conversion functions visible
  /// in current class; including conversion function templates.
  OverloadedFunctionDecl *getVisibleConversionFunctions();
  /// addVisibleConversionFunction - Add a new conversion function to the
  /// list of visible conversion functions.
  void addVisibleConversionFunction(CXXConversionDecl *ConvDecl);
  
  /// \brief Add a new conversion function template to the list of visible
  /// conversion functions.
  void addVisibleConversionFunction(FunctionTemplateDecl *ConvDecl);
  
  /// addConversionFunction - Add a new conversion function to the
  /// list of conversion functions.
  void addConversionFunction(CXXConversionDecl *ConvDecl);

  /// \brief Add a new conversion function template to the list of conversion
  /// functions.
  void addConversionFunction(FunctionTemplateDecl *ConvDecl);

  /// isAggregate - Whether this class is an aggregate (C++
  /// [dcl.init.aggr]), which is a class with no user-declared
  /// constructors, no private or protected non-static data members,
  /// no base classes, and no virtual functions (C++ [dcl.init.aggr]p1).
  bool isAggregate() const { return Aggregate; }

  /// setAggregate - Set whether this class is an aggregate (C++
  /// [dcl.init.aggr]).
  void setAggregate(bool Agg) { Aggregate = Agg; }

  /// isPOD - Whether this class is a POD-type (C++ [class]p4), which is a class
  /// that is an aggregate that has no non-static non-POD data members, no
  /// reference data members, no user-defined copy assignment operator and no
  /// user-defined destructor.
  bool isPOD() const { return PlainOldData; }

  /// setPOD - Set whether this class is a POD-type (C++ [class]p4).
  void setPOD(bool POD) { PlainOldData = POD; }

  /// isEmpty - Whether this class is empty (C++0x [meta.unary.prop]), which
  /// means it has a virtual function, virtual base, data member (other than
  /// 0-width bit-field) or inherits from a non-empty class. Does NOT include
  /// a check for union-ness.
  bool isEmpty() const { return Empty; }

  /// Set whether this class is empty (C++0x [meta.unary.prop])
  void setEmpty(bool Emp) { Empty = Emp; }

  /// isPolymorphic - Whether this class is polymorphic (C++ [class.virtual]),
  /// which means that the class contains or inherits a virtual function.
  bool isPolymorphic() const { return Polymorphic; }

  /// setPolymorphic - Set whether this class is polymorphic (C++
  /// [class.virtual]).
  void setPolymorphic(bool Poly) { Polymorphic = Poly; }

  /// isAbstract - Whether this class is abstract (C++ [class.abstract]),
  /// which means that the class contains or inherits a pure virtual function.
  bool isAbstract() const { return Abstract; }

  /// setAbstract - Set whether this class is abstract (C++ [class.abstract])
  void setAbstract(bool Abs) { Abstract = Abs; }

  // hasTrivialConstructor - Whether this class has a trivial constructor
  // (C++ [class.ctor]p5)
  bool hasTrivialConstructor() const { return HasTrivialConstructor; }

  // setHasTrivialConstructor - Set whether this class has a trivial constructor
  // (C++ [class.ctor]p5)
  void setHasTrivialConstructor(bool TC) { HasTrivialConstructor = TC; }

  // hasTrivialCopyConstructor - Whether this class has a trivial copy
  // constructor (C++ [class.copy]p6)
  bool hasTrivialCopyConstructor() const { return HasTrivialCopyConstructor; }

  // setHasTrivialCopyConstructor - Set whether this class has a trivial
  // copy constructor (C++ [class.copy]p6)
  void setHasTrivialCopyConstructor(bool TC) { HasTrivialCopyConstructor = TC; }

  // hasTrivialCopyAssignment - Whether this class has a trivial copy
  // assignment operator (C++ [class.copy]p11)
  bool hasTrivialCopyAssignment() const { return HasTrivialCopyAssignment; }

  // setHasTrivialCopyAssignment - Set whether this class has a
  // trivial copy assignment operator (C++ [class.copy]p11)
  void setHasTrivialCopyAssignment(bool TC) { HasTrivialCopyAssignment = TC; }

  // hasTrivialDestructor - Whether this class has a trivial destructor
  // (C++ [class.dtor]p3)
  bool hasTrivialDestructor() const { return HasTrivialDestructor; }

  // setHasTrivialDestructor - Set whether this class has a trivial destructor
  // (C++ [class.dtor]p3)
  void setHasTrivialDestructor(bool TC) { HasTrivialDestructor = TC; }

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
  CXXRecordDecl *getInstantiatedFromMemberClass() const {
    return TemplateOrInstantiation.dyn_cast<CXXRecordDecl*>();
  }

  /// \brief Specify that this record is an instantiation of the
  /// member class RD.
  void setInstantiationOfMemberClass(CXXRecordDecl *RD) {
    TemplateOrInstantiation = RD;
  }

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

  /// getDefaultConstructor - Returns the default constructor for this class
  CXXConstructorDecl *getDefaultConstructor(ASTContext &Context);

  /// getDestructor - Returns the destructor decl for this class.
  const CXXDestructorDecl *getDestructor(ASTContext &Context);

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
  bool isDerivedFrom(CXXRecordDecl *Base);
  
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
  bool isDerivedFrom(CXXRecordDecl *Base, CXXBasePaths &Paths);
  
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
  typedef bool BaseMatchesCallback(CXXBaseSpecifier *Specifier,
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
                     CXXBasePaths &Paths);
  
  /// \brief Base-class lookup callback that determines whether the given
  /// base class specifier refers to a specific class declaration.
  ///
  /// This callback can be used with \c lookupInBases() to determine whether
  /// a given derived class has is a base class subobject of a particular type.
  /// The user data pointer should refer to the canonical CXXRecordDecl of the
  /// base class that we are searching for.
  static bool FindBaseClass(CXXBaseSpecifier *Specifier, CXXBasePath &Path,
                            void *BaseRecord);
  
  /// \brief Base-class lookup callback that determines whether there exists
  /// a tag with the given name.
  ///
  /// This callback can be used with \c lookupInBases() to find tag members
  /// of the given name within a C++ class hierarchy. The user data pointer
  /// is an opaque \c DeclarationName pointer.
  static bool FindTagMember(CXXBaseSpecifier *Specifier, CXXBasePath &Path,
                            void *Name);

  /// \brief Base-class lookup callback that determines whether there exists
  /// a member with the given name.
  ///
  /// This callback can be used with \c lookupInBases() to find members
  /// of the given name within a C++ class hierarchy. The user data pointer
  /// is an opaque \c DeclarationName pointer.
  static bool FindOrdinaryMember(CXXBaseSpecifier *Specifier, CXXBasePath &Path,
                                 void *Name);
  
  /// \brief Base-class lookup callback that determines whether there exists
  /// a member with the given name that can be used in a nested-name-specifier.
  ///
  /// This callback can be used with \c lookupInBases() to find membes of
  /// the given name within a C++ class hierarchy that can occur within
  /// nested-name-specifiers.
  static bool FindNestedNameSpecifierMember(CXXBaseSpecifier *Specifier, 
                                            CXXBasePath &Path,
                                            void *UserData);
  
  /// viewInheritance - Renders and displays an inheritance diagram
  /// for this C++ class and all of its base classes (transitively) using
  /// GraphViz.
  void viewInheritance(ASTContext& Context) const;

  static bool classof(const Decl *D) {
    return D->getKind() == CXXRecord ||
           D->getKind() == ClassTemplateSpecialization ||
           D->getKind() == ClassTemplatePartialSpecialization;
  }
  static bool classof(const CXXRecordDecl *D) { return true; }
  static bool classof(const ClassTemplateSpecializationDecl *D) {
    return true;
  }
};

/// CXXMethodDecl - Represents a static or instance method of a
/// struct/union/class.
class CXXMethodDecl : public FunctionDecl {
protected:
  CXXMethodDecl(Kind DK, CXXRecordDecl *RD, SourceLocation L,
                DeclarationName N, QualType T, DeclaratorInfo *DInfo,
                bool isStatic, bool isInline)
    : FunctionDecl(DK, RD, L, N, T, DInfo, (isStatic ? Static : None),
                   isInline) {}

public:
  static CXXMethodDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                              SourceLocation L, DeclarationName N,
                              QualType T, DeclaratorInfo *DInfo,
                              bool isStatic = false,
                              bool isInline = false);

  bool isStatic() const { return getStorageClass() == Static; }
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
  
  const CXXMethodDecl *getCanonicalDecl() const {
    return cast<CXXMethodDecl>(FunctionDecl::getCanonicalDecl());
  }
  CXXMethodDecl *getCanonicalDecl() {
    return cast<CXXMethodDecl>(FunctionDecl::getCanonicalDecl());
  }
  
  ///
  void addOverriddenMethod(const CXXMethodDecl *MD);

  typedef const CXXMethodDecl ** method_iterator;

  method_iterator begin_overridden_methods() const;
  method_iterator end_overridden_methods() const;

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

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= CXXMethod && D->getKind() <= CXXConversion;
  }
  static bool classof(const CXXMethodDecl *D) { return true; }
};

/// CXXBaseOrMemberInitializer - Represents a C++ base or member
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
class CXXBaseOrMemberInitializer {
  /// BaseOrMember - This points to the entity being initialized,
  /// which is either a base class (a Type) or a non-static data
  /// member. When the low bit is 1, it's a base
  /// class; when the low bit is 0, it's a member.
  uintptr_t BaseOrMember;

  /// Args - The arguments used to initialize the base or member.
  Stmt **Args;
  unsigned NumArgs;

  /// \brief Stores either the constructor to call to initialize this base or
  /// member (a CXXConstructorDecl pointer), or stores the anonymous union of
  /// which the initialized value is a member.
  ///
  /// When the value is a FieldDecl pointer, 'BaseOrMember' is class's
  /// anonymous union data member, this field holds the FieldDecl for the
  /// member of the anonymous union being initialized.
  /// @code
  /// struct X {
  ///   X() : au_i1(123) {}
  ///   union {
  ///     int au_i1;
  ///     float au_f1;
  ///   };
  /// };
  /// @endcode
  /// In above example, BaseOrMember holds the field decl. for anonymous union
  /// and AnonUnionMember holds field decl for au_i1.
  llvm::PointerUnion<CXXConstructorDecl *, FieldDecl *> CtorOrAnonUnion;

  /// IdLoc - Location of the id in ctor-initializer list.
  SourceLocation IdLoc;

  /// RParenLoc - Location of the right paren of the ctor-initializer.
  SourceLocation RParenLoc;

public:
  /// CXXBaseOrMemberInitializer - Creates a new base-class initializer.
  explicit
  CXXBaseOrMemberInitializer(QualType BaseType, Expr **Args, unsigned NumArgs,
                             CXXConstructorDecl *C,
                             SourceLocation L, SourceLocation R);

  /// CXXBaseOrMemberInitializer - Creates a new member initializer.
  explicit
  CXXBaseOrMemberInitializer(FieldDecl *Member, Expr **Args, unsigned NumArgs,
                             CXXConstructorDecl *C,
                             SourceLocation L, SourceLocation R);

  /// ~CXXBaseOrMemberInitializer - Destroy the base or member initializer.
  ~CXXBaseOrMemberInitializer();

  /// arg_iterator - Iterates through the member initialization
  /// arguments.
  typedef ExprIterator arg_iterator;

  /// arg_const_iterator - Iterates through the member initialization
  /// arguments.
  typedef ConstExprIterator const_arg_iterator;

  /// getBaseOrMember - get the generic 'member' representing either the field
  /// or a base class.
  void* getBaseOrMember() const { return reinterpret_cast<void*>(BaseOrMember); }

  /// isBaseInitializer - Returns true when this initializer is
  /// initializing a base class.
  bool isBaseInitializer() const { return (BaseOrMember & 0x1) != 0; }

  /// isMemberInitializer - Returns true when this initializer is
  /// initializing a non-static data member.
  bool isMemberInitializer() const { return (BaseOrMember & 0x1) == 0; }

  /// getBaseClass - If this is a base class initializer, returns the
  /// type used to specify the initializer. The resulting type will be
  /// a class type or a typedef of a class type. If this is not a base
  /// class initializer, returns NULL.
  Type *getBaseClass() {
    if (isBaseInitializer())
      return reinterpret_cast<Type*>(BaseOrMember & ~0x01);
    else
      return 0;
  }

  /// getBaseClass - If this is a base class initializer, returns the
  /// type used to specify the initializer. The resulting type will be
  /// a class type or a typedef of a class type. If this is not a base
  /// class initializer, returns NULL.
  const Type *getBaseClass() const {
    if (isBaseInitializer())
      return reinterpret_cast<const Type*>(BaseOrMember & ~0x01);
    else
      return 0;
  }

  /// getMember - If this is a member initializer, returns the
  /// declaration of the non-static data member being
  /// initialized. Otherwise, returns NULL.
  FieldDecl *getMember() {
    if (isMemberInitializer())
      return reinterpret_cast<FieldDecl *>(BaseOrMember);
    else
      return 0;
  }

  void setMember(FieldDecl * anonUnionField) {
    BaseOrMember = reinterpret_cast<uintptr_t>(anonUnionField);
  }

  FieldDecl *getAnonUnionMember() const {
    return CtorOrAnonUnion.dyn_cast<FieldDecl *>();
  }
  void setAnonUnionMember(FieldDecl *anonMember) {
    CtorOrAnonUnion = anonMember;
  }

  const CXXConstructorDecl *getConstructor() const {
    return CtorOrAnonUnion.dyn_cast<CXXConstructorDecl *>();
  }

  SourceLocation getSourceLocation() const { return IdLoc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }

  /// arg_begin() - Retrieve an iterator to the first initializer argument.
  arg_iterator       arg_begin()       { return Args; }
  /// arg_begin() - Retrieve an iterator to the first initializer argument.
  const_arg_iterator const_arg_begin() const { return Args; }

  /// arg_end() - Retrieve an iterator past the last initializer argument.
  arg_iterator       arg_end()       { return Args + NumArgs; }
  /// arg_end() - Retrieve an iterator past the last initializer argument.
  const_arg_iterator const_arg_end() const { return Args + NumArgs; }

  /// getNumArgs - Determine the number of arguments used to
  /// initialize the member or base.
  unsigned getNumArgs() const { return NumArgs; }
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
  /// Explicit - Whether this constructor is explicit.
  bool Explicit : 1;

  /// ImplicitlyDefined - Whether this constructor was implicitly
  /// defined by the compiler. When false, the constructor was defined
  /// by the user. In C++03, this flag will have the same value as
  /// Implicit. In C++0x, however, a constructor that is
  /// explicitly defaulted (i.e., defined with " = default") will have
  /// @c !Implicit && ImplicitlyDefined.
  bool ImplicitlyDefined : 1;

  /// Support for base and member initializers.
  /// BaseOrMemberInitializers - The arguments used to initialize the base
  /// or member.
  CXXBaseOrMemberInitializer **BaseOrMemberInitializers;
  unsigned NumBaseOrMemberInitializers;

  CXXConstructorDecl(CXXRecordDecl *RD, SourceLocation L,
                     DeclarationName N, QualType T, DeclaratorInfo *DInfo,
                     bool isExplicit, bool isInline, bool isImplicitlyDeclared)
    : CXXMethodDecl(CXXConstructor, RD, L, N, T, DInfo, false, isInline),
      Explicit(isExplicit), ImplicitlyDefined(false),
      BaseOrMemberInitializers(0), NumBaseOrMemberInitializers(0) {
    setImplicit(isImplicitlyDeclared);
  }
  virtual void Destroy(ASTContext& C);

public:
  static CXXConstructorDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                                    SourceLocation L, DeclarationName N,
                                    QualType T, DeclaratorInfo *DInfo,
                                    bool isExplicit,
                                    bool isInline, bool isImplicitlyDeclared);

  /// isExplicit - Whether this constructor was marked "explicit" or not.
  bool isExplicit() const { return Explicit; }

  /// isImplicitlyDefined - Whether this constructor was implicitly
  /// defined. If false, then this constructor was defined by the
  /// user. This operation can only be invoked if the constructor has
  /// already been defined.
  bool isImplicitlyDefined(ASTContext &C) const {
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
  typedef CXXBaseOrMemberInitializer **init_iterator;

  /// init_const_iterator - Iterates through the memberbase initializer list.
  typedef CXXBaseOrMemberInitializer * const * init_const_iterator;

  /// init_begin() - Retrieve an iterator to the first initializer.
  init_iterator       init_begin()       { return BaseOrMemberInitializers; }
  /// begin() - Retrieve an iterator to the first initializer.
  init_const_iterator init_begin() const { return BaseOrMemberInitializers; }

  /// init_end() - Retrieve an iterator past the last initializer.
  init_iterator       init_end()       {
    return BaseOrMemberInitializers + NumBaseOrMemberInitializers;
  }
  /// end() - Retrieve an iterator past the last initializer.
  init_const_iterator init_end() const {
    return BaseOrMemberInitializers + NumBaseOrMemberInitializers;
  }

  /// getNumArgs - Determine the number of arguments used to
  /// initialize the member or base.
  unsigned getNumBaseOrMemberInitializers() const {
      return NumBaseOrMemberInitializers;
  }

  void setNumBaseOrMemberInitializers(unsigned numBaseOrMemberInitializers) {
    NumBaseOrMemberInitializers = numBaseOrMemberInitializers;
  }

  void setBaseOrMemberInitializers(CXXBaseOrMemberInitializer ** initializers) {
    BaseOrMemberInitializers = initializers;
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
  bool isCopyConstructor(ASTContext &Context, unsigned &TypeQuals) const;

  /// isCopyConstructor - Whether this constructor is a copy
  /// constructor (C++ [class.copy]p2, which can be used to copy the
  /// class.
  bool isCopyConstructor(ASTContext &Context) const {
    unsigned TypeQuals = 0;
    return isCopyConstructor(Context, TypeQuals);
  }

  /// isConvertingConstructor - Whether this constructor is a
  /// converting constructor (C++ [class.conv.ctor]), which can be
  /// used for user-defined conversions.
  bool isConvertingConstructor(bool AllowExplicit) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() == CXXConstructor;
  }
  static bool classof(const CXXConstructorDecl *D) { return true; }
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
public:
  enum KindOfObjectToDestroy {
    VBASE = 0x1,
    DRCTNONVBASE = 0x2,
    ANYBASE = 0x3
  };
private:
  /// ImplicitlyDefined - Whether this destructor was implicitly
  /// defined by the compiler. When false, the destructor was defined
  /// by the user. In C++03, this flag will have the same value as
  /// Implicit. In C++0x, however, a destructor that is
  /// explicitly defaulted (i.e., defined with " = default") will have
  /// @c !Implicit && ImplicitlyDefined.
  bool ImplicitlyDefined : 1;

  /// Support for base and member destruction.
  /// BaseOrMemberDestructions - The arguments used to destruct the base
  /// or member. Each uintptr_t value represents one of base classes (either
  /// virtual or direct non-virtual base), or non-static data member
  /// to be destroyed. The low two bits encode the kind of object
  /// being destroyed.
  uintptr_t *BaseOrMemberDestructions;
  unsigned NumBaseOrMemberDestructions;

  CXXDestructorDecl(CXXRecordDecl *RD, SourceLocation L,
                    DeclarationName N, QualType T,
                    bool isInline, bool isImplicitlyDeclared)
    : CXXMethodDecl(CXXDestructor, RD, L, N, T, /*DInfo=*/0, false, isInline),
      ImplicitlyDefined(false),
      BaseOrMemberDestructions(0), NumBaseOrMemberDestructions(0) {
    setImplicit(isImplicitlyDeclared);
  }
  virtual void Destroy(ASTContext& C);

public:
  static CXXDestructorDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                                   SourceLocation L, DeclarationName N,
                                   QualType T, bool isInline,
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

  /// destr_iterator - Iterates through the member/base destruction list.

  /// destr_const_iterator - Iterates through the member/base destruction list.
  typedef uintptr_t const destr_const_iterator;

  /// destr_begin() - Retrieve an iterator to the first destructed member/base.
  uintptr_t* destr_begin() {
    return BaseOrMemberDestructions;
  }
  /// destr_begin() - Retrieve an iterator to the first destructed member/base.
  uintptr_t* destr_begin() const {
    return BaseOrMemberDestructions;
  }

  /// destr_end() - Retrieve an iterator past the last destructed member/base.
  uintptr_t* destr_end() {
    return BaseOrMemberDestructions + NumBaseOrMemberDestructions;
  }
  /// destr_end() - Retrieve an iterator past the last destructed member/base.
  uintptr_t* destr_end() const {
    return BaseOrMemberDestructions + NumBaseOrMemberDestructions;
  }

  /// getNumBaseOrMemberDestructions - Number of base and non-static members
  /// to destroy.
  unsigned getNumBaseOrMemberDestructions() const {
    return NumBaseOrMemberDestructions;
  }

  /// setNumBaseOrMemberDestructions - Set number of base and non-static members
  /// to destroy.
  void setNumBaseOrMemberDestructions(unsigned numBaseOrMemberDestructions) {
    NumBaseOrMemberDestructions = numBaseOrMemberDestructions;
  }

  /// getBaseOrMemberToDestroy - get the generic 'member' representing either
  /// the field or a base class.
  uintptr_t* getBaseOrMemberToDestroy() const {
    return BaseOrMemberDestructions;
  }

  /// setBaseOrMemberToDestroy - set the generic 'member' representing either
  /// the field or a base class.
  void setBaseOrMemberDestructions(uintptr_t* baseOrMemberDestructions) {
    BaseOrMemberDestructions = baseOrMemberDestructions;
  }

  /// isVbaseToDestroy - returns true, if object is virtual base.
  bool isVbaseToDestroy(uintptr_t Vbase) const {
    return (Vbase & VBASE) != 0;
  }
  /// isDirectNonVBaseToDestroy - returns true, if object is direct non-virtual
  /// base.
  bool isDirectNonVBaseToDestroy(uintptr_t DrctNonVbase) const {
    return (DrctNonVbase & DRCTNONVBASE) != 0;
  }
  /// isAnyBaseToDestroy - returns true, if object is any base (virtual or
  /// direct non-virtual)
  bool isAnyBaseToDestroy(uintptr_t AnyBase) const {
    return (AnyBase & ANYBASE) != 0;
  }
  /// isMemberToDestroy - returns true if object is a non-static data member.
  bool isMemberToDestroy(uintptr_t Member) const {
    return (Member & ANYBASE)  == 0;
  }
  /// getAnyBaseClassToDestroy - Get the type for the given base class object.
  Type *getAnyBaseClassToDestroy(uintptr_t Base) const {
    if (isAnyBaseToDestroy(Base))
      return reinterpret_cast<Type*>(Base  & ~0x03);
    return 0;
  }
  /// getMemberToDestroy - Get the member for the given object.
  FieldDecl *getMemberToDestroy(uintptr_t Member) const {
    if (isMemberToDestroy(Member))
      return reinterpret_cast<FieldDecl *>(Member);
    return 0;
  }
  /// getVbaseClassToDestroy - Get the virtual base.
  Type *getVbaseClassToDestroy(uintptr_t Vbase) const {
    if (isVbaseToDestroy(Vbase))
      return reinterpret_cast<Type*>(Vbase  & ~0x01);
    return 0;
  }
  /// getDirectNonVBaseClassToDestroy - Get the virtual base.
  Type *getDirectNonVBaseClassToDestroy(uintptr_t Base) const {
    if (isDirectNonVBaseToDestroy(Base))
      return reinterpret_cast<Type*>(Base  & ~0x02);
    return 0;
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() == CXXDestructor;
  }
  static bool classof(const CXXDestructorDecl *D) { return true; }
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
  /// Explicit - Whether this conversion function is marked
  /// "explicit", meaning that it can only be applied when the user
  /// explicitly wrote a cast. This is a C++0x feature.
  bool Explicit : 1;

  CXXConversionDecl(CXXRecordDecl *RD, SourceLocation L,
                    DeclarationName N, QualType T, DeclaratorInfo *DInfo,
                    bool isInline, bool isExplicit)
    : CXXMethodDecl(CXXConversion, RD, L, N, T, DInfo, false, isInline),
      Explicit(isExplicit) { }

public:
  static CXXConversionDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                                   SourceLocation L, DeclarationName N,
                                   QualType T, DeclaratorInfo *DInfo,
                                   bool isInline, bool isExplicit);

  /// isExplicit - Whether this is an explicit conversion operator
  /// (C++0x only). Explicit conversion operators are only considered
  /// when the user has explicitly written a cast.
  bool isExplicit() const { return Explicit; }

  /// getConversionType - Returns the type that this conversion
  /// function is converting to.
  QualType getConversionType() const {
    return getType()->getAs<FunctionType>()->getResultType();
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() == CXXConversion;
  }
  static bool classof(const CXXConversionDecl *D) { return true; }
};

/// FriendDecl - Represents the declaration of a friend entity,
/// which can be a function, a type, or a templated function or type.
//  For example:
///
/// @code
/// template <typename T> class A {
///   friend int foo(T);
///   friend class B;
///   friend T; // only in C++0x
///   template <typename U> friend class C;
///   template <typename U> friend A& operator+=(A&, const U&) { ... }
/// };
/// @endcode
///
/// The semantic context of a friend decl is its declaring class.
class FriendDecl : public Decl {
public:
  typedef llvm::PointerUnion<NamedDecl*,Type*> FriendUnion;

private:
  // The declaration that's a friend of this class.
  FriendUnion Friend;

  // Location of the 'friend' specifier.
  SourceLocation FriendLoc;

  FriendDecl(DeclContext *DC, SourceLocation L, FriendUnion Friend,
             SourceLocation FriendL)
    : Decl(Decl::Friend, DC, L),
      Friend(Friend),
      FriendLoc(FriendL) {
  }

public:
  static FriendDecl *Create(ASTContext &C, DeclContext *DC,
                            SourceLocation L, FriendUnion Friend_,
                            SourceLocation FriendL);

  /// If this friend declaration names an (untemplated but
  /// possibly dependent) type, return the type;  otherwise
  /// return null.  This is used only for C++0x's unelaborated
  /// friend type declarations.
  Type *getFriendType() const {
    return Friend.dyn_cast<Type*>();
  }

  /// If this friend declaration doesn't name an unelaborated
  /// type, return the inner declaration.
  NamedDecl *getFriendDecl() const {
    return Friend.dyn_cast<NamedDecl*>();
  }

  /// Retrieves the location of the 'friend' keyword.
  SourceLocation getFriendLoc() const {
    return FriendLoc;
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() == Decl::Friend;
  }
  static bool classof(const FriendDecl *D) { return true; }
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
  enum LanguageIDs { lang_c = /* DW_LANG_C */ 0x0002,
  lang_cxx = /* DW_LANG_C_plus_plus */ 0x0004 };
private:
  /// Language - The language for this linkage specification.
  LanguageIDs Language;

  /// HadBraces - Whether this linkage specification had curly braces or not.
  bool HadBraces : 1;

  LinkageSpecDecl(DeclContext *DC, SourceLocation L, LanguageIDs lang,
                  bool Braces)
    : Decl(LinkageSpec, DC, L),
      DeclContext(LinkageSpec), Language(lang), HadBraces(Braces) { }

public:
  static LinkageSpecDecl *Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation L, LanguageIDs Lang,
                                 bool Braces);

  LanguageIDs getLanguage() const { return Language; }

  /// hasBraces - Determines whether this linkage specification had
  /// braces in its syntactic form.
  bool hasBraces() const { return HadBraces; }

  static bool classof(const Decl *D) {
    return D->getKind() == LinkageSpec;
  }
  static bool classof(const LinkageSpecDecl *D) { return true; }
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

  /// SourceLocation - Location of 'namespace' token.
  SourceLocation NamespaceLoc;

  /// \brief The source range that covers the nested-name-specifier
  /// preceding the namespace name.
  SourceRange QualifierRange;

  /// \brief The nested-name-specifier that precedes the namespace
  /// name, if any.
  NestedNameSpecifier *Qualifier;

  /// IdentLoc - Location of nominated namespace-name identifier.
  // FIXME: We don't store location of scope specifier.
  SourceLocation IdentLoc;

  /// NominatedNamespace - Namespace nominated by using-directive.
  NamespaceDecl *NominatedNamespace;

  /// Enclosing context containing both using-directive and nomintated
  /// namespace.
  DeclContext *CommonAncestor;

  /// getUsingDirectiveName - Returns special DeclarationName used by
  /// using-directives. This is only used by DeclContext for storing
  /// UsingDirectiveDecls in its lookup structure.
  static DeclarationName getName() {
    return DeclarationName::getUsingDirectiveName();
  }

  UsingDirectiveDecl(DeclContext *DC, SourceLocation L,
                     SourceLocation NamespcLoc,
                     SourceRange QualifierRange,
                     NestedNameSpecifier *Qualifier,
                     SourceLocation IdentLoc,
                     NamespaceDecl *Nominated,
                     DeclContext *CommonAncestor)
    : NamedDecl(Decl::UsingDirective, DC, L, getName()),
      NamespaceLoc(NamespcLoc), QualifierRange(QualifierRange),
      Qualifier(Qualifier), IdentLoc(IdentLoc),
      NominatedNamespace(Nominated? Nominated->getOriginalNamespace() : 0),
      CommonAncestor(CommonAncestor) {
  }

public:
  /// \brief Retrieve the source range of the nested-name-specifier
  /// that qualifiers the namespace name.
  SourceRange getQualifierRange() const { return QualifierRange; }

  /// \brief Retrieve the nested-name-specifier that qualifies the
  /// name of the namespace.
  NestedNameSpecifier *getQualifier() const { return Qualifier; }

  /// getNominatedNamespace - Returns namespace nominated by using-directive.
  NamespaceDecl *getNominatedNamespace() { return NominatedNamespace; }

  const NamespaceDecl *getNominatedNamespace() const {
    return const_cast<UsingDirectiveDecl*>(this)->getNominatedNamespace();
  }

  /// getCommonAncestor - returns common ancestor context of using-directive,
  /// and nominated by it namespace.
  DeclContext *getCommonAncestor() { return CommonAncestor; }
  const DeclContext *getCommonAncestor() const { return CommonAncestor; }

  /// getNamespaceKeyLocation - Returns location of namespace keyword.
  SourceLocation getNamespaceKeyLocation() const { return NamespaceLoc; }

  /// getIdentLocation - Returns location of identifier.
  SourceLocation getIdentLocation() const { return IdentLoc; }

  static UsingDirectiveDecl *Create(ASTContext &C, DeclContext *DC,
                                    SourceLocation L,
                                    SourceLocation NamespaceLoc,
                                    SourceRange QualifierRange,
                                    NestedNameSpecifier *Qualifier,
                                    SourceLocation IdentLoc,
                                    NamespaceDecl *Nominated,
                                    DeclContext *CommonAncestor);

  static bool classof(const Decl *D) {
    return D->getKind() == Decl::UsingDirective;
  }
  static bool classof(const UsingDirectiveDecl *D) { return true; }

  // Friend for getUsingDirectiveName.
  friend class DeclContext;
};

/// NamespaceAliasDecl - Represents a C++ namespace alias. For example:
///
/// @code
/// namespace Foo = Bar;
/// @endcode
class NamespaceAliasDecl : public NamedDecl {
  SourceLocation AliasLoc;

  /// \brief The source range that covers the nested-name-specifier
  /// preceding the namespace name.
  SourceRange QualifierRange;

  /// \brief The nested-name-specifier that precedes the namespace
  /// name, if any.
  NestedNameSpecifier *Qualifier;

  /// IdentLoc - Location of namespace identifier.
  SourceLocation IdentLoc;

  /// Namespace - The Decl that this alias points to. Can either be a
  /// NamespaceDecl or a NamespaceAliasDecl.
  NamedDecl *Namespace;

  NamespaceAliasDecl(DeclContext *DC, SourceLocation L,
                     SourceLocation AliasLoc, IdentifierInfo *Alias,
                     SourceRange QualifierRange,
                     NestedNameSpecifier *Qualifier,
                     SourceLocation IdentLoc, NamedDecl *Namespace)
    : NamedDecl(Decl::NamespaceAlias, DC, L, Alias), AliasLoc(AliasLoc),
      QualifierRange(QualifierRange), Qualifier(Qualifier),
      IdentLoc(IdentLoc), Namespace(Namespace) { }

public:
  /// \brief Retrieve the source range of the nested-name-specifier
  /// that qualifiers the namespace name.
  SourceRange getQualifierRange() const { return QualifierRange; }

  /// \brief Retrieve the nested-name-specifier that qualifies the
  /// name of the namespace.
  NestedNameSpecifier *getQualifier() const { return Qualifier; }

  NamespaceDecl *getNamespace() {
    if (NamespaceAliasDecl *AD = dyn_cast<NamespaceAliasDecl>(Namespace))
      return AD->getNamespace();

    return cast<NamespaceDecl>(Namespace);
  }

  const NamespaceDecl *getNamespace() const {
    return const_cast<NamespaceAliasDecl*>(this)->getNamespace();
  }

  /// \brief Retrieve the namespace that this alias refers to, which
  /// may either be a NamespaceDecl or a NamespaceAliasDecl.
  NamedDecl *getAliasedNamespace() const { return Namespace; }

  static NamespaceAliasDecl *Create(ASTContext &C, DeclContext *DC,
                                    SourceLocation L, SourceLocation AliasLoc,
                                    IdentifierInfo *Alias,
                                    SourceRange QualifierRange,
                                    NestedNameSpecifier *Qualifier,
                                    SourceLocation IdentLoc,
                                    NamedDecl *Namespace);

  static bool classof(const Decl *D) {
    return D->getKind() == Decl::NamespaceAlias;
  }
  static bool classof(const NamespaceAliasDecl *D) { return true; }
};

/// UsingDecl - Represents a C++ using-declaration. For example:
///    using someNameSpace::someIdentifier;
class UsingDecl : public NamedDecl {
  /// \brief The source range that covers the nested-name-specifier
  /// preceding the declaration name.
  SourceRange NestedNameRange;

  /// \brief The source location of the target declaration name.
  SourceLocation TargetNameLocation;

  /// \brief The source location of the "using" location itself.
  SourceLocation UsingLocation;

  /// \brief Target declaration.
  NamedDecl* TargetDecl;

  /// \brief Target nested name specifier.
  NestedNameSpecifier* TargetNestedNameDecl;

  // \brief Has 'typename' keyword.
  bool IsTypeName;

  UsingDecl(DeclContext *DC, SourceLocation L, SourceRange NNR,
            SourceLocation TargetNL, SourceLocation UL, NamedDecl* Target,
            NestedNameSpecifier* TargetNNS, bool IsTypeNameArg)
    : NamedDecl(Decl::Using, DC, L, Target->getDeclName()),
      NestedNameRange(NNR), TargetNameLocation(TargetNL),
      UsingLocation(UL), TargetDecl(Target),
      TargetNestedNameDecl(TargetNNS), IsTypeName(IsTypeNameArg) {
    this->IdentifierNamespace = TargetDecl->getIdentifierNamespace();
  }

public:
  /// \brief Returns the source range that covers the nested-name-specifier
  /// preceding the namespace name.
  SourceRange getNestedNameRange() { return NestedNameRange; }

  /// \brief Returns the source location of the target declaration name.
  SourceLocation getTargetNameLocation() { return TargetNameLocation; }

  /// \brief Returns the source location of the "using" location itself.
  SourceLocation getUsingLocation() { return UsingLocation; }

  /// \brief getTargetDecl - Returns target specified by using-decl.
  NamedDecl *getTargetDecl() { return TargetDecl; }
  const NamedDecl *getTargetDecl() const { return TargetDecl; }

  /// \brief Get target nested name declaration.
  NestedNameSpecifier* getTargetNestedNameDecl() {
    return TargetNestedNameDecl;
  }

  /// isTypeName - Return true if using decl has 'typename'.
  bool isTypeName() const { return IsTypeName; }

  static UsingDecl *Create(ASTContext &C, DeclContext *DC,
      SourceLocation L, SourceRange NNR, SourceLocation TargetNL,
      SourceLocation UL, NamedDecl* Target,
      NestedNameSpecifier* TargetNNS, bool IsTypeNameArg);

  static bool classof(const Decl *D) {
    return D->getKind() == Decl::Using;
  }
  static bool classof(const UsingDecl *D) { return true; }
};

/// UnresolvedUsingDecl - Represents a using declaration whose name can not
/// yet be resolved.
class UnresolvedUsingDecl : public NamedDecl {
  /// \brief The source range that covers the nested-name-specifier
  /// preceding the declaration name.
  SourceRange TargetNestedNameRange;

  /// \brief The source location of the target declaration name.
  SourceLocation TargetNameLocation;

  NestedNameSpecifier *TargetNestedNameSpecifier;

  DeclarationName TargetName;

  // \brief Has 'typename' keyword.
  bool IsTypeName;

  UnresolvedUsingDecl(DeclContext *DC, SourceLocation UsingLoc,
                      SourceRange TargetNNR, NestedNameSpecifier *TargetNNS,
                      SourceLocation TargetNameLoc, DeclarationName TargetName,
                      bool IsTypeNameArg)
  : NamedDecl(Decl::UnresolvedUsing, DC, UsingLoc, TargetName),
    TargetNestedNameRange(TargetNNR), TargetNameLocation(TargetNameLoc),
    TargetNestedNameSpecifier(TargetNNS), TargetName(TargetName),
    IsTypeName(IsTypeNameArg) { }

public:
  /// \brief Returns the source range that covers the nested-name-specifier
  /// preceding the namespace name.
  SourceRange getTargetNestedNameRange() const { return TargetNestedNameRange; }

  /// \brief Get target nested name declaration.
  NestedNameSpecifier* getTargetNestedNameSpecifier() {
    return TargetNestedNameSpecifier;
  }

  /// \brief Returns the source location of the target declaration name.
  SourceLocation getTargetNameLocation() const { return TargetNameLocation; }

  /// \brief Returns the source location of the target declaration name.
  DeclarationName getTargetName() const { return TargetName; }

  bool isTypeName() const { return IsTypeName; }

  static UnresolvedUsingDecl *Create(ASTContext &C, DeclContext *DC,
                                     SourceLocation UsingLoc,
                                     SourceRange TargetNNR,
                                     NestedNameSpecifier *TargetNNS,
                                     SourceLocation TargetNameLoc,
                                     DeclarationName TargetName,
                                     bool IsTypeNameArg);

  static bool classof(const Decl *D) {
    return D->getKind() == Decl::UnresolvedUsing;
  }
  static bool classof(const UnresolvedUsingDecl *D) { return true; }
};

/// StaticAssertDecl - Represents a C++0x static_assert declaration.
class StaticAssertDecl : public Decl {
  Expr *AssertExpr;
  StringLiteral *Message;

  StaticAssertDecl(DeclContext *DC, SourceLocation L,
                   Expr *assertexpr, StringLiteral *message)
  : Decl(StaticAssert, DC, L), AssertExpr(assertexpr), Message(message) { }

public:
  static StaticAssertDecl *Create(ASTContext &C, DeclContext *DC,
                                  SourceLocation L, Expr *AssertExpr,
                                  StringLiteral *Message);

  Expr *getAssertExpr() { return AssertExpr; }
  const Expr *getAssertExpr() const { return AssertExpr; }

  StringLiteral *getMessage() { return Message; }
  const StringLiteral *getMessage() const { return Message; }

  virtual ~StaticAssertDecl();
  virtual void Destroy(ASTContext& C);

  static bool classof(const Decl *D) {
    return D->getKind() == Decl::StaticAssert;
  }
  static bool classof(StaticAssertDecl *D) { return true; }
};

/// Insertion operator for diagnostics.  This allows sending AccessSpecifier's
/// into a diagnostic with <<.
const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                    AccessSpecifier AS);

} // end namespace clang

#endif
