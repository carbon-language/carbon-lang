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

#include "clang/AST/Decl.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

class ClassTemplateDecl;
class CXXRecordDecl;
class CXXConstructorDecl;
class CXXDestructorDecl;
class CXXConversionDecl;
class CXXMethodDecl;
class ClassTemplateSpecializationDecl;

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
  llvm::SmallVector<FunctionDecl *, 4> Functions;

  // FIXME: This should go away when we stop using
  // OverloadedFunctionDecl to store conversions in CXXRecordDecl.
  friend class CXXRecordDecl;

public:
  typedef llvm::SmallVector<FunctionDecl *, 4>::iterator function_iterator;
  typedef llvm::SmallVector<FunctionDecl *, 4>::const_iterator
    function_const_iterator;

  static OverloadedFunctionDecl *Create(ASTContext &C, DeclContext *DC,
                                        DeclarationName N);

  /// addOverload - Add an overloaded function FD to this set of
  /// overloaded functions.
  void addOverload(FunctionDecl *FD) {
    assert((FD->getDeclName() == getDeclName() ||
            isa<CXXConversionDecl>(FD) || isa<CXXConstructorDecl>(FD)) &&
           "Overloaded functions must have the same name");
    Functions.push_back(FD);

    // An overloaded function declaration always has the location of
    // the most-recently-added function declaration.
    if (FD->getLocation().isValid())
      this->setLocation(FD->getLocation());
  }

  function_iterator function_begin() { return Functions.begin(); }
  function_iterator function_end() { return Functions.end(); }
  function_const_iterator function_begin() const { return Functions.begin(); }
  function_const_iterator function_end() const { return Functions.end(); }

  /// getNumFunctions - the number of overloaded functions stored in
  /// this set.
  unsigned getNumFunctions() const { return Functions.size(); }

  /// getFunction - retrieve the ith function in the overload set.
  const FunctionDecl *getFunction(unsigned i) const {
    assert(i < getNumFunctions() && "Illegal function #");
    return Functions[i];
  }
  FunctionDecl *getFunction(unsigned i) {
    assert(i < getNumFunctions() && "Illegal function #");
    return Functions[i];
  }

  // getDeclContext - Get the context of these overloaded functions.
  DeclContext *getDeclContext() {
    assert(getNumFunctions() > 0 && "Context of an empty overload set");
    return getFunction(0)->getDeclContext();
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { 
    return D->getKind() == OverloadedFunction; 
  }
  static bool classof(const OverloadedFunctionDecl *D) { return true; }
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

  /// Polymorphic - True when this class is polymorphic, i.e. has at least one
  /// virtual member or derives from a polymorphic class.
  bool Polymorphic : 1;

  /// Abstract - True when this class is abstract, i.e. has at least one
  /// pure virtual function, (that can come from a base class).
  bool Abstract : 1;
  
  /// HasTrivialConstructor - True when this class has a trivial constructor
  bool HasTrivialConstructor : 1;
  
  /// HasTrivialDestructor - True when this class has a trivial destructor
  bool HasTrivialDestructor : 1;
  
  /// Bases - Base classes of this class.
  /// FIXME: This is wasted space for a union.
  CXXBaseSpecifier *Bases;

  /// NumBases - The number of base class specifiers in Bases.
  unsigned NumBases;

  /// Conversions - Overload set containing the conversion functions
  /// of this C++ class (but not its inherited conversion
  /// functions). Each of the entries in this overload set is a
  /// CXXConversionDecl.
  OverloadedFunctionDecl Conversions;

  /// \brief The template or declaration that is declaration is
  /// instantiated from.
  /// 
  /// For non-templates, this value will be NULL. For record
  /// declarations that describe a class template, this will be a
  /// pointer to a ClassTemplateDecl. For member
  /// classes of class template specializations, this will be the
  /// RecordDecl from which the member class was instantiated.
  llvm::PointerUnion<ClassTemplateDecl*, CXXRecordDecl*>TemplateOrInstantiation;

protected:
  CXXRecordDecl(Kind K, TagKind TK, DeclContext *DC,
                SourceLocation L, IdentifierInfo *Id);

  ~CXXRecordDecl();

public:
  /// base_class_iterator - Iterator that traverses the base classes
  /// of a clas.
  typedef CXXBaseSpecifier*       base_class_iterator;

  /// base_class_const_iterator - Iterator that traverses the base
  /// classes of a clas.
  typedef const CXXBaseSpecifier* base_class_const_iterator;

  static CXXRecordDecl *Create(ASTContext &C, TagKind TK, DeclContext *DC,
                               SourceLocation L, IdentifierInfo *Id,
                               CXXRecordDecl* PrevDecl=0);
  
  /// setBases - Sets the base classes of this struct or class.
  void setBases(CXXBaseSpecifier const * const *Bases, unsigned NumBases);

  /// getNumBases - Retrieves the number of base classes of this
  /// class.
  unsigned getNumBases() const { return NumBases; }

  base_class_iterator       bases_begin()       { return Bases; }
  base_class_const_iterator bases_begin() const { return Bases; }
  base_class_iterator       bases_end()         { return Bases + NumBases; }
  base_class_const_iterator bases_end()   const { return Bases + NumBases; }

  /// hasConstCopyConstructor - Determines whether this class has a
  /// copy constructor that accepts a const-qualified argument.
  bool hasConstCopyConstructor(ASTContext &Context) const;

  /// hasConstCopyAssignment - Determines whether this class has a
  /// copy assignment operator that accepts a const-qualified argument.
  bool hasConstCopyAssignment(ASTContext &Context) const;

  /// addedConstructor - Notify the class that another constructor has
  /// been added. This routine helps maintain information about the
  /// class based on which constructors have been added.
  void addedConstructor(ASTContext &Context, CXXConstructorDecl *ConDecl);

  /// hasUserDeclaredConstructor - Whether this class has any
  /// user-declared constructors. When true, a default constructor
  /// will not be implicitly declared.
  bool hasUserDeclaredConstructor() const { return UserDeclaredConstructor; }

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
    return &Conversions; 
  }
  const OverloadedFunctionDecl *getConversionFunctions() const { 
    return &Conversions; 
  }

  /// addConversionFunction - Add a new conversion function to the
  /// list of conversion functions.
  void addConversionFunction(ASTContext &Context, CXXConversionDecl *ConvDecl);

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
  CXXRecordDecl *getInstantiatedFromMemberClass() {
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
  ClassTemplateDecl *getDescribedClassTemplate() {
    return TemplateOrInstantiation.dyn_cast<ClassTemplateDecl*>();
  }

  void setDescribedClassTemplate(ClassTemplateDecl *Template) {
    TemplateOrInstantiation = Template;
  }

  /// viewInheritance - Renders and displays an inheritance diagram
  /// for this C++ class and all of its base classes (transitively) using
  /// GraphViz.
  void viewInheritance(ASTContext& Context) const;

  static bool classof(const Decl *D) { 
    return D->getKind() == CXXRecord || 
           D->getKind() == ClassTemplateSpecialization; 
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
                DeclarationName N, QualType T,
                bool isStatic, bool isInline)
    : FunctionDecl(DK, RD, L, N, T, (isStatic ? Static : None),
                   isInline) {}

public:
  static CXXMethodDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                              SourceLocation L, DeclarationName N,
                              QualType T, bool isStatic = false,
                              bool isInline = false);
  
  bool isStatic() const { return getStorageClass() == Static; }
  bool isInstance() const { return !isStatic(); }

  bool isOutOfLineDefinition() const {
    return getLexicalDeclContext() != getDeclContext();
  }

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
    return getType()->getAsFunctionProtoType()->getTypeQuals();
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
class CXXBaseOrMemberInitializer {
  /// BaseOrMember - This points to the entity being initialized,
  /// which is either a base class (a Type) or a non-static data
  /// member. When the low bit is 1, it's a base
  /// class; when the low bit is 0, it's a member.
  uintptr_t BaseOrMember;

  /// Args - The arguments used to initialize the base or member.
  Expr **Args;
  unsigned NumArgs;

public:
  /// CXXBaseOrMemberInitializer - Creates a new base-class initializer.
  explicit 
  CXXBaseOrMemberInitializer(QualType BaseType, Expr **Args, unsigned NumArgs);

  /// CXXBaseOrMemberInitializer - Creates a new member initializer.
  explicit 
  CXXBaseOrMemberInitializer(FieldDecl *Member, Expr **Args, unsigned NumArgs);

  /// ~CXXBaseOrMemberInitializer - Destroy the base or member initializer.
  ~CXXBaseOrMemberInitializer();

  /// arg_iterator - Iterates through the member initialization
  /// arguments.
  typedef Expr **arg_iterator;

  /// arg_const_iterator - Iterates through the member initialization
  /// arguments.
  typedef Expr * const * arg_const_iterator;

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

  /// begin() - Retrieve an iterator to the first initializer argument.
  arg_iterator       begin()       { return Args; }
  /// begin() - Retrieve an iterator to the first initializer argument.
  arg_const_iterator begin() const { return Args; }

  /// end() - Retrieve an iterator past the last initializer argument.
  arg_iterator       end()       { return Args + NumArgs; }
  /// end() - Retrieve an iterator past the last initializer argument.
  arg_const_iterator end() const { return Args + NumArgs; }

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

  /// FIXME: Add support for base and member initializers.

  CXXConstructorDecl(CXXRecordDecl *RD, SourceLocation L,
                     DeclarationName N, QualType T,
                     bool isExplicit, bool isInline, bool isImplicitlyDeclared)
    : CXXMethodDecl(CXXConstructor, RD, L, N, T, false, isInline),
      Explicit(isExplicit), ImplicitlyDefined(false) { 
    setImplicit(isImplicitlyDeclared);
  }

public:
  static CXXConstructorDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                                    SourceLocation L, DeclarationName N,
                                    QualType T, bool isExplicit,
                                    bool isInline, bool isImplicitlyDeclared);

  /// isExplicit - Whether this constructor was marked "explicit" or not.  
  bool isExplicit() const { return Explicit; }

  /// isImplicitlyDefined - Whether this constructor was implicitly
  /// defined. If false, then this constructor was defined by the
  /// user. This operation can only be invoked if the constructor has
  /// already been defined.
  bool isImplicitlyDefined(ASTContext &C) const { 
    assert(isThisDeclarationADefinition() && 
           "Can only get the implicit-definition flag once the constructor has been defined");
    return ImplicitlyDefined; 
  }

  /// setImplicitlyDefined - Set whether this constructor was
  /// implicitly defined or not.
  void setImplicitlyDefined(bool ID) { 
    assert(isThisDeclarationADefinition() && 
           "Can only set the implicit-definition flag once the constructor has been defined");
    ImplicitlyDefined = ID; 
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
  bool isConvertingConstructor() const;

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
  /// ImplicitlyDefined - Whether this destructor was implicitly
  /// defined by the compiler. When false, the destructor was defined
  /// by the user. In C++03, this flag will have the same value as
  /// Implicit. In C++0x, however, a destructor that is
  /// explicitly defaulted (i.e., defined with " = default") will have
  /// @c !Implicit && ImplicitlyDefined.
  bool ImplicitlyDefined : 1;

  CXXDestructorDecl(CXXRecordDecl *RD, SourceLocation L,
                    DeclarationName N, QualType T,
                    bool isInline, bool isImplicitlyDeclared)
    : CXXMethodDecl(CXXDestructor, RD, L, N, T, false, isInline),
      ImplicitlyDefined(false) { 
    setImplicit(isImplicitlyDeclared);
  }

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
                    DeclarationName N, QualType T, 
                    bool isInline, bool isExplicit)
    : CXXMethodDecl(CXXConversion, RD, L, N, T, false, isInline),
      Explicit(isExplicit) { }

public:
  static CXXConversionDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                                   SourceLocation L, DeclarationName N,
                                   QualType T, bool isInline, 
                                   bool isExplicit);

  /// isExplicit - Whether this is an explicit conversion operator
  /// (C++0x only). Explicit conversion operators are only considered
  /// when the user has explicitly written a cast.
  bool isExplicit() const { return Explicit; }

  /// getConversionType - Returns the type that this conversion
  /// function is converting to.
  QualType getConversionType() const { 
    return getType()->getAsFunctionType()->getResultType(); 
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { 
    return D->getKind() == CXXConversion;
  }
  static bool classof(const CXXConversionDecl *D) { return true; }
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
                     SourceLocation IdentLoc,
                     NamespaceDecl *Nominated,
                     DeclContext *CommonAncestor)
    : NamedDecl(Decl::UsingDirective, DC, L, getName()),
      NamespaceLoc(NamespcLoc), IdentLoc(IdentLoc),
      NominatedNamespace(Nominated? Nominated->getOriginalNamespace() : 0),
      CommonAncestor(CommonAncestor) {
  }

public:
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
  
  /// IdentLoc - Location of namespace identifier.
  /// FIXME: We don't store location of scope specifier.
  SourceLocation IdentLoc;
  
  /// Namespace - The Decl that this alias points to. Can either be a 
  /// NamespaceDecl or a NamespaceAliasDecl.
  NamedDecl *Namespace;
  
  NamespaceAliasDecl(DeclContext *DC, SourceLocation L, 
                     SourceLocation AliasLoc, IdentifierInfo *Alias, 
                     SourceLocation IdentLoc, NamedDecl *Namespace)
    : NamedDecl(Decl::NamespaceAlias, DC, L, Alias), AliasLoc(AliasLoc), 
      IdentLoc(IdentLoc), Namespace(Namespace) { }

public:

  NamespaceDecl *getNamespace() {
    if (NamespaceAliasDecl *AD = dyn_cast<NamespaceAliasDecl>(Namespace))
      return AD->getNamespace();

    return cast<NamespaceDecl>(Namespace);
  }
  
  const NamespaceDecl *getNamespace() const {
    return const_cast<NamespaceAliasDecl*>(this)->getNamespace();
  }
  
  static NamespaceAliasDecl *Create(ASTContext &C, DeclContext *DC, 
                                    SourceLocation L, SourceLocation AliasLoc, 
                                    IdentifierInfo *Alias, 
                                    SourceLocation IdentLoc, 
                                    NamedDecl *Namespace);
  
  static bool classof(const Decl *D) {
    return D->getKind() == Decl::NamespaceAlias;
  }
  static bool classof(const NamespaceAliasDecl *D) { return true; }
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

/// CXXTempVarDecl - Represents an implicit C++ temporary variable declaration.
class CXXTempVarDecl : public VarDecl {
protected:
  CXXTempVarDecl(DeclContext *DC, QualType T) 
    : VarDecl(CXXTempVar, DC, SourceLocation(), 0, T, None) {}

public:
  static CXXTempVarDecl *Create(ASTContext &C, DeclContext *DC,
                                QualType T);
  
  static bool classof(const Decl *D) {
    return D->getKind() == Decl::CXXTempVar;
  }
  static bool classof(CXXTempVarDecl *D) { return true; }
};

/// Insertion operator for diagnostics.  This allows sending AccessSpecifier's
/// into a diagnostic with <<.
const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                    AccessSpecifier AS);
  
} // end namespace clang

#endif
