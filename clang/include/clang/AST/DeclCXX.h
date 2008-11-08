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
class CXXRecordDecl;
class CXXConstructorDecl;
class CXXDestructorDecl;
class CXXConversionDecl;

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
  OverloadedFunctionDecl(DeclContext *DC, IdentifierInfo *Id)
    : NamedDecl(OverloadedFunction, SourceLocation(), Id) { }

  /// Functions - the set of overloaded functions contained in this
  /// overload set.
  llvm::SmallVector<FunctionDecl *, 4> Functions;
  
public:
  typedef llvm::SmallVector<FunctionDecl *, 4>::iterator function_iterator;
  typedef llvm::SmallVector<FunctionDecl *, 4>::const_iterator
    function_const_iterator;

  static OverloadedFunctionDecl *Create(ASTContext &C, DeclContext *DC,
                                        IdentifierInfo *Id);

  /// addOverload - Add an overloaded function FD to this set of
  /// overloaded functions.
  void addOverload(FunctionDecl *FD) {
    assert((!getNumFunctions() || (FD->getDeclContext() == getDeclContext())) &&
           "Overloaded functions must all be in the same context");
    assert((FD->getIdentifier() == getIdentifier() ||
            isa<CXXConversionDecl>(FD)) &&
           "Overloaded functions must have the same name or be conversions.");
    Functions.push_back(FD);
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

protected:
  /// EmitImpl - Serialize this FunctionDecl.  Called by Decl::Emit.
  virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize an OverloadedFunctionDecl.  Called by
  /// Decl::Create.
  static OverloadedFunctionDecl* CreateImpl(llvm::Deserializer& D, 
                                            ASTContext& C);
  
  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
  friend class CXXRecordDecl;
};

/// CXXFieldDecl - Represents an instance field of a C++ struct/union/class.
class CXXFieldDecl : public FieldDecl {
  CXXRecordDecl *Parent;

  CXXFieldDecl(CXXRecordDecl *RD, SourceLocation L, IdentifierInfo *Id,
               QualType T, Expr *BW = NULL)
    : FieldDecl(CXXField, L, Id, T, BW), Parent(RD) {}
public:
  static CXXFieldDecl *Create(ASTContext &C, CXXRecordDecl *RD,SourceLocation L,
                              IdentifierInfo *Id, QualType T, Expr *BW = NULL);

  void setAccess(AccessSpecifier AS) { Access = AS; }
  AccessSpecifier getAccess() const { return AccessSpecifier(Access); }
  CXXRecordDecl *getParent() const { return Parent; }
    
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == CXXField; }
  static bool classof(const CXXFieldDecl *D) { return true; }
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
/// CXXRecordDecl differs from RecordDecl in several ways. First, it
/// is a DeclContext, because it can contain other
/// declarations. Second, it provides additional C++ fields, including
/// storage for base classes and constructors.
class CXXRecordDecl : public RecordDecl, public DeclContext {
  /// UserDeclaredConstructor - True when this class has a
  /// user-declared constructor. 
  bool UserDeclaredConstructor : 1;

  /// UserDeclaredCopyConstructor - True when this class has a
  /// user-defined copy constructor.
  bool UserDeclaredCopyConstructor : 1;

  /// Aggregate - True when this class is an aggregate.
  bool Aggregate : 1;

  /// Polymorphic - True when this class is polymorphic, i.e. has at least one
  /// virtual member or derives from a polymorphic class.
  bool Polymorphic : 1;

  /// Bases - Base classes of this class.
  /// FIXME: This is wasted space for a union.
  CXXBaseSpecifier *Bases;

  /// NumBases - The number of base class specifiers in Bases.
  unsigned NumBases;

  /// Constructors - Overload set containing the constructors of this
  /// C++ class. Each of the entries in this overload set is a
  /// CXXConstructorDecl.
  OverloadedFunctionDecl Constructors;

  // Destructor - The destructor of this C++ class.
  CXXDestructorDecl *Destructor;

  /// Conversions - Overload set containing the conversion functions
  /// of this C++ class (but not its inherited conversion
  /// functions). Each of the entries in this overload set is a
  /// CXXConversionDecl.
  OverloadedFunctionDecl Conversions;

  CXXRecordDecl(TagKind TK, DeclContext *DC,
                SourceLocation L, IdentifierInfo *Id) 
    : RecordDecl(CXXRecord, TK, DC, L, Id), DeclContext(CXXRecord),
      UserDeclaredConstructor(false), UserDeclaredCopyConstructor(false),
      Aggregate(true), Polymorphic(false), Bases(0), NumBases(0),
      Constructors(DC, Id), Destructor(0), Conversions(DC, Id) { }

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

  const CXXFieldDecl *getMember(unsigned i) const {
    return cast<const CXXFieldDecl>(RecordDecl::getMember(i));
  }
  CXXFieldDecl *getMember(unsigned i) {
    return cast<CXXFieldDecl>(RecordDecl::getMember(i));
  }

  /// getMember - If the member doesn't exist, or there are no members, this 
  /// function will return 0;
  CXXFieldDecl *getMember(IdentifierInfo *name) {
    return cast_or_null<CXXFieldDecl>(RecordDecl::getMember(name));
  }

  /// getConstructors - Retrieve the overload set containing all of
  /// the constructors of this class.
  OverloadedFunctionDecl       *getConstructors()       { return &Constructors; }

  /// getConstructors - Retrieve the overload set containing all of
  /// the constructors of this class.
  const OverloadedFunctionDecl *getConstructors() const { return &Constructors; }

  /// hasConstCopyConstructor - Determines whether this class has a
  /// copy constructor that accepts a const-qualified argument.
  bool hasConstCopyConstructor(ASTContext &Context) const;

  /// addConstructor - Add another constructor to the list of constructors.
  void addConstructor(ASTContext &Context, CXXConstructorDecl *ConDecl);

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

  /// getDestructor - Retrieve the destructor for this class. 
  CXXDestructorDecl *getDestructor() const { return Destructor; }

  /// setDestructor - Set the destructor for this class.
  void setDestructor(CXXDestructorDecl *Destructor) {
    assert(!this->Destructor && "Already have a destructor!");
    this->Destructor = Destructor;
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

  /// isPolymorphic - Whether this class is polymorphic (C++ [class.virtual]),
  /// which means that the class contains or inherits a virtual function.
  bool isPolymorphic() const { return Polymorphic; }

  /// setPolymorphic - Set whether this class is polymorphic (C++
  /// [class.virtual]).
  void setPolymorphic(bool Poly) { Polymorphic = Poly; }

  /// viewInheritance - Renders and displays an inheritance diagram
  /// for this C++ class and all of its base classes (transitively) using
  /// GraphViz.
  void viewInheritance(ASTContext& Context) const;

  static bool classof(const Decl *D) { return D->getKind() == CXXRecord; }
  static bool classof(const CXXRecordDecl *D) { return true; }
  static DeclContext *castToDeclContext(const CXXRecordDecl *D) {
    return static_cast<DeclContext *>(const_cast<CXXRecordDecl*>(D));
  }
  static CXXRecordDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<CXXRecordDecl *>(const_cast<DeclContext*>(DC));
  }

  virtual void Destroy(ASTContext& C);

protected:
  /// EmitImpl - Serialize this CXXRecordDecl.  Called by Decl::Emit.
  // FIXME: Implement this.
  //virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize a CXXRecordDecl.  Called by Decl::Create.
  // FIXME: Implement this.
  static CXXRecordDecl* CreateImpl(Kind DK, llvm::Deserializer& D, ASTContext& C);
  
  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
};

/// CXXMethodDecl - Represents a static or instance method of a
/// struct/union/class.
class CXXMethodDecl : public FunctionDecl {
protected:
  CXXMethodDecl(Kind DK, CXXRecordDecl *RD, SourceLocation L,
                IdentifierInfo *Id, QualType T,
                bool isStatic, bool isInline, ScopedDecl *PrevDecl)
    : FunctionDecl(DK, RD, L, Id, T, (isStatic ? Static : None),
                   isInline, PrevDecl) {}

public:
  static CXXMethodDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                              SourceLocation L, IdentifierInfo *Id,
                              QualType T, bool isStatic = false,
                              bool isInline = false,  ScopedDecl *PrevDecl = 0);
  
  bool isStatic() const { return getStorageClass() == Static; }
  bool isInstance() const { return !isStatic(); }

  void setAccess(AccessSpecifier AS) { Access = AS; }
  AccessSpecifier getAccess() const { return AccessSpecifier(Access); }

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
    return getType()->getAsFunctionTypeProto()->getTypeQuals();
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { 
    return D->getKind() >= CXXMethod && D->getKind() <= CXXConversion;
  }
  static bool classof(const CXXMethodDecl *D) { return true; }

protected:
  /// EmitImpl - Serialize this CXXMethodDecl.  Called by Decl::Emit.
  // FIXME: Implement this.
  //virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize a CXXMethodDecl.  Called by Decl::Create.
  // FIXME: Implement this.
  static CXXMethodDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);
  
  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
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
  /// member (a CXXFieldDecl). When the low bit is 1, it's a base
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
  CXXBaseOrMemberInitializer(CXXFieldDecl *Member, Expr **Args, unsigned NumArgs);

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
  CXXFieldDecl *getMember() { 
    if (isMemberInitializer())
      return reinterpret_cast<CXXFieldDecl *>(BaseOrMember); 
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

  /// ImplicitlyDeclared - Whether this constructor was implicitly
  /// declared. When false, the constructor was declared by the user.
  bool ImplicitlyDeclared : 1;

  /// ImplicitlyDefined - Whether this constructor was implicitly
  /// defined by the compiler. When false, the constructor was defined
  /// by the user. In C++03, this flag will have the same value as
  /// ImplicitlyDeclared. In C++0x, however, a constructor that is
  /// explicitly defaulted (i.e., defined with " = default") will have
  /// @c !ImplicitlyDeclared && ImplicitlyDefined.
  bool ImplicitlyDefined : 1;

  /// FIXME: Add support for base and member initializers.

  CXXConstructorDecl(CXXRecordDecl *RD, SourceLocation L,
                     IdentifierInfo *Id, QualType T,
                     bool isExplicit, bool isInline, bool isImplicitlyDeclared)
    : CXXMethodDecl(CXXConstructor, RD, L, Id, T, false, isInline, /*PrevDecl=*/0),
      Explicit(isExplicit), ImplicitlyDeclared(isImplicitlyDeclared),
      ImplicitlyDefined(false) { }

public:
  static CXXConstructorDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                                    SourceLocation L, IdentifierInfo *Id,
                                    QualType T, bool isExplicit,
                                    bool isInline, bool isImplicitlyDeclared);

  /// isExplicit - Whether this constructor was marked "explicit" or not.  
  bool isExplicit() const { return Explicit; }

  /// isImplicitlyDeclared - Whether this constructor was implicitly
  /// declared. If false, then this constructor was explicitly
  /// declared by the user.
  bool isImplicitlyDeclared() const { return ImplicitlyDeclared; }

  /// isImplicitlyDefined - Whether this constructor was implicitly
  /// defined. If false, then this constructor was defined by the
  /// user. This operation can only be invoked if the constructor has
  /// already been defined.
  bool isImplicitlyDefined() const { 
    assert(getBody() != 0 && 
           "Can only get the implicit-definition flag once the constructor has been defined");
    return ImplicitlyDefined; 
  }

  /// setImplicitlyDefined - Set whether this constructor was
  /// implicitly defined or not.
  void setImplicitlyDefined(bool ID) { 
    assert(getBody() != 0 && 
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

  /// EmitImpl - Serialize this CXXConstructorDecl.  Called by Decl::Emit.
  // FIXME: Implement this.
  //virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize a CXXConstructorDecl.  Called by Decl::Create.
  // FIXME: Implement this.
  static CXXConstructorDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);
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
  /// ImplicitlyDeclared - Whether this destructor was implicitly
  /// declared. When false, the destructor was declared by the user.
  bool ImplicitlyDeclared : 1;

  /// ImplicitlyDefined - Whether this destructor was implicitly
  /// defined by the compiler. When false, the destructor was defined
  /// by the user. In C++03, this flag will have the same value as
  /// ImplicitlyDeclared. In C++0x, however, a destructor that is
  /// explicitly defaulted (i.e., defined with " = default") will have
  /// @c !ImplicitlyDeclared && ImplicitlyDefined.
  bool ImplicitlyDefined : 1;

  CXXDestructorDecl(CXXRecordDecl *RD, SourceLocation L,
                    IdentifierInfo *Id, QualType T,
                    bool isInline, bool isImplicitlyDeclared)
    : CXXMethodDecl(CXXDestructor, RD, L, Id, T, false, isInline, 
                    /*PrevDecl=*/0),
      ImplicitlyDeclared(isImplicitlyDeclared),
      ImplicitlyDefined(false) { }

public:
  static CXXDestructorDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                                   SourceLocation L, IdentifierInfo *Id,
                                   QualType T, bool isInline, 
                                   bool isImplicitlyDeclared);

  /// isImplicitlyDeclared - Whether this destructor was implicitly
  /// declared. If false, then this destructor was explicitly
  /// declared by the user.
  bool isImplicitlyDeclared() const { return ImplicitlyDeclared; }

  /// isImplicitlyDefined - Whether this destructor was implicitly
  /// defined. If false, then this destructor was defined by the
  /// user. This operation can only be invoked if the destructor has
  /// already been defined.
  bool isImplicitlyDefined() const { 
    assert(getBody() != 0 && 
           "Can only get the implicit-definition flag once the destructor has been defined");
    return ImplicitlyDefined; 
  }

  /// setImplicitlyDefined - Set whether this destructor was
  /// implicitly defined or not.
  void setImplicitlyDefined(bool ID) { 
    assert(getBody() != 0 && 
           "Can only set the implicit-definition flag once the destructor has been defined");
    ImplicitlyDefined = ID; 
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { 
    return D->getKind() == CXXDestructor;
  }
  static bool classof(const CXXDestructorDecl *D) { return true; }

  /// EmitImpl - Serialize this CXXDestructorDecl.  Called by Decl::Emit.
  // FIXME: Implement this.
  //virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize a CXXDestructorDecl.  Called by Decl::Create.
  // FIXME: Implement this.
  static CXXDestructorDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);
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
                    IdentifierInfo *Id, QualType T, 
                    bool isInline, bool isExplicit)
    : CXXMethodDecl(CXXConversion, RD, L, Id, T, false, isInline, 
                    /*PrevDecl=*/0),
      Explicit(isExplicit) { }

public:
  static CXXConversionDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                                   SourceLocation L, IdentifierInfo *Id,
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

  /// EmitImpl - Serialize this CXXConversionDecl.  Called by Decl::Emit.
  // FIXME: Implement this.
  //virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize a CXXConversionDecl.  Called by Decl::Create.
  // FIXME: Implement this.
  static CXXConversionDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// CXXClassVarDecl - Represents a static data member of a struct/union/class.
class CXXClassVarDecl : public VarDecl {

  CXXClassVarDecl(CXXRecordDecl *RD, SourceLocation L,
              IdentifierInfo *Id, QualType T, ScopedDecl *PrevDecl)
    : VarDecl(CXXClassVar, RD, L, Id, T, None, PrevDecl) {}
public:
  static CXXClassVarDecl *Create(ASTContext &C, CXXRecordDecl *RD,
                             SourceLocation L,IdentifierInfo *Id,
                             QualType T, ScopedDecl *PrevDecl);
  
  void setAccess(AccessSpecifier AS) { Access = AS; }
  AccessSpecifier getAccess() const { return AccessSpecifier(Access); }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == CXXClassVar; }
  static bool classof(const CXXClassVarDecl *D) { return true; }
  
protected:
  /// EmitImpl - Serialize this CXXClassVarDecl. Called by Decl::Emit.
  // FIXME: Implement this.
  //virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize a CXXClassVarDecl.  Called by Decl::Create.
  // FIXME: Implement this.
  static CXXClassVarDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);

  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
};


/// CXXClassMemberWrapper - A wrapper class for C++ class member decls.
/// Common functions like set/getAccess are included here to avoid bloating
/// the interface of non-C++ specific decl classes, like NamedDecl.
class CXXClassMemberWrapper {
  Decl *MD;

public:
  CXXClassMemberWrapper(Decl *D) : MD(D) {
    assert(isMember(D) && "Not a C++ class member!");
  }

  AccessSpecifier getAccess() const {
    return AccessSpecifier(MD->Access);
  }

  void setAccess(AccessSpecifier AS) {
    assert(AS != AS_none && "Access must be specified.");
    MD->Access = AS;
  }

  CXXRecordDecl *getParent() const {
    if (ScopedDecl *SD = dyn_cast<ScopedDecl>(MD)) {
      return cast<CXXRecordDecl>(SD->getDeclContext());
    }
    return cast<CXXFieldDecl>(MD)->getParent();
  }

  static bool isMember(Decl *D) {
    if (ScopedDecl *SD = dyn_cast<ScopedDecl>(D)) {
      return isa<CXXRecordDecl>(SD->getDeclContext());
    }
    return isa<CXXFieldDecl>(D);
  }
};
  
/// LinkageSpecDecl - This represents a linkage specification.  For example:
///   extern "C" void foo();
///
class LinkageSpecDecl : public Decl {
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
  /// D - This is the Decl of the linkage specification.
  Decl *D;
  
  LinkageSpecDecl(SourceLocation L, LanguageIDs lang, Decl *d)
  : Decl(LinkageSpec, L), Language(lang), D(d) {}
public:
  static LinkageSpecDecl *Create(ASTContext &C, SourceLocation L,
                                 LanguageIDs Lang, Decl *D);
  
  LanguageIDs getLanguage() const { return Language; }
  const Decl *getDecl() const { return D; }
  Decl *getDecl() { return D; }
  
  static bool classof(const Decl *D) {
    return D->getKind() == LinkageSpec;
  }
  static bool classof(const LinkageSpecDecl *D) { return true; }
  
protected:
  void EmitInRec(llvm::Serializer& S) const;
  void ReadInRec(llvm::Deserializer& D, ASTContext& C);
};

  

} // end namespace clang

#endif
