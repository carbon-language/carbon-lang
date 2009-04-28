//===--- Decl.h - Classes for representing declarations ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Decl subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECL_H
#define LLVM_CLANG_AST_DECL_H

#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/ExternalASTSource.h"

namespace clang {
class Expr;
class Stmt;
class CompoundStmt;
class StringLiteral;

/// TranslationUnitDecl - The top declaration context.
class TranslationUnitDecl : public Decl, public DeclContext {
  TranslationUnitDecl()
    : Decl(TranslationUnit, 0, SourceLocation()),
      DeclContext(TranslationUnit) {}
public:
  static TranslationUnitDecl *Create(ASTContext &C);
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == TranslationUnit; }
  static bool classof(const TranslationUnitDecl *D) { return true; }  
  static DeclContext *castToDeclContext(const TranslationUnitDecl *D) {
    return static_cast<DeclContext *>(const_cast<TranslationUnitDecl*>(D));
  }
  static TranslationUnitDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<TranslationUnitDecl *>(const_cast<DeclContext*>(DC));
  }
};

/// NamedDecl - This represents a decl with a name.  Many decls have names such
/// as ObjCMethodDecl, but not @class, etc.
class NamedDecl : public Decl {
  /// Name - The name of this declaration, which is typically a normal
  /// identifier but may also be a special kind of name (C++
  /// constructor, Objective-C selector, etc.)
  DeclarationName Name;

protected:
  NamedDecl(Kind DK, DeclContext *DC, SourceLocation L, DeclarationName N)
    : Decl(DK, DC, L), Name(N) { }

public:
  /// getIdentifier - Get the identifier that names this declaration,
  /// if there is one. This will return NULL if this declaration has
  /// no name (e.g., for an unnamed class) or if the name is a special
  /// name (C++ constructor, Objective-C selector, etc.).
  IdentifierInfo *getIdentifier() const { return Name.getAsIdentifierInfo(); }

  /// getNameAsCString - Get the name of identifier for this declaration as a
  /// C string (const char*).  This requires that the declaration have a name
  /// and that it be a simple identifier.
  const char *getNameAsCString() const {
    assert(getIdentifier() && "Name is not a simple identifier");
    return getIdentifier()->getName();
  }

  /// getDeclName - Get the actual, stored name of the declaration,
  /// which may be a special name.
  DeclarationName getDeclName() const { return Name; }

  /// \brief Set the name of this declaration.
  void setDeclName(DeclarationName N) { Name = N; }

  /// getNameAsString - Get a human-readable name for the declaration, even if
  /// it is one of the special kinds of names (C++ constructor, Objective-C
  /// selector, etc).  Creating this name requires expensive string
  /// manipulation, so it should be called only when performance doesn't matter.
  /// For simple declarations, getNameAsCString() should suffice.
  std::string getNameAsString() const { return Name.getAsString(); }
  
  /// getQualifiedNameAsString - Returns human-readable qualified name for
  /// declaration, like A::B::i, for i being member of namespace A::B.
  /// If declaration is not member of context which can be named (record,
  /// namespace), it will return same result as getNameAsString().
  /// Creating this name is expensive, so it should be called only when
  /// performance doesn't matter.
  std::string getQualifiedNameAsString() const;

  /// declarationReplaces - Determine whether this declaration, if
  /// known to be well-formed within its context, will replace the
  /// declaration OldD if introduced into scope. A declaration will
  /// replace another declaration if, for example, it is a
  /// redeclaration of the same variable or function, but not if it is
  /// a declaration of a different kind (function vs. class) or an
  /// overloaded function.
  bool declarationReplaces(NamedDecl *OldD) const;

  /// \brief Determine whether this declaration has linkage.
  bool hasLinkage() const;

  static bool classof(const Decl *D) {
    return D->getKind() >= NamedFirst && D->getKind() <= NamedLast;
  }
  static bool classof(const NamedDecl *D) { return true; }
};

/// NamespaceDecl - Represent a C++ namespace.
class NamespaceDecl : public NamedDecl, public DeclContext {
  SourceLocation LBracLoc, RBracLoc;
  
  // For extended namespace definitions:
  //
  // namespace A { int x; }
  // namespace A { int y; }
  //
  // there will be one NamespaceDecl for each declaration.
  // NextNamespace points to the next extended declaration.
  // OrigNamespace points to the original namespace declaration.
  // OrigNamespace of the first namespace decl points to itself.
  NamespaceDecl *OrigNamespace, *NextNamespace;
  
  NamespaceDecl(DeclContext *DC, SourceLocation L, IdentifierInfo *Id)
    : NamedDecl(Namespace, DC, L, Id), DeclContext(Namespace) {
    OrigNamespace = this;
    NextNamespace = 0;
  }
public:
  static NamespaceDecl *Create(ASTContext &C, DeclContext *DC,
                               SourceLocation L, IdentifierInfo *Id);
  
  virtual void Destroy(ASTContext& C);

  NamespaceDecl *getNextNamespace() { return NextNamespace; }
  const NamespaceDecl *getNextNamespace() const { return NextNamespace; }
  void setNextNamespace(NamespaceDecl *ND) { NextNamespace = ND; }

  NamespaceDecl *getOriginalNamespace() const {
    return OrigNamespace;
  }
  void setOriginalNamespace(NamespaceDecl *ND) { OrigNamespace = ND; }
  
  SourceRange getSourceRange() const { 
    return SourceRange(LBracLoc, RBracLoc); 
  }

  SourceLocation getLBracLoc() const { return LBracLoc; }
  SourceLocation getRBracLoc() const { return RBracLoc; }
  void setLBracLoc(SourceLocation LBrace) { LBracLoc = LBrace; }
  void setRBracLoc(SourceLocation RBrace) { RBracLoc = RBrace; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Namespace; }
  static bool classof(const NamespaceDecl *D) { return true; }
  static DeclContext *castToDeclContext(const NamespaceDecl *D) {
    return static_cast<DeclContext *>(const_cast<NamespaceDecl*>(D));
  }
  static NamespaceDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<NamespaceDecl *>(const_cast<DeclContext*>(DC));
  }
};

/// ValueDecl - Represent the declaration of a variable (in which case it is 
/// an lvalue) a function (in which case it is a function designator) or
/// an enum constant. 
class ValueDecl : public NamedDecl {
  QualType DeclType;

protected:
  ValueDecl(Kind DK, DeclContext *DC, SourceLocation L,
            DeclarationName N, QualType T) 
    : NamedDecl(DK, DC, L, N), DeclType(T) {}
public:
  QualType getType() const { return DeclType; }
  void setType(QualType newType) { DeclType = newType; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= ValueFirst && D->getKind() <= ValueLast;
  }
  static bool classof(const ValueDecl *D) { return true; }
};

/// VarDecl - An instance of this class is created to represent a variable
/// declaration or definition.
class VarDecl : public ValueDecl {
public:
  enum StorageClass {
    None, Auto, Register, Extern, Static, PrivateExtern
  };

  /// getStorageClassSpecifierString - Return the string used to
  /// specify the storage class \arg SC.
  ///
  /// It is illegal to call this function with SC == None.
  static const char *getStorageClassSpecifierString(StorageClass SC);

private:
  Stmt *Init;
  // FIXME: This can be packed into the bitfields in Decl.
  unsigned SClass : 3;
  bool ThreadSpecified : 1;
  bool HasCXXDirectInit : 1; 

  /// DeclaredInCondition - Whether this variable was declared in a
  /// condition, e.g., if (int x = foo()) { ... }.
  bool DeclaredInCondition : 1;

  /// \brief The previous declaration of this variable.
  VarDecl *PreviousDeclaration;

  // Move to DeclGroup when it is implemented.
  SourceLocation TypeSpecStartLoc;
  friend class StmtIteratorBase;
protected:
  VarDecl(Kind DK, DeclContext *DC, SourceLocation L, IdentifierInfo *Id,
          QualType T, StorageClass SC, SourceLocation TSSL = SourceLocation())
    : ValueDecl(DK, DC, L, Id, T), Init(0),
      ThreadSpecified(false), HasCXXDirectInit(false),
      DeclaredInCondition(false), PreviousDeclaration(0), 
      TypeSpecStartLoc(TSSL) { 
    SClass = SC; 
  }
public:
  static VarDecl *Create(ASTContext &C, DeclContext *DC,
                         SourceLocation L, IdentifierInfo *Id,
                         QualType T, StorageClass S,
                         SourceLocation TypeSpecStartLoc = SourceLocation());

  virtual ~VarDecl();
  virtual void Destroy(ASTContext& C);

  StorageClass getStorageClass() const { return (StorageClass)SClass; }
  void setStorageClass(StorageClass SC) { SClass = SC; }

  SourceLocation getTypeSpecStartLoc() const { return TypeSpecStartLoc; }
  void setTypeSpecStartLoc(SourceLocation SL) {
    TypeSpecStartLoc = SL;
  }

  const Expr *getInit() const { return (const Expr*) Init; }
  Expr *getInit() { return (Expr*) Init; }
  void setInit(Expr *I) { Init = (Stmt*) I; }
      
  /// \brief Retrieve the definition of this variable, which may come
  /// from a previous declaration. Def will be set to the VarDecl that
  /// contains the initializer, and the result will be that
  /// initializer.
  const Expr *getDefinition(const VarDecl *&Def) const;

  void setThreadSpecified(bool T) { ThreadSpecified = T; }
  bool isThreadSpecified() const {
    return ThreadSpecified;
  }

  void setCXXDirectInitializer(bool T) { HasCXXDirectInit = T; }

  /// hasCXXDirectInitializer - If true, the initializer was a direct
  /// initializer, e.g: "int x(1);". The Init expression will be the expression
  /// inside the parens or a "ClassType(a,b,c)" class constructor expression for
  /// class types. Clients can distinguish between "int x(1);" and "int x=1;"
  /// by checking hasCXXDirectInitializer.
  ///
  bool hasCXXDirectInitializer() const {
    return HasCXXDirectInit;
  }
  
  /// isDeclaredInCondition - Whether this variable was declared as
  /// part of a condition in an if/switch/while statement, e.g.,
  /// @code
  /// if (int x = foo()) { ... }
  /// @endcode
  bool isDeclaredInCondition() const {
    return DeclaredInCondition;
  }
  void setDeclaredInCondition(bool InCondition) { 
    DeclaredInCondition = InCondition; 
  }

  /// getPreviousDeclaration - Return the previous declaration of this
  /// variable.
  const VarDecl *getPreviousDeclaration() const { return PreviousDeclaration; }

  void setPreviousDeclaration(VarDecl *PrevDecl) {
    PreviousDeclaration = PrevDecl;
  }

  /// hasLocalStorage - Returns true if a variable with function scope
  ///  is a non-static local variable.
  bool hasLocalStorage() const {
    if (getStorageClass() == None)
      return !isFileVarDecl();
    
    // Return true for:  Auto, Register.
    // Return false for: Extern, Static, PrivateExtern.
    
    return getStorageClass() <= Register;
  }

  /// hasExternStorage - Returns true if a variable has extern or
  /// __private_extern__ storage.
  bool hasExternalStorage() const {
    return getStorageClass() == Extern || getStorageClass() == PrivateExtern;
  }

  /// hasGlobalStorage - Returns true for all variables that do not
  ///  have local storage.  This includs all global variables as well
  ///  as static variables declared within a function.
  bool hasGlobalStorage() const { return !hasLocalStorage(); }

  /// isBlockVarDecl - Returns true for local variable declarations.  Note that
  /// this includes static variables inside of functions.
  ///
  ///   void foo() { int x; static int y; extern int z; }
  ///
  bool isBlockVarDecl() const {
    if (getKind() != Decl::Var)
      return false;
    if (const DeclContext *DC = getDeclContext())
      return DC->getLookupContext()->isFunctionOrMethod();
    return false;
  }
  
  /// \brief Determines whether this is a static data member.
  ///
  /// This will only be true in C++, and applies to, e.g., the
  /// variable 'x' in:
  /// \code
  /// struct S {
  ///   static int x;
  /// };
  /// \endcode
  bool isStaticDataMember() const {
    return getDeclContext()->isRecord();
  }

  /// isFileVarDecl - Returns true for file scoped variable declaration.
  bool isFileVarDecl() const {
    if (getKind() != Decl::Var)
      return false;
    if (const DeclContext *Ctx = getDeclContext()) {
      Ctx = Ctx->getLookupContext();
      if (isa<TranslationUnitDecl>(Ctx) || isa<NamespaceDecl>(Ctx) )
        return true;
    }
    return false;
  }

  /// \brief Determine whether this is a tentative definition of a
  /// variable in C.
  bool isTentativeDefinition(ASTContext &Context) const;
  
  /// \brief Determines whether this variable is a variable with
  /// external, C linkage.
  bool isExternC(ASTContext &Context) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= VarFirst && D->getKind() <= VarLast;
  }
  static bool classof(const VarDecl *D) { return true; }
};

class ImplicitParamDecl : public VarDecl {
protected:
  ImplicitParamDecl(Kind DK, DeclContext *DC, SourceLocation L,
            IdentifierInfo *Id, QualType Tw) 
    : VarDecl(DK, DC, L, Id, Tw, VarDecl::None) {}
public:
  static ImplicitParamDecl *Create(ASTContext &C, DeclContext *DC,
                         SourceLocation L, IdentifierInfo *Id,
                         QualType T);
  // Implement isa/cast/dyncast/etc.
  static bool classof(const ImplicitParamDecl *D) { return true; }
  static bool classof(const Decl *D) { return D->getKind() == ImplicitParam; }
};

/// ParmVarDecl - Represent a parameter to a function.
class ParmVarDecl : public VarDecl {
  // NOTE: VC++ treats enums as signed, avoid using the ObjCDeclQualifier enum
  /// FIXME: Also can be paced into the bitfields in Decl.
  /// in, inout, etc.
  unsigned objcDeclQualifier : 6;
  
  /// Default argument, if any.  [C++ Only]
  Expr *DefaultArg;
protected:
  ParmVarDecl(Kind DK, DeclContext *DC, SourceLocation L,
              IdentifierInfo *Id, QualType T, StorageClass S,
              Expr *DefArg)
    : VarDecl(DK, DC, L, Id, T, S), 
      objcDeclQualifier(OBJC_TQ_None), DefaultArg(DefArg) {}

public:
  static ParmVarDecl *Create(ASTContext &C, DeclContext *DC,
                             SourceLocation L,IdentifierInfo *Id,
                             QualType T, StorageClass S, Expr *DefArg);
  
  ObjCDeclQualifier getObjCDeclQualifier() const {
    return ObjCDeclQualifier(objcDeclQualifier);
  }
  void setObjCDeclQualifier(ObjCDeclQualifier QTVal) {
    objcDeclQualifier = QTVal;
  }
    
  const Expr *getDefaultArg() const { return DefaultArg; }
  Expr *getDefaultArg() { return DefaultArg; }
  void setDefaultArg(Expr *defarg) { DefaultArg = defarg; }

  /// hasUnparsedDefaultArg - Determines whether this parameter has a
  /// default argument that has not yet been parsed. This will occur
  /// during the processing of a C++ class whose member functions have
  /// default arguments, e.g.,
  /// @code
  ///   class X {
  ///   public:
  ///     void f(int x = 17); // x has an unparsed default argument now
  ///   }; // x has a regular default argument now
  /// @endcode
  bool hasUnparsedDefaultArg() const {
    return DefaultArg == reinterpret_cast<Expr *>(-1);
  }

  /// setUnparsedDefaultArg - Specify that this parameter has an
  /// unparsed default argument. The argument will be replaced with a
  /// real default argument via setDefaultArg when the class
  /// definition enclosing the function declaration that owns this
  /// default argument is completed.
  void setUnparsedDefaultArg() { DefaultArg = reinterpret_cast<Expr *>(-1); }

  QualType getOriginalType() const;
  
  /// setOwningFunction - Sets the function declaration that owns this
  /// ParmVarDecl. Since ParmVarDecls are often created before the
  /// FunctionDecls that own them, this routine is required to update
  /// the DeclContext appropriately.
  void setOwningFunction(DeclContext *FD) { setDeclContext(FD); }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { 
    return (D->getKind() == ParmVar ||
            D->getKind() == OriginalParmVar); 
  }
  static bool classof(const ParmVarDecl *D) { return true; }
};

/// OriginalParmVarDecl - Represent a parameter to a function, when
/// the type of the parameter has been promoted. This node represents the
/// parameter to the function with its original type.
///
class OriginalParmVarDecl : public ParmVarDecl {
  friend class ParmVarDecl;
protected:
  QualType OriginalType;
private:
  OriginalParmVarDecl(DeclContext *DC, SourceLocation L,
                              IdentifierInfo *Id, QualType T, 
                              QualType OT, StorageClass S,
                              Expr *DefArg)
  : ParmVarDecl(OriginalParmVar, DC, L, Id, T, S, DefArg), OriginalType(OT) {}
public:
  static OriginalParmVarDecl *Create(ASTContext &C, DeclContext *DC,
                                     SourceLocation L,IdentifierInfo *Id,
                                     QualType T, QualType OT,
                                     StorageClass S, Expr *DefArg);

  void setOriginalType(QualType T) { OriginalType = T; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == OriginalParmVar; }
  static bool classof(const OriginalParmVarDecl *D) { return true; }
};
  
/// FunctionDecl - An instance of this class is created to represent a
/// function declaration or definition. 
///
/// Since a given function can be declared several times in a program,
/// there may be several FunctionDecls that correspond to that
/// function. Only one of those FunctionDecls will be found when
/// traversing the list of declarations in the context of the
/// FunctionDecl (e.g., the translation unit); this FunctionDecl
/// contains all of the information known about the function. Other,
/// previous declarations of the function are available via the
/// getPreviousDeclaration() chain. 
class FunctionDecl : public ValueDecl, public DeclContext {
public:
  enum StorageClass {
    None, Extern, Static, PrivateExtern
  };
private:
  /// ParamInfo - new[]'d array of pointers to VarDecls for the formal
  /// parameters of this function.  This is null if a prototype or if there are
  /// no formals.  TODO: we could allocate this space immediately after the
  /// FunctionDecl object to save an allocation like FunctionType does.
  ParmVarDecl **ParamInfo;
  
  LazyDeclStmtPtr Body;
  
  /// PreviousDeclaration - A link to the previous declaration of this
  /// same function, NULL if this is the first declaration. For
  /// example, in the following code, the PreviousDeclaration can be
  /// traversed several times to see all three declarations of the
  /// function "f", the last of which is also a definition.
  ///
  ///   int f(int x, int y = 1);
  ///   int f(int x = 0, int y);
  ///   int f(int x, int y) { return x + y; }
  FunctionDecl *PreviousDeclaration;

  // FIXME: This can be packed into the bitfields in Decl.
  // NOTE: VC++ treats enums as signed, avoid using the StorageClass enum
  unsigned SClass : 2;
  bool IsInline : 1;
  bool C99InlineDefinition : 1;
  bool IsVirtual : 1;
  bool IsPure : 1;
  bool InheritedPrototype : 1;
  bool HasPrototype : 1;
  bool IsDeleted : 1;

  // Move to DeclGroup when it is implemented.
  SourceLocation TypeSpecStartLoc;
protected:
  FunctionDecl(Kind DK, DeclContext *DC, SourceLocation L,
               DeclarationName N, QualType T,
               StorageClass S, bool isInline,
               SourceLocation TSSL = SourceLocation())
    : ValueDecl(DK, DC, L, N, T), 
      DeclContext(DK),
      ParamInfo(0), Body(), PreviousDeclaration(0),
      SClass(S), IsInline(isInline), C99InlineDefinition(false), 
      IsVirtual(false), IsPure(false), InheritedPrototype(false), 
      HasPrototype(true), IsDeleted(false), TypeSpecStartLoc(TSSL) {}

  virtual ~FunctionDecl() {}
  virtual void Destroy(ASTContext& C);

public:
  static FunctionDecl *Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                              DeclarationName N, QualType T, 
                              StorageClass S = None, bool isInline = false,
                              bool hasPrototype = true,
                              SourceLocation TSStartLoc = SourceLocation());  
  
  SourceLocation getTypeSpecStartLoc() const { return TypeSpecStartLoc; }
  void setTypeSpecStartLoc(SourceLocation TS) { TypeSpecStartLoc = TS; }

  /// getBody - Retrieve the body (definition) of the function. The
  /// function body might be in any of the (re-)declarations of this
  /// function. The variant that accepts a FunctionDecl pointer will
  /// set that function declaration to the actual declaration
  /// containing the body (if there is one).
  Stmt *getBody(ASTContext &Context, const FunctionDecl *&Definition) const;

  virtual Stmt *getBody(ASTContext &Context) const {
    const FunctionDecl* Definition;
    return getBody(Context, Definition);
  }

  /// \brief If the function has a body that is immediately available,
  /// return it.
  Stmt *getBodyIfAvailable() const;

  /// isThisDeclarationADefinition - Returns whether this specific
  /// declaration of the function is also a definition. This does not
  /// determine whether the function has been defined (e.g., in a
  /// previous definition); for that information, use getBody.
  /// FIXME: Should return true if function is deleted or defaulted. However,
  /// CodeGenModule.cpp uses it, and I don't know if this would break it.
  bool isThisDeclarationADefinition() const { return Body; }

  void setBody(Stmt *B) { Body = B; }
  void setLazyBody(uint64_t Offset) { Body = Offset; }

  /// Whether this function is virtual, either by explicit marking, or by
  /// overriding a virtual function. Only valid on C++ member functions.
  bool isVirtual() { return IsVirtual; }
  void setVirtual(bool V = true) { IsVirtual = V; }

  /// Whether this virtual function is pure, i.e. makes the containing class
  /// abstract.
  bool isPure() const { return IsPure; }
  void setPure(bool P = true) { IsPure = P; }

  /// \brief Whether this function has a prototype, either because one
  /// was explicitly written or because it was "inherited" by merging
  /// a declaration without a prototype with a declaration that has a
  /// prototype.
  bool hasPrototype() const { return HasPrototype || InheritedPrototype; }
  void setHasPrototype(bool P) { HasPrototype = P; }

  /// \brief Whether this function inherited its prototype from a
  /// previous declaration.
  bool inheritedPrototype() const { return InheritedPrototype; }
  void setInheritedPrototype(bool P = true) { InheritedPrototype = P; }

  /// \brief Whether this function has been deleted.
  ///
  /// A function that is "deleted" (via the C++0x "= delete" syntax)
  /// acts like a normal function, except that it cannot actually be
  /// called or have its address taken. Deleted functions are
  /// typically used in C++ overload resolution to attract arguments
  /// whose type or lvalue/rvalue-ness would permit the use of a
  /// different overload that would behave incorrectly. For example,
  /// one might use deleted functions to ban implicit conversion from
  /// a floating-point number to an Integer type:
  ///
  /// @code
  /// struct Integer {
  ///   Integer(long); // construct from a long
  ///   Integer(double) = delete; // no construction from float or double
  ///   Integer(long double) = delete; // no construction from long double
  /// };
  /// @endcode
  bool isDeleted() const { return IsDeleted; }
  void setDeleted(bool D = true) { IsDeleted = D; }

  /// \brief Determines whether this is a function "main", which is
  /// the entry point into an executable program.
  bool isMain() const;

  /// \brief Determines whether this function is a function with
  /// external, C linkage.
  bool isExternC(ASTContext &Context) const;

  /// \brief Determines whether this is a global function.
  bool isGlobal() const;

  /// getPreviousDeclaration - Return the previous declaration of this
  /// function.
  const FunctionDecl *getPreviousDeclaration() const {
    return PreviousDeclaration;
  }

  void setPreviousDeclaration(FunctionDecl * PrevDecl) {
    PreviousDeclaration = PrevDecl;
  }

  unsigned getBuiltinID(ASTContext &Context) const;

  unsigned getNumParmVarDeclsFromType() const;
  
  // Iterator access to formal parameters.
  unsigned param_size() const { return getNumParams(); }
  typedef ParmVarDecl **param_iterator;
  typedef ParmVarDecl * const *param_const_iterator;
  
  param_iterator param_begin() { return ParamInfo; }
  param_iterator param_end()   { return ParamInfo+param_size(); }
  
  param_const_iterator param_begin() const { return ParamInfo; }
  param_const_iterator param_end() const   { return ParamInfo+param_size(); }
  
  /// getNumParams - Return the number of parameters this function must have
  /// based on its functiontype.  This is the length of the PararmInfo array
  /// after it has been created.
  unsigned getNumParams() const;
  
  const ParmVarDecl *getParamDecl(unsigned i) const {
    assert(i < getNumParams() && "Illegal param #");
    return ParamInfo[i];
  }
  ParmVarDecl *getParamDecl(unsigned i) {
    assert(i < getNumParams() && "Illegal param #");
    return ParamInfo[i];
  }
  void setParams(ASTContext& C, ParmVarDecl **NewParamInfo, unsigned NumParams);

  /// getMinRequiredArguments - Returns the minimum number of arguments
  /// needed to call this function. This may be fewer than the number of
  /// function parameters, if some of the parameters have default
  /// arguments (in C++).
  unsigned getMinRequiredArguments() const;

  QualType getResultType() const { 
    return getType()->getAsFunctionType()->getResultType();
  }
  StorageClass getStorageClass() const { return StorageClass(SClass); }
  void setStorageClass(StorageClass SC) { SClass = SC; }

  bool isInline() const { return IsInline; }
  void setInline(bool I) { IsInline = I; }

  /// \brief Whether this function is an "inline definition" as
  /// defined by C99.
  bool isC99InlineDefinition() const { return C99InlineDefinition; }
  void setC99InlineDefinition(bool I) { C99InlineDefinition = I; }

  /// \brief Determines whether this function has a gnu_inline
  /// attribute that affects its semantics.
  ///
  /// The gnu_inline attribute only introduces GNU inline semantics
  /// when all of the inline declarations of the function are marked
  /// gnu_inline.
  bool hasActiveGNUInlineAttribute() const;

  /// \brief Determines whether this function is a GNU "extern
  /// inline", which is roughly the opposite of a C99 "extern inline"
  /// function.
  bool isExternGNUInline() const;

  /// isOverloadedOperator - Whether this function declaration
  /// represents an C++ overloaded operator, e.g., "operator+".
  bool isOverloadedOperator() const { 
    return getOverloadedOperator() != OO_None;
  };

  OverloadedOperatorKind getOverloadedOperator() const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= FunctionFirst && D->getKind() <= FunctionLast;
  }
  static bool classof(const FunctionDecl *D) { return true; }
  static DeclContext *castToDeclContext(const FunctionDecl *D) {
    return static_cast<DeclContext *>(const_cast<FunctionDecl*>(D));
  }
  static FunctionDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<FunctionDecl *>(const_cast<DeclContext*>(DC));
  }
};


/// FieldDecl - An instance of this class is created by Sema::ActOnField to 
/// represent a member of a struct/union/class.
class FieldDecl : public ValueDecl {
  // FIXME: This can be packed into the bitfields in Decl.
  bool Mutable : 1;
  Expr *BitWidth;
protected:
  FieldDecl(Kind DK, DeclContext *DC, SourceLocation L, 
            IdentifierInfo *Id, QualType T, Expr *BW, bool Mutable)
    : ValueDecl(DK, DC, L, Id, T), Mutable(Mutable), BitWidth(BW)
      { }

public:
  static FieldDecl *Create(ASTContext &C, DeclContext *DC, SourceLocation L, 
                           IdentifierInfo *Id, QualType T, Expr *BW, 
                           bool Mutable);

  /// isMutable - Determines whether this field is mutable (C++ only).
  bool isMutable() const { return Mutable; }

  /// \brief Set whether this field is mutable (C++ only).
  void setMutable(bool M) { Mutable = M; }

  /// isBitfield - Determines whether this field is a bitfield.
  bool isBitField() const { return BitWidth != NULL; }

  /// @brief Determines whether this is an unnamed bitfield.
  bool isUnnamedBitfield() const { return BitWidth != NULL && !getDeclName(); }

  /// isAnonymousStructOrUnion - Determines whether this field is a
  /// representative for an anonymous struct or union. Such fields are
  /// unnamed and are implicitly generated by the implementation to
  /// store the data for the anonymous union or struct.
  bool isAnonymousStructOrUnion() const;

  Expr *getBitWidth() const { return BitWidth; }
  void setBitWidth(Expr *BW) { BitWidth = BW; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= FieldFirst && D->getKind() <= FieldLast;
  }
  static bool classof(const FieldDecl *D) { return true; }
};

/// EnumConstantDecl - An instance of this object exists for each enum constant
/// that is defined.  For example, in "enum X {a,b}", each of a/b are
/// EnumConstantDecl's, X is an instance of EnumDecl, and the type of a/b is a
/// TagType for the X EnumDecl.
class EnumConstantDecl : public ValueDecl {
  Stmt *Init; // an integer constant expression
  llvm::APSInt Val; // The value.
protected:
  EnumConstantDecl(DeclContext *DC, SourceLocation L,
                   IdentifierInfo *Id, QualType T, Expr *E,
                   const llvm::APSInt &V)
    : ValueDecl(EnumConstant, DC, L, Id, T), Init((Stmt*)E), Val(V) {}

  virtual ~EnumConstantDecl() {}
public:

  static EnumConstantDecl *Create(ASTContext &C, EnumDecl *DC,
                                  SourceLocation L, IdentifierInfo *Id,
                                  QualType T, Expr *E,
                                  const llvm::APSInt &V);
  
  virtual void Destroy(ASTContext& C);

  const Expr *getInitExpr() const { return (const Expr*) Init; }
  Expr *getInitExpr() { return (Expr*) Init; }
  const llvm::APSInt &getInitVal() const { return Val; }

  void setInitExpr(Expr *E) { Init = (Stmt*) E; }
  void setInitVal(const llvm::APSInt &V) { Val = V; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == EnumConstant; }
  static bool classof(const EnumConstantDecl *D) { return true; }
  
  friend class StmtIteratorBase;
};


/// TypeDecl - Represents a declaration of a type.
///
class TypeDecl : public NamedDecl {
  /// TypeForDecl - This indicates the Type object that represents
  /// this TypeDecl.  It is a cache maintained by
  /// ASTContext::getTypedefType, ASTContext::getTagDeclType, and
  /// ASTContext::getTemplateTypeParmType, and TemplateTypeParmDecl.
  mutable Type *TypeForDecl;
  friend class ASTContext;
  friend class DeclContext;
  friend class TagDecl;
  friend class TemplateTypeParmDecl;
  friend class ClassTemplateSpecializationDecl;
  friend class TagType;

protected:
  TypeDecl(Kind DK, DeclContext *DC, SourceLocation L,
           IdentifierInfo *Id)
    : NamedDecl(DK, DC, L, Id), TypeForDecl(0) {}

public:
  // Low-level accessor
  Type *getTypeForDecl() const { return TypeForDecl; }
  void setTypeForDecl(Type *TD) { TypeForDecl = TD; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= TypeFirst && D->getKind() <= TypeLast;
  }
  static bool classof(const TypeDecl *D) { return true; }
};


class TypedefDecl : public TypeDecl {
  /// UnderlyingType - This is the type the typedef is set to.
  QualType UnderlyingType;
  TypedefDecl(DeclContext *DC, SourceLocation L,
              IdentifierInfo *Id, QualType T) 
    : TypeDecl(Typedef, DC, L, Id), UnderlyingType(T) {}

  virtual ~TypedefDecl() {}
public:
  
  static TypedefDecl *Create(ASTContext &C, DeclContext *DC,
                             SourceLocation L,IdentifierInfo *Id,
                             QualType T);
  
  QualType getUnderlyingType() const { return UnderlyingType; }
  void setUnderlyingType(QualType newType) { UnderlyingType = newType; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Typedef; }
  static bool classof(const TypedefDecl *D) { return true; }
};

class TypedefDecl;
  
/// TagDecl - Represents the declaration of a struct/union/class/enum.
class TagDecl : public TypeDecl, public DeclContext {
public:
  enum TagKind {
    TK_struct,
    TK_union,
    TK_class,
    TK_enum
  };

private:
  // FIXME: This can be packed into the bitfields in Decl.
  /// TagDeclKind - The TagKind enum.
  unsigned TagDeclKind : 2;

  /// IsDefinition - True if this is a definition ("struct foo {};"), false if
  /// it is a declaration ("struct foo;").
  bool IsDefinition : 1;
  
  /// TypedefForAnonDecl - If a TagDecl is anonymous and part of a typedef,
  /// this points to the TypedefDecl. Used for mangling.
  TypedefDecl *TypedefForAnonDecl;
  
protected:
  TagDecl(Kind DK, TagKind TK, DeclContext *DC, SourceLocation L,
          IdentifierInfo *Id)
    : TypeDecl(DK, DC, L, Id), DeclContext(DK), TypedefForAnonDecl(0) {
    assert((DK != Enum || TK == TK_enum) &&"EnumDecl not matched with TK_enum");
    TagDeclKind = TK;
    IsDefinition = false;
  }
public:
  
  /// isDefinition - Return true if this decl has its body specified.
  bool isDefinition() const {
    return IsDefinition;
  }
  
  /// @brief Starts the definition of this tag declaration.
  /// 
  /// This method should be invoked at the beginning of the definition
  /// of this tag declaration. It will set the tag type into a state
  /// where it is in the process of being defined.
  void startDefinition();

  /// @brief Completes the definition of this tag declaration.
  void completeDefinition();

  /// getDefinition - Returns the TagDecl that actually defines this 
  ///  struct/union/class/enum.  When determining whether or not a
  ///  struct/union/class/enum is completely defined, one should use this method
  ///  as opposed to 'isDefinition'.  'isDefinition' indicates whether or not a
  ///  specific TagDecl is defining declaration, not whether or not the
  ///  struct/union/class/enum type is defined.  This method returns NULL if
  ///  there is no TagDecl that defines the struct/union/class/enum.
  TagDecl* getDefinition(ASTContext& C) const;
  
  const char *getKindName() const {
    switch (getTagKind()) {
    default: assert(0 && "Unknown TagKind!");
    case TK_struct: return "struct";
    case TK_union:  return "union";
    case TK_class:  return "class";
    case TK_enum:   return "enum";
    }
  }

  TagKind getTagKind() const {
    return TagKind(TagDeclKind);
  }

  void setTagKind(TagKind TK) { TagDeclKind = TK; }

  bool isStruct() const { return getTagKind() == TK_struct; }
  bool isClass()  const { return getTagKind() == TK_class; }
  bool isUnion()  const { return getTagKind() == TK_union; }
  bool isEnum()   const { return getTagKind() == TK_enum; }
  
  TypedefDecl *getTypedefForAnonDecl() const { return TypedefForAnonDecl; }
  void setTypedefForAnonDecl(TypedefDecl *TDD) { TypedefForAnonDecl = TDD; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= TagFirst && D->getKind() <= TagLast;
  }
  static bool classof(const TagDecl *D) { return true; }

  static DeclContext *castToDeclContext(const TagDecl *D) {
    return static_cast<DeclContext *>(const_cast<TagDecl*>(D));
  }
  static TagDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<TagDecl *>(const_cast<DeclContext*>(DC));
  }

  void setDefinition(bool V) { IsDefinition = V; }
};

/// EnumDecl - Represents an enum.  As an extension, we allow forward-declared
/// enums.
class EnumDecl : public TagDecl {
  /// IntegerType - This represent the integer type that the enum corresponds
  /// to for code generation purposes.  Note that the enumerator constants may
  /// have a different type than this does.
  QualType IntegerType;
  
  EnumDecl(DeclContext *DC, SourceLocation L,
           IdentifierInfo *Id)
    : TagDecl(Enum, TK_enum, DC, L, Id) {
      IntegerType = QualType();
    }
public:
  static EnumDecl *Create(ASTContext &C, DeclContext *DC,
                          SourceLocation L, IdentifierInfo *Id,
                          EnumDecl *PrevDecl);
  
  virtual void Destroy(ASTContext& C);

  /// completeDefinition - When created, the EnumDecl corresponds to a
  /// forward-declared enum. This method is used to mark the
  /// declaration as being defined; it's enumerators have already been
  /// added (via DeclContext::addDecl). NewType is the new underlying
  /// type of the enumeration type.
  void completeDefinition(ASTContext &C, QualType NewType);
  
  // enumerator_iterator - Iterates through the enumerators of this
  // enumeration.
  typedef specific_decl_iterator<EnumConstantDecl> enumerator_iterator;

  enumerator_iterator enumerator_begin(ASTContext &Context) const { 
    return enumerator_iterator(this->decls_begin(Context));
  }

  enumerator_iterator enumerator_end(ASTContext &Context) const { 
    return enumerator_iterator(this->decls_end(Context));
  }

  /// getIntegerType - Return the integer type this enum decl corresponds to.
  /// This returns a null qualtype for an enum forward definition.
  QualType getIntegerType() const { return IntegerType; }

  /// \brief Set the underlying integer type.
  void setIntegerType(QualType T) { IntegerType = T; }

  static bool classof(const Decl *D) { return D->getKind() == Enum; }
  static bool classof(const EnumDecl *D) { return true; }
};


/// RecordDecl - Represents a struct/union/class.  For example:
///   struct X;                  // Forward declaration, no "body".
///   union Y { int A, B; };     // Has body with members A and B (FieldDecls).
/// This decl will be marked invalid if *any* members are invalid.
///
class RecordDecl : public TagDecl {
  // FIXME: This can be packed into the bitfields in Decl.
  /// HasFlexibleArrayMember - This is true if this struct ends with a flexible
  /// array member (e.g. int X[]) or if this union contains a struct that does.
  /// If so, this cannot be contained in arrays or other structs as a member.
  bool HasFlexibleArrayMember : 1;

  /// AnonymousStructOrUnion - Whether this is the type of an
  /// anonymous struct or union.
  bool AnonymousStructOrUnion : 1;

protected:
  RecordDecl(Kind DK, TagKind TK, DeclContext *DC,
             SourceLocation L, IdentifierInfo *Id);
  virtual ~RecordDecl();

public:
  static RecordDecl *Create(ASTContext &C, TagKind TK, DeclContext *DC,
                            SourceLocation L, IdentifierInfo *Id,
                            RecordDecl* PrevDecl = 0);

  virtual void Destroy(ASTContext& C);
      
  bool hasFlexibleArrayMember() const { return HasFlexibleArrayMember; }
  void setHasFlexibleArrayMember(bool V) { HasFlexibleArrayMember = V; }

  /// isAnonymousStructOrUnion - Whether this is an anonymous struct
  /// or union. To be an anonymous struct or union, it must have been
  /// declared without a name and there must be no objects of this
  /// type declared, e.g.,
  /// @code
  ///   union { int i; float f; };
  /// @endcode   
  /// is an anonymous union but neither of the following are:
  /// @code
  ///  union X { int i; float f; };
  ///  union { int i; float f; } obj;
  /// @endcode
  bool isAnonymousStructOrUnion() const { return AnonymousStructOrUnion; }
  void setAnonymousStructOrUnion(bool Anon) {
    AnonymousStructOrUnion = Anon;
  }

  /// \brief Determines whether this declaration represents the
  /// injected class name.
  ///
  /// The injected class name in C++ is the name of the class that
  /// appears inside the class itself. For example:
  ///
  /// \code
  /// struct C {
  ///   // C is implicitly declared here as a synonym for the class name.
  /// };
  ///
  /// C::C c; // same as "C c;"
  /// \endcode
  bool isInjectedClassName() const;

  /// getDefinition - Returns the RecordDecl that actually defines this 
  ///  struct/union/class.  When determining whether or not a struct/union/class
  ///  is completely defined, one should use this method as opposed to
  ///  'isDefinition'.  'isDefinition' indicates whether or not a specific
  ///  RecordDecl is defining declaration, not whether or not the record
  ///  type is defined.  This method returns NULL if there is no RecordDecl
  ///  that defines the struct/union/tag.
  RecordDecl* getDefinition(ASTContext& C) const {
    return cast_or_null<RecordDecl>(TagDecl::getDefinition(C));
  }
  
  // Iterator access to field members. The field iterator only visits
  // the non-static data members of this class, ignoring any static
  // data members, functions, constructors, destructors, etc.
  typedef specific_decl_iterator<FieldDecl> field_iterator;

  field_iterator field_begin(ASTContext &Context) const {
    return field_iterator(decls_begin(Context));
  }
  field_iterator field_end(ASTContext &Context) const {
    return field_iterator(decls_end(Context));
  }

  // field_empty - Whether there are any fields (non-static data
  // members) in this record.
  bool field_empty(ASTContext &Context) const { 
    return field_begin(Context) == field_end(Context);
  }

  /// completeDefinition - Notes that the definition of this type is
  /// now complete.
  void completeDefinition(ASTContext& C);

  static bool classof(const Decl *D) {
    return D->getKind() >= RecordFirst && D->getKind() <= RecordLast;
  }
  static bool classof(const RecordDecl *D) { return true; }
};

class FileScopeAsmDecl : public Decl {
  StringLiteral *AsmString;
  FileScopeAsmDecl(DeclContext *DC, SourceLocation L, StringLiteral *asmstring)
    : Decl(FileScopeAsm, DC, L), AsmString(asmstring) {}
public:
  static FileScopeAsmDecl *Create(ASTContext &C, DeclContext *DC,
                                  SourceLocation L, StringLiteral *Str);

  const StringLiteral *getAsmString() const { return AsmString; }
  StringLiteral *getAsmString() { return AsmString; }
  void setAsmString(StringLiteral *Asm) { AsmString = Asm; }

  static bool classof(const Decl *D) {
    return D->getKind() == FileScopeAsm;
  }
  static bool classof(const FileScopeAsmDecl *D) { return true; }  
};

/// BlockDecl - This represents a block literal declaration, which is like an
/// unnamed FunctionDecl.  For example:
/// ^{ statement-body }   or   ^(int arg1, float arg2){ statement-body }
///
class BlockDecl : public Decl, public DeclContext {
  /// ParamInfo - new[]'d array of pointers to ParmVarDecls for the formal
  /// parameters of this function.  This is null if a prototype or if there are
  /// no formals.
  ParmVarDecl **ParamInfo;
  unsigned NumParams;
  
  Stmt *Body;
  
protected:
  BlockDecl(DeclContext *DC, SourceLocation CaretLoc)
    : Decl(Block, DC, CaretLoc), DeclContext(Block), 
      ParamInfo(0), NumParams(0), Body(0) {}

  virtual ~BlockDecl();
  virtual void Destroy(ASTContext& C);

public:
  static BlockDecl *Create(ASTContext &C, DeclContext *DC, SourceLocation L);

  SourceLocation getCaretLocation() const { return getLocation(); }

  CompoundStmt *getBody() const { return (CompoundStmt*) Body; }
  Stmt *getBody(ASTContext &C) const { return (Stmt*) Body; }
  void setBody(CompoundStmt *B) { Body = (Stmt*) B; }

  // Iterator access to formal parameters.
  unsigned param_size() const { return getNumParams(); }
  typedef ParmVarDecl **param_iterator;
  typedef ParmVarDecl * const *param_const_iterator;
  
  bool param_empty() const { return NumParams == 0; }
  param_iterator param_begin()  { return ParamInfo; }
  param_iterator param_end()   { return ParamInfo+param_size(); }
  
  param_const_iterator param_begin() const { return ParamInfo; }
  param_const_iterator param_end() const   { return ParamInfo+param_size(); }
  
  unsigned getNumParams() const;
  const ParmVarDecl *getParamDecl(unsigned i) const {
    assert(i < getNumParams() && "Illegal param #");
    return ParamInfo[i];
  }
  ParmVarDecl *getParamDecl(unsigned i) {
    assert(i < getNumParams() && "Illegal param #");
    return ParamInfo[i];
  }
  void setParams(ASTContext& C, ParmVarDecl **NewParamInfo, unsigned NumParams);
    
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Block; }
  static bool classof(const BlockDecl *D) { return true; }  
  static DeclContext *castToDeclContext(const BlockDecl *D) {
    return static_cast<DeclContext *>(const_cast<BlockDecl*>(D));
  }
  static BlockDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<BlockDecl *>(const_cast<DeclContext*>(DC));
  }
};

/// Insertion operator for diagnostics.  This allows sending NamedDecl's
/// into a diagnostic with <<.
inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           NamedDecl* ND) {
  DB.AddTaggedVal(reinterpret_cast<intptr_t>(ND), Diagnostic::ak_nameddecl);
  return DB;
}

}  // end namespace clang

#endif
