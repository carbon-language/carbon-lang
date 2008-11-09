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

#include "clang/Basic/OperatorKinds.h"
#include "clang/AST/DeclBase.h"
#include "clang/Parse/AccessSpecifier.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
class Expr;
class Stmt;
class CompoundStmt;
class StringLiteral;
class IdentifierInfo;

/// TranslationUnitDecl - The top declaration context.
/// FIXME: The TranslationUnit class should probably be modified to serve as
/// the top decl context. It would have ownership of the top decls so that the
/// AST is self-contained and easily de/serializable.
class TranslationUnitDecl : public Decl, public DeclContext {
  TranslationUnitDecl()
    : Decl(TranslationUnit, SourceLocation()),
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

protected:
  /// EmitImpl - Serialize this TranslationUnitDecl. Called by Decl::Emit.
  virtual void EmitImpl(llvm::Serializer& S) const;

  /// CreateImpl - Deserialize a TranslationUnitDecl.  Called by Decl::Create.
  static TranslationUnitDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);

  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
};

/// NamedDecl - This represents a decl with an identifier for a name.  Many
/// decls have names, but not ObjCMethodDecl, @class, etc.
class NamedDecl : public Decl {
  /// Identifier - The identifier for this declaration (e.g. the name for the
  /// variable, the tag for a struct).
  IdentifierInfo *Identifier;
public:
  NamedDecl(Kind DK, SourceLocation L, IdentifierInfo *Id)
   : Decl(DK, L), Identifier(Id) {}
  
  IdentifierInfo *getIdentifier() const { return Identifier; }
  const char *getName() const;
    
  static bool classof(const Decl *D) {
    return D->getKind() >= NamedFirst && D->getKind() <= NamedLast;
  }
  static bool classof(const NamedDecl *D) { return true; }
  
protected:
  void EmitInRec(llvm::Serializer& S) const;
  void ReadInRec(llvm::Deserializer& D, ASTContext& C);
};

/// ScopedDecl - Represent lexically scoped names, used for all ValueDecl's
/// and TypeDecl's.
class ScopedDecl : public NamedDecl {
  /// NextDeclarator - If this decl was part of a multi-declarator declaration,
  /// such as "int X, Y, *Z;" this indicates Decl for the next declarator.
  ScopedDecl *NextDeclarator;
  
  /// When this decl is in scope while parsing, the Next field contains a
  /// pointer to the shadowed decl of the same name.  When the scope is popped,
  /// Decls are relinked onto a containing decl object.
  ///
  ScopedDecl *Next;

  /// DeclCtx - Holds either a DeclContext* or a MultipleDC*.
  /// For declarations that don't contain C++ scope specifiers, it contains
  /// the DeclContext where the ScopedDecl was declared.
  /// For declarations with C++ scope specifiers, it contains a MultipleDC*
  /// with the context where it semantically belongs (SemanticDC) and the
  /// context where it was lexically declared (LexicalDC).
  /// e.g.:
  ///
  ///   namespace A {
  ///      void f(); // SemanticDC == LexicalDC == 'namespace A'
  ///   }
  ///   void A::f(); // SemanticDC == namespace 'A'
  ///                // LexicalDC == global namespace
  uintptr_t DeclCtx;

  struct MultipleDC {
    DeclContext *SemanticDC;
    DeclContext *LexicalDC;
  };

  inline bool isInSemaDC() const { return (DeclCtx & 0x1) == 0; }
  inline bool isOutOfSemaDC() const { return (DeclCtx & 0x1) != 0; }
  inline MultipleDC *getMultipleDC() const {
    return reinterpret_cast<MultipleDC*>(DeclCtx & ~0x1);
  }

protected:
  ScopedDecl(Kind DK, DeclContext *DC, SourceLocation L,
             IdentifierInfo *Id, ScopedDecl *PrevDecl)
    : NamedDecl(DK, L, Id), NextDeclarator(PrevDecl), Next(0),
      DeclCtx(reinterpret_cast<uintptr_t>(DC)) {}

  virtual ~ScopedDecl();
  
public:
  const DeclContext *getDeclContext() const {
    if (isInSemaDC())
      return reinterpret_cast<DeclContext*>(DeclCtx);
    return getMultipleDC()->SemanticDC;
  }
  DeclContext *getDeclContext() {
    return const_cast<DeclContext*>(
                         const_cast<const ScopedDecl*>(this)->getDeclContext());
  }

  /// getLexicalDeclContext - The declaration context where this ScopedDecl was
  /// lexically declared (LexicalDC). May be different from
  /// getDeclContext() (SemanticDC).
  /// e.g.:
  ///
  ///   namespace A {
  ///      void f(); // SemanticDC == LexicalDC == 'namespace A'
  ///   }
  ///   void A::f(); // SemanticDC == namespace 'A'
  ///                // LexicalDC == global namespace
  const DeclContext *getLexicalDeclContext() const {
    if (isInSemaDC())
      return reinterpret_cast<DeclContext*>(DeclCtx);
    return getMultipleDC()->LexicalDC;
  }
  DeclContext *getLexicalDeclContext() {
    return const_cast<DeclContext*>(
                  const_cast<const ScopedDecl*>(this)->getLexicalDeclContext());
  }

  void setLexicalDeclContext(DeclContext *DC);

  ScopedDecl *getNext() const { return Next; }
  void setNext(ScopedDecl *N) { Next = N; }
  
  /// getNextDeclarator - If this decl was part of a multi-declarator
  /// declaration, such as "int X, Y, *Z;" this returns the decl for the next
  /// declarator.  Otherwise it returns null.
  ScopedDecl *getNextDeclarator() { return NextDeclarator; }
  const ScopedDecl *getNextDeclarator() const { return NextDeclarator; }
  void setNextDeclarator(ScopedDecl *N) { NextDeclarator = N; }
  
  // isDefinedOutsideFunctionOrMethod - This predicate returns true if this
  // scoped decl is defined outside the current function or method.  This is
  // roughly global variables and functions, but also handles enums (which could
  // be defined inside or outside a function etc).
  bool isDefinedOutsideFunctionOrMethod() const {
    if (getDeclContext())
      return !getDeclContext()->isFunctionOrMethod();
    else
      return true;
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= ScopedFirst && D->getKind() <= ScopedLast;
  }
  static bool classof(const ScopedDecl *D) { return true; }
  
protected:
  void EmitInRec(llvm::Serializer& S) const;
  void ReadInRec(llvm::Deserializer& D, ASTContext& C);
  
  void EmitOutRec(llvm::Serializer& S) const;
  void ReadOutRec(llvm::Deserializer& D, ASTContext& C);
  
  friend void Decl::Destroy(ASTContext& C);
};

/// NamespaceDecl - Represent a C++ namespace.
class NamespaceDecl : public ScopedDecl, public DeclContext {
  SourceLocation LBracLoc, RBracLoc;
  
  // For extended namespace definitions:
  //
  // namespace A { int x; }
  // namespace A { int y; }
  //
  // there will be one NamespaceDecl for each declaration.
  // NextDeclarator points to the next extended declaration.
  // OrigNamespace points to the original namespace declaration.
  // OrigNamespace of the first namespace decl points to itself.

  NamespaceDecl *OrigNamespace;

  NamespaceDecl(DeclContext *DC, SourceLocation L, IdentifierInfo *Id)
    : ScopedDecl(Namespace, DC, L, Id, 0), DeclContext(Namespace) {
      OrigNamespace = this;
  }
public:
  static NamespaceDecl *Create(ASTContext &C, DeclContext *DC,
                               SourceLocation L, IdentifierInfo *Id);
  
  virtual void Destroy(ASTContext& C);

  NamespaceDecl *getNextNamespace() {
    return cast_or_null<NamespaceDecl>(getNextDeclarator());
  }
  const NamespaceDecl *getNextNamespace() const {
    return cast_or_null<NamespaceDecl>(getNextDeclarator());
  }
  void setNextNamespace(NamespaceDecl *ND) { setNextDeclarator(ND); }

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
  
protected:
  /// EmitImpl - Serialize this NamespaceDecl. Called by Decl::Emit.
  virtual void EmitImpl(llvm::Serializer& S) const;

  /// CreateImpl - Deserialize a NamespaceDecl.  Called by Decl::Create.
  static NamespaceDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);

  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
};

/// ValueDecl - Represent the declaration of a variable (in which case it is 
/// an lvalue) a function (in which case it is a function designator) or
/// an enum constant. 
class ValueDecl : public ScopedDecl {
  QualType DeclType;

protected:
  ValueDecl(Kind DK, DeclContext *DC, SourceLocation L,
            IdentifierInfo *Id, QualType T, ScopedDecl *PrevDecl) 
    : ScopedDecl(DK, DC, L, Id, PrevDecl), DeclType(T) {}
public:
  QualType getType() const { return DeclType; }
  void setType(QualType newType) { DeclType = newType; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= ValueFirst && D->getKind() <= ValueLast;
  }
  static bool classof(const ValueDecl *D) { return true; }
  
protected:
  void EmitInRec(llvm::Serializer& S) const;
  void ReadInRec(llvm::Deserializer& D, ASTContext& C);
};

/// VarDecl - An instance of this class is created to represent a variable
/// declaration or definition.
class VarDecl : public ValueDecl {
public:
  enum StorageClass {
    None, Auto, Register, Extern, Static, PrivateExtern
  };
private:
  Stmt *Init;
  // FIXME: This can be packed into the bitfields in Decl.
  unsigned SClass : 3;
  bool ThreadSpecified : 1;
  bool HasCXXDirectInit : 1; 
  
  // Move to DeclGroup when it is implemented.
  SourceLocation TypeSpecStartLoc;
  friend class StmtIteratorBase;
protected:
  VarDecl(Kind DK, DeclContext *DC, SourceLocation L, IdentifierInfo *Id,
          QualType T, StorageClass SC, ScopedDecl *PrevDecl, 
          SourceLocation TSSL = SourceLocation())
    : ValueDecl(DK, DC, L, Id, T, PrevDecl), Init(0),
          ThreadSpecified(false), HasCXXDirectInit(false),
          TypeSpecStartLoc(TSSL) { SClass = SC; }
public:
  static VarDecl *Create(ASTContext &C, DeclContext *DC,
                         SourceLocation L, IdentifierInfo *Id,
                         QualType T, StorageClass S, ScopedDecl *PrevDecl,
                         SourceLocation TypeSpecStartLoc = SourceLocation());
  
  StorageClass getStorageClass() const { return (StorageClass)SClass; }

  SourceLocation getTypeSpecStartLoc() const { return TypeSpecStartLoc; }
  
  const Expr *getInit() const { return (const Expr*) Init; }
  Expr *getInit() { return (Expr*) Init; }
  void setInit(Expr *I) { Init = (Stmt*) I; }
      
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
  
  /// hasLocalStorage - Returns true if a variable with function scope
  ///  is a non-static local variable.
  bool hasLocalStorage() const {
    if (getStorageClass() == None)
      return !isFileVarDecl();
    
    // Return true for:  Auto, Register.
    // Return false for: Extern, Static, PrivateExtern.
    
    return getStorageClass() <= Register;
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
      return DC->isFunctionOrMethod();
    return false;
  }
  
  /// isFileVarDecl - Returns true for file scoped variable declaration.
  bool isFileVarDecl() const {
    if (getKind() != Decl::Var)
      return false;
    if (isa<TranslationUnitDecl>(getDeclContext()) ||
        isa<NamespaceDecl>(getDeclContext()) )
      return true;
    return false;
  }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= VarFirst && D->getKind() <= VarLast;
  }
  static bool classof(const VarDecl *D) { return true; }

protected:
  void EmitInRec(llvm::Serializer& S) const;
  void ReadInRec(llvm::Deserializer& D, ASTContext& C);
  
  void EmitOutRec(llvm::Serializer& S) const;
  void ReadOutRec(llvm::Deserializer& D, ASTContext& C);
  
  /// EmitImpl - Serialize this VarDecl. Called by Decl::Emit.
  virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// ReadImpl - Deserialize this VarDecl. Called by subclasses.
  virtual void ReadImpl(llvm::Deserializer& D, ASTContext& C);
  
  /// CreateImpl - Deserialize a VarDecl.  Called by Decl::Create.
  static VarDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

class ImplicitParamDecl : public VarDecl {
protected:
  ImplicitParamDecl(Kind DK, DeclContext *DC, SourceLocation L,
            IdentifierInfo *Id, QualType T, ScopedDecl *PrevDecl) 
    : VarDecl(DK, DC, L, Id, T, VarDecl::None, PrevDecl) {}
public:
  static ImplicitParamDecl *Create(ASTContext &C, DeclContext *DC,
                         SourceLocation L, IdentifierInfo *Id,
                         QualType T, ScopedDecl *PrevDecl);
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

  ParmVarDecl(DeclContext *DC, SourceLocation L,
              IdentifierInfo *Id, QualType T, StorageClass S,
              Expr *DefArg, ScopedDecl *PrevDecl)
    : VarDecl(ParmVar, DC, L, Id, T, S, PrevDecl), 
      objcDeclQualifier(OBJC_TQ_None), DefaultArg(DefArg) {}

public:
  static ParmVarDecl *Create(ASTContext &C, DeclContext *DC,
                             SourceLocation L,IdentifierInfo *Id,
                             QualType T, StorageClass S, Expr *DefArg,
                             ScopedDecl *PrevDecl);
  
  ObjCDeclQualifier getObjCDeclQualifier() const {
    return ObjCDeclQualifier(objcDeclQualifier);
  }
  void setObjCDeclQualifier(ObjCDeclQualifier QTVal) 
  { objcDeclQualifier = QTVal; }
    
  const Expr *getDefaultArg() const { return DefaultArg; }
  Expr *getDefaultArg() { return DefaultArg; }
  void setDefaultArg(Expr *defarg) { DefaultArg = defarg; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == ParmVar; }
  static bool classof(const ParmVarDecl *D) { return true; }
  
protected:
  /// EmitImpl - Serialize this ParmVarDecl. Called by Decl::Emit.
  virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize a ParmVarDecl.  Called by Decl::Create.
  static ParmVarDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);

  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
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
  
  Stmt *Body;  // Null if a prototype.
  
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

  // NOTE: VC++ treats enums as signed, avoid using the StorageClass enum
  unsigned SClass : 2;
  bool IsInline : 1;
  bool IsImplicit : 1;

  // Move to DeclGroup when it is implemented.
  SourceLocation TypeSpecStartLoc;
protected:
  FunctionDecl(Kind DK, DeclContext *DC, SourceLocation L,
               IdentifierInfo *Id, QualType T,
               StorageClass S, bool isInline, ScopedDecl *PrevDecl,
               SourceLocation TSSL = SourceLocation())
    : ValueDecl(DK, DC, L, Id, T, PrevDecl), 
      DeclContext(DK),
      ParamInfo(0), Body(0), PreviousDeclaration(0),
      SClass(S), IsInline(isInline), IsImplicit(0), TypeSpecStartLoc(TSSL) {}

  virtual ~FunctionDecl();
  virtual void Destroy(ASTContext& C);

public:
  static FunctionDecl *Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                              IdentifierInfo *Id, QualType T, 
                              StorageClass S = None, bool isInline = false, 
                              ScopedDecl *PrevDecl = 0,
                              SourceLocation TSStartLoc = SourceLocation());  
  
  SourceLocation getTypeSpecStartLoc() const { return TypeSpecStartLoc; }

  /// getBody - Retrieve the body (definition) of the function. The
  /// function body might be in any of the (re-)declarations of this
  /// function. The variant that accepts a FunctionDecl pointer will
  /// set that function declaration to the actual declaration
  /// containing the body (if there is one).
  Stmt *getBody(const FunctionDecl *&Definition) const;

  virtual Stmt *getBody() const { 
    const FunctionDecl* Definition;
    return getBody(Definition);
  }
  
  /// isThisDeclarationADefinition - Returns whether this specific
  /// declaration of the function is also a definition. This does not
  /// determine whether the function has been defined (e.g., in a
  /// previous definition); for that information, use getBody.
  bool isThisDeclarationADefinition() const { return Body != 0; }

  void setBody(Stmt *B) { Body = B; }
  
  bool isImplicit() { return IsImplicit; }
  void setImplicit() { IsImplicit = true; }

  /// getPreviousDeclaration - Return the previous declaration of this
  /// function.
  const FunctionDecl *getPreviousDeclaration() const {
    return PreviousDeclaration;
  }

  void setPreviousDeclaration(FunctionDecl * PrevDecl) {
    PreviousDeclaration = PrevDecl;
  }

  // Iterator access to formal parameters.
  unsigned param_size() const { return getNumParams(); }
  typedef ParmVarDecl **param_iterator;
  typedef ParmVarDecl * const *param_const_iterator;
  
  param_iterator param_begin() { return ParamInfo; }
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
  void setParams(ParmVarDecl **NewParamInfo, unsigned NumParams);

  /// getMinRequiredArguments - Returns the minimum number of arguments
  /// needed to call this function. This may be fewer than the number of
  /// function parameters, if some of the parameters have default
  /// arguments (in C++).
  unsigned getMinRequiredArguments() const;

  QualType getResultType() const { 
    return getType()->getAsFunctionType()->getResultType();
  }
  StorageClass getStorageClass() const { return StorageClass(SClass); }
  bool isInline() const { return IsInline; }
 
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

protected:
  /// EmitImpl - Serialize this FunctionDecl.  Called by Decl::Emit.
  virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize a FunctionDecl.  Called by Decl::Create.
  static FunctionDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);
  
  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
  friend class CXXRecordDecl;
};


/// FieldDecl - An instance of this class is created by Sema::ActOnField to 
/// represent a member of a struct/union/class.
class FieldDecl : public NamedDecl {
  QualType DeclType;  
  Expr *BitWidth;
protected:
  FieldDecl(Kind DK, SourceLocation L, IdentifierInfo *Id, QualType T,
            Expr *BW = NULL)
    : NamedDecl(DK, L, Id), DeclType(T), BitWidth(BW) {}
  FieldDecl(SourceLocation L, IdentifierInfo *Id, QualType T, Expr *BW)
    : NamedDecl(Field, L, Id), DeclType(T), BitWidth(BW) {}
public:
  static FieldDecl *Create(ASTContext &C, SourceLocation L, IdentifierInfo *Id,
                           QualType T, Expr *BW = NULL);

  QualType getType() const { return DeclType; }
  
  bool isBitField() const { return BitWidth != NULL; }
  Expr *getBitWidth() const { return BitWidth; }
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= FieldFirst && D->getKind() <= FieldLast;
  }
  static bool classof(const FieldDecl *D) { return true; }

protected:
  /// EmitImpl - Serialize this FieldDecl.  Called by Decl::Emit.
  virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize a FieldDecl.  Called by Decl::Create.
  static FieldDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);
  
  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
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
                   const llvm::APSInt &V, ScopedDecl *PrevDecl)
    : ValueDecl(EnumConstant, DC, L, Id, T, PrevDecl), Init((Stmt*)E), Val(V) {}

  virtual ~EnumConstantDecl() {}
public:

  static EnumConstantDecl *Create(ASTContext &C, EnumDecl *DC,
                                  SourceLocation L, IdentifierInfo *Id,
                                  QualType T, Expr *E,
                                  const llvm::APSInt &V, ScopedDecl *PrevDecl);
  
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
  
protected:
  /// EmitImpl - Serialize this EnumConstantDecl.  Called by Decl::Emit.
  virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize a EnumConstantDecl.  Called by Decl::Create.
  static EnumConstantDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);
  
  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
};


/// TypeDecl - Represents a declaration of a type.
///
class TypeDecl : public ScopedDecl {
  /// TypeForDecl - This indicates the Type object that represents this
  /// TypeDecl.  It is a cache maintained by ASTContext::getTypedefType and
  /// ASTContext::getTagDeclType.
  Type *TypeForDecl;
  friend class ASTContext;
protected:
  TypeDecl(Kind DK, DeclContext *DC, SourceLocation L,
           IdentifierInfo *Id, ScopedDecl *PrevDecl)
    : ScopedDecl(DK, DC, L, Id, PrevDecl), TypeForDecl(0) {}
public:
  void setAccess(AccessSpecifier AS) { Access = AS; }
  AccessSpecifier getAccess() const { return AccessSpecifier(Access); }

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
              IdentifierInfo *Id, QualType T, ScopedDecl *PD) 
    : TypeDecl(Typedef, DC, L, Id, PD), UnderlyingType(T) {}

  virtual ~TypedefDecl() {}
public:
  
  static TypedefDecl *Create(ASTContext &C, DeclContext *DC,
                             SourceLocation L,IdentifierInfo *Id,
                             QualType T, ScopedDecl *PD);
  
  QualType getUnderlyingType() const { return UnderlyingType; }
  void setUnderlyingType(QualType newType) { UnderlyingType = newType; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Typedef; }
  static bool classof(const TypedefDecl *D) { return true; }

protected:
  /// EmitImpl - Serialize this TypedefDecl.  Called by Decl::Emit.
  virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize a TypedefDecl.  Called by Decl::Create.
  static TypedefDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);
  
  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
  
};


/// TagDecl - Represents the declaration of a struct/union/class/enum.
class TagDecl : public TypeDecl {
public:
  enum TagKind {
    TK_struct,
    TK_union,
    TK_class,
    TK_enum
  };

private:
  /// TagDeclKind - The TagKind enum.
  unsigned TagDeclKind : 2;

  /// IsDefinition - True if this is a definition ("struct foo {};"), false if
  /// it is a declaration ("struct foo;").
  bool IsDefinition : 1;
protected:
  TagDecl(Kind DK, TagKind TK, DeclContext *DC, SourceLocation L,
          IdentifierInfo *Id, ScopedDecl *PrevDecl)
    : TypeDecl(DK, DC, L, Id, PrevDecl) {
    assert((DK != Enum || TK == TK_enum) &&"EnumDecl not matched with TK_enum");
    TagDeclKind = TK;
    IsDefinition = false;
  }
public:
  
  /// isDefinition - Return true if this decl has its body specified.
  bool isDefinition() const {
    return IsDefinition;
  }
  
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

  bool isStruct() const { return getTagKind() == TK_struct; }
  bool isClass()  const { return getTagKind() == TK_class; }
  bool isUnion()  const { return getTagKind() == TK_union; }
  bool isEnum()   const { return getTagKind() == TK_enum; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= TagFirst && D->getKind() <= TagLast;
  }
  static bool classof(const TagDecl *D) { return true; }
protected:
  void setDefinition(bool V) { IsDefinition = V; }
};

/// EnumDecl - Represents an enum.  As an extension, we allow forward-declared
/// enums.
class EnumDecl : public TagDecl, public DeclContext {
  // EnumDecl's DeclChain points to a linked list of EnumConstantDecl's which
  // are linked together through their getNextDeclarator pointers.

  /// IntegerType - This represent the integer type that the enum corresponds
  /// to for code generation purposes.  Note that the enumerator constants may
  /// have a different type than this does.
  QualType IntegerType;
  
  EnumDecl(DeclContext *DC, SourceLocation L,
           IdentifierInfo *Id, ScopedDecl *PrevDecl)
    : TagDecl(Enum, TK_enum, DC, L, Id, PrevDecl), DeclContext(Enum) {
      IntegerType = QualType();
    }
public:
  static EnumDecl *Create(ASTContext &C, DeclContext *DC,
                          SourceLocation L, IdentifierInfo *Id,
                          ScopedDecl *PrevDecl);
  
  virtual void Destroy(ASTContext& C);

  /// defineElements - When created, EnumDecl correspond to a forward declared
  /// enum.  This method is used to mark the decl as being defined, with the
  /// specified list of enums.
  void defineElements(EnumConstantDecl *ListHead, QualType NewType) {
    assert(!isDefinition() && "Cannot redefine enums!");
    setDeclChain(ListHead);
    setDefinition(true);
    
    IntegerType = NewType;
  }
  
  /// getIntegerType - Return the integer type this enum decl corresponds to.
  /// This returns a null qualtype for an enum forward definition.
  QualType getIntegerType() const { return IntegerType; }
  
  /// getEnumConstantList - Return the first EnumConstantDecl in the enum.
  ///
  EnumConstantDecl *getEnumConstantList() {
    return cast_or_null<EnumConstantDecl>(getDeclChain());
  }
  const EnumConstantDecl *getEnumConstantList() const {
    return cast_or_null<const EnumConstantDecl>(getDeclChain());
  }
  
  static bool classof(const Decl *D) { return D->getKind() == Enum; }
  static bool classof(const EnumDecl *D) { return true; }
  static DeclContext *castToDeclContext(const EnumDecl *D) {
    return static_cast<DeclContext *>(const_cast<EnumDecl*>(D));
  }
  static EnumDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<EnumDecl *>(const_cast<DeclContext*>(DC));
  }
  
protected:
  /// EmitImpl - Serialize this EnumDecl.  Called by Decl::Emit.
  virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize a EnumDecl.  Called by Decl::Create.
  static EnumDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);
  
  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
};


/// RecordDecl - Represents a struct/union/class.  For example:
///   struct X;                  // Forward declaration, no "body".
///   union Y { int A, B; };     // Has body with members A and B (FieldDecls).
/// This decl will be marked invalid if *any* members are invalid.
///
class RecordDecl : public TagDecl {
  /// HasFlexibleArrayMember - This is true if this struct ends with a flexible
  /// array member (e.g. int X[]) or if this union contains a struct that does.
  /// If so, this cannot be contained in arrays or other structs as a member.
  bool HasFlexibleArrayMember : 1;
  
  /// Members/NumMembers - This is a new[]'d array of pointers to Decls.
  FieldDecl **Members;   // Null if not defined.
  int NumMembers;   // -1 if not defined.

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
  
  /// getNumMembers - Return the number of members, or -1 if this is a forward
  /// definition.
  int getNumMembers() const { return NumMembers; }
  const FieldDecl *getMember(unsigned i) const { return Members[i]; }
  FieldDecl *getMember(unsigned i) { return Members[i]; }

  // Iterator access to field members.
  typedef FieldDecl **field_iterator;
  typedef FieldDecl * const *field_const_iterator;

  field_iterator field_begin() {
    assert(isDefinition() && "Not a definition!");
    return Members;
  }
  field_iterator field_end() {
    assert(isDefinition() && "Not a definition!");
    return Members + getNumMembers();
  }

  field_const_iterator field_begin() const {
    assert(isDefinition() && "Not a definition!");
    return Members;
  }
  field_const_iterator field_end() const {
    assert(isDefinition() && "Not a definition!");
    return Members + getNumMembers();
  }

  /// defineBody - When created, RecordDecl's correspond to a forward declared
  /// record.  This method is used to mark the decl as being defined, with the
  /// specified contents.
  void defineBody(ASTContext& C, FieldDecl **Members, unsigned numMembers);

  /// getMember - If the member doesn't exist, or there are no members, this 
  /// function will return 0;
  FieldDecl *getMember(IdentifierInfo *name);

  static bool classof(const Decl *D) {
    return D->getKind() >= RecordFirst && D->getKind() <= RecordLast;
  }
  static bool classof(const RecordDecl *D) { return true; }

protected:
  /// EmitImpl - Serialize this RecordDecl.  Called by Decl::Emit.
  virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize a RecordDecl.  Called by Decl::Create.
  static RecordDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);
  
  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
};

class FileScopeAsmDecl : public Decl {
  StringLiteral *AsmString;
  FileScopeAsmDecl(SourceLocation L, StringLiteral *asmstring)
    : Decl(FileScopeAsm, L), AsmString(asmstring) {}
public:
  static FileScopeAsmDecl *Create(ASTContext &C, SourceLocation L,
                                  StringLiteral *Str);

  const StringLiteral *getAsmString() const { return AsmString; }
  StringLiteral *getAsmString() { return AsmString; }
  static bool classof(const Decl *D) {
    return D->getKind() == FileScopeAsm;
  }
  static bool classof(const FileScopeAsmDecl *D) { return true; }  
protected:
  /// EmitImpl - Serialize this FileScopeAsmDecl. Called by Decl::Emit.
  virtual void EmitImpl(llvm::Serializer& S) const;
  
  /// CreateImpl - Deserialize a FileScopeAsmDecl.  Called by Decl::Create.
  static FileScopeAsmDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);
  
  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
};

/// BlockDecl - This represents a block literal declaration, which is like an
/// unnamed FunctionDecl.  For example:
/// ^{ statement-body }   or   ^(int arg1, float arg2){ statement-body }
///
class BlockDecl : public Decl, public DeclContext {
  llvm::SmallVector<ParmVarDecl*, 8> Args;
  Stmt *Body;
  
  // Since BlockDecl's aren't named/scoped, we need to store the context.
  DeclContext *ParentContext;
protected:
  BlockDecl(DeclContext *DC, SourceLocation CaretLoc)
    : Decl(Block, CaretLoc), DeclContext(Block), Body(0), ParentContext(DC) {}

  virtual ~BlockDecl();
  virtual void Destroy(ASTContext& C);

public:
  static BlockDecl *Create(ASTContext &C, DeclContext *DC, SourceLocation L);

  SourceLocation getCaretLocation() const { return getLocation(); }

  Stmt *getBody() const { return Body; }
  void setBody(Stmt *B) { Body = B; }

  void setArgs(ParmVarDecl **args, unsigned numargs) {
    Args.clear(); 
    Args.insert(Args.begin(), args, args+numargs);
  }
  const DeclContext *getParentContext() const { return ParentContext; }
  DeclContext *getParentContext() { return ParentContext; }
  
  /// arg_iterator - Iterate over the ParmVarDecl's for this block.
  typedef llvm::SmallVector<ParmVarDecl*, 8>::const_iterator param_iterator;
  bool param_empty() const { return Args.empty(); }
  param_iterator param_begin() const { return Args.begin(); }
  param_iterator param_end() const { return Args.end(); }
    
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Block; }
  static bool classof(const BlockDecl *D) { return true; }  
  static DeclContext *castToDeclContext(const BlockDecl *D) {
    return static_cast<DeclContext *>(const_cast<BlockDecl*>(D));
  }
  static BlockDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<BlockDecl *>(const_cast<DeclContext*>(DC));
  }

protected:
  /// EmitImpl - Serialize this BlockDecl. Called by Decl::Emit.
  virtual void EmitImpl(llvm::Serializer& S) const;

  /// CreateImpl - Deserialize a BlockDecl.  Called by Decl::Create.
  static BlockDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);

  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
};

}  // end namespace clang
	
#endif
