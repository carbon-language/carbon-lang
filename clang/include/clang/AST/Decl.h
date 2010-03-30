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

#include "clang/AST/APValue.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Redeclarable.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/Basic/Linkage.h"

namespace clang {
class CXXTemporary;
class Expr;
class FunctionTemplateDecl;
class Stmt;
class CompoundStmt;
class StringLiteral;
class TemplateArgumentList;
class MemberSpecializationInfo;
class FunctionTemplateSpecializationInfo;
class TypeLoc;

/// \brief A container of type source information.
///
/// A client can read the relevant info using TypeLoc wrappers, e.g:
/// @code
/// TypeLoc TL = TypeSourceInfo->getTypeLoc();
/// if (PointerLoc *PL = dyn_cast<PointerLoc>(&TL))
///   PL->getStarLoc().print(OS, SrcMgr);
/// @endcode
///
class TypeSourceInfo {
  QualType Ty;
  // Contains a memory block after the class, used for type source information,
  // allocated by ASTContext.
  friend class ASTContext;
  TypeSourceInfo(QualType ty) : Ty(ty) { }
public:
  /// \brief Return the type wrapped by this type source info.
  QualType getType() const { return Ty; }

  /// \brief Return the TypeLoc wrapper for the type source info.
  TypeLoc getTypeLoc() const;
};

/// TranslationUnitDecl - The top declaration context.
class TranslationUnitDecl : public Decl, public DeclContext {
  ASTContext &Ctx;

  /// The (most recently entered) anonymous namespace for this
  /// translation unit, if one has been created.
  NamespaceDecl *AnonymousNamespace;

  explicit TranslationUnitDecl(ASTContext &ctx)
    : Decl(TranslationUnit, 0, SourceLocation()),
      DeclContext(TranslationUnit),
      Ctx(ctx), AnonymousNamespace(0) {}
public:
  ASTContext &getASTContext() const { return Ctx; }

  NamespaceDecl *getAnonymousNamespace() const { return AnonymousNamespace; }
  void setAnonymousNamespace(NamespaceDecl *D) { AnonymousNamespace = D; }

  static TranslationUnitDecl *Create(ASTContext &C);
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const TranslationUnitDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == TranslationUnit; }
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

  /// getName - Get the name of identifier for this declaration as a StringRef.
  /// This requires that the declaration have a name and that it be a simple
  /// identifier.
  llvm::StringRef getName() const {
    assert(Name.isIdentifier() && "Name is not a simple identifier");
    return getIdentifier() ? getIdentifier()->getName() : "";
  }

  /// getNameAsCString - Get the name of identifier for this declaration as a
  /// C string (const char*).  This requires that the declaration have a name
  /// and that it be a simple identifier.
  //
  // FIXME: Deprecated, move clients to getName().
  const char *getNameAsCString() const {
    assert(Name.isIdentifier() && "Name is not a simple identifier");
    return getIdentifier() ? getIdentifier()->getNameStart() : "";
  }

  /// getNameAsString - Get a human-readable name for the declaration, even if
  /// it is one of the special kinds of names (C++ constructor, Objective-C
  /// selector, etc).  Creating this name requires expensive string
  /// manipulation, so it should be called only when performance doesn't matter.
  /// For simple declarations, getNameAsCString() should suffice.
  //
  // FIXME: This function should be renamed to indicate that it is not just an
  // alternate form of getName(), and clients should move as appropriate.
  //
  // FIXME: Deprecated, move clients to getName().
  std::string getNameAsString() const { return Name.getAsString(); }

  /// getDeclName - Get the actual, stored name of the declaration,
  /// which may be a special name.
  DeclarationName getDeclName() const { return Name; }

  /// \brief Set the name of this declaration.
  void setDeclName(DeclarationName N) { Name = N; }

  /// getQualifiedNameAsString - Returns human-readable qualified name for
  /// declaration, like A::B::i, for i being member of namespace A::B.
  /// If declaration is not member of context which can be named (record,
  /// namespace), it will return same result as getNameAsString().
  /// Creating this name is expensive, so it should be called only when
  /// performance doesn't matter.
  std::string getQualifiedNameAsString() const;
  std::string getQualifiedNameAsString(const PrintingPolicy &Policy) const;

  /// getNameForDiagnostic - Appends a human-readable name for this
  /// declaration into the given string.
  ///
  /// This is the method invoked by Sema when displaying a NamedDecl
  /// in a diagnostic.  It does not necessarily produce the same
  /// result as getNameAsString(); for example, class template
  /// specializations are printed with their template arguments.
  ///
  /// TODO: use an API that doesn't require so many temporary strings
  virtual void getNameForDiagnostic(std::string &S,
                                    const PrintingPolicy &Policy,
                                    bool Qualified) const {
    if (Qualified)
      S += getQualifiedNameAsString(Policy);
    else
      S += getNameAsString();
  }

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

  /// \brief Determine whether this declaration is a C++ class member.
  bool isCXXClassMember() const {
    const DeclContext *DC = getDeclContext();

    // C++0x [class.mem]p1:
    //   The enumerators of an unscoped enumeration defined in
    //   the class are members of the class.
    // FIXME: support C++0x scoped enumerations.
    if (isa<EnumDecl>(DC))
      DC = DC->getParent();

    return DC->isRecord();
  }

  /// \brief Determine what kind of linkage this entity has.
  Linkage getLinkage() const;

  /// \brief Looks through UsingDecls and ObjCCompatibleAliasDecls for
  /// the underlying named decl.
  NamedDecl *getUnderlyingDecl();
  const NamedDecl *getUnderlyingDecl() const {
    return const_cast<NamedDecl*>(this)->getUnderlyingDecl();
  }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const NamedDecl *D) { return true; }
  static bool classofKind(Kind K) { return K >= NamedFirst && K <= NamedLast; }
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
  NamespaceDecl *NextNamespace;

  /// \brief A pointer to either the original namespace definition for
  /// this namespace (if the boolean value is false) or the anonymous
  /// namespace that lives just inside this namespace (if the boolean
  /// value is true).
  ///
  /// We can combine these two notions because the anonymous namespace
  /// must only be stored in one of the namespace declarations (so all
  /// of the namespace declarations can find it). We therefore choose
  /// the original namespace declaration, since all of the namespace
  /// declarations have a link directly to it; the original namespace
  /// declaration itself only needs to know that it is the original
  /// namespace declaration (which the boolean indicates).
  llvm::PointerIntPair<NamespaceDecl *, 1, bool> OrigOrAnonNamespace;

  NamespaceDecl(DeclContext *DC, SourceLocation L, IdentifierInfo *Id)
    : NamedDecl(Namespace, DC, L, Id), DeclContext(Namespace),
      NextNamespace(0), OrigOrAnonNamespace(0, true) { }

public:
  static NamespaceDecl *Create(ASTContext &C, DeclContext *DC,
                               SourceLocation L, IdentifierInfo *Id);

  virtual void Destroy(ASTContext& C);

  // \brief Returns true if this is an anonymous namespace declaration.
  //
  // For example:
  //   namespace {
  //     ...
  //   };
  // q.v. C++ [namespace.unnamed]
  bool isAnonymousNamespace() const {
    return !getIdentifier();
  }

  NamespaceDecl *getNextNamespace() { return NextNamespace; }
  const NamespaceDecl *getNextNamespace() const { return NextNamespace; }
  void setNextNamespace(NamespaceDecl *ND) { NextNamespace = ND; }

  NamespaceDecl *getOriginalNamespace() const {
    if (OrigOrAnonNamespace.getInt())
      return const_cast<NamespaceDecl *>(this);

    return OrigOrAnonNamespace.getPointer();
  }

  void setOriginalNamespace(NamespaceDecl *ND) { 
    if (ND != this) {
      OrigOrAnonNamespace.setPointer(ND);
      OrigOrAnonNamespace.setInt(false);
    }
  }

  NamespaceDecl *getAnonymousNamespace() const {
    return getOriginalNamespace()->OrigOrAnonNamespace.getPointer();
  }

  void setAnonymousNamespace(NamespaceDecl *D) {
    assert(!D || D->isAnonymousNamespace());
    assert(!D || D->getParent() == this);
    getOriginalNamespace()->OrigOrAnonNamespace.setPointer(D);
  }

  virtual NamespaceDecl *getCanonicalDecl() { return getOriginalNamespace(); }
  const NamespaceDecl *getCanonicalDecl() const { 
    return getOriginalNamespace(); 
  }

  virtual SourceRange getSourceRange() const {
    return SourceRange(getLocation(), RBracLoc);
  }

  SourceLocation getLBracLoc() const { return LBracLoc; }
  SourceLocation getRBracLoc() const { return RBracLoc; }
  void setLBracLoc(SourceLocation LBrace) { LBracLoc = LBrace; }
  void setRBracLoc(SourceLocation RBrace) { RBracLoc = RBrace; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const NamespaceDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == Namespace; }
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
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const ValueDecl *D) { return true; }
  static bool classofKind(Kind K) { return K >= ValueFirst && K <= ValueLast; }
};

/// \brief Represents a ValueDecl that came out of a declarator.
/// Contains type source information through TypeSourceInfo.
class DeclaratorDecl : public ValueDecl {
  // A struct representing both a TInfo and a syntactic qualifier,
  // to be used for the (uncommon) case of out-of-line declarations.
  struct ExtInfo {
    TypeSourceInfo *TInfo;
    NestedNameSpecifier *NNS;
    SourceRange NNSRange;
  };

  llvm::PointerUnion<TypeSourceInfo*, ExtInfo*> DeclInfo;

  bool hasExtInfo() const { return DeclInfo.is<ExtInfo*>(); }
  ExtInfo *getExtInfo() { return DeclInfo.get<ExtInfo*>(); }
  const ExtInfo *getExtInfo() const { return DeclInfo.get<ExtInfo*>(); }

protected:
  DeclaratorDecl(Kind DK, DeclContext *DC, SourceLocation L,
                 DeclarationName N, QualType T, TypeSourceInfo *TInfo)
    : ValueDecl(DK, DC, L, N, T), DeclInfo(TInfo) {}

public:
  virtual ~DeclaratorDecl();
  virtual void Destroy(ASTContext &C);

  TypeSourceInfo *getTypeSourceInfo() const {
    return hasExtInfo()
      ? DeclInfo.get<ExtInfo*>()->TInfo
      : DeclInfo.get<TypeSourceInfo*>();
  }
  void setTypeSourceInfo(TypeSourceInfo *TI) {
    if (hasExtInfo())
      DeclInfo.get<ExtInfo*>()->TInfo = TI;
    else
      DeclInfo = TI;
  }

  NestedNameSpecifier *getQualifier() const {
    return hasExtInfo() ? DeclInfo.get<ExtInfo*>()->NNS : 0;
  }
  SourceRange getQualifierRange() const {
    return hasExtInfo() ? DeclInfo.get<ExtInfo*>()->NNSRange : SourceRange();
  }
  void setQualifierInfo(NestedNameSpecifier *Qualifier,
                        SourceRange QualifierRange);

  SourceLocation getTypeSpecStartLoc() const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const DeclaratorDecl *D) { return true; }
  static bool classofKind(Kind K) {
    return K >= DeclaratorFirst && K <= DeclaratorLast;
  }
};

/// \brief Structure used to store a statement, the constant value to
/// which it was evaluated (if any), and whether or not the statement
/// is an integral constant expression (if known).
struct EvaluatedStmt {
  EvaluatedStmt() : WasEvaluated(false), IsEvaluating(false), CheckedICE(false),
                    CheckingICE(false), IsICE(false) { }

  /// \brief Whether this statement was already evaluated.
  bool WasEvaluated : 1;

  /// \brief Whether this statement is being evaluated.
  bool IsEvaluating : 1;

  /// \brief Whether we already checked whether this statement was an
  /// integral constant expression.
  bool CheckedICE : 1;

  /// \brief Whether we are checking whether this statement is an
  /// integral constant expression.
  bool CheckingICE : 1;

  /// \brief Whether this statement is an integral constant
  /// expression. Only valid if CheckedICE is true.
  bool IsICE : 1;

  Stmt *Value;
  APValue Evaluated;
};

// \brief Describes the kind of template specialization that a
// particular template specialization declaration represents.
enum TemplateSpecializationKind {
  /// This template specialization was formed from a template-id but
  /// has not yet been declared, defined, or instantiated.
  TSK_Undeclared = 0,
  /// This template specialization was implicitly instantiated from a
  /// template. (C++ [temp.inst]).
  TSK_ImplicitInstantiation,
  /// This template specialization was declared or defined by an
  /// explicit specialization (C++ [temp.expl.spec]) or partial
  /// specialization (C++ [temp.class.spec]).
  TSK_ExplicitSpecialization,
  /// This template specialization was instantiated from a template
  /// due to an explicit instantiation declaration request
  /// (C++0x [temp.explicit]).
  TSK_ExplicitInstantiationDeclaration,
  /// This template specialization was instantiated from a template
  /// due to an explicit instantiation definition request
  /// (C++ [temp.explicit]).
  TSK_ExplicitInstantiationDefinition
};
  
/// VarDecl - An instance of this class is created to represent a variable
/// declaration or definition.
class VarDecl : public DeclaratorDecl, public Redeclarable<VarDecl> {
public:
  enum StorageClass {
    None, Auto, Register, Extern, Static, PrivateExtern
  };

  /// getStorageClassSpecifierString - Return the string used to
  /// specify the storage class \arg SC.
  ///
  /// It is illegal to call this function with SC == None.
  static const char *getStorageClassSpecifierString(StorageClass SC);

protected:
  /// \brief Placeholder type used in Init to denote an unparsed C++ default
  /// argument.
  struct UnparsedDefaultArgument;

  /// \brief Placeholder type used in Init to denote an uninstantiated C++
  /// default argument.
  struct UninstantiatedDefaultArgument;

  typedef llvm::PointerUnion4<Stmt *, EvaluatedStmt *,
                              UnparsedDefaultArgument *,
                              UninstantiatedDefaultArgument *> InitType;

  /// \brief The initializer for this variable or, for a ParmVarDecl, the
  /// C++ default argument.
  mutable InitType Init;

private:
  // FIXME: This can be packed into the bitfields in Decl.
  unsigned SClass : 3;
  bool ThreadSpecified : 1;
  bool HasCXXDirectInit : 1;

  /// DeclaredInCondition - Whether this variable was declared in a
  /// condition, e.g., if (int x = foo()) { ... }.
  bool DeclaredInCondition : 1;

  friend class StmtIteratorBase;
protected:
  VarDecl(Kind DK, DeclContext *DC, SourceLocation L, IdentifierInfo *Id,
          QualType T, TypeSourceInfo *TInfo, StorageClass SC)
    : DeclaratorDecl(DK, DC, L, Id, T, TInfo), Init(),
      ThreadSpecified(false), HasCXXDirectInit(false),
      DeclaredInCondition(false) {
    SClass = SC;
  }

  typedef Redeclarable<VarDecl> redeclarable_base;
  virtual VarDecl *getNextRedeclaration() { return RedeclLink.getNext(); }

public:
  typedef redeclarable_base::redecl_iterator redecl_iterator;
  redecl_iterator redecls_begin() const {
    return redeclarable_base::redecls_begin();
  }
  redecl_iterator redecls_end() const {
    return redeclarable_base::redecls_end();
  }

  static VarDecl *Create(ASTContext &C, DeclContext *DC,
                         SourceLocation L, IdentifierInfo *Id,
                         QualType T, TypeSourceInfo *TInfo, StorageClass S);

  virtual void Destroy(ASTContext& C);
  virtual ~VarDecl();

  virtual SourceRange getSourceRange() const;

  StorageClass getStorageClass() const { return (StorageClass)SClass; }
  void setStorageClass(StorageClass SC) { SClass = SC; }

  void setThreadSpecified(bool T) { ThreadSpecified = T; }
  bool isThreadSpecified() const {
    return ThreadSpecified;
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

  /// \brief Determines whether this variable is a variable with
  /// external, C linkage.
  bool isExternC() const;

  /// isBlockVarDecl - Returns true for local variable declarations.  Note that
  /// this includes static variables inside of functions. It also includes
  /// variables inside blocks.
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

  /// isFunctionOrMethodVarDecl - Similar to isBlockVarDecl, but excludes
  /// variables declared in blocks.
  bool isFunctionOrMethodVarDecl() const {
    if (getKind() != Decl::Var)
      return false;
    if (const DeclContext *DC = getDeclContext())
      return DC->getLookupContext()->isFunctionOrMethod() &&
             DC->getLookupContext()->getDeclKind() != Decl::Block;
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
    // If it wasn't static, it would be a FieldDecl.
    return getDeclContext()->isRecord();
  }

  virtual VarDecl *getCanonicalDecl();
  const VarDecl *getCanonicalDecl() const {
    return const_cast<VarDecl*>(this)->getCanonicalDecl();
  }

  enum DefinitionKind {
    DeclarationOnly,      ///< This declaration is only a declaration.
    TentativeDefinition,  ///< This declaration is a tentative definition.
    Definition            ///< This declaration is definitely a definition.
  };

  /// \brief Check whether this declaration is a definition. If this could be
  /// a tentative definition (in C), don't check whether there's an overriding
  /// definition.
  DefinitionKind isThisDeclarationADefinition() const;

  /// \brief Get the tentative definition that acts as the real definition in
  /// a TU. Returns null if there is a proper definition available.
  VarDecl *getActingDefinition();
  const VarDecl *getActingDefinition() const {
    return const_cast<VarDecl*>(this)->getActingDefinition();
  }

  /// \brief Determine whether this is a tentative definition of a
  /// variable in C.
  bool isTentativeDefinitionNow() const;

  /// \brief Get the real (not just tentative) definition for this declaration.
  VarDecl *getDefinition();
  const VarDecl *getDefinition() const {
    return const_cast<VarDecl*>(this)->getDefinition();
  }

  /// \brief Determine whether this is or was instantiated from an out-of-line 
  /// definition of a static data member.
  virtual bool isOutOfLine() const;

  /// \brief If this is a static data member, find its out-of-line definition.
  VarDecl *getOutOfLineDefinition();
  
  /// isFileVarDecl - Returns true for file scoped variable declaration.
  bool isFileVarDecl() const {
    if (getKind() != Decl::Var)
      return false;
    if (const DeclContext *Ctx = getDeclContext()) {
      Ctx = Ctx->getLookupContext();
      if (isa<TranslationUnitDecl>(Ctx) || isa<NamespaceDecl>(Ctx) )
        return true;
    }
    if (isStaticDataMember())
      return true;

    return false;
  }

  /// getAnyInitializer - Get the initializer for this variable, no matter which
  /// declaration it is attached to.
  const Expr *getAnyInitializer() const {
    const VarDecl *D;
    return getAnyInitializer(D);
  }

  /// getAnyInitializer - Get the initializer for this variable, no matter which
  /// declaration it is attached to. Also get that declaration.
  const Expr *getAnyInitializer(const VarDecl *&D) const;

  bool hasInit() const {
    return !Init.isNull();
  }
  const Expr *getInit() const {
    if (Init.isNull())
      return 0;

    const Stmt *S = Init.dyn_cast<Stmt *>();
    if (!S) {
      if (EvaluatedStmt *ES = Init.dyn_cast<EvaluatedStmt*>())
        S = ES->Value;
    }
    return (const Expr*) S;
  }
  Expr *getInit() {
    if (Init.isNull())
      return 0;

    Stmt *S = Init.dyn_cast<Stmt *>();
    if (!S) {
      if (EvaluatedStmt *ES = Init.dyn_cast<EvaluatedStmt*>())
        S = ES->Value;
    }

    return (Expr*) S;
  }

  /// \brief Retrieve the address of the initializer expression.
  Stmt **getInitAddress() {
    if (EvaluatedStmt *ES = Init.dyn_cast<EvaluatedStmt*>())
      return &ES->Value;

    // This union hack tip-toes around strict-aliasing rules.
    union {
      InitType *InitPtr;
      Stmt **StmtPtr;
    };

    InitPtr = &Init;
    return StmtPtr;
  }

  void setInit(Expr *I);

  EvaluatedStmt *EnsureEvaluatedStmt() const {
    EvaluatedStmt *Eval = Init.dyn_cast<EvaluatedStmt *>();
    if (!Eval) {
      Stmt *S = Init.get<Stmt *>();
      Eval = new (getASTContext()) EvaluatedStmt;
      Eval->Value = S;
      Init = Eval;
    }
    return Eval;
  }

  /// \brief Check whether we are in the process of checking whether the
  /// initializer can be evaluated.
  bool isEvaluatingValue() const {
    if (EvaluatedStmt *Eval = Init.dyn_cast<EvaluatedStmt *>())
      return Eval->IsEvaluating;

    return false;
  }

  /// \brief Note that we now are checking whether the initializer can be
  /// evaluated.
  void setEvaluatingValue() const {
    EvaluatedStmt *Eval = EnsureEvaluatedStmt();
    Eval->IsEvaluating = true;
  }

  /// \brief Note that constant evaluation has computed the given
  /// value for this variable's initializer.
  void setEvaluatedValue(const APValue &Value) const {
    EvaluatedStmt *Eval = EnsureEvaluatedStmt();
    Eval->IsEvaluating = false;
    Eval->WasEvaluated = true;
    Eval->Evaluated = Value;
  }

  /// \brief Return the already-evaluated value of this variable's
  /// initializer, or NULL if the value is not yet known. Returns pointer
  /// to untyped APValue if the value could not be evaluated.
  APValue *getEvaluatedValue() const {
    if (EvaluatedStmt *Eval = Init.dyn_cast<EvaluatedStmt *>())
      if (Eval->WasEvaluated)
        return &Eval->Evaluated;

    return 0;
  }

  /// \brief Determines whether it is already known whether the
  /// initializer is an integral constant expression or not.
  bool isInitKnownICE() const {
    if (EvaluatedStmt *Eval = Init.dyn_cast<EvaluatedStmt *>())
      return Eval->CheckedICE;

    return false;
  }

  /// \brief Determines whether the initializer is an integral
  /// constant expression.
  ///
  /// \pre isInitKnownICE()
  bool isInitICE() const {
    assert(isInitKnownICE() &&
           "Check whether we already know that the initializer is an ICE");
    return Init.get<EvaluatedStmt *>()->IsICE;
  }

  /// \brief Check whether we are in the process of checking the initializer
  /// is an integral constant expression.
  bool isCheckingICE() const {
    if (EvaluatedStmt *Eval = Init.dyn_cast<EvaluatedStmt *>())
      return Eval->CheckingICE;

    return false;
  }

  /// \brief Note that we now are checking whether the initializer is an
  /// integral constant expression.
  void setCheckingICE() const {
    EvaluatedStmt *Eval = EnsureEvaluatedStmt();
    Eval->CheckingICE = true;
  }

  /// \brief Note that we now know whether the initializer is an
  /// integral constant expression.
  void setInitKnownICE(bool IsICE) const {
    EvaluatedStmt *Eval = EnsureEvaluatedStmt();
    Eval->CheckingICE = false;
    Eval->CheckedICE = true;
    Eval->IsICE = IsICE;
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
  
  /// \brief If this variable is an instantiated static data member of a
  /// class template specialization, returns the templated static data member
  /// from which it was instantiated.
  VarDecl *getInstantiatedFromStaticDataMember() const;

  /// \brief If this variable is a static data member, determine what kind of 
  /// template specialization or instantiation this is.
  TemplateSpecializationKind getTemplateSpecializationKind() const;
  
  /// \brief If this variable is an instantiation of a static data member of a
  /// class template specialization, retrieves the member specialization
  /// information.
  MemberSpecializationInfo *getMemberSpecializationInfo() const;
  
  /// \brief For a static data member that was instantiated from a static
  /// data member of a class template, set the template specialiation kind.
  void setTemplateSpecializationKind(TemplateSpecializationKind TSK,
                        SourceLocation PointOfInstantiation = SourceLocation());

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const VarDecl *D) { return true; }
  static bool classofKind(Kind K) { return K >= VarFirst && K <= VarLast; }
};

class ImplicitParamDecl : public VarDecl {
protected:
  ImplicitParamDecl(Kind DK, DeclContext *DC, SourceLocation L,
                    IdentifierInfo *Id, QualType Tw)
    : VarDecl(DK, DC, L, Id, Tw, /*TInfo=*/0, VarDecl::None) {}
public:
  static ImplicitParamDecl *Create(ASTContext &C, DeclContext *DC,
                                   SourceLocation L, IdentifierInfo *Id,
                                   QualType T);
  // Implement isa/cast/dyncast/etc.
  static bool classof(const ImplicitParamDecl *D) { return true; }
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == ImplicitParam; }
};

/// ParmVarDecl - Represent a parameter to a function.
class ParmVarDecl : public VarDecl {
  // NOTE: VC++ treats enums as signed, avoid using the ObjCDeclQualifier enum
  /// FIXME: Also can be paced into the bitfields in Decl.
  /// in, inout, etc.
  unsigned objcDeclQualifier : 6;
  bool HasInheritedDefaultArg : 1;

protected:
  ParmVarDecl(Kind DK, DeclContext *DC, SourceLocation L,
              IdentifierInfo *Id, QualType T, TypeSourceInfo *TInfo,
              StorageClass S, Expr *DefArg)
  : VarDecl(DK, DC, L, Id, T, TInfo, S),
    objcDeclQualifier(OBJC_TQ_None), HasInheritedDefaultArg(false) {
    setDefaultArg(DefArg);
  }

public:
  static ParmVarDecl *Create(ASTContext &C, DeclContext *DC,
                             SourceLocation L,IdentifierInfo *Id,
                             QualType T, TypeSourceInfo *TInfo,
                             StorageClass S, Expr *DefArg);

  ObjCDeclQualifier getObjCDeclQualifier() const {
    return ObjCDeclQualifier(objcDeclQualifier);
  }
  void setObjCDeclQualifier(ObjCDeclQualifier QTVal) {
    objcDeclQualifier = QTVal;
  }

  Expr *getDefaultArg();
  const Expr *getDefaultArg() const {
    return const_cast<ParmVarDecl *>(this)->getDefaultArg();
  }
  
  void setDefaultArg(Expr *defarg) {
    Init = reinterpret_cast<Stmt *>(defarg);
  }

  unsigned getNumDefaultArgTemporaries() const;
  CXXTemporary *getDefaultArgTemporary(unsigned i);
  const CXXTemporary *getDefaultArgTemporary(unsigned i) const {
    return const_cast<ParmVarDecl *>(this)->getDefaultArgTemporary(i);
  }
  
  /// \brief Retrieve the source range that covers the entire default
  /// argument.
  SourceRange getDefaultArgRange() const;  
  void setUninstantiatedDefaultArg(Expr *arg) {
    Init = reinterpret_cast<UninstantiatedDefaultArgument *>(arg);
  }
  Expr *getUninstantiatedDefaultArg() {
    return (Expr *)Init.get<UninstantiatedDefaultArgument *>();
  }
  const Expr *getUninstantiatedDefaultArg() const {
    return (const Expr *)Init.get<UninstantiatedDefaultArgument *>();
  }

  /// hasDefaultArg - Determines whether this parameter has a default argument,
  /// either parsed or not.
  bool hasDefaultArg() const {
    return getInit() || hasUnparsedDefaultArg() ||
      hasUninstantiatedDefaultArg();
  }

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
    return Init.is<UnparsedDefaultArgument*>();
  }

  bool hasUninstantiatedDefaultArg() const {
    return Init.is<UninstantiatedDefaultArgument*>();
  }

  /// setUnparsedDefaultArg - Specify that this parameter has an
  /// unparsed default argument. The argument will be replaced with a
  /// real default argument via setDefaultArg when the class
  /// definition enclosing the function declaration that owns this
  /// default argument is completed.
  void setUnparsedDefaultArg() {
    Init = (UnparsedDefaultArgument *)0;
  }

  bool hasInheritedDefaultArg() const {
    return HasInheritedDefaultArg;
  }

  void setHasInheritedDefaultArg(bool I = true) {
    HasInheritedDefaultArg = I;
  }

  QualType getOriginalType() const {
    if (getTypeSourceInfo())
      return getTypeSourceInfo()->getType();
    return getType();
  }

  /// setOwningFunction - Sets the function declaration that owns this
  /// ParmVarDecl. Since ParmVarDecls are often created before the
  /// FunctionDecls that own them, this routine is required to update
  /// the DeclContext appropriately.
  void setOwningFunction(DeclContext *FD) { setDeclContext(FD); }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const ParmVarDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == ParmVar; }
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
class FunctionDecl : public DeclaratorDecl, public DeclContext,
                     public Redeclarable<FunctionDecl> {
public:
  enum StorageClass {
    None, Extern, Static, PrivateExtern
  };

private:
  /// ParamInfo - new[]'d array of pointers to VarDecls for the formal
  /// parameters of this function.  This is null if a prototype or if there are
  /// no formals.
  ParmVarDecl **ParamInfo;

  LazyDeclStmtPtr Body;

  // FIXME: This can be packed into the bitfields in Decl.
  // NOTE: VC++ treats enums as signed, avoid using the StorageClass enum
  unsigned SClass : 2;
  bool IsInline : 1;
  bool IsVirtualAsWritten : 1;
  bool IsPure : 1;
  bool HasInheritedPrototype : 1;
  bool HasWrittenPrototype : 1;
  bool IsDeleted : 1;
  bool IsTrivial : 1; // sunk from CXXMethodDecl
  bool IsCopyAssignment : 1;  // sunk from CXXMethodDecl
  bool HasImplicitReturnZero : 1;

  /// \brief End part of this FunctionDecl's source range.
  ///
  /// We could compute the full range in getSourceRange(). However, when we're
  /// dealing with a function definition deserialized from a PCH/AST file,
  /// we can only compute the full range once the function body has been
  /// de-serialized, so it's far better to have the (sometimes-redundant)
  /// EndRangeLoc.
  SourceLocation EndRangeLoc;

  /// \brief The template or declaration that this declaration
  /// describes or was instantiated from, respectively.
  ///
  /// For non-templates, this value will be NULL. For function
  /// declarations that describe a function template, this will be a
  /// pointer to a FunctionTemplateDecl. For member functions
  /// of class template specializations, this will be a MemberSpecializationInfo
  /// pointer containing information about the specialization.
  /// For function template specializations, this will be a
  /// FunctionTemplateSpecializationInfo, which contains information about
  /// the template being specialized and the template arguments involved in
  /// that specialization.
  llvm::PointerUnion3<FunctionTemplateDecl *, 
                      MemberSpecializationInfo *,
                      FunctionTemplateSpecializationInfo *>
    TemplateOrSpecialization;

protected:
  FunctionDecl(Kind DK, DeclContext *DC, SourceLocation L,
               DeclarationName N, QualType T, TypeSourceInfo *TInfo,
               StorageClass S, bool isInline)
    : DeclaratorDecl(DK, DC, L, N, T, TInfo),
      DeclContext(DK),
      ParamInfo(0), Body(),
      SClass(S), IsInline(isInline), 
      IsVirtualAsWritten(false), IsPure(false), HasInheritedPrototype(false),
      HasWrittenPrototype(true), IsDeleted(false), IsTrivial(false),
      IsCopyAssignment(false),
      HasImplicitReturnZero(false),
      EndRangeLoc(L), TemplateOrSpecialization() {}

  virtual ~FunctionDecl() {}
  virtual void Destroy(ASTContext& C);

  typedef Redeclarable<FunctionDecl> redeclarable_base;
  virtual FunctionDecl *getNextRedeclaration() { return RedeclLink.getNext(); }

public:
  typedef redeclarable_base::redecl_iterator redecl_iterator;
  redecl_iterator redecls_begin() const {
    return redeclarable_base::redecls_begin();
  }
  redecl_iterator redecls_end() const {
    return redeclarable_base::redecls_end();
  }

  static FunctionDecl *Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                              DeclarationName N, QualType T,
                              TypeSourceInfo *TInfo,
                              StorageClass S = None, bool isInline = false,
                              bool hasWrittenPrototype = true);

  virtual void getNameForDiagnostic(std::string &S,
                                    const PrintingPolicy &Policy,
                                    bool Qualified) const;

  virtual SourceRange getSourceRange() const {
    return SourceRange(getLocation(), EndRangeLoc);
  }
  void setLocEnd(SourceLocation E) {
    EndRangeLoc = E;
  }

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
  /// FIXME: Should return true if function is deleted or defaulted. However,
  /// CodeGenModule.cpp uses it, and I don't know if this would break it.
  bool isThisDeclarationADefinition() const { return Body; }

  void setBody(Stmt *B);
  void setLazyBody(uint64_t Offset) { Body = Offset; }

  /// Whether this function is marked as virtual explicitly.
  bool isVirtualAsWritten() const { return IsVirtualAsWritten; }
  void setVirtualAsWritten(bool V) { IsVirtualAsWritten = V; }

  /// Whether this virtual function is pure, i.e. makes the containing class
  /// abstract.
  bool isPure() const { return IsPure; }
  void setPure(bool P = true) { IsPure = P; }

  /// Whether this function is "trivial" in some specialized C++ senses.
  /// Can only be true for default constructors, copy constructors,
  /// copy assignment operators, and destructors.  Not meaningful until
  /// the class has been fully built by Sema.
  bool isTrivial() const { return IsTrivial; }
  void setTrivial(bool IT) { IsTrivial = IT; }

  bool isCopyAssignment() const { return IsCopyAssignment; }
  void setCopyAssignment(bool CA) { IsCopyAssignment = CA; }

  /// Whether falling off this function implicitly returns null/zero.
  /// If a more specific implicit return value is required, front-ends
  /// should synthesize the appropriate return statements.
  bool hasImplicitReturnZero() const { return HasImplicitReturnZero; }
  void setHasImplicitReturnZero(bool IRZ) { HasImplicitReturnZero = IRZ; }

  /// \brief Whether this function has a prototype, either because one
  /// was explicitly written or because it was "inherited" by merging
  /// a declaration without a prototype with a declaration that has a
  /// prototype.
  bool hasPrototype() const {
    return HasWrittenPrototype || HasInheritedPrototype;
  }

  bool hasWrittenPrototype() const { return HasWrittenPrototype; }
  void setHasWrittenPrototype(bool P) { HasWrittenPrototype = P; }

  /// \brief Whether this function inherited its prototype from a
  /// previous declaration.
  bool hasInheritedPrototype() const { return HasInheritedPrototype; }
  void setHasInheritedPrototype(bool P = true) { HasInheritedPrototype = P; }

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
  bool isExternC() const;

  /// \brief Determines whether this is a global function.
  bool isGlobal() const;

  void setPreviousDeclaration(FunctionDecl * PrevDecl);

  virtual const FunctionDecl *getCanonicalDecl() const;
  virtual FunctionDecl *getCanonicalDecl();

  unsigned getBuiltinID() const;

  // Iterator access to formal parameters.
  unsigned param_size() const { return getNumParams(); }
  typedef ParmVarDecl **param_iterator;
  typedef ParmVarDecl * const *param_const_iterator;

  param_iterator param_begin() { return ParamInfo; }
  param_iterator param_end()   { return ParamInfo+param_size(); }

  param_const_iterator param_begin() const { return ParamInfo; }
  param_const_iterator param_end() const   { return ParamInfo+param_size(); }

  /// getNumParams - Return the number of parameters this function must have
  /// based on its FunctionType.  This is the length of the ParamInfo array
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
  void setParams(ParmVarDecl **NewParamInfo, unsigned NumParams);

  /// getMinRequiredArguments - Returns the minimum number of arguments
  /// needed to call this function. This may be fewer than the number of
  /// function parameters, if some of the parameters have default
  /// arguments (in C++).
  unsigned getMinRequiredArguments() const;

  QualType getResultType() const {
    return getType()->getAs<FunctionType>()->getResultType();
  }
  StorageClass getStorageClass() const { return StorageClass(SClass); }
  void setStorageClass(StorageClass SC) { SClass = SC; }

  /// \brief Determine whether the "inline" keyword was specified for this
  /// function.
  bool isInlineSpecified() const { return IsInline; }
                       
  /// Set whether the "inline" keyword was specified for this function.
  void setInlineSpecified(bool I) { IsInline = I; }

  /// \brief Determine whether this function should be inlined, because it is
  /// either marked "inline" or is a member function of a C++ class that
  /// was defined in the class body.
  bool isInlined() const;
                       
  bool isInlineDefinitionExternallyVisible() const;
                       
  /// isOverloadedOperator - Whether this function declaration
  /// represents an C++ overloaded operator, e.g., "operator+".
  bool isOverloadedOperator() const {
    return getOverloadedOperator() != OO_None;
  }

  OverloadedOperatorKind getOverloadedOperator() const;

  const IdentifierInfo *getLiteralIdentifier() const;

  /// \brief If this function is an instantiation of a member function
  /// of a class template specialization, retrieves the function from
  /// which it was instantiated.
  ///
  /// This routine will return non-NULL for (non-templated) member
  /// functions of class templates and for instantiations of function
  /// templates. For example, given:
  ///
  /// \code
  /// template<typename T>
  /// struct X {
  ///   void f(T);
  /// };
  /// \endcode
  ///
  /// The declaration for X<int>::f is a (non-templated) FunctionDecl
  /// whose parent is the class template specialization X<int>. For
  /// this declaration, getInstantiatedFromFunction() will return
  /// the FunctionDecl X<T>::A. When a complete definition of
  /// X<int>::A is required, it will be instantiated from the
  /// declaration returned by getInstantiatedFromMemberFunction().
  FunctionDecl *getInstantiatedFromMemberFunction() const;

  /// \brief If this function is an instantiation of a member function of a
  /// class template specialization, retrieves the member specialization
  /// information.
  MemberSpecializationInfo *getMemberSpecializationInfo() const;
                       
  /// \brief Specify that this record is an instantiation of the
  /// member function FD.
  void setInstantiationOfMemberFunction(FunctionDecl *FD,
                                        TemplateSpecializationKind TSK);

  /// \brief Retrieves the function template that is described by this
  /// function declaration.
  ///
  /// Every function template is represented as a FunctionTemplateDecl
  /// and a FunctionDecl (or something derived from FunctionDecl). The
  /// former contains template properties (such as the template
  /// parameter lists) while the latter contains the actual
  /// description of the template's
  /// contents. FunctionTemplateDecl::getTemplatedDecl() retrieves the
  /// FunctionDecl that describes the function template,
  /// getDescribedFunctionTemplate() retrieves the
  /// FunctionTemplateDecl from a FunctionDecl.
  FunctionTemplateDecl *getDescribedFunctionTemplate() const {
    return TemplateOrSpecialization.dyn_cast<FunctionTemplateDecl*>();
  }

  void setDescribedFunctionTemplate(FunctionTemplateDecl *Template) {
    TemplateOrSpecialization = Template;
  }

  /// \brief Determine whether this function is a function template 
  /// specialization.
  bool isFunctionTemplateSpecialization() const {
    return getPrimaryTemplate() != 0;
  }
       
  /// \brief If this function is actually a function template specialization,
  /// retrieve information about this function template specialization. 
  /// Otherwise, returns NULL.
  FunctionTemplateSpecializationInfo *getTemplateSpecializationInfo() const {
    return TemplateOrSpecialization.
             dyn_cast<FunctionTemplateSpecializationInfo*>();
  }

  /// \brief Determines whether this function is a function template
  /// specialization or a member of a class template specialization that can
  /// be implicitly instantiated.
  bool isImplicitlyInstantiable() const;
              
  /// \brief Retrieve the function declaration from which this function could
  /// be instantiated, if it is an instantiation (rather than a non-template
  /// or a specialization, for example).
  FunctionDecl *getTemplateInstantiationPattern() const;

  /// \brief Retrieve the primary template that this function template
  /// specialization either specializes or was instantiated from.
  ///
  /// If this function declaration is not a function template specialization,
  /// returns NULL.
  FunctionTemplateDecl *getPrimaryTemplate() const;

  /// \brief Retrieve the template arguments used to produce this function
  /// template specialization from the primary template.
  ///
  /// If this function declaration is not a function template specialization,
  /// returns NULL.
  const TemplateArgumentList *getTemplateSpecializationArgs() const;

  /// \brief Specify that this function declaration is actually a function
  /// template specialization.
  ///
  /// \param Context the AST context in which this function resides.
  ///
  /// \param Template the function template that this function template
  /// specialization specializes.
  ///
  /// \param TemplateArgs the template arguments that produced this
  /// function template specialization from the template.
  ///
  /// \param InsertPos If non-NULL, the position in the function template
  /// specialization set where the function template specialization data will
  /// be inserted.
  ///
  /// \param TSK the kind of template specialization this is.
  void setFunctionTemplateSpecialization(FunctionTemplateDecl *Template,
                                      const TemplateArgumentList *TemplateArgs,
                                         void *InsertPos,
                    TemplateSpecializationKind TSK = TSK_ImplicitInstantiation);

  /// \brief Determine what kind of template instantiation this function
  /// represents.
  TemplateSpecializationKind getTemplateSpecializationKind() const;

  /// \brief Determine what kind of template instantiation this function
  /// represents.
  void setTemplateSpecializationKind(TemplateSpecializationKind TSK,
                        SourceLocation PointOfInstantiation = SourceLocation());

  /// \brief Retrieve the (first) point of instantiation of a function template
  /// specialization or a member of a class template specialization.
  ///
  /// \returns the first point of instantiation, if this function was 
  /// instantiated from a template; otherwie, returns an invalid source 
  /// location.
  SourceLocation getPointOfInstantiation() const;
                       
  /// \brief Determine whether this is or was instantiated from an out-of-line 
  /// definition of a member function.
  virtual bool isOutOfLine() const;
                       
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const FunctionDecl *D) { return true; }
  static bool classofKind(Kind K) {
    return K >= FunctionFirst && K <= FunctionLast;
  }
  static DeclContext *castToDeclContext(const FunctionDecl *D) {
    return static_cast<DeclContext *>(const_cast<FunctionDecl*>(D));
  }
  static FunctionDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<FunctionDecl *>(const_cast<DeclContext*>(DC));
  }
};


/// FieldDecl - An instance of this class is created by Sema::ActOnField to
/// represent a member of a struct/union/class.
class FieldDecl : public DeclaratorDecl {
  // FIXME: This can be packed into the bitfields in Decl.
  bool Mutable : 1;
  Expr *BitWidth;
protected:
  FieldDecl(Kind DK, DeclContext *DC, SourceLocation L,
            IdentifierInfo *Id, QualType T, TypeSourceInfo *TInfo,
            Expr *BW, bool Mutable)
    : DeclaratorDecl(DK, DC, L, Id, T, TInfo), Mutable(Mutable), BitWidth(BW) {
  }

public:
  static FieldDecl *Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                           IdentifierInfo *Id, QualType T,
                           TypeSourceInfo *TInfo, Expr *BW, bool Mutable);

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

  /// getParent - Returns the parent of this field declaration, which
  /// is the struct in which this method is defined.
  const RecordDecl *getParent() const {
    return cast<RecordDecl>(getDeclContext());
  }

  RecordDecl *getParent() {
    return cast<RecordDecl>(getDeclContext());
  }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const FieldDecl *D) { return true; }
  static bool classofKind(Kind K) { return K >= FieldFirst && K <= FieldLast; }
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
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const EnumConstantDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == EnumConstant; }

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
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const TypeDecl *D) { return true; }
  static bool classofKind(Kind K) { return K >= TypeFirst && K <= TypeLast; }
};


class TypedefDecl : public TypeDecl, public Redeclarable<TypedefDecl> {
  /// UnderlyingType - This is the type the typedef is set to.
  TypeSourceInfo *TInfo;

  TypedefDecl(DeclContext *DC, SourceLocation L,
              IdentifierInfo *Id, TypeSourceInfo *TInfo)
    : TypeDecl(Typedef, DC, L, Id), TInfo(TInfo) {}

  virtual ~TypedefDecl();
public:

  static TypedefDecl *Create(ASTContext &C, DeclContext *DC,
                             SourceLocation L, IdentifierInfo *Id,
                             TypeSourceInfo *TInfo);

  TypeSourceInfo *getTypeSourceInfo() const {
    return TInfo;
  }

  /// Retrieves the canonical declaration of this typedef.
  TypedefDecl *getCanonicalDecl() {
    return getFirstDeclaration();
  }
  const TypedefDecl *getCanonicalDecl() const {
    return getFirstDeclaration();
  }

  QualType getUnderlyingType() const {
    return TInfo->getType();
  }
  void setTypeSourceInfo(TypeSourceInfo *newType) {
    TInfo = newType;
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const TypedefDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == Typedef; }
};

class TypedefDecl;

/// TagDecl - Represents the declaration of a struct/union/class/enum.
class TagDecl
  : public TypeDecl, public DeclContext, public Redeclarable<TagDecl> {
public:
  // This is really ugly.
  typedef ElaboratedType::TagKind TagKind;
  static const TagKind TK_struct = ElaboratedType::TK_struct;
  static const TagKind TK_union = ElaboratedType::TK_union;
  static const TagKind TK_class = ElaboratedType::TK_class;
  static const TagKind TK_enum = ElaboratedType::TK_enum;

private:
  // FIXME: This can be packed into the bitfields in Decl.
  /// TagDeclKind - The TagKind enum.
  unsigned TagDeclKind : 2;

  /// IsDefinition - True if this is a definition ("struct foo {};"), false if
  /// it is a declaration ("struct foo;").
  bool IsDefinition : 1;

  /// IsEmbeddedInDeclarator - True if this tag declaration is
  /// "embedded" (i.e., defined or declared for the very first time)
  /// in the syntax of a declarator.
  bool IsEmbeddedInDeclarator : 1;

  SourceLocation TagKeywordLoc;
  SourceLocation RBraceLoc;

  // A struct representing syntactic qualifier info,
  // to be used for the (uncommon) case of out-of-line declarations.
  struct ExtInfo {
    NestedNameSpecifier *NNS;
    SourceRange NNSRange;
  };

  /// TypedefDeclOrQualifier - If the (out-of-line) tag declaration name
  /// is qualified, it points to the qualifier info (nns and range);
  /// otherwise, if the tag declaration is anonymous and it is part of
  /// a typedef, it points to the TypedefDecl (used for mangling);
  /// otherwise, it is a null (TypedefDecl) pointer.
  llvm::PointerUnion<TypedefDecl*, ExtInfo*> TypedefDeclOrQualifier;

  bool hasExtInfo() const { return TypedefDeclOrQualifier.is<ExtInfo*>(); }
  ExtInfo *getExtInfo() { return TypedefDeclOrQualifier.get<ExtInfo*>(); }
  const ExtInfo *getExtInfo() const {
    return TypedefDeclOrQualifier.get<ExtInfo*>();
  }

protected:
  TagDecl(Kind DK, TagKind TK, DeclContext *DC,
          SourceLocation L, IdentifierInfo *Id,
          TagDecl *PrevDecl, SourceLocation TKL = SourceLocation())
    : TypeDecl(DK, DC, L, Id), DeclContext(DK), TagKeywordLoc(TKL),
      TypedefDeclOrQualifier((TypedefDecl*) 0) {
    assert((DK != Enum || TK == TK_enum) &&"EnumDecl not matched with TK_enum");
    TagDeclKind = TK;
    IsDefinition = false;
    IsEmbeddedInDeclarator = false;
    setPreviousDeclaration(PrevDecl);
  }

  typedef Redeclarable<TagDecl> redeclarable_base;
  virtual TagDecl *getNextRedeclaration() { return RedeclLink.getNext(); }

public:
  void Destroy(ASTContext &C);

  typedef redeclarable_base::redecl_iterator redecl_iterator;
  redecl_iterator redecls_begin() const {
    return redeclarable_base::redecls_begin();
  }
  redecl_iterator redecls_end() const {
    return redeclarable_base::redecls_end();
  }

  SourceLocation getRBraceLoc() const { return RBraceLoc; }
  void setRBraceLoc(SourceLocation L) { RBraceLoc = L; }

  SourceLocation getTagKeywordLoc() const { return TagKeywordLoc; }
  void setTagKeywordLoc(SourceLocation TKL) { TagKeywordLoc = TKL; }

  virtual SourceRange getSourceRange() const;

  virtual TagDecl* getCanonicalDecl();
  const TagDecl* getCanonicalDecl() const {
    return const_cast<TagDecl*>(this)->getCanonicalDecl();
  }

  /// isDefinition - Return true if this decl has its body specified.
  bool isDefinition() const {
    return IsDefinition;
  }

  bool isEmbeddedInDeclarator() const {
    return IsEmbeddedInDeclarator;
  }
  void setEmbeddedInDeclarator(bool isInDeclarator) {
    IsEmbeddedInDeclarator = isInDeclarator;
  }

  /// \brief Whether this declaration declares a type that is
  /// dependent, i.e., a type that somehow depends on template
  /// parameters.
  bool isDependentType() const { return isDependentContext(); }

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
  TagDecl* getDefinition() const;

  void setDefinition(bool V) { IsDefinition = V; }

  const char *getKindName() const {
    return ElaboratedType::getNameForTagKind(getTagKind());
  }

  /// getTagKindForTypeSpec - Converts a type specifier (DeclSpec::TST)
  /// into a tag kind.  It is an error to provide a type specifier
  /// which *isn't* a tag kind here.
  static TagKind getTagKindForTypeSpec(unsigned TypeSpec);

  TagKind getTagKind() const {
    return TagKind(TagDeclKind);
  }

  void setTagKind(TagKind TK) { TagDeclKind = TK; }

  bool isStruct() const { return getTagKind() == TK_struct; }
  bool isClass()  const { return getTagKind() == TK_class; }
  bool isUnion()  const { return getTagKind() == TK_union; }
  bool isEnum()   const { return getTagKind() == TK_enum; }

  TypedefDecl *getTypedefForAnonDecl() const {
    return hasExtInfo() ? 0 : TypedefDeclOrQualifier.get<TypedefDecl*>();
  }
  void setTypedefForAnonDecl(TypedefDecl *TDD) { TypedefDeclOrQualifier = TDD; }

  NestedNameSpecifier *getQualifier() const {
    return hasExtInfo() ? TypedefDeclOrQualifier.get<ExtInfo*>()->NNS : 0;
  }
  SourceRange getQualifierRange() const {
    return hasExtInfo()
      ? TypedefDeclOrQualifier.get<ExtInfo*>()->NNSRange
      : SourceRange();
  }
  void setQualifierInfo(NestedNameSpecifier *Qualifier,
                        SourceRange QualifierRange);

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const TagDecl *D) { return true; }
  static bool classofKind(Kind K) { return K >= TagFirst && K <= TagLast; }

  static DeclContext *castToDeclContext(const TagDecl *D) {
    return static_cast<DeclContext *>(const_cast<TagDecl*>(D));
  }
  static TagDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<TagDecl *>(const_cast<DeclContext*>(DC));
  }
};

/// EnumDecl - Represents an enum.  As an extension, we allow forward-declared
/// enums.
class EnumDecl : public TagDecl {
  /// IntegerType - This represent the integer type that the enum corresponds
  /// to for code generation purposes.  Note that the enumerator constants may
  /// have a different type than this does.
  QualType IntegerType;

  /// PromotionType - The integer type that values of this type should
  /// promote to.  In C, enumerators are generally of an integer type
  /// directly, but gcc-style large enumerators (and all enumerators
  /// in C++) are of the enum type instead.
  QualType PromotionType;

  /// \brief If the enumeration was instantiated from an enumeration
  /// within a class or function template, this pointer refers to the
  /// enumeration declared within the template.
  EnumDecl *InstantiatedFrom;

  EnumDecl(DeclContext *DC, SourceLocation L,
           IdentifierInfo *Id, EnumDecl *PrevDecl, SourceLocation TKL)
    : TagDecl(Enum, TK_enum, DC, L, Id, PrevDecl, TKL), InstantiatedFrom(0) {
      IntegerType = QualType();
    }
public:
  EnumDecl *getCanonicalDecl() {
    return cast<EnumDecl>(TagDecl::getCanonicalDecl());
  }
  const EnumDecl *getCanonicalDecl() const {
    return cast<EnumDecl>(TagDecl::getCanonicalDecl());
  }

  static EnumDecl *Create(ASTContext &C, DeclContext *DC,
                          SourceLocation L, IdentifierInfo *Id,
                          SourceLocation TKL, EnumDecl *PrevDecl);

  virtual void Destroy(ASTContext& C);

  /// completeDefinition - When created, the EnumDecl corresponds to a
  /// forward-declared enum. This method is used to mark the
  /// declaration as being defined; it's enumerators have already been
  /// added (via DeclContext::addDecl). NewType is the new underlying
  /// type of the enumeration type.
  void completeDefinition(QualType NewType,
                          QualType PromotionType);

  // enumerator_iterator - Iterates through the enumerators of this
  // enumeration.
  typedef specific_decl_iterator<EnumConstantDecl> enumerator_iterator;

  enumerator_iterator enumerator_begin() const {
    return enumerator_iterator(this->decls_begin());
  }

  enumerator_iterator enumerator_end() const {
    return enumerator_iterator(this->decls_end());
  }

  /// getPromotionType - Return the integer type that enumerators
  /// should promote to.
  QualType getPromotionType() const { return PromotionType; }

  /// \brief Set the promotion type.
  void setPromotionType(QualType T) { PromotionType = T; }

  /// getIntegerType - Return the integer type this enum decl corresponds to.
  /// This returns a null qualtype for an enum forward definition.
  QualType getIntegerType() const { return IntegerType; }

  /// \brief Set the underlying integer type.
  void setIntegerType(QualType T) { IntegerType = T; }

  /// \brief Returns the enumeration (declared within the template)
  /// from which this enumeration type was instantiated, or NULL if
  /// this enumeration was not instantiated from any template.
  EnumDecl *getInstantiatedFromMemberEnum() const {
    return InstantiatedFrom;
  }

  void setInstantiationOfMemberEnum(EnumDecl *IF) { InstantiatedFrom = IF; }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const EnumDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == Enum; }
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

  /// AnonymousStructOrUnion - Whether this is the type of an anonymous struct
  /// or union.
  bool AnonymousStructOrUnion : 1;

  /// HasObjectMember - This is true if this struct has at least one member
  /// containing an object.
  bool HasObjectMember : 1;

protected:
  RecordDecl(Kind DK, TagKind TK, DeclContext *DC,
             SourceLocation L, IdentifierInfo *Id,
             RecordDecl *PrevDecl, SourceLocation TKL);
  virtual ~RecordDecl();

public:
  static RecordDecl *Create(ASTContext &C, TagKind TK, DeclContext *DC,
                            SourceLocation L, IdentifierInfo *Id,
                            SourceLocation TKL = SourceLocation(),
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

  bool hasObjectMember() const { return HasObjectMember; }
  void setHasObjectMember (bool val) { HasObjectMember = val; }

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
  RecordDecl* getDefinition() const {
    return cast_or_null<RecordDecl>(TagDecl::getDefinition());
  }

  // Iterator access to field members. The field iterator only visits
  // the non-static data members of this class, ignoring any static
  // data members, functions, constructors, destructors, etc.
  typedef specific_decl_iterator<FieldDecl> field_iterator;

  field_iterator field_begin() const {
    return field_iterator(decls_begin());
  }
  field_iterator field_end() const {
    return field_iterator(decls_end());
  }

  // field_empty - Whether there are any fields (non-static data
  // members) in this record.
  bool field_empty() const {
    return field_begin() == field_end();
  }

  /// completeDefinition - Notes that the definition of this type is
  /// now complete.
  void completeDefinition();

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const RecordDecl *D) { return true; }
  static bool classofKind(Kind K) {
    return K >= RecordFirst && K <= RecordLast;
  }
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

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const FileScopeAsmDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == FileScopeAsm; }
};

/// BlockDecl - This represents a block literal declaration, which is like an
/// unnamed FunctionDecl.  For example:
/// ^{ statement-body }   or   ^(int arg1, float arg2){ statement-body }
///
class BlockDecl : public Decl, public DeclContext {
  // FIXME: This can be packed into the bitfields in Decl.
  bool isVariadic : 1;
  /// ParamInfo - new[]'d array of pointers to ParmVarDecls for the formal
  /// parameters of this function.  This is null if a prototype or if there are
  /// no formals.
  ParmVarDecl **ParamInfo;
  unsigned NumParams;

  Stmt *Body;

protected:
  BlockDecl(DeclContext *DC, SourceLocation CaretLoc)
    : Decl(Block, DC, CaretLoc), DeclContext(Block),
      isVariadic(false), ParamInfo(0), NumParams(0), Body(0) {}

  virtual ~BlockDecl();
  virtual void Destroy(ASTContext& C);

public:
  static BlockDecl *Create(ASTContext &C, DeclContext *DC, SourceLocation L);

  SourceLocation getCaretLocation() const { return getLocation(); }

  bool IsVariadic() const { return isVariadic; }
  void setIsVariadic(bool value) { isVariadic = value; }

  CompoundStmt *getCompoundBody() const { return (CompoundStmt*) Body; }
  Stmt *getBody() const { return (Stmt*) Body; }
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
  void setParams(ParmVarDecl **NewParamInfo, unsigned NumParams);

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const BlockDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == Block; }
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
