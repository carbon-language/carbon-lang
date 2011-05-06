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
#include "llvm/ADT/Optional.h"

namespace clang {
class CXXTemporary;
class Expr;
class FunctionTemplateDecl;
class Stmt;
class CompoundStmt;
class StringLiteral;
class NestedNameSpecifier;
class TemplateParameterList;
class TemplateArgumentList;
class MemberSpecializationInfo;
class FunctionTemplateSpecializationInfo;
class DependentFunctionTemplateSpecializationInfo;
class TypeLoc;
class UnresolvedSetImpl;
class LabelStmt;

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
  TypeLoc getTypeLoc() const; // implemented in TypeLoc.h
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

  void printName(llvm::raw_ostream &os) const { return Name.printName(os); }

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

  /// \brief Given that this declaration is a C++ class member,
  /// determine whether it's an instance member of its class.
  bool isCXXInstanceMember() const;

  class LinkageInfo {
    Linkage linkage_;
    Visibility visibility_;
    bool explicit_;

  public:
    LinkageInfo() : linkage_(ExternalLinkage), visibility_(DefaultVisibility),
                    explicit_(false) {}
    LinkageInfo(Linkage L, Visibility V, bool E)
      : linkage_(L), visibility_(V), explicit_(E) {}

    static LinkageInfo external() {
      return LinkageInfo();
    }
    static LinkageInfo internal() {
      return LinkageInfo(InternalLinkage, DefaultVisibility, false);
    }
    static LinkageInfo uniqueExternal() {
      return LinkageInfo(UniqueExternalLinkage, DefaultVisibility, false);
    }
    static LinkageInfo none() {
      return LinkageInfo(NoLinkage, DefaultVisibility, false);
    }

    Linkage linkage() const { return linkage_; }
    Visibility visibility() const { return visibility_; }
    bool visibilityExplicit() const { return explicit_; }

    void setLinkage(Linkage L) { linkage_ = L; }
    void setVisibility(Visibility V) { visibility_ = V; }
    void setVisibility(Visibility V, bool E) { visibility_ = V; explicit_ = E; }
    void setVisibility(LinkageInfo Other) {
      setVisibility(Other.visibility(), Other.visibilityExplicit());
    }

    void mergeLinkage(Linkage L) {
      setLinkage(minLinkage(linkage(), L));
    }
    void mergeLinkage(LinkageInfo Other) {
      setLinkage(minLinkage(linkage(), Other.linkage()));
    }

    void mergeVisibility(Visibility V) {
      setVisibility(minVisibility(visibility(), V));
    }
    void mergeVisibility(Visibility V, bool E) {
      setVisibility(minVisibility(visibility(), V), visibilityExplicit() || E);
    }
    void mergeVisibility(LinkageInfo Other) {
      mergeVisibility(Other.visibility(), Other.visibilityExplicit());
    }

    void merge(LinkageInfo Other) {
      mergeLinkage(Other);
      mergeVisibility(Other);
    }
    void merge(std::pair<Linkage,Visibility> LV) {
      mergeLinkage(LV.first);
      mergeVisibility(LV.second);
    }

    friend LinkageInfo merge(LinkageInfo L, LinkageInfo R) {
      L.merge(R);
      return L;
    }
  };

  /// \brief Determine what kind of linkage this entity has.
  Linkage getLinkage() const;

  /// \brief Determines the visibility of this entity.
  Visibility getVisibility() const { return getLinkageAndVisibility().visibility(); }

  /// \brief Determines the linkage and visibility of this entity.
  LinkageInfo getLinkageAndVisibility() const;

  /// \brief If visibility was explicitly specified for this
  /// declaration, return that visibility.
  llvm::Optional<Visibility> getExplicitVisibility() const;

  /// \brief Clear the linkage cache in response to a change
  /// to the declaration. 
  void ClearLinkageCache();

  /// \brief Looks through UsingDecls and ObjCCompatibleAliasDecls for
  /// the underlying named decl.
  NamedDecl *getUnderlyingDecl();
  const NamedDecl *getUnderlyingDecl() const {
    return const_cast<NamedDecl*>(this)->getUnderlyingDecl();
  }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const NamedDecl *D) { return true; }
  static bool classofKind(Kind K) { return K >= firstNamed && K <= lastNamed; }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const NamedDecl *ND) {
  ND->getDeclName().printName(OS);
  return OS;
}

/// LabelDecl - Represents the declaration of a label.  Labels also have a
/// corresponding LabelStmt, which indicates the position that the label was
/// defined at.  For normal labels, the location of the decl is the same as the
/// location of the statement.  For GNU local labels (__label__), the decl
/// location is where the __label__ is.
class LabelDecl : public NamedDecl {
  LabelStmt *TheStmt;
  /// LocStart - For normal labels, this is the same as the main declaration
  /// label, i.e., the location of the identifier; for GNU local labels,
  /// this is the location of the __label__ keyword.
  SourceLocation LocStart;

  LabelDecl(DeclContext *DC, SourceLocation IdentL, IdentifierInfo *II,
            LabelStmt *S, SourceLocation StartL)
    : NamedDecl(Label, DC, IdentL, II), TheStmt(S), LocStart(StartL) {}

public:
  static LabelDecl *Create(ASTContext &C, DeclContext *DC,
                           SourceLocation IdentL, IdentifierInfo *II);
  static LabelDecl *Create(ASTContext &C, DeclContext *DC,
                           SourceLocation IdentL, IdentifierInfo *II,
                           SourceLocation GnuLabelL);

  LabelStmt *getStmt() const { return TheStmt; }
  void setStmt(LabelStmt *T) { TheStmt = T; }

  bool isGnuLocal() const { return LocStart != getLocation(); }
  void setLocStart(SourceLocation L) { LocStart = L; }

  SourceRange getSourceRange() const {
    return SourceRange(LocStart, getLocation());
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const LabelDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == Label; }
};
  
/// NamespaceDecl - Represent a C++ namespace.
class NamespaceDecl : public NamedDecl, public DeclContext {
  bool IsInline : 1;

  /// LocStart - The starting location of the source range, pointing
  /// to either the namespace or the inline keyword.
  SourceLocation LocStart;
  /// RBraceLoc - The ending location of the source range.
  SourceLocation RBraceLoc;

  // For extended namespace definitions:
  //
  // namespace A { int x; }
  // namespace A { int y; }
  //
  // there will be one NamespaceDecl for each declaration.
  // NextNamespace points to the next extended declaration.
  // OrigNamespace points to the original namespace declaration.
  // OrigNamespace of the first namespace decl points to its anonymous namespace
  LazyDeclPtr NextNamespace;

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

  NamespaceDecl(DeclContext *DC, SourceLocation StartLoc,
                SourceLocation IdLoc, IdentifierInfo *Id)
    : NamedDecl(Namespace, DC, IdLoc, Id), DeclContext(Namespace),
      IsInline(false), LocStart(StartLoc), RBraceLoc(),
      NextNamespace(), OrigOrAnonNamespace(0, true) { }

public:
  static NamespaceDecl *Create(ASTContext &C, DeclContext *DC,
                               SourceLocation StartLoc,
                               SourceLocation IdLoc, IdentifierInfo *Id);

  /// \brief Returns true if this is an anonymous namespace declaration.
  ///
  /// For example:
  /// \code
  ///   namespace {
  ///     ...
  ///   };
  /// \endcode
  /// q.v. C++ [namespace.unnamed]
  bool isAnonymousNamespace() const {
    return !getIdentifier();
  }

  /// \brief Returns true if this is an inline namespace declaration.
  bool isInline() const {
    return IsInline;
  }

  /// \brief Set whether this is an inline namespace declaration.
  void setInline(bool Inline) {
    IsInline = Inline;
  }

  /// \brief Return the next extended namespace declaration or null if there
  /// is none.
  NamespaceDecl *getNextNamespace();
  const NamespaceDecl *getNextNamespace() const { 
    return const_cast<NamespaceDecl *>(this)->getNextNamespace();
  }

  /// \brief Set the next extended namespace declaration.
  void setNextNamespace(NamespaceDecl *ND) { NextNamespace = ND; }

  /// \brief Get the original (first) namespace declaration.
  NamespaceDecl *getOriginalNamespace() const {
    if (OrigOrAnonNamespace.getInt())
      return const_cast<NamespaceDecl *>(this);

    return OrigOrAnonNamespace.getPointer();
  }

  /// \brief Return true if this declaration is an original (first) declaration
  /// of the namespace. This is false for non-original (subsequent) namespace
  /// declarations and anonymous namespaces.
  bool isOriginalNamespace() const {
    return getOriginalNamespace() == this;
  }

  /// \brief Set the original (first) namespace declaration.
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
    assert(!D || D->getParent()->getRedeclContext() == this);
    getOriginalNamespace()->OrigOrAnonNamespace.setPointer(D);
  }

  virtual NamespaceDecl *getCanonicalDecl() { return getOriginalNamespace(); }
  const NamespaceDecl *getCanonicalDecl() const { 
    return getOriginalNamespace(); 
  }

  virtual SourceRange getSourceRange() const {
    return SourceRange(LocStart, RBraceLoc);
  }

  SourceLocation getLocStart() const { return LocStart; }
  SourceLocation getRBraceLoc() const { return RBraceLoc; }
  void setLocStart(SourceLocation L) { LocStart = L; }
  void setRBraceLoc(SourceLocation L) { RBraceLoc = L; }

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
  
  friend class ASTDeclReader;
  friend class ASTDeclWriter;
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
  static bool classofKind(Kind K) { return K >= firstValue && K <= lastValue; }
};

/// QualifierInfo - A struct with extended info about a syntactic
/// name qualifier, to be used for the case of out-of-line declarations.
struct QualifierInfo {
  NestedNameSpecifierLoc QualifierLoc;

  /// NumTemplParamLists - The number of "outer" template parameter lists.
  /// The count includes all of the template parameter lists that were matched
  /// against the template-ids occurring into the NNS and possibly (in the
  /// case of an explicit specialization) a final "template <>".
  unsigned NumTemplParamLists;

  /// TemplParamLists - A new-allocated array of size NumTemplParamLists,
  /// containing pointers to the "outer" template parameter lists.
  /// It includes all of the template parameter lists that were matched
  /// against the template-ids occurring into the NNS and possibly (in the
  /// case of an explicit specialization) a final "template <>".
  TemplateParameterList** TemplParamLists;

  /// Default constructor.
  QualifierInfo() : QualifierLoc(), NumTemplParamLists(0), TemplParamLists(0) {}

  /// setTemplateParameterListsInfo - Sets info about "outer" template
  /// parameter lists.
  void setTemplateParameterListsInfo(ASTContext &Context,
                                     unsigned NumTPLists,
                                     TemplateParameterList **TPLists);
  
private:
  // Copy constructor and copy assignment are disabled.
  QualifierInfo(const QualifierInfo&);
  QualifierInfo& operator=(const QualifierInfo&);
};

/// \brief Represents a ValueDecl that came out of a declarator.
/// Contains type source information through TypeSourceInfo.
class DeclaratorDecl : public ValueDecl {
  // A struct representing both a TInfo and a syntactic qualifier,
  // to be used for the (uncommon) case of out-of-line declarations.
  struct ExtInfo : public QualifierInfo {
    TypeSourceInfo *TInfo;
  };

  llvm::PointerUnion<TypeSourceInfo*, ExtInfo*> DeclInfo;

  /// InnerLocStart - The start of the source range for this declaration,
  /// ignoring outer template declarations.
  SourceLocation InnerLocStart;

  bool hasExtInfo() const { return DeclInfo.is<ExtInfo*>(); }
  ExtInfo *getExtInfo() { return DeclInfo.get<ExtInfo*>(); }
  const ExtInfo *getExtInfo() const { return DeclInfo.get<ExtInfo*>(); }

protected:
  DeclaratorDecl(Kind DK, DeclContext *DC, SourceLocation L,
                 DeclarationName N, QualType T, TypeSourceInfo *TInfo,
                 SourceLocation StartL)
    : ValueDecl(DK, DC, L, N, T), DeclInfo(TInfo), InnerLocStart(StartL) {
  }

public:
  TypeSourceInfo *getTypeSourceInfo() const {
    return hasExtInfo()
      ? getExtInfo()->TInfo
      : DeclInfo.get<TypeSourceInfo*>();
  }
  void setTypeSourceInfo(TypeSourceInfo *TI) {
    if (hasExtInfo())
      getExtInfo()->TInfo = TI;
    else
      DeclInfo = TI;
  }

  /// getInnerLocStart - Return SourceLocation representing start of source
  /// range ignoring outer template declarations.
  SourceLocation getInnerLocStart() const { return InnerLocStart; }
  void setInnerLocStart(SourceLocation L) { InnerLocStart = L; }

  /// getOuterLocStart - Return SourceLocation representing start of source
  /// range taking into account any outer template declarations.
  SourceLocation getOuterLocStart() const;

  virtual SourceRange getSourceRange() const;

  /// \brief Retrieve the nested-name-specifier that qualifies the name of this
  /// declaration, if it was present in the source.
  NestedNameSpecifier *getQualifier() const {
    return hasExtInfo() ? getExtInfo()->QualifierLoc.getNestedNameSpecifier()
                        : 0;
  }
  
  /// \brief Retrieve the nested-name-specifier (with source-location 
  /// information) that qualifies the name of this declaration, if it was 
  /// present in the source.
  NestedNameSpecifierLoc getQualifierLoc() const {
    return hasExtInfo() ? getExtInfo()->QualifierLoc
                        : NestedNameSpecifierLoc();
  }
  
  void setQualifierInfo(NestedNameSpecifierLoc QualifierLoc);

  unsigned getNumTemplateParameterLists() const {
    return hasExtInfo() ? getExtInfo()->NumTemplParamLists : 0;
  }
  TemplateParameterList *getTemplateParameterList(unsigned index) const {
    assert(index < getNumTemplateParameterLists());
    return getExtInfo()->TemplParamLists[index];
  }
  void setTemplateParameterListsInfo(ASTContext &Context, unsigned NumTPLists,
                                     TemplateParameterList **TPLists);

  SourceLocation getTypeSpecStartLoc() const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const DeclaratorDecl *D) { return true; }
  static bool classofKind(Kind K) {
    return K >= firstDeclarator && K <= lastDeclarator;
  }

  friend class ASTDeclReader;
  friend class ASTDeclWriter;
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

/// VarDecl - An instance of this class is created to represent a variable
/// declaration or definition.
class VarDecl : public DeclaratorDecl, public Redeclarable<VarDecl> {
public:
  typedef clang::StorageClass StorageClass;

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
  class VarDeclBitfields {
    friend class VarDecl;
    friend class ASTDeclReader;

    unsigned SClass : 3;
    unsigned SClassAsWritten : 3;
    unsigned ThreadSpecified : 1;
    unsigned HasCXXDirectInit : 1;

    /// \brief Whether this variable is the exception variable in a C++ catch
    /// or an Objective-C @catch statement.
    unsigned ExceptionVar : 1;
  
    /// \brief Whether this local variable could be allocated in the return
    /// slot of its function, enabling the named return value optimization (NRVO).
    unsigned NRVOVariable : 1;

    /// \brief Whether this variable is the for-range-declaration in a C++0x
    /// for-range statement.
    unsigned CXXForRangeDecl : 1;
  };
  enum { NumVarDeclBits = 13 }; // two reserved bits for now

  friend class ASTDeclReader;
  friend class StmtIteratorBase;
  
protected:
  class ParmVarDeclBitfields {
    friend class ParmVarDecl;
    friend class ASTDeclReader;

    unsigned : NumVarDeclBits;

    /// Whether this parameter inherits a default argument from a
    /// prior declaration.
    unsigned HasInheritedDefaultArg : 1;

    /// Whether this parameter undergoes K&R argument promotion.
    unsigned IsKNRPromoted : 1;

    /// Whether this parameter is an ObjC method parameter or not.
    unsigned IsObjCMethodParam : 1;

    /// If IsObjCMethodParam, a Decl::ObjCDeclQualifier.
    /// Otherwise, the number of function parameter scopes enclosing
    /// the function parameter scope in which this parameter was
    /// declared.
    unsigned ScopeDepthOrObjCQuals : 8;

    /// The number of parameters preceding this parameter in the
    /// function parameter scope in which it was declared.
    unsigned ParameterIndex : 8;
  };

  union {
    unsigned AllBits;
    VarDeclBitfields VarDeclBits;
    ParmVarDeclBitfields ParmVarDeclBits;
  };

  VarDecl(Kind DK, DeclContext *DC, SourceLocation StartLoc,
          SourceLocation IdLoc, IdentifierInfo *Id,
          QualType T, TypeSourceInfo *TInfo, StorageClass SC,
          StorageClass SCAsWritten)
    : DeclaratorDecl(DK, DC, IdLoc, Id, T, TInfo, StartLoc), Init() {
    assert(sizeof(VarDeclBitfields) <= sizeof(unsigned));
    assert(sizeof(ParmVarDeclBitfields) <= sizeof(unsigned));
    AllBits = 0;
    VarDeclBits.SClass = SC;
    VarDeclBits.SClassAsWritten = SCAsWritten;
    // Everything else is implicitly initialized to false.
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
                         SourceLocation StartLoc, SourceLocation IdLoc,
                         IdentifierInfo *Id, QualType T, TypeSourceInfo *TInfo,
                         StorageClass S, StorageClass SCAsWritten);

  virtual SourceRange getSourceRange() const;

  StorageClass getStorageClass() const {
    return (StorageClass) VarDeclBits.SClass;
  }
  StorageClass getStorageClassAsWritten() const {
    return (StorageClass) VarDeclBits.SClassAsWritten;
  }
  void setStorageClass(StorageClass SC);
  void setStorageClassAsWritten(StorageClass SC) {
    assert(isLegalForVariable(SC));
    VarDeclBits.SClassAsWritten = SC;
  }

  void setThreadSpecified(bool T) { VarDeclBits.ThreadSpecified = T; }
  bool isThreadSpecified() const {
    return VarDeclBits.ThreadSpecified;
  }

  /// hasLocalStorage - Returns true if a variable with function scope
  ///  is a non-static local variable.
  bool hasLocalStorage() const {
    if (getStorageClass() == SC_None)
      return !isFileVarDecl();

    // Return true for:  Auto, Register.
    // Return false for: Extern, Static, PrivateExtern.

    return getStorageClass() >= SC_Auto;
  }

  /// isStaticLocal - Returns true if a variable with function scope is a 
  /// static local variable.
  bool isStaticLocal() const {
    return getStorageClass() == SC_Static && !isFileVarDecl();
  }
  
  /// hasExternStorage - Returns true if a variable has extern or
  /// __private_extern__ storage.
  bool hasExternalStorage() const {
    return getStorageClass() == SC_Extern ||
           getStorageClass() == SC_PrivateExtern;
  }

  /// hasGlobalStorage - Returns true for all variables that do not
  ///  have local storage.  This includs all global variables as well
  ///  as static variables declared within a function.
  bool hasGlobalStorage() const { return !hasLocalStorage(); }

  /// \brief Determines whether this variable is a variable with
  /// external, C linkage.
  bool isExternC() const;

  /// isLocalVarDecl - Returns true for local variable declarations
  /// other than parameters.  Note that this includes static variables
  /// inside of functions. It also includes variables inside blocks.
  ///
  ///   void foo() { int x; static int y; extern int z; }
  ///
  bool isLocalVarDecl() const {
    if (getKind() != Decl::Var)
      return false;
    if (const DeclContext *DC = getDeclContext())
      return DC->getRedeclContext()->isFunctionOrMethod();
    return false;
  }

  /// isFunctionOrMethodVarDecl - Similar to isLocalVarDecl, but
  /// excludes variables declared in blocks.
  bool isFunctionOrMethodVarDecl() const {
    if (getKind() != Decl::Var)
      return false;
    const DeclContext *DC = getDeclContext()->getRedeclContext();
    return DC->isFunctionOrMethod() && DC->getDeclKind() != Decl::Block;
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
    return getKind() != Decl::ParmVar && getDeclContext()->isRecord();
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

  /// \brief Check whether this variable is defined in this
  /// translation unit.
  DefinitionKind hasDefinition() const;

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
    
    if (getDeclContext()->getRedeclContext()->isFileContext())
      return true;
    
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
    return !Init.isNull() && (Init.is<Stmt *>() || Init.is<EvaluatedStmt *>());
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

  void setCXXDirectInitializer(bool T) { VarDeclBits.HasCXXDirectInit = T; }

  /// hasCXXDirectInitializer - If true, the initializer was a direct
  /// initializer, e.g: "int x(1);". The Init expression will be the expression
  /// inside the parens or a "ClassType(a,b,c)" class constructor expression for
  /// class types. Clients can distinguish between "int x(1);" and "int x=1;"
  /// by checking hasCXXDirectInitializer.
  ///
  bool hasCXXDirectInitializer() const {
    return VarDeclBits.HasCXXDirectInit;
  }

  /// \brief Determine whether this variable is the exception variable in a
  /// C++ catch statememt or an Objective-C @catch statement.
  bool isExceptionVariable() const {
    return VarDeclBits.ExceptionVar;
  }
  void setExceptionVariable(bool EV) { VarDeclBits.ExceptionVar = EV; }
  
  /// \brief Determine whether this local variable can be used with the named
  /// return value optimization (NRVO).
  ///
  /// The named return value optimization (NRVO) works by marking certain
  /// non-volatile local variables of class type as NRVO objects. These
  /// locals can be allocated within the return slot of their containing
  /// function, in which case there is no need to copy the object to the
  /// return slot when returning from the function. Within the function body,
  /// each return that returns the NRVO object will have this variable as its
  /// NRVO candidate.
  bool isNRVOVariable() const { return VarDeclBits.NRVOVariable; }
  void setNRVOVariable(bool NRVO) { VarDeclBits.NRVOVariable = NRVO; }

  /// \brief Determine whether this variable is the for-range-declaration in
  /// a C++0x for-range statement.
  bool isCXXForRangeDecl() const { return VarDeclBits.CXXForRangeDecl; }
  void setCXXForRangeDecl(bool FRD) { VarDeclBits.CXXForRangeDecl = FRD; }
  
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
  static bool classofKind(Kind K) { return K >= firstVar && K <= lastVar; }
};

class ImplicitParamDecl : public VarDecl {
public:
  static ImplicitParamDecl *Create(ASTContext &C, DeclContext *DC,
                                   SourceLocation IdLoc, IdentifierInfo *Id,
                                   QualType T);

  ImplicitParamDecl(DeclContext *DC, SourceLocation IdLoc,
                    IdentifierInfo *Id, QualType Type)
    : VarDecl(ImplicitParam, DC, IdLoc, IdLoc, Id, Type,
              /*tinfo*/ 0, SC_None, SC_None) {
    setImplicit();
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const ImplicitParamDecl *D) { return true; }
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == ImplicitParam; }
};

/// ParmVarDecl - Represents a parameter to a function.
class ParmVarDecl : public VarDecl {
public:
  enum { MaxFunctionScopeDepth = 255 };
  enum { MaxFunctionScopeIndex = 255 };

protected:
  ParmVarDecl(Kind DK, DeclContext *DC, SourceLocation StartLoc,
              SourceLocation IdLoc, IdentifierInfo *Id,
              QualType T, TypeSourceInfo *TInfo,
              StorageClass S, StorageClass SCAsWritten, Expr *DefArg)
    : VarDecl(DK, DC, StartLoc, IdLoc, Id, T, TInfo, S, SCAsWritten) {
    assert(ParmVarDeclBits.HasInheritedDefaultArg == false);
    assert(ParmVarDeclBits.IsKNRPromoted == false);
    assert(ParmVarDeclBits.IsObjCMethodParam == false);
    setDefaultArg(DefArg);
  }

public:
  static ParmVarDecl *Create(ASTContext &C, DeclContext *DC,
                             SourceLocation StartLoc,
                             SourceLocation IdLoc, IdentifierInfo *Id,
                             QualType T, TypeSourceInfo *TInfo,
                             StorageClass S, StorageClass SCAsWritten,
                             Expr *DefArg);

  void setObjCMethodScopeInfo(unsigned parameterIndex) {
    ParmVarDeclBits.IsObjCMethodParam = true;

    ParmVarDeclBits.ParameterIndex = parameterIndex;
    assert(ParmVarDeclBits.ParameterIndex == parameterIndex && "truncation!");
  }

  void setScopeInfo(unsigned scopeDepth, unsigned parameterIndex) {
    assert(!ParmVarDeclBits.IsObjCMethodParam);

    ParmVarDeclBits.ScopeDepthOrObjCQuals = scopeDepth;
    assert(ParmVarDeclBits.ScopeDepthOrObjCQuals == scopeDepth && "truncation!");

    ParmVarDeclBits.ParameterIndex = parameterIndex;
    assert(ParmVarDeclBits.ParameterIndex == parameterIndex && "truncation!");
  }

  bool isObjCMethodParameter() const {
    return ParmVarDeclBits.IsObjCMethodParam;
  }

  unsigned getFunctionScopeDepth() const {
    if (ParmVarDeclBits.IsObjCMethodParam) return 0;
    return ParmVarDeclBits.ScopeDepthOrObjCQuals;
  }

  /// Returns the index of this parameter in its prototype or method scope.
  unsigned getFunctionScopeIndex() const {
    return ParmVarDeclBits.ParameterIndex;
  }

  ObjCDeclQualifier getObjCDeclQualifier() const {
    if (!ParmVarDeclBits.IsObjCMethodParam) return OBJC_TQ_None;
    return ObjCDeclQualifier(ParmVarDeclBits.ScopeDepthOrObjCQuals);
  }
  void setObjCDeclQualifier(ObjCDeclQualifier QTVal) {
    assert(ParmVarDeclBits.IsObjCMethodParam);
    ParmVarDeclBits.ScopeDepthOrObjCQuals = QTVal;
  }

  /// True if the value passed to this parameter must undergo
  /// K&R-style default argument promotion:
  ///
  /// C99 6.5.2.2.
  ///   If the expression that denotes the called function has a type
  ///   that does not include a prototype, the integer promotions are
  ///   performed on each argument, and arguments that have type float
  ///   are promoted to double.
  bool isKNRPromoted() const {
    return ParmVarDeclBits.IsKNRPromoted;
  }
  void setKNRPromoted(bool promoted) {
    ParmVarDeclBits.IsKNRPromoted = promoted;
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
    return ParmVarDeclBits.HasInheritedDefaultArg;
  }

  void setHasInheritedDefaultArg(bool I = true) {
    ParmVarDeclBits.HasInheritedDefaultArg = I;
  }

  QualType getOriginalType() const {
    if (getTypeSourceInfo())
      return getTypeSourceInfo()->getType();
    return getType();
  }

  /// \brief Determine whether this parameter is actually a function
  /// parameter pack.
  bool isParameterPack() const;
  
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
  typedef clang::StorageClass StorageClass;

  /// \brief The kind of templated function a FunctionDecl can be.
  enum TemplatedKind {
    TK_NonTemplate,
    TK_FunctionTemplate,
    TK_MemberSpecialization,
    TK_FunctionTemplateSpecialization,
    TK_DependentFunctionTemplateSpecialization
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
  unsigned SClassAsWritten : 2;
  bool IsInline : 1;
  bool IsInlineSpecified : 1;
  bool IsVirtualAsWritten : 1;
  bool IsPure : 1;
  bool HasInheritedPrototype : 1;
  bool HasWrittenPrototype : 1;
  bool IsDeleted : 1;
  bool IsTrivial : 1; // sunk from CXXMethodDecl
  bool IsDefaulted : 1; // sunk from CXXMethoDecl
  bool IsExplicitlyDefaulted : 1; //sunk from CXXMethodDecl
  bool HasImplicitReturnZero : 1;
  bool IsLateTemplateParsed : 1;

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
  llvm::PointerUnion4<FunctionTemplateDecl *, 
                      MemberSpecializationInfo *,
                      FunctionTemplateSpecializationInfo *,
                      DependentFunctionTemplateSpecializationInfo *>
    TemplateOrSpecialization;

  /// DNLoc - Provides source/type location info for the
  /// declaration name embedded in the DeclaratorDecl base class.
  DeclarationNameLoc DNLoc;

  /// \brief Specify that this function declaration is actually a function
  /// template specialization.
  ///
  /// \param C the ASTContext.
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
  ///
  /// \param TemplateArgsAsWritten location info of template arguments.
  ///
  /// \param PointOfInstantiation point at which the function template
  /// specialization was first instantiated. 
  void setFunctionTemplateSpecialization(ASTContext &C,
                                         FunctionTemplateDecl *Template,
                                       const TemplateArgumentList *TemplateArgs,
                                         void *InsertPos,
                                         TemplateSpecializationKind TSK,
                          const TemplateArgumentListInfo *TemplateArgsAsWritten,
                                         SourceLocation PointOfInstantiation);

  /// \brief Specify that this record is an instantiation of the
  /// member function FD.
  void setInstantiationOfMemberFunction(ASTContext &C, FunctionDecl *FD,
                                        TemplateSpecializationKind TSK);

  void setParams(ASTContext &C, ParmVarDecl **NewParamInfo, unsigned NumParams);

protected:
  FunctionDecl(Kind DK, DeclContext *DC, SourceLocation StartLoc,
               const DeclarationNameInfo &NameInfo,
               QualType T, TypeSourceInfo *TInfo,
               StorageClass S, StorageClass SCAsWritten, bool isInlineSpecified)
    : DeclaratorDecl(DK, DC, NameInfo.getLoc(), NameInfo.getName(), T, TInfo,
                     StartLoc),
      DeclContext(DK),
      ParamInfo(0), Body(),
      SClass(S), SClassAsWritten(SCAsWritten),
      IsInline(isInlineSpecified), IsInlineSpecified(isInlineSpecified),
      IsVirtualAsWritten(false), IsPure(false), HasInheritedPrototype(false),
      HasWrittenPrototype(true), IsDeleted(false), IsTrivial(false),
      HasImplicitReturnZero(false), IsLateTemplateParsed(false),
      EndRangeLoc(NameInfo.getEndLoc()),
      TemplateOrSpecialization(),
      DNLoc(NameInfo.getInfo()) {}

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

  static FunctionDecl *Create(ASTContext &C, DeclContext *DC,
                              SourceLocation StartLoc, SourceLocation NLoc,
                              DeclarationName N, QualType T,
                              TypeSourceInfo *TInfo,
                              StorageClass SC = SC_None,
                              StorageClass SCAsWritten = SC_None,
                              bool isInlineSpecified = false,
                              bool hasWrittenPrototype = true) {
    DeclarationNameInfo NameInfo(N, NLoc);
    return FunctionDecl::Create(C, DC, StartLoc, NameInfo, T, TInfo,
                                SC, SCAsWritten,
                                isInlineSpecified, hasWrittenPrototype);
  }

  static FunctionDecl *Create(ASTContext &C, DeclContext *DC,
                              SourceLocation StartLoc,
                              const DeclarationNameInfo &NameInfo,
                              QualType T, TypeSourceInfo *TInfo,
                              StorageClass SC = SC_None,
                              StorageClass SCAsWritten = SC_None,
                              bool isInlineSpecified = false,
                              bool hasWrittenPrototype = true);

  DeclarationNameInfo getNameInfo() const {
    return DeclarationNameInfo(getDeclName(), getLocation(), DNLoc);
  }

  virtual void getNameForDiagnostic(std::string &S,
                                    const PrintingPolicy &Policy,
                                    bool Qualified) const;

  void setRangeEnd(SourceLocation E) { EndRangeLoc = E; }

  virtual SourceRange getSourceRange() const;

  /// \brief Returns true if the function has a body (definition). The
  /// function body might be in any of the (re-)declarations of this
  /// function. The variant that accepts a FunctionDecl pointer will
  /// set that function declaration to the actual declaration
  /// containing the body (if there is one).
  bool hasBody(const FunctionDecl *&Definition) const;

  virtual bool hasBody() const {
    const FunctionDecl* Definition;
    return hasBody(Definition);
  }

  /// getBody - Retrieve the body (definition) of the function. The
  /// function body might be in any of the (re-)declarations of this
  /// function. The variant that accepts a FunctionDecl pointer will
  /// set that function declaration to the actual declaration
  /// containing the body (if there is one).
  /// NOTE: For checking if there is a body, use hasBody() instead, to avoid
  /// unnecessary AST de-serialization of the body.
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
  bool isThisDeclarationADefinition() const {
    return Body || IsLateTemplateParsed;
  }

  void setBody(Stmt *B);
  void setLazyBody(uint64_t Offset) { Body = Offset; }

  /// Whether this function is variadic.
  bool isVariadic() const;

  /// Whether this function is marked as virtual explicitly.
  bool isVirtualAsWritten() const { return IsVirtualAsWritten; }
  void setVirtualAsWritten(bool V) { IsVirtualAsWritten = V; }

  /// Whether this virtual function is pure, i.e. makes the containing class
  /// abstract.
  bool isPure() const { return IsPure; }
  void setPure(bool P = true);

  /// Whether this is a constexpr function or constexpr constructor.
  // FIXME: C++0x: Implement tracking of the constexpr specifier.
  bool isConstExpr() const { return false; }

  /// Whether this templated function will be late parsed.
  bool isLateTemplateParsed() const { return IsLateTemplateParsed; }
  void setLateTemplateParsed(bool ILT = true) { IsLateTemplateParsed = ILT; }

  /// Whether this function is "trivial" in some specialized C++ senses.
  /// Can only be true for default constructors, copy constructors,
  /// copy assignment operators, and destructors.  Not meaningful until
  /// the class has been fully built by Sema.
  bool isTrivial() const { return IsTrivial; }
  void setTrivial(bool IT) { IsTrivial = IT; }

  /// Whether this function is defaulted per C++0x. Only valid for
  /// special member functions. 
  bool isDefaulted() const { return IsDefaulted; }
  void setDefaulted(bool D = true) { IsDefaulted = D; }

  /// Whether this function is explicitly defaulted per C++0x. Only valid
  /// for special member functions.
  bool isExplicitlyDefaulted() const { return IsExplicitlyDefaulted; }
  void setExplicitlyDefaulted(bool ED = true) { IsExplicitlyDefaulted = ED; }

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
  void setParams(ParmVarDecl **NewParamInfo, unsigned NumParams) {
    setParams(getASTContext(), NewParamInfo, NumParams);
  }

  /// getMinRequiredArguments - Returns the minimum number of arguments
  /// needed to call this function. This may be fewer than the number of
  /// function parameters, if some of the parameters have default
  /// arguments (in C++).
  unsigned getMinRequiredArguments() const;

  QualType getResultType() const {
    return getType()->getAs<FunctionType>()->getResultType();
  }
  
  /// \brief Determine the type of an expression that calls this function.
  QualType getCallResultType() const {
    return getType()->getAs<FunctionType>()->getCallResultType(getASTContext());
  }
                       
  StorageClass getStorageClass() const { return StorageClass(SClass); }
  void setStorageClass(StorageClass SC);

  StorageClass getStorageClassAsWritten() const {
    return StorageClass(SClassAsWritten);
  }

  /// \brief Determine whether the "inline" keyword was specified for this
  /// function.
  bool isInlineSpecified() const { return IsInlineSpecified; }
                       
  /// Set whether the "inline" keyword was specified for this function.
  void setInlineSpecified(bool I) { 
    IsInlineSpecified = I; 
    IsInline = I;
  }

  /// Flag that this function is implicitly inline.
  void setImplicitlyInline() {
    IsInline = true;
  }

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
  
  /// \brief What kind of templated function this is.
  TemplatedKind getTemplatedKind() const;

  /// \brief If this function is an instantiation of a member function of a
  /// class template specialization, retrieves the member specialization
  /// information.
  MemberSpecializationInfo *getMemberSpecializationInfo() const;
                       
  /// \brief Specify that this record is an instantiation of the
  /// member function FD.
  void setInstantiationOfMemberFunction(FunctionDecl *FD,
                                        TemplateSpecializationKind TSK) {
    setInstantiationOfMemberFunction(getASTContext(), FD, TSK);
  }

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

  /// \brief Retrieve the template argument list as written in the sources,
  /// if any.
  ///
  /// If this function declaration is not a function template specialization
  /// or if it had no explicit template argument list, returns NULL.
  /// Note that it an explicit template argument list may be written empty,
  /// e.g., template<> void foo<>(char* s);
  const TemplateArgumentListInfo*
  getTemplateSpecializationArgsAsWritten() const;

  /// \brief Specify that this function declaration is actually a function
  /// template specialization.
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
  ///
  /// \param TemplateArgsAsWritten location info of template arguments.
  ///
  /// \param PointOfInstantiation point at which the function template
  /// specialization was first instantiated. 
  void setFunctionTemplateSpecialization(FunctionTemplateDecl *Template,
                                      const TemplateArgumentList *TemplateArgs,
                                         void *InsertPos,
                    TemplateSpecializationKind TSK = TSK_ImplicitInstantiation,
                    const TemplateArgumentListInfo *TemplateArgsAsWritten = 0,
                    SourceLocation PointOfInstantiation = SourceLocation()) {
    setFunctionTemplateSpecialization(getASTContext(), Template, TemplateArgs,
                                      InsertPos, TSK, TemplateArgsAsWritten,
                                      PointOfInstantiation);
  }

  /// \brief Specifies that this function declaration is actually a
  /// dependent function template specialization.
  void setDependentTemplateSpecialization(ASTContext &Context,
                             const UnresolvedSetImpl &Templates,
                      const TemplateArgumentListInfo &TemplateArgs);

  DependentFunctionTemplateSpecializationInfo *
  getDependentSpecializationInfo() const {
    return TemplateOrSpecialization.
             dyn_cast<DependentFunctionTemplateSpecializationInfo*>();
  }

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
    return K >= firstFunction && K <= lastFunction;
  }
  static DeclContext *castToDeclContext(const FunctionDecl *D) {
    return static_cast<DeclContext *>(const_cast<FunctionDecl*>(D));
  }
  static FunctionDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<FunctionDecl *>(const_cast<DeclContext*>(DC));
  }

  friend class ASTDeclReader;
  friend class ASTDeclWriter;
};


/// FieldDecl - An instance of this class is created by Sema::ActOnField to
/// represent a member of a struct/union/class.
class FieldDecl : public DeclaratorDecl {
  // FIXME: This can be packed into the bitfields in Decl.
  bool Mutable : 1;
  mutable unsigned CachedFieldIndex : 31;

  Expr *BitWidth;
protected:
  FieldDecl(Kind DK, DeclContext *DC, SourceLocation StartLoc,
            SourceLocation IdLoc, IdentifierInfo *Id,
            QualType T, TypeSourceInfo *TInfo, Expr *BW, bool Mutable)
    : DeclaratorDecl(DK, DC, IdLoc, Id, T, TInfo, StartLoc),
      Mutable(Mutable), CachedFieldIndex(0), BitWidth(BW) {
  }

public:
  static FieldDecl *Create(const ASTContext &C, DeclContext *DC,
                           SourceLocation StartLoc, SourceLocation IdLoc,
                           IdentifierInfo *Id, QualType T,
                           TypeSourceInfo *TInfo, Expr *BW, bool Mutable);

  /// getFieldIndex - Returns the index of this field within its record,
  /// as appropriate for passing to ASTRecordLayout::getFieldOffset.
  unsigned getFieldIndex() const;

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

  SourceRange getSourceRange() const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const FieldDecl *D) { return true; }
  static bool classofKind(Kind K) { return K >= firstField && K <= lastField; }
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

public:

  static EnumConstantDecl *Create(ASTContext &C, EnumDecl *DC,
                                  SourceLocation L, IdentifierInfo *Id,
                                  QualType T, Expr *E,
                                  const llvm::APSInt &V);

  const Expr *getInitExpr() const { return (const Expr*) Init; }
  Expr *getInitExpr() { return (Expr*) Init; }
  const llvm::APSInt &getInitVal() const { return Val; }

  void setInitExpr(Expr *E) { Init = (Stmt*) E; }
  void setInitVal(const llvm::APSInt &V) { Val = V; }

  SourceRange getSourceRange() const;
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const EnumConstantDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == EnumConstant; }

  friend class StmtIteratorBase;
};

/// IndirectFieldDecl - An instance of this class is created to represent a
/// field injected from an anonymous union/struct into the parent scope.
/// IndirectFieldDecl are always implicit.
class IndirectFieldDecl : public ValueDecl {
  NamedDecl **Chaining;
  unsigned ChainingSize;

  IndirectFieldDecl(DeclContext *DC, SourceLocation L,
                    DeclarationName N, QualType T,
                    NamedDecl **CH, unsigned CHS)
    : ValueDecl(IndirectField, DC, L, N, T), Chaining(CH), ChainingSize(CHS) {}

public:
  static IndirectFieldDecl *Create(ASTContext &C, DeclContext *DC,
                                   SourceLocation L, IdentifierInfo *Id,
                                   QualType T, NamedDecl **CH, unsigned CHS);
  
  typedef NamedDecl * const *chain_iterator;
  chain_iterator chain_begin() const { return Chaining; }
  chain_iterator chain_end() const  { return Chaining+ChainingSize; }

  unsigned getChainingSize() const { return ChainingSize; }

  FieldDecl *getAnonField() const {
    assert(ChainingSize >= 2);
    return cast<FieldDecl>(Chaining[ChainingSize - 1]);
  }

  VarDecl *getVarDecl() const {
    assert(ChainingSize >= 2);
    return dyn_cast<VarDecl>(*chain_begin());
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const IndirectFieldDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == IndirectField; }
  friend class ASTDeclReader;
};

/// TypeDecl - Represents a declaration of a type.
///
class TypeDecl : public NamedDecl {
  /// TypeForDecl - This indicates the Type object that represents
  /// this TypeDecl.  It is a cache maintained by
  /// ASTContext::getTypedefType, ASTContext::getTagDeclType, and
  /// ASTContext::getTemplateTypeParmType, and TemplateTypeParmDecl.
  mutable const Type *TypeForDecl;
  /// LocStart - The start of the source range for this declaration.
  SourceLocation LocStart;
  friend class ASTContext;
  friend class DeclContext;
  friend class TagDecl;
  friend class TemplateTypeParmDecl;
  friend class TagType;

protected:
  TypeDecl(Kind DK, DeclContext *DC, SourceLocation L, IdentifierInfo *Id,
           SourceLocation StartL = SourceLocation())
    : NamedDecl(DK, DC, L, Id), TypeForDecl(0), LocStart(StartL) {}

public:
  // Low-level accessor
  const Type *getTypeForDecl() const { return TypeForDecl; }
  void setTypeForDecl(const Type *TD) { TypeForDecl = TD; }

  SourceLocation getLocStart() const { return LocStart; }
  void setLocStart(SourceLocation L) { LocStart = L; }
  virtual SourceRange getSourceRange() const {
    if (LocStart.isValid())
      return SourceRange(LocStart, getLocation());
    else
      return SourceRange(getLocation());
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const TypeDecl *D) { return true; }
  static bool classofKind(Kind K) { return K >= firstType && K <= lastType; }
};


/// Base class for declarations which introduce a typedef-name.
class TypedefNameDecl : public TypeDecl, public Redeclarable<TypedefNameDecl> {
  /// UnderlyingType - This is the type the typedef is set to.
  TypeSourceInfo *TInfo;

protected:
  TypedefNameDecl(Kind DK, DeclContext *DC, SourceLocation StartLoc,
                  SourceLocation IdLoc, IdentifierInfo *Id,
                  TypeSourceInfo *TInfo)
    : TypeDecl(DK, DC, IdLoc, Id, StartLoc), TInfo(TInfo) {}

  typedef Redeclarable<TypedefNameDecl> redeclarable_base;
  virtual TypedefNameDecl *getNextRedeclaration() {
    return RedeclLink.getNext();
  }

public:
  typedef redeclarable_base::redecl_iterator redecl_iterator;
  redecl_iterator redecls_begin() const {
    return redeclarable_base::redecls_begin();
  }
  redecl_iterator redecls_end() const {
    return redeclarable_base::redecls_end();
  }

  TypeSourceInfo *getTypeSourceInfo() const {
    return TInfo;
  }

  /// Retrieves the canonical declaration of this typedef-name.
  TypedefNameDecl *getCanonicalDecl() {
    return getFirstDeclaration();
  }
  const TypedefNameDecl *getCanonicalDecl() const {
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
  static bool classof(const TypedefNameDecl *D) { return true; }
  static bool classofKind(Kind K) {
    return K >= firstTypedefName && K <= lastTypedefName;
  }
};

/// TypedefDecl - Represents the declaration of a typedef-name via the 'typedef'
/// type specifier.
class TypedefDecl : public TypedefNameDecl {
  TypedefDecl(DeclContext *DC, SourceLocation StartLoc, SourceLocation IdLoc,
              IdentifierInfo *Id, TypeSourceInfo *TInfo)
    : TypedefNameDecl(Typedef, DC, StartLoc, IdLoc, Id, TInfo) {}

public:
  static TypedefDecl *Create(ASTContext &C, DeclContext *DC,
                             SourceLocation StartLoc, SourceLocation IdLoc,
                             IdentifierInfo *Id, TypeSourceInfo *TInfo);

  SourceRange getSourceRange() const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const TypedefDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == Typedef; }
};

/// TypeAliasDecl - Represents the declaration of a typedef-name via a C++0x
/// alias-declaration.
class TypeAliasDecl : public TypedefNameDecl {
  TypeAliasDecl(DeclContext *DC, SourceLocation StartLoc, SourceLocation IdLoc,
                IdentifierInfo *Id, TypeSourceInfo *TInfo)
    : TypedefNameDecl(TypeAlias, DC, StartLoc, IdLoc, Id, TInfo) {}

public:
  static TypeAliasDecl *Create(ASTContext &C, DeclContext *DC,
                               SourceLocation StartLoc, SourceLocation IdLoc,
                               IdentifierInfo *Id, TypeSourceInfo *TInfo);

  SourceRange getSourceRange() const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const TypeAliasDecl *D) { return true; }
  static bool classofKind(Kind K) { return K == TypeAlias; }
};

/// TagDecl - Represents the declaration of a struct/union/class/enum.
class TagDecl
  : public TypeDecl, public DeclContext, public Redeclarable<TagDecl> {
public:
  // This is really ugly.
  typedef TagTypeKind TagKind;

private:
  // FIXME: This can be packed into the bitfields in Decl.
  /// TagDeclKind - The TagKind enum.
  unsigned TagDeclKind : 2;

  /// IsDefinition - True if this is a definition ("struct foo {};"), false if
  /// it is a declaration ("struct foo;").
  bool IsDefinition : 1;

  /// IsBeingDefined - True if this is currently being defined.
  bool IsBeingDefined : 1;

  /// IsEmbeddedInDeclarator - True if this tag declaration is
  /// "embedded" (i.e., defined or declared for the very first time)
  /// in the syntax of a declarator.
  bool IsEmbeddedInDeclarator : 1;

protected:
  // These are used by (and only defined for) EnumDecl.
  unsigned NumPositiveBits : 8;
  unsigned NumNegativeBits : 8;

  /// IsScoped - True if this tag declaration is a scoped enumeration. Only
  /// possible in C++0x mode.
  bool IsScoped : 1;
  /// IsScopedUsingClassTag - If this tag declaration is a scoped enum,
  /// then this is true if the scoped enum was declared using the class
  /// tag, false if it was declared with the struct tag. No meaning is
  /// associated if this tag declaration is not a scoped enum.
  bool IsScopedUsingClassTag : 1;

  /// IsFixed - True if this is an enumeration with fixed underlying type. Only
  /// possible in C++0x mode.
  bool IsFixed : 1;

private:
  SourceLocation RBraceLoc;

  // A struct representing syntactic qualifier info,
  // to be used for the (uncommon) case of out-of-line declarations.
  typedef QualifierInfo ExtInfo;

  /// TypedefNameDeclOrQualifier - If the (out-of-line) tag declaration name
  /// is qualified, it points to the qualifier info (nns and range);
  /// otherwise, if the tag declaration is anonymous and it is part of
  /// a typedef or alias, it points to the TypedefNameDecl (used for mangling);
  /// otherwise, it is a null (TypedefNameDecl) pointer.
  llvm::PointerUnion<TypedefNameDecl*, ExtInfo*> TypedefNameDeclOrQualifier;

  bool hasExtInfo() const { return TypedefNameDeclOrQualifier.is<ExtInfo*>(); }
  ExtInfo *getExtInfo() { return TypedefNameDeclOrQualifier.get<ExtInfo*>(); }
  const ExtInfo *getExtInfo() const {
    return TypedefNameDeclOrQualifier.get<ExtInfo*>();
  }

protected:
  TagDecl(Kind DK, TagKind TK, DeclContext *DC,
          SourceLocation L, IdentifierInfo *Id,
          TagDecl *PrevDecl, SourceLocation StartL)
    : TypeDecl(DK, DC, L, Id, StartL), DeclContext(DK),
      TypedefNameDeclOrQualifier((TypedefNameDecl*) 0) {
    assert((DK != Enum || TK == TTK_Enum) &&
           "EnumDecl not matched with TTK_Enum");
    TagDeclKind = TK;
    IsDefinition = false;
    IsBeingDefined = false;
    IsEmbeddedInDeclarator = false;
    setPreviousDeclaration(PrevDecl);
  }

  typedef Redeclarable<TagDecl> redeclarable_base;
  virtual TagDecl *getNextRedeclaration() { return RedeclLink.getNext(); }

  /// @brief Completes the definition of this tag declaration.
  ///
  /// This is a helper function for derived classes.
  void completeDefinition();    
    
public:
  typedef redeclarable_base::redecl_iterator redecl_iterator;
  redecl_iterator redecls_begin() const {
    return redeclarable_base::redecls_begin();
  }
  redecl_iterator redecls_end() const {
    return redeclarable_base::redecls_end();
  }

  SourceLocation getRBraceLoc() const { return RBraceLoc; }
  void setRBraceLoc(SourceLocation L) { RBraceLoc = L; }

  /// getInnerLocStart - Return SourceLocation representing start of source
  /// range ignoring outer template declarations.
  SourceLocation getInnerLocStart() const { return getLocStart(); }

  /// getOuterLocStart - Return SourceLocation representing start of source
  /// range taking into account any outer template declarations.
  SourceLocation getOuterLocStart() const;
  virtual SourceRange getSourceRange() const;

  virtual TagDecl* getCanonicalDecl();
  const TagDecl* getCanonicalDecl() const {
    return const_cast<TagDecl*>(this)->getCanonicalDecl();
  }

  /// isThisDeclarationADefinition() - Return true if this declaration
  /// defines the type.  Provided for consistency.
  bool isThisDeclarationADefinition() const {
    return isDefinition();
  }

  /// isDefinition - Return true if this decl has its body specified.
  bool isDefinition() const {
    return IsDefinition;
  }

  /// isBeingDefined - Return true if this decl is currently being defined.
  bool isBeingDefined() const {
    return IsBeingDefined;
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
    return TypeWithKeyword::getTagTypeKindName(getTagKind());
  }

  TagKind getTagKind() const {
    return TagKind(TagDeclKind);
  }

  void setTagKind(TagKind TK) { TagDeclKind = TK; }

  bool isStruct() const { return getTagKind() == TTK_Struct; }
  bool isClass()  const { return getTagKind() == TTK_Class; }
  bool isUnion()  const { return getTagKind() == TTK_Union; }
  bool isEnum()   const { return getTagKind() == TTK_Enum; }

  TypedefNameDecl *getTypedefNameForAnonDecl() const {
    return hasExtInfo() ? 0 : TypedefNameDeclOrQualifier.get<TypedefNameDecl*>();
  }

  void setTypedefNameForAnonDecl(TypedefNameDecl *TDD);

  /// \brief Retrieve the nested-name-specifier that qualifies the name of this
  /// declaration, if it was present in the source.
  NestedNameSpecifier *getQualifier() const {
    return hasExtInfo() ? getExtInfo()->QualifierLoc.getNestedNameSpecifier()
                        : 0;
  }
  
  /// \brief Retrieve the nested-name-specifier (with source-location 
  /// information) that qualifies the name of this declaration, if it was 
  /// present in the source.
  NestedNameSpecifierLoc getQualifierLoc() const {
    return hasExtInfo() ? getExtInfo()->QualifierLoc
                        : NestedNameSpecifierLoc();
  }
    
  void setQualifierInfo(NestedNameSpecifierLoc QualifierLoc);

  unsigned getNumTemplateParameterLists() const {
    return hasExtInfo() ? getExtInfo()->NumTemplParamLists : 0;
  }
  TemplateParameterList *getTemplateParameterList(unsigned i) const {
    assert(i < getNumTemplateParameterLists());
    return getExtInfo()->TemplParamLists[i];
  }
  void setTemplateParameterListsInfo(ASTContext &Context, unsigned NumTPLists,
                                     TemplateParameterList **TPLists);

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const TagDecl *D) { return true; }
  static bool classofKind(Kind K) { return K >= firstTag && K <= lastTag; }

  static DeclContext *castToDeclContext(const TagDecl *D) {
    return static_cast<DeclContext *>(const_cast<TagDecl*>(D));
  }
  static TagDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<TagDecl *>(const_cast<DeclContext*>(DC));
  }

  friend class ASTDeclReader;
  friend class ASTDeclWriter;
};

/// EnumDecl - Represents an enum.  As an extension, we allow forward-declared
/// enums.
class EnumDecl : public TagDecl {
  /// IntegerType - This represent the integer type that the enum corresponds
  /// to for code generation purposes.  Note that the enumerator constants may
  /// have a different type than this does.
  ///
  /// If the underlying integer type was explicitly stated in the source
  /// code, this is a TypeSourceInfo* for that type. Otherwise this type
  /// was automatically deduced somehow, and this is a Type*.
  ///
  /// Normally if IsFixed(), this would contain a TypeSourceInfo*, but in
  /// some cases it won't.
  ///
  /// The underlying type of an enumeration never has any qualifiers, so
  /// we can get away with just storing a raw Type*, and thus save an
  /// extra pointer when TypeSourceInfo is needed.

  llvm::PointerUnion<const Type*, TypeSourceInfo*> IntegerType;

  /// PromotionType - The integer type that values of this type should
  /// promote to.  In C, enumerators are generally of an integer type
  /// directly, but gcc-style large enumerators (and all enumerators
  /// in C++) are of the enum type instead.
  QualType PromotionType;

  /// \brief If the enumeration was instantiated from an enumeration
  /// within a class or function template, this pointer refers to the
  /// enumeration declared within the template.
  EnumDecl *InstantiatedFrom;

  // The number of positive and negative bits required by the
  // enumerators are stored in the SubclassBits field.
  enum {
    NumBitsWidth = 8,
    NumBitsMask = (1 << NumBitsWidth) - 1
  };

  EnumDecl(DeclContext *DC, SourceLocation StartLoc, SourceLocation IdLoc,
           IdentifierInfo *Id, EnumDecl *PrevDecl,
           bool Scoped, bool ScopedUsingClassTag, bool Fixed)
    : TagDecl(Enum, TTK_Enum, DC, IdLoc, Id, PrevDecl, StartLoc),
      InstantiatedFrom(0) {
    assert(Scoped || !ScopedUsingClassTag);
    IntegerType = (const Type*)0;
    NumNegativeBits = 0;
    NumPositiveBits = 0;
    IsScoped = Scoped;
    IsScopedUsingClassTag = ScopedUsingClassTag;
    IsFixed = Fixed;
  }
public:
  EnumDecl *getCanonicalDecl() {
    return cast<EnumDecl>(TagDecl::getCanonicalDecl());
  }
  const EnumDecl *getCanonicalDecl() const {
    return cast<EnumDecl>(TagDecl::getCanonicalDecl());
  }

  const EnumDecl *getPreviousDeclaration() const {
    return cast_or_null<EnumDecl>(TagDecl::getPreviousDeclaration());
  }
  EnumDecl *getPreviousDeclaration() {
    return cast_or_null<EnumDecl>(TagDecl::getPreviousDeclaration());
  }

  static EnumDecl *Create(ASTContext &C, DeclContext *DC,
                          SourceLocation StartLoc, SourceLocation IdLoc,
                          IdentifierInfo *Id, EnumDecl *PrevDecl,
                          bool IsScoped, bool IsScopedUsingClassTag,
                          bool IsFixed);
  static EnumDecl *Create(ASTContext &C, EmptyShell Empty);

  /// completeDefinition - When created, the EnumDecl corresponds to a
  /// forward-declared enum. This method is used to mark the
  /// declaration as being defined; it's enumerators have already been
  /// added (via DeclContext::addDecl). NewType is the new underlying
  /// type of the enumeration type.
  void completeDefinition(QualType NewType,
                          QualType PromotionType,
                          unsigned NumPositiveBits,
                          unsigned NumNegativeBits);

  // enumerator_iterator - Iterates through the enumerators of this
  // enumeration.
  typedef specific_decl_iterator<EnumConstantDecl> enumerator_iterator;

  enumerator_iterator enumerator_begin() const {
    const EnumDecl *E = cast_or_null<EnumDecl>(getDefinition());
    if (!E)
      E = this;
    return enumerator_iterator(E->decls_begin());
  }

  enumerator_iterator enumerator_end() const {
    const EnumDecl *E = cast_or_null<EnumDecl>(getDefinition());
    if (!E)
      E = this;
    return enumerator_iterator(E->decls_end());
  }

  /// getPromotionType - Return the integer type that enumerators
  /// should promote to.
  QualType getPromotionType() const { return PromotionType; }

  /// \brief Set the promotion type.
  void setPromotionType(QualType T) { PromotionType = T; }

  /// getIntegerType - Return the integer type this enum decl corresponds to.
  /// This returns a null qualtype for an enum forward definition.
  QualType getIntegerType() const {
    if (!IntegerType)
      return QualType();
    if (const Type* T = IntegerType.dyn_cast<const Type*>())
      return QualType(T, 0);
    return IntegerType.get<TypeSourceInfo*>()->getType();
  }

  /// \brief Set the underlying integer type.
  void setIntegerType(QualType T) { IntegerType = T.getTypePtrOrNull(); }

  /// \brief Set the underlying integer type source info.
  void setIntegerTypeSourceInfo(TypeSourceInfo* TInfo) { IntegerType = TInfo; }

  /// \brief Return the type source info for the underlying integer type,
  /// if no type source info exists, return 0.
  TypeSourceInfo* getIntegerTypeSourceInfo() const {
    return IntegerType.dyn_cast<TypeSourceInfo*>();
  }

  /// \brief Returns the width in bits required to store all the
  /// non-negative enumerators of this enum.
  unsigned getNumPositiveBits() const {
    return NumPositiveBits;
  }
  void setNumPositiveBits(unsigned Num) {
    NumPositiveBits = Num;
    assert(NumPositiveBits == Num && "can't store this bitcount");
  }

  /// \brief Returns the width in bits required to store all the
  /// negative enumerators of this enum.  These widths include
  /// the rightmost leading 1;  that is:
  /// 
  /// MOST NEGATIVE ENUMERATOR     PATTERN     NUM NEGATIVE BITS
  /// ------------------------     -------     -----------------
  ///                       -1     1111111                     1
  ///                      -10     1110110                     5
  ///                     -101     1001011                     8
  unsigned getNumNegativeBits() const {
    return NumNegativeBits;
  }
  void setNumNegativeBits(unsigned Num) {
    NumNegativeBits = Num;
  }

  /// \brief Returns true if this is a C++0x scoped enumeration.
  bool isScoped() const {
    return IsScoped;
  }

  /// \brief Returns true if this is a C++0x scoped enumeration.
  bool isScopedUsingClassTag() const {
    return IsScopedUsingClassTag;
  }

  /// \brief Returns true if this is a C++0x enumeration with fixed underlying
  /// type.
  bool isFixed() const {
    return IsFixed;
  }

  /// \brief Returns true if this can be considered a complete type.
  bool isComplete() const {
    return isDefinition() || isFixed();
  }

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

  friend class ASTDeclReader;
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

  /// \brief Whether the field declarations of this record have been loaded
  /// from external storage. To avoid unnecessary deserialization of
  /// methods/nested types we allow deserialization of just the fields
  /// when needed.
  mutable bool LoadedFieldsFromExternalStorage : 1;
  friend class DeclContext;

protected:
  RecordDecl(Kind DK, TagKind TK, DeclContext *DC,
             SourceLocation StartLoc, SourceLocation IdLoc,
             IdentifierInfo *Id, RecordDecl *PrevDecl);

public:
  static RecordDecl *Create(const ASTContext &C, TagKind TK, DeclContext *DC,
                            SourceLocation StartLoc, SourceLocation IdLoc,
                            IdentifierInfo *Id, RecordDecl* PrevDecl = 0);
  static RecordDecl *Create(const ASTContext &C, EmptyShell Empty);

  const RecordDecl *getPreviousDeclaration() const {
    return cast_or_null<RecordDecl>(TagDecl::getPreviousDeclaration());
  }
  RecordDecl *getPreviousDeclaration() {
    return cast_or_null<RecordDecl>(TagDecl::getPreviousDeclaration());
  }

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

  field_iterator field_begin() const;

  field_iterator field_end() const {
    return field_iterator(decl_iterator());
  }

  // field_empty - Whether there are any fields (non-static data
  // members) in this record.
  bool field_empty() const {
    return field_begin() == field_end();
  }

  /// completeDefinition - Notes that the definition of this type is
  /// now complete.
  virtual void completeDefinition();

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classof(const RecordDecl *D) { return true; }
  static bool classofKind(Kind K) {
    return K >= firstRecord && K <= lastRecord;
  }

private:
  /// \brief Deserialize just the fields.
  void LoadFieldsFromExternalStorage() const;
};

class FileScopeAsmDecl : public Decl {
  StringLiteral *AsmString;
  SourceLocation RParenLoc;
  FileScopeAsmDecl(DeclContext *DC, StringLiteral *asmstring,
                   SourceLocation StartL, SourceLocation EndL)
    : Decl(FileScopeAsm, DC, StartL), AsmString(asmstring), RParenLoc(EndL) {}
public:
  static FileScopeAsmDecl *Create(ASTContext &C, DeclContext *DC,
                                  StringLiteral *Str, SourceLocation AsmLoc,
                                  SourceLocation RParenLoc);

  SourceLocation getAsmLoc() const { return getLocation(); }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }
  SourceRange getSourceRange() const {
    return SourceRange(getAsmLoc(), getRParenLoc());
  }

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
public:
  /// A class which contains all the information about a particular
  /// captured value.
  class Capture {
    enum {
      flag_isByRef = 0x1,
      flag_isNested = 0x2
    };

    /// The variable being captured.
    llvm::PointerIntPair<VarDecl*, 2> VariableAndFlags;

    /// The copy expression, expressed in terms of a DeclRef (or
    /// BlockDeclRef) to the captured variable.  Only required if the
    /// variable has a C++ class type.
    Expr *CopyExpr;

  public:
    Capture(VarDecl *variable, bool byRef, bool nested, Expr *copy)
      : VariableAndFlags(variable,
                  (byRef ? flag_isByRef : 0) | (nested ? flag_isNested : 0)),
        CopyExpr(copy) {}

    /// The variable being captured.
    VarDecl *getVariable() const { return VariableAndFlags.getPointer(); }

    /// Whether this is a "by ref" capture, i.e. a capture of a __block
    /// variable.
    bool isByRef() const { return VariableAndFlags.getInt() & flag_isByRef; }

    /// Whether this is a nested capture, i.e. the variable captured
    /// is not from outside the immediately enclosing function/block.
    bool isNested() const { return VariableAndFlags.getInt() & flag_isNested; }

    bool hasCopyExpr() const { return CopyExpr != 0; }
    Expr *getCopyExpr() const { return CopyExpr; }
    void setCopyExpr(Expr *e) { CopyExpr = e; }
  };

private:
  // FIXME: This can be packed into the bitfields in Decl.
  bool IsVariadic : 1;
  bool CapturesCXXThis : 1;
  /// ParamInfo - new[]'d array of pointers to ParmVarDecls for the formal
  /// parameters of this function.  This is null if a prototype or if there are
  /// no formals.
  ParmVarDecl **ParamInfo;
  unsigned NumParams;

  Stmt *Body;
  TypeSourceInfo *SignatureAsWritten;

  Capture *Captures;
  unsigned NumCaptures;

protected:
  BlockDecl(DeclContext *DC, SourceLocation CaretLoc)
    : Decl(Block, DC, CaretLoc), DeclContext(Block),
      IsVariadic(false), CapturesCXXThis(false),
      ParamInfo(0), NumParams(0), Body(0),
      SignatureAsWritten(0), Captures(0), NumCaptures(0) {}

public:
  static BlockDecl *Create(ASTContext &C, DeclContext *DC, SourceLocation L);

  SourceLocation getCaretLocation() const { return getLocation(); }

  bool isVariadic() const { return IsVariadic; }
  void setIsVariadic(bool value) { IsVariadic = value; }

  CompoundStmt *getCompoundBody() const { return (CompoundStmt*) Body; }
  Stmt *getBody() const { return (Stmt*) Body; }
  void setBody(CompoundStmt *B) { Body = (Stmt*) B; }

  void setSignatureAsWritten(TypeSourceInfo *Sig) { SignatureAsWritten = Sig; }
  TypeSourceInfo *getSignatureAsWritten() const { return SignatureAsWritten; }

  // Iterator access to formal parameters.
  unsigned param_size() const { return getNumParams(); }
  typedef ParmVarDecl **param_iterator;
  typedef ParmVarDecl * const *param_const_iterator;

  bool param_empty() const { return NumParams == 0; }
  param_iterator param_begin()  { return ParamInfo; }
  param_iterator param_end()   { return ParamInfo+param_size(); }

  param_const_iterator param_begin() const { return ParamInfo; }
  param_const_iterator param_end() const   { return ParamInfo+param_size(); }

  unsigned getNumParams() const { return NumParams; }
  const ParmVarDecl *getParamDecl(unsigned i) const {
    assert(i < getNumParams() && "Illegal param #");
    return ParamInfo[i];
  }
  ParmVarDecl *getParamDecl(unsigned i) {
    assert(i < getNumParams() && "Illegal param #");
    return ParamInfo[i];
  }
  void setParams(ParmVarDecl **NewParamInfo, unsigned NumParams);

  /// hasCaptures - True if this block (or its nested blocks) captures
  /// anything of local storage from its enclosing scopes.
  bool hasCaptures() const { return NumCaptures != 0 || CapturesCXXThis; }

  /// getNumCaptures - Returns the number of captured variables.
  /// Does not include an entry for 'this'.
  unsigned getNumCaptures() const { return NumCaptures; }

  typedef const Capture *capture_iterator;
  typedef const Capture *capture_const_iterator;
  capture_iterator capture_begin() { return Captures; }
  capture_iterator capture_end() { return Captures + NumCaptures; }
  capture_const_iterator capture_begin() const { return Captures; }
  capture_const_iterator capture_end() const { return Captures + NumCaptures; }

  bool capturesCXXThis() const { return CapturesCXXThis; }

  void setCaptures(ASTContext &Context,
                   const Capture *begin,
                   const Capture *end,
                   bool capturesCXXThis);

  virtual SourceRange getSourceRange() const;
  
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

template<typename decl_type>
void Redeclarable<decl_type>::setPreviousDeclaration(decl_type *PrevDecl) {
  // Note: This routine is implemented here because we need both NamedDecl
  // and Redeclarable to be defined.

  decl_type *First;
  
  if (PrevDecl) {
    // Point to previous. Make sure that this is actually the most recent
    // redeclaration, or we can build invalid chains. If the most recent
    // redeclaration is invalid, it won't be PrevDecl, but we want it anyway.
    RedeclLink = PreviousDeclLink(llvm::cast<decl_type>(
                                                        PrevDecl->getMostRecentDeclaration()));
    First = PrevDecl->getFirstDeclaration();
    assert(First->RedeclLink.NextIsLatest() && "Expected first");
  } else {
    // Make this first.
    First = static_cast<decl_type*>(this);
  }
  
  // First one will point to this one as latest.
  First->RedeclLink = LatestDeclLink(static_cast<decl_type*>(this));
  if (NamedDecl *ND = dyn_cast<NamedDecl>(static_cast<decl_type*>(this)))
    ND->ClearLinkageCache();
}

}  // end namespace clang

#endif
