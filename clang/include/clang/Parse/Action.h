//===--- Action.h - Parser Action Interface ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Action and EmptyAction interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_ACTION_H
#define LLVM_CLANG_PARSE_ACTION_H

#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TemplateKinds.h"
#include "clang/Basic/TypeTraits.h"
#include "clang/Parse/AccessSpecifier.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/Ownership.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/ADT/PointerUnion.h"

namespace clang {
  // Semantic.
  class DeclSpec;
  class ObjCDeclSpec;
  class CXXScopeSpec;
  class Declarator;
  class AttributeList;
  struct FieldDeclarator;
  // Parse.
  class Scope;
  class Action;
  class Selector;
  class Designation;
  class InitListDesignations;
  // Lex.
  class Preprocessor;
  class Token;

  // We can re-use the low bit of expression, statement, base, and
  // member-initializer pointers for the "invalid" flag of
  // ActionResult.
  template<> struct IsResultPtrLowBitFree<0> { static const bool value = true;};
  template<> struct IsResultPtrLowBitFree<1> { static const bool value = true;};
  template<> struct IsResultPtrLowBitFree<3> { static const bool value = true;};
  template<> struct IsResultPtrLowBitFree<4> { static const bool value = true;};
  template<> struct IsResultPtrLowBitFree<5> { static const bool value = true;};

/// Action - As the parser reads the input file and recognizes the productions
/// of the grammar, it invokes methods on this class to turn the parsed input
/// into something useful: e.g. a parse tree.
///
/// The callback methods that this class provides are phrased as actions that
/// the parser has just done or is about to do when the method is called.  They
/// are not requests that the actions module do the specified action.
///
/// All of the methods here are optional except getTypeName() and
/// isCurrentClassName(), which must be specified in order for the
/// parse to complete accurately.  The MinimalAction class does this
/// bare-minimum of tracking to implement this functionality.
class Action : public ActionBase {
public:
  /// Out-of-line virtual destructor to provide home for this class.
  virtual ~Action();

  // Types - Though these don't actually enforce strong typing, they document
  // what types are required to be identical for the actions.
  typedef ActionBase::ExprTy ExprTy;
  typedef ActionBase::StmtTy StmtTy;

  /// Expr/Stmt/Type/BaseResult - Provide a unique type to wrap
  /// ExprTy/StmtTy/TypeTy/BaseTy, providing strong typing and
  /// allowing for failure.
  typedef ActionResult<0> ExprResult;
  typedef ActionResult<1> StmtResult;
  typedef ActionResult<2> TypeResult;
  typedef ActionResult<3> BaseResult;
  typedef ActionResult<4> MemInitResult;
  typedef ActionResult<5, DeclPtrTy> DeclResult;

  /// Same, but with ownership.
  typedef ASTOwningResult<&ActionBase::DeleteExpr> OwningExprResult;
  typedef ASTOwningResult<&ActionBase::DeleteStmt> OwningStmtResult;
  // Note that these will replace ExprResult and StmtResult when the transition
  // is complete.

  /// Single expressions or statements as arguments.
#if !defined(DISABLE_SMART_POINTERS)
  typedef ASTOwningResult<&ActionBase::DeleteExpr> ExprArg;
  typedef ASTOwningResult<&ActionBase::DeleteStmt> StmtArg;
#else
  typedef ASTOwningPtr<&ActionBase::DeleteExpr> ExprArg;
  typedef ASTOwningPtr<&ActionBase::DeleteStmt> StmtArg;
#endif

  /// Multiple expressions or statements as arguments.
  typedef ASTMultiPtr<&ActionBase::DeleteExpr> MultiExprArg;
  typedef ASTMultiPtr<&ActionBase::DeleteStmt> MultiStmtArg;
  typedef ASTMultiPtr<&ActionBase::DeleteTemplateParams> MultiTemplateParamsArg;

  class FullExprArg {
  public:
    // FIXME: The const_cast here is ugly. RValue references would make this
    // much nicer (or we could duplicate a bunch of the move semantics
    // emulation code from Ownership.h).
    FullExprArg(const FullExprArg& Other)
      : Expr(move(const_cast<FullExprArg&>(Other).Expr)) {}

    OwningExprResult release() {
      return move(Expr);
    }

    ExprArg* operator->() {
      return &Expr;
    }

  private:
    // FIXME: No need to make the entire Action class a friend when it's just
    // Action::FullExpr that needs access to the constructor below.
    friend class Action;

    explicit FullExprArg(ExprArg expr)
      : Expr(move(expr)) {}

    ExprArg Expr;
  };

  template<typename T>
  FullExprArg FullExpr(T &Arg) {
      return FullExprArg(ActOnFinishFullExpr(move(Arg)));
  }

  // Utilities for Action implementations to return smart results.

  OwningExprResult ExprError() { return OwningExprResult(*this, true); }
  OwningStmtResult StmtError() { return OwningStmtResult(*this, true); }

  OwningExprResult ExprError(const DiagnosticBuilder&) { return ExprError(); }
  OwningStmtResult StmtError(const DiagnosticBuilder&) { return StmtError(); }

  OwningExprResult ExprEmpty() { return OwningExprResult(*this, false); }
  OwningStmtResult StmtEmpty() { return OwningStmtResult(*this, false); }

  /// Statistics.
  virtual void PrintStats() const {}

  /// getDeclName - Return a pretty name for the specified decl if possible, or
  /// an empty string if not.  This is used for pretty crash reporting.
  virtual std::string getDeclName(DeclPtrTy D) { return ""; }

  /// \brief Invoked for each comment in the source code, providing the source
  /// range that contains the comment.
  virtual void ActOnComment(SourceRange Comment) { }

  //===--------------------------------------------------------------------===//
  // Declaration Tracking Callbacks.
  //===--------------------------------------------------------------------===//

  /// ConvertDeclToDeclGroup - If the parser has one decl in a context where it
  /// needs a decl group, it calls this to convert between the two
  /// representations.
  virtual DeclGroupPtrTy ConvertDeclToDeclGroup(DeclPtrTy Ptr) {
    return DeclGroupPtrTy();
  }

  /// getTypeName - Return non-null if the specified identifier is a type name
  /// in the current scope.
  ///
  /// \param II the identifier for which we are performing name lookup
  ///
  /// \param NameLoc the location of the identifier
  ///
  /// \param S the scope in which this name lookup occurs
  ///
  /// \param SS if non-NULL, the C++ scope specifier that precedes the
  /// identifier
  ///
  /// \param isClassName whether this is a C++ class-name production, in
  /// which we can end up referring to a member of an unknown specialization
  /// that we know (from the grammar) is supposed to be a type. For example,
  /// this occurs when deriving from "std::vector<T>::allocator_type", where T
  /// is a template parameter.
  ///
  /// \returns the type referred to by this identifier, or NULL if the type
  /// does not name an identifier.
  virtual TypeTy *getTypeName(IdentifierInfo &II, SourceLocation NameLoc,
                              Scope *S, const CXXScopeSpec *SS = 0,
                              bool isClassName = false) = 0;

  /// isTagName() - This method is called *for error recovery purposes only*
  /// to determine if the specified name is a valid tag name ("struct foo").  If
  /// so, this returns the TST for the tag corresponding to it (TST_enum,
  /// TST_union, TST_struct, TST_class).  This is used to diagnose cases in C
  /// where the user forgot to specify the tag.
  virtual DeclSpec::TST isTagName(IdentifierInfo &II, Scope *S) {
    return DeclSpec::TST_unspecified;
  }

  /// isCurrentClassName - Return true if the specified name is the
  /// name of the innermost C++ class type currently being defined.
  virtual bool isCurrentClassName(const IdentifierInfo &II, Scope *S,
                                  const CXXScopeSpec *SS = 0) = 0;

  /// \brief Determine whether the given identifier refers to the name of a
  /// template.
  ///
  /// \param S the scope in which name lookup occurs
  ///
  /// \param II the identifier that we are querying to determine whether it
  /// is a template.
  ///
  /// \param IdLoc the source location of the identifier
  ///
  /// \param SS the C++ scope specifier that precedes the template name, if
  /// any.
  ///
  /// \param EnteringContext whether we are potentially entering the context
  /// referred to by the scope specifier \p SS
  ///
  /// \param Template if the name does refer to a template, the declaration
  /// of the template that the name refers to.
  ///
  /// \returns the kind of template that this name refers to.
  virtual TemplateNameKind isTemplateName(Scope *S,
                                          const IdentifierInfo &II,
                                          SourceLocation IdLoc,
                                          const CXXScopeSpec *SS,
                                          TypeTy *ObjectType,
                                          bool EnteringContext,
                                          TemplateTy &Template) = 0;

  /// ActOnCXXGlobalScopeSpecifier - Return the object that represents the
  /// global scope ('::').
  virtual CXXScopeTy *ActOnCXXGlobalScopeSpecifier(Scope *S,
                                                   SourceLocation CCLoc) {
    return 0;
  }

  /// \brief Parsed an identifier followed by '::' in a C++
  /// nested-name-specifier.
  ///
  /// \param S the scope in which the nested-name-specifier was parsed.
  ///
  /// \param SS the nested-name-specifier that precedes the identifier. For
  /// example, if we are parsing "foo::bar::", \p SS will describe the "foo::"
  /// that has already been parsed.
  ///
  /// \param IdLoc the location of the identifier we have just parsed (e.g.,
  /// the "bar" in "foo::bar::".
  ///
  /// \param CCLoc the location of the '::' at the end of the
  /// nested-name-specifier.
  ///
  /// \param II the identifier that represents the scope that this
  /// nested-name-specifier refers to, e.g., the "bar" in "foo::bar::".
  ///
  /// \param ObjectType if this nested-name-specifier occurs as part of a
  /// C++ member access expression such as "x->Base::f", the type of the base
  /// object (e.g., *x in the example, if "x" were a pointer).
  ///
  /// \param EnteringContext if true, then we intend to immediately enter the
  /// context of this nested-name-specifier, e.g., for an out-of-line
  /// definition of a class member.
  ///
  /// \returns a CXXScopeTy* object representing the C++ scope.
  virtual CXXScopeTy *ActOnCXXNestedNameSpecifier(Scope *S,
                                                  const CXXScopeSpec &SS,
                                                  SourceLocation IdLoc,
                                                  SourceLocation CCLoc,
                                                  IdentifierInfo &II,
                                                  TypeTy *ObjectType,
                                                  bool EnteringContext) {
    return 0;
  }

  /// ActOnCXXNestedNameSpecifier - Called during parsing of a
  /// nested-name-specifier that involves a template-id, e.g.,
  /// "foo::bar<int, float>::", and now we need to build a scope
  /// specifier. \p SS is empty or the previously parsed nested-name
  /// part ("foo::"), \p Type is the already-parsed class template
  /// specialization (or other template-id that names a type), \p
  /// TypeRange is the source range where the type is located, and \p
  /// CCLoc is the location of the trailing '::'.
  virtual CXXScopeTy *ActOnCXXNestedNameSpecifier(Scope *S,
                                                  const CXXScopeSpec &SS,
                                                  TypeTy *Type,
                                                  SourceRange TypeRange,
                                                  SourceLocation CCLoc) {
    return 0;
  }

  /// ActOnCXXEnterDeclaratorScope - Called when a C++ scope specifier (global
  /// scope or nested-name-specifier) is parsed, part of a declarator-id.
  /// After this method is called, according to [C++ 3.4.3p3], names should be
  /// looked up in the declarator-id's scope, until the declarator is parsed and
  /// ActOnCXXExitDeclaratorScope is called.
  /// The 'SS' should be a non-empty valid CXXScopeSpec.
  virtual void ActOnCXXEnterDeclaratorScope(Scope *S, const CXXScopeSpec &SS) {
  }

  /// ActOnCXXExitDeclaratorScope - Called when a declarator that previously
  /// invoked ActOnCXXEnterDeclaratorScope(), is finished. 'SS' is the same
  /// CXXScopeSpec that was passed to ActOnCXXEnterDeclaratorScope as well.
  /// Used to indicate that names should revert to being looked up in the
  /// defining scope.
  virtual void ActOnCXXExitDeclaratorScope(Scope *S, const CXXScopeSpec &SS) {
  }

  /// ActOnCXXEnterDeclInitializer - Invoked when we are about to parse an
  /// initializer for the declaration 'Dcl'.
  /// After this method is called, according to [C++ 3.4.1p13], if 'Dcl' is a
  /// static data member of class X, names should be looked up in the scope of
  /// class X.
  virtual void ActOnCXXEnterDeclInitializer(Scope *S, DeclPtrTy Dcl) {
  }

  /// ActOnCXXExitDeclInitializer - Invoked after we are finished parsing an
  /// initializer for the declaration 'Dcl'.
  virtual void ActOnCXXExitDeclInitializer(Scope *S, DeclPtrTy Dcl) {
  }

  /// ActOnDeclarator - This callback is invoked when a declarator is parsed and
  /// 'Init' specifies the initializer if any.  This is for things like:
  /// "int X = 4" or "typedef int foo".
  ///
  virtual DeclPtrTy ActOnDeclarator(Scope *S, Declarator &D) {
    return DeclPtrTy();
  }

  /// ActOnParamDeclarator - This callback is invoked when a parameter
  /// declarator is parsed. This callback only occurs for functions
  /// with prototypes. S is the function prototype scope for the
  /// parameters (C++ [basic.scope.proto]).
  virtual DeclPtrTy ActOnParamDeclarator(Scope *S, Declarator &D) {
    return DeclPtrTy();
  }

  /// AddInitializerToDecl - This action is called immediately after
  /// ActOnDeclarator (when an initializer is present). The code is factored
  /// this way to make sure we are able to handle the following:
  ///   void func() { int xx = xx; }
  /// This allows ActOnDeclarator to register "xx" prior to parsing the
  /// initializer. The declaration above should still result in a warning,
  /// since the reference to "xx" is uninitialized.
  virtual void AddInitializerToDecl(DeclPtrTy Dcl, ExprArg Init) {
    return;
  }

  /// SetDeclDeleted - This action is called immediately after ActOnDeclarator
  /// if =delete is parsed. C++0x [dcl.fct.def]p10
  /// Note that this can be called even for variable declarations. It's the
  /// action's job to reject it.
  virtual void SetDeclDeleted(DeclPtrTy Dcl, SourceLocation DelLoc) {
    return;
  }

  /// ActOnUninitializedDecl - This action is called immediately after
  /// ActOnDeclarator (when an initializer is *not* present).
  /// If TypeContainsUndeducedAuto is true, then the type of the declarator
  /// has an undeduced 'auto' type somewhere.
  virtual void ActOnUninitializedDecl(DeclPtrTy Dcl,
                                      bool TypeContainsUndeducedAuto) {
    return;
  }

  /// FinalizeDeclaratorGroup - After a sequence of declarators are parsed, this
  /// gives the actions implementation a chance to process the group as a whole.
  virtual DeclGroupPtrTy FinalizeDeclaratorGroup(Scope *S, const DeclSpec& DS,
                                                 DeclPtrTy *Group,
                                                 unsigned NumDecls) {
    return DeclGroupPtrTy();
  }


  /// @brief Indicates that all K&R-style parameter declarations have
  /// been parsed prior to a function definition.
  /// @param S  The function prototype scope.
  /// @param D  The function declarator.
  virtual void ActOnFinishKNRParamDeclarations(Scope *S, Declarator &D,
                                               SourceLocation LocAfterDecls) {
  }

  /// ActOnStartOfFunctionDef - This is called at the start of a function
  /// definition, instead of calling ActOnDeclarator.  The Declarator includes
  /// information about formal arguments that are part of this function.
  virtual DeclPtrTy ActOnStartOfFunctionDef(Scope *FnBodyScope, Declarator &D) {
    // Default to ActOnDeclarator.
    return ActOnStartOfFunctionDef(FnBodyScope,
                                   ActOnDeclarator(FnBodyScope, D));
  }

  /// ActOnStartOfFunctionDef - This is called at the start of a function
  /// definition, after the FunctionDecl has already been created.
  virtual DeclPtrTy ActOnStartOfFunctionDef(Scope *FnBodyScope, DeclPtrTy D) {
    return D;
  }

  virtual void ActOnStartOfObjCMethodDef(Scope *FnBodyScope, DeclPtrTy D) {
    return;
  }

  /// ActOnFinishFunctionBody - This is called when a function body has
  /// completed parsing.  Decl is returned by ParseStartOfFunctionDef.
  virtual DeclPtrTy ActOnFinishFunctionBody(DeclPtrTy Decl, StmtArg Body) {
    return Decl;
  }

  virtual DeclPtrTy ActOnFileScopeAsmDecl(SourceLocation Loc,
                                          ExprArg AsmString) {
    return DeclPtrTy();
  }

  /// ActOnPopScope - This callback is called immediately before the specified
  /// scope is popped and deleted.
  virtual void ActOnPopScope(SourceLocation Loc, Scope *S) {}

  /// ActOnTranslationUnitScope - This callback is called once, immediately
  /// after creating the translation unit scope (in Parser::Initialize).
  virtual void ActOnTranslationUnitScope(SourceLocation Loc, Scope *S) {}

  /// ParsedFreeStandingDeclSpec - This method is invoked when a declspec with
  /// no declarator (e.g. "struct foo;") is parsed.
  virtual DeclPtrTy ParsedFreeStandingDeclSpec(Scope *S, DeclSpec &DS) {
    return DeclPtrTy();
  }

  /// ActOnStartLinkageSpecification - Parsed the beginning of a C++
  /// linkage specification, including the language and (if present)
  /// the '{'. ExternLoc is the location of the 'extern', LangLoc is
  /// the location of the language string literal, which is provided
  /// by Lang/StrSize. LBraceLoc, if valid, provides the location of
  /// the '{' brace. Otherwise, this linkage specification does not
  /// have any braces.
  virtual DeclPtrTy ActOnStartLinkageSpecification(Scope *S,
                                                   SourceLocation ExternLoc,
                                                   SourceLocation LangLoc,
                                                   const char *Lang,
                                                   unsigned StrSize,
                                                   SourceLocation LBraceLoc) {
    return DeclPtrTy();
  }

  /// ActOnFinishLinkageSpecification - Completely the definition of
  /// the C++ linkage specification LinkageSpec. If RBraceLoc is
  /// valid, it's the position of the closing '}' brace in a linkage
  /// specification that uses braces.
  virtual DeclPtrTy ActOnFinishLinkageSpecification(Scope *S,
                                                    DeclPtrTy LinkageSpec,
                                                    SourceLocation RBraceLoc) {
    return LinkageSpec;
  }

  /// ActOnEndOfTranslationUnit - This is called at the very end of the
  /// translation unit when EOF is reached and all but the top-level scope is
  /// popped.
  virtual void ActOnEndOfTranslationUnit() {}

  //===--------------------------------------------------------------------===//
  // Type Parsing Callbacks.
  //===--------------------------------------------------------------------===//

  /// ActOnTypeName - A type-name (type-id in C++) was parsed.
  virtual TypeResult ActOnTypeName(Scope *S, Declarator &D) {
    return TypeResult();
  }

  enum TagUseKind {
    TUK_Reference,   // Reference to a tag:  'struct foo *X;'
    TUK_Declaration, // Fwd decl of a tag:   'struct foo;'
    TUK_Definition,  // Definition of a tag: 'struct foo { int X; } Y;'
    TUK_Friend       // Friend declaration:  'friend struct foo;'
  };

  /// \brief The parser has encountered a tag (e.g., "class X") that should be
  /// turned into a declaration by the action module.
  ///
  /// \param S the scope in which this tag occurs.
  ///
  /// \param TagSpec an instance of DeclSpec::TST, indicating what kind of tag
  /// this is (struct/union/enum/class).
  ///
  /// \param TUK how the tag we have encountered is being used, which
  /// can be a reference to a (possibly pre-existing) tag, a
  /// declaration of that tag, or the beginning of a definition of
  /// that tag.
  ///
  /// \param KWLoc the location of the "struct", "class", "union", or "enum"
  /// keyword.
  ///
  /// \param SS C++ scope specifier that precedes the name of the tag, e.g.,
  /// the "std::" in "class std::type_info".
  ///
  /// \param Name the name of the tag, e.g., "X" in "struct X". This parameter
  /// may be NULL, to indicate an anonymous class/struct/union/enum type.
  ///
  /// \param NameLoc the location of the name of the tag.
  ///
  /// \param Attr the set of attributes that appertain to the tag.
  ///
  /// \param AS when this tag occurs within a C++ class, provides the
  /// current access specifier (AS_public, AS_private, AS_protected).
  /// Otherwise, it will be AS_none.
  ///
  /// \param TemplateParameterLists the set of C++ template parameter lists
  /// that apply to this tag, if the tag is a declaration or definition (see
  /// the \p TK parameter). The action module is responsible for determining,
  /// based on the template parameter lists and the scope specifier, whether
  /// the declared tag is a class template or not.
  ///
  /// \param OwnedDecl the callee should set this flag true when the returned
  /// declaration is "owned" by this reference. Ownership is handled entirely
  /// by the action module.
  ///
  /// \returns the declaration to which this tag refers.
  virtual DeclPtrTy ActOnTag(Scope *S, unsigned TagSpec, TagUseKind TUK,
                             SourceLocation KWLoc, const CXXScopeSpec &SS,
                             IdentifierInfo *Name, SourceLocation NameLoc,
                             AttributeList *Attr, AccessSpecifier AS,
                             MultiTemplateParamsArg TemplateParameterLists,
                             bool &OwnedDecl, bool &IsDependent) {
    return DeclPtrTy();
  }

  /// Acts on a reference to a dependent tag name.  This arises in
  /// cases like:
  ///
  ///    template <class T> class A;
  ///    template <class T> class B {
  ///      friend class A<T>::M;  // here
  ///    };
  ///
  /// \param TagSpec an instance of DeclSpec::TST corresponding to the
  /// tag specifier.
  ///
  /// \param TUK the tag use kind (either TUK_Friend or TUK_Reference)
  ///
  /// \param SS the scope specifier (always defined)
  virtual TypeResult ActOnDependentTag(Scope *S,
                                       unsigned TagSpec,
                                       TagUseKind TUK,
                                       const CXXScopeSpec &SS,
                                       IdentifierInfo *Name,
                                       SourceLocation KWLoc,
                                       SourceLocation NameLoc) {
    return TypeResult();
  }

  /// Act on @defs() element found when parsing a structure.  ClassName is the
  /// name of the referenced class.
  virtual void ActOnDefs(Scope *S, DeclPtrTy TagD, SourceLocation DeclStart,
                         IdentifierInfo *ClassName,
                         llvm::SmallVectorImpl<DeclPtrTy> &Decls) {}
  virtual DeclPtrTy ActOnField(Scope *S, DeclPtrTy TagD,
                               SourceLocation DeclStart,
                               Declarator &D, ExprTy *BitfieldWidth) {
    return DeclPtrTy();
  }

  virtual DeclPtrTy ActOnIvar(Scope *S, SourceLocation DeclStart,
                              DeclPtrTy IntfDecl,
                              Declarator &D, ExprTy *BitfieldWidth,
                              tok::ObjCKeywordKind visibility) {
    return DeclPtrTy();
  }

  virtual void ActOnFields(Scope* S, SourceLocation RecLoc, DeclPtrTy TagDecl,
                           DeclPtrTy *Fields, unsigned NumFields,
                           SourceLocation LBrac, SourceLocation RBrac,
                           AttributeList *AttrList) {}

  /// ActOnTagStartDefinition - Invoked when we have entered the
  /// scope of a tag's definition (e.g., for an enumeration, class,
  /// struct, or union).
  virtual void ActOnTagStartDefinition(Scope *S, DeclPtrTy TagDecl) { }

  /// ActOnTagFinishDefinition - Invoked once we have finished parsing
  /// the definition of a tag (enumeration, class, struct, or union).
  virtual void ActOnTagFinishDefinition(Scope *S, DeclPtrTy TagDecl,
                                        SourceLocation RBraceLoc) { }

  virtual DeclPtrTy ActOnEnumConstant(Scope *S, DeclPtrTy EnumDecl,
                                      DeclPtrTy LastEnumConstant,
                                      SourceLocation IdLoc, IdentifierInfo *Id,
                                      SourceLocation EqualLoc, ExprTy *Val) {
    return DeclPtrTy();
  }
  virtual void ActOnEnumBody(SourceLocation EnumLoc, SourceLocation LBraceLoc,
                             SourceLocation RBraceLoc, DeclPtrTy EnumDecl,
                             DeclPtrTy *Elements, unsigned NumElements,
                             Scope *S, AttributeList *AttrList) {}

  //===--------------------------------------------------------------------===//
  // Statement Parsing Callbacks.
  //===--------------------------------------------------------------------===//

  virtual OwningStmtResult ActOnNullStmt(SourceLocation SemiLoc) {
    return StmtEmpty();
  }

  virtual OwningStmtResult ActOnCompoundStmt(SourceLocation L, SourceLocation R,
                                             MultiStmtArg Elts,
                                             bool isStmtExpr) {
    return StmtEmpty();
  }
  virtual OwningStmtResult ActOnDeclStmt(DeclGroupPtrTy Decl,
                                         SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
    return StmtEmpty();
  }

  virtual OwningStmtResult ActOnExprStmt(FullExprArg Expr) {
    return OwningStmtResult(*this, Expr->release());
  }

  /// ActOnCaseStmt - Note that this handles the GNU 'case 1 ... 4' extension,
  /// which can specify an RHS value.  The sub-statement of the case is
  /// specified in a separate action.
  virtual OwningStmtResult ActOnCaseStmt(SourceLocation CaseLoc, ExprArg LHSVal,
                                         SourceLocation DotDotDotLoc,
                                         ExprArg RHSVal,
                                         SourceLocation ColonLoc) {
    return StmtEmpty();
  }

  /// ActOnCaseStmtBody - This installs a statement as the body of a case.
  virtual void ActOnCaseStmtBody(StmtTy *CaseStmt, StmtArg SubStmt) {}

  virtual OwningStmtResult ActOnDefaultStmt(SourceLocation DefaultLoc,
                                            SourceLocation ColonLoc,
                                            StmtArg SubStmt, Scope *CurScope){
    return StmtEmpty();
  }

  virtual OwningStmtResult ActOnLabelStmt(SourceLocation IdentLoc,
                                          IdentifierInfo *II,
                                          SourceLocation ColonLoc,
                                          StmtArg SubStmt) {
    return StmtEmpty();
  }

  virtual OwningStmtResult ActOnIfStmt(SourceLocation IfLoc,
                                       FullExprArg CondVal, StmtArg ThenVal,
                                       SourceLocation ElseLoc,
                                       StmtArg ElseVal) {
    return StmtEmpty();
  }

  virtual OwningStmtResult ActOnStartOfSwitchStmt(ExprArg Cond) {
    return StmtEmpty();
  }

  virtual OwningStmtResult ActOnFinishSwitchStmt(SourceLocation SwitchLoc,
                                                 StmtArg Switch, StmtArg Body) {
    return StmtEmpty();
  }

  virtual OwningStmtResult ActOnWhileStmt(SourceLocation WhileLoc,
                                          FullExprArg Cond, StmtArg Body) {
    return StmtEmpty();
  }
  virtual OwningStmtResult ActOnDoStmt(SourceLocation DoLoc, StmtArg Body,
                                       SourceLocation WhileLoc,
                                       SourceLocation CondLParen,
                                       ExprArg Cond,
                                       SourceLocation CondRParen) {
    return StmtEmpty();
  }
  virtual OwningStmtResult ActOnForStmt(SourceLocation ForLoc,
                                        SourceLocation LParenLoc,
                                        StmtArg First, ExprArg Second,
                                        ExprArg Third, SourceLocation RParenLoc,
                                        StmtArg Body) {
    return StmtEmpty();
  }
  virtual OwningStmtResult ActOnObjCForCollectionStmt(SourceLocation ForColLoc,
                                       SourceLocation LParenLoc,
                                       StmtArg First, ExprArg Second,
                                       SourceLocation RParenLoc, StmtArg Body) {
    return StmtEmpty();
  }
  virtual OwningStmtResult ActOnGotoStmt(SourceLocation GotoLoc,
                                         SourceLocation LabelLoc,
                                         IdentifierInfo *LabelII) {
    return StmtEmpty();
  }
  virtual OwningStmtResult ActOnIndirectGotoStmt(SourceLocation GotoLoc,
                                                 SourceLocation StarLoc,
                                                 ExprArg DestExp) {
    return StmtEmpty();
  }
  virtual OwningStmtResult ActOnContinueStmt(SourceLocation ContinueLoc,
                                             Scope *CurScope) {
    return StmtEmpty();
  }
  virtual OwningStmtResult ActOnBreakStmt(SourceLocation GotoLoc,
                                          Scope *CurScope) {
    return StmtEmpty();
  }
  virtual OwningStmtResult ActOnReturnStmt(SourceLocation ReturnLoc,
                                           ExprArg RetValExp) {
    return StmtEmpty();
  }
  virtual OwningStmtResult ActOnAsmStmt(SourceLocation AsmLoc,
                                        bool IsSimple,
                                        bool IsVolatile,
                                        unsigned NumOutputs,
                                        unsigned NumInputs,
                                        std::string *Names,
                                        MultiExprArg Constraints,
                                        MultiExprArg Exprs,
                                        ExprArg AsmString,
                                        MultiExprArg Clobbers,
                                        SourceLocation RParenLoc) {
    return StmtEmpty();
  }

  // Objective-c statements
  virtual OwningStmtResult ActOnObjCAtCatchStmt(SourceLocation AtLoc,
                                                SourceLocation RParen,
                                                DeclPtrTy Parm, StmtArg Body,
                                                StmtArg CatchList) {
    return StmtEmpty();
  }

  virtual OwningStmtResult ActOnObjCAtFinallyStmt(SourceLocation AtLoc,
                                                  StmtArg Body) {
    return StmtEmpty();
  }

  virtual OwningStmtResult ActOnObjCAtTryStmt(SourceLocation AtLoc,
                                              StmtArg Try, StmtArg Catch,
                                              StmtArg Finally) {
    return StmtEmpty();
  }

  virtual OwningStmtResult ActOnObjCAtThrowStmt(SourceLocation AtLoc,
                                                ExprArg Throw,
                                                Scope *CurScope) {
    return StmtEmpty();
  }

  virtual OwningStmtResult ActOnObjCAtSynchronizedStmt(SourceLocation AtLoc,
                                                       ExprArg SynchExpr,
                                                       StmtArg SynchBody) {
    return StmtEmpty();
  }

  // C++ Statements
  virtual DeclPtrTy ActOnExceptionDeclarator(Scope *S, Declarator &D) {
    return DeclPtrTy();
  }

  virtual OwningStmtResult ActOnCXXCatchBlock(SourceLocation CatchLoc,
                                              DeclPtrTy ExceptionDecl,
                                              StmtArg HandlerBlock) {
    return StmtEmpty();
  }

  virtual OwningStmtResult ActOnCXXTryBlock(SourceLocation TryLoc,
                                            StmtArg TryBlock,
                                            MultiStmtArg Handlers) {
    return StmtEmpty();
  }

  //===--------------------------------------------------------------------===//
  // Expression Parsing Callbacks.
  //===--------------------------------------------------------------------===//

  /// \brief Describes how the expressions currently being parsed are
  /// evaluated at run-time, if at all.
  enum ExpressionEvaluationContext {
    /// \brief The current expression and its subexpressions occur within an
    /// unevaluated operand (C++0x [expr]p8), such as a constant expression
    /// or the subexpression of \c sizeof, where the type or the value of the
    /// expression may be significant but no code will be generated to evaluate
    /// the value of the expression at run time.
    Unevaluated,

    /// \brief The current expression is potentially evaluated at run time,
    /// which means that code may be generated to evaluate the value of the
    /// expression at run time.
    PotentiallyEvaluated,

    /// \brief The current expression may be potentially evaluated or it may
    /// be unevaluated, but it is impossible to tell from the lexical context.
    /// This evaluation context is used primary for the operand of the C++
    /// \c typeid expression, whose argument is potentially evaluated only when
    /// it is an lvalue of polymorphic class type (C++ [basic.def.odr]p2).
    PotentiallyPotentiallyEvaluated
  };

  /// \brief The parser is entering a new expression evaluation context.
  ///
  /// \param NewContext is the new expression evaluation context.
  ///
  /// \returns the previous expression evaluation context.
  virtual ExpressionEvaluationContext
  PushExpressionEvaluationContext(ExpressionEvaluationContext NewContext) {
    return PotentiallyEvaluated;
  }

  /// \brief The parser is existing an expression evaluation context.
  ///
  /// \param OldContext the expression evaluation context that the parser is
  /// leaving.
  ///
  /// \param NewContext the expression evaluation context that the parser is
  /// returning to.
  virtual void
  PopExpressionEvaluationContext(ExpressionEvaluationContext OldContext,
                                 ExpressionEvaluationContext NewContext) { }

  // Primary Expressions.

  /// \brief Retrieve the source range that corresponds to the given
  /// expression.
  virtual SourceRange getExprRange(ExprTy *E) const {
    return SourceRange();
  }

  /// ActOnIdentifierExpr - Parse an identifier in expression context.
  /// 'HasTrailingLParen' indicates whether or not the identifier has a '('
  /// token immediately after it.
  /// An optional CXXScopeSpec can be passed to indicate the C++ scope (class or
  /// namespace) that the identifier must be a member of.
  /// i.e. for "foo::bar", 'II' will be "bar" and 'SS' will be "foo::".
  virtual OwningExprResult ActOnIdentifierExpr(Scope *S, SourceLocation Loc,
                                               IdentifierInfo &II,
                                               bool HasTrailingLParen,
                                               const CXXScopeSpec *SS = 0,
                                               bool isAddressOfOperand = false){
    return ExprEmpty();
  }

  /// ActOnOperatorFunctionIdExpr - Parse a C++ overloaded operator
  /// name (e.g., @c operator+ ) as an expression. This is very
  /// similar to ActOnIdentifierExpr, except that instead of providing
  /// an identifier the parser provides the kind of overloaded
  /// operator that was parsed.
  virtual OwningExprResult ActOnCXXOperatorFunctionIdExpr(
                             Scope *S, SourceLocation OperatorLoc,
                             OverloadedOperatorKind Op,
                             bool HasTrailingLParen, const CXXScopeSpec &SS,
                             bool isAddressOfOperand = false) {
    return ExprEmpty();
  }

  /// ActOnCXXConversionFunctionExpr - Parse a C++ conversion function
  /// name (e.g., @c operator void const *) as an expression. This is
  /// very similar to ActOnIdentifierExpr, except that instead of
  /// providing an identifier the parser provides the type of the
  /// conversion function.
  virtual OwningExprResult ActOnCXXConversionFunctionExpr(
                             Scope *S, SourceLocation OperatorLoc,
                             TypeTy *Type, bool HasTrailingLParen,
                             const CXXScopeSpec &SS,
                             bool isAddressOfOperand = false) {
    return ExprEmpty();
  }

  virtual OwningExprResult ActOnPredefinedExpr(SourceLocation Loc,
                                               tok::TokenKind Kind) {
    return ExprEmpty();
  }
  virtual OwningExprResult ActOnCharacterConstant(const Token &) {
    return ExprEmpty();
  }
  virtual OwningExprResult ActOnNumericConstant(const Token &) {
    return ExprEmpty();
  }

  /// ActOnStringLiteral - The specified tokens were lexed as pasted string
  /// fragments (e.g. "foo" "bar" L"baz").
  virtual OwningExprResult ActOnStringLiteral(const Token *Toks,
                                              unsigned NumToks) {
    return ExprEmpty();
  }

  virtual OwningExprResult ActOnParenExpr(SourceLocation L, SourceLocation R,
                                          ExprArg Val) {
    return move(Val);  // Default impl returns operand.
  }

  virtual OwningExprResult ActOnParenListExpr(SourceLocation L,
                                              SourceLocation R,
                                              MultiExprArg Val) {
    return ExprEmpty();
  }

  // Postfix Expressions.
  virtual OwningExprResult ActOnPostfixUnaryOp(Scope *S, SourceLocation OpLoc,
                                               tok::TokenKind Kind,
                                               ExprArg Input) {
    return ExprEmpty();
  }
  virtual OwningExprResult ActOnArraySubscriptExpr(Scope *S, ExprArg Base,
                                                   SourceLocation LLoc,
                                                   ExprArg Idx,
                                                   SourceLocation RLoc) {
    return ExprEmpty();
  }
  virtual OwningExprResult ActOnMemberReferenceExpr(Scope *S, ExprArg Base,
                                                    SourceLocation OpLoc,
                                                    tok::TokenKind OpKind,
                                                    SourceLocation MemberLoc,
                                                    IdentifierInfo &Member,
                                                    DeclPtrTy ObjCImpDecl,
                                                const CXXScopeSpec *SS = 0) {
    return ExprEmpty();
  }

  /// ActOnCallExpr - Handle a call to Fn with the specified array of arguments.
  /// This provides the location of the left/right parens and a list of comma
  /// locations.  There are guaranteed to be one fewer commas than arguments,
  /// unless there are zero arguments.
  virtual OwningExprResult ActOnCallExpr(Scope *S, ExprArg Fn,
                                         SourceLocation LParenLoc,
                                         MultiExprArg Args,
                                         SourceLocation *CommaLocs,
                                         SourceLocation RParenLoc) {
    return ExprEmpty();
  }

  // Unary Operators.  'Tok' is the token for the operator.
  virtual OwningExprResult ActOnUnaryOp(Scope *S, SourceLocation OpLoc,
                                        tok::TokenKind Op, ExprArg Input) {
    return ExprEmpty();
  }
  virtual OwningExprResult
    ActOnSizeOfAlignOfExpr(SourceLocation OpLoc, bool isSizeof, bool isType,
                           void *TyOrEx, const SourceRange &ArgRange) {
    return ExprEmpty();
  }

  virtual OwningExprResult ActOnCompoundLiteral(SourceLocation LParen,
                                                TypeTy *Ty,
                                                SourceLocation RParen,
                                                ExprArg Op) {
    return ExprEmpty();
  }
  virtual OwningExprResult ActOnInitList(SourceLocation LParenLoc,
                                         MultiExprArg InitList,
                                         SourceLocation RParenLoc) {
    return ExprEmpty();
  }
  /// @brief Parsed a C99 designated initializer.
  ///
  /// @param Desig Contains the designation with one or more designators.
  ///
  /// @param Loc The location of the '=' or ':' prior to the
  /// initialization expression.
  ///
  /// @param GNUSyntax If true, then this designated initializer used
  /// the deprecated GNU syntax @c fieldname:foo or @c [expr]foo rather
  /// than the C99 syntax @c .fieldname=foo or @c [expr]=foo.
  ///
  /// @param Init The value that the entity (or entities) described by
  /// the designation will be initialized with.
  virtual OwningExprResult ActOnDesignatedInitializer(Designation &Desig,
                                                      SourceLocation Loc,
                                                      bool GNUSyntax,
                                                      OwningExprResult Init) {
    return ExprEmpty();
  }

  virtual OwningExprResult ActOnCastExpr(Scope *S, SourceLocation LParenLoc,
                                         TypeTy *Ty, SourceLocation RParenLoc,
                                         ExprArg Op) {
    return ExprEmpty();
  }

  virtual OwningExprResult ActOnBinOp(Scope *S, SourceLocation TokLoc,
                                      tok::TokenKind Kind,
                                      ExprArg LHS, ExprArg RHS) {
    return ExprEmpty();
  }

  /// ActOnConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
  /// in the case of a the GNU conditional expr extension.
  virtual OwningExprResult ActOnConditionalOp(SourceLocation QuestionLoc,
                                              SourceLocation ColonLoc,
                                              ExprArg Cond, ExprArg LHS,
                                              ExprArg RHS) {
    return ExprEmpty();
  }

  //===---------------------- GNU Extension Expressions -------------------===//

  virtual OwningExprResult ActOnAddrLabel(SourceLocation OpLoc,
                                          SourceLocation LabLoc,
                                          IdentifierInfo *LabelII) { // "&&foo"
    return ExprEmpty();
  }

  virtual OwningExprResult ActOnStmtExpr(SourceLocation LPLoc, StmtArg SubStmt,
                                         SourceLocation RPLoc) { // "({..})"
    return ExprEmpty();
  }

  // __builtin_offsetof(type, identifier(.identifier|[expr])*)
  struct OffsetOfComponent {
    SourceLocation LocStart, LocEnd;
    bool isBrackets;  // true if [expr], false if .ident
    union {
      IdentifierInfo *IdentInfo;
      ExprTy *E;
    } U;
  };

  virtual OwningExprResult ActOnBuiltinOffsetOf(Scope *S,
                                                SourceLocation BuiltinLoc,
                                                SourceLocation TypeLoc,
                                                TypeTy *Arg1,
                                                OffsetOfComponent *CompPtr,
                                                unsigned NumComponents,
                                                SourceLocation RParenLoc) {
    return ExprEmpty();
  }

  // __builtin_types_compatible_p(type1, type2)
  virtual OwningExprResult ActOnTypesCompatibleExpr(SourceLocation BuiltinLoc,
                                                    TypeTy *arg1, TypeTy *arg2,
                                                    SourceLocation RPLoc) {
    return ExprEmpty();
  }
  // __builtin_choose_expr(constExpr, expr1, expr2)
  virtual OwningExprResult ActOnChooseExpr(SourceLocation BuiltinLoc,
                                           ExprArg cond, ExprArg expr1,
                                           ExprArg expr2, SourceLocation RPLoc){
    return ExprEmpty();
  }

  // __builtin_va_arg(expr, type)
  virtual OwningExprResult ActOnVAArg(SourceLocation BuiltinLoc,
                                      ExprArg expr, TypeTy *type,
                                      SourceLocation RPLoc) {
    return ExprEmpty();
  }

  /// ActOnGNUNullExpr - Parsed the GNU __null expression, the token
  /// for which is at position TokenLoc.
  virtual OwningExprResult ActOnGNUNullExpr(SourceLocation TokenLoc) {
    return ExprEmpty();
  }

  //===------------------------- "Block" Extension ------------------------===//

  /// ActOnBlockStart - This callback is invoked when a block literal is
  /// started.  The result pointer is passed into the block finalizers.
  virtual void ActOnBlockStart(SourceLocation CaretLoc, Scope *CurScope) {}

  /// ActOnBlockArguments - This callback allows processing of block arguments.
  /// If there are no arguments, this is still invoked.
  virtual void ActOnBlockArguments(Declarator &ParamInfo, Scope *CurScope) {}

  /// ActOnBlockError - If there is an error parsing a block, this callback
  /// is invoked to pop the information about the block from the action impl.
  virtual void ActOnBlockError(SourceLocation CaretLoc, Scope *CurScope) {}

  /// ActOnBlockStmtExpr - This is called when the body of a block statement
  /// literal was successfully completed.  ^(int x){...}
  virtual OwningExprResult ActOnBlockStmtExpr(SourceLocation CaretLoc,
                                              StmtArg Body,
                                              Scope *CurScope) {
    return ExprEmpty();
  }

  //===------------------------- C++ Declarations -------------------------===//

  /// ActOnStartNamespaceDef - This is called at the start of a namespace
  /// definition.
  virtual DeclPtrTy ActOnStartNamespaceDef(Scope *S, SourceLocation IdentLoc,
                                           IdentifierInfo *Ident,
                                           SourceLocation LBrace) {
    return DeclPtrTy();
  }

  /// ActOnFinishNamespaceDef - This callback is called after a namespace is
  /// exited. Decl is returned by ActOnStartNamespaceDef.
  virtual void ActOnFinishNamespaceDef(DeclPtrTy Dcl, SourceLocation RBrace) {
    return;
  }

  /// ActOnUsingDirective - This is called when using-directive is parsed.
  virtual DeclPtrTy ActOnUsingDirective(Scope *CurScope,
                                        SourceLocation UsingLoc,
                                        SourceLocation NamespcLoc,
                                        const CXXScopeSpec &SS,
                                        SourceLocation IdentLoc,
                                        IdentifierInfo *NamespcName,
                                        AttributeList *AttrList);

  /// ActOnNamespaceAliasDef - This is called when a namespace alias definition
  /// is parsed.
  virtual DeclPtrTy ActOnNamespaceAliasDef(Scope *CurScope,
                                           SourceLocation NamespaceLoc,
                                           SourceLocation AliasLoc,
                                           IdentifierInfo *Alias,
                                           const CXXScopeSpec &SS,
                                           SourceLocation IdentLoc,
                                           IdentifierInfo *Ident) {
    return DeclPtrTy();
  }

  /// ActOnUsingDirective - This is called when using-directive is parsed.
  virtual DeclPtrTy ActOnUsingDeclaration(Scope *CurScope,
                                          AccessSpecifier AS,
                                          SourceLocation UsingLoc,
                                          const CXXScopeSpec &SS,
                                          SourceLocation IdentLoc,
                                          IdentifierInfo *TargetName,
                                          OverloadedOperatorKind Op,
                                          AttributeList *AttrList,
                                          bool IsTypeName);

  /// ActOnParamDefaultArgument - Parse default argument for function parameter
  virtual void ActOnParamDefaultArgument(DeclPtrTy param,
                                         SourceLocation EqualLoc,
                                         ExprArg defarg) {
  }

  /// ActOnParamUnparsedDefaultArgument - We've seen a default
  /// argument for a function parameter, but we can't parse it yet
  /// because we're inside a class definition. Note that this default
  /// argument will be parsed later.
  virtual void ActOnParamUnparsedDefaultArgument(DeclPtrTy param,
                                                 SourceLocation EqualLoc,
                                                 SourceLocation ArgLoc) { }

  /// ActOnParamDefaultArgumentError - Parsing or semantic analysis of
  /// the default argument for the parameter param failed.
  virtual void ActOnParamDefaultArgumentError(DeclPtrTy param) { }

  /// AddCXXDirectInitializerToDecl - This action is called immediately after
  /// ActOnDeclarator, when a C++ direct initializer is present.
  /// e.g: "int x(1);"
  virtual void AddCXXDirectInitializerToDecl(DeclPtrTy Dcl,
                                             SourceLocation LParenLoc,
                                             MultiExprArg Exprs,
                                             SourceLocation *CommaLocs,
                                             SourceLocation RParenLoc) {
    return;
  }

  /// \brief Called when we re-enter a template parameter scope.
  ///
  /// This action occurs when we are going to parse an member
  /// function's default arguments or inline definition after the
  /// outermost class definition has been completed, and when one or
  /// more of the class definitions enclosing the member function is a
  /// template. The "entity" in the given scope will be set as it was
  /// when we entered the scope of the template initially, and should
  /// be used to, e.g., reintroduce the names of template parameters
  /// into the current scope so that they can be found by name lookup.
  ///
  /// \param S The (new) template parameter scope.
  ///
  /// \param Template the class template declaration whose template
  /// parameters should be reintroduced into the current scope.
  virtual void ActOnReenterTemplateScope(Scope *S, DeclPtrTy Template) {
  }

  /// ActOnStartDelayedCXXMethodDeclaration - We have completed
  /// parsing a top-level (non-nested) C++ class, and we are now
  /// parsing those parts of the given Method declaration that could
  /// not be parsed earlier (C++ [class.mem]p2), such as default
  /// arguments. This action should enter the scope of the given
  /// Method declaration as if we had just parsed the qualified method
  /// name. However, it should not bring the parameters into scope;
  /// that will be performed by ActOnDelayedCXXMethodParameter.
  virtual void ActOnStartDelayedCXXMethodDeclaration(Scope *S,
                                                     DeclPtrTy Method) {
  }

  /// ActOnDelayedCXXMethodParameter - We've already started a delayed
  /// C++ method declaration. We're (re-)introducing the given
  /// function parameter into scope for use in parsing later parts of
  /// the method declaration. For example, we could see an
  /// ActOnParamDefaultArgument event for this parameter.
  virtual void ActOnDelayedCXXMethodParameter(Scope *S, DeclPtrTy Param) {
  }

  /// ActOnFinishDelayedCXXMethodDeclaration - We have finished
  /// processing the delayed method declaration for Method. The method
  /// declaration is now considered finished. There may be a separate
  /// ActOnStartOfFunctionDef action later (not necessarily
  /// immediately!) for this method, if it was also defined inside the
  /// class body.
  virtual void ActOnFinishDelayedCXXMethodDeclaration(Scope *S,
                                                      DeclPtrTy Method) {
  }

  /// ActOnStaticAssertDeclaration - Parse a C++0x static_assert declaration.
  virtual DeclPtrTy ActOnStaticAssertDeclaration(SourceLocation AssertLoc,
                                                 ExprArg AssertExpr,
                                                 ExprArg AssertMessageExpr) {
    return DeclPtrTy();
  }

  /// ActOnFriendFunctionDecl - Parsed a friend function declarator.
  /// The name is actually a slight misnomer, because the declarator
  /// is not necessarily a function declarator.
  virtual DeclPtrTy ActOnFriendFunctionDecl(Scope *S,
                                            Declarator &D,
                                            bool IsDefinition,
                                            MultiTemplateParamsArg TParams) {
    return DeclPtrTy();
  }

  /// ActOnFriendTypeDecl - Parsed a friend type declaration.
  virtual DeclPtrTy ActOnFriendTypeDecl(Scope *S,
                                        const DeclSpec &DS,
                                        MultiTemplateParamsArg TParams) {
    return DeclPtrTy();
  }

  //===------------------------- C++ Expressions --------------------------===//

  /// ActOnCXXNamedCast - Parse {dynamic,static,reinterpret,const}_cast's.
  virtual OwningExprResult ActOnCXXNamedCast(SourceLocation OpLoc,
                                             tok::TokenKind Kind,
                                             SourceLocation LAngleBracketLoc,
                                             TypeTy *Ty,
                                             SourceLocation RAngleBracketLoc,
                                             SourceLocation LParenLoc,
                                             ExprArg Op,
                                             SourceLocation RParenLoc) {
    return ExprEmpty();
  }

  /// ActOnCXXTypeidOfType - Parse typeid( type-id ).
  virtual OwningExprResult ActOnCXXTypeid(SourceLocation OpLoc,
                                          SourceLocation LParenLoc, bool isType,
                                          void *TyOrExpr,
                                          SourceLocation RParenLoc) {
    return ExprEmpty();
  }

  /// ActOnCXXThis - Parse the C++ 'this' pointer.
  virtual OwningExprResult ActOnCXXThis(SourceLocation ThisLoc) {
    return ExprEmpty();
  }

  /// ActOnCXXBoolLiteral - Parse {true,false} literals.
  virtual OwningExprResult ActOnCXXBoolLiteral(SourceLocation OpLoc,
                                               tok::TokenKind Kind) {
    return ExprEmpty();
  }

  /// ActOnCXXNullPtrLiteral - Parse 'nullptr'.
  virtual OwningExprResult ActOnCXXNullPtrLiteral(SourceLocation Loc) {
    return ExprEmpty();
  }

  /// ActOnCXXThrow - Parse throw expressions.
  virtual OwningExprResult ActOnCXXThrow(SourceLocation OpLoc, ExprArg Op) {
    return ExprEmpty();
  }

  /// ActOnCXXTypeConstructExpr - Parse construction of a specified type.
  /// Can be interpreted either as function-style casting ("int(x)")
  /// or class type construction ("ClassType(x,y,z)")
  /// or creation of a value-initialized type ("int()").
  virtual OwningExprResult ActOnCXXTypeConstructExpr(SourceRange TypeRange,
                                                     TypeTy *TypeRep,
                                                     SourceLocation LParenLoc,
                                                     MultiExprArg Exprs,
                                                     SourceLocation *CommaLocs,
                                                     SourceLocation RParenLoc) {
    return ExprEmpty();
  }

  /// ActOnCXXConditionDeclarationExpr - Parsed a condition declaration of a
  /// C++ if/switch/while/for statement.
  /// e.g: "if (int x = f()) {...}"
  virtual OwningExprResult ActOnCXXConditionDeclarationExpr(Scope *S,
                                                      SourceLocation StartLoc,
                                                      Declarator &D,
                                                      SourceLocation EqualLoc,
                                                      ExprArg AssignExprVal) {
    return ExprEmpty();
  }

  /// ActOnCXXNew - Parsed a C++ 'new' expression. UseGlobal is true if the
  /// new was qualified (::new). In a full new like
  /// @code new (p1, p2) type(c1, c2) @endcode
  /// the p1 and p2 expressions will be in PlacementArgs and the c1 and c2
  /// expressions in ConstructorArgs. The type is passed as a declarator.
  virtual OwningExprResult ActOnCXXNew(SourceLocation StartLoc, bool UseGlobal,
                                       SourceLocation PlacementLParen,
                                       MultiExprArg PlacementArgs,
                                       SourceLocation PlacementRParen,
                                       bool ParenTypeId, Declarator &D,
                                       SourceLocation ConstructorLParen,
                                       MultiExprArg ConstructorArgs,
                                       SourceLocation ConstructorRParen) {
    return ExprEmpty();
  }

  /// ActOnCXXDelete - Parsed a C++ 'delete' expression. UseGlobal is true if
  /// the delete was qualified (::delete). ArrayForm is true if the array form
  /// was used (delete[]).
  virtual OwningExprResult ActOnCXXDelete(SourceLocation StartLoc,
                                          bool UseGlobal, bool ArrayForm,
                                          ExprArg Operand) {
    return ExprEmpty();
  }

  virtual OwningExprResult ActOnUnaryTypeTrait(UnaryTypeTrait OTT,
                                               SourceLocation KWLoc,
                                               SourceLocation LParen,
                                               TypeTy *Ty,
                                               SourceLocation RParen) {
    return ExprEmpty();
  }

  /// \brief Invoked when the parser is starting to parse a C++ member access
  /// expression such as x.f or x->f.
  ///
  /// \param S the scope in which the member access expression occurs.
  ///
  /// \param Base the expression in which a member is being accessed, e.g., the
  /// "x" in "x.f".
  ///
  /// \param OpLoc the location of the member access operator ("." or "->")
  ///
  /// \param OpKind the kind of member access operator ("." or "->")
  ///
  /// \param ObjectType originally NULL. The action should fill in this type
  /// with the type into which name lookup should look to find the member in
  /// the member access expression.
  ///
  /// \returns the (possibly modified) \p Base expression
  virtual OwningExprResult ActOnStartCXXMemberReference(Scope *S,
                                                        ExprArg Base,
                                                        SourceLocation OpLoc,
                                                        tok::TokenKind OpKind,
                                                        TypeTy *&ObjectType) {
    return ExprEmpty();
  }

  /// ActOnDestructorReferenceExpr - Parsed a destructor reference, for example:
  ///
  /// t->~T();
  virtual OwningExprResult
  ActOnDestructorReferenceExpr(Scope *S, ExprArg Base,
                               SourceLocation OpLoc,
                               tok::TokenKind OpKind,
                               SourceLocation ClassNameLoc,
                               IdentifierInfo *ClassName,
                               const CXXScopeSpec &SS,
                               bool HasTrailingLParen) {
    return ExprEmpty();
  }

  /// ActOnOverloadedOperatorReferenceExpr - Parsed an overloaded operator
  /// reference, for example:
  ///
  /// t.operator++();
  virtual OwningExprResult
  ActOnOverloadedOperatorReferenceExpr(Scope *S, ExprArg Base,
                                       SourceLocation OpLoc,
                                       tok::TokenKind OpKind,
                                       SourceLocation ClassNameLoc,
                                       OverloadedOperatorKind OverOpKind,
                                       const CXXScopeSpec *SS = 0) {
    return ExprEmpty();
  }

  /// ActOnConversionOperatorReferenceExpr - Parsed an overloaded conversion
  /// function reference, for example:
  ///
  /// t.operator int();
  virtual OwningExprResult
  ActOnConversionOperatorReferenceExpr(Scope *S, ExprArg Base,
                                       SourceLocation OpLoc,
                                       tok::TokenKind OpKind,
                                       SourceLocation ClassNameLoc,
                                       TypeTy *Ty,
                                       const CXXScopeSpec *SS = 0) {
    return ExprEmpty();
  }

  /// \brief Parsed a reference to a member template-id.
  ///
  /// This callback will occur instead of ActOnMemberReferenceExpr() when the
  /// member in question is a template for which the code provides an
  /// explicitly-specified template argument list, e.g.,
  ///
  /// \code
  /// x.f<int>()
  /// \endcode
  ///
  /// \param S the scope in which the member reference expression occurs
  ///
  /// \param Base the expression to the left of the "." or "->".
  ///
  /// \param OpLoc the location of the "." or "->".
  ///
  /// \param OpKind the kind of operator, which will be "." or "->".
  ///
  /// \param SS the scope specifier that precedes the template-id in, e.g.,
  /// \c x.Base::f<int>().
  ///
  /// \param Template the declaration of the template that is being referenced.
  ///
  /// \param TemplateNameLoc the location of the template name referred to by
  /// \p Template.
  ///
  /// \param LAngleLoc the location of the left angle bracket ('<')
  ///
  /// \param TemplateArgs the (possibly-empty) template argument list provided
  /// as part of the member reference.
  ///
  /// \param RAngleLoc the location of the right angle bracket ('>')
  virtual OwningExprResult
  ActOnMemberTemplateIdReferenceExpr(Scope *S, ExprArg Base,
                                     SourceLocation OpLoc,
                                     tok::TokenKind OpKind,
                                     const CXXScopeSpec &SS,
                                     // FIXME: "template" keyword?
                                     TemplateTy Template,
                                     SourceLocation TemplateNameLoc,
                                     SourceLocation LAngleLoc,
                                     ASTTemplateArgsPtr TemplateArgs,
                                     SourceLocation *TemplateArgLocs,
                                     SourceLocation RAngleLoc) {
    return ExprEmpty();
  }

  /// ActOnFinishFullExpr - Called whenever a full expression has been parsed.
  /// (C++ [intro.execution]p12).
  virtual OwningExprResult ActOnFinishFullExpr(ExprArg Expr) {
    return move(Expr);
  }

  //===---------------------------- C++ Classes ---------------------------===//
  /// ActOnBaseSpecifier - Parsed a base specifier
  virtual BaseResult ActOnBaseSpecifier(DeclPtrTy classdecl,
                                        SourceRange SpecifierRange,
                                        bool Virtual, AccessSpecifier Access,
                                        TypeTy *basetype,
                                        SourceLocation BaseLoc) {
    return BaseResult();
  }

  virtual void ActOnBaseSpecifiers(DeclPtrTy ClassDecl, BaseTy **Bases,
                                   unsigned NumBases) {
  }

  /// ActOnCXXMemberDeclarator - This is invoked when a C++ class member
  /// declarator is parsed. 'AS' is the access specifier, 'BitfieldWidth'
  /// specifies the bitfield width if there is one and 'Init' specifies the
  /// initializer if any.  'Deleted' is true if there's a =delete
  /// specifier on the function.
  virtual DeclPtrTy ActOnCXXMemberDeclarator(Scope *S, AccessSpecifier AS,
                                             Declarator &D,
                                 MultiTemplateParamsArg TemplateParameterLists,
                                             ExprTy *BitfieldWidth,
                                             ExprTy *Init,
                                             bool Deleted = false) {
    return DeclPtrTy();
  }

  virtual MemInitResult ActOnMemInitializer(DeclPtrTy ConstructorDecl,
                                            Scope *S,
                                            const CXXScopeSpec &SS,
                                            IdentifierInfo *MemberOrBase,
                                            TypeTy *TemplateTypeTy,
                                            SourceLocation IdLoc,
                                            SourceLocation LParenLoc,
                                            ExprTy **Args, unsigned NumArgs,
                                            SourceLocation *CommaLocs,
                                            SourceLocation RParenLoc) {
    return true;
  }

  /// ActOnMemInitializers - This is invoked when all of the member
  /// initializers of a constructor have been parsed. ConstructorDecl
  /// is the function declaration (which will be a C++ constructor in
  /// a well-formed program), ColonLoc is the location of the ':' that
  /// starts the constructor initializer, and MemInit/NumMemInits
  /// contains the individual member (and base) initializers.
  virtual void ActOnMemInitializers(DeclPtrTy ConstructorDecl,
                                    SourceLocation ColonLoc,
                                    MemInitTy **MemInits, unsigned NumMemInits){
  }

 virtual void ActOnDefaultCtorInitializers(DeclPtrTy CDtorDecl) {}

  /// ActOnFinishCXXMemberSpecification - Invoked after all member declarators
  /// are parsed but *before* parsing of inline method definitions.
  virtual void ActOnFinishCXXMemberSpecification(Scope* S, SourceLocation RLoc,
                                                 DeclPtrTy TagDecl,
                                                 SourceLocation LBrac,
                                                 SourceLocation RBrac) {
  }

  //===---------------------------C++ Templates----------------------------===//

  /// ActOnTypeParameter - Called when a C++ template type parameter
  /// (e.g., "typename T") has been parsed. Typename specifies whether
  /// the keyword "typename" was used to declare the type parameter
  /// (otherwise, "class" was used), ellipsis specifies whether this is a
  /// C++0x parameter pack, EllipsisLoc specifies the start of the ellipsis,
  /// and KeyLoc is the location of the "class" or "typename" keyword.
  //  ParamName is the name of the parameter (NULL indicates an unnamed template
  //  parameter) and ParamNameLoc is the location of the parameter name (if any)
  /// If the type parameter has a default argument, it will be added
  /// later via ActOnTypeParameterDefault. Depth and Position provide
  /// the number of enclosing templates (see
  /// ActOnTemplateParameterList) and the number of previous
  /// parameters within this template parameter list.
  virtual DeclPtrTy ActOnTypeParameter(Scope *S, bool Typename, bool Ellipsis,
                                       SourceLocation EllipsisLoc,
                                       SourceLocation KeyLoc,
                                       IdentifierInfo *ParamName,
                                       SourceLocation ParamNameLoc,
                                       unsigned Depth, unsigned Position) {
    return DeclPtrTy();
  }

  /// ActOnTypeParameterDefault - Adds a default argument (the type
  /// Default) to the given template type parameter (TypeParam).
  virtual void ActOnTypeParameterDefault(DeclPtrTy TypeParam,
                                         SourceLocation EqualLoc,
                                         SourceLocation DefaultLoc,
                                         TypeTy *Default) {
  }

  /// ActOnNonTypeTemplateParameter - Called when a C++ non-type
  /// template parameter (e.g., "int Size" in "template<int Size>
  /// class Array") has been parsed. S is the current scope and D is
  /// the parsed declarator. Depth and Position provide the number of
  /// enclosing templates (see
  /// ActOnTemplateParameterList) and the number of previous
  /// parameters within this template parameter list.
  virtual DeclPtrTy ActOnNonTypeTemplateParameter(Scope *S, Declarator &D,
                                                  unsigned Depth,
                                                  unsigned Position) {
    return DeclPtrTy();
  }

  /// \brief Adds a default argument to the given non-type template
  /// parameter.
  virtual void ActOnNonTypeTemplateParameterDefault(DeclPtrTy TemplateParam,
                                                    SourceLocation EqualLoc,
                                                    ExprArg Default) {
  }

  /// ActOnTemplateTemplateParameter - Called when a C++ template template
  /// parameter (e.g., "int T" in "template<template <typename> class T> class
  /// Array") has been parsed. TmpLoc is the location of the "template" keyword,
  /// TemplateParams is the sequence of parameters required by the template,
  /// ParamName is the name of the parameter (null if unnamed), and ParamNameLoc
  /// is the source location of the identifier (if given).
  virtual DeclPtrTy ActOnTemplateTemplateParameter(Scope *S,
                                                   SourceLocation TmpLoc,
                                                   TemplateParamsTy *Params,
                                                   IdentifierInfo *ParamName,
                                                   SourceLocation ParamNameLoc,
                                                   unsigned Depth,
                                                   unsigned Position) {
    return DeclPtrTy();
  }

  /// \brief Adds a default argument to the given template template
  /// parameter.
  virtual void ActOnTemplateTemplateParameterDefault(DeclPtrTy TemplateParam,
                                                     SourceLocation EqualLoc,
                                                     ExprArg Default) {
  }

  /// ActOnTemplateParameterList - Called when a complete template
  /// parameter list has been parsed, e.g.,
  ///
  /// @code
  /// export template<typename T, T Size>
  /// @endcode
  ///
  /// Depth is the number of enclosing template parameter lists. This
  /// value does not include templates from outer scopes. For example:
  ///
  /// @code
  /// template<typename T> // depth = 0
  ///   class A {
  ///     template<typename U> // depth = 0
  ///       class B;
  ///   };
  ///
  /// template<typename T> // depth = 0
  ///   template<typename U> // depth = 1
  ///     class A<T>::B { ... };
  /// @endcode
  ///
  /// ExportLoc, if valid, is the position of the "export"
  /// keyword. Otherwise, "export" was not specified.
  /// TemplateLoc is the position of the template keyword, LAngleLoc
  /// is the position of the left angle bracket, and RAngleLoc is the
  /// position of the corresponding right angle bracket.
  /// Params/NumParams provides the template parameters that were
  /// parsed as part of the template-parameter-list.
  virtual TemplateParamsTy *
  ActOnTemplateParameterList(unsigned Depth,
                             SourceLocation ExportLoc,
                             SourceLocation TemplateLoc,
                             SourceLocation LAngleLoc,
                             DeclPtrTy *Params, unsigned NumParams,
                             SourceLocation RAngleLoc) {
    return 0;
  }

  /// \brief Form a type from a template and a list of template
  /// arguments.
  ///
  /// This action merely forms the type for the template-id, possibly
  /// checking well-formedness of the template arguments. It does not
  /// imply the declaration of any entity.
  ///
  /// \param Template  A template whose specialization results in a
  /// type, e.g., a class template or template template parameter.
  virtual TypeResult ActOnTemplateIdType(TemplateTy Template,
                                         SourceLocation TemplateLoc,
                                         SourceLocation LAngleLoc,
                                         ASTTemplateArgsPtr TemplateArgs,
                                         SourceLocation *TemplateArgLocs,
                                         SourceLocation RAngleLoc) {
    return TypeResult();
  };

  /// \brief Note that a template ID was used with a tag.
  ///
  /// \param Type The result of ActOnTemplateIdType.
  ///
  /// \param TUK Either TUK_Reference or TUK_Friend.  Declarations and
  /// definitions are interpreted as explicit instantiations or
  /// specializations.
  ///
  /// \param TagSpec The tag keyword that was provided as part of the
  /// elaborated-type-specifier;  either class, struct, union, or enum.
  ///
  /// \param TagLoc The location of the tag keyword.
  virtual TypeResult ActOnTagTemplateIdType(TypeResult Type,
                                            TagUseKind TUK,
                                            DeclSpec::TST TagSpec,
                                            SourceLocation TagLoc) {
    return TypeResult();
  }

  /// \brief Form a reference to a template-id (that will refer to a function)
  /// from a template and a list of template arguments.
  ///
  /// This action forms an expression that references the given template-id,
  /// possibly checking well-formedness of the template arguments. It does not
  /// imply the declaration of any entity.
  ///
  /// \param Template  A template whose specialization results in a
  /// function or a dependent template.
  virtual OwningExprResult ActOnTemplateIdExpr(TemplateTy Template,
                                               SourceLocation TemplateNameLoc,
                                               SourceLocation LAngleLoc,
                                               ASTTemplateArgsPtr TemplateArgs,
                                               SourceLocation *TemplateArgLocs,
                                               SourceLocation RAngleLoc) {
    return ExprError();
  }

  /// \brief Form a dependent template name.
  ///
  /// This action forms a dependent template name given the template
  /// name and its (presumably dependent) scope specifier. For
  /// example, given "MetaFun::template apply", the scope specifier \p
  /// SS will be "MetaFun::", \p TemplateKWLoc contains the location
  /// of the "template" keyword, and "apply" is the \p Name.
  ///
  /// \param TemplateKWLoc the location of the "template" keyword (if any).
  ///
  /// \param Name the name of the template (an identifier)
  ///
  /// \param NameLoc the location of the identifier
  ///
  /// \param SS the nested-name-specifier that precedes the "template" keyword
  /// or the template name. FIXME: If the dependent template name occurs in
  /// a member access expression, e.g., "x.template f<T>", this
  /// nested-name-specifier will be empty.
  ///
  /// \param ObjectType if this dependent template name occurs in the
  /// context of a member access expression, the type of the object being
  /// accessed.
  virtual TemplateTy ActOnDependentTemplateName(SourceLocation TemplateKWLoc,
                                                const IdentifierInfo &Name,
                                                SourceLocation NameLoc,
                                                const CXXScopeSpec &SS,
                                                TypeTy *ObjectType) {
    return TemplateTy();
  }

  /// \brief Process the declaration or definition of an explicit
  /// class template specialization or a class template partial
  /// specialization.
  ///
  /// This routine is invoked when an explicit class template
  /// specialization or a class template partial specialization is
  /// declared or defined, to introduce the (partial) specialization
  /// and produce a declaration for it. In the following example,
  /// ActOnClassTemplateSpecialization will be invoked for the
  /// declarations at both A and B:
  ///
  /// \code
  /// template<typename T> class X;
  /// template<> class X<int> { }; // A: explicit specialization
  /// template<typename T> class X<T*> { }; // B: partial specialization
  /// \endcode
  ///
  /// Note that it is the job of semantic analysis to determine which
  /// of the two cases actually occurred in the source code, since
  /// they are parsed through the same path. The formulation of the
  /// template parameter lists describes which case we are in.
  ///
  /// \param S the current scope
  ///
  /// \param TagSpec whether this declares a class, struct, or union
  /// (template)
  ///
  /// \param TUK whether this is a declaration or a definition
  ///
  /// \param KWLoc the location of the 'class', 'struct', or 'union'
  /// keyword.
  ///
  /// \param SS the scope specifier preceding the template-id
  ///
  /// \param Template the declaration of the class template that we
  /// are specializing.
  ///
  /// \param Attr attributes on the specialization
  ///
  /// \param TemplateParameterLists the set of template parameter
  /// lists that apply to this declaration. In a well-formed program,
  /// the number of template parameter lists will be one more than the
  /// number of template-ids in the scope specifier. However, it is
  /// common for users to provide the wrong number of template
  /// parameter lists (such as a missing \c template<> prior to a
  /// specialization); the parser does not check this condition.
  virtual DeclResult
  ActOnClassTemplateSpecialization(Scope *S, unsigned TagSpec, TagUseKind TUK,
                                   SourceLocation KWLoc,
                                   const CXXScopeSpec &SS,
                                   TemplateTy Template,
                                   SourceLocation TemplateNameLoc,
                                   SourceLocation LAngleLoc,
                                   ASTTemplateArgsPtr TemplateArgs,
                                   SourceLocation *TemplateArgLocs,
                                   SourceLocation RAngleLoc,
                                   AttributeList *Attr,
                              MultiTemplateParamsArg TemplateParameterLists) {
    return DeclResult();
  }

  /// \brief Invoked when a declarator that has one or more template parameter
  /// lists has been parsed.
  ///
  /// This action is similar to ActOnDeclarator(), except that the declaration
  /// being created somehow involves a template, e.g., it is a template
  /// declaration or specialization.
  virtual DeclPtrTy ActOnTemplateDeclarator(Scope *S,
                              MultiTemplateParamsArg TemplateParameterLists,
                                            Declarator &D) {
    return DeclPtrTy();
  }

  /// \brief Invoked when the parser is beginning to parse a function template
  /// or function template specialization definition.
  virtual DeclPtrTy ActOnStartOfFunctionTemplateDef(Scope *FnBodyScope,
                                MultiTemplateParamsArg TemplateParameterLists,
                                                    Declarator &D) {
    return DeclPtrTy();
  }

  /// \brief Process the explicit instantiation of a class template
  /// specialization.
  ///
  /// This routine is invoked when an explicit instantiation of a
  /// class template specialization is encountered. In the following
  /// example, ActOnExplicitInstantiation will be invoked to force the
  /// instantiation of X<int>:
  ///
  /// \code
  /// template<typename T> class X { /* ... */ };
  /// template class X<int>; // explicit instantiation
  /// \endcode
  ///
  /// \param S the current scope
  ///
  /// \param ExternLoc the location of the 'extern' keyword that specifies that
  /// this is an extern template (if any).
  ///
  /// \param TemplateLoc the location of the 'template' keyword that
  /// specifies that this is an explicit instantiation.
  ///
  /// \param TagSpec whether this declares a class, struct, or union
  /// (template).
  ///
  /// \param KWLoc the location of the 'class', 'struct', or 'union'
  /// keyword.
  ///
  /// \param SS the scope specifier preceding the template-id.
  ///
  /// \param Template the declaration of the class template that we
  /// are instantiation.
  ///
  /// \param LAngleLoc the location of the '<' token in the template-id.
  ///
  /// \param TemplateArgs the template arguments used to form the
  /// template-id.
  ///
  /// \param TemplateArgLocs the locations of the template arguments.
  ///
  /// \param RAngleLoc the location of the '>' token in the template-id.
  ///
  /// \param Attr attributes that apply to this instantiation.
  virtual DeclResult
  ActOnExplicitInstantiation(Scope *S,
                             SourceLocation ExternLoc,
                             SourceLocation TemplateLoc,
                             unsigned TagSpec,
                             SourceLocation KWLoc,
                             const CXXScopeSpec &SS,
                             TemplateTy Template,
                             SourceLocation TemplateNameLoc,
                             SourceLocation LAngleLoc,
                             ASTTemplateArgsPtr TemplateArgs,
                             SourceLocation *TemplateArgLocs,
                             SourceLocation RAngleLoc,
                             AttributeList *Attr) {
    return DeclResult();
  }

  /// \brief Process the explicit instantiation of a member class of a
  /// class template specialization.
  ///
  /// This routine is invoked when an explicit instantiation of a
  /// member class of a class template specialization is
  /// encountered. In the following example,
  /// ActOnExplicitInstantiation will be invoked to force the
  /// instantiation of X<int>::Inner:
  ///
  /// \code
  /// template<typename T> class X { class Inner { /* ... */}; };
  /// template class X<int>::Inner; // explicit instantiation
  /// \endcode
  ///
  /// \param S the current scope
  ///
  /// \param ExternLoc the location of the 'extern' keyword that specifies that
  /// this is an extern template (if any).
  ///
  /// \param TemplateLoc the location of the 'template' keyword that
  /// specifies that this is an explicit instantiation.
  ///
  /// \param TagSpec whether this declares a class, struct, or union
  /// (template).
  ///
  /// \param KWLoc the location of the 'class', 'struct', or 'union'
  /// keyword.
  ///
  /// \param SS the scope specifier preceding the template-id.
  ///
  /// \param Template the declaration of the class template that we
  /// are instantiation.
  ///
  /// \param LAngleLoc the location of the '<' token in the template-id.
  ///
  /// \param TemplateArgs the template arguments used to form the
  /// template-id.
  ///
  /// \param TemplateArgLocs the locations of the template arguments.
  ///
  /// \param RAngleLoc the location of the '>' token in the template-id.
  ///
  /// \param Attr attributes that apply to this instantiation.
  virtual DeclResult
  ActOnExplicitInstantiation(Scope *S,
                             SourceLocation ExternLoc,
                             SourceLocation TemplateLoc,
                             unsigned TagSpec,
                             SourceLocation KWLoc,
                             const CXXScopeSpec &SS,
                             IdentifierInfo *Name,
                             SourceLocation NameLoc,
                             AttributeList *Attr) {
    return DeclResult();
  }

  /// \brief Called when the parser has parsed a C++ typename
  /// specifier that ends in an identifier, e.g., "typename T::type".
  ///
  /// \param TypenameLoc the location of the 'typename' keyword
  /// \param SS the nested-name-specifier following the typename (e.g., 'T::').
  /// \param II the identifier we're retrieving (e.g., 'type' in the example).
  /// \param IdLoc the location of the identifier.
  virtual TypeResult
  ActOnTypenameType(SourceLocation TypenameLoc, const CXXScopeSpec &SS,
                    const IdentifierInfo &II, SourceLocation IdLoc) {
    return TypeResult();
  }

  /// \brief Called when the parser has parsed a C++ typename
  /// specifier that ends in a template-id, e.g.,
  /// "typename MetaFun::template apply<T1, T2>".
  ///
  /// \param TypenameLoc the location of the 'typename' keyword
  /// \param SS the nested-name-specifier following the typename (e.g., 'T::').
  /// \param TemplateLoc the location of the 'template' keyword, if any.
  /// \param Ty the type that the typename specifier refers to.
  virtual TypeResult
  ActOnTypenameType(SourceLocation TypenameLoc, const CXXScopeSpec &SS,
                    SourceLocation TemplateLoc, TypeTy *Ty) {
    return TypeResult();
  }

  //===----------------------- Obj-C Declarations -------------------------===//

  // ActOnStartClassInterface - this action is called immediately after parsing
  // the prologue for a class interface (before parsing the instance
  // variables). Instance variables are processed by ActOnFields().
  virtual DeclPtrTy ActOnStartClassInterface(SourceLocation AtInterfaceLoc,
                                             IdentifierInfo *ClassName,
                                             SourceLocation ClassLoc,
                                             IdentifierInfo *SuperName,
                                             SourceLocation SuperLoc,
                                             const DeclPtrTy *ProtoRefs,
                                             unsigned NumProtoRefs,
                                             SourceLocation EndProtoLoc,
                                             AttributeList *AttrList) {
    return DeclPtrTy();
  }

  /// ActOnCompatiblityAlias - this action is called after complete parsing of
  /// @compaatibility_alias declaration. It sets up the alias relationships.
  virtual DeclPtrTy ActOnCompatiblityAlias(
    SourceLocation AtCompatibilityAliasLoc,
    IdentifierInfo *AliasName,  SourceLocation AliasLocation,
    IdentifierInfo *ClassName, SourceLocation ClassLocation) {
    return DeclPtrTy();
  }

  // ActOnStartProtocolInterface - this action is called immdiately after
  // parsing the prologue for a protocol interface.
  virtual DeclPtrTy ActOnStartProtocolInterface(SourceLocation AtProtoLoc,
                                                IdentifierInfo *ProtocolName,
                                                SourceLocation ProtocolLoc,
                                                const DeclPtrTy *ProtoRefs,
                                                unsigned NumProtoRefs,
                                                SourceLocation EndProtoLoc,
                                                AttributeList *AttrList) {
    return DeclPtrTy();
  }
  // ActOnStartCategoryInterface - this action is called immdiately after
  // parsing the prologue for a category interface.
  virtual DeclPtrTy ActOnStartCategoryInterface(SourceLocation AtInterfaceLoc,
                                                IdentifierInfo *ClassName,
                                                SourceLocation ClassLoc,
                                                IdentifierInfo *CategoryName,
                                                SourceLocation CategoryLoc,
                                                const DeclPtrTy *ProtoRefs,
                                                unsigned NumProtoRefs,
                                                SourceLocation EndProtoLoc) {
    return DeclPtrTy();
  }
  // ActOnStartClassImplementation - this action is called immdiately after
  // parsing the prologue for a class implementation. Instance variables are
  // processed by ActOnFields().
  virtual DeclPtrTy ActOnStartClassImplementation(
    SourceLocation AtClassImplLoc,
    IdentifierInfo *ClassName,
    SourceLocation ClassLoc,
    IdentifierInfo *SuperClassname,
    SourceLocation SuperClassLoc) {
    return DeclPtrTy();
  }
  // ActOnStartCategoryImplementation - this action is called immdiately after
  // parsing the prologue for a category implementation.
  virtual DeclPtrTy ActOnStartCategoryImplementation(
    SourceLocation AtCatImplLoc,
    IdentifierInfo *ClassName,
    SourceLocation ClassLoc,
    IdentifierInfo *CatName,
    SourceLocation CatLoc) {
    return DeclPtrTy();
  }
  // ActOnPropertyImplDecl - called for every property implementation
  virtual DeclPtrTy ActOnPropertyImplDecl(
   SourceLocation AtLoc,              // location of the @synthesize/@dynamic
   SourceLocation PropertyNameLoc,    // location for the property name
   bool ImplKind,                     // true for @synthesize, false for
                                      // @dynamic
   DeclPtrTy ClassImplDecl,           // class or category implementation
   IdentifierInfo *propertyId,        // name of property
   IdentifierInfo *propertyIvar) {    // name of the ivar
    return DeclPtrTy();
  }

  struct ObjCArgInfo {
    IdentifierInfo *Name;
    SourceLocation NameLoc;
    // The Type is null if no type was specified, and the DeclSpec is invalid
    // in this case.
    TypeTy *Type;
    ObjCDeclSpec DeclSpec;

    /// ArgAttrs - Attribute list for this argument.
    AttributeList *ArgAttrs;
  };

  // ActOnMethodDeclaration - called for all method declarations.
  virtual DeclPtrTy ActOnMethodDeclaration(
    SourceLocation BeginLoc,   // location of the + or -.
    SourceLocation EndLoc,     // location of the ; or {.
    tok::TokenKind MethodType, // tok::minus for instance, tok::plus for class.
    DeclPtrTy ClassDecl,       // class this methods belongs to.
    ObjCDeclSpec &ReturnQT,    // for return type's in inout etc.
    TypeTy *ReturnType,        // the method return type.
    Selector Sel,              // a unique name for the method.
    ObjCArgInfo *ArgInfo,      // ArgInfo: Has 'Sel.getNumArgs()' entries.
    llvm::SmallVectorImpl<Declarator> &Cdecls, // c-style args
    AttributeList *MethodAttrList, // optional
    // tok::objc_not_keyword, tok::objc_optional, tok::objc_required
    tok::ObjCKeywordKind impKind,
    bool isVariadic = false) {
    return DeclPtrTy();
  }
  // ActOnAtEnd - called to mark the @end. For declarations (interfaces,
  // protocols, categories), the parser passes all methods/properties.
  // For class implementations, these values default to 0. For implementations,
  // methods are processed incrementally (by ActOnMethodDeclaration above).
  virtual void ActOnAtEnd(SourceLocation AtEndLoc,
                          DeclPtrTy classDecl,
                          DeclPtrTy *allMethods = 0,
                          unsigned allNum = 0,
                          DeclPtrTy *allProperties = 0,
                          unsigned pNum = 0,
                          DeclGroupPtrTy *allTUVars = 0,
                          unsigned tuvNum = 0) {
  }
  // ActOnProperty - called to build one property AST
  virtual DeclPtrTy ActOnProperty(Scope *S, SourceLocation AtLoc,
                                  FieldDeclarator &FD, ObjCDeclSpec &ODS,
                                  Selector GetterSel, Selector SetterSel,
                                  DeclPtrTy ClassCategory,
                                  bool *OverridingProperty,
                                  tok::ObjCKeywordKind MethodImplKind) {
    return DeclPtrTy();
  }

  virtual OwningExprResult ActOnClassPropertyRefExpr(
    IdentifierInfo &receiverName,
    IdentifierInfo &propertyName,
    SourceLocation &receiverNameLoc,
    SourceLocation &propertyNameLoc) {
    return ExprEmpty();
  }

  // ActOnClassMessage - used for both unary and keyword messages.
  // ArgExprs is optional - if it is present, the number of expressions
  // is obtained from NumArgs.
  virtual ExprResult ActOnClassMessage(
    Scope *S,
    IdentifierInfo *receivingClassName,
    Selector Sel,
    SourceLocation lbrac, SourceLocation receiverLoc,
    SourceLocation selectorLoc,
    SourceLocation rbrac,
    ExprTy **ArgExprs, unsigned NumArgs) {
    return ExprResult();
  }
  // ActOnInstanceMessage - used for both unary and keyword messages.
  // ArgExprs is optional - if it is present, the number of expressions
  // is obtained from NumArgs.
  virtual ExprResult ActOnInstanceMessage(
    ExprTy *receiver, Selector Sel,
    SourceLocation lbrac, SourceLocation selectorLoc, SourceLocation rbrac,
    ExprTy **ArgExprs, unsigned NumArgs) {
    return ExprResult();
  }
  virtual DeclPtrTy ActOnForwardClassDeclaration(
    SourceLocation AtClassLoc,
    IdentifierInfo **IdentList,
    unsigned NumElts) {
    return DeclPtrTy();
  }
  virtual DeclPtrTy ActOnForwardProtocolDeclaration(
    SourceLocation AtProtocolLoc,
    const IdentifierLocPair*IdentList,
    unsigned NumElts,
    AttributeList *AttrList) {
    return DeclPtrTy();
  }

  /// FindProtocolDeclaration - This routine looks up protocols and
  /// issues error if they are not declared. It returns list of valid
  /// protocols found.
  virtual void FindProtocolDeclaration(bool WarnOnDeclarations,
                                       const IdentifierLocPair *ProtocolId,
                                       unsigned NumProtocols,
                                 llvm::SmallVectorImpl<DeclPtrTy> &ResProtos) {
  }

  //===----------------------- Obj-C Expressions --------------------------===//

  virtual ExprResult ParseObjCStringLiteral(SourceLocation *AtLocs,
                                            ExprTy **Strings,
                                            unsigned NumStrings) {
    return ExprResult();
  }

  virtual ExprResult ParseObjCEncodeExpression(SourceLocation AtLoc,
                                               SourceLocation EncLoc,
                                               SourceLocation LParenLoc,
                                               TypeTy *Ty,
                                               SourceLocation RParenLoc) {
    return ExprResult();
  }

  virtual ExprResult ParseObjCSelectorExpression(Selector Sel,
                                                 SourceLocation AtLoc,
                                                 SourceLocation SelLoc,
                                                 SourceLocation LParenLoc,
                                                 SourceLocation RParenLoc) {
    return ExprResult();
  }

  virtual ExprResult ParseObjCProtocolExpression(IdentifierInfo *ProtocolId,
                                                 SourceLocation AtLoc,
                                                 SourceLocation ProtoLoc,
                                                 SourceLocation LParenLoc,
                                                 SourceLocation RParenLoc) {
    return ExprResult();
  }

  //===---------------------------- Pragmas -------------------------------===//

  enum PragmaPackKind {
    PPK_Default, // #pragma pack([n])
    PPK_Show,    // #pragma pack(show), only supported by MSVC.
    PPK_Push,    // #pragma pack(push, [identifier], [n])
    PPK_Pop      // #pragma pack(pop, [identifier], [n])
  };

  /// ActOnPragmaPack - Called on well formed #pragma pack(...).
  virtual void ActOnPragmaPack(PragmaPackKind Kind,
                               IdentifierInfo *Name,
                               ExprTy *Alignment,
                               SourceLocation PragmaLoc,
                               SourceLocation LParenLoc,
                               SourceLocation RParenLoc) {
    return;
  }

  /// ActOnPragmaUnused - Called on well formed #pragma unused(...).
  virtual void ActOnPragmaUnused(const Token *Identifiers,
                                 unsigned NumIdentifiers, Scope *CurScope,
                                 SourceLocation PragmaLoc,
                                 SourceLocation LParenLoc,
                                 SourceLocation RParenLoc) {
    return;
  }

  /// ActOnPragmaWeakID - Called on well formed #pragma weak ident.
  virtual void ActOnPragmaWeakID(IdentifierInfo* WeakName,
                                 SourceLocation PragmaLoc,
                                 SourceLocation WeakNameLoc) {
    return;
  }

  /// ActOnPragmaWeakAlias - Called on well formed #pragma weak ident = ident.
  virtual void ActOnPragmaWeakAlias(IdentifierInfo* WeakName,
                                    IdentifierInfo* AliasName,
                                    SourceLocation PragmaLoc,
                                    SourceLocation WeakNameLoc,
                                    SourceLocation AliasNameLoc) {
    return;
  }
};

/// MinimalAction - Minimal actions are used by light-weight clients of the
/// parser that do not need name resolution or significant semantic analysis to
/// be performed.  The actions implemented here are in the form of unresolved
/// identifiers.  By using a simpler interface than the SemanticAction class,
/// the parser doesn't have to build complex data structures and thus runs more
/// quickly.
class MinimalAction : public Action {
  /// Translation Unit Scope - useful to Objective-C actions that need
  /// to lookup file scope declarations in the "ordinary" C decl namespace.
  /// For example, user-defined classes, built-in "id" type, etc.
  Scope *TUScope;
  IdentifierTable &Idents;
  Preprocessor &PP;
  void *TypeNameInfoTablePtr;
public:
  MinimalAction(Preprocessor &pp);
  ~MinimalAction();

  /// getTypeName - This looks at the IdentifierInfo::FETokenInfo field to
  /// determine whether the name is a typedef or not in this scope.
  ///
  /// \param II the identifier for which we are performing name lookup
  ///
  /// \param NameLoc the location of the identifier
  ///
  /// \param S the scope in which this name lookup occurs
  ///
  /// \param SS if non-NULL, the C++ scope specifier that precedes the
  /// identifier
  ///
  /// \param isClassName whether this is a C++ class-name production, in
  /// which we can end up referring to a member of an unknown specialization
  /// that we know (from the grammar) is supposed to be a type. For example,
  /// this occurs when deriving from "std::vector<T>::allocator_type", where T
  /// is a template parameter.
  ///
  /// \returns the type referred to by this identifier, or NULL if the type
  /// does not name an identifier.
  virtual TypeTy *getTypeName(IdentifierInfo &II, SourceLocation NameLoc,
                              Scope *S, const CXXScopeSpec *SS,
                              bool isClassName = false);

  /// isCurrentClassName - Always returns false, because MinimalAction
  /// does not support C++ classes with constructors.
  virtual bool isCurrentClassName(const IdentifierInfo& II, Scope *S,
                                  const CXXScopeSpec *SS);

  virtual TemplateNameKind isTemplateName(Scope *S,
                                          const IdentifierInfo &II,
                                          SourceLocation IdLoc,
                                          const CXXScopeSpec *SS,
                                          TypeTy *ObjectType,
                                          bool EnteringContext,
                                          TemplateTy &Template);

  /// ActOnDeclarator - If this is a typedef declarator, we modify the
  /// IdentifierInfo::FETokenInfo field to keep track of this fact, until S is
  /// popped.
  virtual DeclPtrTy ActOnDeclarator(Scope *S, Declarator &D);

  /// ActOnPopScope - When a scope is popped, if any typedefs are now
  /// out-of-scope, they are removed from the IdentifierInfo::FETokenInfo field.
  virtual void ActOnPopScope(SourceLocation Loc, Scope *S);
  virtual void ActOnTranslationUnitScope(SourceLocation Loc, Scope *S);

  virtual DeclPtrTy ActOnForwardClassDeclaration(SourceLocation AtClassLoc,
                                                 IdentifierInfo **IdentList,
                                                 unsigned NumElts);

  virtual DeclPtrTy ActOnStartClassInterface(SourceLocation interLoc,
                                             IdentifierInfo *ClassName,
                                             SourceLocation ClassLoc,
                                             IdentifierInfo *SuperName,
                                             SourceLocation SuperLoc,
                                             const DeclPtrTy *ProtoRefs,
                                             unsigned NumProtoRefs,
                                             SourceLocation EndProtoLoc,
                                             AttributeList *AttrList);
};

/// PrettyStackTraceActionsDecl - If a crash occurs in the parser while parsing
/// something related to a virtualized decl, include that virtualized decl in
/// the stack trace.
class PrettyStackTraceActionsDecl : public llvm::PrettyStackTraceEntry {
  Action::DeclPtrTy TheDecl;
  SourceLocation Loc;
  Action &Actions;
  SourceManager &SM;
  const char *Message;
public:
  PrettyStackTraceActionsDecl(Action::DeclPtrTy Decl, SourceLocation L,
                              Action &actions, SourceManager &sm,
                              const char *Msg)
  : TheDecl(Decl), Loc(L), Actions(actions), SM(sm), Message(Msg) {}

  virtual void print(llvm::raw_ostream &OS) const;
};

/// \brief RAII object that enters a new expression evaluation context.
class EnterExpressionEvaluationContext {
  /// \brief The action object.
  Action &Actions;

  /// \brief The previous expression evaluation context.
  Action::ExpressionEvaluationContext PrevContext;

  /// \brief The current expression evaluation context.
  Action::ExpressionEvaluationContext CurContext;

public:
  EnterExpressionEvaluationContext(Action &Actions,
                              Action::ExpressionEvaluationContext NewContext)
    : Actions(Actions), CurContext(NewContext) {
      PrevContext = Actions.PushExpressionEvaluationContext(NewContext);
  }

  ~EnterExpressionEvaluationContext() {
    Actions.PopExpressionEvaluationContext(CurContext, PrevContext);
  }
};

}  // end namespace clang

#endif
