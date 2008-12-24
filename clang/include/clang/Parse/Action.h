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
#include "clang/Parse/AccessSpecifier.h"
#include "clang/Parse/Ownership.h"

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
  class InitListDesignations;
  // Lex.
  class Preprocessor;
  class Token;

/// Action - As the parser reads the input file and recognizes the productions
/// of the grammar, it invokes methods on this class to turn the parsed input
/// into something useful: e.g. a parse tree.
///
/// The callback methods that this class provides are phrased as actions that
/// the parser has just done or is about to do when the method is called.  They
/// are not requests that the actions module do the specified action.
///
/// All of the methods here are optional except isTypeName() and
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
  typedef void DeclTy;
  typedef void TypeTy;
  typedef void AttrTy;
  typedef void BaseTy;
  typedef void MemInitTy;
  typedef void CXXScopeTy;
  typedef void TemplateParamsTy;
  typedef void TemplateArgTy;

  /// Expr/Stmt/Type/BaseResult - Provide a unique type to wrap
  /// ExprTy/StmtTy/TypeTy/BaseTy, providing strong typing and
  /// allowing for failure.
  typedef ActionResult<0> ExprResult;
  typedef ActionResult<1> StmtResult;
  typedef ActionResult<2> TypeResult;
  typedef ActionResult<3> BaseResult;
  typedef ActionResult<4> MemInitResult;

  /// Same, but with ownership.
  typedef ASTOwningResult<&ActionBase::DeleteExpr> OwningExprResult;
  typedef ASTOwningResult<&ActionBase::DeleteStmt> OwningStmtResult;
  typedef ASTOwningResult<&ActionBase::DeleteTemplateArg> 
    OwningTemplateArgResult;
  // Note that these will replace ExprResult and StmtResult when the transition
  // is complete.

  /// Single expressions or statements as arguments.
  typedef ASTOwningPtr<&ActionBase::DeleteExpr> ExprArg;
  typedef ASTOwningPtr<&ActionBase::DeleteStmt> StmtArg;
  typedef ASTOwningPtr<&ActionBase::DeleteTemplateArg> TemplateArgArg;

  /// Multiple expressions or statements as arguments.
  typedef ASTMultiPtr<&ActionBase::DeleteExpr> MultiExprArg;
  typedef ASTMultiPtr<&ActionBase::DeleteStmt> MultiStmtArg;
  typedef ASTMultiPtr<&ActionBase::DeleteTemplateParams> MultiTemplateParamsArg;
  typedef ASTMultiPtr<&ActionBase::DeleteTemplateArg> MultiTemplateArgArg;

  // Utilities for Action implementations to return smart results.

  OwningExprResult ExprError() { return OwningExprResult(*this, true); }
  OwningStmtResult StmtError() { return OwningStmtResult(*this, true); }
  OwningTemplateArgResult TemplateArgError() { 
    return OwningTemplateArgResult(*this, true); 
  }

  OwningExprResult ExprError(const DiagnosticBuilder&) { return ExprError(); }
  OwningStmtResult StmtError(const DiagnosticBuilder&) { return StmtError(); }
  OwningTemplateArgResult TemplateArgError(const DiagnosticBuilder&) {
    return TemplateArgError();
  }

  OwningExprResult ExprEmpty() { return OwningExprResult(*this, false); }
  OwningStmtResult StmtEmpty() { return OwningStmtResult(*this, false); }
  OwningTemplateArgResult TemplateArgEmpty() { 
    return OwningTemplateArgResult(*this, false); 
  }

  /// Statistics.
  virtual void PrintStats() const {}
  //===--------------------------------------------------------------------===//
  // Declaration Tracking Callbacks.
  //===--------------------------------------------------------------------===//
  
  /// isTypeName - Return non-null if the specified identifier is a type name
  /// in the current scope.
  /// An optional CXXScopeSpec can be passed to indicate the C++ scope (class or
  /// namespace) that the identifier must be a member of.
  /// i.e. for "foo::bar", 'II' will be "bar" and 'SS' will be "foo::".
  virtual TypeTy *isTypeName(IdentifierInfo &II, Scope *S,
                             const CXXScopeSpec *SS = 0) = 0;

  /// isCurrentClassName - Return true if the specified name is the
  /// name of the innermost C++ class type currently being defined.
  virtual bool isCurrentClassName(const IdentifierInfo &II, Scope *S,
                                  const CXXScopeSpec *SS = 0) = 0;

  /// isTemplateName - Determines whether the identifier II is a
  /// template name in the current scope, and returns the template
  /// declaration if II names a template. An optional CXXScope can be
  /// passed to indicate the C++ scope in which the identifier will be
  /// found. 
  virtual DeclTy *isTemplateName(IdentifierInfo &II, Scope *S,
                                 const CXXScopeSpec *SS = 0) = 0;

  /// ActOnCXXGlobalScopeSpecifier - Return the object that represents the
  /// global scope ('::').
  virtual CXXScopeTy *ActOnCXXGlobalScopeSpecifier(Scope *S,
                                                   SourceLocation CCLoc) {
    return 0;
  }

  /// ActOnCXXNestedNameSpecifier - Called during parsing of a
  /// nested-name-specifier. e.g. for "foo::bar::" we parsed "foo::" and now
  /// we want to resolve "bar::". 'SS' is empty or the previously parsed
  /// nested-name part ("foo::"), 'IdLoc' is the source location of 'bar',
  /// 'CCLoc' is the location of '::' and 'II' is the identifier for 'bar'.
  /// Returns a CXXScopeTy* object representing the C++ scope.
  virtual CXXScopeTy *ActOnCXXNestedNameSpecifier(Scope *S,
                                                  const CXXScopeSpec &SS,
                                                  SourceLocation IdLoc,
                                                  SourceLocation CCLoc,
                                                  IdentifierInfo &II) {
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

  /// ActOnDeclarator - This callback is invoked when a declarator is parsed and
  /// 'Init' specifies the initializer if any.  This is for things like:
  /// "int X = 4" or "typedef int foo".
  ///
  /// LastInGroup is non-null for cases where one declspec has multiple
  /// declarators on it.  For example in 'int A, B', ActOnDeclarator will be
  /// called with LastInGroup=A when invoked for B.
  virtual DeclTy *ActOnDeclarator(Scope *S, Declarator &D,DeclTy *LastInGroup) {
    return 0;
  }

  /// ActOnParamDeclarator - This callback is invoked when a parameter
  /// declarator is parsed. This callback only occurs for functions
  /// with prototypes. S is the function prototype scope for the
  /// parameters (C++ [basic.scope.proto]).
  virtual DeclTy *ActOnParamDeclarator(Scope *S, Declarator &D) {
    return 0;
  }

  /// AddInitializerToDecl - This action is called immediately after 
  /// ActOnDeclarator (when an initializer is present). The code is factored 
  /// this way to make sure we are able to handle the following:
  ///   void func() { int xx = xx; }
  /// This allows ActOnDeclarator to register "xx" prior to parsing the
  /// initializer. The declaration above should still result in a warning, 
  /// since the reference to "xx" is uninitialized.
  virtual void AddInitializerToDecl(DeclTy *Dcl, ExprArg Init) {
    return;
  }

  /// ActOnUninitializedDecl - This action is called immediately after
  /// ActOnDeclarator (when an initializer is *not* present).
  virtual void ActOnUninitializedDecl(DeclTy *Dcl) {
    return;
  }

  /// FinalizeDeclaratorGroup - After a sequence of declarators are parsed, this
  /// gives the actions implementation a chance to process the group as a whole.
  virtual DeclTy *FinalizeDeclaratorGroup(Scope *S, DeclTy *Group) {
    return Group;
  }

  /// ActOnStartOfFunctionDef - This is called at the start of a function
  /// definition, instead of calling ActOnDeclarator.  The Declarator includes
  /// information about formal arguments that are part of this function.
  virtual DeclTy *ActOnStartOfFunctionDef(Scope *FnBodyScope, Declarator &D) {
    // Default to ActOnDeclarator.
    return ActOnStartOfFunctionDef(FnBodyScope,
                                   ActOnDeclarator(FnBodyScope, D, 0));
  }

  /// ActOnStartOfFunctionDef - This is called at the start of a function
  /// definition, after the FunctionDecl has already been created.
  virtual DeclTy *ActOnStartOfFunctionDef(Scope *FnBodyScope, DeclTy *D) {
    return D;
  }

  virtual void ObjCActOnStartOfMethodDef(Scope *FnBodyScope, DeclTy *D) {
    return;
  }

  /// ActOnFinishFunctionBody - This is called when a function body has completed
  /// parsing.  Decl is the DeclTy returned by ParseStartOfFunctionDef.
  virtual DeclTy *ActOnFinishFunctionBody(DeclTy *Decl, StmtArg Body) {
    return Decl;
  }

  virtual DeclTy *ActOnFileScopeAsmDecl(SourceLocation Loc, ExprArg AsmString) {
    return 0;
  }
  
  /// ActOnPopScope - This callback is called immediately before the specified
  /// scope is popped and deleted.
  virtual void ActOnPopScope(SourceLocation Loc, Scope *S) {}

  /// ActOnTranslationUnitScope - This callback is called once, immediately
  /// after creating the translation unit scope (in Parser::Initialize).
  virtual void ActOnTranslationUnitScope(SourceLocation Loc, Scope *S) {}
    
  /// ParsedFreeStandingDeclSpec - This method is invoked when a declspec with
  /// no declarator (e.g. "struct foo;") is parsed.
  virtual DeclTy *ParsedFreeStandingDeclSpec(Scope *S, DeclSpec &DS) {
    return 0;
  }

  /// ActOnLinkageSpec - Parsed a C++ linkage-specification that
  /// contained braces. Lang/StrSize contains the language string that
  /// was parsed at location Loc. Decls/NumDecls provides the
  /// declarations parsed inside the linkage specification.
  virtual DeclTy *ActOnLinkageSpec(SourceLocation Loc, SourceLocation LBrace,
                                   SourceLocation RBrace, const char *Lang,
                                   unsigned StrSize, 
                                   DeclTy **Decls, unsigned NumDecls) {
    return 0;
  }

  /// ActOnLinkageSpec - Parsed a C++ linkage-specification without
  /// braces. Lang/StrSize contains the language string that was
  /// parsed at location Loc. D is the declaration parsed.
  virtual DeclTy *ActOnLinkageSpec(SourceLocation Loc, const char *Lang,
                                   unsigned StrSize, DeclTy *D) {
    return 0;
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
    return 0;
  }
  
  enum TagKind {
    TK_Reference,   // Reference to a tag:  'struct foo *X;'
    TK_Declaration, // Fwd decl of a tag:   'struct foo;'
    TK_Definition   // Definition of a tag: 'struct foo { int X; } Y;'
  };
  virtual DeclTy *ActOnTag(Scope *S, unsigned TagType, TagKind TK,
                           SourceLocation KWLoc, const CXXScopeSpec &SS,
                           IdentifierInfo *Name, SourceLocation NameLoc,
                           AttributeList *Attr,
                           MultiTemplateParamsArg TemplateParameterLists) {
    // TagType is an instance of DeclSpec::TST, indicating what kind of tag this
    // is (struct/union/enum/class).
    return 0;
  }
  
  /// Act on @defs() element found when parsing a structure.  ClassName is the
  /// name of the referenced class.   
  virtual void ActOnDefs(Scope *S, DeclTy *TagD, SourceLocation DeclStart,
                         IdentifierInfo *ClassName,
                         llvm::SmallVectorImpl<DeclTy*> &Decls) {}
  virtual DeclTy *ActOnField(Scope *S, DeclTy *TagD, SourceLocation DeclStart,
                             Declarator &D, ExprTy *BitfieldWidth) {
    return 0;
  }
  
  virtual DeclTy *ActOnIvar(Scope *S, SourceLocation DeclStart,
                            Declarator &D, ExprTy *BitfieldWidth,
                            tok::ObjCKeywordKind visibility) {
    return 0;
  }
  
  virtual void ActOnFields(Scope* S, SourceLocation RecLoc, DeclTy *TagDecl,
                           DeclTy **Fields, unsigned NumFields, 
                           SourceLocation LBrac, SourceLocation RBrac,
                           AttributeList *AttrList) {}
  
  virtual DeclTy *ActOnEnumConstant(Scope *S, DeclTy *EnumDecl,
                                    DeclTy *LastEnumConstant,
                                    SourceLocation IdLoc, IdentifierInfo *Id,
                                    SourceLocation EqualLoc, ExprTy *Val) {
    return 0;
  }
  virtual void ActOnEnumBody(SourceLocation EnumLoc, DeclTy *EnumDecl,
                             DeclTy **Elements, unsigned NumElements) {}

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
  virtual OwningStmtResult ActOnDeclStmt(DeclTy *Decl, SourceLocation StartLoc,
                                   SourceLocation EndLoc) {
    return StmtEmpty();
  }

  virtual OwningStmtResult ActOnExprStmt(ExprArg Expr) {
    return OwningStmtResult(*this, Expr.release());
  }

  /// ActOnCaseStmt - Note that this handles the GNU 'case 1 ... 4' extension,
  /// which can specify an RHS value.
  virtual StmtResult ActOnCaseStmt(SourceLocation CaseLoc, ExprTy *LHSVal,
                                   SourceLocation DotDotDotLoc, ExprTy *RHSVal,
                                   SourceLocation ColonLoc, StmtTy *SubStmt) {
    return 0;
  }
  virtual StmtResult ActOnDefaultStmt(SourceLocation DefaultLoc,
                                      SourceLocation ColonLoc, StmtTy *SubStmt,
                                      Scope *CurScope){
    return 0;
  }
  
  virtual StmtResult ActOnLabelStmt(SourceLocation IdentLoc, IdentifierInfo *II,
                                    SourceLocation ColonLoc, StmtTy *SubStmt) {
    return 0;
  }
  
  virtual StmtResult ActOnIfStmt(SourceLocation IfLoc, ExprTy *CondVal,
                                 StmtTy *ThenVal, SourceLocation ElseLoc,
                                 StmtTy *ElseVal) {
    return 0; 
  }
  
  virtual StmtResult ActOnStartOfSwitchStmt(ExprTy *Cond) {
    return 0;
  }
  
  virtual StmtResult ActOnFinishSwitchStmt(SourceLocation SwitchLoc, 
                                           StmtTy *Switch, ExprTy *Body) {
    return 0;
  }

  virtual StmtResult ActOnWhileStmt(SourceLocation WhileLoc, ExprTy *Cond,
                                    StmtTy *Body) {
    return 0;
  }
  virtual StmtResult ActOnDoStmt(SourceLocation DoLoc, StmtTy *Body,
                                 SourceLocation WhileLoc, ExprTy *Cond) {
    return 0;
  }
  virtual StmtResult ActOnForStmt(SourceLocation ForLoc, 
                                  SourceLocation LParenLoc, 
                                  StmtTy *First, ExprTy *Second, ExprTy *Third,
                                  SourceLocation RParenLoc, StmtTy *Body) {
    return 0;
  }
  virtual StmtResult ActOnObjCForCollectionStmt(SourceLocation ForColLoc, 
                                  SourceLocation LParenLoc, 
                                  StmtTy *First, ExprTy *Second,
                                  SourceLocation RParenLoc, StmtTy *Body) {
    return 0;
  }
  virtual StmtResult ActOnGotoStmt(SourceLocation GotoLoc,
                                   SourceLocation LabelLoc,
                                   IdentifierInfo *LabelII) {
    return 0;
  }
  virtual StmtResult ActOnIndirectGotoStmt(SourceLocation GotoLoc,
                                           SourceLocation StarLoc,
                                           ExprTy *DestExp) {
    return 0;
  }
  virtual StmtResult ActOnContinueStmt(SourceLocation ContinueLoc,
                                       Scope *CurScope) {
    return 0;
  }
  virtual StmtResult ActOnBreakStmt(SourceLocation GotoLoc, Scope *CurScope) {
    return 0;
  }
  virtual StmtResult ActOnReturnStmt(SourceLocation ReturnLoc,
                                     ExprTy *RetValExp) {
    return 0;
  }
  virtual StmtResult ActOnAsmStmt(SourceLocation AsmLoc,
                                  bool IsSimple,                                  
                                  bool IsVolatile,
                                  unsigned NumOutputs,
                                  unsigned NumInputs,
                                  std::string *Names,
                                  ExprTy **Constraints,
                                  ExprTy **Exprs,
                                  ExprTy *AsmString,
                                  unsigned NumClobbers,
                                  ExprTy **Clobbers,
                                  SourceLocation RParenLoc) {
    return 0;
  }
  
  // Objective-c statements
  virtual StmtResult ActOnObjCAtCatchStmt(SourceLocation AtLoc, 
                                          SourceLocation RParen, StmtTy *Parm, 
                                          StmtTy *Body, StmtTy *CatchList) {
    return 0;
  }
  
  virtual StmtResult ActOnObjCAtFinallyStmt(SourceLocation AtLoc, 
                                            StmtTy *Body) {
    return 0;
  }
  
  virtual StmtResult ActOnObjCAtTryStmt(SourceLocation AtLoc, 
                                        StmtTy *Try, 
                                        StmtTy *Catch, StmtTy *Finally) {
    return 0;
  }
  
  virtual StmtResult ActOnObjCAtThrowStmt(SourceLocation AtLoc, 
                                          StmtTy *Throw) {
    return 0;
  }
  
  virtual StmtResult ActOnObjCAtSynchronizedStmt(SourceLocation AtLoc, 
                                        ExprTy *SynchExpr, 
                                        StmtTy *SynchBody) {
    return 0;
  }

  // C++ Statements
  virtual DeclTy *ActOnExceptionDeclarator(Scope *S, Declarator &D) {
    return 0;
  }

  virtual OwningStmtResult ActOnCXXCatchBlock(SourceLocation CatchLoc,
                                              DeclTy *ExceptionDecl,
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
  
  // Primary Expressions.
  
  /// ActOnIdentifierExpr - Parse an identifier in expression context.
  /// 'HasTrailingLParen' indicates whether or not the identifier has a '('
  /// token immediately after it.
  /// An optional CXXScopeSpec can be passed to indicate the C++ scope (class or
  /// namespace) that the identifier must be a member of.
  /// i.e. for "foo::bar", 'II' will be "bar" and 'SS' will be "foo::".
  virtual ExprResult ActOnIdentifierExpr(Scope *S, SourceLocation Loc,
                                         IdentifierInfo &II,
                                         bool HasTrailingLParen,
                                         const CXXScopeSpec *SS = 0) {
    return 0;
  }

  /// ActOnOperatorFunctionIdExpr - Parse a C++ overloaded operator
  /// name (e.g., @c operator+ ) as an expression. This is very
  /// similar to ActOnIdentifierExpr, except that instead of providing
  /// an identifier the parser provides the kind of overloaded
  /// operator that was parsed.
  virtual ExprResult ActOnCXXOperatorFunctionIdExpr(Scope *S, 
                                                    SourceLocation OperatorLoc,
                                                    OverloadedOperatorKind Op,
                                                    bool HasTrailingLParen,
                                                    const CXXScopeSpec &SS) {
    return 0;
  }
  
  /// ActOnCXXConversionFunctionExpr - Parse a C++ conversion function
  /// name (e.g., @c operator void const *) as an expression. This is
  /// very similar to ActOnIdentifierExpr, except that instead of
  /// providing an identifier the parser provides the type of the
  /// conversion function.
  virtual ExprResult ActOnCXXConversionFunctionExpr(Scope *S, 
                                                    SourceLocation OperatorLoc,
                                                    TypeTy *Type,
                                                    bool HasTrailingLParen,
                                                    const CXXScopeSpec &SS) {
    return 0;
  }

  virtual ExprResult ActOnPredefinedExpr(SourceLocation Loc,
                                         tok::TokenKind Kind) {
    return 0;
  }
  virtual ExprResult ActOnCharacterConstant(const Token &) { return 0; }
  virtual ExprResult ActOnNumericConstant(const Token &) { return 0; }
  
  /// ActOnStringLiteral - The specified tokens were lexed as pasted string
  /// fragments (e.g. "foo" "bar" L"baz").
  virtual ExprResult ActOnStringLiteral(const Token *Toks, unsigned NumToks) {
    return 0;
  }
  
  virtual ExprResult ActOnParenExpr(SourceLocation L, SourceLocation R,
                                    ExprTy *Val) {
    return Val;  // Default impl returns operand.
  }
  
  // Postfix Expressions.
  virtual ExprResult ActOnPostfixUnaryOp(Scope *S, SourceLocation OpLoc, 
                                         tok::TokenKind Kind, ExprTy *Input) {
    return 0;
  }
  virtual ExprResult ActOnArraySubscriptExpr(Scope *S,
                                             ExprTy *Base, SourceLocation LLoc,
                                             ExprTy *Idx, SourceLocation RLoc) {
    return 0;
  }
  virtual ExprResult ActOnMemberReferenceExpr(Scope *S, ExprTy *Base,
                                              SourceLocation OpLoc,
                                              tok::TokenKind OpKind,
                                              SourceLocation MemberLoc,
                                              IdentifierInfo &Member) {
    return 0;
  }
  
  /// ActOnCallExpr - Handle a call to Fn with the specified array of arguments.
  /// This provides the location of the left/right parens and a list of comma
  /// locations.  There are guaranteed to be one fewer commas than arguments,
  /// unless there are zero arguments.
  virtual ExprResult ActOnCallExpr(Scope *S, ExprTy *Fn, 
                                   SourceLocation LParenLoc,
                                   ExprTy **Args, unsigned NumArgs,
                                   SourceLocation *CommaLocs,
                                   SourceLocation RParenLoc) {
    return 0;
  }
  
  // Unary Operators.  'Tok' is the token for the operator.
  virtual ExprResult ActOnUnaryOp(Scope *S, SourceLocation OpLoc, 
                                  tok::TokenKind Op, ExprTy *Input) {
    return 0;
  }
  virtual ExprResult 
    ActOnSizeOfAlignOfExpr(SourceLocation OpLoc, bool isSizeof, bool isType,
                           void *TyOrEx, const SourceRange &ArgRange) {
    return 0;
  }
  
  virtual ExprResult ActOnCompoundLiteral(SourceLocation LParen, TypeTy *Ty,
                                          SourceLocation RParen, ExprTy *Op) {
    return 0;
  }
  virtual ExprResult ActOnInitList(SourceLocation LParenLoc,
                                   ExprTy **InitList, unsigned NumInit,
                                   InitListDesignations &Designators,
                                   SourceLocation RParenLoc) {
    return 0;
  }
  virtual ExprResult ActOnCastExpr(SourceLocation LParenLoc, TypeTy *Ty,
                                   SourceLocation RParenLoc, ExprTy *Op) {
    return 0;
  }
  
  virtual ExprResult ActOnBinOp(Scope *S, SourceLocation TokLoc, 
                                tok::TokenKind Kind,
                                ExprTy *LHS, ExprTy *RHS) {
    return 0;
  }

  /// ActOnConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
  /// in the case of a the GNU conditional expr extension.
  virtual ExprResult ActOnConditionalOp(SourceLocation QuestionLoc, 
                                        SourceLocation ColonLoc,
                                        ExprTy *Cond, ExprTy *LHS, ExprTy *RHS){
    return 0;
  }
  
  //===---------------------- GNU Extension Expressions -------------------===//

  virtual ExprResult ActOnAddrLabel(SourceLocation OpLoc, SourceLocation LabLoc,
                                    IdentifierInfo *LabelII) { // "&&foo"
    return 0;
  }
  
  virtual ExprResult ActOnStmtExpr(SourceLocation LPLoc, StmtTy *SubStmt,
                                   SourceLocation RPLoc) { // "({..})"
    return 0;
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
  
  virtual ExprResult ActOnBuiltinOffsetOf(Scope *S, SourceLocation BuiltinLoc,
                                          SourceLocation TypeLoc, TypeTy *Arg1,
                                          OffsetOfComponent *CompPtr,
                                          unsigned NumComponents,
                                          SourceLocation RParenLoc) {
    return 0;
  }
  
  // __builtin_types_compatible_p(type1, type2)
  virtual ExprResult ActOnTypesCompatibleExpr(SourceLocation BuiltinLoc, 
                                              TypeTy *arg1, TypeTy *arg2,
                                              SourceLocation RPLoc) {
    return 0;
  }
  // __builtin_choose_expr(constExpr, expr1, expr2)
  virtual ExprResult ActOnChooseExpr(SourceLocation BuiltinLoc, 
                                     ExprTy *cond, ExprTy *expr1, ExprTy *expr2,
                                     SourceLocation RPLoc) {
    return 0;
  }
  // __builtin_overload(...)
  virtual ExprResult ActOnOverloadExpr(ExprTy **Args, unsigned NumArgs,
                                       SourceLocation *CommaLocs,
                                       SourceLocation BuiltinLoc, 
                                       SourceLocation RPLoc) {
    return 0;
  }
  

  // __builtin_va_arg(expr, type)
  virtual ExprResult ActOnVAArg(SourceLocation BuiltinLoc,
                                ExprTy *expr, TypeTy *type,
                                SourceLocation RPLoc) {
    return 0;
  }

  /// ActOnGNUNullExpr - Parsed the GNU __null expression, the token
  /// for which is at position TokenLoc.
  virtual ExprResult ActOnGNUNullExpr(SourceLocation TokenLoc) {
    return 0;
  }

  //===------------------------- "Block" Extension ------------------------===//

  /// ActOnBlockStart - This callback is invoked when a block literal is
  /// started.  The result pointer is passed into the block finalizers.
  virtual void ActOnBlockStart(SourceLocation CaretLoc, Scope *CurScope) {}

  /// ActOnBlockArguments - This callback allows processing of block arguments.
  /// If there are no arguments, this is still invoked.
  virtual void ActOnBlockArguments(Declarator &ParamInfo) {}
  
  /// ActOnBlockError - If there is an error parsing a block, this callback
  /// is invoked to pop the information about the block from the action impl.
  virtual void ActOnBlockError(SourceLocation CaretLoc, Scope *CurScope) {}
  
  /// ActOnBlockStmtExpr - This is called when the body of a block statement
  /// literal was successfully completed.  ^(int x){...}
  virtual ExprResult ActOnBlockStmtExpr(SourceLocation CaretLoc, StmtTy *Body,
                                        Scope *CurScope) { return 0; }

  //===------------------------- C++ Declarations -------------------------===//

  /// ActOnStartNamespaceDef - This is called at the start of a namespace
  /// definition.
  virtual DeclTy *ActOnStartNamespaceDef(Scope *S, SourceLocation IdentLoc,
                                        IdentifierInfo *Ident,
                                        SourceLocation LBrace) {
    return 0;
  }

  /// ActOnFinishNamespaceDef - This callback is called after a namespace is
  /// exited. Decl is the DeclTy returned by ActOnStartNamespaceDef.
  virtual void ActOnFinishNamespaceDef(DeclTy *Dcl,SourceLocation RBrace) {
    return;
  }

  /// ActOnParamDefaultArgument - Parse default argument for function parameter
  virtual void ActOnParamDefaultArgument(DeclTy *param,
                                         SourceLocation EqualLoc,
                                         ExprTy *defarg) {
  }

  /// ActOnParamUnparsedDefaultArgument - We've seen a default
  /// argument for a function parameter, but we can't parse it yet
  /// because we're inside a class definition. Note that this default
  /// argument will be parsed later.
  virtual void ActOnParamUnparsedDefaultArgument(DeclTy *param, 
                                                 SourceLocation EqualLoc) { }

  /// ActOnParamDefaultArgumentError - Parsing or semantic analysis of
  /// the default argument for the parameter param failed.
  virtual void ActOnParamDefaultArgumentError(DeclTy *param) { }

  /// AddCXXDirectInitializerToDecl - This action is called immediately after 
  /// ActOnDeclarator, when a C++ direct initializer is present.
  /// e.g: "int x(1);"
  virtual void AddCXXDirectInitializerToDecl(DeclTy *Dcl,
                                             SourceLocation LParenLoc,
                                             ExprTy **Exprs, unsigned NumExprs,
                                             SourceLocation *CommaLocs,
                                             SourceLocation RParenLoc) {
    return;
  }
  
  /// ActOnStartDelayedCXXMethodDeclaration - We have completed
  /// parsing a top-level (non-nested) C++ class, and we are now
  /// parsing those parts of the given Method declaration that could
  /// not be parsed earlier (C++ [class.mem]p2), such as default
  /// arguments. This action should enter the scope of the given
  /// Method declaration as if we had just parsed the qualified method
  /// name. However, it should not bring the parameters into scope;
  /// that will be performed by ActOnDelayedCXXMethodParameter.
  virtual void ActOnStartDelayedCXXMethodDeclaration(Scope *S, DeclTy *Method) {
  }

  /// ActOnDelayedCXXMethodParameter - We've already started a delayed
  /// C++ method declaration. We're (re-)introducing the given
  /// function parameter into scope for use in parsing later parts of
  /// the method declaration. For example, we could see an
  /// ActOnParamDefaultArgument event for this parameter.
  virtual void ActOnDelayedCXXMethodParameter(Scope *S, DeclTy *Param) {
  }

  /// ActOnFinishDelayedCXXMethodDeclaration - We have finished
  /// processing the delayed method declaration for Method. The method
  /// declaration is now considered finished. There may be a separate
  /// ActOnStartOfFunctionDef action later (not necessarily
  /// immediately!) for this method, if it was also defined inside the
  /// class body.
  virtual void ActOnFinishDelayedCXXMethodDeclaration(Scope *S, DeclTy *Method) {
  }

  //===------------------------- C++ Expressions --------------------------===//
  
  /// ActOnCXXNamedCast - Parse {dynamic,static,reinterpret,const}_cast's.
  virtual ExprResult ActOnCXXNamedCast(SourceLocation OpLoc, tok::TokenKind Kind,
                                       SourceLocation LAngleBracketLoc, TypeTy *Ty,
                                       SourceLocation RAngleBracketLoc,
                                       SourceLocation LParenLoc, ExprTy *Op,
                                       SourceLocation RParenLoc) {
    return 0;
  }

  /// ActOnCXXTypeidOfType - Parse typeid( type-id ).
  virtual ExprResult ActOnCXXTypeid(SourceLocation OpLoc,
                                    SourceLocation LParenLoc, bool isType,
                                    void *TyOrExpr, SourceLocation RParenLoc) {
    return 0;
  }

  /// ActOnCXXThis - Parse the C++ 'this' pointer.
  virtual ExprResult ActOnCXXThis(SourceLocation ThisLoc) {
    return 0;
  }

  /// ActOnCXXBoolLiteral - Parse {true,false} literals.
  virtual ExprResult ActOnCXXBoolLiteral(SourceLocation OpLoc,
                                         tok::TokenKind Kind) {
    return 0;
  }

  /// ActOnCXXThrow - Parse throw expressions.
  virtual ExprResult ActOnCXXThrow(SourceLocation OpLoc,
                                   ExprTy *Op = 0) {
    return 0;
  }

  /// ActOnCXXTypeConstructExpr - Parse construction of a specified type.
  /// Can be interpreted either as function-style casting ("int(x)")
  /// or class type construction ("ClassType(x,y,z)")
  /// or creation of a value-initialized type ("int()").
  virtual ExprResult ActOnCXXTypeConstructExpr(SourceRange TypeRange,
                                               TypeTy *TypeRep,
                                               SourceLocation LParenLoc,
                                               ExprTy **Exprs,
                                               unsigned NumExprs,
                                               SourceLocation *CommaLocs,
                                               SourceLocation RParenLoc) {
    return 0;
  }

  /// ActOnCXXConditionDeclarationExpr - Parsed a condition declaration of a
  /// C++ if/switch/while/for statement.
  /// e.g: "if (int x = f()) {...}"
  virtual ExprResult ActOnCXXConditionDeclarationExpr(Scope *S,
                                                      SourceLocation StartLoc,
                                                      Declarator &D,
                                                      SourceLocation EqualLoc,
                                                      ExprTy *AssignExprVal) {
    return 0;
  }

  /// ActOnCXXNew - Parsed a C++ 'new' expression. UseGlobal is true if the
  /// new was qualified (::new). In a full new like
  /// @code new (p1, p2) type(c1, c2) @endcode
  /// the p1 and p2 expressions will be in PlacementArgs and the c1 and c2
  /// expressions in ConstructorArgs. The type is passed as a declarator.
  virtual ExprResult ActOnCXXNew(SourceLocation StartLoc, bool UseGlobal,
                                 SourceLocation PlacementLParen,
                                 ExprTy **PlacementArgs, unsigned NumPlaceArgs,
                                 SourceLocation PlacementRParen,
                                 bool ParenTypeId, Declarator &D,
                                 SourceLocation ConstructorLParen,
                                 ExprTy **ConstructorArgs, unsigned NumConsArgs,
                                 SourceLocation ConstructorRParen) {
    return 0;
  }

  /// ActOnCXXDelete - Parsed a C++ 'delete' expression. UseGlobal is true if
  /// the delete was qualified (::delete). ArrayForm is true if the array form
  /// was used (delete[]).
  virtual ExprResult ActOnCXXDelete(SourceLocation StartLoc, bool UseGlobal,
                                    bool ArrayForm, ExprTy *Operand) {
    return 0;
  }

  //===---------------------------- C++ Classes ---------------------------===//
  /// ActOnBaseSpecifier - Parsed a base specifier
  virtual BaseResult ActOnBaseSpecifier(DeclTy *classdecl, 
                                        SourceRange SpecifierRange,
                                        bool Virtual, AccessSpecifier Access,
                                        TypeTy *basetype, 
                                        SourceLocation BaseLoc) {
    return 0;
  }

  virtual void ActOnBaseSpecifiers(DeclTy *ClassDecl, BaseTy **Bases, 
                                   unsigned NumBases) {
  }
                                   
  /// ActOnStartCXXClassDef - This is called at the start of a class/struct/union
  /// definition, when on C++.
  virtual void ActOnStartCXXClassDef(Scope *S, DeclTy *TagDecl,
                                     SourceLocation LBrace) {
  }
  
  /// ActOnCXXMemberDeclarator - This is invoked when a C++ class member
  /// declarator is parsed. 'AS' is the access specifier, 'BitfieldWidth'
  /// specifies the bitfield width if there is one and 'Init' specifies the
  /// initializer if any. 'LastInGroup' is non-null for cases where one declspec
  /// has multiple declarators on it.
  virtual DeclTy *ActOnCXXMemberDeclarator(Scope *S, AccessSpecifier AS,
                                           Declarator &D, ExprTy *BitfieldWidth,
                                           ExprTy *Init, DeclTy *LastInGroup) {
    return 0;
  }

  virtual MemInitResult ActOnMemInitializer(DeclTy *ConstructorDecl,
                                            Scope *S,
                                            IdentifierInfo *MemberOrBase,
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
  virtual void ActOnMemInitializers(DeclTy *ConstructorDecl, 
                                    SourceLocation ColonLoc,
                                    MemInitTy **MemInits, unsigned NumMemInits) {
  }

  /// ActOnFinishCXXMemberSpecification - Invoked after all member declarators
  /// are parsed but *before* parsing of inline method definitions.
  virtual void ActOnFinishCXXMemberSpecification(Scope* S, SourceLocation RLoc,
                                                 DeclTy *TagDecl,
                                                 SourceLocation LBrac,
                                                 SourceLocation RBrac) {
  }

  /// ActOnFinishCXXClassDef - This is called when a class/struct/union has
  /// completed parsing, when on C++.
  virtual void ActOnFinishCXXClassDef(DeclTy *TagDecl) {
  }

  //===---------------------------C++ Templates----------------------------===//

  /// ActOnTypeParameter - Called when a C++ template type parameter
  /// (e.g., "typename T") has been parsed. Typename specifies whether
  /// the keyword "typename" was used to declare the type parameter
  /// (otherwise, "class" was used), and KeyLoc is the location of the
  /// "class" or "typename" keyword. ParamName is the name of the
  /// parameter (NULL indicates an unnamed template parameter) and
  /// ParamName is the location of the parameter name (if any). 
  /// If the type parameter has a default argument, it will be added
  /// later via ActOnTypeParameterDefault. Depth and Position provide
  /// the number of enclosing templates (see
  /// ActOnTemplateParameterList) and the number of previous
  /// parameters within this template parameter list.
  virtual DeclTy *ActOnTypeParameter(Scope *S, bool Typename, 
				     SourceLocation KeyLoc,
				     IdentifierInfo *ParamName,
				     SourceLocation ParamNameLoc,
                                     unsigned Depth, unsigned Position) {
    return 0;
  }

  /// ActOnTypeParameterDefault - Adds a default argument (the type
  /// Default) to the given template type parameter (TypeParam). 
  virtual void ActOnTypeParameterDefault(DeclTy *TypeParam, TypeTy *Default) {
  }

  /// ActOnNonTypeTemplateParameter - Called when a C++ non-type
  /// template parameter (e.g., "int Size" in "template<int Size>
  /// class Array") has been parsed. S is the current scope and D is
  /// the parsed declarator. Depth and Position provide           
  /// the number of enclosing templates (see
  /// ActOnTemplateParameterList) and the number of previous
  /// parameters within this template parameter list.
  virtual DeclTy *ActOnNonTypeTemplateParameter(Scope *S, Declarator &D,
                                                unsigned Depth, 
                                                unsigned Position) {
    return 0;
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
                             DeclTy **Params, unsigned NumParams,
                             SourceLocation RAngleLoc) {
    return 0;
  }

  //===----------------------- Obj-C Declarations -------------------------===//
  
  // ActOnStartClassInterface - this action is called immediately after parsing
  // the prologue for a class interface (before parsing the instance 
  // variables). Instance variables are processed by ActOnFields().
  virtual DeclTy *ActOnStartClassInterface(SourceLocation AtInterfaceLoc,
                                           IdentifierInfo *ClassName, 
                                           SourceLocation ClassLoc,
                                           IdentifierInfo *SuperName, 
                                           SourceLocation SuperLoc,
                                           DeclTy * const *ProtoRefs, 
                                           unsigned NumProtoRefs,
                                           SourceLocation EndProtoLoc,
                                           AttributeList *AttrList) {
    return 0;
  }
  
  /// ActOnCompatiblityAlias - this action is called after complete parsing of
  /// @compaatibility_alias declaration. It sets up the alias relationships.
  virtual DeclTy *ActOnCompatiblityAlias(
    SourceLocation AtCompatibilityAliasLoc,
    IdentifierInfo *AliasName,  SourceLocation AliasLocation,
    IdentifierInfo *ClassName, SourceLocation ClassLocation) {
    return 0;
  }
  
  // ActOnStartProtocolInterface - this action is called immdiately after
  // parsing the prologue for a protocol interface.
  virtual DeclTy *ActOnStartProtocolInterface(SourceLocation AtProtoLoc,
                                              IdentifierInfo *ProtocolName, 
                                              SourceLocation ProtocolLoc,
                                              DeclTy * const *ProtoRefs,
                                              unsigned NumProtoRefs,
                                              SourceLocation EndProtoLoc,
                                              AttributeList *AttrList) {
    return 0;
  }
  // ActOnStartCategoryInterface - this action is called immdiately after
  // parsing the prologue for a category interface.
  virtual DeclTy *ActOnStartCategoryInterface(SourceLocation AtInterfaceLoc,
                                              IdentifierInfo *ClassName, 
                                              SourceLocation ClassLoc,
                                              IdentifierInfo *CategoryName, 
                                              SourceLocation CategoryLoc,
                                              DeclTy * const *ProtoRefs,
                                              unsigned NumProtoRefs,
                                              SourceLocation EndProtoLoc) {
    return 0;
  }
  // ActOnStartClassImplementation - this action is called immdiately after
  // parsing the prologue for a class implementation. Instance variables are 
  // processed by ActOnFields().
  virtual DeclTy *ActOnStartClassImplementation(
    SourceLocation AtClassImplLoc,
    IdentifierInfo *ClassName, 
    SourceLocation ClassLoc,
    IdentifierInfo *SuperClassname, 
    SourceLocation SuperClassLoc) {
    return 0;
  }
  // ActOnStartCategoryImplementation - this action is called immdiately after
  // parsing the prologue for a category implementation.
  virtual DeclTy *ActOnStartCategoryImplementation(
    SourceLocation AtCatImplLoc,
    IdentifierInfo *ClassName, 
    SourceLocation ClassLoc,
    IdentifierInfo *CatName,
    SourceLocation CatLoc) {
    return 0;
  }  
  // ActOnPropertyImplDecl - called for every property implementation
  virtual DeclTy *ActOnPropertyImplDecl(
   SourceLocation AtLoc,              // location of the @synthesize/@dynamic
   SourceLocation PropertyNameLoc,    // location for the property name
   bool ImplKind,                     // true for @synthesize, false for
                                      // @dynamic
   DeclTy *ClassImplDecl,             // class or category implementation
   IdentifierInfo *propertyId,        // name of property
   IdentifierInfo *propertyIvar) {    // name of the ivar
    return 0;
  }

  // ActOnMethodDeclaration - called for all method declarations. 
  virtual DeclTy *ActOnMethodDeclaration(
    SourceLocation BeginLoc,   // location of the + or -.
    SourceLocation EndLoc,     // location of the ; or {.
    tok::TokenKind MethodType, // tok::minus for instance, tok::plus for class.
    DeclTy *ClassDecl,         // class this methods belongs to.
    ObjCDeclSpec &ReturnQT,    // for return type's in inout etc.
    TypeTy *ReturnType,        // the method return type.
    Selector Sel,              // a unique name for the method.
    ObjCDeclSpec *ArgQT,       // for arguments' in inout etc.
    TypeTy **ArgTypes,         // non-zero when Sel.getNumArgs() > 0
    IdentifierInfo **ArgNames, // non-zero when Sel.getNumArgs() > 0
    AttributeList *AttrList,   // optional
    // tok::objc_not_keyword, tok::objc_optional, tok::objc_required    
    tok::ObjCKeywordKind impKind,
    bool isVariadic = false) {
    return 0;
  }
  // ActOnAtEnd - called to mark the @end. For declarations (interfaces,
  // protocols, categories), the parser passes all methods/properties. 
  // For class implementations, these values default to 0. For implementations,
  // methods are processed incrementally (by ActOnMethodDeclaration above).
  virtual void ActOnAtEnd(
    SourceLocation AtEndLoc, 
    DeclTy *classDecl,
    DeclTy **allMethods = 0, 
    unsigned allNum = 0,
    DeclTy **allProperties = 0, 
    unsigned pNum = 0) {
    return;
  }
  // ActOnProperty - called to build one property AST
  virtual DeclTy *ActOnProperty (Scope *S, SourceLocation AtLoc,
                                 FieldDeclarator &FD, ObjCDeclSpec &ODS,
                                 Selector GetterSel, Selector SetterSel,
				 DeclTy *ClassCategory,
				 bool *OverridingProperty,
                                 tok::ObjCKeywordKind MethodImplKind) {
    return 0;
  }
                                     
  // ActOnClassMessage - used for both unary and keyword messages.
  // ArgExprs is optional - if it is present, the number of expressions
  // is obtained from NumArgs.
  virtual ExprResult ActOnClassMessage(
    Scope *S,
    IdentifierInfo *receivingClassName, 
    Selector Sel,
    SourceLocation lbrac,
    SourceLocation receiverLoc,
    SourceLocation rbrac, 
    ExprTy **ArgExprs, unsigned NumArgs) {
    return 0;
  }
  // ActOnInstanceMessage - used for both unary and keyword messages.
  // ArgExprs is optional - if it is present, the number of expressions
  // is obtained from NumArgs.
  virtual ExprResult ActOnInstanceMessage(
    ExprTy *receiver, Selector Sel,
    SourceLocation lbrac, SourceLocation rbrac, 
    ExprTy **ArgExprs, unsigned NumArgs) {
    return 0;
  }
  virtual DeclTy *ActOnForwardClassDeclaration(
    SourceLocation AtClassLoc,
    IdentifierInfo **IdentList,
    unsigned NumElts) {
    return 0;
  }
  virtual DeclTy *ActOnForwardProtocolDeclaration(
    SourceLocation AtProtocolLoc,
    const IdentifierLocPair*IdentList,
    unsigned NumElts,
    AttributeList *AttrList) {
    return 0;
  }
  
  /// FindProtocolDeclaration - This routine looks up protocols and
  /// issues error if they are not declared. It returns list of valid
  /// protocols found.
  virtual void FindProtocolDeclaration(bool WarnOnDeclarations,
                                       const IdentifierLocPair *ProtocolId,
                                       unsigned NumProtocols,
                                 llvm::SmallVectorImpl<DeclTy*> &ResProtos) {
  }

  //===----------------------- Obj-C Expressions --------------------------===//

  virtual ExprResult ParseObjCStringLiteral(SourceLocation *AtLocs, 
                                            ExprTy **Strings,
                                            unsigned NumStrings) {
    return 0;
  }

  virtual ExprResult ParseObjCEncodeExpression(SourceLocation AtLoc,
                                               SourceLocation EncLoc,
                                               SourceLocation LParenLoc,
                                               TypeTy *Ty,
                                               SourceLocation RParenLoc) {
    return 0;
  }
  
  virtual ExprResult ParseObjCSelectorExpression(Selector Sel,
                                                 SourceLocation AtLoc,
                                                 SourceLocation SelLoc,
                                                 SourceLocation LParenLoc,
                                                 SourceLocation RParenLoc) {
    return 0;
  }
  
  virtual ExprResult ParseObjCProtocolExpression(IdentifierInfo *ProtocolId,
                                                 SourceLocation AtLoc,
                                                 SourceLocation ProtoLoc,
                                                 SourceLocation LParenLoc,
                                                 SourceLocation RParenLoc) {
    return 0;
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
public:
  MinimalAction(Preprocessor &pp);
  
  /// isTypeName - This looks at the IdentifierInfo::FETokenInfo field to
  /// determine whether the name is a typedef or not in this scope.
  virtual TypeTy *isTypeName(IdentifierInfo &II, Scope *S,
                             const CXXScopeSpec *SS);

  /// isCurrentClassName - Always returns false, because MinimalAction
  /// does not support C++ classes with constructors.
  virtual bool isCurrentClassName(const IdentifierInfo& II, Scope *S,
                                  const CXXScopeSpec *SS);

  /// isTemplateName - Determines whether the identifier II is a
  /// template name in the current scope, and returns the template
  /// declaration if II names a template. An optional CXXScope can be
  /// passed to indicate the C++ scope in which the identifier will be
  /// found. 
  virtual DeclTy *isTemplateName(IdentifierInfo &II, Scope *S,
                                 const CXXScopeSpec *SS = 0);

  /// ActOnDeclarator - If this is a typedef declarator, we modify the
  /// IdentifierInfo::FETokenInfo field to keep track of this fact, until S is
  /// popped.
  virtual DeclTy *ActOnDeclarator(Scope *S, Declarator &D, DeclTy *LastInGroup);
  
  /// ActOnPopScope - When a scope is popped, if any typedefs are now 
  /// out-of-scope, they are removed from the IdentifierInfo::FETokenInfo field.
  virtual void ActOnPopScope(SourceLocation Loc, Scope *S);
  virtual void ActOnTranslationUnitScope(SourceLocation Loc, Scope *S);
  
  virtual DeclTy *ActOnForwardClassDeclaration(SourceLocation AtClassLoc,
                                               IdentifierInfo **IdentList,
                                               unsigned NumElts);
  
  virtual DeclTy *ActOnStartClassInterface(SourceLocation interLoc,
                                           IdentifierInfo *ClassName,
                                           SourceLocation ClassLoc,
                                           IdentifierInfo *SuperName,
                                           SourceLocation SuperLoc,
                                           DeclTy * const *ProtoRefs, 
                                           unsigned NumProtoRefs,
                                           SourceLocation EndProtoLoc,
                                           AttributeList *AttrList);
};

}  // end namespace clang

#endif
