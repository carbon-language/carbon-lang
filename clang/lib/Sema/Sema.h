//===--- Sema.h - Semantic Analysis & AST Building --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Sema class, which performs semantic analysis and
// builds ASTs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_SEMA_H
#define LLVM_CLANG_AST_SEMA_H

#include "IdentifierResolver.h"
#include "CXXFieldCollector.h"
#include "clang/Parse/Action.h"
#include "clang/Parse/DeclSpec.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/OwningPtr.h"
#include <vector>
#include <string>

namespace llvm {
  class APSInt;
}

namespace clang {
  class ASTContext;
  class ASTConsumer;
  class Preprocessor;
  class Decl;
  class DeclContext;
  class NamedDecl;
  class ScopedDecl;
  class Expr;
  class InitListExpr;
  class CallExpr;
  class VarDecl;
  class ParmVarDecl;
  class TypedefDecl;
  class FunctionDecl;
  class QualType;
  struct LangOptions;
  class Token;
  class IntegerLiteral;
  class StringLiteral;
  class ArrayType;
  class LabelStmt;
  class SwitchStmt;
  class ExtVectorType;
  class TypedefDecl;
  class ObjCInterfaceDecl;
  class ObjCCompatibleAliasDecl;
  class ObjCProtocolDecl;
  class ObjCImplementationDecl;
  class ObjCCategoryImplDecl;
  class ObjCCategoryDecl;
  class ObjCIvarDecl;
  class ObjCMethodDecl;
  class ObjCPropertyDecl;

/// Sema - This implements semantic analysis and AST building for C.
class Sema : public Action {
public:
  Preprocessor &PP;
  ASTContext &Context;
  ASTConsumer &Consumer;

  /// CurContext - This is the current declaration context of parsing.
  DeclContext *CurContext;

  /// LabelMap - This is a mapping from label identifiers to the LabelStmt for
  /// it (which acts like the label decl in some ways).  Forward referenced
  /// labels have a LabelStmt created for them with a null location & SubStmt.
  llvm::DenseMap<IdentifierInfo*, LabelStmt*> LabelMap;
  
  llvm::SmallVector<SwitchStmt*, 8> SwitchStack;
  
  /// ExtVectorDecls - This is a list all the extended vector types. This allows
  /// us to associate a raw vector type with one of the ext_vector type names.
  /// This is only necessary for issuing pretty diagnostics.
  llvm::SmallVector<TypedefDecl*, 24> ExtVectorDecls;

  /// ObjCImplementations - Keep track of all of the classes with
  /// @implementation's, so that we can emit errors on duplicates.
  llvm::DenseMap<IdentifierInfo*, ObjCImplementationDecl*> ObjCImplementations;
  
  /// ObjCProtocols - Keep track of all protocol declarations declared
  /// with @protocol keyword, so that we can emit errors on duplicates and
  /// find the declarations when needed.
  llvm::DenseMap<IdentifierInfo*, ObjCProtocolDecl*> ObjCProtocols;

  /// ObjCInterfaceDecls - Keep track of all class declarations declared
  /// with @interface, so that we can emit errors on duplicates and
  /// find the declarations when needed. 
  typedef llvm::DenseMap<const IdentifierInfo*, 
                         ObjCInterfaceDecl*> ObjCInterfaceDeclsTy;
  ObjCInterfaceDeclsTy ObjCInterfaceDecls;
    
  /// ObjCAliasDecls - Keep track of all class declarations declared
  /// with @compatibility_alias, so that we can emit errors on duplicates and
  /// find the declarations when needed. This construct is ancient and will
  /// likely never be seen. Nevertheless, it is here for compatibility.
  typedef llvm::DenseMap<const IdentifierInfo*, 
                         ObjCCompatibleAliasDecl*> ObjCAliasTy;
  ObjCAliasTy ObjCAliasDecls;

  /// FieldCollector - Collects CXXFieldDecls during parsing of C++ classes.
  llvm::OwningPtr<CXXFieldCollector> FieldCollector;

  IdentifierResolver IdResolver;

  // Enum values used by KnownFunctionIDs (see below).
  enum {
    id_printf,
    id_fprintf,
    id_sprintf,
    id_snprintf,
    id_asprintf,
    id_NSLog,
    id_vsnprintf,
    id_vasprintf,
    id_vfprintf,
    id_vsprintf,
    id_vprintf,
    id_num_known_functions
  };
  
  /// KnownFunctionIDs - This is a list of IdentifierInfo objects to a set
  /// of known functions used by the semantic analysis to do various
  /// kinds of checking (e.g. checking format string errors in printf calls).
  /// This list is populated upon the creation of a Sema object.    
  IdentifierInfo* KnownFunctionIDs[ id_num_known_functions ];
  
  /// Translation Unit Scope - useful to Objective-C actions that need
  /// to lookup file scope declarations in the "ordinary" C decl namespace.
  /// For example, user-defined classes, built-in "id" type, etc.
  Scope *TUScope;
  
  /// ObjCMethodList - a linked list of methods with different signatures.
  struct ObjCMethodList {
    ObjCMethodDecl *Method;
    ObjCMethodList *Next;
    
    ObjCMethodList() {
      Method = 0; 
      Next = 0;
    }
    ObjCMethodList(ObjCMethodDecl *M, ObjCMethodList *C) {
      Method = M;
      Next = C;
    }
  };
  /// Instance/Factory Method Pools - allows efficient lookup when typechecking
  /// messages to "id". We need to maintain a list, since selectors can have
  /// differing signatures across classes. In Cocoa, this happens to be 
  /// extremely uncommon (only 1% of selectors are "overloaded").
  llvm::DenseMap<Selector, ObjCMethodList> InstanceMethodPool;
  llvm::DenseMap<Selector, ObjCMethodList> FactoryMethodPool;
public:
  Sema(Preprocessor &pp, ASTContext &ctxt, ASTConsumer &consumer);
  
  const LangOptions &getLangOptions() const;
  
  /// The primitive diagnostic helpers - always returns true, which simplifies 
  /// error handling (i.e. less code).
  bool Diag(SourceLocation Loc, unsigned DiagID);
  bool Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg);
  bool Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg1,
            const std::string &Msg2);

  /// More expressive diagnostic helpers for expressions (say that 6 times:-)
  bool Diag(SourceLocation Loc, unsigned DiagID, SourceRange R1);
  bool Diag(SourceLocation Loc, unsigned DiagID, 
            SourceRange R1, SourceRange R2);
  bool Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg,
            SourceRange R1);
  bool Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg,
            SourceRange R1, SourceRange R2);
  bool Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg1, 
            const std::string &Msg2, SourceRange R1);
  bool Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg1, 
            const std::string &Msg2, const std::string &Msg3, SourceRange R1);
  bool Diag(SourceLocation Loc, unsigned DiagID, 
            const std::string &Msg1, const std::string &Msg2, 
            SourceRange R1, SourceRange R2);
  
  virtual void DeleteExpr(ExprTy *E);
  virtual void DeleteStmt(StmtTy *S);

  //===--------------------------------------------------------------------===//
  // Type Analysis / Processing: SemaType.cpp.
  //
  QualType ConvertDeclSpecToType(const DeclSpec &DS);
  void ProcessTypeAttributeList(QualType &Result, const AttributeList *AL);
  QualType GetTypeForDeclarator(Declarator &D, Scope *S);

  
  QualType ObjCGetTypeForMethodDefinition(DeclTy *D);

  
  virtual TypeResult ActOnTypeName(Scope *S, Declarator &D);
private:
  //===--------------------------------------------------------------------===//
  // Symbol table / Decl tracking callbacks: SemaDecl.cpp.
  //
  virtual DeclTy *isTypeName(const IdentifierInfo &II, Scope *S);
  virtual DeclTy *ActOnDeclarator(Scope *S, Declarator &D, DeclTy *LastInGroup);
  virtual DeclTy *ActOnParamDeclarator(Scope *S, Declarator &D);
  virtual void ActOnParamDefaultArgument(DeclTy *param, 
                                         SourceLocation EqualLoc,
                                         ExprTy *defarg);
  void AddInitializerToDecl(DeclTy *dcl, ExprTy *init);
  virtual DeclTy *FinalizeDeclaratorGroup(Scope *S, DeclTy *Group);

  virtual DeclTy *ActOnStartOfFunctionDef(Scope *S, Declarator &D);
  virtual DeclTy *ActOnStartOfFunctionDef(Scope *S, DeclTy *D);
  virtual void ObjCActOnStartOfMethodDef(Scope *S, DeclTy *D);
  
  virtual DeclTy *ActOnFinishFunctionBody(DeclTy *Decl, StmtTy *Body);
  virtual DeclTy *ActOnLinkageSpec(SourceLocation Loc, SourceLocation LBrace,
                                   SourceLocation RBrace, const char *Lang,
                                   unsigned StrSize, DeclTy *D);
  virtual DeclTy *ActOnFileScopeAsmDecl(SourceLocation Loc, ExprTy *expr);

  /// Scope actions.
  virtual void ActOnPopScope(SourceLocation Loc, Scope *S);
  virtual void ActOnTranslationUnitScope(SourceLocation Loc, Scope *S);

  /// ParsedFreeStandingDeclSpec - This method is invoked when a declspec with
  /// no declarator (e.g. "struct foo;") is parsed.
  virtual DeclTy *ParsedFreeStandingDeclSpec(Scope *S, DeclSpec &DS);  
  
  virtual DeclTy *ActOnTag(Scope *S, unsigned TagType, TagKind TK,
                           SourceLocation KWLoc, IdentifierInfo *Name,
                           SourceLocation NameLoc, AttributeList *Attr);
  virtual void ActOnDefs(Scope *S, SourceLocation DeclStart, IdentifierInfo
      *ClassName, llvm::SmallVectorImpl<DeclTy*> &Decls);
  virtual DeclTy *ActOnField(Scope *S, SourceLocation DeclStart,
                             Declarator &D, ExprTy *BitfieldWidth);
  
  virtual DeclTy *ActOnIvar(Scope *S, SourceLocation DeclStart,
                            Declarator &D, ExprTy *BitfieldWidth,
                            tok::ObjCKeywordKind visibility);

  // This is used for both record definitions and ObjC interface declarations.
  virtual void ActOnFields(Scope* S,
                           SourceLocation RecLoc, DeclTy *TagDecl,
                           DeclTy **Fields, unsigned NumFields,
                           SourceLocation LBrac, SourceLocation RBrac);
  virtual DeclTy *ActOnEnumConstant(Scope *S, DeclTy *EnumDecl,
                                    DeclTy *LastEnumConstant,
                                    SourceLocation IdLoc, IdentifierInfo *Id,
                                    SourceLocation EqualLoc, ExprTy *Val);
  virtual void ActOnEnumBody(SourceLocation EnumLoc, DeclTy *EnumDecl,
                             DeclTy **Elements, unsigned NumElements);
private:
  DeclContext *getDCParent(DeclContext *DC);

  /// Set the current declaration context until it gets popped.
  void PushDeclContext(DeclContext *DC);
  void PopDeclContext();
  
  /// CurFunctionDecl - If inside of a function body, this returns a pointer to
  /// the function decl for the function being parsed.
  FunctionDecl *getCurFunctionDecl() {
    return dyn_cast<FunctionDecl>(CurContext);
  }

  /// CurMethodDecl - If inside of a method body, this returns a pointer to
  /// the method decl for the method being parsed.
  ObjCMethodDecl *getCurMethodDecl() {
    return dyn_cast<ObjCMethodDecl>(CurContext);
  }

  /// Add this decl to the scope shadowed decl chains.
  void PushOnScopeChains(NamedDecl *D, Scope *S);

  /// Subroutines of ActOnDeclarator().
  TypedefDecl *ParseTypedefDecl(Scope *S, Declarator &D, QualType T,
                                ScopedDecl *LastDecl);
  TypedefDecl *MergeTypeDefDecl(TypedefDecl *New, Decl *Old);
  FunctionDecl *MergeFunctionDecl(FunctionDecl *New, Decl *Old, 
                                  bool &Redeclaration);
  VarDecl *MergeVarDecl(VarDecl *New, Decl *Old);
  FunctionDecl *MergeCXXFunctionDecl(FunctionDecl *New, FunctionDecl *Old);

  /// Helpers for dealing with function parameters
  bool CheckParmsForFunctionDef(FunctionDecl *FD);
  ImplicitParamDecl *CreateImplicitParameter(Scope *S, IdentifierInfo *Id, 
                                       SourceLocation IdLoc, QualType Type);
  void CheckCXXDefaultArguments(FunctionDecl *FD);
  void CheckExtraCXXDefaultArguments(Declarator &D);

  /// More parsing and symbol table subroutines...
  Decl *LookupDecl(const IdentifierInfo *II, unsigned NSI, Scope *S,
                   bool enableLazyBuiltinCreation = true);
  ObjCInterfaceDecl *getObjCInterfaceDecl(IdentifierInfo *Id);
  ScopedDecl *LazilyCreateBuiltin(IdentifierInfo *II, unsigned ID, 
                                  Scope *S);
  ScopedDecl *ImplicitlyDefineFunction(SourceLocation Loc, IdentifierInfo &II,
                                 Scope *S);
  // Decl attributes - this routine is the top level dispatcher. 
  void ProcessDeclAttributes(Decl *D, const Declarator &PD);
  void ProcessDeclAttributeList(Decl *D, const AttributeList *AttrList);

  void WarnUndefinedMethod(SourceLocation ImpLoc, ObjCMethodDecl *method,
                           bool &IncompleteImpl);
                           
  /// CheckProtocolMethodDefs - This routine checks unimpletented methods
  /// Declared in protocol, and those referenced by it.
  void CheckProtocolMethodDefs(SourceLocation ImpLoc,
                               ObjCProtocolDecl *PDecl,
                               bool& IncompleteImpl,
                               const llvm::DenseSet<Selector> &InsMap,
                               const llvm::DenseSet<Selector> &ClsMap);
  
  /// CheckImplementationIvars - This routine checks if the instance variables
  /// listed in the implelementation match those listed in the interface. 
  void CheckImplementationIvars(ObjCImplementationDecl *ImpDecl,
                                ObjCIvarDecl **Fields, unsigned nIvars,
                                SourceLocation Loc);
  
  /// ImplMethodsVsClassMethods - This is main routine to warn if any method
  /// remains unimplemented in the @implementation class.
  void ImplMethodsVsClassMethods(ObjCImplementationDecl* IMPDecl, 
                                 ObjCInterfaceDecl* IDecl);
  
  /// ImplCategoryMethodsVsIntfMethods - Checks that methods declared in the
  /// category interface is implemented in the category @implementation.
  void ImplCategoryMethodsVsIntfMethods(ObjCCategoryImplDecl *CatImplDecl,
                                        ObjCCategoryDecl *CatClassDecl);
  /// MatchTwoMethodDeclarations - Checks if two methods' type match and returns
  /// true, or false, accordingly.
  bool MatchTwoMethodDeclarations(const ObjCMethodDecl *Method, 
                                  const ObjCMethodDecl *PrevMethod); 

  /// isBuiltinObjCType - Returns true of the type is "id", "SEL", "Class"
  /// or "Protocol".
  bool isBuiltinObjCType(TypedefDecl *TD);

  /// AddInstanceMethodToGlobalPool - All instance methods in a translation
  /// unit are added to a global pool. This allows us to efficiently associate
  /// a selector with a method declaraation for purposes of typechecking
  /// messages sent to "id" (where the class of the object is unknown).
  void AddInstanceMethodToGlobalPool(ObjCMethodDecl *Method);
  
  /// AddFactoryMethodToGlobalPool - Same as above, but for factory methods.
  void AddFactoryMethodToGlobalPool(ObjCMethodDecl *Method);
  //===--------------------------------------------------------------------===//
  // Statement Parsing Callbacks: SemaStmt.cpp.
public:
  virtual StmtResult ActOnExprStmt(ExprTy *Expr);
  
  virtual StmtResult ActOnNullStmt(SourceLocation SemiLoc);
  virtual StmtResult ActOnCompoundStmt(SourceLocation L, SourceLocation R,
                                       StmtTy **Elts, unsigned NumElts,
                                       bool isStmtExpr);
  virtual StmtResult ActOnDeclStmt(DeclTy *Decl, SourceLocation StartLoc,
                                   SourceLocation EndLoc);
  virtual StmtResult ActOnCaseStmt(SourceLocation CaseLoc, ExprTy *LHSVal,
                                   SourceLocation DotDotDotLoc, ExprTy *RHSVal,
                                   SourceLocation ColonLoc, StmtTy *SubStmt);
  virtual StmtResult ActOnDefaultStmt(SourceLocation DefaultLoc,
                                      SourceLocation ColonLoc, StmtTy *SubStmt,
                                      Scope *CurScope);
  virtual StmtResult ActOnLabelStmt(SourceLocation IdentLoc, IdentifierInfo *II,
                                    SourceLocation ColonLoc, StmtTy *SubStmt);
  virtual StmtResult ActOnIfStmt(SourceLocation IfLoc, ExprTy *CondVal,
                                 StmtTy *ThenVal, SourceLocation ElseLoc,
                                 StmtTy *ElseVal);
  virtual StmtResult ActOnStartOfSwitchStmt(ExprTy *Cond);
  virtual StmtResult ActOnFinishSwitchStmt(SourceLocation SwitchLoc,
                                           StmtTy *Switch, ExprTy *Body);
  virtual StmtResult ActOnWhileStmt(SourceLocation WhileLoc, ExprTy *Cond,
                                    StmtTy *Body);
  virtual StmtResult ActOnDoStmt(SourceLocation DoLoc, StmtTy *Body,
                                 SourceLocation WhileLoc, ExprTy *Cond);
  
  virtual StmtResult ActOnForStmt(SourceLocation ForLoc, 
                                  SourceLocation LParenLoc, 
                                  StmtTy *First, ExprTy *Second, ExprTy *Third,
                                  SourceLocation RParenLoc, StmtTy *Body);
  virtual StmtResult ActOnObjCForCollectionStmt(SourceLocation ForColLoc, 
                                  SourceLocation LParenLoc, 
                                  StmtTy *First, ExprTy *Second,
                                  SourceLocation RParenLoc, StmtTy *Body);
  
  virtual StmtResult ActOnGotoStmt(SourceLocation GotoLoc,
                                   SourceLocation LabelLoc,
                                   IdentifierInfo *LabelII);
  virtual StmtResult ActOnIndirectGotoStmt(SourceLocation GotoLoc,
                                           SourceLocation StarLoc,
                                           ExprTy *DestExp);
  virtual StmtResult ActOnContinueStmt(SourceLocation ContinueLoc,
                                       Scope *CurScope);
  virtual StmtResult ActOnBreakStmt(SourceLocation GotoLoc, Scope *CurScope);
  
  virtual StmtResult ActOnReturnStmt(SourceLocation ReturnLoc,
                                     ExprTy *RetValExp);
  
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
                                  SourceLocation RParenLoc);
  
  virtual StmtResult ActOnObjCAtCatchStmt(SourceLocation AtLoc, 
                                          SourceLocation RParen, StmtTy *Parm, 
                                          StmtTy *Body, StmtTy *CatchList);
  
  virtual StmtResult ActOnObjCAtFinallyStmt(SourceLocation AtLoc, 
                                            StmtTy *Body);
  
  virtual StmtResult ActOnObjCAtTryStmt(SourceLocation AtLoc, 
                                        StmtTy *Try, 
                                        StmtTy *Catch, StmtTy *Finally);
  
  virtual StmtResult ActOnObjCAtThrowStmt(SourceLocation AtLoc, 
                                          StmtTy *Throw);
  virtual StmtResult ActOnObjCAtSynchronizedStmt(SourceLocation AtLoc, 
                                                 ExprTy *SynchExpr, 
                                                 StmtTy *SynchBody);
  
  //===--------------------------------------------------------------------===//
  // Expression Parsing Callbacks: SemaExpr.cpp.

  // Primary Expressions.
  virtual ExprResult ActOnIdentifierExpr(Scope *S, SourceLocation Loc,
                                         IdentifierInfo &II,
                                         bool HasTrailingLParen);
  virtual ExprResult ActOnPreDefinedExpr(SourceLocation Loc,
                                            tok::TokenKind Kind);
  virtual ExprResult ActOnNumericConstant(const Token &);
  virtual ExprResult ActOnCharacterConstant(const Token &);
  virtual ExprResult ActOnParenExpr(SourceLocation L, SourceLocation R,
                                    ExprTy *Val);

  /// ActOnStringLiteral - The specified tokens were lexed as pasted string
  /// fragments (e.g. "foo" "bar" L"baz").
  virtual ExprResult ActOnStringLiteral(const Token *Toks, unsigned NumToks);
    
  // Binary/Unary Operators.  'Tok' is the token for the operator.
  virtual ExprResult ActOnUnaryOp(SourceLocation OpLoc, tok::TokenKind Op,
                                  ExprTy *Input);
  virtual ExprResult 
    ActOnSizeOfAlignOfTypeExpr(SourceLocation OpLoc, bool isSizeof, 
                               SourceLocation LParenLoc, TypeTy *Ty,
                               SourceLocation RParenLoc);
  
  virtual ExprResult ActOnPostfixUnaryOp(SourceLocation OpLoc, 
                                         tok::TokenKind Kind, ExprTy *Input);
  
  virtual ExprResult ActOnArraySubscriptExpr(ExprTy *Base, SourceLocation LLoc,
                                             ExprTy *Idx, SourceLocation RLoc);
  virtual ExprResult ActOnMemberReferenceExpr(ExprTy *Base,SourceLocation OpLoc,
                                              tok::TokenKind OpKind,
                                              SourceLocation MemberLoc,
                                              IdentifierInfo &Member);
  
  /// ActOnCallExpr - Handle a call to Fn with the specified array of arguments.
  /// This provides the location of the left/right parens and a list of comma
  /// locations.
  virtual ExprResult ActOnCallExpr(ExprTy *Fn, SourceLocation LParenLoc,
                                   ExprTy **Args, unsigned NumArgs,
                                   SourceLocation *CommaLocs,
                                   SourceLocation RParenLoc);
  
  virtual ExprResult ActOnCastExpr(SourceLocation LParenLoc, TypeTy *Ty,
                                   SourceLocation RParenLoc, ExprTy *Op);
                                   
  virtual ExprResult ActOnCompoundLiteral(SourceLocation LParenLoc, TypeTy *Ty,
                                          SourceLocation RParenLoc, ExprTy *Op);
  
  virtual ExprResult ActOnInitList(SourceLocation LParenLoc, 
                                   ExprTy **InitList, unsigned NumInit,
                                   SourceLocation RParenLoc);
                                   
  virtual ExprResult ActOnBinOp(SourceLocation TokLoc, tok::TokenKind Kind,
                                ExprTy *LHS,ExprTy *RHS);
  
  /// ActOnConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
  /// in the case of a the GNU conditional expr extension.
  virtual ExprResult ActOnConditionalOp(SourceLocation QuestionLoc, 
                                        SourceLocation ColonLoc,
                                        ExprTy *Cond, ExprTy *LHS, ExprTy *RHS);

  /// ActOnAddrLabel - Parse the GNU address of label extension: "&&foo".
  virtual ExprResult ActOnAddrLabel(SourceLocation OpLoc, SourceLocation LabLoc,
                                    IdentifierInfo *LabelII);
  
  virtual ExprResult ActOnStmtExpr(SourceLocation LPLoc, StmtTy *SubStmt,
                                   SourceLocation RPLoc); // "({..})"

  /// __builtin_offsetof(type, a.b[123][456].c)
  virtual ExprResult ActOnBuiltinOffsetOf(SourceLocation BuiltinLoc,
                                          SourceLocation TypeLoc, TypeTy *Arg1,
                                          OffsetOfComponent *CompPtr,
                                          unsigned NumComponents,
                                          SourceLocation RParenLoc);
    
  // __builtin_types_compatible_p(type1, type2)
  virtual ExprResult ActOnTypesCompatibleExpr(SourceLocation BuiltinLoc, 
                                              TypeTy *arg1, TypeTy *arg2,
                                              SourceLocation RPLoc);
                                              
  // __builtin_choose_expr(constExpr, expr1, expr2)
  virtual ExprResult ActOnChooseExpr(SourceLocation BuiltinLoc, 
                                     ExprTy *cond, ExprTy *expr1, ExprTy *expr2,
                                     SourceLocation RPLoc);
  
  // __builtin_overload(...)
  virtual ExprResult ActOnOverloadExpr(ExprTy **Args, unsigned NumArgs,
                                       SourceLocation *CommaLocs,
                                       SourceLocation BuiltinLoc, 
                                       SourceLocation RParenLoc);

  // __builtin_va_arg(expr, type)
  virtual ExprResult ActOnVAArg(SourceLocation BuiltinLoc,
                                ExprTy *expr, TypeTy *type,
                                SourceLocation RPLoc);
  
  // Act on C++ namespaces
  virtual DeclTy *ActOnStartNamespaceDef(Scope *S, SourceLocation IdentLoc,
                                        IdentifierInfo *Ident,
                                        SourceLocation LBrace);
  virtual void ActOnFinishNamespaceDef(DeclTy *Dcl, SourceLocation RBrace);

  /// ActOnCXXCasts - Parse {dynamic,static,reinterpret,const}_cast's.
  virtual ExprResult ActOnCXXCasts(SourceLocation OpLoc, tok::TokenKind Kind,
                                   SourceLocation LAngleBracketLoc, TypeTy *Ty,
                                   SourceLocation RAngleBracketLoc,
                                   SourceLocation LParenLoc, ExprTy *E,
                                   SourceLocation RParenLoc);

  //// ActOnCXXThis -  Parse 'this' pointer.
  virtual ExprResult ActOnCXXThis(SourceLocation ThisLoc);

  /// ActOnCXXBoolLiteral - Parse {true,false} literals.
  virtual ExprResult ActOnCXXBoolLiteral(SourceLocation OpLoc,
                                         tok::TokenKind Kind);
  
  //// ActOnCXXThrow -  Parse throw expressions.
  virtual ExprResult ActOnCXXThrow(SourceLocation OpLoc,
                                   ExprTy *expr);

  // ParseObjCStringLiteral - Parse Objective-C string literals.
  virtual ExprResult ParseObjCStringLiteral(SourceLocation *AtLocs, 
                                            ExprTy **Strings,
                                            unsigned NumStrings);
  virtual ExprResult ParseObjCEncodeExpression(SourceLocation AtLoc,
                                               SourceLocation EncodeLoc,
                                               SourceLocation LParenLoc,
                                               TypeTy *Ty,
                                               SourceLocation RParenLoc);
  
  // ParseObjCSelectorExpression - Build selector expression for @selector
  virtual ExprResult ParseObjCSelectorExpression(Selector Sel,
                                                 SourceLocation AtLoc,
                                                 SourceLocation SelLoc,
                                                 SourceLocation LParenLoc,
                                                 SourceLocation RParenLoc);
  
  // ParseObjCProtocolExpression - Build protocol expression for @protocol
  virtual ExprResult ParseObjCProtocolExpression(IdentifierInfo * ProtocolName,
                                                 SourceLocation AtLoc,
                                                 SourceLocation ProtoLoc,
                                                 SourceLocation LParenLoc,
                                                 SourceLocation RParenLoc);

  //===--------------------------------------------------------------------===//
  // C++ Classes
  //
  /// ActOnBaseSpecifier - Parsed a base specifier
  virtual void ActOnBaseSpecifier(DeclTy *classdecl, SourceRange SpecifierRange,
                                  bool Virtual, AccessSpecifier Access,
                                  DeclTy *basetype, SourceLocation BaseLoc);
  
  virtual void ActOnStartCXXClassDef(Scope *S, DeclTy *TagDecl,
                                     SourceLocation LBrace);

  virtual DeclTy *ActOnCXXMemberDeclarator(Scope *S, AccessSpecifier AS,
                                           Declarator &D, ExprTy *BitfieldWidth,
                                           ExprTy *Init, DeclTy *LastInGroup);

  virtual void ActOnFinishCXXMemberSpecification(Scope* S, SourceLocation RLoc,
                                                 DeclTy *TagDecl,
                                                 SourceLocation LBrac,
                                                 SourceLocation RBrac);

  virtual void ActOnFinishCXXClassDef(DeclTy *TagDecl,SourceLocation RBrace);
  

  // Objective-C declarations.
  virtual DeclTy *ActOnStartClassInterface(
                    SourceLocation AtInterafceLoc,
                    IdentifierInfo *ClassName, SourceLocation ClassLoc,
                    IdentifierInfo *SuperName, SourceLocation SuperLoc,
                    const IdentifierLocPair *ProtocolNames,
                    unsigned NumProtocols,
                    SourceLocation EndProtoLoc, AttributeList *AttrList);
  
  virtual DeclTy *ActOnCompatiblityAlias(
                    SourceLocation AtCompatibilityAliasLoc,
                    IdentifierInfo *AliasName,  SourceLocation AliasLocation,
                    IdentifierInfo *ClassName, SourceLocation ClassLocation);
                    
  virtual DeclTy *ActOnStartProtocolInterface(
                    SourceLocation AtProtoInterfaceLoc,
                    IdentifierInfo *ProtocolName, SourceLocation ProtocolLoc,
                    DeclTy * const *ProtoRefNames, unsigned NumProtoRefs,
                    SourceLocation EndProtoLoc);
  
  virtual DeclTy *ActOnStartCategoryInterface(
                    SourceLocation AtInterfaceLoc,
                    IdentifierInfo *ClassName, SourceLocation ClassLoc,
                    IdentifierInfo *CategoryName, SourceLocation CategoryLoc,
                    const IdentifierLocPair *ProtoRefNames,
                    unsigned NumProtoRefs,
                    SourceLocation EndProtoLoc);
  
  virtual DeclTy *ActOnStartClassImplementation(
                    SourceLocation AtClassImplLoc,
                    IdentifierInfo *ClassName, SourceLocation ClassLoc,
                    IdentifierInfo *SuperClassname, 
                    SourceLocation SuperClassLoc);
  
  virtual DeclTy *ActOnStartCategoryImplementation(
                                                  SourceLocation AtCatImplLoc,
                                                  IdentifierInfo *ClassName, 
                                                  SourceLocation ClassLoc,
                                                  IdentifierInfo *CatName,
                                                  SourceLocation CatLoc);
  
  virtual DeclTy *ActOnForwardClassDeclaration(SourceLocation Loc,
                                               IdentifierInfo **IdentList,
                                               unsigned NumElts);
  
  virtual DeclTy *ActOnForwardProtocolDeclaration(SourceLocation AtProtocolLoc,
                                            const IdentifierLocPair *IdentList,
                                                  unsigned NumElts);
  
  virtual void FindProtocolDeclaration(bool WarnOnDeclarations,
                                       const IdentifierLocPair *ProtocolId,
                                       unsigned NumProtocols,
                                   llvm::SmallVectorImpl<DeclTy *> &Protocols);
  
  void DiagnosePropertyMismatch(ObjCPropertyDecl *Property, 
                                ObjCPropertyDecl *SuperProperty,
                                const char *Name);
  void ComparePropertiesInBaseAndSuper(ObjCInterfaceDecl *IDecl);
  
  void MergeProtocolPropertiesIntoClass(ObjCInterfaceDecl *IDecl,
                                        DeclTy *MergeProtocols);
  
  void MergeOneProtocolPropertiesIntoClass(ObjCInterfaceDecl *IDecl,
                                           ObjCProtocolDecl *PDecl);
  
  virtual void ActOnAtEnd(SourceLocation AtEndLoc, DeclTy *classDecl,
                      DeclTy **allMethods = 0, unsigned allNum = 0,
                      DeclTy **allProperties = 0, unsigned pNum = 0);
  
  virtual DeclTy *ActOnProperty(Scope *S, SourceLocation AtLoc,
                                FieldDeclarator &FD, ObjCDeclSpec &ODS,
                                Selector GetterSel, Selector SetterSel,
                                tok::ObjCKeywordKind MethodImplKind);
  
  virtual DeclTy *ActOnPropertyImplDecl(SourceLocation AtLoc, 
                                        SourceLocation PropertyLoc,
                                        bool ImplKind, DeclTy *ClassImplDecl,
                                        IdentifierInfo *PropertyId,
                                        IdentifierInfo *PropertyIvar);
  
  virtual DeclTy *ActOnMethodDeclaration(
    SourceLocation BeginLoc, // location of the + or -.
    SourceLocation EndLoc,   // location of the ; or {.
    tok::TokenKind MethodType, 
    DeclTy *ClassDecl, ObjCDeclSpec &ReturnQT, TypeTy *ReturnType, 
    Selector Sel,
    // optional arguments. The number of types/arguments is obtained
    // from the Sel.getNumArgs().
    ObjCDeclSpec *ArgQT, TypeTy **ArgTypes, IdentifierInfo **ArgNames,
    AttributeList *AttrList, tok::ObjCKeywordKind MethodImplKind,
    bool isVariadic = false);

  // ActOnClassMessage - used for both unary and keyword messages.
  // ArgExprs is optional - if it is present, the number of expressions
  // is obtained from NumArgs.
  virtual ExprResult ActOnClassMessage(
    Scope *S,
    IdentifierInfo *receivingClassName, Selector Sel,
    SourceLocation lbrac, SourceLocation rbrac, 
    ExprTy **ArgExprs, unsigned NumArgs);

  // ActOnInstanceMessage - used for both unary and keyword messages.
  // ArgExprs is optional - if it is present, the number of expressions
  // is obtained from NumArgs.
  virtual ExprResult ActOnInstanceMessage(
    ExprTy *receiver, Selector Sel,
    SourceLocation lbrac, SourceLocation rbrac, 
    ExprTy **ArgExprs, unsigned NumArgs);
private:
  /// ImpCastExprToType - If Expr is not of type 'Type', insert an implicit
  /// cast.  If there is already an implicit cast, merge into the existing one.
  void ImpCastExprToType(Expr *&Expr, QualType Type);

  // UsualUnaryConversions - promotes integers (C99 6.3.1.1p2) and converts
  // functions and arrays to their respective pointers (C99 6.3.2.1).
  Expr *UsualUnaryConversions(Expr *&expr); 

  // UsualUnaryConversionType - Same as UsualUnaryConversions, but works
  // on types instead of expressions
  QualType UsualUnaryConversionType(QualType Ty); 
  
  // DefaultFunctionArrayConversion - converts functions and arrays
  // to their respective pointers (C99 6.3.2.1). 
  void DefaultFunctionArrayConversion(Expr *&expr);
  
  // DefaultArgumentPromotion (C99 6.5.2.2p6). Used for function calls that
  // do not have a prototype. Integer promotions are performed on each 
  // argument, and arguments that have type float are promoted to double.
  void DefaultArgumentPromotion(Expr *&Expr);
  
  // UsualArithmeticConversions - performs the UsualUnaryConversions on it's
  // operands and then handles various conversions that are common to binary
  // operators (C99 6.3.1.8). If both operands aren't arithmetic, this
  // routine returns the first non-arithmetic type found. The client is 
  // responsible for emitting appropriate error diagnostics.
  QualType UsualArithmeticConversions(Expr *&lExpr, Expr *&rExpr,
                                      bool isCompAssign = false);
  
  /// AssignConvertType - All of the 'assignment' semantic checks return this
  /// enum to indicate whether the assignment was allowed.  These checks are
  /// done for simple assignments, as well as initialization, return from
  /// function, argument passing, etc.  The query is phrased in terms of a
  /// source and destination type.
  enum AssignConvertType {
    /// Compatible - the types are compatible according to the standard.
    Compatible,
    
    /// PointerToInt - The assignment converts a pointer to an int, which we
    /// accept as an extension.
    PointerToInt,
    
    /// IntToPointer - The assignment converts an int to a pointer, which we
    /// accept as an extension.
    IntToPointer,
    
    /// FunctionVoidPointer - The assignment is between a function pointer and
    /// void*, which the standard doesn't allow, but we accept as an extension.
    FunctionVoidPointer,

    /// IncompatiblePointer - The assignment is between two pointers types that
    /// are not compatible, but we accept them as an extension.
    IncompatiblePointer,
    
    /// CompatiblePointerDiscardsQualifiers - The assignment discards
    /// c/v/r qualifiers, which we accept as an extension.
    CompatiblePointerDiscardsQualifiers,
    
    /// Incompatible - We reject this conversion outright, it is invalid to
    /// represent it in the AST.
    Incompatible
  };
  
  /// DiagnoseAssignmentResult - Emit a diagnostic, if required, for the
  /// assignment conversion type specified by ConvTy.  This returns true if the
  /// conversion was invalid or false if the conversion was accepted.
  bool DiagnoseAssignmentResult(AssignConvertType ConvTy,
                                SourceLocation Loc,
                                QualType DstType, QualType SrcType,
                                Expr *SrcExpr, const char *Flavor);
  
  /// CheckAssignmentConstraints - Perform type checking for assignment, 
  /// argument passing, variable initialization, and function return values. 
  /// This routine is only used by the following two methods. C99 6.5.16.
  AssignConvertType CheckAssignmentConstraints(QualType lhs, QualType rhs);
  
  // CheckSingleAssignmentConstraints - Currently used by ActOnCallExpr,
  // CheckAssignmentOperands, and ActOnReturnStmt. Prior to type checking, 
  // this routine performs the default function/array converions.
  AssignConvertType CheckSingleAssignmentConstraints(QualType lhs, 
                                                     Expr *&rExpr);
  // CheckCompoundAssignmentConstraints - Type check without performing any 
  // conversions. For compound assignments, the "Check...Operands" methods 
  // perform the necessary conversions. 
  AssignConvertType CheckCompoundAssignmentConstraints(QualType lhs, 
                                                       QualType rhs);
  
  // Helper function for CheckAssignmentConstraints (C99 6.5.16.1p1)
  AssignConvertType CheckPointerTypesForAssignment(QualType lhsType, 
                                                   QualType rhsType);
  
  /// the following "Check" methods will return a valid/converted QualType
  /// or a null QualType (indicating an error diagnostic was issued).
    
  /// type checking binary operators (subroutines of ActOnBinOp).
  inline QualType InvalidOperands(SourceLocation l, Expr *&lex, Expr *&rex);
  inline QualType CheckMultiplyDivideOperands( // C99 6.5.5
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, bool isCompAssign = false); 
  inline QualType CheckRemainderOperands( // C99 6.5.5
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, bool isCompAssign = false); 
  inline QualType CheckAdditionOperands( // C99 6.5.6
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, bool isCompAssign = false);
  inline QualType CheckSubtractionOperands( // C99 6.5.6
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, bool isCompAssign = false);
  inline QualType CheckShiftOperands( // C99 6.5.7
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, bool isCompAssign = false);
  inline QualType CheckCompareOperands( // C99 6.5.8/9
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, bool isRelational);
  inline QualType CheckBitwiseOperands( // C99 6.5.[10...12]
    Expr *&lex, Expr *&rex, SourceLocation OpLoc, bool isCompAssign = false); 
  inline QualType CheckLogicalOperands( // C99 6.5.[13,14]
    Expr *&lex, Expr *&rex, SourceLocation OpLoc);
  // CheckAssignmentOperands is used for both simple and compound assignment.
  // For simple assignment, pass both expressions and a null converted type.
  // For compound assignment, pass both expressions and the converted type.
  inline QualType CheckAssignmentOperands( // C99 6.5.16.[1,2]
    Expr *lex, Expr *&rex, SourceLocation OpLoc, QualType convertedType);
  inline QualType CheckCommaOperands( // C99 6.5.17
    Expr *&lex, Expr *&rex, SourceLocation OpLoc);
  inline QualType CheckConditionalOperands( // C99 6.5.15
    Expr *&cond, Expr *&lhs, Expr *&rhs, SourceLocation questionLoc);

  /// type checking for vector binary operators.
  inline QualType CheckVectorOperands(SourceLocation l, Expr *&lex, Expr *&rex);
  inline QualType CheckVectorCompareOperands(Expr *&lex, Expr *&rx,
                                             SourceLocation l, bool isRel);
  
  /// type checking unary operators (subroutines of ActOnUnaryOp).
  /// C99 6.5.3.1, 6.5.3.2, 6.5.3.4
  QualType CheckIncrementDecrementOperand(Expr *op, SourceLocation OpLoc);   
  QualType CheckAddressOfOperand(Expr *op, SourceLocation OpLoc);
  QualType CheckIndirectionOperand(Expr *op, SourceLocation OpLoc);
  QualType CheckSizeOfAlignOfOperand(QualType type, SourceLocation OpLoc, 
                                     const SourceRange &R, bool isSizeof);
  QualType CheckRealImagOperand(Expr *&Op, SourceLocation OpLoc);
  
  /// type checking primary expressions.
  QualType CheckExtVectorComponent(QualType baseType, SourceLocation OpLoc,
                                   IdentifierInfo &Comp, SourceLocation CmpLoc);
  
  /// type checking declaration initializers (C99 6.7.8)
  friend class InitListChecker;
  bool CheckInitializerTypes(Expr *&simpleInit_or_initList, QualType &declType);
  bool CheckSingleInitializer(Expr *&simpleInit, QualType declType);
  bool CheckForConstantInitializer(Expr *e, QualType t);
  bool CheckArithmeticConstantExpression(const Expr* e);
  bool CheckAddressConstantExpression(const Expr* e);
  bool CheckAddressConstantExpressionLValue(const Expr* e);
  
  StringLiteral *IsStringLiteralInit(Expr *Init, QualType DeclType);
  bool CheckStringLiteralInit(StringLiteral *strLiteral, QualType &DeclT);
  
  // CheckVectorCast - check type constraints for vectors. 
  // Since vectors are an extension, there are no C standard reference for this.
  // We allow casting between vectors and integer datatypes of the same size.
  // returns true if the cast is invalid
  bool CheckVectorCast(SourceRange R, QualType VectorTy, QualType Ty);
  
  // returns true if there were any incompatible arguments.                           
  bool CheckMessageArgumentTypes(Expr **Args, unsigned NumArgs,
                                 ObjCMethodDecl *Method);
                    
  /// ConvertIntegerToTypeWarnOnOverflow - Convert the specified APInt to have
  /// the specified width and sign.  If an overflow occurs, detect it and emit
  /// the specified diagnostic.
  void ConvertIntegerToTypeWarnOnOverflow(llvm::APSInt &OldVal, 
                                          unsigned NewWidth, bool NewSign,
                                          SourceLocation Loc, unsigned DiagID);
  
  bool ObjCQualifiedIdTypesAreCompatible(QualType LHS, QualType RHS,
                                         bool ForCompare);

  
  void InitBuiltinVaListType();

  // Helper method to turn variable array types into
  // constant array types in certain situations which would otherwise
  // be errors
  QualType TryFixInvalidVariablyModifiedType(QualType T);
  
  //===--------------------------------------------------------------------===//
  // Extra semantic analysis beyond the C type system
private:
  Action::ExprResult CheckFunctionCall(FunctionDecl *FDecl, CallExpr *TheCall);
  bool CheckBuiltinCFStringArgument(Expr* Arg);
  bool SemaBuiltinVAStart(CallExpr *TheCall);
  bool SemaBuiltinUnorderedCompare(CallExpr *TheCall);
  bool SemaBuiltinStackAddress(CallExpr *TheCall);
  Action::ExprResult SemaBuiltinShuffleVector(CallExpr *TheCall);
  bool SemaBuiltinPrefetch(CallExpr *TheCall); 
  void CheckPrintfArguments(CallExpr *TheCall,
                            bool HasVAListArg, unsigned format_idx);
  void CheckReturnStackAddr(Expr *RetValExp, QualType lhsType,
                            SourceLocation ReturnLoc);
  void CheckFloatComparison(SourceLocation loc, Expr* lex, Expr* rex);
};

class InitListChecker {
  Sema *SemaRef;
  bool hadError;
  
  void CheckImplicitInitList(InitListExpr *ParentIList, QualType T, 
                             unsigned &Index);
  void CheckExplicitInitList(InitListExpr *IList, QualType &T,
                             unsigned &Index);

  void CheckListElementTypes(InitListExpr *IList, QualType &DeclType, 
                             unsigned &Index);
  void CheckSubElementType(InitListExpr *IList, QualType ElemType, 
                           unsigned &Index);
  // FIXME: Does DeclType need to be a reference type?
  void CheckScalarType(InitListExpr *IList, QualType &DeclType, 
                       unsigned &Index);
  void CheckVectorType(InitListExpr *IList, QualType DeclType, unsigned &Index);
  void CheckStructUnionTypes(InitListExpr *IList, QualType DeclType, 
                             unsigned &Index);
  void CheckArrayType(InitListExpr *IList, QualType &DeclType, unsigned &Index);
  
  int numArrayElements(QualType DeclType);
  int numStructUnionElements(QualType DeclType);
public:
  InitListChecker(Sema *S, InitListExpr *IL, QualType &T);
  bool HadError() { return hadError; }
};


}  // end namespace clang

#endif
