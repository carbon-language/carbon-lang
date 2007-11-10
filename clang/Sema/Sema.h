//===--- Sema.h - Semantic Analysis & AST Building --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Sema class, which performs semantic analysis and
// builds ASTs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_SEMA_H
#define LLVM_CLANG_AST_SEMA_H

#include "clang/Parse/Action.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <vector>
#include <string>

namespace llvm {
  class APSInt;
}

namespace clang {
  class ASTContext;
  class Preprocessor;
  class Decl;
  class ScopedDecl;
  class Expr;
  class InitListExpr;
  class VarDecl;
  class ParmVarDecl;
  class TypedefDecl;
  class FunctionDecl;
  class QualType;
  struct LangOptions;
  struct DeclaratorChunk;
  class Token;
  class IntegerLiteral;
  class ArrayType;
  class LabelStmt;
  class SwitchStmt;
  class OCUVectorType;
  class TypedefDecl;
  class ObjcInterfaceDecl;
  class ObjcProtocolDecl;
  class ObjcImplementationDecl;
  class ObjcCategoryImplDecl;
  class ObjcCategoryDecl;
  class ObjcIvarDecl;
  class ObjcMethodDecl;

/// Sema - This implements semantic analysis and AST building for C.
class Sema : public Action {
  Preprocessor &PP;
  
  ASTContext &Context;
  
  /// CurFunctionDecl - If inside of a function body, this contains a pointer to
  /// the function decl for the function being parsed.
  FunctionDecl *CurFunctionDecl;
  
  /// LastInGroupList - This vector is populated when there are multiple
  /// declarators in a single decl group (e.g. "int A, B, C").  In this case,
  /// all but the last decl will be entered into this.  This is used by the
  /// ASTStreamer.
  std::vector<Decl*> &LastInGroupList;
  
  /// LabelMap - This is a mapping from label identifiers to the LabelStmt for
  /// it (which acts like the label decl in some ways).  Forward referenced
  /// labels have a LabelStmt created for them with a null location & SubStmt.
  llvm::DenseMap<IdentifierInfo*, LabelStmt*> LabelMap;
  
  llvm::SmallVector<SwitchStmt*, 8> SwitchStack;
  
  /// OCUVectorDecls - This is a list all the OCU vector types. This allows
  /// us to associate a raw vector type with one of the OCU type names.
  /// This is only necessary for issuing pretty diagnostics.
  llvm::SmallVector<TypedefDecl*, 24> OCUVectorDecls;

  /// ObjcImplementations - Keep track of all of the classes with
  /// @implementation's, so that we can emit errors on duplicates.
  llvm::SmallPtrSet<IdentifierInfo*, 8> ObjcImplementations;
  
  /// ObjcProtocols - Keep track of all protocol declarations declared
  /// with @protocol keyword, so that we can emit errors on duplicates and
  /// find the declarations when needed.
  llvm::DenseMap<IdentifierInfo*, ObjcProtocolDecl*> ObjcProtocols;
  
  // Enum values used by KnownFunctionIDs (see below).
  enum {
    id_printf,
    id_fprintf,
    id_sprintf,
    id_snprintf,
    id_asprintf,
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
  struct ObjcMethodList {
    ObjcMethodDecl *Method;
    ObjcMethodList *Next;
    
    ObjcMethodList() {
      Method = 0; 
      Next = 0;
    }
    ObjcMethodList(ObjcMethodDecl *M, ObjcMethodList *C) {
      Method = M;
      Next = C;
    }
  };
  /// Instance/Factory Method Pools - allows efficient lookup when typechecking
  /// messages to "id". We need to maintain a list, since selectors can have
  /// differing signatures across classes. In Cocoa, this happens to be 
  /// extremely uncommon (only 1% of selectors are "overloaded").
  llvm::DenseMap<Selector, ObjcMethodList> InstanceMethodPool;
  llvm::DenseMap<Selector, ObjcMethodList> FactoryMethodPool;
public:
  Sema(Preprocessor &pp, ASTContext &ctxt, std::vector<Decl*> &prevInGroup);
  
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
  bool Diag(SourceLocation Loc, unsigned DiagID, 
            const std::string &Msg1, const std::string &Msg2, 
            SourceRange R1, SourceRange R2);
  
  virtual void DeleteExpr(ExprTy *E);
  virtual void DeleteStmt(StmtTy *S);

  //===--------------------------------------------------------------------===//
  // Type Analysis / Processing: SemaType.cpp.
  //
  QualType GetTypeForDeclarator(Declarator &D, Scope *S);
  
  QualType ObjcGetTypeForMethodDefinition(DeclTy *D, Scope *S);

  
  virtual TypeResult ActOnTypeName(Scope *S, Declarator &D);
  
  virtual TypeResult ActOnParamDeclaratorType(Scope *S, Declarator &D);
private:
  //===--------------------------------------------------------------------===//
  // Symbol table / Decl tracking callbacks: SemaDecl.cpp.
  //
  virtual DeclTy *isTypeName(const IdentifierInfo &II, Scope *S) const;
  virtual DeclTy *ActOnDeclarator(Scope *S, Declarator &D, DeclTy *LastInGroup);
  virtual DeclTy *ObjcActOnMethodDefinition(Scope *S, DeclTy *D, 
					    DeclTy *LastInGroup);
  void AddInitializerToDecl(DeclTy *dcl, ExprTy *init);
  virtual DeclTy *FinalizeDeclaratorGroup(Scope *S, DeclTy *Group);

  virtual DeclTy *ActOnStartOfFunctionDef(Scope *S, Declarator &D);
  virtual void ObjcActOnStartOfMethodDef(Scope *S, DeclTy *D);
  virtual DeclTy *ActOnFunctionDefBody(DeclTy *Decl, StmtTy *Body);
  virtual void ActOnMethodDefBody(DeclTy *Decl, StmtTy *Body);
  
  /// Scope actions.
  virtual void ActOnPopScope(SourceLocation Loc, Scope *S);
  virtual void ActOnTranslationUnitScope(SourceLocation Loc, Scope *S);

  /// ParsedFreeStandingDeclSpec - This method is invoked when a declspec with
  /// no declarator (e.g. "struct foo;") is parsed.
  virtual DeclTy *ParsedFreeStandingDeclSpec(Scope *S, DeclSpec &DS);  
  
  virtual DeclTy *ActOnTag(Scope *S, unsigned TagType, TagKind TK,
                           SourceLocation KWLoc, IdentifierInfo *Name,
                           SourceLocation NameLoc, AttributeList *Attr);
  virtual DeclTy *ActOnField(Scope *S, DeclTy *TagDecl,SourceLocation DeclStart,
                             Declarator &D, ExprTy *BitfieldWidth);
                                      
  // This is used for both record definitions and ObjC interface declarations.
  virtual void ActOnFields(Scope* S,
                           SourceLocation RecLoc, DeclTy *TagDecl,
                           DeclTy **Fields, unsigned NumFields,
                           SourceLocation LBrac, SourceLocation RBrac,
                           tok::ObjCKeywordKind *visibility = 0);
  virtual DeclTy *ActOnEnumConstant(Scope *S, DeclTy *EnumDecl,
                                    DeclTy *LastEnumConstant,
                                    SourceLocation IdLoc, IdentifierInfo *Id,
                                    SourceLocation EqualLoc, ExprTy *Val);
  virtual void ActOnEnumBody(SourceLocation EnumLoc, DeclTy *EnumDecl,
                             DeclTy **Elements, unsigned NumElements);
private:
  /// Subroutines of ActOnDeclarator()...
  TypedefDecl *ParseTypedefDecl(Scope *S, Declarator &D, ScopedDecl *LastDecl);
  TypedefDecl *MergeTypeDefDecl(TypedefDecl *New, ScopedDecl *Old);
  FunctionDecl *MergeFunctionDecl(FunctionDecl *New, ScopedDecl *Old);
  VarDecl *MergeVarDecl(VarDecl *New, ScopedDecl *Old);
  /// AddTopLevelDecl - called after the decl has been fully processed.
  /// Allows for bookkeeping and post-processing of each declaration.
  void AddTopLevelDecl(Decl *current, Decl *last);

  /// More parsing and symbol table subroutines...
  ParmVarDecl *ParseParamDeclarator(DeclaratorChunk &FI, unsigned ArgNo,
                                    Scope *FnBodyScope);
  ParmVarDecl *ObjcBuildMethodParameter(ParmVarDecl *param, Scope *FnBodyScope);
  
  ScopedDecl *LookupScopedDecl(IdentifierInfo *II, unsigned NSI, 
                               SourceLocation IdLoc, Scope *S);
  ScopedDecl *LookupInterfaceDecl(IdentifierInfo *II);
  ObjcInterfaceDecl *getObjCInterfaceDecl(IdentifierInfo *Id);
  ScopedDecl *LazilyCreateBuiltin(IdentifierInfo *II, unsigned ID, Scope *S);
  ScopedDecl *ImplicitlyDefineFunction(SourceLocation Loc, IdentifierInfo &II,
                                 Scope *S);
  // Decl attributes - this routine is the top level dispatcher. 
  void HandleDeclAttributes(Decl *New, AttributeList *declspec_prefix,
                            AttributeList *declarator_postfix);
  void HandleDeclAttribute(Decl *New, AttributeList *rawAttr);
                                
  // HandleVectorTypeAttribute - this attribute is only applicable to 
  // integral and float scalars, although arrays, pointers, and function
  // return values are allowed in conjunction with this construct. Aggregates
  // with this attribute are invalid, even if they are of the same size as a
  // corresponding scalar.
  // The raw attribute should contain precisely 1 argument, the vector size 
  // for the variable, measured in bytes. If curType and rawAttr are well
  // formed, this routine will return a new vector type.
  QualType HandleVectorTypeAttribute(QualType curType, AttributeList *rawAttr);
  void HandleOCUVectorTypeAttribute(TypedefDecl *d, AttributeList *rawAttr);
  
  /// CheckProtocolMethodDefs - This routine checks unimpletented methods
  /// Declared in protocol, and those referenced by it.
  void CheckProtocolMethodDefs(ObjcProtocolDecl *PDecl,
                               bool& IncompleteImpl,
                               const llvm::DenseSet<Selector> &InsMap,
                               const llvm::DenseSet<Selector> &ClsMap);
  
  /// CheckImplementationIvars - This routine checks if the instance variables
  /// listed in the implelementation match those listed in the interface. 
  void CheckImplementationIvars(ObjcImplementationDecl *ImpDecl,
                                ObjcIvarDecl **Fields, unsigned nIvars,
				SourceLocation Loc);
  
  /// ImplMethodsVsClassMethods - This is main routine to warn if any method
  /// remains unimplemented in the @implementation class.
  void ImplMethodsVsClassMethods(ObjcImplementationDecl* IMPDecl, 
                                 ObjcInterfaceDecl* IDecl);
  
  /// ImplCategoryMethodsVsIntfMethods - Checks that methods declared in the
  /// category interface is implemented in the category @implementation.
  void ImplCategoryMethodsVsIntfMethods(ObjcCategoryImplDecl *CatImplDecl,
                                        ObjcCategoryDecl *CatClassDecl);
  /// MatchTwoMethodDeclarations - Checks if two methods' type match and returns
  /// true, or false, accordingly.
  bool MatchTwoMethodDeclarations(const ObjcMethodDecl *Method, 
                                  const ObjcMethodDecl *PrevMethod); 

  /// GetObjcSelType - Getter for the build-in "Protocol *" type.
  QualType GetObjcProtoType(SourceLocation Loc = SourceLocation());
  
  /// isBuiltinObjcType - Returns true of the type is "id", "SEL", "Class".
  bool isBuiltinObjcType(TypedefDecl *TD);

  /// AddInstanceMethodToGlobalPool - All instance methods in a translation
  /// unit are added to a global pool. This allows us to efficiently associate
  /// a selector with a method declaraation for purposes of typechecking
  /// messages sent to "id" (where the class of the object is unknown).
  void AddInstanceMethodToGlobalPool(ObjcMethodDecl *Method);
  
  /// AddFactoryMethodToGlobalPool - Same as above, but for factory methods.
  void AddFactoryMethodToGlobalPool(ObjcMethodDecl *Method);
  //===--------------------------------------------------------------------===//
  // Statement Parsing Callbacks: SemaStmt.cpp.
public:
  virtual StmtResult ActOnExprStmt(ExprTy *Expr);
  
  virtual StmtResult ActOnNullStmt(SourceLocation SemiLoc);
  virtual StmtResult ActOnCompoundStmt(SourceLocation L, SourceLocation R,
                                       StmtTy **Elts, unsigned NumElts,
                                       bool isStmtExpr);
  virtual StmtResult ActOnDeclStmt(DeclTy *Decl);
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
                                  SourceLocation RParenLoc);
  
  virtual StmtResult ActOnObjcAtCatchStmt(SourceLocation AtLoc, 
                                          SourceLocation RParen, StmtTy *Parm, 
                                          StmtTy *Body, StmtTy *CatchList);
  
  virtual StmtResult ActOnObjcAtFinallyStmt(SourceLocation AtLoc, 
                                            StmtTy *Body);
  
  virtual StmtResult ActOnObjcAtTryStmt(SourceLocation AtLoc, 
                                        StmtTy *Try, 
                                        StmtTy *Catch, StmtTy *Finally);
  
  virtual StmtResult ActOnObjcAtThrowStmt(SourceLocation AtLoc, 
                                          StmtTy *Throw);
  
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
  
  // __builtin_va_arg(expr, type)
  virtual ExprResult ActOnVAArg(SourceLocation BuiltinLoc,
                                ExprTy *expr, TypeTy *type,
                                SourceLocation RPLoc);
  
  /// ActOnCXXCasts - Parse {dynamic,static,reinterpret,const}_cast's.
  virtual ExprResult ActOnCXXCasts(SourceLocation OpLoc, tok::TokenKind Kind,
                                   SourceLocation LAngleBracketLoc, TypeTy *Ty,
                                   SourceLocation RAngleBracketLoc,
                                   SourceLocation LParenLoc, ExprTy *E,
                                   SourceLocation RParenLoc);

  /// ActOnCXXBoolLiteral - Parse {true,false} literals.
  virtual ExprResult ActOnCXXBoolLiteral(SourceLocation OpLoc,
                                         tok::TokenKind Kind);
  
  // ParseObjCStringLiteral - Parse Objective-C string literals.
  virtual ExprResult ParseObjCStringLiteral(SourceLocation AtLoc,
                                            ExprTy *string);
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
  
  // Objective-C declarations.
  virtual DeclTy *ActOnStartClassInterface(
		    SourceLocation AtInterafceLoc,
                    IdentifierInfo *ClassName, SourceLocation ClassLoc,
                    IdentifierInfo *SuperName, SourceLocation SuperLoc,
                    IdentifierInfo **ProtocolNames, unsigned NumProtocols,
                    SourceLocation EndProtoLoc, AttributeList *AttrList);
  
  virtual DeclTy *ActOnCompatiblityAlias(
                    SourceLocation AtCompatibilityAliasLoc,
                    IdentifierInfo *AliasName,  SourceLocation AliasLocation,
                    IdentifierInfo *ClassName, SourceLocation ClassLocation);
                    
  virtual DeclTy *ActOnStartProtocolInterface(
		    SourceLocation AtProtoInterfaceLoc,
                    IdentifierInfo *ProtocolName, SourceLocation ProtocolLoc,
                    IdentifierInfo **ProtoRefNames, unsigned NumProtoRefs,
                    SourceLocation EndProtoLoc);
  
  virtual DeclTy *ActOnStartCategoryInterface(
		    SourceLocation AtInterfaceLoc,
                    IdentifierInfo *ClassName, SourceLocation ClassLoc,
                    IdentifierInfo *CategoryName, SourceLocation CategoryLoc,
                    IdentifierInfo **ProtoRefNames, unsigned NumProtoRefs,
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
                                                  IdentifierInfo **IdentList,
                                                  unsigned NumElts);
  
  virtual void FindProtocolDeclaration(SourceLocation TypeLoc,
                                       IdentifierInfo **ProtocolId,
                                       unsigned NumProtocols,
                                       llvm::SmallVector<DeclTy *, 8> & 
                                       Protocols);

  virtual void ActOnAddMethodsToObjcDecl(Scope* S, DeclTy *ClassDecl, 
				         DeclTy **allMethods, unsigned allNum,
                                         DeclTy **allProperties, unsigned pNum,
                                         SourceLocation AtEndLoc);
  
  virtual DeclTy *ActOnAddObjcProperties(SourceLocation AtLoc, 
                                         DeclTy **allProperties,
                                         unsigned NumProperties,
                                         ObjcDeclSpec &DS);
  
  virtual DeclTy *ActOnMethodDeclaration(
    SourceLocation BeginLoc, // location of the + or -.
    SourceLocation EndLoc,   // location of the ; or {.
    tok::TokenKind MethodType, 
    DeclTy *ClassDecl, ObjcDeclSpec &ReturnQT, TypeTy *ReturnType, 
    Selector Sel,
    // optional arguments. The number of types/arguments is obtained
    // from the Sel.getNumArgs().
    ObjcDeclSpec *ArgQT, TypeTy **ArgTypes, IdentifierInfo **ArgNames,
    AttributeList *AttrList, tok::ObjCKeywordKind MethodImplKind);

  // ActOnClassMessage - used for both unary and keyword messages.
  // ArgExprs is optional - if it is present, the number of expressions
  // is obtained from Sel.getNumArgs().
  virtual ExprResult ActOnClassMessage(
    IdentifierInfo *receivingClassName, Selector Sel,
    SourceLocation lbrac, SourceLocation rbrac, ExprTy **ArgExprs);

  // ActOnInstanceMessage - used for both unary and keyword messages.
  // ArgExprs is optional - if it is present, the number of expressions
  // is obtained from Sel.getNumArgs().
  virtual ExprResult ActOnInstanceMessage(
    ExprTy *receiver, Selector Sel,
    SourceLocation lbrac, SourceLocation rbrac, ExprTy **ArgExprs);
private:
  // UsualUnaryConversions - promotes integers (C99 6.3.1.1p2) and converts
  // functions and arrays to their respective pointers (C99 6.3.2.1). 
  void UsualUnaryConversions(Expr *&expr); 
  
  // DefaultFunctionArrayConversion - converts functions and arrays
  // to their respective pointers (C99 6.3.2.1). 
  void DefaultFunctionArrayConversion(Expr *&expr);
  
  // DefaultArgumentPromotion (C99 6.5.2.2p6). Used for function calls that
  // do not have a prototype. Integer promotions are performed on each 
  // argument, and arguments that have type float are promoted to double.
  void DefaultArgumentPromotion(Expr *&expr);
  
  // UsualArithmeticConversions - performs the UsualUnaryConversions on it's
  // operands and then handles various conversions that are common to binary
  // operators (C99 6.3.1.8). If both operands aren't arithmetic, this
  // routine returns the first non-arithmetic type found. The client is 
  // responsible for emitting appropriate error diagnostics.
  QualType UsualArithmeticConversions(Expr *&lExpr, Expr *&rExpr,
                                      bool isCompAssign = false);
  enum AssignmentCheckResult {
    Compatible,
    Incompatible,
    PointerFromInt, 
    IntFromPointer,
    IncompatiblePointer,
    CompatiblePointerDiscardsQualifiers
  };
  // CheckAssignmentConstraints - Perform type checking for assignment, 
  // argument passing, variable initialization, and function return values. 
  // This routine is only used by the following two methods. C99 6.5.16.
  AssignmentCheckResult CheckAssignmentConstraints(QualType lhs, QualType rhs);
  
  // CheckSingleAssignmentConstraints - Currently used by ActOnCallExpr,
  // CheckAssignmentOperands, and ActOnReturnStmt. Prior to type checking, 
  // this routine performs the default function/array converions.
  AssignmentCheckResult CheckSingleAssignmentConstraints(QualType lhs, 
                                                         Expr *&rExpr);
  // CheckCompoundAssignmentConstraints - Type check without performing any 
  // conversions. For compound assignments, the "Check...Operands" methods 
  // perform the necessary conversions. 
  AssignmentCheckResult CheckCompoundAssignmentConstraints(QualType lhs, 
                                                           QualType rhs);
  
  // Helper function for CheckAssignmentConstraints (C99 6.5.16.1p1)
  AssignmentCheckResult CheckPointerTypesForAssignment(QualType lhsType, 
                                                       QualType rhsType);
  
  /// the following "Check" methods will return a valid/converted QualType
  /// or a null QualType (indicating an error diagnostic was issued).
    
  /// type checking binary operators (subroutines of ActOnBinOp).
  inline void InvalidOperands(SourceLocation l, Expr *&lex, Expr *&rex);
  inline QualType CheckVectorOperands(SourceLocation l, Expr *&lex, Expr *&rex);
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
  
  /// type checking unary operators (subroutines of ActOnUnaryOp).
  /// C99 6.5.3.1, 6.5.3.2, 6.5.3.4
  QualType CheckIncrementDecrementOperand(Expr *op, SourceLocation OpLoc);   
  QualType CheckAddressOfOperand(Expr *op, SourceLocation OpLoc);
  QualType CheckIndirectionOperand(Expr *op, SourceLocation OpLoc);
  QualType CheckSizeOfAlignOfOperand(QualType type, SourceLocation loc, 
                                     bool isSizeof);
  QualType CheckRealImagOperand(Expr *&Op, SourceLocation OpLoc);
  
  /// type checking primary expressions.
  QualType CheckOCUVectorComponent(QualType baseType, SourceLocation OpLoc,
                                   IdentifierInfo &Comp, SourceLocation CmpLoc);
  
  /// type checking declaration initializers (C99 6.7.8)
  bool CheckInitializer(Expr *&simpleInit_or_initList, QualType &declType,
                        bool isStatic);
  bool CheckSingleInitializer(Expr *&simpleInit, bool isStatic, 
                              QualType declType);
  bool CheckInitExpr(Expr *expr, InitListExpr *IList, unsigned slot,
                     bool isStatic, QualType ElementType);
  void CheckVariableInitList(QualType DeclType, InitListExpr *IList, 
                             QualType ElementType, bool isStatic, 
                             int &nInitializers, bool &hadError);
  void CheckConstantInitList(QualType DeclType, InitListExpr *IList, 
                             QualType ElementType, bool isStatic, 
                             int &nInitializers, bool &hadError);
                             
  // returns true if there were any incompatible arguments.                           
  bool CheckMessageArgumentTypes(Expr **Args, unsigned NumArgs,
                                 ObjcMethodDecl *Method);
                    
  /// ConvertIntegerToTypeWarnOnOverflow - Convert the specified APInt to have
  /// the specified width and sign.  If an overflow occurs, detect it and emit
  /// the specified diagnostic.
  void ConvertIntegerToTypeWarnOnOverflow(llvm::APSInt &OldVal, 
                                          unsigned NewWidth, bool NewSign,
                                          SourceLocation Loc, unsigned DiagID);
  
  void InitBuiltinVaListType();
  
  //===--------------------------------------------------------------------===//
  // Extra semantic analysis beyond the C type system
  private:
  
  bool CheckFunctionCall(Expr *Fn,
                         SourceLocation LParenLoc, SourceLocation RParenLoc,
                         FunctionDecl *FDecl,
                         Expr** Args, unsigned NumArgsInCall);

  void CheckPrintfArguments(Expr *Fn,
                            SourceLocation LParenLoc, SourceLocation RParenLoc,
                            bool HasVAListArg, FunctionDecl *FDecl,
                            unsigned format_idx, Expr** Args,
                            unsigned NumArgsInCall);
                            
  void CheckReturnStackAddr(Expr *RetValExp, QualType lhsType,
                            SourceLocation ReturnLoc);

  
  bool CheckBuiltinCFStringArgument(Expr* Arg);
};


}  // end namespace clang

#endif
