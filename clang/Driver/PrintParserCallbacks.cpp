//===--- PrintParserActions.cpp - Implement -parse-print-callbacks mode ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This code simply runs the preprocessor on the input file and prints out the
// result.  This is the traditional behavior of the -E option.
//
//===----------------------------------------------------------------------===//

#include "clang.h"
#include "clang/Parse/Action.h"
#include "clang/Parse/DeclSpec.h"
#include "llvm/Support/Streams.h"
using namespace clang;

namespace {
  class ParserPrintActions : public MinimalAction {
    
  public:
    ParserPrintActions(IdentifierTable &IT) : MinimalAction(IT) {}

    // Printing Functions which also must call MinimalAction

    /// ActOnDeclarator - This callback is invoked when a declarator is parsed
    /// and 'Init' specifies the initializer if any.  This is for things like:
    /// "int X = 4" or "typedef int foo".
    virtual DeclTy *ActOnDeclarator(Scope *S, Declarator &D,
                                    DeclTy *LastInGroup) {
      llvm::cout << __FUNCTION__ << " ";
      if (IdentifierInfo *II = D.getIdentifier()) {
        llvm::cout << "'" << II->getName() << "'";
      } else {
        llvm::cout << "<anon>";
      }
      llvm::cout << "\n";
      
      // Pass up to EmptyActions so that the symbol table is maintained right.
      return MinimalAction::ActOnDeclarator(S, D, LastInGroup);
    }
    /// ActOnPopScope - This callback is called immediately before the specified
    /// scope is popped and deleted.
    virtual void ActOnPopScope(SourceLocation Loc, Scope *S) {
      llvm::cout << __FUNCTION__ << "\n";
      return MinimalAction::ActOnPopScope(Loc, S);
    }

    /// ActOnTranslationUnitScope - This callback is called once, immediately
    /// after creating the translation unit scope (in Parser::Initialize).
    virtual void ActOnTranslationUnitScope(SourceLocation Loc, Scope *S) {
      llvm::cout << __FUNCTION__ << "\n";
      MinimalAction::ActOnTranslationUnitScope(Loc, S);
    }


    Action::DeclTy *ActOnStartClassInterface(SourceLocation AtInterfaceLoc,
                                             IdentifierInfo *ClassName,
                                             SourceLocation ClassLoc,
                                             IdentifierInfo *SuperName,
                                             SourceLocation SuperLoc,
                                             DeclTy * const *ProtoRefs,
                                             unsigned NumProtocols,
                                             SourceLocation EndProtoLoc,
                                             AttributeList *AttrList) {
      llvm::cout << __FUNCTION__ << "\n";
      return MinimalAction::ActOnStartClassInterface(AtInterfaceLoc,
                                                     ClassName, ClassLoc, 
                                                     SuperName, SuperLoc,
                                                     ProtoRefs, NumProtocols,
                                                     EndProtoLoc, AttrList);
    }

    /// ActOnForwardClassDeclaration - 
    /// Scope will always be top level file scope. 
    Action::DeclTy *ActOnForwardClassDeclaration(SourceLocation AtClassLoc,
                                                 IdentifierInfo **IdentList, 
                                                 unsigned NumElts) {
      llvm::cout << __FUNCTION__ << "\n";
      return MinimalAction::ActOnForwardClassDeclaration(AtClassLoc, IdentList,
                                                         NumElts);
    }

    // Pure Printing

    /// ActOnParamDeclarator - This callback is invoked when a parameter
    /// declarator is parsed. This callback only occurs for functions
    /// with prototypes. S is the function prototype scope for the
    /// parameters (C++ [basic.scope.proto]).
    virtual DeclTy *ActOnParamDeclarator(Scope *S, Declarator &D) {
      llvm::cout << __FUNCTION__ << " ";
      if (IdentifierInfo *II = D.getIdentifier()) {
        llvm::cout << "'" << II->getName() << "'";
      } else {
        llvm::cout << "<anon>";
      }
      llvm::cout << "\n";
      return 0;
    }
    
    /// AddInitializerToDecl - This action is called immediately after 
    /// ParseDeclarator (when an initializer is present). The code is factored 
    /// this way to make sure we are able to handle the following:
    ///   void func() { int xx = xx; }
    /// This allows ActOnDeclarator to register "xx" prior to parsing the
    /// initializer. The declaration above should still result in a warning, 
    /// since the reference to "xx" is uninitialized.
    virtual void AddInitializerToDecl(DeclTy *Dcl, ExprTy *Init) {
      llvm::cout << __FUNCTION__ << "\n";
    }

    /// FinalizeDeclaratorGroup - After a sequence of declarators are parsed, this
    /// gives the actions implementation a chance to process the group as a whole.
    virtual DeclTy *FinalizeDeclaratorGroup(Scope *S, DeclTy *Group) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }

    /// ActOnStartOfFunctionDef - This is called at the start of a function
    /// definition, instead of calling ActOnDeclarator.  The Declarator includes
    /// information about formal arguments that are part of this function.
    virtual DeclTy *ActOnStartOfFunctionDef(Scope *FnBodyScope, Declarator &D) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }

    /// ActOnStartOfFunctionDef - This is called at the start of a function
    /// definition, after the FunctionDecl has already been created.
    virtual DeclTy *ActOnStartOfFunctionDef(Scope *FnBodyScope, DeclTy *D) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }

    virtual void ObjCActOnStartOfMethodDef(Scope *FnBodyScope, DeclTy *D) {
      llvm::cout << __FUNCTION__ << "\n";
    }
  
    /// ActOnFunctionDefBody - This is called when a function body has completed
    /// parsing.  Decl is the DeclTy returned by ParseStartOfFunctionDef.
    virtual DeclTy *ActOnFinishFunctionBody(DeclTy *Decl, StmtTy *Body) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }

    virtual DeclTy *ActOnFileScopeAsmDecl(SourceLocation Loc, ExprTy *AsmString) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    /// ParsedFreeStandingDeclSpec - This method is invoked when a declspec with
    /// no declarator (e.g. "struct foo;") is parsed.
    virtual DeclTy *ParsedFreeStandingDeclSpec(Scope *S, DeclSpec &DS) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }

    virtual DeclTy *ActOnLinkageSpec(SourceLocation Loc, SourceLocation LBrace,
                                     SourceLocation RBrace, const char *Lang,
                                     unsigned StrSize, DeclTy *D) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    //===--------------------------------------------------------------------===//
    // Type Parsing Callbacks.
    //===--------------------------------------------------------------------===//
  
    virtual TypeResult ActOnTypeName(Scope *S, Declarator &D) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual DeclTy *ActOnTag(Scope *S, unsigned TagType, TagKind TK,
                             SourceLocation KWLoc, IdentifierInfo *Name,
                             SourceLocation NameLoc, AttributeList *Attr) {
      // TagType is an instance of DeclSpec::TST, indicating what kind of tag this
      // is (struct/union/enum/class).
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    /// Act on @defs() element found when parsing a structure.  ClassName is the
    /// name of the referenced class.   
    virtual void ActOnDefs(Scope *S, SourceLocation DeclStart,
                           IdentifierInfo *ClassName,
                           llvm::SmallVectorImpl<DeclTy*> &Decls) {
      llvm::cout << __FUNCTION__ << "\n";
    }

    virtual DeclTy *ActOnField(Scope *S, SourceLocation DeclStart,
                               Declarator &D, ExprTy *BitfieldWidth) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual DeclTy *ActOnIvar(Scope *S, SourceLocation DeclStart,
                              Declarator &D, ExprTy *BitfieldWidth,
                              tok::ObjCKeywordKind visibility) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual void ActOnFields(Scope* S, SourceLocation RecLoc, DeclTy *TagDecl,
                             DeclTy **Fields, unsigned NumFields, 
                             SourceLocation LBrac, SourceLocation RBrac) {
      llvm::cout << __FUNCTION__ << "\n";
    }
  
    virtual DeclTy *ActOnEnumConstant(Scope *S, DeclTy *EnumDecl,
                                      DeclTy *LastEnumConstant,
                                      SourceLocation IdLoc, IdentifierInfo *Id,
                                      SourceLocation EqualLoc, ExprTy *Val) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }

    virtual void ActOnEnumBody(SourceLocation EnumLoc, DeclTy *EnumDecl,
                               DeclTy **Elements, unsigned NumElements) {
      llvm::cout << __FUNCTION__ << "\n";
    }

    //===--------------------------------------------------------------------===//
    // Statement Parsing Callbacks.
    //===--------------------------------------------------------------------===//
  
    virtual StmtResult ActOnNullStmt(SourceLocation SemiLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual StmtResult ActOnCompoundStmt(SourceLocation L, SourceLocation R,
                                         StmtTy **Elts, unsigned NumElts,
                                         bool isStmtExpr) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
    virtual StmtResult ActOnDeclStmt(DeclTy *Decl, SourceLocation StartLoc,
                                     SourceLocation EndLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual StmtResult ActOnExprStmt(ExprTy *Expr) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtResult(Expr);
    }
  
    /// ActOnCaseStmt - Note that this handles the GNU 'case 1 ... 4' extension,
    /// which can specify an RHS value.
    virtual StmtResult ActOnCaseStmt(SourceLocation CaseLoc, ExprTy *LHSVal,
                                     SourceLocation DotDotDotLoc, ExprTy *RHSVal,
                                     SourceLocation ColonLoc, StmtTy *SubStmt) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
    virtual StmtResult ActOnDefaultStmt(SourceLocation DefaultLoc,
                                        SourceLocation ColonLoc, StmtTy *SubStmt,
                                        Scope *CurScope){
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual StmtResult ActOnLabelStmt(SourceLocation IdentLoc, IdentifierInfo *II,
                                      SourceLocation ColonLoc, StmtTy *SubStmt) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual StmtResult ActOnIfStmt(SourceLocation IfLoc, ExprTy *CondVal,
                                   StmtTy *ThenVal, SourceLocation ElseLoc,
                                   StmtTy *ElseVal) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0; 
    }
  
    virtual StmtResult ActOnStartOfSwitchStmt(ExprTy *Cond) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual StmtResult ActOnFinishSwitchStmt(SourceLocation SwitchLoc, 
                                             StmtTy *Switch, ExprTy *Body) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }

    virtual StmtResult ActOnWhileStmt(SourceLocation WhileLoc, ExprTy *Cond,
                                      StmtTy *Body) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
    virtual StmtResult ActOnDoStmt(SourceLocation DoLoc, StmtTy *Body,
                                   SourceLocation WhileLoc, ExprTy *Cond) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
    virtual StmtResult ActOnForStmt(SourceLocation ForLoc, 
                                    SourceLocation LParenLoc, 
                                    StmtTy *First, ExprTy *Second, ExprTy *Third,
                                    SourceLocation RParenLoc, StmtTy *Body) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
    virtual StmtResult ActOnObjCForCollectionStmt(SourceLocation ForColLoc, 
                                                  SourceLocation LParenLoc, 
                                                  StmtTy *First, ExprTy *Second,
                                                  SourceLocation RParenLoc, StmtTy *Body) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
    virtual StmtResult ActOnGotoStmt(SourceLocation GotoLoc,
                                     SourceLocation LabelLoc,
                                     IdentifierInfo *LabelII) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
    virtual StmtResult ActOnIndirectGotoStmt(SourceLocation GotoLoc,
                                             SourceLocation StarLoc,
                                             ExprTy *DestExp) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
    virtual StmtResult ActOnContinueStmt(SourceLocation ContinueLoc,
                                         Scope *CurScope) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
    virtual StmtResult ActOnBreakStmt(SourceLocation GotoLoc, Scope *CurScope) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
    virtual StmtResult ActOnReturnStmt(SourceLocation ReturnLoc,
                                       ExprTy *RetValExp) {
      llvm::cout << __FUNCTION__ << "\n";
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
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    // Objective-c statements
    virtual StmtResult ActOnObjCAtCatchStmt(SourceLocation AtLoc, 
                                            SourceLocation RParen, StmtTy *Parm, 
                                            StmtTy *Body, StmtTy *CatchList) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual StmtResult ActOnObjCAtFinallyStmt(SourceLocation AtLoc, 
                                              StmtTy *Body) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual StmtResult ActOnObjCAtTryStmt(SourceLocation AtLoc, 
                                          StmtTy *Try, 
                                          StmtTy *Catch, StmtTy *Finally) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual StmtResult ActOnObjCAtThrowStmt(SourceLocation AtLoc, 
                                            StmtTy *Throw) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual StmtResult ActOnObjCAtSynchronizedStmt(SourceLocation AtLoc, 
                                                   ExprTy *SynchExpr, 
                                                   StmtTy *SynchBody) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    //===--------------------------------------------------------------------===//
    // Expression Parsing Callbacks.
    //===--------------------------------------------------------------------===//
  
    // Primary Expressions.
  
    /// ActOnIdentifierExpr - Parse an identifier in expression context.
    /// 'HasTrailingLParen' indicates whether or not the identifier has a '('
    /// token immediately after it.
    virtual ExprResult ActOnIdentifierExpr(Scope *S, SourceLocation Loc,
                                           IdentifierInfo &II,
                                           bool HasTrailingLParen) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual ExprResult ActOnPredefinedExpr(SourceLocation Loc,
                                           tok::TokenKind Kind) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }

    virtual ExprResult ActOnCharacterConstant(const Token &) { 
      llvm::cout << __FUNCTION__ << "\n";
      return 0; 
    }

    virtual ExprResult ActOnNumericConstant(const Token &) { 
      llvm::cout << __FUNCTION__ << "\n";
      return 0; 
    }
  
    /// ActOnStringLiteral - The specified tokens were lexed as pasted string
    /// fragments (e.g. "foo" "bar" L"baz").
    virtual ExprResult ActOnStringLiteral(const Token *Toks, unsigned NumToks) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual ExprResult ActOnParenExpr(SourceLocation L, SourceLocation R,
                                      ExprTy *Val) {
      llvm::cout << __FUNCTION__ << "\n";
      return Val;  // Default impl returns operand.
    }
  
    // Postfix Expressions.
    virtual ExprResult ActOnPostfixUnaryOp(SourceLocation OpLoc, 
                                           tok::TokenKind Kind, ExprTy *Input) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
    virtual ExprResult ActOnArraySubscriptExpr(ExprTy *Base, SourceLocation LLoc,
                                               ExprTy *Idx, SourceLocation RLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
    virtual ExprResult ActOnMemberReferenceExpr(ExprTy *Base,SourceLocation OpLoc,
                                                tok::TokenKind OpKind,
                                                SourceLocation MemberLoc,
                                                IdentifierInfo &Member) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    /// ActOnCallExpr - Handle a call to Fn with the specified array of arguments.
    /// This provides the location of the left/right parens and a list of comma
    /// locations.  There are guaranteed to be one fewer commas than arguments,
    /// unless there are zero arguments.
    virtual ExprResult ActOnCallExpr(ExprTy *Fn, SourceLocation LParenLoc,
                                     ExprTy **Args, unsigned NumArgs,
                                     SourceLocation *CommaLocs,
                                     SourceLocation RParenLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    // Unary Operators.  'Tok' is the token for the operator.
    virtual ExprResult ActOnUnaryOp(SourceLocation OpLoc, tok::TokenKind Op,
                                    ExprTy *Input) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
    virtual ExprResult 
    ActOnSizeOfAlignOfTypeExpr(SourceLocation OpLoc, bool isSizeof, 
                               SourceLocation LParenLoc, TypeTy *Ty,
                               SourceLocation RParenLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual ExprResult ActOnCompoundLiteral(SourceLocation LParen, TypeTy *Ty,
                                            SourceLocation RParen, ExprTy *Op) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
    virtual ExprResult ActOnInitList(SourceLocation LParenLoc,
                                     ExprTy **InitList, unsigned NumInit,
                                     SourceLocation RParenLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
    virtual ExprResult ActOnCastExpr(SourceLocation LParenLoc, TypeTy *Ty,
                                     SourceLocation RParenLoc, ExprTy *Op) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual ExprResult ActOnBinOp(SourceLocation TokLoc, tok::TokenKind Kind,
                                  ExprTy *LHS, ExprTy *RHS) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }

    /// ActOnConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
    /// in the case of a the GNU conditional expr extension.
    virtual ExprResult ActOnConditionalOp(SourceLocation QuestionLoc, 
                                          SourceLocation ColonLoc,
                                          ExprTy *Cond, ExprTy *LHS, ExprTy *RHS){
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    //===---------------------- GNU Extension Expressions -------------------===//

    virtual ExprResult ActOnAddrLabel(SourceLocation OpLoc, SourceLocation LabLoc,
                                      IdentifierInfo *LabelII) { // "&&foo"
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual ExprResult ActOnStmtExpr(SourceLocation LPLoc, StmtTy *SubStmt,
                                     SourceLocation RPLoc) { // "({..})"
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    virtual ExprResult ActOnBuiltinOffsetOf(SourceLocation BuiltinLoc,
                                            SourceLocation TypeLoc, TypeTy *Arg1,
                                            OffsetOfComponent *CompPtr,
                                            unsigned NumComponents,
                                            SourceLocation RParenLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  
    // __builtin_types_compatible_p(type1, type2)
    virtual ExprResult ActOnTypesCompatibleExpr(SourceLocation BuiltinLoc, 
                                                TypeTy *arg1, TypeTy *arg2,
                                                SourceLocation RPLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
    // __builtin_choose_expr(constExpr, expr1, expr2)
    virtual ExprResult ActOnChooseExpr(SourceLocation BuiltinLoc, 
                                       ExprTy *cond, ExprTy *expr1, ExprTy *expr2,
                                       SourceLocation RPLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
    // __builtin_overload(...)
    virtual ExprResult ActOnOverloadExpr(ExprTy **Args, unsigned NumArgs,
                                         SourceLocation *CommaLocs,
                                         SourceLocation BuiltinLoc, 
                                         SourceLocation RPLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  

    // __builtin_va_arg(expr, type)
    virtual ExprResult ActOnVAArg(SourceLocation BuiltinLoc,
                                  ExprTy *expr, TypeTy *type,
                                  SourceLocation RPLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return 0;
    }
  };
}

MinimalAction *clang::CreatePrintParserActionsAction(IdentifierTable &IT) {
  return new ParserPrintActions(IT);
}
