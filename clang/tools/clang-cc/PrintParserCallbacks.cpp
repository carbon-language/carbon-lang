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

#include "clang-cc.h"
#include "clang/Parse/Action.h"
#include "clang/Parse/DeclSpec.h"
#include "llvm/Support/Streams.h"
using namespace clang;

namespace {
  class ParserPrintActions : public MinimalAction {
    
  public:
    ParserPrintActions(Preprocessor &PP) : MinimalAction(PP) {}

    // Printing Functions which also must call MinimalAction

    /// ActOnDeclarator - This callback is invoked when a declarator is parsed
    /// and 'Init' specifies the initializer if any.  This is for things like:
    /// "int X = 4" or "typedef int foo".
    virtual DeclPtrTy ActOnDeclarator(Scope *S, Declarator &D) {
      llvm::cout << __FUNCTION__ << " ";
      if (IdentifierInfo *II = D.getIdentifier()) {
        llvm::cout << "'" << II->getName() << "'";
      } else {
        llvm::cout << "<anon>";
      }
      llvm::cout << "\n";
      
      // Pass up to EmptyActions so that the symbol table is maintained right.
      return MinimalAction::ActOnDeclarator(S, D);
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


    Action::DeclPtrTy ActOnStartClassInterface(SourceLocation AtInterfaceLoc,
                                               IdentifierInfo *ClassName,
                                               SourceLocation ClassLoc,
                                               IdentifierInfo *SuperName,
                                               SourceLocation SuperLoc,
                                               const DeclPtrTy *ProtoRefs,
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
    Action::DeclPtrTy ActOnForwardClassDeclaration(SourceLocation AtClassLoc,
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
    virtual DeclPtrTy ActOnParamDeclarator(Scope *S, Declarator &D) {
      llvm::cout << __FUNCTION__ << " ";
      if (IdentifierInfo *II = D.getIdentifier()) {
        llvm::cout << "'" << II->getName() << "'";
      } else {
        llvm::cout << "<anon>";
      }
      llvm::cout << "\n";
      return DeclPtrTy();
    }
    
    /// AddInitializerToDecl - This action is called immediately after 
    /// ParseDeclarator (when an initializer is present). The code is factored 
    /// this way to make sure we are able to handle the following:
    ///   void func() { int xx = xx; }
    /// This allows ActOnDeclarator to register "xx" prior to parsing the
    /// initializer. The declaration above should still result in a warning, 
    /// since the reference to "xx" is uninitialized.
    virtual void AddInitializerToDecl(DeclPtrTy Dcl, ExprArg Init) {
      llvm::cout << __FUNCTION__ << "\n";
    }

    /// FinalizeDeclaratorGroup - After a sequence of declarators are parsed,
    /// this gives the actions implementation a chance to process the group as
    /// a whole.
    virtual DeclGroupPtrTy FinalizeDeclaratorGroup(Scope *S, DeclPtrTy *Group,
                                                   unsigned NumDecls) {
      llvm::cout << __FUNCTION__ << "\n";
      return DeclGroupPtrTy();
    }

    /// ActOnStartOfFunctionDef - This is called at the start of a function
    /// definition, instead of calling ActOnDeclarator.  The Declarator includes
    /// information about formal arguments that are part of this function.
    virtual DeclPtrTy ActOnStartOfFunctionDef(Scope *FnBodyScope,
                                              Declarator &D){
      llvm::cout << __FUNCTION__ << "\n";
      return DeclPtrTy();
    }

    /// ActOnStartOfFunctionDef - This is called at the start of a function
    /// definition, after the FunctionDecl has already been created.
    virtual DeclPtrTy ActOnStartOfFunctionDef(Scope *FnBodyScope, DeclPtrTy D) {
      llvm::cout << __FUNCTION__ << "\n";
      return DeclPtrTy();
    }

    virtual void ActOnStartOfObjCMethodDef(Scope *FnBodyScope, DeclPtrTy D) {
      llvm::cout << __FUNCTION__ << "\n";
    }
  
    /// ActOnFunctionDefBody - This is called when a function body has completed
    /// parsing.  Decl is the DeclTy returned by ParseStartOfFunctionDef.
    virtual DeclPtrTy ActOnFinishFunctionBody(DeclPtrTy Decl, StmtArg Body) {
      llvm::cout << __FUNCTION__ << "\n";
      return DeclPtrTy();
    }

    virtual DeclPtrTy ActOnFileScopeAsmDecl(SourceLocation Loc,
                                            ExprArg AsmString) {
      llvm::cout << __FUNCTION__ << "\n";
      return DeclPtrTy();
    }
  
    /// ParsedFreeStandingDeclSpec - This method is invoked when a declspec with
    /// no declarator (e.g. "struct foo;") is parsed.
    virtual DeclPtrTy ParsedFreeStandingDeclSpec(Scope *S, DeclSpec &DS) {
      llvm::cout << __FUNCTION__ << "\n";
      return DeclPtrTy();
    }
    
    /// ActOnLinkageSpec - Parsed a C++ linkage-specification that
    /// contained braces. Lang/StrSize contains the language string that
    /// was parsed at location Loc. Decls/NumDecls provides the
    /// declarations parsed inside the linkage specification.
    virtual DeclPtrTy ActOnLinkageSpec(SourceLocation Loc,
                                       SourceLocation LBrace,
                                       SourceLocation RBrace, const char *Lang,
                                       unsigned StrSize, 
                                       DeclPtrTy *Decls, unsigned NumDecls) {
      llvm::cout << __FUNCTION__ << "\n";
      return DeclPtrTy();
    }
    
    /// ActOnLinkageSpec - Parsed a C++ linkage-specification without
    /// braces. Lang/StrSize contains the language string that was
    /// parsed at location Loc. D is the declaration parsed.
    virtual DeclPtrTy ActOnLinkageSpec(SourceLocation Loc, const char *Lang,
                                       unsigned StrSize, DeclPtrTy D) {
      return DeclPtrTy();
    }
    
    //===--------------------------------------------------------------------===//
    // Type Parsing Callbacks.
    //===--------------------------------------------------------------------===//
  
    virtual TypeResult ActOnTypeName(Scope *S, Declarator &D) {
      llvm::cout << __FUNCTION__ << "\n";
      return TypeResult();
    }
  
    virtual DeclPtrTy ActOnTag(Scope *S, unsigned TagType, TagKind TK,
                               SourceLocation KWLoc, const CXXScopeSpec &SS,
                               IdentifierInfo *Name, SourceLocation NameLoc,
                               AttributeList *Attr, AccessSpecifier AS) {
      // TagType is an instance of DeclSpec::TST, indicating what kind of tag this
      // is (struct/union/enum/class).
      llvm::cout << __FUNCTION__ << "\n";
      return DeclPtrTy();
    }
  
    /// Act on @defs() element found when parsing a structure.  ClassName is the
    /// name of the referenced class.   
    virtual void ActOnDefs(Scope *S, DeclPtrTy TagD, SourceLocation DeclStart,
                           IdentifierInfo *ClassName,
                           llvm::SmallVectorImpl<DeclPtrTy> &Decls) {
      llvm::cout << __FUNCTION__ << "\n";
    }

    virtual DeclPtrTy ActOnField(Scope *S, DeclPtrTy TagD, 
                                 SourceLocation DeclStart,
                                 Declarator &D, ExprTy *BitfieldWidth) {
      llvm::cout << __FUNCTION__ << "\n";
      return DeclPtrTy();
    }
  
    virtual DeclPtrTy ActOnIvar(Scope *S, SourceLocation DeclStart,
                                Declarator &D, ExprTy *BitfieldWidth,
                                tok::ObjCKeywordKind visibility) {
      llvm::cout << __FUNCTION__ << "\n";
      return DeclPtrTy();
    }
  
    virtual void ActOnFields(Scope* S, SourceLocation RecLoc, DeclPtrTy TagDecl,
                             DeclPtrTy *Fields, unsigned NumFields, 
                             SourceLocation LBrac, SourceLocation RBrac,
                             AttributeList *AttrList) {
      llvm::cout << __FUNCTION__ << "\n";
    }
  
    virtual DeclPtrTy ActOnEnumConstant(Scope *S, DeclPtrTy EnumDecl,
                                        DeclPtrTy LastEnumConstant,
                                        SourceLocation IdLoc,IdentifierInfo *Id,
                                        SourceLocation EqualLoc, ExprTy *Val) {
      llvm::cout << __FUNCTION__ << "\n";
      return DeclPtrTy();
    }

    virtual void ActOnEnumBody(SourceLocation EnumLoc, DeclPtrTy EnumDecl,
                               DeclPtrTy *Elements, unsigned NumElements) {
      llvm::cout << __FUNCTION__ << "\n";
    }

    //===--------------------------------------------------------------------===//
    // Statement Parsing Callbacks.
    //===--------------------------------------------------------------------===//

    virtual OwningStmtResult ActOnNullStmt(SourceLocation SemiLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }

    virtual OwningStmtResult ActOnCompoundStmt(SourceLocation L,
                                               SourceLocation R,
                                               MultiStmtArg Elts,
                                               bool isStmtExpr) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }
    virtual OwningStmtResult ActOnDeclStmt(DeclGroupPtrTy Decl,
                                           SourceLocation StartLoc,
                                           SourceLocation EndLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }
  
    virtual OwningStmtResult ActOnExprStmt(ExprArg Expr) {
      llvm::cout << __FUNCTION__ << "\n";
      return OwningStmtResult(*this, Expr.release());
    }
  
    /// ActOnCaseStmt - Note that this handles the GNU 'case 1 ... 4' extension,
    /// which can specify an RHS value.
    virtual OwningStmtResult ActOnCaseStmt(SourceLocation CaseLoc,
                                           ExprArg LHSVal,
                                           SourceLocation DotDotDotLoc,
                                           ExprArg RHSVal,
                                           SourceLocation ColonLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }
    virtual OwningStmtResult ActOnDefaultStmt(SourceLocation DefaultLoc,
                                              SourceLocation ColonLoc,
                                              StmtArg SubStmt, Scope *CurScope){
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }

    virtual OwningStmtResult ActOnLabelStmt(SourceLocation IdentLoc,
                                            IdentifierInfo *II,
                                            SourceLocation ColonLoc,
                                            StmtArg SubStmt) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }

    virtual OwningStmtResult ActOnIfStmt(SourceLocation IfLoc, ExprArg CondVal,
                                         StmtArg ThenVal,SourceLocation ElseLoc,
                                         StmtArg ElseVal) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }

    virtual OwningStmtResult ActOnStartOfSwitchStmt(ExprArg Cond) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }

    virtual OwningStmtResult ActOnFinishSwitchStmt(SourceLocation SwitchLoc,
                                                   StmtArg Switch,
                                                   StmtArg Body) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }

    virtual OwningStmtResult ActOnWhileStmt(SourceLocation WhileLoc,
                                            ExprArg Cond, StmtArg Body) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }
    virtual OwningStmtResult ActOnDoStmt(SourceLocation DoLoc, StmtArg Body,
                                         SourceLocation WhileLoc, ExprArg Cond){
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }
    virtual OwningStmtResult ActOnForStmt(SourceLocation ForLoc,
                                        SourceLocation LParenLoc,
                                        StmtArg First, ExprArg Second,
                                        ExprArg Third, SourceLocation RParenLoc,
                                        StmtArg Body) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }
    virtual OwningStmtResult ActOnObjCForCollectionStmt(
                                       SourceLocation ForColLoc,
                                       SourceLocation LParenLoc,
                                       StmtArg First, ExprArg Second,
                                       SourceLocation RParenLoc, StmtArg Body) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }
    virtual OwningStmtResult ActOnGotoStmt(SourceLocation GotoLoc,
                                           SourceLocation LabelLoc,
                                           IdentifierInfo *LabelII) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }
    virtual OwningStmtResult ActOnIndirectGotoStmt(SourceLocation GotoLoc,
                                                   SourceLocation StarLoc,
                                                   ExprArg DestExp) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }
    virtual OwningStmtResult ActOnContinueStmt(SourceLocation ContinueLoc,
                                               Scope *CurScope) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }
    virtual OwningStmtResult ActOnBreakStmt(SourceLocation GotoLoc,
                                            Scope *CurScope) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }
    virtual OwningStmtResult ActOnReturnStmt(SourceLocation ReturnLoc,
                                             ExprArg RetValExp) {
      llvm::cout << __FUNCTION__ << "\n";
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
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }

    // Objective-c statements
    virtual OwningStmtResult ActOnObjCAtCatchStmt(SourceLocation AtLoc,
                                                  SourceLocation RParen,
                                                  DeclPtrTy Parm, StmtArg Body,
                                                  StmtArg CatchList) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }

    virtual OwningStmtResult ActOnObjCAtFinallyStmt(SourceLocation AtLoc,
                                                    StmtArg Body) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }

    virtual OwningStmtResult ActOnObjCAtTryStmt(SourceLocation AtLoc,
                                                StmtArg Try, StmtArg Catch,
                                                StmtArg Finally) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }

    virtual OwningStmtResult ActOnObjCAtThrowStmt(SourceLocation AtLoc,
                                                  ExprArg Throw,
                                                  Scope *CurScope) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }

    virtual OwningStmtResult ActOnObjCAtSynchronizedStmt(SourceLocation AtLoc,
                                                         ExprArg SynchExpr,
                                                         StmtArg SynchBody) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }

    // C++ Statements
    virtual DeclPtrTy ActOnExceptionDeclarator(Scope *S, Declarator &D) {
      llvm::cout << __FUNCTION__ << "\n";
      return DeclPtrTy();
    }

    virtual OwningStmtResult ActOnCXXCatchBlock(SourceLocation CatchLoc,
                                                DeclPtrTy ExceptionDecl,
                                                StmtArg HandlerBlock) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }

    virtual OwningStmtResult ActOnCXXTryBlock(SourceLocation TryLoc,
                                              StmtArg TryBlock,
                                              MultiStmtArg Handlers) {
      llvm::cout << __FUNCTION__ << "\n";
      return StmtEmpty();
    }

    //===--------------------------------------------------------------------===//
    // Expression Parsing Callbacks.
    //===--------------------------------------------------------------------===//

    // Primary Expressions.

    /// ActOnIdentifierExpr - Parse an identifier in expression context.
    /// 'HasTrailingLParen' indicates whether or not the identifier has a '('
    /// token immediately after it.
    virtual OwningExprResult ActOnIdentifierExpr(Scope *S, SourceLocation Loc,
                                                 IdentifierInfo &II,
                                                 bool HasTrailingLParen,
                                                 const CXXScopeSpec *SS,
                                                 bool isAddressOfOperand) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnCXXOperatorFunctionIdExpr(
                               Scope *S, SourceLocation OperatorLoc,
                               OverloadedOperatorKind Op,
                               bool HasTrailingLParen, const CXXScopeSpec &SS,
                               bool isAddressOfOperand) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnCXXConversionFunctionExpr(
                               Scope *S, SourceLocation OperatorLoc,
                               TypeTy *Type, bool HasTrailingLParen,
                               const CXXScopeSpec &SS,bool isAddressOfOperand) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnPredefinedExpr(SourceLocation Loc,
                                                 tok::TokenKind Kind) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnCharacterConstant(const Token &) { 
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnNumericConstant(const Token &) { 
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    /// ActOnStringLiteral - The specified tokens were lexed as pasted string
    /// fragments (e.g. "foo" "bar" L"baz").
    virtual OwningExprResult ActOnStringLiteral(const Token *Toks,
                                                unsigned NumToks) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnParenExpr(SourceLocation L, SourceLocation R,
                                            ExprArg Val) {
      llvm::cout << __FUNCTION__ << "\n";
      return move(Val);  // Default impl returns operand.
    }

    // Postfix Expressions.
    virtual OwningExprResult ActOnPostfixUnaryOp(Scope *S, SourceLocation OpLoc, 
                                                 tok::TokenKind Kind,
                                                 ExprArg Input) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }
    virtual OwningExprResult ActOnArraySubscriptExpr(Scope *S, ExprArg Base,
                                                     SourceLocation LLoc,
                                                     ExprArg Idx,
                                                     SourceLocation RLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }
    virtual OwningExprResult ActOnMemberReferenceExpr(Scope *S, ExprArg Base,
                                                      SourceLocation OpLoc,
                                                      tok::TokenKind OpKind,
                                                      SourceLocation MemberLoc,
                                                      IdentifierInfo &Member,
                                                      DeclPtrTy ImplDecl) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnCallExpr(Scope *S, ExprArg Fn,
                                           SourceLocation LParenLoc,
                                           MultiExprArg Args,
                                           SourceLocation *CommaLocs,
                                           SourceLocation RParenLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    // Unary Operators.  'Tok' is the token for the operator.
    virtual OwningExprResult ActOnUnaryOp(Scope *S, SourceLocation OpLoc,
                                          tok::TokenKind Op, ExprArg Input) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }
    virtual OwningExprResult
      ActOnSizeOfAlignOfExpr(SourceLocation OpLoc, bool isSizeof, bool isType,
                             void *TyOrEx, const SourceRange &ArgRange) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnCompoundLiteral(SourceLocation LParen,
                                                  TypeTy *Ty,
                                                  SourceLocation RParen,
                                                  ExprArg Op) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }
    virtual OwningExprResult ActOnInitList(SourceLocation LParenLoc,
                                           MultiExprArg InitList,
                                           SourceLocation RParenLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }
    virtual OwningExprResult ActOnCastExpr(SourceLocation LParenLoc, TypeTy *Ty,
                                           SourceLocation RParenLoc,ExprArg Op){
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnBinOp(Scope *S, SourceLocation TokLoc,
                                        tok::TokenKind Kind,
                                        ExprArg LHS, ExprArg RHS) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    /// ActOnConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
    /// in the case of a the GNU conditional expr extension.
    virtual OwningExprResult ActOnConditionalOp(SourceLocation QuestionLoc,
                                                SourceLocation ColonLoc,
                                                ExprArg Cond, ExprArg LHS,
                                                ExprArg RHS) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    //===--------------------- GNU Extension Expressions ------------------===//

    virtual OwningExprResult ActOnAddrLabel(SourceLocation OpLoc,
                                            SourceLocation LabLoc,
                                            IdentifierInfo *LabelII) {// "&&foo"
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnStmtExpr(SourceLocation LPLoc,
                                           StmtArg SubStmt,
                                           SourceLocation RPLoc) { // "({..})"
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnBuiltinOffsetOf(Scope *S,
                                                  SourceLocation BuiltinLoc,
                                                  SourceLocation TypeLoc,
                                                  TypeTy *Arg1,
                                                  OffsetOfComponent *CompPtr,
                                                  unsigned NumComponents,
                                                  SourceLocation RParenLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    // __builtin_types_compatible_p(type1, type2)
    virtual OwningExprResult ActOnTypesCompatibleExpr(SourceLocation BuiltinLoc,
                                                      TypeTy *arg1,TypeTy *arg2,
                                                      SourceLocation RPLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }
    // __builtin_choose_expr(constExpr, expr1, expr2)
    virtual OwningExprResult ActOnChooseExpr(SourceLocation BuiltinLoc,
                                             ExprArg cond, ExprArg expr1,
                                             ExprArg expr2,
                                             SourceLocation RPLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    // __builtin_va_arg(expr, type)
    virtual OwningExprResult ActOnVAArg(SourceLocation BuiltinLoc,
                                  ExprArg expr, TypeTy *type,
                                  SourceLocation RPLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnGNUNullExpr(SourceLocation TokenLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual void ActOnBlockStart(SourceLocation CaretLoc, Scope *CurScope) {
      llvm::cout << __FUNCTION__ << "\n";
    }

    virtual void ActOnBlockArguments(Declarator &ParamInfo, Scope *CurScope) {
      llvm::cout << __FUNCTION__ << "\n";
    }

    virtual void ActOnBlockError(SourceLocation CaretLoc, Scope *CurScope) {
      llvm::cout << __FUNCTION__ << "\n";
    }

    virtual OwningExprResult ActOnBlockStmtExpr(SourceLocation CaretLoc,
                                                StmtArg Body,
                                                Scope *CurScope) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual DeclPtrTy ActOnStartNamespaceDef(Scope *S, SourceLocation IdentLoc,
                                             IdentifierInfo *Ident,
                                             SourceLocation LBrace) {
      llvm::cout << __FUNCTION__ << "\n";
      return DeclPtrTy();
    }

    virtual void ActOnFinishNamespaceDef(DeclPtrTy Dcl, SourceLocation RBrace) {
      llvm::cout << __FUNCTION__ << "\n";
      return;
    }

#if 0
    // FIXME: AttrList should be deleted by this function, but the definition
    // would have to be available.
    virtual DeclPtrTy ActOnUsingDirective(Scope *CurScope,
                                          SourceLocation UsingLoc,
                                          SourceLocation NamespcLoc,
                                          const CXXScopeSpec &SS,
                                          SourceLocation IdentLoc,
                                          IdentifierInfo *NamespcName,
                                          AttributeList *AttrList) {
      llvm::cout << __FUNCTION__ << "\n";
      return DeclPtrTy();
    }
#endif

    virtual void ActOnParamDefaultArgument(DeclPtrTy param,
                                           SourceLocation EqualLoc,
                                           ExprArg defarg) {
      llvm::cout << __FUNCTION__ << "\n";
    }

    virtual void ActOnParamUnparsedDefaultArgument(DeclPtrTy param,
                                                   SourceLocation EqualLoc) {
      llvm::cout << __FUNCTION__ << "\n";
    }

    virtual void ActOnParamDefaultArgumentError(DeclPtrTy param) {
      llvm::cout << __FUNCTION__ << "\n";
    }

    virtual void AddCXXDirectInitializerToDecl(DeclPtrTy Dcl,
                                               SourceLocation LParenLoc,
                                               MultiExprArg Exprs,
                                               SourceLocation *CommaLocs,
                                               SourceLocation RParenLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return;
    }

    virtual void ActOnStartDelayedCXXMethodDeclaration(Scope *S,
                                                       DeclPtrTy Method)
    {
      llvm::cout << __FUNCTION__ << "\n";
    }

    virtual void ActOnDelayedCXXMethodParameter(Scope *S, DeclPtrTy Param) {
      llvm::cout << __FUNCTION__ << "\n";
    }

    virtual void ActOnFinishDelayedCXXMethodDeclaration(Scope *S,
                                                        DeclPtrTy Method) {
      llvm::cout << __FUNCTION__ << "\n";
    }

    virtual DeclPtrTy ActOnStaticAssertDeclaration(SourceLocation AssertLoc,
                                                   ExprArg AssertExpr,
                                                   ExprArg AssertMessageExpr) {
      llvm::cout << __FUNCTION__ << "\n";
      return DeclPtrTy();
    }

    virtual OwningExprResult ActOnCXXNamedCast(SourceLocation OpLoc,
                                               tok::TokenKind Kind,
                                               SourceLocation LAngleBracketLoc,
                                               TypeTy *Ty,
                                               SourceLocation RAngleBracketLoc,
                                               SourceLocation LParenLoc,
                                               ExprArg Op,
                                               SourceLocation RParenLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnCXXTypeid(SourceLocation OpLoc,
                                            SourceLocation LParenLoc,
                                            bool isType, void *TyOrExpr,
                                            SourceLocation RParenLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnCXXThis(SourceLocation ThisLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnCXXBoolLiteral(SourceLocation OpLoc,
                                                 tok::TokenKind Kind) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnCXXThrow(SourceLocation OpLoc, ExprArg Op) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnCXXTypeConstructExpr(SourceRange TypeRange,
                                                     TypeTy *TypeRep,
                                                     SourceLocation LParenLoc,
                                                     MultiExprArg Exprs,
                                                     SourceLocation *CommaLocs,
                                                     SourceLocation RParenLoc) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnCXXConditionDeclarationExpr(Scope *S,
                                                        SourceLocation StartLoc,
                                                        Declarator &D,
                                                        SourceLocation EqualLoc,
                                                        ExprArg AssignExprVal) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnCXXNew(SourceLocation StartLoc,
                                         bool UseGlobal,
                                         SourceLocation PlacementLParen,
                                         MultiExprArg PlacementArgs,
                                         SourceLocation PlacementRParen,
                                         bool ParenTypeId, Declarator &D,
                                         SourceLocation ConstructorLParen,
                                         MultiExprArg ConstructorArgs,
                                         SourceLocation ConstructorRParen) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnCXXDelete(SourceLocation StartLoc,
                                            bool UseGlobal, bool ArrayForm,
                                            ExprArg Operand) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }

    virtual OwningExprResult ActOnUnaryTypeTrait(UnaryTypeTrait OTT,
                                                 SourceLocation KWLoc,
                                                 SourceLocation LParen,
                                                 TypeTy *Ty,
                                                 SourceLocation RParen) {
      llvm::cout << __FUNCTION__ << "\n";
      return ExprEmpty();
    }
  };
}

MinimalAction *clang::CreatePrintParserActionsAction(Preprocessor &PP) {
  return new ParserPrintActions(PP);
}
