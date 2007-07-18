//===--- Action.h - Parser Action Interface ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Action and EmptyAction interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_ACTION_H
#define LLVM_CLANG_PARSE_ACTION_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TokenKinds.h"

namespace clang {
  // Semantic.
  class DeclSpec;
  class Declarator;
  class AttributeList;
  // Parse.
  class Scope;
  class Action;
  // Lex.
  class IdentifierInfo;
  class LexerToken;

/// Action - As the parser reads the input file and recognizes the productions
/// of the grammar, it invokes methods on this class to turn the parsed input
/// into something useful: e.g. a parse tree.
///
/// The callback methods that this class provides are phrased as actions that
/// the parser has just done or is about to do when the method is called.  They
/// are not requests that the actions module do the specified action.
///
/// All of the methods here are optional except isTypeName(), which must be
/// specified in order for the parse to complete accurately.  The EmptyAction
/// class does this bare-minimum of tracking to implement this functionality.
class Action {
public:
  /// Out-of-line virtual destructor to provide home for this class.
  virtual ~Action();
  
  // Types - Though these don't actually enforce strong typing, they document
  // what types are required to be identical for the actions.
  typedef void ExprTy;
  typedef void StmtTy;
  typedef void DeclTy;
  typedef void TypeTy;
  typedef void AttrTy;
  
  /// ActionResult - This structure is used while parsing/acting on expressions,
  /// stmts, etc.  It encapsulates both the object returned by the action, plus
  /// a sense of whether or not it is valid.
  template<unsigned UID>
  struct ActionResult {
    void *Val;
    bool isInvalid;
    
    ActionResult(bool Invalid = false) : Val(0), isInvalid(Invalid) {}
    template<typename ActualExprTy>
    ActionResult(ActualExprTy *val) : Val(val), isInvalid(false) {}
    
    const ActionResult &operator=(void *RHS) {
      Val = RHS;
      isInvalid = false;
      return *this;
    }
  };

  /// Expr/Stmt/TypeResult - Provide a unique type to wrap ExprTy/StmtTy/TypeTy,
  /// providing strong typing and allowing for failure.
  typedef ActionResult<0> ExprResult;
  typedef ActionResult<1> StmtResult;
  typedef ActionResult<2> TypeResult;
  
  //===--------------------------------------------------------------------===//
  // Declaration Tracking Callbacks.
  //===--------------------------------------------------------------------===//
  
  /// isTypeName - Return non-null if the specified identifier is a typedef name
  /// in the current scope.
  virtual DeclTy *isTypeName(const IdentifierInfo &II, Scope *S) const = 0;
  
  /// ParseDeclarator - This callback is invoked when a declarator is parsed and
  /// 'Init' specifies the initializer if any.  This is for things like:
  /// "int X = 4" or "typedef int foo".
  ///
  /// LastInGroup is non-null for cases where one declspec has multiple
  /// declarators on it.  For example in 'int A, B', ParseDeclarator will be
  /// called with LastInGroup=A when invoked for B.
  virtual DeclTy *ParseDeclarator(Scope *S, Declarator &D,
                                  ExprTy *Init, DeclTy *LastInGroup) {
    return 0;
  }

  /// FinalizeDeclaratorGroup - After a sequence of declarators are parsed, this
  /// gives the actions implementation a chance to process the group as a whole.
  virtual DeclTy *FinalizeDeclaratorGroup(Scope *S, DeclTy *Group) {
    return Group;
  }

  /// ParseStartOfFunctionDef - This is called at the start of a function
  /// definition, instead of calling ParseDeclarator.  The Declarator includes
  /// information about formal arguments that are part of this function.
  virtual DeclTy *ParseStartOfFunctionDef(Scope *FnBodyScope, Declarator &D) {
    // Default to ParseDeclarator.
    return ParseDeclarator(FnBodyScope, D, 0, 0);
  }

  /// ParseFunctionDefBody - This is called when a function body has completed
  /// parsing.  Decl is the DeclTy returned by ParseStartOfFunctionDef.
  virtual DeclTy *ParseFunctionDefBody(DeclTy *Decl, StmtTy *Body) {
    return Decl;
  }

  
  /// PopScope - This callback is called immediately before the specified scope
  /// is popped and deleted.
  virtual void PopScope(SourceLocation Loc, Scope *S) {}
  
  /// ParsedFreeStandingDeclSpec - This method is invoked when a declspec with
  /// no declarator (e.g. "struct foo;") is parsed.
  virtual DeclTy *ParsedFreeStandingDeclSpec(Scope *S, DeclSpec &DS) {
    return 0;
  }
  
  virtual DeclTy *ParsedObjcClassDeclaration(Scope *S,
                                             IdentifierInfo **IdentList,
                                             unsigned NumElts) {
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Type Parsing Callbacks.
  //===--------------------------------------------------------------------===//
  
  virtual TypeResult ParseTypeName(Scope *S, Declarator &D) {
    return 0;
  }
  
  virtual TypeResult ParseParamDeclaratorType(Scope *S, Declarator &D) {
    return 0;
  }
  
  enum TagKind {
    TK_Reference,   // Reference to a tag:  'struct foo *X;'
    TK_Declaration, // Fwd decl of a tag:   'struct foo;'
    TK_Definition   // Definition of a tag: 'struct foo { int X; } Y;'
  };
  virtual DeclTy *ParseTag(Scope *S, unsigned TagType, TagKind TK,
                           SourceLocation KWLoc, IdentifierInfo *Name,
                           SourceLocation NameLoc, AttributeList *Attr) {
    // TagType is an instance of DeclSpec::TST, indicating what kind of tag this
    // is (struct/union/enum/class).
    return 0;
  }
  
  virtual DeclTy *ParseField(Scope *S, DeclTy *TagDecl,SourceLocation DeclStart,
                             Declarator &D, ExprTy *BitfieldWidth) {
    return 0;
  }
  virtual void ParseRecordBody(SourceLocation RecLoc, DeclTy *TagDecl,
                               DeclTy **Fields, unsigned NumFields) {}

  virtual DeclTy *ParseEnumConstant(Scope *S, DeclTy *EnumDecl,
                                    DeclTy *LastEnumConstant,
                                    SourceLocation IdLoc, IdentifierInfo *Id,
                                    SourceLocation EqualLoc, ExprTy *Val) {
    return 0;
  }
  virtual void ParseEnumBody(SourceLocation EnumLoc, DeclTy *EnumDecl,
                             DeclTy **Elements, unsigned NumElements) {}

  //===--------------------------------------------------------------------===//
  // Statement Parsing Callbacks.
  //===--------------------------------------------------------------------===//
  
  virtual StmtResult ParseNullStmt(SourceLocation SemiLoc) {
    return 0;
  }
  
  virtual StmtResult ParseCompoundStmt(SourceLocation L, SourceLocation R,
                                       StmtTy **Elts, unsigned NumElts) {
    return 0;
  }
  virtual StmtResult ParseDeclStmt(DeclTy *Decl) {
    return 0;
  }
  
  virtual StmtResult ParseExprStmt(ExprTy *Expr) {
    return StmtResult(Expr);
  }
  
  /// ParseCaseStmt - Note that this handles the GNU 'case 1 ... 4' extension,
  /// which can specify an RHS value.
  virtual StmtResult ParseCaseStmt(SourceLocation CaseLoc, ExprTy *LHSVal,
                                   SourceLocation DotDotDotLoc, ExprTy *RHSVal,
                                   SourceLocation ColonLoc, StmtTy *SubStmt) {
    return 0;
  }
  virtual StmtResult ParseDefaultStmt(SourceLocation DefaultLoc,
                                      SourceLocation ColonLoc, StmtTy *SubStmt,
                                      Scope *CurScope){
    return 0;
  }
  
  virtual StmtResult ParseLabelStmt(SourceLocation IdentLoc, IdentifierInfo *II,
                                    SourceLocation ColonLoc, StmtTy *SubStmt) {
    return 0;
  }
  
  virtual StmtResult ParseIfStmt(SourceLocation IfLoc, ExprTy *CondVal,
                                 StmtTy *ThenVal, SourceLocation ElseLoc,
                                 StmtTy *ElseVal) {
    return 0; 
  }
  
  virtual StmtResult ParseSwitchStmt(SourceLocation SwitchLoc, ExprTy *Cond,
                                     StmtTy *Body) {
    return 0;
  }
  virtual StmtResult ParseWhileStmt(SourceLocation WhileLoc, ExprTy *Cond,
                                    StmtTy *Body) {
    return 0;
  }
  virtual StmtResult ParseDoStmt(SourceLocation DoLoc, StmtTy *Body,
                                 SourceLocation WhileLoc, ExprTy *Cond) {
    return 0;
  }
  virtual StmtResult ParseForStmt(SourceLocation ForLoc, 
                                  SourceLocation LParenLoc, 
                                  StmtTy *First, ExprTy *Second, ExprTy *Third,
                                  SourceLocation RParenLoc, StmtTy *Body) {
    return 0;
  }
  virtual StmtResult ParseGotoStmt(SourceLocation GotoLoc,
                                   SourceLocation LabelLoc,
                                   IdentifierInfo *LabelII) {
    return 0;
  }
  virtual StmtResult ParseIndirectGotoStmt(SourceLocation GotoLoc,
                                           SourceLocation StarLoc,
                                           ExprTy *DestExp) {
    return 0;
  }
  virtual StmtResult ParseContinueStmt(SourceLocation ContinueLoc,
                                       Scope *CurScope) {
    return 0;
  }
  virtual StmtResult ParseBreakStmt(SourceLocation GotoLoc, Scope *CurScope) {
    return 0;
  }
  virtual StmtResult ParseReturnStmt(SourceLocation ReturnLoc,
                                     ExprTy *RetValExp) {
    return 0;
  }
  
  //===--------------------------------------------------------------------===//
  // Expression Parsing Callbacks.
  //===--------------------------------------------------------------------===//
  
  // Primary Expressions.
  
  /// ParseIdentifierExpr - Parse an identifier in expression context.
  /// 'HasTrailingLParen' indicates whether or not the identifier has a '('
  /// token immediately after it.
  virtual ExprResult ParseIdentifierExpr(Scope *S, SourceLocation Loc,
                                         IdentifierInfo &II,
                                         bool HasTrailingLParen) {
    return 0;
  }
  
  virtual ExprResult ParseSimplePrimaryExpr(SourceLocation Loc,
                                            tok::TokenKind Kind) {
    return 0;
  }
  virtual ExprResult ParseCharacterConstant(const LexerToken &) { return 0; }
  virtual ExprResult ParseNumericConstant(const LexerToken &) { return 0; }
  
  /// ParseStringLiteral - The specified tokens were lexed as pasted string
  /// fragments (e.g. "foo" "bar" L"baz").
  virtual ExprResult ParseStringLiteral(const LexerToken *Toks, unsigned NumToks) {
    return 0;
  }
  
  virtual ExprResult ParseParenExpr(SourceLocation L, SourceLocation R,
                                    ExprTy *Val) {
    return Val;  // Default impl returns operand.
  }
  
  // Postfix Expressions.
  virtual ExprResult ParsePostfixUnaryOp(SourceLocation OpLoc, 
                                         tok::TokenKind Kind, ExprTy *Input) {
    return 0;
  }
  virtual ExprResult ParseArraySubscriptExpr(ExprTy *Base, SourceLocation LLoc,
                                             ExprTy *Idx, SourceLocation RLoc) {
    return 0;
  }
  virtual ExprResult ParseMemberReferenceExpr(ExprTy *Base,SourceLocation OpLoc,
                                              tok::TokenKind OpKind,
                                              SourceLocation MemberLoc,
                                              IdentifierInfo &Member) {
    return 0;
  }
  
  /// ParseCallExpr - Handle a call to Fn with the specified array of arguments.
  /// This provides the location of the left/right parens and a list of comma
  /// locations.  There are guaranteed to be one fewer commas than arguments,
  /// unless there are zero arguments.
  virtual ExprResult ParseCallExpr(ExprTy *Fn, SourceLocation LParenLoc,
                                   ExprTy **Args, unsigned NumArgs,
                                   SourceLocation *CommaLocs,
                                   SourceLocation RParenLoc) {
    return 0;
  }
  
  // Unary Operators.  'Tok' is the token for the operator.
  virtual ExprResult ParseUnaryOp(SourceLocation OpLoc, tok::TokenKind Op,
                                  ExprTy *Input) {
    return 0;
  }
  virtual ExprResult 
    ParseSizeOfAlignOfTypeExpr(SourceLocation OpLoc, bool isSizeof, 
                               SourceLocation LParenLoc, TypeTy *Ty,
                               SourceLocation RParenLoc) {
    return 0;
  }
  
  virtual ExprResult ParseCastExpr(SourceLocation LParenLoc, TypeTy *Ty,
                                   SourceLocation RParenLoc, ExprTy *Op) {
    return 0;
  }
  
  virtual ExprResult ParseBinOp(SourceLocation TokLoc, tok::TokenKind Kind,
                                ExprTy *LHS, ExprTy *RHS) {
    return 0;
  }

  /// ParseConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
  /// in the case of a the GNU conditional expr extension.
  virtual ExprResult ParseConditionalOp(SourceLocation QuestionLoc, 
                                        SourceLocation ColonLoc,
                                        ExprTy *Cond, ExprTy *LHS, ExprTy *RHS){
    return 0;
  }
  
  virtual ExprResult ParseAddrLabel(SourceLocation OpLoc, SourceLocation LabLoc,
                                    IdentifierInfo *LabelII) { // "&&foo"
    return 0;
  }
  
  
  /// ParseCXXCasts - Parse {dynamic,static,reinterpret,const}_cast's.
  virtual ExprResult ParseCXXCasts(SourceLocation OpLoc, tok::TokenKind Kind,
                                   SourceLocation LAngleBracketLoc, TypeTy *Ty,
                                   SourceLocation RAngleBracketLoc,
                                   SourceLocation LParenLoc, ExprTy *Op,
                                   SourceLocation RParenLoc) {
    return 0;
  }

  /// ParseCXXBoolLiteral - Parse {true,false} literals.
  virtual ExprResult ParseCXXBoolLiteral(SourceLocation OpLoc,
                                         tok::TokenKind Kind) {
    return 0;
  }
};

/// MinimalAction - Minimal actions are used by light-weight clients of the
/// parser that do not need name resolution or significant semantic analysis to
/// be performed.  The actions implemented here are in the form of unresolved
/// identifiers.  By using a simpler interface than the SemanticAction class,
/// the parser doesn't have to build complex data structures and thus runs more
/// quickly.
class MinimalAction : public Action {
public:
  /// isTypeName - This looks at the IdentifierInfo::FETokenInfo field to
  /// determine whether the name is a typedef or not in this scope.
  virtual DeclTy *isTypeName(const IdentifierInfo &II, Scope *S) const;
  
  /// ParseDeclarator - If this is a typedef declarator, we modify the
  /// IdentifierInfo::FETokenInfo field to keep track of this fact, until S is
  /// popped.
  virtual DeclTy *ParseDeclarator(Scope *S, Declarator &D, ExprTy *Init,
                                  DeclTy *LastInGroup);
  
  /// PopScope - When a scope is popped, if any typedefs are now out-of-scope,
  /// they are removed from the IdentifierInfo::FETokenInfo field.
  virtual void PopScope(SourceLocation Loc, Scope *S);
  
  virtual DeclTy *ParsedObjcClassDeclaration(Scope *S,
                                             IdentifierInfo **IdentList,
                                             unsigned NumElts);
  
};

}  // end namespace clang

#endif
