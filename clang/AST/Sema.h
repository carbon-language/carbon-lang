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
#include <vector>
#include <string>

namespace llvm {
namespace clang {
  class ASTContext;
  class Preprocessor;
  class Decl;
  class TypeRef;
  
/// Sema - This implements semantic analysis and AST building for C.
class Sema : public Action {
  ASTContext &Context;
  
  /// LastInGroupList - This vector is populated when there are multiple
  /// declarators in a single decl group (e.g. "int A, B, C").  In this case,
  /// all but the last decl will be entered into this.  This is used by the
  /// ASTStreamer.
  std::vector<Decl*> &LastInGroupList;
public:
  Sema(ASTContext &ctx, std::vector<Decl*> &prevInGroup)
    : Context(ctx), LastInGroupList(prevInGroup) {
  }
  
  void Diag(SourceLocation Loc, unsigned DiagID,
            const std::string &Msg = std::string());
  
  //===--------------------------------------------------------------------===//
  // Type Analysis / Processing: SemaType.cpp.
  //
  TypeRef GetTypeForDeclarator(Declarator &D, Scope *S);
  
  
  //===--------------------------------------------------------------------===//
  // Symbol table / Decl tracking callbacks: SemaDecl.cpp.
  //
  virtual bool isTypeName(const IdentifierInfo &II, Scope *S) const;
  virtual DeclTy *ParseDeclarator(Scope *S, Declarator &D, ExprTy *Init,
                                  DeclTy *LastInGroup);
  virtual DeclTy *ParseFunctionDefinition(Scope *S, Declarator &D,
                                          StmtTy *Body);
  virtual void PopScope(SourceLocation Loc, Scope *S);
  
  //===--------------------------------------------------------------------===//
  // Statement Parsing Callbacks: SemaStmt.cpp.

  virtual StmtResult ParseCompoundStmt(SourceLocation L, SourceLocation R,
                                       StmtTy **Elts, unsigned NumElts);
  virtual StmtResult ParseExprStmt(ExprTy *Expr) {
    return Expr; // Exprs are Stmts.
  }
  virtual StmtResult ParseCaseStmt(SourceLocation CaseLoc, ExprTy *LHSVal,
                                   SourceLocation DotDotDotLoc, ExprTy *RHSVal,
                                   SourceLocation ColonLoc, StmtTy *SubStmt);
  virtual StmtResult ParseDefaultStmt(SourceLocation DefaultLoc,
                                      SourceLocation ColonLoc, StmtTy *SubStmt);
  virtual StmtResult ParseLabelStmt(SourceLocation IdentLoc, IdentifierInfo *II,
                                    SourceLocation ColonLoc, StmtTy *SubStmt);
  virtual StmtResult ParseIfStmt(SourceLocation IfLoc, ExprTy *CondVal,
                                 StmtTy *ThenVal, SourceLocation ElseLoc,
                                 StmtTy *ElseVal);
  virtual StmtResult ParseSwitchStmt(SourceLocation SwitchLoc, ExprTy *Cond,
                                     StmtTy *Body);
  virtual StmtResult ParseWhileStmt(SourceLocation WhileLoc, ExprTy *Cond,
                                    StmtTy *Body);
  virtual StmtResult ParseDoStmt(SourceLocation DoLoc, StmtTy *Body,
                                 SourceLocation WhileLoc, ExprTy *Cond);
  
  virtual StmtResult ParseForStmt(SourceLocation ForLoc, 
                                  SourceLocation LParenLoc, 
                                  StmtTy *First, ExprTy *Second, ExprTy *Third,
                                  SourceLocation RParenLoc, StmtTy *Body);
  virtual StmtResult ParseGotoStmt(SourceLocation GotoLoc,
                                   SourceLocation LabelLoc,
                                   IdentifierInfo *LabelII);
  virtual StmtResult ParseIndirectGotoStmt(SourceLocation GotoLoc,
                                           SourceLocation StarLoc,
                                           ExprTy *DestExp);
  virtual StmtResult ParseContinueStmt(SourceLocation ContinueLoc,
                                       Scope *CurScope);
  virtual StmtResult ParseBreakStmt(SourceLocation GotoLoc, Scope *CurScope);
  
  virtual StmtResult ParseReturnStmt(SourceLocation ReturnLoc,
                                     ExprTy *RetValExp);
  
  //===--------------------------------------------------------------------===//
  // Expression Parsing Callbacks: SemaExpr.cpp.

  // Primary Expressions.
  virtual ExprResult ParseSimplePrimaryExpr(SourceLocation Loc,
                                            tok::TokenKind Kind);
  virtual ExprResult ParseIntegerConstant(SourceLocation Loc);
  virtual ExprResult ParseFloatingConstant(SourceLocation Loc);
  virtual ExprResult ParseParenExpr(SourceLocation L, SourceLocation R,
                                    ExprTy *Val);

  /// ParseStringExpr - The specified tokens were lexed as pasted string
  /// fragments (e.g. "foo" "bar" L"baz").
  virtual ExprResult ParseStringExpr(const LexerToken *Toks, unsigned NumToks);
    
  // Binary/Unary Operators.  'Tok' is the token for the operator.
  virtual ExprResult ParseUnaryOp(SourceLocation OpLoc, tok::TokenKind Op,
                                  ExprTy *Input);
  virtual ExprResult 
    ParseSizeOfAlignOfTypeExpr(SourceLocation OpLoc, bool isSizeof, 
                               SourceLocation LParenLoc, TypeTy *Ty,
                               SourceLocation RParenLoc);
  
  virtual ExprResult ParsePostfixUnaryOp(SourceLocation OpLoc, 
                                         tok::TokenKind Kind, ExprTy *Input);
  
  virtual ExprResult ParseArraySubscriptExpr(ExprTy *Base, SourceLocation LLoc,
                                             ExprTy *Idx, SourceLocation RLoc);
  virtual ExprResult ParseMemberReferenceExpr(ExprTy *Base,SourceLocation OpLoc,
                                              tok::TokenKind OpKind,
                                              SourceLocation MemberLoc,
                                              IdentifierInfo &Member);
  
  /// ParseCallExpr - Handle a call to Fn with the specified array of arguments.
  /// This provides the location of the left/right parens and a list of comma
  /// locations.
  virtual ExprResult ParseCallExpr(ExprTy *Fn, SourceLocation LParenLoc,
                                   ExprTy **Args, unsigned NumArgs,
                                   SourceLocation *CommaLocs,
                                   SourceLocation RParenLoc);
  
  virtual ExprResult ParseCastExpr(SourceLocation LParenLoc, TypeTy *Ty,
                                   SourceLocation RParenLoc, ExprTy *Op);
  
  virtual ExprResult ParseBinOp(SourceLocation TokLoc, tok::TokenKind Kind,
                                ExprTy *LHS,ExprTy *RHS);
  
  /// ParseConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
  /// in the case of a the GNU conditional expr extension.
  virtual ExprResult ParseConditionalOp(SourceLocation QuestionLoc, 
                                        SourceLocation ColonLoc,
                                        ExprTy *Cond, ExprTy *LHS, ExprTy *RHS);
};


}  // end namespace clang
}  // end namespace llvm

#endif
