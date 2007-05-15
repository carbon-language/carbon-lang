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
  class Expr;
  class VarDecl;
  class TypedefDecl;
  class FunctionDecl;
  class QualType;
  class LangOptions;
  class DeclaratorChunk;
  class LexerToken;
  class IntegerLiteral;
  class ArrayType;
  
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
public:
  Sema(Preprocessor &pp, ASTContext &ctxt, std::vector<Decl*> &prevInGroup);
  
  const LangOptions &getLangOptions() const;
  
  /// always returns true, which simplifies error handling (i.e. less code).
  bool Diag(SourceLocation Loc, unsigned DiagID,
            const std::string &Msg = std::string());
  bool Diag(const LexerToken &Tok, unsigned DiagID,
            const std::string &M = std::string());
  bool Diag(SourceLocation Loc, unsigned DiagID, QualType t);
  
  //===--------------------------------------------------------------------===//
  // Type Analysis / Processing: SemaType.cpp.
  //
  QualType GetTypeForDeclarator(Declarator &D, Scope *S);
  
  virtual TypeResult ParseTypeName(Scope *S, Declarator &D);
  
  virtual TypeResult ParseParamDeclaratorType(Scope *S, Declarator &D);
private:
  //===--------------------------------------------------------------------===//
  // Symbol table / Decl tracking callbacks: SemaDecl.cpp.
  //
  virtual DeclTy *isTypeName(const IdentifierInfo &II, Scope *S) const;
  virtual DeclTy *ParseDeclarator(Scope *S, Declarator &D, ExprTy *Init,
                                  DeclTy *LastInGroup);
  virtual DeclTy *ParseStartOfFunctionDef(Scope *S, Declarator &D);
  virtual DeclTy *ParseFunctionDefBody(DeclTy *Decl, StmtTy *Body);
  virtual void PopScope(SourceLocation Loc, Scope *S);

  /// ParsedFreeStandingDeclSpec - This method is invoked when a declspec with
  /// no declarator (e.g. "struct foo;") is parsed.
  virtual DeclTy *ParsedFreeStandingDeclSpec(Scope *S, DeclSpec &DS);  
  
  virtual DeclTy *ParseTag(Scope *S, unsigned TagType, TagKind TK,
                           SourceLocation KWLoc, IdentifierInfo *Name,
                           SourceLocation NameLoc);
  virtual DeclTy *ParseField(Scope *S, DeclTy *TagDecl,SourceLocation DeclStart,
                             Declarator &D, ExprTy *BitfieldWidth);
  virtual void ParseRecordBody(SourceLocation RecLoc, DeclTy *TagDecl,
                               DeclTy **Fields, unsigned NumFields);
  virtual DeclTy *ParseEnumConstant(Scope *S, DeclTy *EnumDecl,
                                    SourceLocation IdLoc, IdentifierInfo *Id,
                                    SourceLocation EqualLoc, ExprTy *Val);
  virtual void ParseEnumBody(SourceLocation EnumLoc, DeclTy *EnumDecl,
                             DeclTy **Elements, unsigned NumElements);
private:
  /// Subroutines of ParseDeclarator()...
  TypedefDecl *ParseTypedefDecl(Scope *S, Declarator &D);
  TypedefDecl *MergeTypeDefDecl(TypedefDecl *New, Decl *Old);
  FunctionDecl *MergeFunctionDecl(FunctionDecl *New, Decl *Old);
  VarDecl *MergeVarDecl(VarDecl *New, Decl *Old);
  /// AddTopLevelDecl - called after the decl has been fully processed.
  /// Allows for bookkeeping and post-processing of each declaration.
  void AddTopLevelDecl(Decl *current, Decl *last);

  /// More parsing and symbol table subroutines...
  VarDecl *ParseParamDeclarator(DeclaratorChunk &FI, unsigned ArgNo,
                                Scope *FnBodyScope);
  Decl *LookupScopedDecl(IdentifierInfo *II, unsigned NSI, SourceLocation IdLoc,
                         Scope *S);  
  Decl *LazilyCreateBuiltin(IdentifierInfo *II, unsigned ID, Scope *S);
  Decl *ImplicitlyDefineFunction(SourceLocation Loc, IdentifierInfo &II,
                                 Scope *S);
  
  //===--------------------------------------------------------------------===//
  // Statement Parsing Callbacks: SemaStmt.cpp.
public:
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
  virtual ExprResult ParseIdentifierExpr(Scope *S, SourceLocation Loc,
                                         IdentifierInfo &II,
                                         bool HasTrailingLParen);
  virtual ExprResult ParseSimplePrimaryExpr(SourceLocation Loc,
                                            tok::TokenKind Kind);
  virtual ExprResult ParseNumericConstant(const LexerToken &);
  virtual ExprResult ParseCharacterConstant(const LexerToken &);
  virtual ExprResult ParseParenExpr(SourceLocation L, SourceLocation R,
                                    ExprTy *Val);

  /// ParseStringLiteral - The specified tokens were lexed as pasted string
  /// fragments (e.g. "foo" "bar" L"baz").
  virtual ExprResult ParseStringLiteral(const LexerToken *Toks, unsigned NumToks);
    
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

  /// ParseCXXCasts - Parse {dynamic,static,reinterpret,const}_cast's.
  virtual ExprResult ParseCXXCasts(SourceLocation OpLoc, tok::TokenKind Kind,
                                   SourceLocation LAngleBracketLoc, TypeTy *Ty,
                                   SourceLocation RAngleBracketLoc,
                                   SourceLocation LParenLoc, ExprTy *E,
                                   SourceLocation RParenLoc);

  /// ParseCXXBoolLiteral - Parse {true,false} literals.
  virtual ExprResult ParseCXXBoolLiteral(SourceLocation OpLoc,
                                         tok::TokenKind Kind);
private:
  QualType UsualUnaryConversion(QualType t); // C99 6.3
  QualType UsualArithmeticConversions(QualType t1, QualType t2); // C99 6.3.1.8
  
  enum AssignmentConversionResult {
    Compatible,
    Incompatible,
    PointerFromInt, 
    IntFromPointer,
    IncompatiblePointer,
    CompatiblePointerDiscardsQualifiers
  };
  // Conversions for assignment, argument passing, initialization, and
  // function return values. UsualAssignmentConversions is currently used by 
  // CheckSimpleAssignment, CheckCompoundAssignment and ParseCallExpr. 
  QualType UsualAssignmentConversions(QualType lhs, QualType rhs, // C99 6.5.16
                                      AssignmentConversionResult &r);
  // Helper function for UsualAssignmentConversions (C99 6.5.16.1p1)
  QualType CheckPointerTypesForAssignment(QualType lhsType, QualType rhsType,
                                          AssignmentConversionResult &r);
  
  /// the following "Check" methods will return a valid/converted QualType
  /// or a null QualType (indicating an error diagnostic was issued).
    
  /// type checking binary operators (subroutines of ParseBinOp).
  inline QualType CheckMultiplyDivideOperands( // C99 6.5.5
    Expr *lex, Expr *rex, SourceLocation OpLoc); 
  inline QualType CheckRemainderOperands( // C99 6.5.5
    Expr *lex, Expr *rex, SourceLocation OpLoc); 
  inline QualType CheckAdditionOperands( // C99 6.5.6
    Expr *lex, Expr *rex, SourceLocation OpLoc);
  inline QualType CheckSubtractionOperands( // C99 6.5.6
    Expr *lex, Expr *rex, SourceLocation OpLoc);
  inline QualType CheckShiftOperands( // C99 6.5.7
    Expr *lex, Expr *rex, SourceLocation OpLoc);
  inline QualType CheckRelationalOperands( // C99 6.5.8
    Expr *lex, Expr *rex, SourceLocation OpLoc);
  inline QualType CheckEqualityOperands( // C99 6.5.9
    Expr *lex, Expr *rex, SourceLocation OpLoc); 
  inline QualType CheckBitwiseOperands( // C99 6.5.[10...12]
    Expr *lex, Expr *rex, SourceLocation OpLoc); 
  inline QualType CheckLogicalOperands( // C99 6.5.[13,14]
    Expr *lex, Expr *rex, SourceLocation OpLoc);
  // CheckAssignmentOperands is used for both simple and compound assignment.
  // For simple assignment, pass both expressions and a null converted type.
  // For compound assignment, pass both expressions and the converted type.
  inline QualType CheckAssignmentOperands( // C99 6.5.16.[1,2]
    Expr *lex, Expr *rex, SourceLocation OpLoc, QualType convertedType);
  inline QualType CheckCommaOperands( // C99 6.5.17
    Expr *lex, Expr *rex, SourceLocation OpLoc);
  inline QualType CheckConditionalOperands( // C99 6.5.15
    Expr *cond, Expr *lhs, Expr *rhs, SourceLocation questionLoc);
  
  /// type checking unary operators (subroutines of ParseUnaryOp).
  QualType CheckIncrementDecrementOperand( // C99 6.5.3.1 
    Expr *op, SourceLocation loc);
  QualType CheckAddressOfOperand( // C99 6.5.3.2
    Expr *op, SourceLocation loc);
  QualType CheckIndirectionOperand( // C99 6.5.3.2
    Expr *op, SourceLocation loc);
  QualType CheckSizeOfAlignOfOperand( // C99 6.5.3.4
    QualType type, SourceLocation loc, bool isSizeof);
    
  // C99: 6.7.5p3: Used by ParseDeclarator/ParseField to make sure we have
  // a constant expression of type int with a value greater than zero.
  bool isConstantArrayType(ArrayType *ary, SourceLocation loc); 
};


}  // end namespace clang
}  // end namespace llvm

#endif
