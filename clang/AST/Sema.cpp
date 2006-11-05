//===--- ASTBuilder.cpp - AST Builder Implementation ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the actions class which builds an AST out of a parse
// stream.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTBuilder.h"
#include "clang/Parse/Action.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/Parse/Scope.h"
#include "clang/Lex/IdentifierTable.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// Symbol table tracking callbacks.
//===----------------------------------------------------------------------===//

bool ASTBuilder::isTypeName(const IdentifierInfo &II, Scope *S) const {
  Decl *D = II.getFETokenInfo<Decl>();
  return D != 0 && D->getDeclSpec().StorageClassSpec == DeclSpec::SCS_typedef;
}

Action::DeclTy *
ASTBuilder::ParseDeclarator(Scope *S, Declarator &D, ExprTy *Init, 
                            DeclTy *LastInGroup) {
  IdentifierInfo *II = D.getIdentifier();
  Decl *PrevDecl = II ? II->getFETokenInfo<Decl>() : 0;

  Decl *New;
  if (D.isFunctionDeclarator())
    New = new FunctionDecl(II, D, PrevDecl);
  else
    New = new VarDecl(II, D, PrevDecl);
  
  // If this has an identifier, add it to the scope stack.
  if (II) {
    // If PrevDecl includes conflicting name here, emit a diagnostic.
    II->setFETokenInfo(New);
    S->AddDecl(II);
  }
  
  // If this is a top-level decl that is chained to some other (e.g. int A,B,C;)
  // remember this in the LastInGroupList list.
  if (LastInGroup && S->getParent() == 0)
    LastInGroupList.push_back((Decl*)LastInGroup);
  
  return New;
}

Action::DeclTy *
ASTBuilder::ParseFunctionDefinition(Scope *S, Declarator &D, StmtTy *Body) {
  FunctionDecl *FD = (FunctionDecl *)ParseDeclarator(S, D, 0, 0);
  
  FD->setBody((Stmt*)Body);

  return FD;
}

void ASTBuilder::PopScope(SourceLocation Loc, Scope *S) {
  for (Scope::decl_iterator I = S->decl_begin(), E = S->decl_end();
       I != E; ++I) {
    IdentifierInfo &II = *static_cast<IdentifierInfo*>(*I);
    Decl *D = II.getFETokenInfo<Decl>();
    assert(D && "This decl didn't get pushed??");
    
    Decl *Next = D->getNext();

    // FIXME: Push the decl on the parent function list if in a function.
    delete D;
    
    II.setFETokenInfo(Next);
  }
}

//===--------------------------------------------------------------------===//
// Statement Parsing Callbacks.
//===--------------------------------------------------------------------===//

Action::StmtResult 
ASTBuilder::ParseCompoundStmt(SourceLocation L, SourceLocation R,
                              StmtTy **Elts, unsigned NumElts) {
  if (NumElts > 1)
    return new CompoundStmt((Stmt**)Elts, NumElts);
  else if (NumElts == 1)
    return Elts[0];        // {stmt} -> stmt
  else
    return 0;              // {}  -> ;
}

Action::StmtResult
ASTBuilder::ParseCaseStmt(SourceLocation CaseLoc, ExprTy *LHSVal,
                          SourceLocation DotDotDotLoc, ExprTy *RHSVal,
                          SourceLocation ColonLoc, StmtTy *SubStmt) {
  return new CaseStmt((Expr*)LHSVal, (Expr*)RHSVal, (Stmt*)SubStmt);
}

Action::StmtResult
ASTBuilder::ParseDefaultStmt(SourceLocation DefaultLoc,
                             SourceLocation ColonLoc, StmtTy *SubStmt) {
  return new DefaultStmt((Stmt*)SubStmt);
}

Action::StmtResult
ASTBuilder::ParseLabelStmt(SourceLocation IdentLoc, IdentifierInfo *II,
                           SourceLocation ColonLoc, StmtTy *SubStmt) {
  return new LabelStmt(II, (Stmt*)SubStmt);
}

Action::StmtResult 
ASTBuilder::ParseIfStmt(SourceLocation IfLoc, ExprTy *CondVal,
                        StmtTy *ThenVal, SourceLocation ElseLoc,
                        StmtTy *ElseVal) {
  return new IfStmt((Expr*)CondVal, (Stmt*)ThenVal, (Stmt*)ElseVal);
}
Action::StmtResult
ASTBuilder::ParseSwitchStmt(SourceLocation SwitchLoc, ExprTy *Cond,
                            StmtTy *Body) {
  return new SwitchStmt((Expr*)Cond, (Stmt*)Body);
}

Action::StmtResult
ASTBuilder::ParseWhileStmt(SourceLocation WhileLoc, ExprTy *Cond, StmtTy *Body){
  return new WhileStmt((Expr*)Cond, (Stmt*)Body);
}

Action::StmtResult
ASTBuilder::ParseDoStmt(SourceLocation DoLoc, StmtTy *Body,
                        SourceLocation WhileLoc, ExprTy *Cond) {
  return new DoStmt((Stmt*)Body, (Expr*)Cond);
}

Action::StmtResult 
ASTBuilder::ParseForStmt(SourceLocation ForLoc, SourceLocation LParenLoc, 
                         StmtTy *First, ExprTy *Second, ExprTy *Third,
                         SourceLocation RParenLoc, StmtTy *Body) {
  return new ForStmt((Stmt*)First, (Expr*)Second, (Expr*)Third, (Stmt*)Body);
}


Action::StmtResult 
ASTBuilder::ParseGotoStmt(SourceLocation GotoLoc, SourceLocation LabelLoc,
                          IdentifierInfo *LabelII) {
  return new GotoStmt(LabelII);
}
Action::StmtResult 
ASTBuilder::ParseIndirectGotoStmt(SourceLocation GotoLoc,SourceLocation StarLoc,
                                  ExprTy *DestExp) {
  return new IndirectGotoStmt((Expr*)DestExp);
}

Action::StmtResult 
ASTBuilder::ParseContinueStmt(SourceLocation ContinueLoc) {
  return new ContinueStmt();
}

Action::StmtResult 
ASTBuilder::ParseBreakStmt(SourceLocation GotoLoc) {
  return new BreakStmt();
}


Action::StmtResult
ASTBuilder::ParseReturnStmt(SourceLocation ReturnLoc,
                            ExprTy *RetValExp) {
  return new ReturnStmt((Expr*)RetValExp);
}

//===--------------------------------------------------------------------===//
// Expression Parsing Callbacks.
//===--------------------------------------------------------------------===//

Action::ExprResult ASTBuilder::ParseSimplePrimaryExpr(SourceLocation Loc,
                                                      tok::TokenKind Kind) {
  switch (Kind) {
  default:
    assert(0 && "Unknown simple primary expr!");
  case tok::identifier: {
    // Could be enum-constant or decl.
    //Tok.getIdentifierInfo()
    return new DeclRefExpr(*(Decl*)0);
  }
    
  case tok::char_constant:     // constant: character-constant
  case tok::kw___func__:       // primary-expression: __func__ [C99 6.4.2.2]
  case tok::kw___FUNCTION__:   // primary-expression: __FUNCTION__ [GNU]
  case tok::kw___PRETTY_FUNCTION__:  // primary-expression: __P..Y_F..N__ [GNU]
    //assert(0 && "FIXME: Unimp so far!");
    return new DeclRefExpr(*(Decl*)0);
  }
}

Action::ExprResult ASTBuilder::ParseIntegerConstant(SourceLocation Loc) {
  return new IntegerConstant();
}
Action::ExprResult ASTBuilder::ParseFloatingConstant(SourceLocation Loc) {
  return new FloatingConstant();
}

Action::ExprResult ASTBuilder::ParseParenExpr(SourceLocation L, 
                                              SourceLocation R,
                                              ExprTy *Val) {
  return Val;
}

/// ParseStringExpr - This accepts a string after semantic analysis. This string
/// may be the result of string concatenation ([C99 5.1.1.2, translation phase
/// #6]), so it may come from multiple tokens.
/// 
Action::ExprResult ASTBuilder::
ParseStringExpr(const char *StrData, unsigned StrLen, bool isWide,
                SourceLocation *TokLocs, unsigned NumToks) {
  assert(NumToks && "Must have at least one string!");
  return new StringExpr(StrData, StrLen, isWide);
}


// Unary Operators.  'Tok' is the token for the operator.
Action::ExprResult ASTBuilder::ParseUnaryOp(SourceLocation OpLoc,
                                            tok::TokenKind Op,
                                            ExprTy *Input) {
  UnaryOperator::Opcode Opc;
  switch (Op) {
  default: assert(0 && "Unknown unary op!");
  case tok::plusplus:     Opc = UnaryOperator::PreInc; break;
  case tok::minusminus:   Opc = UnaryOperator::PreDec; break;
  case tok::amp:          Opc = UnaryOperator::AddrOf; break;
  case tok::star:         Opc = UnaryOperator::Deref; break;
  case tok::plus:         Opc = UnaryOperator::Plus; break;
  case tok::minus:        Opc = UnaryOperator::Minus; break;
  case tok::tilde:        Opc = UnaryOperator::Not; break;
  case tok::exclaim:      Opc = UnaryOperator::LNot; break;
  case tok::kw_sizeof:    Opc = UnaryOperator::SizeOf; break;
  case tok::kw___alignof: Opc = UnaryOperator::AlignOf; break;
  case tok::kw___real:    Opc = UnaryOperator::Real; break;
  case tok::kw___imag:    Opc = UnaryOperator::Imag; break;
  case tok::ampamp:       Opc = UnaryOperator::AddrLabel; break;
  case tok::kw___extension__: 
    return Input;
    //Opc = UnaryOperator::Extension;
    //break;
  }

  return new UnaryOperator((Expr*)Input, Opc);
}

Action::ExprResult ASTBuilder::
ParseSizeOfAlignOfTypeExpr(SourceLocation OpLoc, bool isSizeof, 
                           SourceLocation LParenLoc, TypeTy *Ty,
                           SourceLocation RParenLoc) {
  return new SizeOfAlignOfTypeExpr(isSizeof, (Type*)Ty);
}


Action::ExprResult ASTBuilder::ParsePostfixUnaryOp(SourceLocation OpLoc, 
                                                   tok::TokenKind Kind,
                                                   ExprTy *Input) {
  UnaryOperator::Opcode Opc;
  switch (Kind) {
  default: assert(0 && "Unknown unary op!");
  case tok::plusplus:   Opc = UnaryOperator::PostInc; break;
  case tok::minusminus: Opc = UnaryOperator::PostDec; break;
  }
  
  return new UnaryOperator((Expr*)Input, Opc);
}

Action::ExprResult ASTBuilder::
ParseArraySubscriptExpr(ExprTy *Base, SourceLocation LLoc,
                        ExprTy *Idx, SourceLocation RLoc) {
  return new ArraySubscriptExpr((Expr*)Base, (Expr*)Idx);
}

Action::ExprResult ASTBuilder::
ParseMemberReferenceExpr(ExprTy *Base, SourceLocation OpLoc,
                         tok::TokenKind OpKind, SourceLocation MemberLoc,
                         IdentifierInfo &Member) {
  Decl *MemberDecl = 0;
  // TODO: Look up MemberDecl.
  return new MemberExpr((Expr*)Base, OpKind == tok::arrow, MemberDecl);
}

/// ParseCallExpr - Handle a call to Fn with the specified array of arguments.
/// This provides the location of the left/right parens and a list of comma
/// locations.
Action::ExprResult ASTBuilder::
ParseCallExpr(ExprTy *Fn, SourceLocation LParenLoc,
              ExprTy **Args, unsigned NumArgs,
              SourceLocation *CommaLocs, SourceLocation RParenLoc) {
  return new CallExpr((Expr*)Fn, (Expr**)Args, NumArgs);
}

Action::ExprResult ASTBuilder::
ParseCastExpr(SourceLocation LParenLoc, TypeTy *Ty,
              SourceLocation RParenLoc, ExprTy *Op) {
  return new CastExpr((Type*)Ty, (Expr*)Op);
}



// Binary Operators.  'Tok' is the token for the operator.
Action::ExprResult ASTBuilder::ParseBinOp(SourceLocation TokLoc, 
                                          tok::TokenKind Kind, ExprTy *LHS,
                                          ExprTy *RHS) {
  BinaryOperator::Opcode Opc;
  switch (Kind) {
  default: assert(0 && "Unknown binop!");
  case tok::star:                 Opc = BinaryOperator::Mul; break;
  case tok::slash:                Opc = BinaryOperator::Div; break;
  case tok::percent:              Opc = BinaryOperator::Rem; break;
  case tok::plus:                 Opc = BinaryOperator::Add; break;
  case tok::minus:                Opc = BinaryOperator::Sub; break;
  case tok::lessless:             Opc = BinaryOperator::Shl; break;
  case tok::greatergreater:       Opc = BinaryOperator::Shr; break;
  case tok::lessequal:            Opc = BinaryOperator::LE; break;
  case tok::less:                 Opc = BinaryOperator::LT; break;
  case tok::greaterequal:         Opc = BinaryOperator::GE; break;
  case tok::greater:              Opc = BinaryOperator::GT; break;
  case tok::exclaimequal:         Opc = BinaryOperator::NE; break;
  case tok::equalequal:           Opc = BinaryOperator::EQ; break;
  case tok::amp:                  Opc = BinaryOperator::And; break;
  case tok::caret:                Opc = BinaryOperator::Xor; break;
  case tok::pipe:                 Opc = BinaryOperator::Or; break;
  case tok::ampamp:               Opc = BinaryOperator::LAnd; break;
  case tok::pipepipe:             Opc = BinaryOperator::LOr; break;
  case tok::equal:                Opc = BinaryOperator::Assign; break;
  case tok::starequal:            Opc = BinaryOperator::MulAssign; break;
  case tok::slashequal:           Opc = BinaryOperator::DivAssign; break;
  case tok::percentequal:         Opc = BinaryOperator::RemAssign; break;
  case tok::plusequal:            Opc = BinaryOperator::AddAssign; break;
  case tok::minusequal:           Opc = BinaryOperator::SubAssign; break;
  case tok::lesslessequal:        Opc = BinaryOperator::ShlAssign; break;
  case tok::greatergreaterequal:  Opc = BinaryOperator::ShrAssign; break;
  case tok::ampequal:             Opc = BinaryOperator::AndAssign; break;
  case tok::caretequal:           Opc = BinaryOperator::XorAssign; break;
  case tok::pipeequal:            Opc = BinaryOperator::OrAssign; break;
  case tok::comma:                Opc = BinaryOperator::Comma; break;
  }
  
  return new BinaryOperator((Expr*)LHS, (Expr*)RHS, Opc);
}

/// ParseConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
/// in the case of a the GNU conditional expr extension.
Action::ExprResult ASTBuilder::ParseConditionalOp(SourceLocation QuestionLoc, 
                                                  SourceLocation ColonLoc,
                                                  ExprTy *Cond, ExprTy *LHS,
                                                  ExprTy *RHS) {
  return new ConditionalOperator((Expr*)Cond, (Expr*)LHS, (Expr*)RHS);
}

