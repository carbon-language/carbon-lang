//===--- Builder.cpp - AST Builder Implementation -------------------------===//
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

#include "clang/Parse/Action.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/Parse/Scope.h"
#include "clang/Lex/IdentifierTable.h"
#include "clang/Lex/LexerToken.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;
using namespace clang;

/// ASTBuilder
namespace {
class VISIBILITY_HIDDEN ASTBuilder : public Action {
  Preprocessor &PP;
  
  /// FullLocInfo - If this is true, the ASTBuilder constructs AST Nodes that
  /// capture maximal location information for each source-language construct.
  bool FullLocInfo;
  
  /// LastInGroupList - This vector is populated when there are multiple
  /// declarators in a single decl group (e.g. "int A, B, C").  In this case,
  /// all but the last decl will be entered into this.  This is used by the
  /// ASTStreamer.
  std::vector<Decl*> &LastInGroupList;
public:
  ASTBuilder(Preprocessor &pp, bool fullLocInfo,
             std::vector<Decl*> &prevInGroup)
    : PP(pp), FullLocInfo(fullLocInfo), LastInGroupList(prevInGroup) {}
  
  //===--------------------------------------------------------------------===//
  // Symbol table tracking callbacks.
  //
  virtual bool isTypedefName(const IdentifierInfo &II, Scope *S) const;
  virtual DeclTy *ParseDeclarator(Scope *S, Declarator &D, ExprTy *Init,
                                  DeclTy *LastInGroup);
  virtual DeclTy *ParseFunctionDefinition(Scope *S, Declarator &D,
                                          StmtTy *Body);
  virtual void PopScope(SourceLocation Loc, Scope *S);
  
  //===--------------------------------------------------------------------===//
  // Expression Parsing Callbacks.

  // Primary Expressions.
  virtual ExprResult ParseSimplePrimaryExpr(const LexerToken &Tok);
  virtual ExprResult ParseIntegerConstant(const LexerToken &Tok);
  virtual ExprResult ParseFloatingConstant(const LexerToken &Tok);
  virtual ExprResult ParseParenExpr(SourceLocation L, SourceLocation R,
                                    ExprTy *Val);
  virtual ExprResult ParseStringExpr(const char *StrData, unsigned StrLen,
                                     bool isWide,
                                     const LexerToken *Toks, unsigned NumToks);
  
  // Binary/Unary Operators.  'Tok' is the token for the operator.
  virtual ExprResult ParseUnaryOp(const LexerToken &Tok, ExprTy *Input);
  virtual ExprResult 
    ParseSizeOfAlignOfTypeExpr(SourceLocation OpLoc, bool isSizeof, 
                               SourceLocation LParenLoc, TypeTy *Ty,
                               SourceLocation RParenLoc);
  
  virtual ExprResult ParsePostfixUnaryOp(const LexerToken &Tok, ExprTy *Input);
  
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
  
  virtual ExprResult ParseBinOp(const LexerToken &Tok, ExprTy *LHS,ExprTy *RHS);
  
  /// ParseConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
  /// in the case of a the GNU conditional expr extension.
  virtual ExprResult ParseConditionalOp(SourceLocation QuestionLoc, 
                                        SourceLocation ColonLoc,
                                        ExprTy *Cond, ExprTy *LHS, ExprTy *RHS);
};
} // end anonymous namespace


//===----------------------------------------------------------------------===//
// Symbol table tracking callbacks.
//===----------------------------------------------------------------------===//

bool ASTBuilder::isTypedefName(const IdentifierInfo &II, Scope *S) const {
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
  
  if (LastInGroup) LastInGroupList.push_back((Decl*)LastInGroup);
  
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
// Expression Parsing Callbacks.
//===--------------------------------------------------------------------===//

Action::ExprResult ASTBuilder::ParseSimplePrimaryExpr(const LexerToken &Tok) {
  switch (Tok.getKind()) {
  default:
    assert(0 && "Unknown simple primary expr!");
  case tok::identifier: {
    // Could be enum-constant or decl.
    //Tok.getIdentifierInfo()
    return new DeclExpr(*(Decl*)0);
  }
    
  case tok::char_constant:     // constant: character-constant
  case tok::kw___func__:       // primary-expression: __func__ [C99 6.4.2.2]
  case tok::kw___FUNCTION__:   // primary-expression: __FUNCTION__ [GNU]
  case tok::kw___PRETTY_FUNCTION__:  // primary-expression: __P..Y_F..N__ [GNU]
    //assert(0 && "FIXME: Unimp so far!");
    return new DeclExpr(*(Decl*)0);
  }
}

Action::ExprResult ASTBuilder::ParseIntegerConstant(const LexerToken &Tok) {
  return new IntegerConstant();
}
Action::ExprResult ASTBuilder::ParseFloatingConstant(const LexerToken &Tok) {
  return new FloatingConstant();
}

Action::ExprResult ASTBuilder::ParseParenExpr(SourceLocation L, 
                                              SourceLocation R,
                                              ExprTy *Val) {
  if (!FullLocInfo) return Val;
  
  return new ParenExpr(L, R, (Expr*)Val);
}

/// ParseStringExpr - This accepts a string after semantic analysis. This string
/// may be the result of string concatenation ([C99 5.1.1.2, translation phase
/// #6]), so it may come from multiple tokens.
/// 
Action::ExprResult ASTBuilder::
ParseStringExpr(const char *StrData, unsigned StrLen, bool isWide,
                const LexerToken *Toks, unsigned NumToks) {
  assert(NumToks && "Must have at least one string!");
  
  if (!FullLocInfo)
    return new StringExpr(StrData, StrLen, isWide);
  else {
    SmallVector<SourceLocation, 4> Locs;
    for (unsigned i = 0; i != NumToks; ++i)
      Locs.push_back(Toks[i].getLocation());
    return new StringExprLOC(StrData, StrLen, isWide, &Locs[0], Locs.size());
  }
}


// Unary Operators.  'Tok' is the token for the operator.
Action::ExprResult ASTBuilder::ParseUnaryOp(const LexerToken &Tok,
                                            ExprTy *Input) {
  UnaryOperator::Opcode Opc;
  switch (Tok.getKind()) {
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
  }

  if (!FullLocInfo)
    return new UnaryOperator((Expr*)Input, Opc);
  else
    return new UnaryOperatorLOC(Tok.getLocation(), (Expr*)Input, Opc);
}

Action::ExprResult ASTBuilder::
ParseSizeOfAlignOfTypeExpr(SourceLocation OpLoc, bool isSizeof, 
                           SourceLocation LParenLoc, TypeTy *Ty,
                           SourceLocation RParenLoc) {
  if (!FullLocInfo)
    return new SizeOfAlignOfTypeExpr(isSizeof, (Type*)Ty);
  else
    return new SizeOfAlignOfTypeExprLOC(OpLoc, isSizeof, LParenLoc, (Type*)Ty,
                                        RParenLoc);
}


Action::ExprResult ASTBuilder::ParsePostfixUnaryOp(const LexerToken &Tok,
                                                   ExprTy *Input) {
  UnaryOperator::Opcode Opc;
  switch (Tok.getKind()) {
  default: assert(0 && "Unknown unary op!");
  case tok::plusplus:   Opc = UnaryOperator::PostInc; break;
  case tok::minusminus: Opc = UnaryOperator::PostDec; break;
  }
  
  if (!FullLocInfo)
    return new UnaryOperator((Expr*)Input, Opc);
  else
    return new UnaryOperatorLOC(Tok.getLocation(), (Expr*)Input, Opc);
}

Action::ExprResult ASTBuilder::
ParseArraySubscriptExpr(ExprTy *Base, SourceLocation LLoc,
                        ExprTy *Idx, SourceLocation RLoc) {
  if (!FullLocInfo)
    return new ArraySubscriptExpr((Expr*)Base, (Expr*)Idx);
  else
    return new ArraySubscriptExprLOC((Expr*)Base, LLoc, (Expr*)Idx, RLoc);
}

Action::ExprResult ASTBuilder::
ParseMemberReferenceExpr(ExprTy *Base, SourceLocation OpLoc,
                         tok::TokenKind OpKind, SourceLocation MemberLoc,
                         IdentifierInfo &Member) {
  Decl *MemberDecl = 0;
  // TODO: Look up MemberDecl.
  if (!FullLocInfo)
    return new MemberExpr((Expr*)Base, OpKind == tok::arrow, MemberDecl);
  else
    return new MemberExprLOC((Expr*)Base, OpLoc, OpKind == tok::arrow,
                             MemberLoc, MemberDecl);
}

/// ParseCallExpr - Handle a call to Fn with the specified array of arguments.
/// This provides the location of the left/right parens and a list of comma
/// locations.
Action::ExprResult ASTBuilder::
ParseCallExpr(ExprTy *Fn, SourceLocation LParenLoc,
              ExprTy **Args, unsigned NumArgs,
              SourceLocation *CommaLocs, SourceLocation RParenLoc) {
  if (!FullLocInfo)
    return new CallExpr((Expr*)Fn, (Expr**)Args, NumArgs);
  else
    return new CallExprLOC((Expr*)Fn, LParenLoc, (Expr**)Args, NumArgs,
                           CommaLocs, RParenLoc);
}

Action::ExprResult ASTBuilder::
ParseCastExpr(SourceLocation LParenLoc, TypeTy *Ty,
              SourceLocation RParenLoc, ExprTy *Op) {
  if (!FullLocInfo)
    return new CastExpr((Type*)Ty, (Expr*)Op);
  else
    return new CastExprLOC(LParenLoc, (Type*)Ty, RParenLoc, (Expr*)Op);
}



// Binary Operators.  'Tok' is the token for the operator.
Action::ExprResult ASTBuilder::ParseBinOp(const LexerToken &Tok, ExprTy *LHS,
                                          ExprTy *RHS) {
  BinaryOperator::Opcode Opc;
  switch (Tok.getKind()) {
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
  
  if (!FullLocInfo)
    return new BinaryOperator((Expr*)LHS, (Expr*)RHS, Opc);
  else
    return new BinaryOperatorLOC((Expr*)LHS, Tok.getLocation(), (Expr*)RHS,Opc);
}

/// ParseConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
/// in the case of a the GNU conditional expr extension.
Action::ExprResult ASTBuilder::ParseConditionalOp(SourceLocation QuestionLoc, 
                                                  SourceLocation ColonLoc,
                                                  ExprTy *Cond, ExprTy *LHS,
                                                  ExprTy *RHS) {
  if (!FullLocInfo)
    return new ConditionalOperator((Expr*)Cond, (Expr*)LHS, (Expr*)RHS);
  else
    return new ConditionalOperatorLOC((Expr*)Cond, QuestionLoc, (Expr*)LHS,
                                      ColonLoc, (Expr*)RHS);
}


/// Interface to the Builder.cpp file.
///
Action *CreateASTBuilderActions(Preprocessor &PP, bool FullLocInfo,
                                std::vector<Decl*> &LastInGroupList) {
  return new ASTBuilder(PP, FullLocInfo, LastInGroupList);
}

