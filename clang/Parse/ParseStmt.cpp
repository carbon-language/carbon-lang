//===--- ParseStmt.cpp - Statement and Block Parser -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Statement and Block portions of the Parser
// interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/Scope.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// C99 6.8: Statements and Blocks.
//===----------------------------------------------------------------------===//

/// ParseStatementOrDeclaration - Read 'statement' or 'declaration'.
///       StatementOrDeclaration:
///         statement
///         declaration
///
///       statement:
///         labeled-statement
///         compound-statement
///         expression-statement
///         selection-statement
///         iteration-statement
///         jump-statement
/// [OBC]   objc-throw-statement
/// [OBC]   objc-try-catch-statement
/// [OBC]   objc-synchronized-statement  [TODO]
/// [GNU]   asm-statement
/// [OMP]   openmp-construct             [TODO]
///
///       labeled-statement:
///         identifier ':' statement
///         'case' constant-expression ':' statement
///         'default' ':' statement
///
///       selection-statement:
///         if-statement
///         switch-statement
///
///       iteration-statement:
///         while-statement
///         do-statement
///         for-statement
///
///       expression-statement:
///         expression[opt] ';'
///
///       jump-statement:
///         'goto' identifier ';'
///         'continue' ';'
///         'break' ';'
///         'return' expression[opt] ';'
/// [GNU]   'goto' '*' expression ';'
///
/// [OBC] objc-throw-statement:
/// [OBC]   '@' 'throw' expression ';'
/// [OBC]   '@' 'throw' ';' 
/// 
Parser::StmtResult Parser::ParseStatementOrDeclaration(bool OnlyStatement) {
  const char *SemiError = 0;
  Parser::StmtResult Res;
  
  // Cases in this switch statement should fall through if the parser expects
  // the token to end in a semicolon (in which case SemiError should be set),
  // or they directly 'return;' if not.
  tok::TokenKind Kind  = Tok.getKind();
  SourceLocation AtLoc;
  switch (Kind) {
  case tok::identifier:             // C99 6.8.1: labeled-statement
    // identifier ':' statement
    // declaration                  (if !OnlyStatement)
    // expression[opt] ';'
    return ParseIdentifierStatement(OnlyStatement);

  case tok::at: // May be a @try or @throw statement
    {
      AtLoc = ConsumeToken();  // consume @
      if (Tok.getIdentifierInfo()->getObjCKeywordID() == tok::objc_try)
        return ParseObjCTryStmt(AtLoc);
      else if (Tok.getIdentifierInfo()->getObjCKeywordID() == tok::objc_throw)
        return ParseObjCThrowStmt(AtLoc);
      ExprResult Res = ParseExpressionWithLeadingAt(AtLoc);
      if (Res.isInvalid) {
        // If the expression is invalid, skip ahead to the next semicolon. Not
        // doing this opens us up to the possibility of infinite loops if
        // ParseExpression does not consume any tokens.
        SkipUntil(tok::semi);
        return true;
      }
      // Otherwise, eat the semicolon.
      ExpectAndConsume(tok::semi, diag::err_expected_semi_after_expr);
      return Actions.ActOnExprStmt(Res.Val);
    }

  default:
    if (!OnlyStatement && isDeclarationSpecifier()) {
      return Actions.ActOnDeclStmt(ParseDeclaration(Declarator::BlockContext));
    } else if (Tok.is(tok::r_brace)) {
      Diag(Tok, diag::err_expected_statement);
      return true;
    } else {
      // expression[opt] ';'
      ExprResult Res = ParseExpression();
      if (Res.isInvalid) {
        // If the expression is invalid, skip ahead to the next semicolon.  Not
        // doing this opens us up to the possibility of infinite loops if
        // ParseExpression does not consume any tokens.
        SkipUntil(tok::semi);
        return true;
      }
      // Otherwise, eat the semicolon.
      ExpectAndConsume(tok::semi, diag::err_expected_semi_after_expr);
      return Actions.ActOnExprStmt(Res.Val);
    }
    
  case tok::kw_case:                // C99 6.8.1: labeled-statement
    return ParseCaseStatement();
  case tok::kw_default:             // C99 6.8.1: labeled-statement
    return ParseDefaultStatement();
    
  case tok::l_brace:                // C99 6.8.2: compound-statement
    return ParseCompoundStatement();
  case tok::semi:                   // C99 6.8.3p3: expression[opt] ';'
    return Actions.ActOnNullStmt(ConsumeToken());
    
  case tok::kw_if:                  // C99 6.8.4.1: if-statement
    return ParseIfStatement();
  case tok::kw_switch:              // C99 6.8.4.2: switch-statement
    return ParseSwitchStatement();
    
  case tok::kw_while:               // C99 6.8.5.1: while-statement
    return ParseWhileStatement();
  case tok::kw_do:                  // C99 6.8.5.2: do-statement
    Res = ParseDoStatement();
    SemiError = "do/while loop";
    break;
  case tok::kw_for:                 // C99 6.8.5.3: for-statement
    return ParseForStatement();

  case tok::kw_goto:                // C99 6.8.6.1: goto-statement
    Res = ParseGotoStatement();
    SemiError = "goto statement";
    break;
  case tok::kw_continue:            // C99 6.8.6.2: continue-statement
    Res = ParseContinueStatement();
    SemiError = "continue statement";
    break;
  case tok::kw_break:               // C99 6.8.6.3: break-statement
    Res = ParseBreakStatement();
    SemiError = "break statement";
    break;
  case tok::kw_return:              // C99 6.8.6.4: return-statement
    Res = ParseReturnStatement();
    SemiError = "return statement";
    break;
    
  case tok::kw_asm:
    Res = ParseAsmStatement();
    SemiError = "asm statement";
    break;
  }
  
  // If we reached this code, the statement must end in a semicolon.
  if (Tok.is(tok::semi)) {
    ConsumeToken();
  } else {
    Diag(Tok, diag::err_expected_semi_after, SemiError);
    SkipUntil(tok::semi);
  }
  return Res;
}

/// ParseIdentifierStatement - Because we don't have two-token lookahead, we
/// have a bit of a quandry here.  Reading the identifier is necessary to see if
/// there is a ':' after it.  If there is, this is a label, regardless of what
/// else the identifier can mean.  If not, this is either part of a declaration
/// (if the identifier is a type-name) or part of an expression.
///
///       labeled-statement:
///         identifier ':' statement
/// [GNU]   identifier ':' attributes[opt] statement
///         declaration                  (if !OnlyStatement)
///         expression[opt] ';'
///
Parser::StmtResult Parser::ParseIdentifierStatement(bool OnlyStatement) {
  assert(Tok.is(tok::identifier) && Tok.getIdentifierInfo() &&
         "Not an identifier!");

  Token IdentTok = Tok;  // Save the whole token.
  ConsumeToken();  // eat the identifier.
  
  // identifier ':' statement
  if (Tok.is(tok::colon)) {
    SourceLocation ColonLoc = ConsumeToken();

    // Read label attributes, if present.
    DeclTy *AttrList = 0;
    if (Tok.is(tok::kw___attribute))
      // TODO: save these somewhere.
      AttrList = ParseAttributes();

    StmtResult SubStmt = ParseStatement();
    
    // Broken substmt shouldn't prevent the label from being added to the AST.
    if (SubStmt.isInvalid)
      SubStmt = Actions.ActOnNullStmt(ColonLoc);
    
    return Actions.ActOnLabelStmt(IdentTok.getLocation(), 
                                  IdentTok.getIdentifierInfo(),
                                  ColonLoc, SubStmt.Val);
  }
  
  // Check to see if this is a declaration.
  void *TypeRep;
  if (!OnlyStatement &&
      (TypeRep = Actions.isTypeName(*IdentTok.getIdentifierInfo(), CurScope))) {
    // Handle this.  Warn/disable if in middle of block and !C99.
    DeclSpec DS;
    
    // Add the typedef name to the start of the decl-specs.
    const char *PrevSpec = 0;
    int isInvalid = DS.SetTypeSpecType(DeclSpec::TST_typedef,
                                       IdentTok.getLocation(), PrevSpec,
                                       TypeRep);
    assert(!isInvalid && "First declspec can't be invalid!");
    if (Tok.is(tok::less)) {
      llvm::SmallVector<IdentifierInfo *, 8> ProtocolRefs;
      ParseObjCProtocolReferences(ProtocolRefs);
      llvm::SmallVector<DeclTy *, 8> *ProtocolDecl = 
              new llvm::SmallVector<DeclTy *, 8>;
      DS.setProtocolQualifiers(ProtocolDecl);
      Actions.FindProtocolDeclaration(IdentTok.getLocation(), 
                                      &ProtocolRefs[0], ProtocolRefs.size(),
                                      *ProtocolDecl);
    }    
    
    // ParseDeclarationSpecifiers will continue from there.
    ParseDeclarationSpecifiers(DS);

    // C99 6.7.2.3p6: Handle "struct-or-union identifier;", "enum { X };"
    // declaration-specifiers init-declarator-list[opt] ';'
    if (Tok.is(tok::semi)) {
      // TODO: emit error on 'int;' or 'const enum foo;'.
      // if (!DS.isMissingDeclaratorOk()) Diag(...);
      
      ConsumeToken();
      // FIXME: Return this as a type decl.
      return 0;
    }
    
    // Parse all the declarators.
    Declarator DeclaratorInfo(DS, Declarator::BlockContext);
    ParseDeclarator(DeclaratorInfo);
    
    DeclTy *Decl = ParseInitDeclaratorListAfterFirstDeclarator(DeclaratorInfo);
    return Decl ? Actions.ActOnDeclStmt(Decl) : 0;
  }
  
  // Otherwise, this is an expression.  Seed it with II and parse it.
  ExprResult Res = ParseExpressionWithLeadingIdentifier(IdentTok);
  if (Res.isInvalid) {
    SkipUntil(tok::semi);
    return true;
  } else if (Tok.isNot(tok::semi)) {
    Diag(Tok, diag::err_expected_semi_after, "expression");
    SkipUntil(tok::semi);
    return true;
  } else {
    ConsumeToken();
    // Convert expr to a stmt.
    return Actions.ActOnExprStmt(Res.Val);
  }
}

/// ParseCaseStatement
///       labeled-statement:
///         'case' constant-expression ':' statement
/// [GNU]   'case' constant-expression '...' constant-expression ':' statement
///
/// Note that this does not parse the 'statement' at the end.
///
Parser::StmtResult Parser::ParseCaseStatement() {
  assert(Tok.is(tok::kw_case) && "Not a case stmt!");
  SourceLocation CaseLoc = ConsumeToken();  // eat the 'case'.

  ExprResult LHS = ParseConstantExpression();
  if (LHS.isInvalid) {
    SkipUntil(tok::colon);
    return true;
  }
  
  // GNU case range extension.
  SourceLocation DotDotDotLoc;
  ExprTy *RHSVal = 0;
  if (Tok.is(tok::ellipsis)) {
    Diag(Tok, diag::ext_gnu_case_range);
    DotDotDotLoc = ConsumeToken();
    
    ExprResult RHS = ParseConstantExpression();
    if (RHS.isInvalid) {
      SkipUntil(tok::colon);
      return true;
    }
    RHSVal = RHS.Val;
  }
  
  if (Tok.isNot(tok::colon)) {
    Diag(Tok, diag::err_expected_colon_after, "'case'");
    SkipUntil(tok::colon);
    return true;
  }
  
  SourceLocation ColonLoc = ConsumeToken();
  
  // Diagnose the common error "switch (X) { case 4: }", which is not valid.
  if (Tok.is(tok::r_brace)) {
    Diag(Tok, diag::err_label_end_of_compound_statement);
    return true;
  }
  
  StmtResult SubStmt = ParseStatement();

  // Broken substmt shouldn't prevent the case from being added to the AST.
  if (SubStmt.isInvalid)
    SubStmt = Actions.ActOnNullStmt(ColonLoc);
  
  return Actions.ActOnCaseStmt(CaseLoc, LHS.Val, DotDotDotLoc, RHSVal, ColonLoc,
                               SubStmt.Val);
}

/// ParseDefaultStatement
///       labeled-statement:
///         'default' ':' statement
/// Note that this does not parse the 'statement' at the end.
///
Parser::StmtResult Parser::ParseDefaultStatement() {
  assert(Tok.is(tok::kw_default) && "Not a default stmt!");
  SourceLocation DefaultLoc = ConsumeToken();  // eat the 'default'.

  if (Tok.isNot(tok::colon)) {
    Diag(Tok, diag::err_expected_colon_after, "'default'");
    SkipUntil(tok::colon);
    return true;
  }
  
  SourceLocation ColonLoc = ConsumeToken();
  
  // Diagnose the common error "switch (X) {... default: }", which is not valid.
  if (Tok.is(tok::r_brace)) {
    Diag(Tok, diag::err_label_end_of_compound_statement);
    return true;
  }

  StmtResult SubStmt = ParseStatement();
  if (SubStmt.isInvalid)
    return true;
  
  return Actions.ActOnDefaultStmt(DefaultLoc, ColonLoc, SubStmt.Val, CurScope);
}


/// ParseCompoundStatement - Parse a "{}" block.
///
///       compound-statement: [C99 6.8.2]
///         { block-item-list[opt] }
/// [GNU]   { label-declarations block-item-list } [TODO]
///
///       block-item-list:
///         block-item
///         block-item-list block-item
///
///       block-item:
///         declaration
/// [GNU]   '__extension__' declaration
///         statement
/// [OMP]   openmp-directive            [TODO]
///
/// [GNU] label-declarations:
/// [GNU]   label-declaration
/// [GNU]   label-declarations label-declaration
///
/// [GNU] label-declaration:
/// [GNU]   '__label__' identifier-list ';'
///
/// [OMP] openmp-directive:             [TODO]
/// [OMP]   barrier-directive
/// [OMP]   flush-directive
///
Parser::StmtResult Parser::ParseCompoundStatement(bool isStmtExpr) {
  assert(Tok.is(tok::l_brace) && "Not a compount stmt!");
  
  // Enter a scope to hold everything within the compound stmt.  Compound
  // statements can always hold declarations.
  EnterScope(Scope::DeclScope);

  // Parse the statements in the body.
  StmtResult Body = ParseCompoundStatementBody(isStmtExpr);

  ExitScope();
  return Body;
}


/// ParseCompoundStatementBody - Parse a sequence of statements and invoke the
/// ActOnCompoundStmt action.  This expects the '{' to be the current token, and
/// consume the '}' at the end of the block.  It does not manipulate the scope
/// stack.
Parser::StmtResult Parser::ParseCompoundStatementBody(bool isStmtExpr) {
  SourceLocation LBraceLoc = ConsumeBrace();  // eat the '{'.

  // TODO: "__label__ X, Y, Z;" is the GNU "Local Label" extension.  These are
  // only allowed at the start of a compound stmt regardless of the language.
  
  llvm::SmallVector<StmtTy*, 32> Stmts;
  while (Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)) {
    StmtResult R;
    if (Tok.isNot(tok::kw___extension__)) {
      R = ParseStatementOrDeclaration(false);
    } else {
      // __extension__ can start declarations and it can also be a unary
      // operator for expressions.  Consume multiple __extension__ markers here
      // until we can determine which is which.
      SourceLocation ExtLoc = ConsumeToken();
      while (Tok.is(tok::kw___extension__))
        ConsumeToken();
      
      // If this is the start of a declaration, parse it as such.
      if (isDeclarationSpecifier()) {
        // FIXME: Save the __extension__ on the decl as a node somehow.
        // FIXME: disable extwarns.
        R = Actions.ActOnDeclStmt(ParseDeclaration(Declarator::BlockContext));
      } else {
        // Otherwise this was a unary __extension__ marker.  Parse the
        // subexpression and add the __extension__ unary op. 
        // FIXME: disable extwarns.
        ExprResult Res = ParseCastExpression(false);
        if (Res.isInvalid) {
          SkipUntil(tok::semi);
          continue;
        }
        
        // Add the __extension__ node to the AST.
        Res = Actions.ActOnUnaryOp(ExtLoc, tok::kw___extension__, Res.Val);
        if (Res.isInvalid)
          continue;
        
        // Eat the semicolon at the end of stmt and convert the expr into a stmt.
        ExpectAndConsume(tok::semi, diag::err_expected_semi_after_expr);
        R = Actions.ActOnExprStmt(Res.Val);
      }
    }
    
    if (!R.isInvalid && R.Val)
      Stmts.push_back(R.Val);
  }
  
  // We broke out of the while loop because we found a '}' or EOF.
  if (Tok.isNot(tok::r_brace)) {
    Diag(Tok, diag::err_expected_rbrace);
    return 0;
  }
  
  SourceLocation RBraceLoc = ConsumeBrace();
  return Actions.ActOnCompoundStmt(LBraceLoc, RBraceLoc,
                                   &Stmts[0], Stmts.size(), isStmtExpr);
}

/// ParseIfStatement
///       if-statement: [C99 6.8.4.1]
///         'if' '(' expression ')' statement
///         'if' '(' expression ')' statement 'else' statement
///
Parser::StmtResult Parser::ParseIfStatement() {
  assert(Tok.is(tok::kw_if) && "Not an if stmt!");
  SourceLocation IfLoc = ConsumeToken();  // eat the 'if'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after, "if");
    SkipUntil(tok::semi);
    return true;
  }
  
  // C99 6.8.4p3 - In C99, the if statement is a block.  This is not
  // the case for C90.
  if (getLang().C99)
    EnterScope(Scope::DeclScope);

  // Parse the condition.
  ExprResult CondExp = ParseSimpleParenExpression();
  if (CondExp.isInvalid) {
    SkipUntil(tok::semi);
    if (getLang().C99)
      ExitScope();
    return true;
  }
  
  // C99 6.8.4p3 - In C99, the body of the if statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  bool NeedsInnerScope = getLang().C99 && Tok.isNot(tok::l_brace);
  if (NeedsInnerScope) EnterScope(Scope::DeclScope);
  
  // Read the if condition.
  StmtResult CondStmt = ParseStatement();

  // Broken substmt shouldn't prevent the label from being added to the AST.
  if (CondStmt.isInvalid)
    CondStmt = Actions.ActOnNullStmt(Tok.getLocation());
  
  // Pop the 'if' scope if needed.
  if (NeedsInnerScope) ExitScope();
  
  // If it has an else, parse it.
  SourceLocation ElseLoc;
  StmtResult ElseStmt(false);
  if (Tok.is(tok::kw_else)) {
    ElseLoc = ConsumeToken();
    
    // C99 6.8.4p3 - In C99, the body of the if statement is a scope, even if
    // there is no compound stmt.  C90 does not have this clause.  We only do
    // this if the body isn't a compound statement to avoid push/pop in common
    // cases.
    NeedsInnerScope = getLang().C99 && Tok.isNot(tok::l_brace);
    if (NeedsInnerScope) EnterScope(Scope::DeclScope);
    
    ElseStmt = ParseStatement();

    // Pop the 'else' scope if needed.
    if (NeedsInnerScope) ExitScope();
    
    if (ElseStmt.isInvalid)
      ElseStmt = Actions.ActOnNullStmt(ElseLoc);
  }
  
  if (getLang().C99)
    ExitScope();

  return Actions.ActOnIfStmt(IfLoc, CondExp.Val, CondStmt.Val,
                             ElseLoc, ElseStmt.Val);
}

/// ParseSwitchStatement
///       switch-statement:
///         'switch' '(' expression ')' statement
Parser::StmtResult Parser::ParseSwitchStatement() {
  assert(Tok.is(tok::kw_switch) && "Not a switch stmt!");
  SourceLocation SwitchLoc = ConsumeToken();  // eat the 'switch'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after, "switch");
    SkipUntil(tok::semi);
    return true;
  }

  // C99 6.8.4p3 - In C99, the switch statement is a block.  This is
  // not the case for C90.  Start the switch scope.
  if (getLang().C99)
    EnterScope(Scope::BreakScope|Scope::DeclScope);
  else
    EnterScope(Scope::BreakScope);

  // Parse the condition.
  ExprResult Cond = ParseSimpleParenExpression();
  
  if (Cond.isInvalid) {
    ExitScope();
    return true;
  }
    
  StmtResult Switch = Actions.ActOnStartOfSwitchStmt(Cond.Val);
  
  // C99 6.8.4p3 - In C99, the body of the switch statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  bool NeedsInnerScope = getLang().C99 && Tok.isNot(tok::l_brace);
  if (NeedsInnerScope) EnterScope(Scope::DeclScope);
  
  // Read the body statement.
  StmtResult Body = ParseStatement();

  // Pop the body scope if needed.
  if (NeedsInnerScope) ExitScope();
  
  if (Body.isInvalid) {
    Body = Actions.ActOnNullStmt(Tok.getLocation());
    // FIXME: Remove the case statement list from the Switch statement.
  }
  
  ExitScope();
  
  return Actions.ActOnFinishSwitchStmt(SwitchLoc, Switch.Val, Body.Val);
}

/// ParseWhileStatement
///       while-statement: [C99 6.8.5.1]
///         'while' '(' expression ')' statement
Parser::StmtResult Parser::ParseWhileStatement() {
  assert(Tok.is(tok::kw_while) && "Not a while stmt!");
  SourceLocation WhileLoc = Tok.getLocation();
  ConsumeToken();  // eat the 'while'.
  
  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after, "while");
    SkipUntil(tok::semi);
    return true;
  }
  
  // C99 6.8.5p5 - In C99, the while statement is a block.  This is not
  // the case for C90.  Start the loop scope.
  if (getLang().C99)
    EnterScope(Scope::BreakScope | Scope::ContinueScope | Scope::DeclScope);
  else
    EnterScope(Scope::BreakScope | Scope::ContinueScope);

  // Parse the condition.
  ExprResult Cond = ParseSimpleParenExpression();
  
  // C99 6.8.5p5 - In C99, the body of the if statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  bool NeedsInnerScope = getLang().C99 && Tok.isNot(tok::l_brace);
  if (NeedsInnerScope) EnterScope(Scope::DeclScope);
  
  // Read the body statement.
  StmtResult Body = ParseStatement();

  // Pop the body scope if needed.
  if (NeedsInnerScope) ExitScope();

  ExitScope();
  
  if (Cond.isInvalid || Body.isInvalid) return true;
  
  return Actions.ActOnWhileStmt(WhileLoc, Cond.Val, Body.Val);
}

/// ParseDoStatement
///       do-statement: [C99 6.8.5.2]
///         'do' statement 'while' '(' expression ')' ';'
/// Note: this lets the caller parse the end ';'.
Parser::StmtResult Parser::ParseDoStatement() {
  assert(Tok.is(tok::kw_do) && "Not a do stmt!");
  SourceLocation DoLoc = ConsumeToken();  // eat the 'do'.
  
  // C99 6.8.5p5 - In C99, the do statement is a block.  This is not
  // the case for C90.  Start the loop scope.
  if (getLang().C99)
    EnterScope(Scope::BreakScope | Scope::ContinueScope | Scope::DeclScope);
  else
    EnterScope(Scope::BreakScope | Scope::ContinueScope);

  // C99 6.8.5p5 - In C99, the body of the if statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause. We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  bool NeedsInnerScope = getLang().C99 && Tok.isNot(tok::l_brace);
  if (NeedsInnerScope) EnterScope(Scope::DeclScope);
  
  // Read the body statement.
  StmtResult Body = ParseStatement();

  // Pop the body scope if needed.
  if (NeedsInnerScope) ExitScope();

  if (Tok.isNot(tok::kw_while)) {
    ExitScope();
    Diag(Tok, diag::err_expected_while);
    Diag(DoLoc, diag::err_matching, "do");
    SkipUntil(tok::semi);
    return true;
  }
  SourceLocation WhileLoc = ConsumeToken();
  
  if (Tok.isNot(tok::l_paren)) {
    ExitScope();
    Diag(Tok, diag::err_expected_lparen_after, "do/while");
    SkipUntil(tok::semi);
    return true;
  }
  
  // Parse the condition.
  ExprResult Cond = ParseSimpleParenExpression();
  
  ExitScope();
  
  if (Cond.isInvalid || Body.isInvalid) return true;
  
  return Actions.ActOnDoStmt(DoLoc, Body.Val, WhileLoc, Cond.Val);
}

/// ParseForStatement
///       for-statement: [C99 6.8.5.3]
///         'for' '(' expr[opt] ';' expr[opt] ';' expr[opt] ')' statement
///         'for' '(' declaration expr[opt] ';' expr[opt] ')' statement
Parser::StmtResult Parser::ParseForStatement() {
  assert(Tok.is(tok::kw_for) && "Not a for stmt!");
  SourceLocation ForLoc = ConsumeToken();  // eat the 'for'.
  
  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after, "for");
    SkipUntil(tok::semi);
    return true;
  }
  
  // C99 6.8.5p5 - In C99, the for statement is a block.  This is not
  // the case for C90.  Start the loop scope.
  if (getLang().C99)
    EnterScope(Scope::BreakScope | Scope::ContinueScope | Scope::DeclScope);
  else
    EnterScope(Scope::BreakScope | Scope::ContinueScope);

  SourceLocation LParenLoc = ConsumeParen();
  ExprResult Value;
  
  StmtTy *FirstPart = 0;
  ExprTy *SecondPart = 0;
  StmtTy *ThirdPart = 0;
  
  // Parse the first part of the for specifier.
  if (Tok.is(tok::semi)) {  // for (;
    // no first part, eat the ';'.
    ConsumeToken();
  } else if (isDeclarationSpecifier()) {  // for (int X = 4;
    // Parse declaration, which eats the ';'.
    if (!getLang().C99)   // Use of C99-style for loops in C90 mode?
      Diag(Tok, diag::ext_c99_variable_decl_in_for_loop);
    DeclTy *aBlockVarDecl = ParseDeclaration(Declarator::ForContext);
    StmtResult stmtResult = Actions.ActOnDeclStmt(aBlockVarDecl);
    FirstPart = stmtResult.isInvalid ? 0 : stmtResult.Val;
  } else {
    Value = ParseExpression();

    // Turn the expression into a stmt.
    if (!Value.isInvalid) {
      StmtResult R = Actions.ActOnExprStmt(Value.Val);
      if (!R.isInvalid)
        FirstPart = R.Val;
    }
      
    if (Tok.is(tok::semi)) {
      ConsumeToken();
    } else {
      if (!Value.isInvalid) Diag(Tok, diag::err_expected_semi_for);
      SkipUntil(tok::semi);
    }
  }
  
  // Parse the second part of the for specifier.
  if (Tok.is(tok::semi)) {  // for (...;;
    // no second part.
    Value = ExprResult();
  } else {
    Value = ParseExpression();
    if (!Value.isInvalid)
      SecondPart = Value.Val;
  }
  
  if (Tok.is(tok::semi)) {
    ConsumeToken();
  } else {
    if (!Value.isInvalid) Diag(Tok, diag::err_expected_semi_for);
    SkipUntil(tok::semi);
  }
  
  // Parse the third part of the for specifier.
  if (Tok.is(tok::r_paren)) {  // for (...;...;)
    // no third part.
    Value = ExprResult();
  } else {
    Value = ParseExpression();
    if (!Value.isInvalid) {
      // Turn the expression into a stmt.
      StmtResult R = Actions.ActOnExprStmt(Value.Val);
      if (!R.isInvalid)
        ThirdPart = R.Val;
    }
  }
  
  // Match the ')'.
  SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);
  
  // C99 6.8.5p5 - In C99, the body of the if statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  bool NeedsInnerScope = getLang().C99 && Tok.isNot(tok::l_brace);
  if (NeedsInnerScope) EnterScope(Scope::DeclScope);
  
  // Read the body statement.
  StmtResult Body = ParseStatement();
  
  // Pop the body scope if needed.
  if (NeedsInnerScope) ExitScope();

  // Leave the for-scope.
  ExitScope();
    
  if (Body.isInvalid)
    return Body;
  
  return Actions.ActOnForStmt(ForLoc, LParenLoc, FirstPart, SecondPart,
                              ThirdPart, RParenLoc, Body.Val);
}

/// ParseGotoStatement
///       jump-statement:
///         'goto' identifier ';'
/// [GNU]   'goto' '*' expression ';'
///
/// Note: this lets the caller parse the end ';'.
///
Parser::StmtResult Parser::ParseGotoStatement() {
  assert(Tok.is(tok::kw_goto) && "Not a goto stmt!");
  SourceLocation GotoLoc = ConsumeToken();  // eat the 'goto'.
  
  StmtResult Res;
  if (Tok.is(tok::identifier)) {
    Res = Actions.ActOnGotoStmt(GotoLoc, Tok.getLocation(),
                                Tok.getIdentifierInfo());
    ConsumeToken();
  } else if (Tok.is(tok::star) && !getLang().NoExtensions) {
    // GNU indirect goto extension.
    Diag(Tok, diag::ext_gnu_indirect_goto);
    SourceLocation StarLoc = ConsumeToken();
    ExprResult R = ParseExpression();
    if (R.isInvalid) {  // Skip to the semicolon, but don't consume it.
      SkipUntil(tok::semi, false, true);
      return true;
    }
    Res = Actions.ActOnIndirectGotoStmt(GotoLoc, StarLoc, R.Val);
  } else {
    Diag(Tok, diag::err_expected_ident);
    return true;
  }
    
  return Res;
}

/// ParseContinueStatement
///       jump-statement:
///         'continue' ';'
///
/// Note: this lets the caller parse the end ';'.
///
Parser::StmtResult Parser::ParseContinueStatement() {
  SourceLocation ContinueLoc = ConsumeToken();  // eat the 'continue'.
  return Actions.ActOnContinueStmt(ContinueLoc, CurScope);
}

/// ParseBreakStatement
///       jump-statement:
///         'break' ';'
///
/// Note: this lets the caller parse the end ';'.
///
Parser::StmtResult Parser::ParseBreakStatement() {
  SourceLocation BreakLoc = ConsumeToken();  // eat the 'break'.
  return Actions.ActOnBreakStmt(BreakLoc, CurScope);
}

/// ParseReturnStatement
///       jump-statement:
///         'return' expression[opt] ';'
Parser::StmtResult Parser::ParseReturnStatement() {
  assert(Tok.is(tok::kw_return) && "Not a return stmt!");
  SourceLocation ReturnLoc = ConsumeToken();  // eat the 'return'.
  
  ExprResult R(0);
  if (Tok.isNot(tok::semi)) {
    R = ParseExpression();
    if (R.isInvalid) {  // Skip to the semicolon, but don't consume it.
      SkipUntil(tok::semi, false, true);
      return true;
    }
  }
  return Actions.ActOnReturnStmt(ReturnLoc, R.Val);
}

/// ParseAsmStatement - Parse a GNU extended asm statement.
/// [GNU] asm-statement:
///         'asm' type-qualifier[opt] '(' asm-argument ')' ';'
///
/// [GNU] asm-argument:
///         asm-string-literal
///         asm-string-literal ':' asm-operands[opt]
///         asm-string-literal ':' asm-operands[opt] ':' asm-operands[opt]
///         asm-string-literal ':' asm-operands[opt] ':' asm-operands[opt]
///                 ':' asm-clobbers
///
/// [GNU] asm-clobbers:
///         asm-string-literal
///         asm-clobbers ',' asm-string-literal
///
Parser::StmtResult Parser::ParseAsmStatement() {
  assert(Tok.is(tok::kw_asm) && "Not an asm stmt");
  SourceLocation AsmLoc = ConsumeToken();
  
  DeclSpec DS;
  SourceLocation Loc = Tok.getLocation();
  ParseTypeQualifierListOpt(DS);
  
  // GNU asms accept, but warn, about type-qualifiers other than volatile.
  if (DS.getTypeQualifiers() & DeclSpec::TQ_const)
    Diag(Loc, diag::w_asm_qualifier_ignored, "const");
  if (DS.getTypeQualifiers() & DeclSpec::TQ_restrict)
    Diag(Loc, diag::w_asm_qualifier_ignored, "restrict");
  
  // Remember if this was a volatile asm.
  //bool isVolatile = DS.TypeQualifiers & DeclSpec::TQ_volatile;
  
  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after, "asm");
    SkipUntil(tok::r_paren);
    return true;
  }
  Loc = ConsumeParen();
  
  ParseAsmStringLiteral();
  
  // Parse Outputs, if present.
  ParseAsmOperandsOpt();
  
  // Parse Inputs, if present.
  ParseAsmOperandsOpt();
  
  // Parse the clobbers, if present.
  if (Tok.is(tok::colon)) {
    ConsumeToken();
    
    if (isTokenStringLiteral()) {
      // Parse the asm-string list for clobbers.
      while (1) {
        ParseAsmStringLiteral();

        if (Tok.isNot(tok::comma)) break;
        ConsumeToken();
      }
    }
  }
  
  SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren, Loc);
  
  // FIXME: Pass all the details down to the action.
  return Actions.ActOnAsmStmt(AsmLoc, RParenLoc);
}

/// ParseAsmOperands - Parse the asm-operands production as used by
/// asm-statement.  We also parse a leading ':' token.  If the leading colon is
/// not present, we do not parse anything.
///
/// [GNU] asm-operands:
///         asm-operand
///         asm-operands ',' asm-operand
///
/// [GNU] asm-operand:
///         asm-string-literal '(' expression ')'
///         '[' identifier ']' asm-string-literal '(' expression ')'
///
void Parser::ParseAsmOperandsOpt() {
  // Only do anything if this operand is present.
  if (Tok.isNot(tok::colon)) return;
  ConsumeToken();
  
  // 'asm-operands' isn't present?
  if (!isTokenStringLiteral() && Tok.isNot(tok::l_square))
    return;
  
  while (1) {
    // Read the [id] if present.
    if (Tok.is(tok::l_square)) {
      SourceLocation Loc = ConsumeBracket();
      
      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident);
        SkipUntil(tok::r_paren);
        return;
      }
      
      // Eat the identifier, FIXME: capture it.
      ConsumeToken();
      
      MatchRHSPunctuation(tok::r_square, Loc);
    }
    
    ParseAsmStringLiteral();

    if (Tok.isNot(tok::l_paren)) {
      Diag(Tok, diag::err_expected_lparen_after, "asm operand");
      SkipUntil(tok::r_paren);
      return;
    }
    
    // Read the parenthesized expression.
    ExprResult Res = ParseSimpleParenExpression();
    if (Res.isInvalid) {
      SkipUntil(tok::r_paren);
      return;
    }

    // Eat the comma and continue parsing if it exists.
    if (Tok.isNot(tok::comma)) return;
    ConsumeToken();
  }
}
