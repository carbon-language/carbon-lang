//===--- ParseStmt.cpp - Statement and Block Parser -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Statement and Block portions of the Parser
// interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
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
/// [C++]   declaration-statement
/// [OBC]   objc-throw-statement
/// [OBC]   objc-try-catch-statement
/// [OBC]   objc-synchronized-statement
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
  case tok::at: // May be a @try or @throw statement
    {
      AtLoc = ConsumeToken();  // consume @
      return ParseObjCAtStatement(AtLoc);
    }

  case tok::identifier:
    if (NextToken().is(tok::colon)) { // C99 6.8.1: labeled-statement
      // identifier ':' statement
      return ParseLabeledStatement();
    }
    // PASS THROUGH.

  default:
    if ((getLang().CPlusPlus || !OnlyStatement) && isDeclarationSpecifier()) {
      SourceLocation DeclStart = Tok.getLocation();
      DeclTy *Res = ParseDeclaration(Declarator::BlockContext);
      // FIXME: Pass in the right location for the end of the declstmt.
      return Actions.ActOnDeclStmt(Res, DeclStart, DeclStart);
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
    bool msAsm = false;
    Res = ParseAsmStatement(msAsm);
    if (msAsm) return Res;
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

/// ParseLabeledStatement - We have an identifier and a ':' after it.
///
///       labeled-statement:
///         identifier ':' statement
/// [GNU]   identifier ':' attributes[opt] statement
///
Parser::StmtResult Parser::ParseLabeledStatement() {
  assert(Tok.is(tok::identifier) && Tok.getIdentifierInfo() &&
         "Not an identifier!");

  Token IdentTok = Tok;  // Save the whole token.
  ConsumeToken();  // eat the identifier.

  assert(Tok.is(tok::colon) && "Not a label!");
  
  // identifier ':' statement
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
      
      // __extension__ silences extension warnings in the subexpression.
      bool SavedExtWarn = Diags.getWarnOnExtensions();
      Diags.setWarnOnExtensions(false);
      
      // If this is the start of a declaration, parse it as such.
      if (isDeclarationSpecifier()) {
        // FIXME: Save the __extension__ on the decl as a node somehow.
        SourceLocation DeclStart = Tok.getLocation();
        DeclTy *Res = ParseDeclaration(Declarator::BlockContext);
        // FIXME: Pass in the right location for the end of the declstmt.
        R = Actions.ActOnDeclStmt(Res, DeclStart, DeclStart);
        
        Diags.setWarnOnExtensions(SavedExtWarn);
      } else {
        // Otherwise this was a unary __extension__ marker.  Parse the
        // subexpression and add the __extension__ unary op. 
        ExprResult Res = ParseCastExpression(false);
        Diags.setWarnOnExtensions(SavedExtWarn);

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
    return true;
  }
  
  SourceLocation RBraceLoc = ConsumeBrace();
  return Actions.ActOnCompoundStmt(LBraceLoc, RBraceLoc,
                                   &Stmts[0], Stmts.size(), isStmtExpr);
}

/// ParseIfStatement
///       if-statement: [C99 6.8.4.1]
///         'if' '(' expression ')' statement
///         'if' '(' expression ')' statement 'else' statement
/// [C++]   'if' '(' condition ')' statement
/// [C++]   'if' '(' condition ')' statement 'else' statement
///
Parser::StmtResult Parser::ParseIfStatement() {
  assert(Tok.is(tok::kw_if) && "Not an if stmt!");
  SourceLocation IfLoc = ConsumeToken();  // eat the 'if'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after, "if");
    SkipUntil(tok::semi);
    return true;
  }

  bool C99orCXX = getLang().C99 || getLang().CPlusPlus;

  // C99 6.8.4p3 - In C99, the if statement is a block.  This is not
  // the case for C90.
  if (C99orCXX)
    EnterScope(Scope::DeclScope | Scope::ControlScope);

  // Parse the condition.
  ExprResult CondExp;
  if (getLang().CPlusPlus) {
    SourceLocation LParenLoc = ConsumeParen();
    CondExp = ParseCXXCondition();
    MatchRHSPunctuation(tok::r_paren, LParenLoc);
  } else {
    CondExp = ParseSimpleParenExpression();
  }

  if (CondExp.isInvalid) {
    SkipUntil(tok::semi);
    if (C99orCXX)
      ExitScope();
    return true;
  }
  
  // C99 6.8.4p3 - In C99, the body of the if statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  bool NeedsInnerScope = C99orCXX && Tok.isNot(tok::l_brace);
  if (NeedsInnerScope) EnterScope(Scope::DeclScope);

  // Read the 'then' stmt.
  SourceLocation ThenStmtLoc = Tok.getLocation();
  StmtResult ThenStmt = ParseStatement();

  // Pop the 'if' scope if needed.
  if (NeedsInnerScope) ExitScope();
  
  // If it has an else, parse it.
  SourceLocation ElseLoc;
  SourceLocation ElseStmtLoc;
  StmtResult ElseStmt(false);
  
  if (Tok.is(tok::kw_else)) {
    ElseLoc = ConsumeToken();
    
    // C99 6.8.4p3 - In C99, the body of the if statement is a scope, even if
    // there is no compound stmt.  C90 does not have this clause.  We only do
    // this if the body isn't a compound statement to avoid push/pop in common
    // cases.
    NeedsInnerScope = C99orCXX && Tok.isNot(tok::l_brace);
    if (NeedsInnerScope) EnterScope(Scope::DeclScope);
    
    ElseStmtLoc = Tok.getLocation();
    ElseStmt = ParseStatement();

    // Pop the 'else' scope if needed.
    if (NeedsInnerScope) ExitScope();
  }
  
  if (C99orCXX)
    ExitScope();

  // If the then or else stmt is invalid and the other is valid (and present),
  // make turn the invalid one into a null stmt to avoid dropping the other 
  // part.  If both are invalid, return error.
  if ((ThenStmt.isInvalid && ElseStmt.isInvalid) ||
      (ThenStmt.isInvalid && ElseStmt.Val == 0) ||
      (ThenStmt.Val == 0  && ElseStmt.isInvalid)) {
    // Both invalid, or one is invalid and other is non-present: delete cond and
    // return error.
    Actions.DeleteExpr(CondExp.Val);
    return true;
  }
  
  // Now if either are invalid, replace with a ';'.
  if (ThenStmt.isInvalid)
    ThenStmt = Actions.ActOnNullStmt(ThenStmtLoc);
  if (ElseStmt.isInvalid)
    ElseStmt = Actions.ActOnNullStmt(ElseStmtLoc);
  
  return Actions.ActOnIfStmt(IfLoc, CondExp.Val, ThenStmt.Val,
                             ElseLoc, ElseStmt.Val);
}

/// ParseSwitchStatement
///       switch-statement:
///         'switch' '(' expression ')' statement
/// [C++]   'switch' '(' condition ')' statement
Parser::StmtResult Parser::ParseSwitchStatement() {
  assert(Tok.is(tok::kw_switch) && "Not a switch stmt!");
  SourceLocation SwitchLoc = ConsumeToken();  // eat the 'switch'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after, "switch");
    SkipUntil(tok::semi);
    return true;
  }

  bool C99orCXX = getLang().C99 || getLang().CPlusPlus;

  // C99 6.8.4p3 - In C99, the switch statement is a block.  This is
  // not the case for C90.  Start the switch scope.
  if (C99orCXX)
    EnterScope(Scope::BreakScope | Scope::DeclScope | Scope::ControlScope);
  else
    EnterScope(Scope::BreakScope);

  // Parse the condition.
  ExprResult Cond;
  if (getLang().CPlusPlus) {
    SourceLocation LParenLoc = ConsumeParen();
    Cond = ParseCXXCondition();
    MatchRHSPunctuation(tok::r_paren, LParenLoc);
  } else {
    Cond = ParseSimpleParenExpression();
  }
  
  if (Cond.isInvalid) {
    ExitScope();
    return true;
  }
    
  StmtResult Switch = Actions.ActOnStartOfSwitchStmt(Cond.Val);
  
  // C99 6.8.4p3 - In C99, the body of the switch statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  bool NeedsInnerScope = C99orCXX && Tok.isNot(tok::l_brace);
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
/// [C++]   'while' '(' condition ')' statement
Parser::StmtResult Parser::ParseWhileStatement() {
  assert(Tok.is(tok::kw_while) && "Not a while stmt!");
  SourceLocation WhileLoc = Tok.getLocation();
  ConsumeToken();  // eat the 'while'.
  
  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after, "while");
    SkipUntil(tok::semi);
    return true;
  }
  
  bool C99orCXX = getLang().C99 || getLang().CPlusPlus;

  // C99 6.8.5p5 - In C99, the while statement is a block.  This is not
  // the case for C90.  Start the loop scope.
  if (C99orCXX)
    EnterScope(Scope::BreakScope | Scope::ContinueScope |
               Scope::DeclScope  | Scope::ControlScope);
  else
    EnterScope(Scope::BreakScope | Scope::ContinueScope);

  // Parse the condition.
  ExprResult Cond;
  if (getLang().CPlusPlus) {
    SourceLocation LParenLoc = ConsumeParen();
    Cond = ParseCXXCondition();
    MatchRHSPunctuation(tok::r_paren, LParenLoc);
  } else {
    Cond = ParseSimpleParenExpression();
  }
  
  // C99 6.8.5p5 - In C99, the body of the if statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  bool NeedsInnerScope = C99orCXX && Tok.isNot(tok::l_brace);
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
/// [C++]   'for' '(' for-init-statement condition[opt] ';' expression[opt] ')'
/// [C++]       statement
/// [OBJC2] 'for' '(' declaration 'in' expr ')' statement
/// [OBJC2] 'for' '(' expr 'in' expr ')' statement
///
/// [C++] for-init-statement:
/// [C++]   expression-statement
/// [C++]   simple-declaration
///
Parser::StmtResult Parser::ParseForStatement() {
  assert(Tok.is(tok::kw_for) && "Not a for stmt!");
  SourceLocation ForLoc = ConsumeToken();  // eat the 'for'.
  
  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after, "for");
    SkipUntil(tok::semi);
    return true;
  }
  
  bool C99orCXX = getLang().C99 || getLang().CPlusPlus;

  // C99 6.8.5p5 - In C99, the for statement is a block.  This is not
  // the case for C90.  Start the loop scope.
  if (C99orCXX)
    EnterScope(Scope::BreakScope | Scope::ContinueScope |
               Scope::DeclScope  | Scope::ControlScope);
  else
    EnterScope(Scope::BreakScope | Scope::ContinueScope);

  SourceLocation LParenLoc = ConsumeParen();
  ExprResult Value;
  
  StmtTy *FirstPart = 0;
  ExprTy *SecondPart = 0;
  StmtTy *ThirdPart = 0;
  bool ForEach = false;
  
  // Parse the first part of the for specifier.
  if (Tok.is(tok::semi)) {  // for (;
    // no first part, eat the ';'.
    ConsumeToken();
  } else if (isDeclarationSpecifier()) {  // for (int X = 4;
    // Parse declaration, which eats the ';'.
    if (!C99orCXX)   // Use of C99-style for loops in C90 mode?
      Diag(Tok, diag::ext_c99_variable_decl_in_for_loop);
    
    SourceLocation DeclStart = Tok.getLocation();
    DeclTy *aBlockVarDecl = ParseSimpleDeclaration(Declarator::ForContext);
    // FIXME: Pass in the right location for the end of the declstmt.
    StmtResult stmtResult = Actions.ActOnDeclStmt(aBlockVarDecl, DeclStart,
                                                  DeclStart);
    FirstPart = stmtResult.isInvalid ? 0 : stmtResult.Val;
    if ((ForEach = isTokIdentifier_in())) {
      ConsumeToken(); // consume 'in'
      Value = ParseExpression();
      if (!Value.isInvalid)
        SecondPart = Value.Val;
    }
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
    }
    else if ((ForEach = isTokIdentifier_in())) {
      ConsumeToken(); // consume 'in'
      Value = ParseExpression();
      if (!Value.isInvalid)
        SecondPart = Value.Val;
    }
    else {
      if (!Value.isInvalid) Diag(Tok, diag::err_expected_semi_for);
      SkipUntil(tok::semi);
    }
  }
  if (!ForEach) {
    // Parse the second part of the for specifier.
    if (Tok.is(tok::semi)) {  // for (...;;
      // no second part.
      Value = ExprResult();
    } else {
      Value = getLang().CPlusPlus ? ParseCXXCondition()
                                  : ParseExpression();
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
  }
  // Match the ')'.
  SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);
  
  // C99 6.8.5p5 - In C99, the body of the if statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  bool NeedsInnerScope = C99orCXX && Tok.isNot(tok::l_brace);
  if (NeedsInnerScope) EnterScope(Scope::DeclScope);
  
  // Read the body statement.
  StmtResult Body = ParseStatement();
  
  // Pop the body scope if needed.
  if (NeedsInnerScope) ExitScope();

  // Leave the for-scope.
  ExitScope();
    
  if (Body.isInvalid)
    return Body;
  
  if (!ForEach) 
    return Actions.ActOnForStmt(ForLoc, LParenLoc, FirstPart, 
                                SecondPart, ThirdPart, RParenLoc, Body.Val);
  else
    return Actions.ActOnObjCForCollectionStmt(ForLoc, LParenLoc, FirstPart, 
                                              SecondPart, RParenLoc, Body.Val);
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

/// FuzzyParseMicrosoftAsmStatement. When -fms-extensions is enabled, this
/// routine is called to skip/ignore tokens that comprise the MS asm statement.
Parser::StmtResult Parser::FuzzyParseMicrosoftAsmStatement() {
  if (Tok.is(tok::l_brace)) {
    unsigned short savedBraceCount = BraceCount;
    do {
      ConsumeAnyToken();
    } while (BraceCount > savedBraceCount && Tok.isNot(tok::eof));
  } else { 
    // From the MS website: If used without braces, the __asm keyword means
    // that the rest of the line is an assembly-language statement.
    SourceManager &SrcMgr = PP.getSourceManager();
    SourceLocation TokLoc = Tok.getLocation();
    unsigned lineNo = SrcMgr.getLogicalLineNumber(TokLoc);
    do {
      ConsumeAnyToken();
      TokLoc = Tok.getLocation();
    } while ((SrcMgr.getLogicalLineNumber(TokLoc) == lineNo) && 
             Tok.isNot(tok::r_brace) && Tok.isNot(tok::semi) && 
             Tok.isNot(tok::eof));
  }
  return Actions.ActOnNullStmt(Tok.getLocation());
}

/// ParseAsmStatement - Parse a GNU extended asm statement.
///       asm-statement:
///         gnu-asm-statement
///         ms-asm-statement
///
/// [GNU] gnu-asm-statement:
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
/// [MS]  ms-asm-statement:
///         '__asm' assembly-instruction ';'[opt]
///         '__asm' '{' assembly-instruction-list '}' ';'[opt]
///
/// [MS]  assembly-instruction-list:
///         assembly-instruction ';'[opt]
///         assembly-instruction-list ';' assembly-instruction ';'[opt]
///
Parser::StmtResult Parser::ParseAsmStatement(bool &msAsm) {
  assert(Tok.is(tok::kw_asm) && "Not an asm stmt");
  SourceLocation AsmLoc = ConsumeToken();
  
  if (getLang().Microsoft && Tok.isNot(tok::l_paren) && !isTypeQualifier()) {
    msAsm = true;
    return FuzzyParseMicrosoftAsmStatement();
  }
  DeclSpec DS;
  SourceLocation Loc = Tok.getLocation();
  ParseTypeQualifierListOpt(DS);
  
  // GNU asms accept, but warn, about type-qualifiers other than volatile.
  if (DS.getTypeQualifiers() & DeclSpec::TQ_const)
    Diag(Loc, diag::w_asm_qualifier_ignored, "const");
  if (DS.getTypeQualifiers() & DeclSpec::TQ_restrict)
    Diag(Loc, diag::w_asm_qualifier_ignored, "restrict");
  
  // Remember if this was a volatile asm.
  bool isVolatile = DS.getTypeQualifiers() & DeclSpec::TQ_volatile;
  bool isSimple = false;
  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after, "asm");
    SkipUntil(tok::r_paren);
    return true;
  }
  Loc = ConsumeParen();
  
  ExprResult AsmString = ParseAsmStringLiteral();
  if (AsmString.isInvalid)
    return true;
  
  llvm::SmallVector<std::string, 4> Names;
  llvm::SmallVector<ExprTy*, 4> Constraints;
  llvm::SmallVector<ExprTy*, 4> Exprs;
  llvm::SmallVector<ExprTy*, 4> Clobbers;

  unsigned NumInputs = 0, NumOutputs = 0;
  
  SourceLocation RParenLoc;
  if (Tok.is(tok::r_paren)) {
    // We have a simple asm expression
    isSimple = true;
    
    RParenLoc = ConsumeParen();
  } else {
    // Parse Outputs, if present. 
    if (ParseAsmOperandsOpt(Names, Constraints, Exprs))
        return true;
  
    NumOutputs = Names.size();
  
    // Parse Inputs, if present.
    if (ParseAsmOperandsOpt(Names, Constraints, Exprs))
        return true;
      
    assert(Names.size() == Constraints.size() &&
           Constraints.size() == Exprs.size() 
           && "Input operand size mismatch!");

    NumInputs = Names.size() - NumOutputs;
  
    // Parse the clobbers, if present.
    if (Tok.is(tok::colon)) {
      ConsumeToken();
    
      // Parse the asm-string list for clobbers.
      while (1) {
        ExprResult Clobber = ParseAsmStringLiteral();

        if (Clobber.isInvalid)
          break;
      
        Clobbers.push_back(Clobber.Val);
      
        if (Tok.isNot(tok::comma)) break;
        ConsumeToken();
      }
    }
  
    RParenLoc = MatchRHSPunctuation(tok::r_paren, Loc);
  }
  
  return Actions.ActOnAsmStmt(AsmLoc, isSimple, isVolatile,
                              NumOutputs, NumInputs,
                              &Names[0], &Constraints[0], &Exprs[0],
                              AsmString.Val,
                              Clobbers.size(), &Clobbers[0],
                              RParenLoc);
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
bool Parser::ParseAsmOperandsOpt(llvm::SmallVectorImpl<std::string> &Names,
                                 llvm::SmallVectorImpl<ExprTy*> &Constraints,
                                 llvm::SmallVectorImpl<ExprTy*> &Exprs) {
  // Only do anything if this operand is present.
  if (Tok.isNot(tok::colon)) return false;
  ConsumeToken();
  
  // 'asm-operands' isn't present?
  if (!isTokenStringLiteral() && Tok.isNot(tok::l_square))
    return false;
  
  while (1) {   
    // Read the [id] if present.
    if (Tok.is(tok::l_square)) {
      SourceLocation Loc = ConsumeBracket();
      
      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident);
        SkipUntil(tok::r_paren);
        return true;
      }
      
      IdentifierInfo *II = Tok.getIdentifierInfo();
      ConsumeToken();

      Names.push_back(std::string(II->getName(), II->getLength()));
      MatchRHSPunctuation(tok::r_square, Loc);
    } else
      Names.push_back(std::string());
    
    ExprResult Constraint = ParseAsmStringLiteral();
    if (Constraint.isInvalid) {
        SkipUntil(tok::r_paren);
        return true;
    }
    Constraints.push_back(Constraint.Val);

    if (Tok.isNot(tok::l_paren)) {
      Diag(Tok, diag::err_expected_lparen_after, "asm operand");
      SkipUntil(tok::r_paren);
      return true;
    }
    
    // Read the parenthesized expression.
    ExprResult Res = ParseSimpleParenExpression();
    if (Res.isInvalid) {
      SkipUntil(tok::r_paren);
      return true;
    }
    Exprs.push_back(Res.Val);
    // Eat the comma and continue parsing if it exists.
    if (Tok.isNot(tok::comma)) return false;
    ConsumeToken();
  }

  return true;
}

Parser::DeclTy *Parser::ParseFunctionStatementBody(DeclTy *Decl, 
                                           SourceLocation L, SourceLocation R) {
  // Do not enter a scope for the brace, as the arguments are in the same scope
  // (the function body) as the body itself.  Instead, just read the statement
  // list and put it into a CompoundStmt for safe keeping.
  StmtResult FnBody = ParseCompoundStatementBody();
  
  // If the function body could not be parsed, make a bogus compoundstmt.
  if (FnBody.isInvalid)
    FnBody = Actions.ActOnCompoundStmt(L, R, 0, 0, false);
  
  // Leave the function body scope.
  ExitScope();
  
  return Actions.ActOnFinishFunctionBody(Decl, FnBody.Val);
}
