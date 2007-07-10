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
/// [OBC]   objc-throw-statement         [TODO]
/// [OBC]   objc-try-catch-statement     [TODO]
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
/// [OBC] objc-throw-statement:           [TODO]
/// [OBC]   '@' 'throw' expression ';'    [TODO]
/// [OBC]   '@' 'throw' ';'               [TODO]
/// 
Parser::StmtResult Parser::ParseStatementOrDeclaration(bool OnlyStatement) {
  const char *SemiError = 0;
  Parser::StmtResult Res;
  
  // Cases in this switch statement should fall through if the parser expects
  // the token to end in a semicolon (in which case SemiError should be set),
  // or they directly 'return;' if not.
  switch (Tok.getKind()) {
  case tok::identifier:             // C99 6.8.1: labeled-statement
    // identifier ':' statement
    // declaration                  (if !OnlyStatement)
    // expression[opt] ';'
    return ParseIdentifierStatement(OnlyStatement);

  default:
    if (!OnlyStatement && isDeclarationSpecifier()) {
      // TODO: warn/disable if declaration is in the middle of a block and !C99.
      return Actions.ParseDeclStmt(ParseDeclaration(Declarator::BlockContext));
    } else if (Tok.getKind() == tok::r_brace) {
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
      return Actions.ParseExprStmt(Res.Val);
    }
    
  case tok::kw_case:                // C99 6.8.1: labeled-statement
    return ParseCaseStatement();
  case tok::kw_default:             // C99 6.8.1: labeled-statement
    return ParseDefaultStatement();
    
  case tok::l_brace:                // C99 6.8.2: compound-statement
    return ParseCompoundStatement();
  case tok::semi:                   // C99 6.8.3p3: expression[opt] ';'
    return Actions.ParseNullStmt(ConsumeToken());
    
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
  if (Tok.getKind() == tok::semi) {
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
  assert(Tok.getKind() == tok::identifier && Tok.getIdentifierInfo() &&
         "Not an identifier!");

  LexerToken IdentTok = Tok;  // Save the whole token.
  ConsumeToken();  // eat the identifier.
  
  // identifier ':' statement
  if (Tok.getKind() == tok::colon) {
    SourceLocation ColonLoc = ConsumeToken();

    // Read label attributes, if present.
    DeclTy *AttrList = 0;
    if (Tok.getKind() == tok::kw___attribute)
      // TODO: save these somewhere.
      AttrList = ParseAttributes();

    StmtResult SubStmt = ParseStatement();
    
    // Broken substmt shouldn't prevent the label from being added to the AST.
    if (SubStmt.isInvalid)
      SubStmt = Actions.ParseNullStmt(ColonLoc);
    
    return Actions.ParseLabelStmt(IdentTok.getLocation(), 
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
    
    // ParseDeclarationSpecifiers will continue from there.
    ParseDeclarationSpecifiers(DS);

    // C99 6.7.2.3p6: Handle "struct-or-union identifier;", "enum { X };"
    // declaration-specifiers init-declarator-list[opt] ';'
    if (Tok.getKind() == tok::semi) {
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
    return Decl ? Actions.ParseDeclStmt(Decl) : 0;
  }
  
  // Otherwise, this is an expression.  Seed it with II and parse it.
  ExprResult Res = ParseExpressionWithLeadingIdentifier(IdentTok);
  if (Res.isInvalid) {
    SkipUntil(tok::semi);
    return true;
  } else if (Tok.getKind() != tok::semi) {
    Diag(Tok, diag::err_expected_semi_after, "expression");
    SkipUntil(tok::semi);
    return true;
  } else {
    ConsumeToken();
    // Convert expr to a stmt.
    return Actions.ParseExprStmt(Res.Val);
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
  assert(Tok.getKind() == tok::kw_case && "Not a case stmt!");
  SourceLocation CaseLoc = ConsumeToken();  // eat the 'case'.

  ExprResult LHS = ParseConstantExpression();
  if (LHS.isInvalid) {
    SkipUntil(tok::colon);
    return true;
  }
  
  // GNU case range extension.
  SourceLocation DotDotDotLoc;
  ExprTy *RHSVal = 0;
  if (Tok.getKind() == tok::ellipsis) {
    Diag(Tok, diag::ext_gnu_case_range);
    DotDotDotLoc = ConsumeToken();
    
    ExprResult RHS = ParseConstantExpression();
    if (RHS.isInvalid) {
      SkipUntil(tok::colon);
      return true;
    }
    RHSVal = RHS.Val;
  }
  
  if (Tok.getKind() != tok::colon) {
    Diag(Tok, diag::err_expected_colon_after, "'case'");
    SkipUntil(tok::colon);
    return true;
  }
  
  SourceLocation ColonLoc = ConsumeToken();
  
  // Diagnose the common error "switch (X) { case 4: }", which is not valid.
  if (Tok.getKind() == tok::r_brace) {
    Diag(Tok, diag::err_label_end_of_compound_statement);
    return true;
  }
  
  StmtResult SubStmt = ParseStatement();

  // Broken substmt shouldn't prevent the case from being added to the AST.
  if (SubStmt.isInvalid)
    SubStmt = Actions.ParseNullStmt(ColonLoc);
  
  // TODO: look up enclosing switch stmt.
  return Actions.ParseCaseStmt(CaseLoc, LHS.Val, DotDotDotLoc, RHSVal, ColonLoc,
                               SubStmt.Val);
}

/// ParseDefaultStatement
///       labeled-statement:
///         'default' ':' statement
/// Note that this does not parse the 'statement' at the end.
///
Parser::StmtResult Parser::ParseDefaultStatement() {
  assert(Tok.getKind() == tok::kw_default && "Not a default stmt!");
  SourceLocation DefaultLoc = ConsumeToken();  // eat the 'default'.

  if (Tok.getKind() != tok::colon) {
    Diag(Tok, diag::err_expected_colon_after, "'default'");
    SkipUntil(tok::colon);
    return true;
  }
  
  SourceLocation ColonLoc = ConsumeToken();
  
  // Diagnose the common error "switch (X) {... default: }", which is not valid.
  if (Tok.getKind() == tok::r_brace) {
    Diag(Tok, diag::err_label_end_of_compound_statement);
    return true;
  }

  StmtResult SubStmt = ParseStatement();
  if (SubStmt.isInvalid)
    return true;
  
  // TODO: look up enclosing switch stmt.
  return Actions.ParseDefaultStmt(DefaultLoc, ColonLoc, SubStmt.Val);
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
/// [GNU]   '__extension__' declaration [TODO]
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
Parser::StmtResult Parser::ParseCompoundStatement() {
  assert(Tok.getKind() == tok::l_brace && "Not a compount stmt!");
  
  // Enter a scope to hold everything within the compound stmt.
  EnterScope(0);

  // Parse the statements in the body.
  StmtResult Body = ParseCompoundStatementBody();

  ExitScope();
  return Body;
}


/// ParseCompoundStatementBody - Parse a sequence of statements and invoke the
/// ParseCompoundStmt action.  This expects the '{' to be the current token, and
/// consume the '}' at the end of the block.  It does not manipulate the scope
/// stack.
Parser::StmtResult Parser::ParseCompoundStatementBody() {
  SourceLocation LBraceLoc = ConsumeBrace();  // eat the '{'.

  // TODO: "__label__ X, Y, Z;" is the GNU "Local Label" extension.  These are
  // only allowed at the start of a compound stmt.
  
  llvm::SmallVector<StmtTy*, 32> Stmts;
  while (Tok.getKind() != tok::r_brace && Tok.getKind() != tok::eof) {
    StmtResult R = ParseStatementOrDeclaration(false);
    if (!R.isInvalid && R.Val)
      Stmts.push_back(R.Val);
  }
  
  // We broke out of the while loop because we found a '}' or EOF.
  if (Tok.getKind() != tok::r_brace) {
    Diag(Tok, diag::err_expected_rbrace);
    return 0;
  }
  
  SourceLocation RBraceLoc = ConsumeBrace();
  return Actions.ParseCompoundStmt(LBraceLoc, RBraceLoc,
                                   &Stmts[0], Stmts.size());
}

/// ParseIfStatement
///       if-statement: [C99 6.8.4.1]
///         'if' '(' expression ')' statement
///         'if' '(' expression ')' statement 'else' statement
///
Parser::StmtResult Parser::ParseIfStatement() {
  assert(Tok.getKind() == tok::kw_if && "Not an if stmt!");
  SourceLocation IfLoc = ConsumeToken();  // eat the 'if'.

  if (Tok.getKind() != tok::l_paren) {
    Diag(Tok, diag::err_expected_lparen_after, "if");
    SkipUntil(tok::semi);
    return true;
  }
  
  // Parse the condition.
  ExprResult CondExp = ParseSimpleParenExpression();
  if (CondExp.isInvalid) {
    SkipUntil(tok::semi);
    return true;
  }
  
  // Read the if condition.
  StmtResult CondStmt = ParseStatement();

  // Broken substmt shouldn't prevent the label from being added to the AST.
  if (CondStmt.isInvalid)
    CondStmt = Actions.ParseNullStmt(Tok.getLocation());
  
  
  // If it has an else, parse it.
  SourceLocation ElseLoc;
  StmtResult ElseStmt(false);
  if (Tok.getKind() == tok::kw_else) {
    ElseLoc = ConsumeToken();
    ElseStmt = ParseStatement();
    
    if (ElseStmt.isInvalid)
      ElseStmt = Actions.ParseNullStmt(ElseLoc);
  }
  
  return Actions.ParseIfStmt(IfLoc, CondExp.Val, CondStmt.Val,
                             ElseLoc, ElseStmt.Val);
}

/// ParseSwitchStatement
///       switch-statement:
///         'switch' '(' expression ')' statement
Parser::StmtResult Parser::ParseSwitchStatement() {
  assert(Tok.getKind() == tok::kw_switch && "Not a switch stmt!");
  SourceLocation SwitchLoc = ConsumeToken();  // eat the 'switch'.

  if (Tok.getKind() != tok::l_paren) {
    Diag(Tok, diag::err_expected_lparen_after, "switch");
    SkipUntil(tok::semi);
    return true;
  }
  
  // Start the switch scope.
  EnterScope(Scope::BreakScope);

  // Parse the condition.
  ExprResult Cond = ParseSimpleParenExpression();
  
  // Read the body statement.
  StmtResult Body = ParseStatement();

  ExitScope();
  
  if (Cond.isInvalid || Body.isInvalid) return true;
  
  return Actions.ParseSwitchStmt(SwitchLoc, Cond.Val, Body.Val);
}

/// ParseWhileStatement
///       while-statement: [C99 6.8.5.1]
///         'while' '(' expression ')' statement
Parser::StmtResult Parser::ParseWhileStatement() {
  assert(Tok.getKind() == tok::kw_while && "Not a while stmt!");
  SourceLocation WhileLoc = Tok.getLocation();
  ConsumeToken();  // eat the 'while'.
  
  if (Tok.getKind() != tok::l_paren) {
    Diag(Tok, diag::err_expected_lparen_after, "while");
    SkipUntil(tok::semi);
    return true;
  }
  
  // Start the loop scope.
  EnterScope(Scope::BreakScope | Scope::ContinueScope);

  // Parse the condition.
  ExprResult Cond = ParseSimpleParenExpression();
  
  // Read the body statement.
  StmtResult Body = ParseStatement();

  ExitScope();
  
  if (Cond.isInvalid || Body.isInvalid) return true;
  
  return Actions.ParseWhileStmt(WhileLoc, Cond.Val, Body.Val);
}

/// ParseDoStatement
///       do-statement: [C99 6.8.5.2]
///         'do' statement 'while' '(' expression ')' ';'
/// Note: this lets the caller parse the end ';'.
Parser::StmtResult Parser::ParseDoStatement() {
  assert(Tok.getKind() == tok::kw_do && "Not a do stmt!");
  SourceLocation DoLoc = ConsumeToken();  // eat the 'do'.
  
  // Start the loop scope.
  EnterScope(Scope::BreakScope | Scope::ContinueScope);

  // Read the body statement.
  StmtResult Body = ParseStatement();

  if (Tok.getKind() != tok::kw_while) {
    ExitScope();
    Diag(Tok, diag::err_expected_while);
    Diag(DoLoc, diag::err_matching, "do");
    SkipUntil(tok::semi);
    return true;
  }
  SourceLocation WhileLoc = ConsumeToken();
  
  if (Tok.getKind() != tok::l_paren) {
    ExitScope();
    Diag(Tok, diag::err_expected_lparen_after, "do/while");
    SkipUntil(tok::semi);
    return true;
  }
  
  // Parse the condition.
  ExprResult Cond = ParseSimpleParenExpression();
  
  ExitScope();
  
  if (Cond.isInvalid || Body.isInvalid) return true;
  
  return Actions.ParseDoStmt(DoLoc, Body.Val, WhileLoc, Cond.Val);
}

/// ParseForStatement
///       for-statement: [C99 6.8.5.3]
///         'for' '(' expr[opt] ';' expr[opt] ';' expr[opt] ')' statement
///         'for' '(' declaration expr[opt] ';' expr[opt] ')' statement
Parser::StmtResult Parser::ParseForStatement() {
  assert(Tok.getKind() == tok::kw_for && "Not a for stmt!");
  SourceLocation ForLoc = ConsumeToken();  // eat the 'for'.
  
  if (Tok.getKind() != tok::l_paren) {
    Diag(Tok, diag::err_expected_lparen_after, "for");
    SkipUntil(tok::semi);
    return true;
  }
  
  EnterScope(Scope::BreakScope | Scope::ContinueScope);

  SourceLocation LParenLoc = ConsumeParen();
  ExprResult Value;
  
  StmtTy *FirstPart = 0;
  ExprTy *SecondPart = 0;
  StmtTy *ThirdPart = 0;
  
  // Parse the first part of the for specifier.
  if (Tok.getKind() == tok::semi) {  // for (;
    // no first part, eat the ';'.
    ConsumeToken();
  } else if (isDeclarationSpecifier()) {  // for (int X = 4;
    // Parse declaration, which eats the ';'.
    if (!getLang().C99)   // Use of C99-style for loops in C90 mode?
      Diag(Tok, diag::ext_c99_variable_decl_in_for_loop);
    DeclTy *aBlockVarDecl = ParseDeclaration(Declarator::ForContext);
    StmtResult stmtResult = Actions.ParseDeclStmt(aBlockVarDecl);
    FirstPart = stmtResult.isInvalid ? 0 : stmtResult.Val;
  } else {
    Value = ParseExpression();

    // Turn the expression into a stmt.
    if (!Value.isInvalid) {
      StmtResult R = Actions.ParseExprStmt(Value.Val);
      if (!R.isInvalid)
        FirstPart = R.Val;
    }
      
    if (Tok.getKind() == tok::semi) {
      ConsumeToken();
    } else {
      if (!Value.isInvalid) Diag(Tok, diag::err_expected_semi_for);
      SkipUntil(tok::semi);
    }
  }
  
  // Parse the second part of the for specifier.
  if (Tok.getKind() == tok::semi) {  // for (...;;
    // no second part.
    Value = ExprResult();
  } else {
    Value = ParseExpression();
    if (!Value.isInvalid)
      SecondPart = Value.Val;
  }
  
  if (Tok.getKind() == tok::semi) {
    ConsumeToken();
  } else {
    if (!Value.isInvalid) Diag(Tok, diag::err_expected_semi_for);
    SkipUntil(tok::semi);
  }
  
  // Parse the third part of the for specifier.
  if (Tok.getKind() == tok::r_paren) {  // for (...;...;)
    // no third part.
    Value = ExprResult();
  } else {
    Value = ParseExpression();
    if (!Value.isInvalid) {
      // Turn the expression into a stmt.
      StmtResult R = Actions.ParseExprStmt(Value.Val);
      if (!R.isInvalid)
        ThirdPart = R.Val;
    }
  }
  
  // Match the ')'.
  SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);
  
  // Read the body statement.
  StmtResult Body = ParseStatement();
  
  // Leave the for-scope.
  ExitScope();
    
  if (Body.isInvalid)
    return Body;
  
  return Actions.ParseForStmt(ForLoc, LParenLoc, FirstPart, SecondPart,
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
  assert(Tok.getKind() == tok::kw_goto && "Not a goto stmt!");
  SourceLocation GotoLoc = ConsumeToken();  // eat the 'goto'.
  
  StmtResult Res;
  if (Tok.getKind() == tok::identifier) {
    Res = Actions.ParseGotoStmt(GotoLoc, Tok.getLocation(),
                                Tok.getIdentifierInfo());
    ConsumeToken();
  } else if (Tok.getKind() == tok::star && !getLang().NoExtensions) {
    // GNU indirect goto extension.
    Diag(Tok, diag::ext_gnu_indirect_goto);
    SourceLocation StarLoc = ConsumeToken();
    ExprResult R = ParseExpression();
    if (R.isInvalid) {  // Skip to the semicolon, but don't consume it.
      SkipUntil(tok::semi, false, true);
      return true;
    }
    Res = Actions.ParseIndirectGotoStmt(GotoLoc, StarLoc, R.Val);
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
  return Actions.ParseContinueStmt(ContinueLoc, CurScope);
}

/// ParseBreakStatement
///       jump-statement:
///         'break' ';'
///
/// Note: this lets the caller parse the end ';'.
///
Parser::StmtResult Parser::ParseBreakStatement() {
  SourceLocation BreakLoc = ConsumeToken();  // eat the 'break'.
  return Actions.ParseBreakStmt(BreakLoc, CurScope);
}

/// ParseReturnStatement
///       jump-statement:
///         'return' expression[opt] ';'
Parser::StmtResult Parser::ParseReturnStatement() {
  assert(Tok.getKind() == tok::kw_return && "Not a return stmt!");
  SourceLocation ReturnLoc = ConsumeToken();  // eat the 'return'.
  
  ExprResult R(0);
  if (Tok.getKind() != tok::semi) {
    R = ParseExpression();
    if (R.isInvalid) {  // Skip to the semicolon, but don't consume it.
      SkipUntil(tok::semi, false, true);
      return true;
    }
  }
  return Actions.ParseReturnStmt(ReturnLoc, R.Val);
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
  assert(Tok.getKind() == tok::kw_asm && "Not an asm stmt");
  ConsumeToken();
  
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
  
  if (Tok.getKind() != tok::l_paren) {
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
  if (Tok.getKind() == tok::colon) {
    ConsumeToken();
    
    if (isTokenStringLiteral()) {
      // Parse the asm-string list for clobbers.
      while (1) {
        ParseAsmStringLiteral();

        if (Tok.getKind() != tok::comma) break;
        ConsumeToken();
      }
    }
  }
  
  MatchRHSPunctuation(tok::r_paren, Loc);
  
  // FIXME: Implement action for asm parsing.
  return false;
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
  if (Tok.getKind() != tok::colon) return;
  ConsumeToken();
  
  // 'asm-operands' isn't present?
  if (!isTokenStringLiteral() && Tok.getKind() != tok::l_square)
    return;
  
  while (1) {
    // Read the [id] if present.
    if (Tok.getKind() == tok::l_square) {
      SourceLocation Loc = ConsumeBracket();
      
      if (Tok.getKind() != tok::identifier) {
        Diag(Tok, diag::err_expected_ident);
        SkipUntil(tok::r_paren);
        return;
      }
      MatchRHSPunctuation(tok::r_square, Loc);
    }
    
    ParseAsmStringLiteral();

    if (Tok.getKind() != tok::l_paren) {
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
    if (Tok.getKind() != tok::comma) return;
    ConsumeToken();
  }
}
