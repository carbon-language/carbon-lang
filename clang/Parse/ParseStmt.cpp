//===--- Statement.cpp - Statement and Block Parser -----------------------===//
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
#include "clang/Parse/Declarations.h"
using namespace llvm;
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
/// [GNU]   asm-statement                [TODO]
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
/// [GNU]   'goto' '*' expression ';'     [TODO]
///
/// [OBC] objc-throw-statement:           [TODO]
/// [OBC]   '@' 'throw' expression ';'    [TODO]
/// [OBC]   '@' 'throw' ';'               [TODO]
/// 
void Parser::ParseStatementOrDeclaration(bool OnlyStatement) {
  switch (Tok.getKind()) {
  default:
    Diag(Tok, diag::err_expected_statement_declaration);
    SkipUntil(tok::semi);
    break;
    
    
  case tok::l_brace:                // C99 6.8.2: compound-statement
    ParseCompoundStatement();
    break;
  case tok::semi:                   // C99 6.8.3: expression[opt] ';'
    ConsumeToken();
    break;
  case tok::kw_if:                  // C99 6.8.4.1: if-statement
    ParseIfStatement();
    break;
  case tok::kw_switch:              // C99 6.8.4.2: switch-statement
    ParseSwitchStatement();
    break;
  case tok::kw_while:               // C99 6.8.5.1: while-statement
    ParseWhileStatement();
    break;
  case tok::kw_do:                  // C99 6.8.5.2: do-statement
    ParseDoStatement();
    break;
  case tok::kw_for:                 // C99 6.8.5.3: for-statement
    ParseForStatement();
    break;
    // TODO: Handle OnlyStatement..
  }
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
void Parser::ParseCompoundStatement() {
  assert(Tok.getKind() == tok::l_brace && "Not a compount stmt!");
  ConsumeBrace();  // eat the '{'.
  
  while (Tok.getKind() != tok::r_brace && Tok.getKind() != tok::eof)
    ParseStatementOrDeclaration(false);
  
  // We broke out of the while loop because we found a '}' or EOF.
  if (Tok.getKind() == tok::r_brace)
    ConsumeBrace();
  else
    Diag(Tok, diag::err_expected_rbrace);
}

/// ParseIfStatement
///       if-statement: [C99 6.8.4.1]
///         'if' '(' expression ')' statement
///         'if' '(' expression ')' statement 'else' statement
void Parser::ParseIfStatement() {
  assert(Tok.getKind() == tok::kw_if && "Not an if stmt!");
  ConsumeToken();  // eat the 'if'.

  if (Tok.getKind() != tok::l_paren) {
    Diag(Tok, diag::err_expected_lparen_after, "if");
    SkipUntil(tok::semi);
    return;
  }
  
  // Parse the condition.
  ParseParenExpression();
  
  // Read the if condition.
  ParseStatement();
  
  // If it has an else, parse it.
  if (Tok.getKind() == tok::kw_else) {
    ConsumeToken();
    ParseStatement();
  }
}

/// ParseSwitchStatement
///       switch-statement:
///         'switch' '(' expression ')' statement
void Parser::ParseSwitchStatement() {
  assert(Tok.getKind() == tok::kw_switch && "Not a switch stmt!");
  ConsumeToken();  // eat the 'switch'.

  if (Tok.getKind() != tok::l_paren) {
    Diag(Tok, diag::err_expected_lparen_after, "switch");
    SkipUntil(tok::semi);
    return;
  }
  
  // Parse the condition.
  ParseParenExpression();
  
  // Read the body statement.
  ParseStatement();
}

/// ParseWhileStatement
///       while-statement: [C99 6.8.5.1]
///         'while' '(' expression ')' statement
void Parser::ParseWhileStatement() {
  assert(Tok.getKind() == tok::kw_while && "Not a while stmt!");
  ConsumeToken();  // eat the 'while'.
  
  if (Tok.getKind() != tok::l_paren) {
    Diag(Tok, diag::err_expected_lparen_after, "while");
    SkipUntil(tok::semi);
    return;
  }
  
  // Parse the condition.
  ParseParenExpression();
  
  // Read the body statement.
  ParseStatement();
}

/// ParseDoStatement
///       do-statement: [C99 6.8.5.2]
///         'do' statement 'while' '(' expression ')' ';'
void Parser::ParseDoStatement() {
  assert(Tok.getKind() == tok::kw_do && "Not a do stmt!");
  SourceLocation DoLoc = Tok.getLocation();
  ConsumeToken();  // eat the 'do'.
  
  // Read the body statement.
  ParseStatement();

  if (Tok.getKind() != tok::kw_while) {
    Diag(Tok, diag::err_expected_while);
    Diag(DoLoc, diag::err_matching);
    SkipUntil(tok::semi);
    return;
  }
  ConsumeToken();
  
  if (Tok.getKind() != tok::l_paren) {
    Diag(Tok, diag::err_expected_lparen_after, "do/while");
    SkipUntil(tok::semi);
    return;
  }
  
  // Parse the condition.
  ParseParenExpression();
  
  if (Tok.getKind() != tok::semi) {
    Diag(Tok, diag::err_expected_semi_do_while);
    SkipUntil(tok::semi);
    return;
  }
}

/// ParseForStatement
///       for-statement: [C99 6.8.5.3]
///         'for' '(' expr[opt] ';' expr[opt] ';' expr[opt] ')' statement
///         'for' '(' declaration expr[opt] ';' expr[opt] ')' statement
void Parser::ParseForStatement() {
  assert(Tok.getKind() == tok::kw_for && "Not a for stmt!");
  SourceLocation ForLoc = Tok.getLocation();
  ConsumeToken();  // eat the 'for'.
  
  if (Tok.getKind() != tok::l_paren) {
    Diag(Tok, diag::err_expected_lparen_after, "for");
    SkipUntil(tok::semi);
    return;
  }

  SourceLocation LParenLoc = Tok.getLocation();
  ConsumeParen();
  
  // Parse the first part of the for specifier.
  if (Tok.getKind() == tok::semi) {  // for (;
    // no first part, eat the ';'.
    ConsumeToken();
  } else if (isDeclarationSpecifier()) {  // for (int X = 4;
    // Parse declaration, which eats the ';'.
    if (!getLang().C99)   // Use of C99-style for loops in C90 mode?
      Diag(Tok, diag::ext_c99_variable_decl_in_for_loop);
    ParseDeclaration(Declarator::ForContext);
  } else {
    ParseExpression();
  
    if (Tok.getKind() == tok::semi) {
      ConsumeToken();
    } else {
      Diag(Tok, diag::err_expected_semi_for);
      Diag(ForLoc, diag::err_matching);
      SkipUntil(tok::semi);
    }
  }
  
  // Parse the second part of the for specifier.
  if (Tok.getKind() == tok::semi) {  // for (...;;
    // no second part.
  } else {
    ParseExpression();
  }
  
  if (Tok.getKind() == tok::semi) {
    ConsumeToken();
  } else {
    Diag(Tok, diag::err_expected_semi_for);
    Diag(ForLoc, diag::err_matching);
    SkipUntil(tok::semi);
  }
  
  // Parse the third part of the for specifier.
  if (Tok.getKind() == tok::r_paren) {  // for (...;...;)
    // no third part.
  } else {
    ParseExpression();
  }
  
  if (Tok.getKind() == tok::r_paren) {
    ConsumeParen();
  } else {
    Diag(Tok, diag::err_expected_rparen);
    Diag(LParenLoc, diag::err_matching);
    SkipUntil(tok::r_paren);
    return;
  }
  
  // Read the body statement.
  ParseStatement();
}

