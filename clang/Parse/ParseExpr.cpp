//===--- Expression.cpp - Expression Parsing ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Expression parsing implementation.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Basic/Diagnostic.h"
using namespace llvm;
using namespace clang;

// C99 6.7.8
void Parser::ParseInitializer() {
  // FIXME: STUB.
  ParseAssignmentExpression();
}



Parser::ExprTy Parser::ParseExpression() {
  ParsePostfixExpression();
  return 0;
}

// Expr that doesn't include commas.
void Parser::ParseAssignmentExpression() {
  ParseExpression();
}

/// ParsePostfixExpression
///       postfix-expression: [C99 6.5.2]
///         primary-expression
///         postfix-expression '[' expression ']'
///         postfix-expression '(' argument-expression-list[opt] ')'
///         postfix-expression '.' identifier
///         postfix-expression '->' identifier
///         postfix-expression '++'
///         postfix-expression '--'
///         '(' type-name ')' '{' initializer-list '}'
///         '(' type-name ')' '{' initializer-list ',' '}'
///
///       argument-expression-list: [C99 6.5.2]
///         argument-expression
///         argument-expression-list ',' argument-expression
///
///       primary-expression: [C99 6.5.1]
///         identifier
///         constant
///         string-literal
///         '(' expression ')'
///         '__func__'        [C99 6.4.2.2]
/// [GNU]   '__FUNCTION__'
/// [GNU]   '__PRETTY_FUNCTION__'
/// [GNU]   '(' compound-statement ')'
/// [GNU]   '__builtin_va_arg' '(' assignment-expression ',' type-name ')'
/// [GNU]   '__builtin_offsetof' '(' type-name ',' offsetof-member-designator')'
/// [GNU]   '__builtin_choose_expr' '(' assign-expr ',' assign-expr ','
///                                     assign-expr ')'
/// [GNU]   '__builtin_types_compatible_p' '(' type-name ',' type-name ')'
/// [OBC]   '[' objc-receiver objc-message-args ']'    [TODO]
/// [OBC]   '@selector' '(' objc-selector-arg ')'      [TODO]
/// [OBC]   '@protocol' '(' identifier ')'             [TODO]
/// [OBC]   '@encode' '(' type-name ')'                [TODO]
/// [OBC]   objc-string-literal                        [TODO]
///
///       constant: [C99 6.4.4]
///         integer-constant
///         floating-constant
///         enumeration-constant -> identifier
///         character-constant
/// 
/// [GNU] offsetof-member-designator:
/// [GNU]   identifier
/// [GNU]   offsetof-member-designator '.' identifier
/// [GNU]   offsetof-member-designator '[' expression ']'
///
void Parser::ParsePostfixExpression() {
  // First step, parse the primary expression.
  switch (Tok.getKind()) {
    // primary-expression
  case tok::identifier:        // primary-expression: identifier
                               // constant: enumeration-constant
  case tok::numeric_constant:  // constant: integer-constant
                               // constant: floating-constant
  case tok::char_constant:     // constant: character-constant
  case tok::kw___func__:       // primary-expression: __func__ [C99 6.4.2.2]
  case tok::kw___FUNCTION__:   // primary-expression: __FUNCTION__ [GNU]
  case tok::kw___PRETTY_FUNCTION__:  // primary-expression: __P..Y_F..N__ [GNU]
    ConsumeToken();
    break;
  case tok::string_literal:    // primary-expression: string-literal
    ParseStringLiteralExpression();
    break;
  case tok::l_paren:          // primary-expression: '(' expression ')'
    ParseParenExpression();
    break;
    
  default:
    Diag(Tok, diag::err_expected_expression);
    return;
  }  
}

/// ParseStringLiteralExpression - This handles the various token types that
/// form string literals, and also handles string concatenation [C99 5.1.1.2,
/// translation phase #6].
///
///       primary-expression: [C99 6.5.1]
///         string-literal
void Parser::ParseStringLiteralExpression() {
  assert(isStringLiteral() && "Not a string literal!");
  ConsumeStringToken();
  
  // String concat.  Note that keywords like __func__ and __FUNCTION__ aren't
  // considered to be strings.
  while (isStringLiteral())
    ConsumeStringToken();
}


/// ParseParenExpression - C99 c.5.1p5
///       primary-expression:
///         '(' expression ')'
void Parser::ParseParenExpression() {
  assert(Tok.getKind() == tok::l_paren && "Not a paren expr!");
  SourceLocation OpenLoc = Tok.getLocation();
  ConsumeParen();
  
  ParseExpression();
  
  if (Tok.getKind() == tok::r_paren) {
    ConsumeParen();
  } else {
    Diag(Tok, diag::err_expected_rparen);
    Diag(OpenLoc, diag::err_matching);
  }
}
