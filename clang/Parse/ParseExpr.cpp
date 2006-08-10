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
  case tok::l_paren:           // primary-expression: '(' expression ')'
                               // primary-expression: '(' compound-statement ')'
    ParseParenExpression(false/*allow statement exprs, initializers */);
    break;
  case tok::kw___builtin_va_arg:
  case tok::kw___builtin_offsetof:
  case tok::kw___builtin_choose_expr:
  case tok::kw___builtin_types_compatible_p:
    assert(0 && "FIXME: UNIMP!");
  default:
    Diag(Tok, diag::err_expected_expression);
    // Guarantee forward progress.
    // FIXME: this dies on 'if ({1; });'.  Expression parsing should return a
    // bool for failure.  This will cause higher-level parsing stuff to do the
    // right thing.
    ConsumeToken();
    return;
  }
  
  // Now that the primary-expression piece of the postfix-expression has been
  // parsed, see if there are any postfix-expression pieces here.
  SourceLocation Loc;
  while (1) {
    switch (Tok.getKind()) {
    default:
      return;
    case tok::l_square:    // postfix-expression: p-e '[' expression ']'
      Loc = Tok.getLocation();
      ConsumeBracket();
      ParseExpression();
      if (Tok.getKind() == tok::r_square) {
        ConsumeBracket();
      } else {
        Diag(Tok, diag::err_expected_rsquare);
        Diag(Loc, diag::err_matching);
        SkipUntil(tok::r_square);
      }
      break;
      
    case tok::l_paren:     // p-e: p-e '(' argument-expression-list[opt] ')'
      Loc = Tok.getLocation();
      ConsumeParen();

      while (1) {
        // FIXME: This should be argument-expression!
        ParseAssignmentExpression();
        
        if (Tok.getKind() != tok::comma)
          break;
        ConsumeToken();  // Next argument.
      }
        
      if (Tok.getKind() == tok::r_paren) {
        ConsumeParen();
      } else {
        Diag(Tok, diag::err_expected_rparen);
        Diag(Loc, diag::err_matching);
        SkipUntil(tok::r_paren);
      }
      break;
      
    case tok::arrow:       // postfix-expression: p-e '->' identifier
    case tok::period:      // postfix-expression: p-e '.' identifier
      ConsumeToken();
      if (Tok.getKind() != tok::identifier) {
        Diag(Tok, diag::err_expected_ident);
        return;
      }
      ConsumeToken();
      break;
      
    case tok::plusplus:    // postfix-expression: postfix-expression '++'
    case tok::minusminus:  // postfix-expression: postfix-expression '--'
      ConsumeToken();
      break;
    }
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


/// ParseParenExpression - C99 6.5.1p5
///       primary-expression:
///         '(' expression ')'
/// [GNU]   '(' compound-statement ')'      (if !ParenExprOnly)
///       postfix-expression: [C99 6.5.2]
///         '(' type-name ')' '{' initializer-list '}'
///         '(' type-name ')' '{' initializer-list ',' '}'
///
void Parser::ParseParenExpression(bool ParenExprOnly) {
  assert(Tok.getKind() == tok::l_paren && "Not a paren expr!");
  SourceLocation OpenLoc = Tok.getLocation();
  ConsumeParen();
  
  if (!ParenExprOnly && Tok.getKind() == tok::l_brace &&
      !getLang().NoExtensions) {
    Diag(Tok, diag::ext_gnu_statement_expr);
    ParseCompoundStatement();
  } else if (!ParenExprOnly && 0 /*type */) { 
    // FIXME: Implement compound literals: C99 6.5.2.5.  Type-name: C99 6.7.6.
  } else {
    ParseExpression();
  }
  
  if (Tok.getKind() == tok::r_paren) {
    ConsumeParen();
  } else {
    Diag(Tok, diag::err_expected_rparen);
    Diag(OpenLoc, diag::err_matching);
  }
}
