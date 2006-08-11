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
  if (Tok.getKind() == tok::l_brace) {
    ConsumeBrace();
    
    if (Tok.getKind() == tok::numeric_constant)
      ConsumeToken();
    
    // FIXME: initializer-list
    // Match the '}'.
    MatchRHSPunctuation(tok::r_brace, Tok.getLocation(), "{",
                        diag::err_expected_rbrace);
    return;
  }
  
  ParseAssignmentExpression();
}



Parser::ExprTy Parser::ParseExpression() {
  ParseCastExpression();
  return 0;
}

// Expr that doesn't include commas.
void Parser::ParseAssignmentExpression() {
  ParseExpression();
}

/// ParseCastExpression
///       cast-expression: [C99 6.5.4]
///         unary-expression
///         '(' type-name ')' cast-expression
///
void Parser::ParseCastExpression() {
  // If this doesn't start with an '(', then it is a unary-expression.
  if (Tok.getKind() != tok::l_paren)
    return ParseUnaryExpression();
  
  ParenParseOption ParenExprType = CastExpr;
  ParseParenExpression(ParenExprType);

  switch (ParenExprType) {
  case SimpleExpr: break;    // Nothing to do.
  case CompoundStmt: break;  // Nothing to do.
  case CompoundLiteral:
    // We parsed '(' type-name ')' '{' ... '}'.  If any suffixes of
    // postfix-expression exist, parse them now.
    //Diag(Tok, diag::err_parse_error);
    //assert(0 && "FIXME");
    break;
  case CastExpr:
    // We parsed '(' type-name ')' and the thing after it wasn't a '{'.  Parse
    // the cast-expression that follows it next.
    ParseCastExpression();
    break;
  }
}

/// ParseUnaryExpression
///       unary-expression:  [C99 6.5.3]
///         postfix-expression
///         '++' unary-expression
///         '--' unary-expression
///         unary-operator cast-expression
///         'sizeof' unary-expression
///         'sizeof' '(' type-name ')'
/// [GNU]   '__alignof' unary-expression
/// [GNU]   '__alignof' '(' type-name ')'
/// [GNU]   '&&' identifier
///       unary-operator: one of
///         '&'  '*'  '+'  '-'  '~'  '!'
/// [GNU]   '__extension__'  '__real'  '__imag'
///
void Parser::ParseUnaryExpression() {
  switch (Tok.getKind()) {
  default:                 // unary-expression: postfix-expression
    ParsePostfixExpression();
    break;
  case tok::plusplus:      // unary-expression: '++' unary-expression
  case tok::minusminus:    // unary-expression: '--' unary-expression
    ConsumeToken();
    ParseUnaryExpression();
    break;
  case tok::amp:           // unary-expression: '&' cast-expression
  case tok::star:          // unary-expression: '*' cast-expression
  case tok::plus:          // unary-expression: '+' cast-expression
  case tok::minus:         // unary-expression: '-' cast-expression
  case tok::tilde:         // unary-expression: '~' cast-expression
  case tok::exclaim:       // unary-expression: '!' cast-expression
  case tok::kw___real:     // unary-expression: '__real' cast-expression [GNU]
  case tok::kw___imag:     // unary-expression: '__real' cast-expression [GNU]
  //case tok::kw__extension__:  [TODO]
    ConsumeToken();
    ParseCastExpression();
    break;
    
  case tok::kw_sizeof:     // unary-expression: 'sizeof' unary-expression
                           // unary-expression: 'sizeof' '(' type-name ')'
  case tok::kw___alignof:  // unary-expression: '__alignof' unary-expression
                           // unary-expression: '__alignof' '(' type-name ')'
    ParseSizeofAlignofExpression();
    break;
  case tok::ampamp:        // unary-expression: '&&' identifier
    Diag(Tok, diag::ext_gnu_address_of_label);
    ConsumeToken();
    if (Tok.getKind() == tok::identifier) {
      ConsumeToken();
    } else {
      Diag(Tok, diag::err_expected_ident);
      ConsumeToken(); // FIXME: Should just return error!
      return;
    }
    break;
  }
}

/// ParseSizeofAlignofExpression - Parse a sizeof or alignof expression.
///       unary-expression:  [C99 6.5.3]
///         'sizeof' unary-expression
///         'sizeof' '(' type-name ')'
/// [GNU]   '__alignof' unary-expression
/// [GNU]   '__alignof' '(' type-name ')'
void Parser::ParseSizeofAlignofExpression() {
  assert((Tok.getKind() == tok::kw_sizeof ||
          Tok.getKind() == tok::kw___alignof) &&
         "Not a sizeof/alignof expression!");
  ConsumeToken();
  
  // If the operand doesn't start with an '(', it must be an expression.
  if (Tok.getKind() != tok::l_paren) {
    ParseUnaryExpression();
    return;
  }
  
  // If it starts with a '(', we know that it is either a parenthesized
  // type-name, or it is a unary-expression that starts with a compound literal,
  // or starts with a primary-expression that is a parenthesized expression.
  ParenParseOption ExprType = CastExpr;
  ParseParenExpression(ExprType);
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
  case tok::l_paren: {  // primary-expr: '(' expression ')'
                        // primary-expr: '(' compound-statement ')'
                        // postfix-expr: '(' type-name ')' '{' init-list '}'
                        // postfix-expr: '(' type-name ')' '{' init-list ',' '}'
    ParenParseOption ParenExprType = CompoundLiteral;
    ParseParenExpression(ParenExprType);
    break;
  }
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
      // Match the ']'.
      MatchRHSPunctuation(tok::r_square, Loc, "[", diag::err_expected_rsquare);
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
        
      // Match the ')'.
      MatchRHSPunctuation(tok::r_paren, Loc, "(", diag::err_expected_rparen);
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
  assert(isTokenStringLiteral() && "Not a string literal!");
  ConsumeStringToken();
  
  // String concat.  Note that keywords like __func__ and __FUNCTION__ aren't
  // considered to be strings.
  while (isTokenStringLiteral())
    ConsumeStringToken();
}


/// ParseParenExpression - This parses the unit that starts with a '(' token,
/// based on what is allowed by ExprType.  The actual thing parsed is returned
/// in ExprType.
///
///       primary-expression: [C99 6.5.1]
///         '(' expression ')'
/// [GNU]   '(' compound-statement ')'      (if !ParenExprOnly)
///       postfix-expression: [C99 6.5.2]
///         '(' type-name ')' '{' initializer-list '}'
///         '(' type-name ')' '{' initializer-list ',' '}'
///       cast-expression: [C99 6.5.4]
///         '(' type-name ')' cast-expression
///
void Parser::ParseParenExpression(ParenParseOption &ExprType) {
  assert(Tok.getKind() == tok::l_paren && "Not a paren expr!");
  SourceLocation OpenLoc = Tok.getLocation();
  ConsumeParen();
  
  if (ExprType >= CompoundStmt && Tok.getKind() == tok::l_brace &&
      !getLang().NoExtensions) {
    Diag(Tok, diag::ext_gnu_statement_expr);
    ParseCompoundStatement();
    ExprType = CompoundStmt;
  } else if (ExprType >= CompoundLiteral && isTypeSpecifierQualifier()) {
    // Otherwise, this is a compound expression.
    ParseTypeName();

    // Match the ')'.
    MatchRHSPunctuation(tok::r_paren, OpenLoc, "(", diag::err_expected_rparen);

    if (Tok.getKind() == tok::l_brace) {
      ParseInitializer();
      ExprType = CompoundLiteral;
    } else if (ExprType == CastExpr) {
      // Note that this doesn't parse the subsequence cast-expression.
      ExprType = CastExpr;
    } else {
      Diag(Tok, diag::err_expected_lbrace_in_compound_literal);
      return;
    }
    return;
  } else {
    ParseExpression();
    ExprType = SimpleExpr;
  }
  
  // Match the ')'.
  MatchRHSPunctuation(tok::r_paren, OpenLoc, "(", diag::err_expected_rparen);
}
