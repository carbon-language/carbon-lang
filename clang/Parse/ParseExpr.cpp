//===--- Expression.cpp - Expression Parsing ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Expression parsing implementation.  Expressions in
// C99 basically consist of a bunch of binary operators with unary operators and
// other random stuff at the leaves.
//
// In the C99 grammar, these unary operators bind tightest and are represented
// as the 'cast-expression' production.  Everything else is either a binary
// operator (e.g. '/') or a ternary operator ("?:").  The unary leaves are
// handled by ParseCastExpression, the higher level pieces are handled by
// ParseBinaryExpression.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/Alloca.h"
using namespace llvm;
using namespace clang;

/// PrecedenceLevels - These are precedences for the binary/ternary operators in
/// the C99 grammar.  These have been named to relate with the C99 grammar
/// productions.  Low precedences numbers bind more weakly than high numbers.
namespace prec {
  enum Level {
    Unknown        = 0,    // Not binary operator.
    Comma          = 1,    // ,
    Assignment     = 2,    // =, *=, /=, %=, +=, -=, <<=, >>=, &=, ^=, |=
    Conditional    = 3,    // ?
    LogicalOr      = 4,    // ||
    LogicalAnd     = 5,    // &&
    InclusiveOr    = 6,    // |
    ExclusiveOr    = 7,    // ^
    And            = 8,    // &
    MinMax         = 9,   // <?, >?           min, max (GCC extensions)
    Equality       = 10,   // ==, !=
    Relational     = 11,   //  >=, <=, >, <
    Shift          = 12,   // <<, >>
    Additive       = 13,   // -, +
    Multiplicative = 14    // *, /, %
  };
}


/// getBinOpPrecedence - Return the precedence of the specified binary operator
/// token.  This returns:
///
static prec::Level getBinOpPrecedence(tok::TokenKind Kind) {
  switch (Kind) {
  default:                        return prec::Unknown;
  case tok::comma:                return prec::Comma;
  case tok::equal:
  case tok::starequal:
  case tok::slashequal:
  case tok::percentequal:
  case tok::plusequal:
  case tok::minusequal:
  case tok::lesslessequal:
  case tok::greatergreaterequal:
  case tok::ampequal:
  case tok::caretequal:
  case tok::pipeequal:            return prec::Assignment;
  case tok::question:             return prec::Conditional;
  case tok::pipepipe:             return prec::LogicalOr;
  case tok::ampamp:               return prec::LogicalAnd;
  case tok::pipe:                 return prec::InclusiveOr;
  case tok::caret:                return prec::ExclusiveOr;
  case tok::amp:                  return prec::And;
  case tok::lessquestion:
  case tok::greaterquestion:      return prec::MinMax;
  case tok::exclaimequal:
  case tok::equalequal:           return prec::Equality;
  case tok::lessequal:
  case tok::less:
  case tok::greaterequal:
  case tok::greater:              return prec::Relational;
  case tok::lessless:
  case tok::greatergreater:       return prec::Shift;
  case tok::plus:
  case tok::minus:                return prec::Additive;
  case tok::percent:
  case tok::slash:
  case tok::star:                 return prec::Multiplicative;
  }
}


/// ParseExpression - Simple precedence-based parser for binary/ternary
/// operators.
///
/// Note: we diverge from the C99 grammar when parsing the assignment-expression
/// production.  C99 specifies that the LHS of an assignment operator should be
/// parsed as a unary-expression, but consistency dictates that it be a
/// conditional-expession.  In practice, the important thing here is that the
/// LHS of an assignment has to be an l-value, which productions between
/// unary-expression and conditional-expression don't produce.  Because we want
/// consistency, we parse the LHS as a conditional-expression, then check for
/// l-value-ness in semantic analysis stages.
///
///       multiplicative-expression: [C99 6.5.5]
///         cast-expression
///         multiplicative-expression '*' cast-expression
///         multiplicative-expression '/' cast-expression
///         multiplicative-expression '%' cast-expression
///
///       additive-expression: [C99 6.5.6]
///         multiplicative-expression
///         additive-expression '+' multiplicative-expression
///         additive-expression '-' multiplicative-expression
///
///       shift-expression: [C99 6.5.7]
///         additive-expression
///         shift-expression '<<' additive-expression
///         shift-expression '>>' additive-expression
///
///       relational-expression: [C99 6.5.8]
///         shift-expression
///         relational-expression '<' shift-expression
///         relational-expression '>' shift-expression
///         relational-expression '<=' shift-expression
///         relational-expression '>=' shift-expression
///
///       equality-expression: [C99 6.5.9]
///         relational-expression
///         equality-expression '==' relational-expression
///         equality-expression '!=' relational-expression
///
///       AND-expression: [C99 6.5.10]
///         equality-expression
///         AND-expression '&' equality-expression
///
///       exclusive-OR-expression: [C99 6.5.11]
///         AND-expression
///         exclusive-OR-expression '^' AND-expression
///
///       inclusive-OR-expression: [C99 6.5.12]
///         exclusive-OR-expression
///         inclusive-OR-expression '|' exclusive-OR-expression
///
///       logical-AND-expression: [C99 6.5.13]
///         inclusive-OR-expression
///         logical-AND-expression '&&' inclusive-OR-expression
///
///       logical-OR-expression: [C99 6.5.14]
///         logical-AND-expression
///         logical-OR-expression '||' logical-AND-expression
///
///       conditional-expression: [C99 6.5.15]
///         logical-OR-expression
///         logical-OR-expression '?' expression ':' conditional-expression
/// [GNU]   logical-OR-expression '?' ':' conditional-expression
///
///       assignment-expression: [C99 6.5.16]
///         conditional-expression
///         unary-expression assignment-operator assignment-expression
///
///       assignment-operator: one of
///         = *= /= %= += -= <<= >>= &= ^= |=
///
///       expression: [C99 6.5.17]
///         assignment-expression
///         expression ',' assignment-expression
///
Parser::ExprResult Parser::ParseExpression() {
  ExprResult LHS = ParseCastExpression(false);
  if (LHS.isInvalid) return LHS;
  
  return ParseRHSOfBinaryExpression(LHS, prec::Comma);
}

/// ParseAssignmentExpression - Parse an expr that doesn't include commas.
///
Parser::ExprResult Parser::ParseAssignmentExpression() {
  ExprResult LHS = ParseCastExpression(false);
  if (LHS.isInvalid) return LHS;
  
  return ParseRHSOfBinaryExpression(LHS, prec::Assignment);
}

Parser::ExprResult Parser::ParseConstantExpression() {
  ExprResult LHS = ParseCastExpression(false);
  if (LHS.isInvalid) return LHS;
  
  // TODO: Validate that this is a constant expr!
  return ParseRHSOfBinaryExpression(LHS, prec::Conditional);
}

/// ParseExpressionWithLeadingIdentifier - This special purpose method is used
/// in contexts where we have already consumed an identifier (which we saved in
/// 'Tok'), then discovered that the identifier was really the leading token of
/// part of an expression.  For example, in "A[1]+B", we consumed "A" (which is
/// now in 'Tok') and the current token is "[".
Parser::ExprResult Parser::
ParseExpressionWithLeadingIdentifier(const LexerToken &Tok) {
  // We know that 'Tok' must correspond to this production:
  //   primary-expression: identifier
  
  // TODO: Pass 'Tok' to the action.
  ExprResult Res = ExprResult(false);
  
  // Because we have to parse an entire cast-expression before starting the
  // ParseRHSOfBinaryExpression method (which parses any trailing binops), we
  // need to handle the 'postfix-expression' rules.  We do this by invoking
  // ParsePostfixExpressionSuffix to consume any postfix-expression suffixes:
  Res = ParsePostfixExpressionSuffix(Res);
  if (Res.isInvalid) return Res;

  // At this point, the "A[1]" part of "A[1]+B" has been consumed. Once this is
  // done, we know we don't have to do anything for cast-expression, because the
  // only non-postfix-expression production starts with a '(' token, and we know
  // we have an identifier.  As such, we can invoke ParseRHSOfBinaryExpression
  // to consume any trailing operators (e.g. "+" in this example) and connected
  // chunks of the expression.
  return ParseRHSOfBinaryExpression(Res, prec::Comma);
}

/// ParseExpressionWithLeadingIdentifier - This special purpose method is used
/// in contexts where we have already consumed an identifier (which we saved in
/// 'Tok'), then discovered that the identifier was really the leading token of
/// part of an assignment-expression.  For example, in "A[1]+B", we consumed "A"
/// (which is now in 'Tok') and the current token is "[".
Parser::ExprResult Parser::
ParseAssignmentExprWithLeadingIdentifier(const LexerToken &Tok) {
  // We know that 'Tok' must correspond to this production:
  //   primary-expression: identifier
  
  // TODO: Pass 'Tok' to the action.
  ExprResult Res = ExprResult(false);
  
  // Because we have to parse an entire cast-expression before starting the
  // ParseRHSOfBinaryExpression method (which parses any trailing binops), we
  // need to handle the 'postfix-expression' rules.  We do this by invoking
  // ParsePostfixExpressionSuffix to consume any postfix-expression suffixes:
  Res = ParsePostfixExpressionSuffix(Res);
  if (Res.isInvalid) return Res;
  
  // At this point, the "A[1]" part of "A[1]+B" has been consumed. Once this is
  // done, we know we don't have to do anything for cast-expression, because the
  // only non-postfix-expression production starts with a '(' token, and we know
  // we have an identifier.  As such, we can invoke ParseRHSOfBinaryExpression
  // to consume any trailing operators (e.g. "+" in this example) and connected
  // chunks of the expression.
  return ParseRHSOfBinaryExpression(Res, prec::Assignment);
}


/// ParseAssignmentExpressionWithLeadingStar - This special purpose method is
/// used in contexts where we have already consumed a '*' (which we saved in
/// 'Tok'), then discovered that the '*' was really the leading token of an
/// expression.  For example, in "*(int*)P+B", we consumed "*" (which is
/// now in 'Tok') and the current token is "(".
Parser::ExprResult Parser::
ParseAssignmentExpressionWithLeadingStar(const LexerToken &Tok) {
  // We know that 'Tok' must correspond to this production:
  //  unary-expression: unary-operator cast-expression
  // where 'unary-operator' is '*'.
  
  // Parse the cast-expression that follows the '*'.  This will parse the
  // "*(int*)P" part of "*(int*)P+B".
  ExprResult Res = ParseCastExpression(false);
  if (Res.isInvalid) return Res;

  // TODO: Combine Tok + Res to get the new AST.
  
  // We have to parse an entire cast-expression before starting the
  // ParseRHSOfBinaryExpression method (which parses any trailing binops). Since
  // we know that the only production above us is the cast-expression
  // production, and because the only alternative productions start with a '('
  // token (we know we had a '*'), there is no work to do to get a whole
  // cast-expression.
  
  // At this point, the "*(int*)P" part of "*(int*)P+B" has been consumed. Once
  // this is done, we can invoke ParseRHSOfBinaryExpression to consume any
  // trailing operators (e.g. "+" in this example) and connected chunks of the
  // assignment-expression.
  return ParseRHSOfBinaryExpression(Res, prec::Assignment);
}


/// ParseRHSOfBinaryExpression - Parse a binary expression that starts with
/// LHS and has a precedence of at least MinPrec.
Parser::ExprResult
Parser::ParseRHSOfBinaryExpression(ExprResult LHS, unsigned MinPrec) {
  unsigned NextTokPrec = getBinOpPrecedence(Tok.getKind());
  SourceLocation ColonLoc;

  while (1) {
    // If this token has a lower precedence than we are allowed to parse (e.g.
    // because we are called recursively, or because the token is not a binop),
    // then we are done!
    if (NextTokPrec < MinPrec)
      return LHS;

    // Consume the operator, saving the operator token for error reporting.
    LexerToken OpToken = Tok;
    ConsumeToken();
    
    // Special case handling for the ternary operator.
    ExprResult TernaryMiddle(true);
    if (NextTokPrec == prec::Conditional) {
      if (Tok.getKind() != tok::colon) {
        // Handle this production specially:
        //   logical-OR-expression '?' expression ':' conditional-expression
        // In particular, the RHS of the '?' is 'expression', not
        // 'logical-OR-expression' as we might expect.
        TernaryMiddle = ParseExpression();
        if (TernaryMiddle.isInvalid) return TernaryMiddle;
      } else {
        // Special case handling of "X ? Y : Z" where Y is empty:
        //   logical-OR-expression '?' ':' conditional-expression   [GNU]
        TernaryMiddle = ExprResult(false);
        Diag(Tok, diag::ext_gnu_conditional_expr);
      }
      
      if (Tok.getKind() != tok::colon) {
        Diag(Tok, diag::err_expected_colon);
        Diag(OpToken, diag::err_matching, "?");
        return ExprResult(true);
      }
      
      // Eat the colon.
      ColonLoc = ConsumeToken();
    }
    
    // Parse another leaf here for the RHS of the operator.
    ExprResult RHS = ParseCastExpression(false);
    if (RHS.isInvalid) return RHS;

    // Remember the precedence of this operator and get the precedence of the
    // operator immediately to the right of the RHS.
    unsigned ThisPrec = NextTokPrec;
    NextTokPrec = getBinOpPrecedence(Tok.getKind());

    // Assignment and conditional expressions are right-associative.
    bool isRightAssoc = NextTokPrec == prec::Conditional ||
                        NextTokPrec == prec::Assignment;

    // Get the precedence of the operator to the right of the RHS.  If it binds
    // more tightly with RHS than we do, evaluate it completely first.
    if (ThisPrec < NextTokPrec ||
        (ThisPrec == NextTokPrec && isRightAssoc)) {
      // If this is left-associative, only parse things on the RHS that bind
      // more tightly than the current operator.  If it is left-associative, it
      // is okay, to bind exactly as tightly.  For example, compile A=B=C=D as
      // A=(B=(C=D)), where each paren is a level of recursion here.
      RHS = ParseRHSOfBinaryExpression(RHS, ThisPrec + !isRightAssoc);
      if (RHS.isInvalid) return RHS;

      NextTokPrec = getBinOpPrecedence(Tok.getKind());
    }
    assert(NextTokPrec <= ThisPrec && "Recursion didn't work!");
  
    // Combine the LHS and RHS into the LHS (e.g. build AST).
    if (TernaryMiddle.isInvalid)
      LHS = Actions.ParseBinOp(OpToken, LHS.Val, RHS.Val);
    else
      LHS = Actions.ParseConditionalOp(OpToken.getLocation(), ColonLoc,
                                       LHS.Val, TernaryMiddle.Val, RHS.Val);
  }
}

/// ParseCastExpression - Parse a cast-expression, or, if isUnaryExpression is
/// true, parse a unary-expression.
///
///       cast-expression: [C99 6.5.4]
///         unary-expression
///         '(' type-name ')' cast-expression
///
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
///
///       unary-operator: one of
///         '&'  '*'  '+'  '-'  '~'  '!'
/// [GNU]   '__extension__'  '__real'  '__imag'
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
Parser::ExprResult Parser::ParseCastExpression(bool isUnaryExpression) {
  ExprResult Res;
  tok::TokenKind SavedKind = Tok.getKind();
  
  // This handles all of cast-expression, unary-expression, postfix-expression,
  // and primary-expression.  We handle them together like this for efficiency
  // and to simplify handling of an expression starting with a '(' token: which
  // may be one of a parenthesized expression, cast-expression, compound literal
  // expression, or statement expression.
  //
  // If the parsed tokens consist of a primary-expression, the cases below
  // call ParsePostfixExpressionSuffix to handle the postfix expression
  // suffixes.  Cases that cannot be followed by postfix exprs should
  // return without invoking ParsePostfixExpressionSuffix.
  switch (Tok.getKind()) {
  case tok::l_paren: {
    // If this expression is limited to being a unary-expression, the parent can
    // not start a cast expression.
    ParenParseOption ParenExprType =
      isUnaryExpression ? CompoundLiteral : CastExpr;
    TypeTy *CastTy;
    SourceLocation LParenLoc = Tok.getLocation();
    SourceLocation RParenLoc;
    Res = ParseParenExpression(ParenExprType, CastTy, RParenLoc);
    if (Res.isInvalid) return Res;
    
    switch (ParenExprType) {
    case SimpleExpr:   break;    // Nothing else to do.
    case CompoundStmt: break;  // Nothing else to do.
    case CompoundLiteral:
      // We parsed '(' type-name ')' '{' ... '}'.  If any suffixes of
      // postfix-expression exist, parse them now.
      break;
    case CastExpr:
      // We parsed '(' type-name ')' and the thing after it wasn't a '{'.  Parse
      // the cast-expression that follows it next.
      // TODO: For cast expression with CastTy.
      Res = ParseCastExpression(false);
      if (!Res.isInvalid)
        Res = Actions.ParseCastExpr(LParenLoc, CastTy, RParenLoc, Res.Val);
      return Res;
    }
      
    // These can be followed by postfix-expr pieces.
    return ParsePostfixExpressionSuffix(Res);
  }
    
    // primary-expression
  case tok::numeric_constant:
    // constant: integer-constant
    // constant: floating-constant
    
    // TODO: Validate whether this is an integer or floating-constant or
    // neither.
    if (1) {
      Res = Actions.ParseIntegerConstant(Tok);
    } else {
      Res = Actions.ParseFloatingConstant(Tok);
    }
    ConsumeToken();
    
    // These can be followed by postfix-expr pieces.
    return ParsePostfixExpressionSuffix(Res);

  case tok::identifier:        // primary-expression: identifier
                               // constant: enumeration-constant
  case tok::char_constant:     // constant: character-constant
  case tok::kw___func__:       // primary-expression: __func__ [C99 6.4.2.2]
  case tok::kw___FUNCTION__:   // primary-expression: __FUNCTION__ [GNU]
  case tok::kw___PRETTY_FUNCTION__:  // primary-expression: __P..Y_F..N__ [GNU]
    Res = Actions.ParseSimplePrimaryExpr(Tok);
    ConsumeToken();
    // These can be followed by postfix-expr pieces.
    return ParsePostfixExpressionSuffix(Res);
  case tok::string_literal:    // primary-expression: string-literal
  case tok::wide_string_literal:
    Res = ParseStringLiteralExpression();
    if (Res.isInvalid) return Res;
    // This can be followed by postfix-expr pieces (e.g. "foo"[1]).
    return ParsePostfixExpressionSuffix(Res);
  case tok::kw___builtin_va_arg:
  case tok::kw___builtin_offsetof:
  case tok::kw___builtin_choose_expr:
  case tok::kw___builtin_types_compatible_p:
    return ParseBuiltinPrimaryExpression();
  case tok::plusplus:      // unary-expression: '++' unary-expression
  case tok::minusminus: {  // unary-expression: '--' unary-expression
    SourceLocation SavedLoc = ConsumeToken();
    Res = ParseCastExpression(true);
    if (!Res.isInvalid)
      Res = Actions.ParseUnaryOp(SavedLoc, SavedKind, Res.Val);
    return Res;
  }
  case tok::amp:           // unary-expression: '&' cast-expression
  case tok::star:          // unary-expression: '*' cast-expression
  case tok::plus:          // unary-expression: '+' cast-expression
  case tok::minus:         // unary-expression: '-' cast-expression
  case tok::tilde:         // unary-expression: '~' cast-expression
  case tok::exclaim:       // unary-expression: '!' cast-expression
  case tok::kw___real:     // unary-expression: '__real' cast-expression [GNU]
  case tok::kw___imag:     // unary-expression: '__imag' cast-expression [GNU]
  case tok::kw___extension__:{//unary-expression:'__extension__' cast-expr [GNU]
    // FIXME: Extension not handled correctly here!
    SourceLocation SavedLoc = ConsumeToken();
    Res = ParseCastExpression(false);
    if (!Res.isInvalid)
      Res = Actions.ParseUnaryOp(SavedLoc, SavedKind, Res.Val);
    return Res;
  }
  case tok::kw_sizeof:     // unary-expression: 'sizeof' unary-expression
                           // unary-expression: 'sizeof' '(' type-name ')'
  case tok::kw___alignof:  // unary-expression: '__alignof' unary-expression
                           // unary-expression: '__alignof' '(' type-name ')'
    return ParseSizeofAlignofExpression();
  case tok::ampamp: {      // unary-expression: '&&' identifier
    Diag(Tok, diag::ext_gnu_address_of_label);
    SourceLocation SavedLoc = ConsumeToken();
    
    if (Tok.getKind() != tok::identifier) {
      Diag(Tok, diag::err_expected_ident);
      return ExprResult(true);
    }
    // FIXME: Create a label ref for Tok.Ident.
    Res = Actions.ParseUnaryOp(SavedLoc, SavedKind, 0);
    ConsumeToken();
      
    return Res;
  }
  default:
    Diag(Tok, diag::err_expected_expression);
    return ExprResult(true);
  }
  
  // unreachable.
  abort();
}

/// ParsePostfixExpressionSuffix - Once the leading part of a postfix-expression
/// is parsed, this method parses any suffixes that apply.
///
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
///         argument-expression-list ',' assignment-expression
///
Parser::ExprResult Parser::ParsePostfixExpressionSuffix(ExprResult LHS) {
  
  // Now that the primary-expression piece of the postfix-expression has been
  // parsed, see if there are any postfix-expression pieces here.
  SourceLocation Loc;
  while (1) {
    switch (Tok.getKind()) {
    default:  // Not a postfix-expression suffix.
      return LHS;
    case tok::l_square: {  // postfix-expression: p-e '[' expression ']'
      Loc = ConsumeBracket();
      ExprResult Idx = ParseExpression();
      
      SourceLocation RLoc = Tok.getLocation();
      
      if (!LHS.isInvalid && !Idx.isInvalid && Tok.getKind() == tok::r_square)
        LHS = Actions.ParseArraySubscriptExpr(LHS.Val, Loc, Idx.Val, RLoc);
      else
        LHS = ExprResult(true);

      // Match the ']'.
      MatchRHSPunctuation(tok::r_square, Loc);
      break;
    }
      
    case tok::l_paren: {   // p-e: p-e '(' argument-expression-list[opt] ')'
      SmallVector<ExprTy*, 8> ArgExprs;
      SmallVector<SourceLocation, 8> CommaLocs;
      bool ArgExprsOk = true;
      
      Loc = ConsumeParen();
      
      if (Tok.getKind() != tok::r_paren) {
        while (1) {
          ExprResult ArgExpr = ParseAssignmentExpression();
          if (ArgExpr.isInvalid)
            ArgExprsOk = false;
          else
            ArgExprs.push_back(ArgExpr.Val);
          
          if (Tok.getKind() != tok::comma)
            break;
          // Move to the next argument, remember where the comma was.
          CommaLocs.push_back(ConsumeToken());
        }
      }
        
      // Match the ')'.
      if (!LHS.isInvalid && ArgExprsOk && Tok.getKind() == tok::r_paren) {
        assert((ArgExprs.size() == 0 || ArgExprs.size()-1 == CommaLocs.size())&&
               "Unexpected number of commas!");
        LHS = Actions.ParseCallExpr(LHS.Val, Loc, &ArgExprs[0], ArgExprs.size(),
                                    &CommaLocs[0], Tok.getLocation());
      }
      
      MatchRHSPunctuation(tok::r_paren, Loc);
      break;
    }
    case tok::arrow:       // postfix-expression: p-e '->' identifier
    case tok::period: {    // postfix-expression: p-e '.' identifier
      tok::TokenKind OpKind = Tok.getKind();
      SourceLocation OpLoc = ConsumeToken();  // Eat the "." or "->" token.
      
      if (Tok.getKind() != tok::identifier) {
        Diag(Tok, diag::err_expected_ident);
        return ExprResult(true);
      }
      
      if (!LHS.isInvalid)
        LHS = Actions.ParseMemberReferenceExpr(LHS.Val, OpLoc, OpKind,
                                               Tok.getLocation(),
                                               *Tok.getIdentifierInfo());
      ConsumeToken();
      break;
    }
    case tok::plusplus:    // postfix-expression: postfix-expression '++'
    case tok::minusminus:  // postfix-expression: postfix-expression '--'
      if (!LHS.isInvalid)
        LHS = Actions.ParsePostfixUnaryOp(Tok, LHS.Val);
      ConsumeToken();
      break;
    }
  }
}


/// ParseSizeofAlignofExpression - Parse a sizeof or alignof expression.
///       unary-expression:  [C99 6.5.3]
///         'sizeof' unary-expression
///         'sizeof' '(' type-name ')'
/// [GNU]   '__alignof' unary-expression
/// [GNU]   '__alignof' '(' type-name ')'
Parser::ExprResult Parser::ParseSizeofAlignofExpression() {
  assert((Tok.getKind() == tok::kw_sizeof ||
          Tok.getKind() == tok::kw___alignof) &&
         "Not a sizeof/alignof expression!");
  LexerToken OpTok = Tok;
  ConsumeToken();
  
  // If the operand doesn't start with an '(', it must be an expression.
  ExprResult Operand;
  if (Tok.getKind() != tok::l_paren) {
    Operand = ParseCastExpression(true);
  } else {
    // If it starts with a '(', we know that it is either a parenthesized
    // type-name, or it is a unary-expression that starts with a compound
    // literal, or starts with a primary-expression that is a parenthesized
    // expression.
    ParenParseOption ExprType = CastExpr;
    TypeTy *CastTy;
    SourceLocation LParenLoc = Tok.getLocation(), RParenLoc;
    Operand = ParseParenExpression(ExprType, CastTy, RParenLoc);
    
    // If ParseParenExpression parsed a '(typename)' sequence only, the this is
    // sizeof/alignof a type.  Otherwise, it is sizeof/alignof an expression.
    if (ExprType == CastExpr) {
      return Actions.ParseSizeOfAlignOfTypeExpr(OpTok.getLocation(),
                                              OpTok.getKind() == tok::kw_sizeof,
                                                LParenLoc, CastTy, RParenLoc);
    }
  }
  
  // If we get here, the operand to the sizeof/alignof was an expresion.
  if (!Operand.isInvalid)
    Operand = Actions.ParseUnaryOp(OpTok.getLocation(), OpTok.getKind(),
                                   Operand.Val);
  return Operand;
}

/// ParseBuiltinPrimaryExpression
///
///       primary-expression: [C99 6.5.1]
/// [GNU]   '__builtin_va_arg' '(' assignment-expression ',' type-name ')'
/// [GNU]   '__builtin_offsetof' '(' type-name ',' offsetof-member-designator')'
/// [GNU]   '__builtin_choose_expr' '(' assign-expr ',' assign-expr ','
///                                     assign-expr ')'
/// [GNU]   '__builtin_types_compatible_p' '(' type-name ',' type-name ')'
/// 
/// [GNU] offsetof-member-designator:
/// [GNU]   identifier
/// [GNU]   offsetof-member-designator '.' identifier
/// [GNU]   offsetof-member-designator '[' expression ']'
///
Parser::ExprResult Parser::ParseBuiltinPrimaryExpression() {
  ExprResult Res(false);
  const IdentifierInfo *BuiltinII = Tok.getIdentifierInfo();

  tok::TokenKind T = Tok.getKind();
  SourceLocation StartLoc = ConsumeToken();   // Eat the builtin identifier.

  // All of these start with an open paren.
  if (Tok.getKind() != tok::l_paren) {
    Diag(Tok, diag::err_expected_lparen_after, BuiltinII->getName());
    return ExprResult(true);
  }
  
  SourceLocation LParenLoc = ConsumeParen();
  // TODO: Build AST.

  switch (T) {
  default: assert(0 && "Not a builtin primary expression!");
  case tok::kw___builtin_va_arg:
    Res = ParseAssignmentExpression();
    if (Res.isInvalid) {
      SkipUntil(tok::r_paren);
      return Res;
    }

    if (ExpectAndConsume(tok::comma, diag::err_expected_comma, "",tok::r_paren))
      return ExprResult(true);

    ParseTypeName();
    break;
    
  case tok::kw___builtin_offsetof:
    ParseTypeName();

    if (ExpectAndConsume(tok::comma, diag::err_expected_comma, "",tok::r_paren))
      return ExprResult(true);
    
    // We must have at least one identifier here.
    if (ExpectAndConsume(tok::identifier, diag::err_expected_ident, "",
                         tok::r_paren))
      return ExprResult(true);

    while (1) {
      if (Tok.getKind() == tok::period) {
        // offsetof-member-designator: offsetof-member-designator '.' identifier
        ConsumeToken();
        
        if (ExpectAndConsume(tok::identifier, diag::err_expected_ident, "",
                             tok::r_paren))
          return ExprResult(true);
      } else if (Tok.getKind() == tok::l_square) {
        // offsetof-member-designator: offsetof-member-design '[' expression ']'
        SourceLocation LSquareLoc = ConsumeBracket();
        Res = ParseExpression();
        if (Res.isInvalid) {
          SkipUntil(tok::r_paren);
          return Res;
        }

        MatchRHSPunctuation(tok::r_square, LSquareLoc);
      } else {
        break;
      }
    }
    break;
  case tok::kw___builtin_choose_expr:
    Res = ParseAssignmentExpression();
    
    if (ExpectAndConsume(tok::comma, diag::err_expected_comma, "",tok::r_paren))
      return ExprResult(true);
    
    Res = ParseAssignmentExpression();
    
    if (ExpectAndConsume(tok::comma, diag::err_expected_comma, "",tok::r_paren))
      return ExprResult(true);
    
    Res = ParseAssignmentExpression();
    break;
  case tok::kw___builtin_types_compatible_p:
    ParseTypeName();
    
    if (ExpectAndConsume(tok::comma, diag::err_expected_comma, "",tok::r_paren))
      return ExprResult(true);
    
    ParseTypeName();
    break;
  }      
  
  MatchRHSPunctuation(tok::r_paren, LParenLoc);
  
  // These can be followed by postfix-expr pieces because they are
  // primary-expressions.
  return ParsePostfixExpressionSuffix(Res);
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
Parser::ExprResult Parser::ParseParenExpression(ParenParseOption &ExprType,
                                                TypeTy *&CastTy,
                                                SourceLocation &RParenLoc) {
  assert(Tok.getKind() == tok::l_paren && "Not a paren expr!");
  SourceLocation OpenLoc = ConsumeParen();
  ExprResult Result(false);
  CastTy = 0;
  
  if (ExprType >= CompoundStmt && Tok.getKind() == tok::l_brace &&
      !getLang().NoExtensions) {
    Diag(Tok, diag::ext_gnu_statement_expr);
    ParseCompoundStatement();
    ExprType = CompoundStmt;
    // TODO: Build AST for GNU compound stmt.
  } else if (ExprType >= CompoundLiteral && isTypeSpecifierQualifier()) {
    // Otherwise, this is a compound literal expression or cast expression.
    TypeTy *Ty = ParseTypeName();

    // Match the ')'.
    if (Tok.getKind() == tok::r_paren)
      RParenLoc = ConsumeParen();
    else
      MatchRHSPunctuation(tok::r_paren, OpenLoc);
    
    if (Tok.getKind() == tok::l_brace) {
      if (!getLang().C99)   // Compound literals don't exist in C90.
        Diag(OpenLoc, diag::ext_c99_compound_literal);
      Result = ParseInitializer();
      ExprType = CompoundLiteral;
      // TODO: Build AST for compound literal.
    } else if (ExprType == CastExpr) {
      // Note that this doesn't parse the subsequence cast-expression, it just
      // returns the parsed type to the callee.
      ExprType = CastExpr;
      CastTy = Ty;
      return ExprResult(false);
    } else {
      Diag(Tok, diag::err_expected_lbrace_in_compound_literal);
      return ExprResult(true);
    }
    return Result;
  } else {
    Result = ParseExpression();
    ExprType = SimpleExpr;
    if (!Result.isInvalid && Tok.getKind() == tok::r_paren)
      Result = Actions.ParseParenExpr(OpenLoc, Tok.getLocation(), Result.Val);
  }
  
  // Match the ')'.
  if (Result.isInvalid)
    SkipUntil(tok::r_paren);
  else {
    if (Tok.getKind() == tok::r_paren)
      RParenLoc = ConsumeParen();
    else
      MatchRHSPunctuation(tok::r_paren, OpenLoc);
  }
  
  return Result;
}

/// HexDigitValue - Return the value of the specified hex digit, or -1 if it's
/// not valid.
static int HexDigitValue(char C) {
  if (C >= '0' && C <= '9') return C-'0';
  if (C >= 'a' && C <= 'f') return C-'a'+10;
  if (C >= 'A' && C <= 'F') return C-'A'+10;
  return -1;
}

/// ParseStringLiteralExpression - This handles the various token types that
/// form string literals, and also handles string concatenation [C99 5.1.1.2,
/// translation phase #6].
///
///       primary-expression: [C99 6.5.1]
///         string-literal
Parser::ExprResult Parser::ParseStringLiteralExpression() {
  assert(isTokenStringLiteral() && "Not a string literal!");
  
  // String concat.  Note that keywords like __func__ and __FUNCTION__ are not
  // considered to be strings for concatenation purposes.
  SmallVector<LexerToken, 4> StringToks;
  
  // While we're looking at all of the string portions, remember the max
  // individual token length, computing a bound on the concatenated string
  // length, and see whether any piece is a wide-string.  If any of the string
  // portions is a wide-string literal, the result is also a wide-string literal
  // [C99 6.4.5p4].
  unsigned SizeBound = 0, MaxTokenLength = 0;
  bool AnyWide = false;
  do {
    // The string could be shorter than this if it needs cleaning, but this is a
    // reasonable bound, which is all we need.
    SizeBound += Tok.getLength()-2;  // -2 for "".
    
    // Find maximum string piece length.
    if (Tok.getLength() > MaxTokenLength) 
      MaxTokenLength = Tok.getLength();
    
    // Remember if we see any wide strings.
    AnyWide |= Tok.getKind() == tok::wide_string_literal;
    
    // Remember the string token.
    StringToks.push_back(Tok);
    ConsumeStringToken();
  } while (isTokenStringLiteral());
  
  // Include space for the null terminator.
  ++SizeBound;
  
  // TODO: K&R warning: "traditional C rejects string constant concatenation"
  
  // Get the width in bytes of wchar_t.  If no wchar_t strings are used, do not
  // query the target.  As such, wchar_tByteWidth is only valid if AnyWide=true.
  unsigned wchar_tByteWidth = ~0U;
  if (AnyWide)
    wchar_tByteWidth=getTargetInfo().getWCharWidth(StringToks[0].getLocation());
  
  // The output buffer size needs to be large enough to hold wide characters.
  // This is a worst-case assumption which basically corresponds to L"" "long".
  if (AnyWide)
    SizeBound *= wchar_tByteWidth;
  
  // Create a temporary buffer to hold the result string data.  If it is "big",
  // use malloc, otherwise use alloca.
  char *ResultBuf;
  if (SizeBound > 512)
    ResultBuf = (char*)malloc(SizeBound);
  else
    ResultBuf = (char*)alloca(SizeBound);
  
  // Likewise, but for each string piece.
  char *TokenBuf;
  if (MaxTokenLength > 512)
    TokenBuf = (char*)malloc(MaxTokenLength);
  else
    TokenBuf = (char*)alloca(MaxTokenLength);
  
  // Loop over all the strings, getting their spelling, and expanding them to
  // wide strings as appropriate.
  char *ResultPtr = ResultBuf;   // Next byte to fill in.
  
  for (unsigned i = 0, e = StringToks.size(); i != e; ++i) {
    const char *ThisTokBuf = TokenBuf;
    // Get the spelling of the token, which eliminates trigraphs, etc.  We know
    // that ThisTokBuf points to a buffer that is big enough for the whole token
    // and 'spelled' tokens can only shrink.
    unsigned ThisTokLen = PP.getSpelling(StringToks[i], ThisTokBuf);
    const char *ThisTokEnd = ThisTokBuf+ThisTokLen-1;  // Skip end quote.
    
    // TODO: Input character set mapping support.
    
    // Skip L marker for wide strings.
    if (ThisTokBuf[0] == 'L') ++ThisTokBuf;
    
    assert(ThisTokBuf[0] == '"' && "Expected quote, lexer broken?");
    ++ThisTokBuf;
    
    while (ThisTokBuf != ThisTokEnd) {
      // Is this a span of non-escape characters?
      if (ThisTokBuf[0] != '\\') {
        const char *InStart = ThisTokBuf;
        do {
          ++ThisTokBuf;
        } while (ThisTokBuf != ThisTokEnd && ThisTokBuf[0] != '\\');
        
        // Copy the character span over.
        unsigned Len = ThisTokBuf-InStart;
        if (!AnyWide) {
          memcpy(ResultPtr, InStart, Len);
          ResultPtr += Len;
        } else {
          // Note: our internal rep of wide char tokens is always little-endian.
          for (; Len; --Len, ++InStart) {
            *ResultPtr++ = InStart[0];
            // Add zeros at the end.
            for (unsigned i = 1, e = wchar_tByteWidth; i != e; ++i)
              *ResultPtr++ = 0;
          }
        }
        continue;
      }
      
      // Otherwise, this is an escape character.  Skip the '\' char.
      ++ThisTokBuf;
      
      // We know that this character can't be off the end of the buffer, because
      // that would have been \", which would not have been the end of string.
      unsigned ResultChar = *ThisTokBuf++;
      switch (ResultChar) {
      // These map to themselves.
      case '\\': case '\'': case '"': case '?': break;
        
      // These have fixed mappings.
      case 'a':
        // TODO: K&R: the meaning of '\\a' is different in traditional C
        ResultChar = 7;
        break;
      case 'b':
        ResultChar = 8;
        break;
      case 'e':
        PP.Diag(StringToks[i], diag::ext_nonstandard_escape, "e");
        ResultChar = 27;
        break;
      case 'f':
        ResultChar = 12;
        break;
      case 'n':
        ResultChar = 10;
        break;
      case 'r':
        ResultChar = 13;
        break;
      case 't':
        ResultChar = 9;
        break;
      case 'v':
        ResultChar = 11;
        break;
        
      //case 'u': case 'U':  // FIXME: UCNs.
      case 'x': // Hex escape.
        if (ThisTokBuf == ThisTokEnd ||
            (ResultChar = HexDigitValue(*ThisTokBuf)) == ~0U) {
          PP.Diag(StringToks[i], diag::err_hex_escape_no_digits);
          ResultChar = 0;
          break;
        }
        ++ThisTokBuf; // Consumed one hex digit.
        
        assert(0 && "hex escape: unimp!");
        break;
      case '0': case '1': case '2': case '3':
      case '4': case '5': case '6': case '7':
        // Octal escapes.
        assert(0 && "octal escape: unimp!");
        break;
        
      // Otherwise, these are not valid escapes.
      case '(': case '{': case '[': case '%':
        // GCC accepts these as extensions.  We warn about them as such though.
        if (!PP.getLangOptions().NoExtensions) {
          PP.Diag(StringToks[i], diag::ext_nonstandard_escape,
                  std::string()+(char)ResultChar);
          break;
        }
        // FALL THROUGH.
      default:
        if (isgraph(ThisTokBuf[0])) {
          PP.Diag(StringToks[i], diag::ext_unknown_escape,
                  std::string()+(char)ResultChar);
        } else {
          PP.Diag(StringToks[i], diag::ext_unknown_escape,
                  "x"+utohexstr(ResultChar));
        }
      }

      // Note: our internal rep of wide char tokens is always little-endian.
      *ResultPtr++ = ResultChar & 0xFF;
      
      if (AnyWide) {
        for (unsigned i = 1, e = wchar_tByteWidth; i != e; ++i)
          *ResultPtr++ = ResultChar >> i*8;
      }
    }
  }
  
  // Add zero terminator.
  *ResultPtr = 0;
  if (AnyWide) {
    for (unsigned i = 1, e = wchar_tByteWidth; i != e; ++i)
      *ResultPtr++ = 0;
  }
  
  // Hand this off to the Actions.
  ExprResult Res = Actions.ParseStringExpr(ResultBuf, ResultPtr-ResultBuf,
                                           AnyWide,
                                           &StringToks[0], StringToks.size());
  
  // If either buffer was heap allocated, release it now.
  if (MaxTokenLength > 512) free(TokenBuf);
  if (SizeBound > 512) free(ResultBuf);
  
  return Res;
}

