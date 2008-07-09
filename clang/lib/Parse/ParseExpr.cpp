//===--- ParseExpr.cpp - Expression Parsing -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallString.h"
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
    Equality       = 9,    // ==, !=
    Relational     = 10,   //  >=, <=, >, <
    Shift          = 11,   // <<, >>
    Additive       = 12,   // -, +
    Multiplicative = 13    // *, /, %
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
/// [C++]   throw-expression [C++ 15]
///
///       assignment-operator: one of
///         = *= /= %= += -= <<= >>= &= ^= |=
///
///       expression: [C99 6.5.17]
///         assignment-expression
///         expression ',' assignment-expression
///
Parser::ExprResult Parser::ParseExpression() {
  if (Tok.is(tok::kw_throw))
    return ParseThrowExpression();

  ExprResult LHS = ParseCastExpression(false);
  if (LHS.isInvalid) return LHS;
  
  return ParseRHSOfBinaryExpression(LHS, prec::Comma);
}

/// This routine is called when the '@' is seen and consumed. 
/// Current token is an Identifier and is not a 'try'. This
/// routine is necessary to disambiguate @try-statement from,
/// for example, @encode-expression.
///
Parser::ExprResult Parser::ParseExpressionWithLeadingAt(SourceLocation AtLoc) {
  ExprResult LHS = ParseObjCAtExpression(AtLoc);
  if (LHS.isInvalid) return LHS;
 
  return ParseRHSOfBinaryExpression(LHS, prec::Comma);
}

/// ParseAssignmentExpression - Parse an expr that doesn't include commas.
///
Parser::ExprResult Parser::ParseAssignmentExpression() {
  if (Tok.is(tok::kw_throw))
    return ParseThrowExpression();

  ExprResult LHS = ParseCastExpression(false);
  if (LHS.isInvalid) return LHS;
  
  return ParseRHSOfBinaryExpression(LHS, prec::Assignment);
}

/// ParseAssignmentExprWithObjCMessageExprStart - Parse an assignment expression
/// where part of an objc message send has already been parsed.  In this case
/// LBracLoc indicates the location of the '[' of the message send, and either
/// ReceiverName or ReceiverExpr is non-null indicating the receiver of the
/// message.
///
/// Since this handles full assignment-expression's, it handles postfix
/// expressions and other binary operators for these expressions as well.
Parser::ExprResult 
Parser::ParseAssignmentExprWithObjCMessageExprStart(SourceLocation LBracLoc,
                                                   IdentifierInfo *ReceiverName,
                                                    ExprTy *ReceiverExpr) {
  ExprResult R = ParseObjCMessageExpressionBody(LBracLoc, ReceiverName,
                                                ReceiverExpr);
  if (R.isInvalid) return R;
  R = ParsePostfixExpressionSuffix(R);
  if (R.isInvalid) return R;
  return ParseRHSOfBinaryExpression(R, 2);
}


Parser::ExprResult Parser::ParseConstantExpression() {
  ExprResult LHS = ParseCastExpression(false);
  if (LHS.isInvalid) return LHS;
  
  return ParseRHSOfBinaryExpression(LHS, prec::Conditional);
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
    Token OpToken = Tok;
    ConsumeToken();
    
    // Special case handling for the ternary operator.
    ExprResult TernaryMiddle(true);
    if (NextTokPrec == prec::Conditional) {
      if (Tok.isNot(tok::colon)) {
        // Handle this production specially:
        //   logical-OR-expression '?' expression ':' conditional-expression
        // In particular, the RHS of the '?' is 'expression', not
        // 'logical-OR-expression' as we might expect.
        TernaryMiddle = ParseExpression();
        if (TernaryMiddle.isInvalid) {
          Actions.DeleteExpr(LHS.Val);
          return TernaryMiddle;
        }
      } else {
        // Special case handling of "X ? Y : Z" where Y is empty:
        //   logical-OR-expression '?' ':' conditional-expression   [GNU]
        TernaryMiddle = ExprResult(false);
        Diag(Tok, diag::ext_gnu_conditional_expr);
      }
      
      if (Tok.isNot(tok::colon)) {
        Diag(Tok, diag::err_expected_colon);
        Diag(OpToken, diag::err_matching, "?");
        Actions.DeleteExpr(LHS.Val);
        Actions.DeleteExpr(TernaryMiddle.Val);
        return ExprResult(true);
      }
      
      // Eat the colon.
      ColonLoc = ConsumeToken();
    }
    
    // Parse another leaf here for the RHS of the operator.
    ExprResult RHS = ParseCastExpression(false);
    if (RHS.isInvalid) {
      Actions.DeleteExpr(LHS.Val);
      Actions.DeleteExpr(TernaryMiddle.Val);
      return RHS;
    }

    // Remember the precedence of this operator and get the precedence of the
    // operator immediately to the right of the RHS.
    unsigned ThisPrec = NextTokPrec;
    NextTokPrec = getBinOpPrecedence(Tok.getKind());

    // Assignment and conditional expressions are right-associative.
    bool isRightAssoc = ThisPrec == prec::Conditional ||
                        ThisPrec == prec::Assignment;

    // Get the precedence of the operator to the right of the RHS.  If it binds
    // more tightly with RHS than we do, evaluate it completely first.
    if (ThisPrec < NextTokPrec ||
        (ThisPrec == NextTokPrec && isRightAssoc)) {
      // If this is left-associative, only parse things on the RHS that bind
      // more tightly than the current operator.  If it is left-associative, it
      // is okay, to bind exactly as tightly.  For example, compile A=B=C=D as
      // A=(B=(C=D)), where each paren is a level of recursion here.
      RHS = ParseRHSOfBinaryExpression(RHS, ThisPrec + !isRightAssoc);
      if (RHS.isInvalid) {
        Actions.DeleteExpr(LHS.Val);
        Actions.DeleteExpr(TernaryMiddle.Val);
        return RHS;
      }

      NextTokPrec = getBinOpPrecedence(Tok.getKind());
    }
    assert(NextTokPrec <= ThisPrec && "Recursion didn't work!");
  
    if (!LHS.isInvalid) {
      // Combine the LHS and RHS into the LHS (e.g. build AST).
      if (TernaryMiddle.isInvalid)
        LHS = Actions.ActOnBinOp(OpToken.getLocation(), OpToken.getKind(),
                                 LHS.Val, RHS.Val);
      else
        LHS = Actions.ActOnConditionalOp(OpToken.getLocation(), ColonLoc,
                                         LHS.Val, TernaryMiddle.Val, RHS.Val);
    } else {
      // We had a semantic error on the LHS.  Just free the RHS and continue.
      Actions.DeleteExpr(TernaryMiddle.Val);
      Actions.DeleteExpr(RHS.Val);
    }
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
/// [C++]   boolean-literal  [C++ 2.13.5]
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
/// [OBJC]  '[' objc-message-expr ']'    
/// [OBJC]  '@selector' '(' objc-selector-arg ')'
/// [OBJC]  '@protocol' '(' identifier ')'             
/// [OBJC]  '@encode' '(' type-name ')'                
/// [OBJC]  objc-string-literal
/// [C++]   'const_cast' '<' type-name '>' '(' expression ')'       [C++ 5.2p1]
/// [C++]   'dynamic_cast' '<' type-name '>' '(' expression ')'     [C++ 5.2p1]
/// [C++]   'reinterpret_cast' '<' type-name '>' '(' expression ')' [C++ 5.2p1]
/// [C++]   'static_cast' '<' type-name '>' '(' expression ')'      [C++ 5.2p1]
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
  switch (SavedKind) {
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
        Res = Actions.ActOnCastExpr(LParenLoc, CastTy, RParenLoc, Res.Val);
      return Res;
    }
      
    // These can be followed by postfix-expr pieces.
    return ParsePostfixExpressionSuffix(Res);
  }
    
    // primary-expression
  case tok::numeric_constant:
    // constant: integer-constant
    // constant: floating-constant
    
    Res = Actions.ActOnNumericConstant(Tok);
    ConsumeToken();
    
    // These can be followed by postfix-expr pieces.
    return ParsePostfixExpressionSuffix(Res);

  case tok::kw_true:
  case tok::kw_false:
    return ParseCXXBoolLiteral();

  case tok::identifier: {      // primary-expression: identifier
                               // constant: enumeration-constant
    // Consume the identifier so that we can see if it is followed by a '('.
    // Function designators are allowed to be undeclared (C99 6.5.1p2), so we
    // need to know whether or not this identifier is a function designator or
    // not.
    IdentifierInfo &II = *Tok.getIdentifierInfo();
    SourceLocation L = ConsumeToken();
    Res = Actions.ActOnIdentifierExpr(CurScope, L, II, Tok.is(tok::l_paren));
    // These can be followed by postfix-expr pieces.
    return ParsePostfixExpressionSuffix(Res);
  }
  case tok::char_constant:     // constant: character-constant
    Res = Actions.ActOnCharacterConstant(Tok);
    ConsumeToken();
    // These can be followed by postfix-expr pieces.
    return ParsePostfixExpressionSuffix(Res);    
  case tok::kw___func__:       // primary-expression: __func__ [C99 6.4.2.2]
  case tok::kw___FUNCTION__:   // primary-expression: __FUNCTION__ [GNU]
  case tok::kw___PRETTY_FUNCTION__:  // primary-expression: __P..Y_F..N__ [GNU]
    Res = Actions.ActOnPreDefinedExpr(Tok.getLocation(), SavedKind);
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
  case tok::kw___builtin_overload:
  case tok::kw___builtin_types_compatible_p:
    return ParseBuiltinPrimaryExpression();
  case tok::plusplus:      // unary-expression: '++' unary-expression
  case tok::minusminus: {  // unary-expression: '--' unary-expression
    SourceLocation SavedLoc = ConsumeToken();
    Res = ParseCastExpression(true);
    if (!Res.isInvalid)
      Res = Actions.ActOnUnaryOp(SavedLoc, SavedKind, Res.Val);
    return Res;
  }
  case tok::amp:           // unary-expression: '&' cast-expression
  case tok::star:          // unary-expression: '*' cast-expression
  case tok::plus:          // unary-expression: '+' cast-expression
  case tok::minus:         // unary-expression: '-' cast-expression
  case tok::tilde:         // unary-expression: '~' cast-expression
  case tok::exclaim:       // unary-expression: '!' cast-expression
  case tok::kw___real:     // unary-expression: '__real' cast-expression [GNU]
  case tok::kw___imag: {   // unary-expression: '__imag' cast-expression [GNU]
    SourceLocation SavedLoc = ConsumeToken();
    Res = ParseCastExpression(false);
    if (!Res.isInvalid)
      Res = Actions.ActOnUnaryOp(SavedLoc, SavedKind, Res.Val);
    return Res;
  }    
      
  case tok::kw___extension__:{//unary-expression:'__extension__' cast-expr [GNU]
    // __extension__ silences extension warnings in the subexpression.
    bool SavedExtWarn = Diags.getWarnOnExtensions();
    Diags.setWarnOnExtensions(false);
    SourceLocation SavedLoc = ConsumeToken();
    Res = ParseCastExpression(false);
    if (!Res.isInvalid)
      Res = Actions.ActOnUnaryOp(SavedLoc, SavedKind, Res.Val);
    Diags.setWarnOnExtensions(SavedExtWarn);
    return Res;
  }
  case tok::kw_sizeof:     // unary-expression: 'sizeof' unary-expression
                           // unary-expression: 'sizeof' '(' type-name ')'
  case tok::kw___alignof:  // unary-expression: '__alignof' unary-expression
                           // unary-expression: '__alignof' '(' type-name ')'
    return ParseSizeofAlignofExpression();
  case tok::ampamp: {      // unary-expression: '&&' identifier
    SourceLocation AmpAmpLoc = ConsumeToken();
    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected_ident);
      return ExprResult(true);
    }
    
    Diag(AmpAmpLoc, diag::ext_gnu_address_of_label);
    Res = Actions.ActOnAddrLabel(AmpAmpLoc, Tok.getLocation(),
                                 Tok.getIdentifierInfo());
    ConsumeToken();
    return Res;
  }
  case tok::kw_const_cast:
  case tok::kw_dynamic_cast:
  case tok::kw_reinterpret_cast:
  case tok::kw_static_cast:
    return ParseCXXCasts();
  case tok::kw_this:
    return ParseCXXThis();
  case tok::at: {
    SourceLocation AtLoc = ConsumeToken();
    return ParseObjCAtExpression(AtLoc);
  }
  case tok::l_square:
    // These can be followed by postfix-expr pieces.
    if (getLang().ObjC1)
      return ParsePostfixExpressionSuffix(ParseObjCMessageExpression());
    // FALL THROUGH.
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
      
      if (!LHS.isInvalid && !Idx.isInvalid && Tok.is(tok::r_square))
        LHS = Actions.ActOnArraySubscriptExpr(LHS.Val, Loc, Idx.Val, RLoc);
      else 
        LHS = ExprResult(true);

      // Match the ']'.
      MatchRHSPunctuation(tok::r_square, Loc);
      break;
    }
      
    case tok::l_paren: {   // p-e: p-e '(' argument-expression-list[opt] ')'
      llvm::SmallVector<ExprTy*, 8> ArgExprs;
      llvm::SmallVector<SourceLocation, 8> CommaLocs;
      
      Loc = ConsumeParen();
      
      if (Tok.isNot(tok::r_paren)) {
        while (1) {
          ExprResult ArgExpr = ParseAssignmentExpression();
          if (ArgExpr.isInvalid) {
            SkipUntil(tok::r_paren);
            return ExprResult(true);
          } else
            ArgExprs.push_back(ArgExpr.Val);
          
          if (Tok.isNot(tok::comma))
            break;
          // Move to the next argument, remember where the comma was.
          CommaLocs.push_back(ConsumeToken());
        }
      }
        
      // Match the ')'.
      if (!LHS.isInvalid && Tok.is(tok::r_paren)) {
        assert((ArgExprs.size() == 0 || ArgExprs.size()-1 == CommaLocs.size())&&
               "Unexpected number of commas!");
        LHS = Actions.ActOnCallExpr(LHS.Val, Loc, &ArgExprs[0], ArgExprs.size(),
                                    &CommaLocs[0], Tok.getLocation());
      }
      
      MatchRHSPunctuation(tok::r_paren, Loc);
      break;
    }
    case tok::arrow:       // postfix-expression: p-e '->' identifier
    case tok::period: {    // postfix-expression: p-e '.' identifier
      tok::TokenKind OpKind = Tok.getKind();
      SourceLocation OpLoc = ConsumeToken();  // Eat the "." or "->" token.
      
      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident);
        return ExprResult(true);
      }
      
      if (!LHS.isInvalid)
        LHS = Actions.ActOnMemberReferenceExpr(LHS.Val, OpLoc, OpKind,
                                               Tok.getLocation(),
                                               *Tok.getIdentifierInfo());
      ConsumeToken();
      break;
    }
    case tok::plusplus:    // postfix-expression: postfix-expression '++'
    case tok::minusminus:  // postfix-expression: postfix-expression '--'
      if (!LHS.isInvalid)
        LHS = Actions.ActOnPostfixUnaryOp(Tok.getLocation(), Tok.getKind(),
                                          LHS.Val);
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
  assert((Tok.is(tok::kw_sizeof) || Tok.is(tok::kw___alignof)) &&
         "Not a sizeof/alignof expression!");
  Token OpTok = Tok;
  ConsumeToken();
  
  // If the operand doesn't start with an '(', it must be an expression.
  ExprResult Operand;
  if (Tok.isNot(tok::l_paren)) {
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
    if (ExprType == CastExpr)
      return Actions.ActOnSizeOfAlignOfTypeExpr(OpTok.getLocation(),
                                                OpTok.is(tok::kw_sizeof),
                                                LParenLoc, CastTy, RParenLoc);
    
    // If this is a parenthesized expression, it is the start of a 
    // unary-expression, but doesn't include any postfix pieces.  Parse these
    // now if present.
    Operand = ParsePostfixExpressionSuffix(Operand);
  }
  
  // If we get here, the operand to the sizeof/alignof was an expresion.
  if (!Operand.isInvalid)
    Operand = Actions.ActOnUnaryOp(OpTok.getLocation(), OpTok.getKind(),
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
/// [CLANG] '__builtin_overload' '(' expr (',' expr)* ')'
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
  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after, BuiltinII->getName());
    return ExprResult(true);
  }
  
  SourceLocation LParenLoc = ConsumeParen();
  // TODO: Build AST.

  switch (T) {
  default: assert(0 && "Not a builtin primary expression!");
  case tok::kw___builtin_va_arg: {
    ExprResult Expr = ParseAssignmentExpression();
    if (Expr.isInvalid) {
      SkipUntil(tok::r_paren);
      return Res;
    }

    if (ExpectAndConsume(tok::comma, diag::err_expected_comma, "",tok::r_paren))
      return ExprResult(true);

    TypeTy *Ty = ParseTypeName();
    
    if (Tok.isNot(tok::r_paren)) {
      Diag(Tok, diag::err_expected_rparen);
      return ExprResult(true);
    }
    Res = Actions.ActOnVAArg(StartLoc, Expr.Val, Ty, ConsumeParen());
    break;
  }
  case tok::kw___builtin_offsetof: {
    SourceLocation TypeLoc = Tok.getLocation();
    TypeTy *Ty = ParseTypeName();

    if (ExpectAndConsume(tok::comma, diag::err_expected_comma, "",tok::r_paren))
      return ExprResult(true);
    
    // We must have at least one identifier here.
    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected_ident);
      SkipUntil(tok::r_paren);
      return true;
    }
    
    // Keep track of the various subcomponents we see.
    llvm::SmallVector<Action::OffsetOfComponent, 4> Comps;
    
    Comps.push_back(Action::OffsetOfComponent());
    Comps.back().isBrackets = false;
    Comps.back().U.IdentInfo = Tok.getIdentifierInfo();
    Comps.back().LocStart = Comps.back().LocEnd = ConsumeToken();

    while (1) {
      if (Tok.is(tok::period)) {
        // offsetof-member-designator: offsetof-member-designator '.' identifier
        Comps.push_back(Action::OffsetOfComponent());
        Comps.back().isBrackets = false;
        Comps.back().LocStart = ConsumeToken();
        
        if (Tok.isNot(tok::identifier)) {
          Diag(Tok, diag::err_expected_ident);
          SkipUntil(tok::r_paren);
          return true;
        }
        Comps.back().U.IdentInfo = Tok.getIdentifierInfo();
        Comps.back().LocEnd = ConsumeToken();
        
      } else if (Tok.is(tok::l_square)) {
        // offsetof-member-designator: offsetof-member-design '[' expression ']'
        Comps.push_back(Action::OffsetOfComponent());
        Comps.back().isBrackets = true;
        Comps.back().LocStart = ConsumeBracket();
        Res = ParseExpression();
        if (Res.isInvalid) {
          SkipUntil(tok::r_paren);
          return Res;
        }
        Comps.back().U.E = Res.Val;

        Comps.back().LocEnd =
          MatchRHSPunctuation(tok::r_square, Comps.back().LocStart);
      } else if (Tok.is(tok::r_paren)) {
        Res = Actions.ActOnBuiltinOffsetOf(StartLoc, TypeLoc, Ty, &Comps[0],
                                           Comps.size(), ConsumeParen());
        break;
      } else {
        // Error occurred.
        return ExprResult(true);
      }
    }
    break;
  }
  case tok::kw___builtin_choose_expr: {
    ExprResult Cond = ParseAssignmentExpression();
    if (Cond.isInvalid) {
      SkipUntil(tok::r_paren);
      return Cond;
    }
    if (ExpectAndConsume(tok::comma, diag::err_expected_comma, "",tok::r_paren))
      return ExprResult(true);
    
    ExprResult Expr1 = ParseAssignmentExpression();
    if (Expr1.isInvalid) {
      SkipUntil(tok::r_paren);
      return Expr1;
    }    
    if (ExpectAndConsume(tok::comma, diag::err_expected_comma, "",tok::r_paren))
      return ExprResult(true);
    
    ExprResult Expr2 = ParseAssignmentExpression();
    if (Expr2.isInvalid) {
      SkipUntil(tok::r_paren);
      return Expr2;
    }    
    if (Tok.isNot(tok::r_paren)) {
      Diag(Tok, diag::err_expected_rparen);
      return ExprResult(true);
    }
    Res = Actions.ActOnChooseExpr(StartLoc, Cond.Val, Expr1.Val, Expr2.Val,
                                  ConsumeParen());
    break;
  }
  case tok::kw___builtin_overload: {
    llvm::SmallVector<ExprTy*, 8> ArgExprs;
    llvm::SmallVector<SourceLocation, 8> CommaLocs;

    // For each iteration through the loop look for assign-expr followed by a
    // comma.  If there is no comma, break and attempt to match r-paren.
    if (Tok.isNot(tok::r_paren)) {
      while (1) {
        ExprResult ArgExpr = ParseAssignmentExpression();
        if (ArgExpr.isInvalid) {
          SkipUntil(tok::r_paren);
          return ExprResult(true);
        } else
          ArgExprs.push_back(ArgExpr.Val);
        
        if (Tok.isNot(tok::comma))
          break;
        // Move to the next argument, remember where the comma was.
        CommaLocs.push_back(ConsumeToken());
      }
    }
    
    // Attempt to consume the r-paren
    if (Tok.isNot(tok::r_paren)) {
      Diag(Tok, diag::err_expected_rparen);
      SkipUntil(tok::r_paren);
      return ExprResult(true);
    }
    Res = Actions.ActOnOverloadExpr(&ArgExprs[0], ArgExprs.size(), 
                                    &CommaLocs[0], StartLoc, ConsumeParen());
    break;
  }
  case tok::kw___builtin_types_compatible_p:
    TypeTy *Ty1 = ParseTypeName();
    
    if (ExpectAndConsume(tok::comma, diag::err_expected_comma, "",tok::r_paren))
      return ExprResult(true);
    
    TypeTy *Ty2 = ParseTypeName();
    
    if (Tok.isNot(tok::r_paren)) {
      Diag(Tok, diag::err_expected_rparen);
      return ExprResult(true);
    }
    Res = Actions.ActOnTypesCompatibleExpr(StartLoc, Ty1, Ty2, ConsumeParen());
    break;
  }      
  
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
  assert(Tok.is(tok::l_paren) && "Not a paren expr!");
  SourceLocation OpenLoc = ConsumeParen();
  ExprResult Result(true);
  CastTy = 0;
  
  if (ExprType >= CompoundStmt && Tok.is(tok::l_brace)) {
    Diag(Tok, diag::ext_gnu_statement_expr);
    Parser::StmtResult Stmt = ParseCompoundStatement(true);
    ExprType = CompoundStmt;
    
    // If the substmt parsed correctly, build the AST node.
    if (!Stmt.isInvalid && Tok.is(tok::r_paren))
      Result = Actions.ActOnStmtExpr(OpenLoc, Stmt.Val, Tok.getLocation());
    
  } else if (ExprType >= CompoundLiteral && isTypeSpecifierQualifier()) {
    // Otherwise, this is a compound literal expression or cast expression.
    TypeTy *Ty = ParseTypeName();

    // Match the ')'.
    if (Tok.is(tok::r_paren))
      RParenLoc = ConsumeParen();
    else
      MatchRHSPunctuation(tok::r_paren, OpenLoc);
    
    if (Tok.is(tok::l_brace)) {
      if (!getLang().C99)   // Compound literals don't exist in C90.
        Diag(OpenLoc, diag::ext_c99_compound_literal);
      Result = ParseInitializer();
      ExprType = CompoundLiteral;
      if (!Result.isInvalid)
        return Actions.ActOnCompoundLiteral(OpenLoc, Ty, RParenLoc, Result.Val);
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
    if (!Result.isInvalid && Tok.is(tok::r_paren))
      Result = Actions.ActOnParenExpr(OpenLoc, Tok.getLocation(), Result.Val);
  }
  
  // Match the ')'.
  if (Result.isInvalid)
    SkipUntil(tok::r_paren);
  else {
    if (Tok.is(tok::r_paren))
      RParenLoc = ConsumeParen();
    else
      MatchRHSPunctuation(tok::r_paren, OpenLoc);
  }
  
  return Result;
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
  llvm::SmallVector<Token, 4> StringToks;
  
  do {
    StringToks.push_back(Tok);
    ConsumeStringToken();
  } while (isTokenStringLiteral());

  // Pass the set of string tokens, ready for concatenation, to the actions.
  return Actions.ActOnStringLiteral(&StringToks[0], StringToks.size());
}
