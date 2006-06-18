//===--- PPExpressions.cpp - Preprocessor Expression Evaluation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Preprocessor::EvaluateDirectiveExpression method.
//
//===----------------------------------------------------------------------===//
//
// FIXME: implement testing for asserts.
// FIXME: Parse integer constants correctly.  Reject 123.0, etc.
// FIXME: Track signed/unsigned correctly.
// FIXME: Track and report integer overflow correctly.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Basic/Diagnostic.h"
using namespace llvm;
using namespace clang;

/// EvaluateDirectiveExpression - Evaluate an integer constant expression that
/// may occur after a #if or #elif directive.  Sets Result to the result of
/// the expression.  Returns false normally, true if lexing must be aborted.
///
/// MinPrec is the minimum precedence that this range of the expression is
/// allowed to include.
bool Preprocessor::EvaluateDirectiveExpression(bool &Result) {
  // Peek ahead one token.
  LexerToken Tok;
  if (Lex(Tok)) return true;

  // In error cases, bail out with false value.
  Result = false;
  
  bool StopParse = false;
  
  int ResVal = 0;
  if (EvaluateValue(ResVal, Tok, StopParse)) {
    // Skip the rest of the macro line.
    if (!StopParse && Tok.getKind() != tok::eom)
      StopParse |= DiscardUntilEndOfDirective();
    return StopParse;
  }
  
  if (EvaluateDirectiveSubExpr(ResVal, 1, Tok, StopParse)) {
    // Skip the rest of the macro line.
    if (!StopParse && Tok.getKind() != tok::eom)
      StopParse |= DiscardUntilEndOfDirective();
    return StopParse;
  }
  
  // If we aren't at the tok::eom token, something bad happened, like an extra
  // ')' token.
  if (Tok.getKind() != tok::eom) {
    return Diag(Tok, diag::err_pp_expected_eol) ||
           DiscardUntilEndOfDirective();
  }
  
  Result = ResVal != 0;
  return false;
}

/// EvaluateValue - Evaluate the token PeekTok (and any others needed) and
/// return the computed value in Result.  Return true if there was an error
/// parsing, setting StopParse if parsing should be aborted.
bool Preprocessor::EvaluateValue(int &Result, LexerToken &PeekTok, 
                                 bool &StopParse) {
  Result = 0;
  
  // If this token's spelling is a pp-identifier, check to see if it is
  // 'defined' or if it is a macro.  Note that we check here because many
  // keywords are pp-identifiers, so we can't check the kind.
  if (const IdentifierTokenInfo *II = PeekTok.getIdentifierInfo()) {
    // If this identifier isn't 'defined' and it wasn't macro expanded, it turns
    // into a simple 0.
    if (strcmp(II->getName(), "defined")) {
      Result = 0;
      return (StopParse = Lex(PeekTok));
    }

    // Handle "defined X" and "defined(X)".
    assert(!DisableMacroExpansion &&
           "How could macro exp already be disabled?");
    // Turn off macro expansion.
    DisableMacroExpansion = true;

    // Get the next token.
    if ((StopParse = Lex(PeekTok))) return true;

    // Two options, it can either be a pp-identifier or a (.
    bool InParens = false;
    if (PeekTok.getKind() == tok::l_paren) {
      // Found a paren, remember we saw it and skip it.
      InParens = true;
      if ((StopParse = Lex(PeekTok))) return true;
    }
    
    // If we don't have a pp-identifier now, this is an error.
    if ((II = PeekTok.getIdentifierInfo()) == 0) {
      DisableMacroExpansion = false;
      StopParse = Diag(PeekTok, diag::err_pp_defined_requires_identifier);
      return true;
    }
    
    // Otherwise, we got an identifier, is it defined to something?
    Result = II->getMacroInfo() != 0;

    // Consume identifier.
    if ((StopParse = Lex(PeekTok))) return true;

    // If we are in parens, ensure we have a trailing ).
    if (InParens) {
      if (PeekTok.getKind() != tok::r_paren) {
        StopParse = Diag(PeekTok, diag::err_pp_missing_rparen);
        return true;
      }
      // Consume the ).
      if ((StopParse = Lex(PeekTok))) return true;
    }
    
    DisableMacroExpansion = false;
    return false;
  }
  
  switch (PeekTok.getKind()) {
  default:  // Non-value token.
    StopParse = Diag(PeekTok, diag::err_pp_expr_bad_token);
    return true;
  case tok::eom:
  case tok::r_paren:
    // If there is no expression, report and exit.
    StopParse = Diag(PeekTok, diag::err_pp_expected_value_in_expr);
    return true;
  case tok::numeric_constant: {
    // FIXME: faster.  FIXME: track signs.
    std::string Spell = Lexer::getSpelling(PeekTok, getLangOptions());
    // FIXME: COMPUTE integer constants CORRECTLY.
    Result = atoi(Spell.c_str());
    return (StopParse = Lex(PeekTok));
  }
  case tok::l_paren:
    if (StopParse = Lex(PeekTok)) return true;  // Eat the (.
    // Parse the value.
    if (EvaluateValue(Result, PeekTok, StopParse)) return true;
      
    // If there are any binary operators involved, parse them.
    if (EvaluateDirectiveSubExpr(Result, 1, PeekTok, StopParse))
      return StopParse;

    if (PeekTok.getKind() != tok::r_paren) {
      StopParse = Diag(PeekTok, diag::err_pp_expected_rparen);
      return true;
    }
    if (StopParse = Lex(PeekTok)) return true;  // Eat the ).
    return false;
 
  case tok::plus:
    // Unary plus doesn't modify the value.
    if (StopParse = Lex(PeekTok)) return true;
    return EvaluateValue(Result, PeekTok, StopParse);
  case tok::minus:
    if (StopParse = Lex(PeekTok)) return true;
    if (EvaluateValue(Result, PeekTok, StopParse)) return true;
    Result = -Result;
    return false;
    
  case tok::tilde:
    if (StopParse = Lex(PeekTok)) return true;
    if (EvaluateValue(Result, PeekTok, StopParse)) return true;
    Result = ~Result;
    return false;
    
  case tok::exclaim:
    if (StopParse = Lex(PeekTok)) return true;
    if (EvaluateValue(Result, PeekTok, StopParse)) return true;
    Result = !Result;
    return false;
    
  // FIXME: Handle #assert
  }
}



/// getPrecedence - Return the precedence of the specified binary operator
/// token.  This returns:
///   ~0 - Invalid token.
///   15 - *,/,%
///   14 - -,+
///   13 - <<,>>
///   12 - >=, <=, >, <
///   11 - ==, !=
///   10 - <?, >?           min, max (GCC extensions)
///    9 - &
///    8 - ^
///    7 - |
///    6 - &&
///    5 - ||
///    4 - ?
///    3 - :
///    0 - eom, )
static unsigned getPrecedence(tok::TokenKind Kind) {
  switch (Kind) {
  default: return ~0U;
  case tok::percent:
  case tok::slash:
  case tok::star:                 return 15;
  case tok::plus:
  case tok::minus:                return 14;
  case tok::lessless:
  case tok::greatergreater:       return 13;
  case tok::lessequal:
  case tok::less:
  case tok::greaterequal:
  case tok::greater:              return 12;
  case tok::exclaimequal:
  case tok::equalequal:           return 11;
  case tok::lessquestion:
  case tok::greaterquestion:      return 10;
  case tok::amp:                  return 9;
  case tok::caret:                return 8;
  case tok::pipe:                 return 7;
  case tok::ampamp:               return 6;
  case tok::pipepipe:             return 5;
  case tok::question:             return 4;
  case tok::colon:                return 3;
  case tok::comma:                return 2;
  case tok::r_paren:              return 0;   // Lowest priority, end of expr.
  case tok::eom:                  return 0;   // Lowest priority, end of macro.
  }
}


/// EvaluateDirectiveSubExpr - Evaluate the subexpression whose first token is
/// PeekTok, and whose precedence is PeekPrec.
bool Preprocessor::EvaluateDirectiveSubExpr(int &LHS, unsigned MinPrec,
                                            LexerToken &PeekTok,
                                            bool &StopParse) {
  unsigned PeekPrec = getPrecedence(PeekTok.getKind());
  // If this token isn't valid, report the error.
  if (PeekPrec == ~0U) {
    StopParse = Diag(PeekTok, diag::err_pp_expr_bad_token);
    return true;
  }
  
  while (1) {
    // If this token has a lower precedence than we are allowed to parse, return
    // it so that higher levels of the recursion can parse it.
    if (PeekPrec < MinPrec)
      return false;
    
    tok::TokenKind Operator = PeekTok.getKind();

    // Consume the operator, saving the operator token for error reporting.
    LexerToken OpToken = PeekTok;
    if (StopParse = Lex(PeekTok)) return true;

    int RHS;
    // Parse the RHS of the operator.
    if (EvaluateValue(RHS, PeekTok, StopParse)) return true;

    // Remember the precedence of this operator and get the precedence of the
    // operator immediately to the right of the RHS.
    unsigned ThisPrec = PeekPrec;
    PeekPrec = getPrecedence(PeekTok.getKind());

    // If this token isn't valid, report the error.
    if (PeekPrec == ~0U) {
      StopParse = Diag(PeekTok, diag::err_pp_expr_bad_token);
      return true;
    }
    
    bool isRightAssoc = Operator == tok::question;
    
    // Get the precedence of the operator to the right of the RHS.  If it binds
    // more tightly with RHS than we do, evaluate it completely first.
    if (ThisPrec < PeekPrec ||
        (ThisPrec == PeekPrec && isRightAssoc)) {
      if (EvaluateDirectiveSubExpr(RHS, ThisPrec+1, PeekTok, StopParse))
        return true;
      PeekPrec = getPrecedence(PeekTok.getKind());
    }
    assert(PeekPrec <= ThisPrec && "Recursion didn't work!");
    
    switch (Operator) {
    default: assert(0 && "Unknown operator token!");
    case tok::percent:
      if (RHS == 0) {
        StopParse = Diag(OpToken, diag::err_pp_remainder_by_zero);
        return true;
      }
      LHS %= RHS;
      break;
    case tok::slash:
      if (RHS == 0) {
        StopParse = Diag(OpToken, diag::err_pp_division_by_zero);
        return true;
      }
      LHS /= RHS;
      break;
    case tok::star :           LHS *= RHS; break;
    case tok::lessless:        LHS << RHS; break;  // FIXME: shift amt overflow?
    case tok::greatergreater:  LHS >> RHS; break;  // FIXME: signed vs unsigned
    case tok::plus :           LHS += RHS; break;
    case tok::minus:           LHS -= RHS; break;
    case tok::lessequal:       LHS = LHS <= RHS; break;
    case tok::less:            LHS = LHS <  RHS; break;
    case tok::greaterequal:    LHS = LHS >= RHS; break;
    case tok::greater:         LHS = LHS >  RHS; break;
    case tok::exclaimequal:    LHS = LHS != RHS; break;
    case tok::equalequal:      LHS = LHS == RHS; break;
    case tok::lessquestion:    // Deprecation warning emitted by the lexer.
      LHS = std::min(LHS, RHS);
      break; 
    case tok::greaterquestion: // Deprecation warning emitted by the lexer.
      LHS = std::max(LHS, RHS);
      break;
    case tok::amp:             LHS &= RHS; break;
    case tok::caret:           LHS ^= RHS; break;
    case tok::pipe:            LHS |= RHS; break;
    case tok::ampamp:          LHS = LHS && RHS; break;
    case tok::pipepipe:        LHS = LHS || RHS; break;
    case tok::comma:
      if ((StopParse = Diag(OpToken, diag::ext_pp_comma_expr)))
        return true;
      LHS = RHS; // LHS = LHS,RHS -> RHS.
      break; 
    case tok::question: {
      // Parse the : part of the expression.
      if (PeekTok.getKind() != tok::colon) {
        StopParse = Diag(OpToken, diag::err_pp_question_without_colon);
        return true;
      }
      // Consume the :.
      if (StopParse = Lex(PeekTok)) return true;

      // Evaluate the value after the :.
      int AfterColonVal = 0;
      if (EvaluateValue(AfterColonVal, PeekTok, StopParse)) return true;

      // Parse anything after the : RHS that has a higher precedence than ?.
      if (EvaluateDirectiveSubExpr(AfterColonVal, ThisPrec+1,
                                   PeekTok, StopParse))
        return true;
      
      // Now that we have the condition, the LHS and the RHS of the :, evaluate.
      LHS = LHS ? RHS : AfterColonVal;
      
      // Figure out the precedence of the token after the : part.
      PeekPrec = getPrecedence(PeekTok.getKind());
      break;
    }
    case tok::colon:
      // Don't allow :'s to float around without being part of ?: exprs.
      StopParse = Diag(OpToken, diag::err_pp_colon_without_question);
      return true;
    }
  }
  
  return false;
}
