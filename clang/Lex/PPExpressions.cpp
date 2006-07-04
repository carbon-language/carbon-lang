//===--- PPExpressions.cpp - Preprocessor Expression Evaluation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Preprocessor::EvaluateDirectiveExpression method,
// which parses and evaluates integer constant expressions for #if directives.
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
#include "clang/Lex/MacroInfo.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Basic/Diagnostic.h"
using namespace llvm;
using namespace clang;

/// EvaluateDirectiveExpression - Evaluate an integer constant expression that
/// may occur after a #if or #elif directive.  If the expression is equivalent
/// to "!defined(X)" return X in IfNDefMacro.
bool Preprocessor::
EvaluateDirectiveExpression(IdentifierInfo *&IfNDefMacro) {
  // Peek ahead one token.
  LexerToken Tok;
  Lex(Tok);

  int ResVal = 0;
  if (EvaluateValue(ResVal, Tok)) {
    // Parse error, skip the rest of the macro line.
    if (Tok.getKind() != tok::eom)
      DiscardUntilEndOfDirective();
    return false;
  }
  
  // If we are at the end of the expression after just parsing a value, there
  // must be no (unparenthesized) binary operators involved, so we can exit
  // directly.
  if (Tok.getKind() == tok::eom) {
    return ResVal != 0;
  }
  
  // Otherwise, we must have a binary operator (e.g. "#if 1 < 2"), so parse the
  // operator and the stuff after it.
  if (EvaluateDirectiveSubExpr(ResVal, 1, Tok)) {
    // Parse error, skip the rest of the macro line.
    if (Tok.getKind() != tok::eom)
      DiscardUntilEndOfDirective();
    return false;
  }
  
  // If we aren't at the tok::eom token, something bad happened, like an extra
  // ')' token.
  if (Tok.getKind() != tok::eom) {
    Diag(Tok, diag::err_pp_expected_eol);
    DiscardUntilEndOfDirective();
  }
  
  return ResVal != 0;
}

/// EvaluateValue - Evaluate the token PeekTok (and any others needed) and
/// return the computed value in Result.  Return true if there was an error
/// parsing.
bool Preprocessor::EvaluateValue(int &Result, LexerToken &PeekTok) {
  Result = 0;
  
  // If this token's spelling is a pp-identifier, check to see if it is
  // 'defined' or if it is a macro.  Note that we check here because many
  // keywords are pp-identifiers, so we can't check the kind.
  if (const IdentifierInfo *II = PeekTok.getIdentifierInfo()) {
    // If this identifier isn't 'defined' and it wasn't macro expanded, it turns
    // into a simple 0.
    if (strcmp(II->getName(), "defined")) {
      Result = 0;
      Lex(PeekTok);
      return false;
    }

    // Handle "defined X" and "defined(X)".
    assert(!DisableMacroExpansion &&
           "How could macro exp already be disabled?");
    // Turn off macro expansion.
    DisableMacroExpansion = true;

    // Get the next token.
    Lex(PeekTok);

    // Two options, it can either be a pp-identifier or a (.
    bool InParens = false;
    if (PeekTok.getKind() == tok::l_paren) {
      // Found a paren, remember we saw it and skip it.
      InParens = true;
      Lex(PeekTok);
    }
    
    // If we don't have a pp-identifier now, this is an error.
    if ((II = PeekTok.getIdentifierInfo()) == 0) {
      DisableMacroExpansion = false;
      Diag(PeekTok, diag::err_pp_defined_requires_identifier);
      return true;
    }
    
    // Otherwise, we got an identifier, is it defined to something?
    Result = II->getMacroInfo() != 0;
    
    // If there is a macro, mark it used.
    if (Result) II->getMacroInfo()->setIsUsed(true);

    // Consume identifier.
    Lex(PeekTok);

    // If we are in parens, ensure we have a trailing ).
    if (InParens) {
      if (PeekTok.getKind() != tok::r_paren) {
        Diag(PeekTok, diag::err_pp_missing_rparen);
        return true;
      }
      // Consume the ).
      Lex(PeekTok);
    }
    
    DisableMacroExpansion = false;
    return false;
  }
  
  switch (PeekTok.getKind()) {
  default:  // Non-value token.
    Diag(PeekTok, diag::err_pp_expr_bad_token);
    return true;
  case tok::eom:
  case tok::r_paren:
    // If there is no expression, report and exit.
    Diag(PeekTok, diag::err_pp_expected_value_in_expr);
    return true;
  case tok::numeric_constant: {
    // FIXME: faster.  FIXME: track signs.
    std::string Spell = getSpelling(PeekTok);
    // FIXME: COMPUTE integer constants CORRECTLY.
    Result = atoi(Spell.c_str());
    Lex(PeekTok);
    return false;
  }
  case tok::l_paren:
    Lex(PeekTok);  // Eat the (.
    // Parse the value and if there are any binary operators involved, parse
    // them.
    if (EvaluateValue(Result, PeekTok) ||
        EvaluateDirectiveSubExpr(Result, 1, PeekTok))
      return true;

    if (PeekTok.getKind() != tok::r_paren) {
      Diag(PeekTok, diag::err_pp_expected_rparen);
      return true;
    }
    Lex(PeekTok);  // Eat the ).
    return false;
 
  case tok::plus:
    // Unary plus doesn't modify the value.
    Lex(PeekTok);
    return EvaluateValue(Result, PeekTok);
  case tok::minus:
    Lex(PeekTok);
    if (EvaluateValue(Result, PeekTok)) return true;
    Result = -Result;
    return false;
    
  case tok::tilde:
    Lex(PeekTok);
    if (EvaluateValue(Result, PeekTok)) return true;
    Result = ~Result;
    return false;
    
  case tok::exclaim:
    Lex(PeekTok);
    if (EvaluateValue(Result, PeekTok)) return true;
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
                                            LexerToken &PeekTok) {
  unsigned PeekPrec = getPrecedence(PeekTok.getKind());
  // If this token isn't valid, report the error.
  if (PeekPrec == ~0U) {
    Diag(PeekTok, diag::err_pp_expr_bad_token);
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
    Lex(PeekTok);

    int RHS;
    // Parse the RHS of the operator.
    if (EvaluateValue(RHS, PeekTok)) return true;

    // Remember the precedence of this operator and get the precedence of the
    // operator immediately to the right of the RHS.
    unsigned ThisPrec = PeekPrec;
    PeekPrec = getPrecedence(PeekTok.getKind());

    // If this token isn't valid, report the error.
    if (PeekPrec == ~0U) {
      Diag(PeekTok, diag::err_pp_expr_bad_token);
      return true;
    }
    
    bool isRightAssoc = Operator == tok::question;
    
    // Get the precedence of the operator to the right of the RHS.  If it binds
    // more tightly with RHS than we do, evaluate it completely first.
    if (ThisPrec < PeekPrec ||
        (ThisPrec == PeekPrec && isRightAssoc)) {
      if (EvaluateDirectiveSubExpr(RHS, ThisPrec+1, PeekTok))
        return true;
      PeekPrec = getPrecedence(PeekTok.getKind());
    }
    assert(PeekPrec <= ThisPrec && "Recursion didn't work!");
    
    switch (Operator) {
    default: assert(0 && "Unknown operator token!");
    case tok::percent:
      if (RHS == 0) {
        Diag(OpToken, diag::err_pp_remainder_by_zero);
        return true;
      }
      LHS %= RHS;
      break;
    case tok::slash:
      if (RHS == 0) {
        Diag(OpToken, diag::err_pp_division_by_zero);
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
      Diag(OpToken, diag::ext_pp_comma_expr);
      LHS = RHS; // LHS = LHS,RHS -> RHS.
      break; 
    case tok::question: {
      // Parse the : part of the expression.
      if (PeekTok.getKind() != tok::colon) {
        Diag(OpToken, diag::err_pp_question_without_colon);
        return true;
      }
      // Consume the :.
      Lex(PeekTok);

      // Evaluate the value after the :.
      int AfterColonVal = 0;
      if (EvaluateValue(AfterColonVal, PeekTok)) return true;

      // Parse anything after the : RHS that has a higher precedence than ?.
      if (EvaluateDirectiveSubExpr(AfterColonVal, ThisPrec+1,
                                   PeekTok))
        return true;
      
      // Now that we have the condition, the LHS and the RHS of the :, evaluate.
      LHS = LHS ? RHS : AfterColonVal;
      
      // Figure out the precedence of the token after the : part.
      PeekPrec = getPrecedence(PeekTok.getKind());
      break;
    }
    case tok::colon:
      // Don't allow :'s to float around without being part of ?: exprs.
      Diag(OpToken, diag::err_pp_colon_without_question);
      return true;
    }
  }
  
  return false;
}
