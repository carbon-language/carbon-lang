//===--- PPExpressions.cpp - Preprocessor Expression Evaluation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Preprocessor::EvaluateDirectiveExpression method,
// which parses and evaluates integer constant expressions for #if directives.
//
//===----------------------------------------------------------------------===//
//
// FIXME: implement testing for #assert's.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallString.h"
using namespace clang;

static bool EvaluateDirectiveSubExpr(llvm::APSInt &LHS, unsigned MinPrec,
                                     Token &PeekTok, bool ValueLive,
                                     Preprocessor &PP);

/// DefinedTracker - This struct is used while parsing expressions to keep track
/// of whether !defined(X) has been seen.
///
/// With this simple scheme, we handle the basic forms:
///    !defined(X)   and !defined X
/// but we also trivially handle (silly) stuff like:
///    !!!defined(X) and +!defined(X) and !+!+!defined(X) and !(defined(X)).
struct DefinedTracker {
  /// Each time a Value is evaluated, it returns information about whether the
  /// parsed value is of the form defined(X), !defined(X) or is something else.
  enum TrackerState {
    DefinedMacro,        // defined(X)
    NotDefinedMacro,     // !defined(X)
    Unknown              // Something else.
  } State;
  /// TheMacro - When the state is DefinedMacro or NotDefinedMacro, this
  /// indicates the macro that was checked.
  IdentifierInfo *TheMacro;
};



/// EvaluateValue - Evaluate the token PeekTok (and any others needed) and
/// return the computed value in Result.  Return true if there was an error
/// parsing.  This function also returns information about the form of the
/// expression in DT.  See above for information on what DT means.
///
/// If ValueLive is false, then this value is being evaluated in a context where
/// the result is not used.  As such, avoid diagnostics that relate to
/// evaluation.
static bool EvaluateValue(llvm::APSInt &Result, Token &PeekTok,
                          DefinedTracker &DT, bool ValueLive,
                          Preprocessor &PP) {
  Result = 0;
  DT.State = DefinedTracker::Unknown;
  
  // If this token's spelling is a pp-identifier, check to see if it is
  // 'defined' or if it is a macro.  Note that we check here because many
  // keywords are pp-identifiers, so we can't check the kind.
  if (IdentifierInfo *II = PeekTok.getIdentifierInfo()) {
    // If this identifier isn't 'defined' and it wasn't macro expanded, it turns
    // into a simple 0, unless it is the C++ keyword "true", in which case it
    // turns into "1".
    if (II->getPPKeywordID() != tok::pp_defined) {
      PP.Diag(PeekTok, diag::warn_pp_undef_identifier, II->getName());
      Result = II->getTokenID() == tok::kw_true;
      Result.setIsUnsigned(false);  // "0" is signed intmax_t 0.
      PP.LexNonComment(PeekTok);
      return false;
    }

    // Handle "defined X" and "defined(X)".

    // Get the next token, don't expand it.
    PP.LexUnexpandedToken(PeekTok);

    // Two options, it can either be a pp-identifier or a (.
    bool InParens = false;
    if (PeekTok.is(tok::l_paren)) {
      // Found a paren, remember we saw it and skip it.
      InParens = true;
      PP.LexUnexpandedToken(PeekTok);
    }
    
    // If we don't have a pp-identifier now, this is an error.
    if ((II = PeekTok.getIdentifierInfo()) == 0) {
      PP.Diag(PeekTok, diag::err_pp_defined_requires_identifier);
      return true;
    }
    
    // Otherwise, we got an identifier, is it defined to something?
    Result = II->hasMacroDefinition();
    Result.setIsUnsigned(false);  // Result is signed intmax_t.

    // If there is a macro, mark it used.
    if (Result != 0 && ValueLive) {
      MacroInfo *Macro = PP.getMacroInfo(II);
      Macro->setIsUsed(true);
    }

    // Consume identifier.
    PP.LexNonComment(PeekTok);

    // If we are in parens, ensure we have a trailing ).
    if (InParens) {
      if (PeekTok.isNot(tok::r_paren)) {
        PP.Diag(PeekTok, diag::err_pp_missing_rparen);
        return true;
      }
      // Consume the ).
      PP.LexNonComment(PeekTok);
    }
    
    // Success, remember that we saw defined(X).
    DT.State = DefinedTracker::DefinedMacro;
    DT.TheMacro = II;
    return false;
  }
  
  switch (PeekTok.getKind()) {
  default:  // Non-value token.
    PP.Diag(PeekTok, diag::err_pp_expr_bad_token_start_expr);
    return true;
  case tok::eom:
  case tok::r_paren:
    // If there is no expression, report and exit.
    PP.Diag(PeekTok, diag::err_pp_expected_value_in_expr);
    return true;
  case tok::numeric_constant: {
    llvm::SmallString<64> IntegerBuffer;
    IntegerBuffer.resize(PeekTok.getLength());
    const char *ThisTokBegin = &IntegerBuffer[0];
    unsigned ActualLength = PP.getSpelling(PeekTok, ThisTokBegin);
    NumericLiteralParser Literal(ThisTokBegin, ThisTokBegin+ActualLength, 
                                 PeekTok.getLocation(), PP);
    if (Literal.hadError)
      return true; // a diagnostic was already reported.
    
    if (Literal.isFloatingLiteral() || Literal.isImaginary) {
      PP.Diag(PeekTok, diag::err_pp_illegal_floating_literal);
      return true;
    }
    assert(Literal.isIntegerLiteral() && "Unknown ppnumber");

    // long long is a C99 feature.
    if (!PP.getLangOptions().C99 && !PP.getLangOptions().CPlusPlus0x
        && Literal.isLongLong)
      PP.Diag(PeekTok, diag::ext_longlong);

    // Parse the integer literal into Result.
    if (Literal.GetIntegerValue(Result)) {
      // Overflow parsing integer literal.
      if (ValueLive) PP.Diag(PeekTok, diag::warn_integer_too_large);
      Result.setIsUnsigned(true);
    } else {
      // Set the signedness of the result to match whether there was a U suffix
      // or not.
      Result.setIsUnsigned(Literal.isUnsigned);
    
      // Detect overflow based on whether the value is signed.  If signed
      // and if the value is too large, emit a warning "integer constant is so
      // large that it is unsigned" e.g. on 12345678901234567890 where intmax_t
      // is 64-bits.
      if (!Literal.isUnsigned && Result.isNegative()) {
        if (ValueLive)PP.Diag(PeekTok, diag::warn_integer_too_large_for_signed);
        Result.setIsUnsigned(true);
      }
    }
    
    // Consume the token.
    PP.LexNonComment(PeekTok);
    return false;
  }
  case tok::char_constant: {   // 'x'
    llvm::SmallString<32> CharBuffer;
    CharBuffer.resize(PeekTok.getLength());
    const char *ThisTokBegin = &CharBuffer[0];
    unsigned ActualLength = PP.getSpelling(PeekTok, ThisTokBegin);
    CharLiteralParser Literal(ThisTokBegin, ThisTokBegin+ActualLength, 
                              PeekTok.getLocation(), PP);
    if (Literal.hadError())
      return true;  // A diagnostic was already emitted.

    // Character literals are always int or wchar_t, expand to intmax_t.
    TargetInfo &TI = PP.getTargetInfo();
    unsigned NumBits = TI.getCharWidth(Literal.isWide());
    
    // Set the width.
    llvm::APSInt Val(NumBits);
    // Set the value.
    Val = Literal.getValue();
    // Set the signedness.
    Val.setIsUnsigned(!TI.isCharSigned());
    
    if (Result.getBitWidth() > Val.getBitWidth()) {
      Result = Val.extend(Result.getBitWidth());
    } else {
      assert(Result.getBitWidth() == Val.getBitWidth() &&
             "intmax_t smaller than char/wchar_t?");
      Result = Val;
    }

    // Consume the token.
    PP.LexNonComment(PeekTok);
    return false;
  }
  case tok::l_paren:
    PP.LexNonComment(PeekTok);  // Eat the (.
    // Parse the value and if there are any binary operators involved, parse
    // them.
    if (EvaluateValue(Result, PeekTok, DT, ValueLive, PP)) return true;

    // If this is a silly value like (X), which doesn't need parens, check for
    // !(defined X).
    if (PeekTok.is(tok::r_paren)) {
      // Just use DT unmodified as our result.
    } else {
      if (EvaluateDirectiveSubExpr(Result, 1, PeekTok, ValueLive, PP))
        return true;
      
      if (PeekTok.isNot(tok::r_paren)) {
        PP.Diag(PeekTok, diag::err_pp_expected_rparen);
        return true;
      }
      DT.State = DefinedTracker::Unknown;
    }
    PP.LexNonComment(PeekTok);  // Eat the ).
    return false;
 
  case tok::plus:
    // Unary plus doesn't modify the value.
    PP.LexNonComment(PeekTok);
    return EvaluateValue(Result, PeekTok, DT, ValueLive, PP);
  case tok::minus: {
    SourceLocation Loc = PeekTok.getLocation();
    PP.LexNonComment(PeekTok);
    if (EvaluateValue(Result, PeekTok, DT, ValueLive, PP)) return true;
    // C99 6.5.3.3p3: The sign of the result matches the sign of the operand.
    Result = -Result;
    
    bool Overflow = false;
    if (Result.isUnsigned())
      Overflow = Result.isNegative();
    else if (Result.isMinSignedValue())
      Overflow = true;   // -MININT is the only thing that overflows.
      
    // If this operator is live and overflowed, report the issue.
    if (Overflow && ValueLive)
      PP.Diag(Loc, diag::warn_pp_expr_overflow);
    
    DT.State = DefinedTracker::Unknown;
    return false;
  }
    
  case tok::tilde:
    PP.LexNonComment(PeekTok);
    if (EvaluateValue(Result, PeekTok, DT, ValueLive, PP)) return true;
    // C99 6.5.3.3p4: The sign of the result matches the sign of the operand.
    Result = ~Result;
    DT.State = DefinedTracker::Unknown;
    return false;
    
  case tok::exclaim:
    PP.LexNonComment(PeekTok);
    if (EvaluateValue(Result, PeekTok, DT, ValueLive, PP)) return true;
    Result = !Result;
    // C99 6.5.3.3p5: The sign of the result is 'int', aka it is signed.
    Result.setIsUnsigned(false);
    
    if (DT.State == DefinedTracker::DefinedMacro)
      DT.State = DefinedTracker::NotDefinedMacro;
    else if (DT.State == DefinedTracker::NotDefinedMacro)
      DT.State = DefinedTracker::DefinedMacro;
    return false;
    
  // FIXME: Handle #assert
  }
}



/// getPrecedence - Return the precedence of the specified binary operator
/// token.  This returns:
///   ~0 - Invalid token.
///   14 - *,/,%
///   13 - -,+
///   12 - <<,>>
///   11 - >=, <=, >, <
///   10 - ==, !=
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
  case tok::star:                 return 14;
  case tok::plus:
  case tok::minus:                return 13;
  case tok::lessless:
  case tok::greatergreater:       return 12;
  case tok::lessequal:
  case tok::less:
  case tok::greaterequal:
  case tok::greater:              return 11;
  case tok::exclaimequal:
  case tok::equalequal:           return 10;
  case tok::amp:                  return 9;
  case tok::caret:                return 8;
  case tok::pipe:                 return 7;
  case tok::ampamp:               return 6;
  case tok::pipepipe:             return 5;
  case tok::comma:                return 4;
  case tok::question:             return 3;
  case tok::colon:                return 2;
  case tok::r_paren:              return 0;   // Lowest priority, end of expr.
  case tok::eom:                  return 0;   // Lowest priority, end of macro.
  }
}


/// EvaluateDirectiveSubExpr - Evaluate the subexpression whose first token is
/// PeekTok, and whose precedence is PeekPrec.
///
/// If ValueLive is false, then this value is being evaluated in a context where
/// the result is not used.  As such, avoid diagnostics that relate to
/// evaluation.
static bool EvaluateDirectiveSubExpr(llvm::APSInt &LHS, unsigned MinPrec,
                                     Token &PeekTok, bool ValueLive,
                                     Preprocessor &PP) {
  unsigned PeekPrec = getPrecedence(PeekTok.getKind());
  // If this token isn't valid, report the error.
  if (PeekPrec == ~0U) {
    PP.Diag(PeekTok, diag::err_pp_expr_bad_token_binop);
    return true;
  }
  
  while (1) {
    // If this token has a lower precedence than we are allowed to parse, return
    // it so that higher levels of the recursion can parse it.
    if (PeekPrec < MinPrec)
      return false;
    
    tok::TokenKind Operator = PeekTok.getKind();
    
    // If this is a short-circuiting operator, see if the RHS of the operator is
    // dead.  Note that this cannot just clobber ValueLive.  Consider 
    // "0 && 1 ? 4 : 1 / 0", which is parsed as "(0 && 1) ? 4 : (1 / 0)".  In
    // this example, the RHS of the && being dead does not make the rest of the
    // expr dead.
    bool RHSIsLive;
    if (Operator == tok::ampamp && LHS == 0)
      RHSIsLive = false;   // RHS of "0 && x" is dead.
    else if (Operator == tok::pipepipe && LHS != 0)
      RHSIsLive = false;   // RHS of "1 || x" is dead.
    else if (Operator == tok::question && LHS == 0)
      RHSIsLive = false;   // RHS (x) of "0 ? x : y" is dead.
    else
      RHSIsLive = ValueLive;

    // Consume the operator, saving the operator token for error reporting.
    Token OpToken = PeekTok;
    PP.LexNonComment(PeekTok);

    llvm::APSInt RHS(LHS.getBitWidth());
    // Parse the RHS of the operator.
    DefinedTracker DT;
    if (EvaluateValue(RHS, PeekTok, DT, RHSIsLive, PP)) return true;

    // Remember the precedence of this operator and get the precedence of the
    // operator immediately to the right of the RHS.
    unsigned ThisPrec = PeekPrec;
    PeekPrec = getPrecedence(PeekTok.getKind());

    // If this token isn't valid, report the error.
    if (PeekPrec == ~0U) {
      PP.Diag(PeekTok, diag::err_pp_expr_bad_token_binop);
      return true;
    }
    
    bool isRightAssoc = Operator == tok::question;
    
    // Get the precedence of the operator to the right of the RHS.  If it binds
    // more tightly with RHS than we do, evaluate it completely first.
    if (ThisPrec < PeekPrec ||
        (ThisPrec == PeekPrec && isRightAssoc)) {
      if (EvaluateDirectiveSubExpr(RHS, ThisPrec+1, PeekTok, RHSIsLive, PP))
        return true;
      PeekPrec = getPrecedence(PeekTok.getKind());
    }
    assert(PeekPrec <= ThisPrec && "Recursion didn't work!");
    
    // Usual arithmetic conversions (C99 6.3.1.8p1): result is unsigned if
    // either operand is unsigned.  
    llvm::APSInt Res(LHS.getBitWidth());
    switch (Operator) {
    case tok::question:       // No UAC for x and y in "x ? y : z".
    case tok::lessless:       // Shift amount doesn't UAC with shift value.
    case tok::greatergreater: // Shift amount doesn't UAC with shift value.
    case tok::comma:          // Comma operands are not subject to UACs.
    case tok::pipepipe:       // Logical || does not do UACs.
    case tok::ampamp:         // Logical && does not do UACs.
      break;                  // No UAC
    default:
      Res.setIsUnsigned(LHS.isUnsigned()|RHS.isUnsigned());
      // If this just promoted something from signed to unsigned, and if the
      // value was negative, warn about it.
      if (ValueLive && Res.isUnsigned()) {
        if (!LHS.isUnsigned() && LHS.isNegative())
          PP.Diag(OpToken, diag::warn_pp_convert_lhs_to_positive,
                  LHS.toStringSigned() + " to " + LHS.toStringUnsigned());
        if (!RHS.isUnsigned() && RHS.isNegative())
          PP.Diag(OpToken, diag::warn_pp_convert_rhs_to_positive,
                  RHS.toStringSigned() + " to " + RHS.toStringUnsigned());
      }
      LHS.setIsUnsigned(Res.isUnsigned());
      RHS.setIsUnsigned(Res.isUnsigned());
    }
    
    // FIXME: All of these should detect and report overflow??
    bool Overflow = false;
    switch (Operator) {
    default: assert(0 && "Unknown operator token!");
    case tok::percent:
      if (RHS == 0) {
        if (ValueLive) {
          PP.Diag(OpToken, diag::err_pp_remainder_by_zero);
          return true;
        }
      } else {
        Res = LHS % RHS;
      }
      break;
    case tok::slash:
      if (RHS == 0) {
        if (ValueLive) {
          PP.Diag(OpToken, diag::err_pp_division_by_zero);
          return true;
        }
        break;
      }

      Res = LHS / RHS;
      if (LHS.isSigned())
        Overflow = LHS.isMinSignedValue() && RHS.isAllOnesValue(); // MININT/-1
      break;
        
    case tok::star:
      Res = LHS * RHS;
      if (LHS != 0 && RHS != 0)
        Overflow = Res/RHS != LHS || Res/LHS != RHS;
      break;
    case tok::lessless: {
      // Determine whether overflow is about to happen.
      unsigned ShAmt = static_cast<unsigned>(RHS.getLimitedValue());
      if (ShAmt >= LHS.getBitWidth())
        Overflow = true, ShAmt = LHS.getBitWidth()-1;
      else if (LHS.isUnsigned())
        Overflow = ShAmt > LHS.countLeadingZeros();
      else if (LHS.isNonNegative())
        Overflow = ShAmt >= LHS.countLeadingZeros(); // Don't allow sign change.
      else
        Overflow = ShAmt >= LHS.countLeadingOnes();
      
      Res = LHS << ShAmt;
      break;
    }
    case tok::greatergreater: {
      // Determine whether overflow is about to happen.
      unsigned ShAmt = static_cast<unsigned>(RHS.getLimitedValue());
      if (ShAmt >= LHS.getBitWidth())
        Overflow = true, ShAmt = LHS.getBitWidth()-1;
      Res = LHS >> ShAmt;
      break;
    }
    case tok::plus:
      Res = LHS + RHS;
      if (LHS.isUnsigned())
        Overflow = Res.ult(LHS);
      else if (LHS.isNonNegative() == RHS.isNonNegative() &&
               Res.isNonNegative() != LHS.isNonNegative())
        Overflow = true;  // Overflow for signed addition.
      break;
    case tok::minus:
      Res = LHS - RHS;
      if (LHS.isUnsigned())
        Overflow = Res.ugt(LHS);
      else if (LHS.isNonNegative() != RHS.isNonNegative() &&
               Res.isNonNegative() != LHS.isNonNegative())
        Overflow = true;  // Overflow for signed subtraction.
      break;
    case tok::lessequal:
      Res = LHS <= RHS;
      Res.setIsUnsigned(false);  // C99 6.5.8p6, result is always int (signed)
      break;
    case tok::less:
      Res = LHS < RHS;
      Res.setIsUnsigned(false);  // C99 6.5.8p6, result is always int (signed)
      break;
    case tok::greaterequal:
      Res = LHS >= RHS;
      Res.setIsUnsigned(false);  // C99 6.5.8p6, result is always int (signed)
      break;
    case tok::greater:
      Res = LHS > RHS;
      Res.setIsUnsigned(false);  // C99 6.5.8p6, result is always int (signed)
      break;
    case tok::exclaimequal:
      Res = LHS != RHS;
      Res.setIsUnsigned(false);  // C99 6.5.9p3, result is always int (signed)
      break;
    case tok::equalequal:
      Res = LHS == RHS;
      Res.setIsUnsigned(false);  // C99 6.5.9p3, result is always int (signed)
      break;
    case tok::amp:
      Res = LHS & RHS;
      break;
    case tok::caret:
      Res = LHS ^ RHS;
      break;
    case tok::pipe:
      Res = LHS | RHS;
      break;
    case tok::ampamp:
      Res = (LHS != 0 && RHS != 0);
      Res.setIsUnsigned(false);  // C99 6.5.13p3, result is always int (signed)
      break;
    case tok::pipepipe:
      Res = (LHS != 0 || RHS != 0);
      Res.setIsUnsigned(false);  // C99 6.5.14p3, result is always int (signed)
      break;
    case tok::comma:
      // Comma is invalid in pp expressions in c89/c++ mode, but is valid in C99
      // if not being evaluated.
      if (!PP.getLangOptions().C99 || ValueLive)
        PP.Diag(OpToken, diag::ext_pp_comma_expr);
      Res = RHS; // LHS = LHS,RHS -> RHS.
      break; 
    case tok::question: {
      // Parse the : part of the expression.
      if (PeekTok.isNot(tok::colon)) {
        PP.Diag(OpToken, diag::err_pp_question_without_colon);
        return true;
      }
      // Consume the :.
      PP.LexNonComment(PeekTok);

      // Evaluate the value after the :.
      bool AfterColonLive = ValueLive && LHS == 0;
      llvm::APSInt AfterColonVal(LHS.getBitWidth());
      DefinedTracker DT;
      if (EvaluateValue(AfterColonVal, PeekTok, DT, AfterColonLive, PP))
        return true;

      // Parse anything after the : RHS that has a higher precedence than ?.
      if (EvaluateDirectiveSubExpr(AfterColonVal, ThisPrec+1,
                                   PeekTok, AfterColonLive, PP))
        return true;
      
      // Now that we have the condition, the LHS and the RHS of the :, evaluate.
      Res = LHS != 0 ? RHS : AfterColonVal;

      // Usual arithmetic conversions (C99 6.3.1.8p1): result is unsigned if
      // either operand is unsigned.
      Res.setIsUnsigned(RHS.isUnsigned() | AfterColonVal.isUnsigned());
      
      // Figure out the precedence of the token after the : part.
      PeekPrec = getPrecedence(PeekTok.getKind());
      break;
    }
    case tok::colon:
      // Don't allow :'s to float around without being part of ?: exprs.
      PP.Diag(OpToken, diag::err_pp_colon_without_question);
      return true;
    }

    // If this operator is live and overflowed, report the issue.
    if (Overflow && ValueLive)
      PP.Diag(OpToken, diag::warn_pp_expr_overflow);
    
    // Put the result back into 'LHS' for our next iteration.
    LHS = Res;
  }
  
  return false;
}

/// EvaluateDirectiveExpression - Evaluate an integer constant expression that
/// may occur after a #if or #elif directive.  If the expression is equivalent
/// to "!defined(X)" return X in IfNDefMacro.
bool Preprocessor::
EvaluateDirectiveExpression(IdentifierInfo *&IfNDefMacro) {
  // Peek ahead one token.
  Token Tok;
  Lex(Tok);
  
  // C99 6.10.1p3 - All expressions are evaluated as intmax_t or uintmax_t.
  unsigned BitWidth = getTargetInfo().getIntMaxTWidth();
    
  llvm::APSInt ResVal(BitWidth);
  DefinedTracker DT;
  if (EvaluateValue(ResVal, Tok, DT, true, *this)) {
    // Parse error, skip the rest of the macro line.
    if (Tok.isNot(tok::eom))
      DiscardUntilEndOfDirective();
    return false;
  }
  
  // If we are at the end of the expression after just parsing a value, there
  // must be no (unparenthesized) binary operators involved, so we can exit
  // directly.
  if (Tok.is(tok::eom)) {
    // If the expression we parsed was of the form !defined(macro), return the
    // macro in IfNDefMacro.
    if (DT.State == DefinedTracker::NotDefinedMacro)
      IfNDefMacro = DT.TheMacro;
    
    return ResVal != 0;
  }
  
  // Otherwise, we must have a binary operator (e.g. "#if 1 < 2"), so parse the
  // operator and the stuff after it.
  if (EvaluateDirectiveSubExpr(ResVal, 1, Tok, true, *this)) {
    // Parse error, skip the rest of the macro line.
    if (Tok.isNot(tok::eom))
      DiscardUntilEndOfDirective();
    return false;
  }
  
  // If we aren't at the tok::eom token, something bad happened, like an extra
  // ')' token.
  if (Tok.isNot(tok::eom)) {
    Diag(Tok, diag::err_pp_expected_eol);
    DiscardUntilEndOfDirective();
  }
  
  return ResVal != 0;
}

