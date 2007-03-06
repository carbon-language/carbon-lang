//===--- SemaExpr.cpp - Semantic Analysis for Expressions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for expressions.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;
using namespace clang;



/// HexDigitValue - Return the value of the specified hex digit, or -1 if it's
/// not valid.
static int HexDigitValue(char C) {
  if (C >= '0' && C <= '9') return C-'0';
  if (C >= 'a' && C <= 'f') return C-'a'+10;
  if (C >= 'A' && C <= 'F') return C-'A'+10;
  return -1;
}

/// ParseStringLiteral - The specified tokens were lexed as pasted string
/// fragments (e.g. "foo" "bar" L"baz").  The result string has to handle string
/// concatenation ([C99 5.1.1.2, translation phase #6]), so it may come from
/// multiple tokens.  However, the common case is that StringToks points to one
/// string.
/// 
Action::ExprResult
Sema::ParseStringLiteral(const LexerToken *StringToks, unsigned NumStringToks) {
  assert(NumStringToks && "Must have at least one string!");

  // Scan all of the string portions, remember the max individual token length,
  // computing a bound on the concatenated string length, and see whether any
  // piece is a wide-string.  If any of the string portions is a wide-string
  // literal, the result is a wide-string literal [C99 6.4.5p4].
  unsigned MaxTokenLength = StringToks[0].getLength();
  unsigned SizeBound = StringToks[0].getLength()-2;  // -2 for "".
  bool AnyWide = StringToks[0].getKind() == tok::wide_string_literal;
  
  // The common case is that there is only one string fragment.
  for (unsigned i = 1; i != NumStringToks; ++i) {
    // The string could be shorter than this if it needs cleaning, but this is a
    // reasonable bound, which is all we need.
    SizeBound += StringToks[i].getLength()-2;  // -2 for "".

    // Remember maximum string piece length.
    if (StringToks[i].getLength() > MaxTokenLength) 
      MaxTokenLength = StringToks[i].getLength();
    
    // Remember if we see any wide strings.
    AnyWide |= StringToks[i].getKind() == tok::wide_string_literal;
  }
  
  
  // Include space for the null terminator.
  ++SizeBound;
  
  // TODO: K&R warning: "traditional C rejects string constant concatenation"
  
  // Get the width in bytes of wchar_t.  If no wchar_t strings are used, do not
  // query the target.  As such, wchar_tByteWidth is only valid if AnyWide=true.
  unsigned wchar_tByteWidth = ~0U;
  if (AnyWide)
    wchar_tByteWidth =Context.Target.getWCharWidth(StringToks[0].getLocation());
  
  // The output buffer size needs to be large enough to hold wide characters.
  // This is a worst-case assumption which basically corresponds to L"" "long".
  if (AnyWide)
    SizeBound *= wchar_tByteWidth;
  
  // Create a temporary buffer to hold the result string data.
  SmallString<512> ResultBuf;
  ResultBuf.resize(SizeBound);
  
  // Likewise, but for each string piece.
  SmallString<512> TokenBuf;
  TokenBuf.resize(MaxTokenLength);
  
  // Loop over all the strings, getting their spelling, and expanding them to
  // wide strings as appropriate.
  char *ResultPtr = &ResultBuf[0];   // Next byte to fill in.
  
  for (unsigned i = 0, e = NumStringToks; i != e; ++i) {
    const char *ThisTokBuf = &TokenBuf[0];
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
        Diag(StringToks[i].getLocation(), diag::ext_nonstandard_escape, "e");
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
          Diag(StringToks[i].getLocation(), diag::err_hex_escape_no_digits);
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
          Diag(StringToks[i].getLocation(), diag::ext_nonstandard_escape,
               std::string()+(char)ResultChar);
          break;
        }
        // FALL THROUGH.
      default:
        if (isgraph(ThisTokBuf[0])) {
          Diag(StringToks[i].getLocation(), diag::ext_unknown_escape,
               std::string()+(char)ResultChar);
        } else {
          Diag(StringToks[i].getLocation(), diag::ext_unknown_escape,
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
  
  SmallVector<SourceLocation, 4> StringTokLocs;
  for (unsigned i = 0; i != NumStringToks; ++i)
    StringTokLocs.push_back(StringToks[i].getLocation());
  
  // FIXME: use factory.
  
  // Pass &StringTokLocs[0], StringTokLocs.size() to factory!
  return new StringLiteral(&ResultBuf[0], ResultPtr-&ResultBuf[0], AnyWide);
}


/// ParseIdentifierExpr - The parser read an identifier in expression context,
/// validate it per-C99 6.5.1.  HasTrailingLParen indicates whether this
/// identifier is used in an function call context.
Sema::ExprResult Sema::ParseIdentifierExpr(Scope *S, SourceLocation Loc,
                                           IdentifierInfo &II,
                                           bool HasTrailingLParen) {
  // Could be enum-constant or decl.
  Decl *D = LookupScopedDecl(&II, Decl::IDNS_Ordinary, Loc, S);
  if (D == 0) {
    // Otherwise, this could be an implicitly declared function reference (legal
    // in C90, extension in C99).
    if (HasTrailingLParen &&
        // Not in C++.
        !getLangOptions().CPlusPlus) {
      D = ImplicitlyDefineFunction(Loc, II, S);
    } else {
      // If this name wasn't predeclared and if this is not a function call,
      // diagnose the problem.
      Diag(Loc, diag::err_undeclared_var_use, II.getName());
      return true;
    }
  }
  
  if (isa<TypedefDecl>(D)) {
    Diag(Loc, diag::err_unexpected_typedef, II.getName());
    return true;
  }
    
  return new DeclRefExpr(D);
}

Sema::ExprResult Sema::ParseSimplePrimaryExpr(SourceLocation Loc,
                                              tok::TokenKind Kind) {
  switch (Kind) {
  default:
    assert(0 && "Unknown simple primary expr!");
  case tok::char_constant:     // constant: character-constant
    // TODO: MOVE this to be some other callback.
  case tok::kw___func__:       // primary-expression: __func__ [C99 6.4.2.2]
  case tok::kw___FUNCTION__:   // primary-expression: __FUNCTION__ [GNU]
  case tok::kw___PRETTY_FUNCTION__:  // primary-expression: __P..Y_F..N__ [GNU]
    return 0;
  }
}

///       integer-constant: [C99 6.4.4.1]
///         decimal-constant integer-suffix
///         octal-constant integer-suffix
///         hexadecimal-constant integer-suffix
///       decimal-constant: 
///         nonzero-digit
///         decimal-constant digit
///       octal-constant: 
///         0
///         octal-constant octal-digit
///       hexadecimal-constant: 
///         hexadecimal-prefix hexadecimal-digit
///         hexadecimal-constant hexadecimal-digit
///       hexadecimal-prefix: one of
///         0x 0X
///       integer-suffix:
///         unsigned-suffix [long-suffix]
///         unsigned-suffix [long-long-suffix]
///         long-suffix [unsigned-suffix]
///         long-long-suffix [unsigned-sufix]
///       nonzero-digit:
///         1 2 3 4 5 6 7 8 9
///       octal-digit:
///         0 1 2 3 4 5 6 7
///       hexadecimal-digit:
///         0 1 2 3 4 5 6 7 8 9
///         a b c d e f
///         A B C D E F
///       unsigned-suffix: one of
///         u U
///       long-suffix: one of
///         l L
///       long-long-suffix: one of 
///         ll LL
///
///       floating-constant: [C99 6.4.4.2]
///         TODO: add rules...
///
Action::ExprResult Sema::ParseNumericConstant(const LexerToken &Tok) {
  SmallString<512> IntegerBuffer;
  IntegerBuffer.resize(Tok.getLength());
  const char *ThisTokBegin = &IntegerBuffer[0];
  
  // Get the spelling of the token, which eliminates trigraphs, etc.  Notes:
  // - We know that ThisTokBuf points to a buffer that is big enough for the 
  //   whole token and 'spelled' tokens can only shrink.
  // - In practice, the local buffer is only used when the spelling doesn't
  //   match the original token (which is rare). The common case simply returns
  //   a pointer to a *constant* buffer (avoiding a copy). 
  
  unsigned ActualLength = PP.getSpelling(Tok, ThisTokBegin);
  ExprResult Res;

  // This is an optimization for single digits (which are very common).
  if (ActualLength == 1) {
    return ExprResult(new IntegerLiteral());
  }
  const char *ThisTokEnd = ThisTokBegin+ActualLength; 
  const char *s = ThisTokBegin;
  unsigned int radix;
  bool saw_exponent = false, saw_period = false;
  Expr *literal_expr = 0;
  
  if (*s == '0') { // parse radix
    s++;
    if ((*s == 'x' || *s == 'X') && isxdigit(s[1])) { // need 1 digit
      s++;
      radix = 16;
      while (s < ThisTokEnd) {
        if (isxdigit(*s)) {
          s++;
        } else if (*s == '.') {
          s++;
          if (saw_period) {
            Diag(Tok, diag::err_too_many_decimal_points);
            return ExprResult(true);
          } else
            saw_period = true;
        } else if (*s == 'p' || *s == 'P') { // binary exponent
          saw_exponent = true;
          s++;
          if (*s == '+' || *s == '-')  s++; // sign is optional
          if (isdigit(*s)) { 
            do { s++; } while (isdigit(*s)); // a decimal integer
          } else {
            Diag(Tok, diag::err_exponent_has_no_digits);
            return ExprResult(true);
          }
          break;
        } else
          break;
      }
      if (saw_period && !saw_exponent) { 
        Diag(Tok, diag::err_hexconstant_requires_exponent);
        return ExprResult(true);
      }
    } else {
      // For now, the radix is set to 8. If we discover that we have a
      // floating point constant, the radix will change to 10. Octal floating
      // point constants are not permitted (only decimal and hexadecimal). 
      radix = 8;
      while (s < ThisTokEnd) {
        if ((*s >= '0') && (*s <= '7')) {
          s++;
        } else if (*s == '.') {
          s++;
          if (saw_period) {
            Diag(Tok, diag::err_too_many_decimal_points);
            return ExprResult(true);
          }
          saw_period = true;
          radix = 10;
        } else if (*s == 'e' || *s == 'E') { // exponent
          saw_exponent = true;
          s++;
          if (*s == '+' || *s == '-')  s++; // sign
          if (isdigit(*s)) { 
            do { s++; } while (isdigit(*s)); // a decimal integer
          } else {
            Diag(Tok, diag::err_exponent_has_no_digits);
            return ExprResult(true);
          }
          radix = 10;
          break;
        } else
          break;
      }
    }
  } else { // the first digit is non-zero
    radix = 10;
    while (s < ThisTokEnd) {
      if (isdigit(*s)) {
        s++;
      } else if (*s == '.') {
        s++;
        if (saw_period) {
          Diag(Tok, diag::err_too_many_decimal_points);
          return ExprResult(true);
        } else
          saw_period = true;
      } else if (*s == 'e' || *s == 'E') { // exponent
        saw_exponent = true;
        s++;
        if (*s == '+' || *s == '-')  s++; // sign
        if (isdigit(*s)) { 
          do { s++; } while (isdigit(*s)); // a decimal integer
        } else {
          Diag(Tok, diag::err_exponent_has_no_digits);
          return ExprResult(true);
        }
        break;
      } else
        break;
    }
  }

  const char *suffix_start = s;
  bool invalid_suffix = false;
  
  if (saw_period || saw_exponent) {
    bool saw_float_suffix = false, saw_long_suffix = false;
        
    while (s < ThisTokEnd) {
      // parse float suffix - they can appear in any order.
      if (*s == 'f' || *s == 'F') {
        if (saw_float_suffix) {
          invalid_suffix = true;
          break;
        } else {
          saw_float_suffix = true;
          s++;
        }
      } else if (*s == 'l' || *s == 'L') {
        if (saw_long_suffix) {
          invalid_suffix = true;
          break;
        } else {
          saw_long_suffix = true;
          s++;
        }
      } else {
        invalid_suffix = true;
        break;
      }
    }
    if (invalid_suffix) {
      Diag(Tok, diag::err_invalid_suffix_float_constant, 
           std::string(suffix_start, ThisTokEnd));
      return ExprResult(true);
    }
    literal_expr = new FloatingLiteral();
  } else {
    bool saw_unsigned = false;
    int long_cnt = 0;

    // if there is no suffix, this loop won't be executed (s == ThisTokEnd)
    while (s < ThisTokEnd) {
      // parse int suffix - they can appear in any order ("ul", "lu", "llu").
      if (*s == 'u' || *s == 'U') {
        if (saw_unsigned) {
          invalid_suffix = true;
          break;
        } else {
          saw_unsigned = true;
          s++;
        }
      } else if (*s == 'l' || *s == 'L') {
        long_cnt++;
        // l's need to be adjacent and the same case.
        if ((long_cnt == 2 && (*s != *(s-1))) || long_cnt == 3) {
          invalid_suffix = true;
          break;
        } else {
          s++;
        }
      } else {
        invalid_suffix = true;
        break;
      }
    }
    if (invalid_suffix) {
      Diag(Tok, diag::err_invalid_suffix_integer_constant, 
           std::string(suffix_start, ThisTokEnd));
      return ExprResult(true);
    }
    literal_expr = new IntegerLiteral();
  }
  return ExprResult(literal_expr);
}

Action::ExprResult Sema::ParseParenExpr(SourceLocation L, SourceLocation R,
                                        ExprTy *Val) {
  return Val;
}


// Unary Operators.  'Tok' is the token for the operator.
Action::ExprResult Sema::ParseUnaryOp(SourceLocation OpLoc, tok::TokenKind Op,
                                      ExprTy *Input) {
  UnaryOperator::Opcode Opc;
  switch (Op) {
  default: assert(0 && "Unknown unary op!");
  case tok::plusplus:     Opc = UnaryOperator::PreInc; break;
  case tok::minusminus:   Opc = UnaryOperator::PreDec; break;
  case tok::amp:          Opc = UnaryOperator::AddrOf; break;
  case tok::star:         Opc = UnaryOperator::Deref; break;
  case tok::plus:         Opc = UnaryOperator::Plus; break;
  case tok::minus:        Opc = UnaryOperator::Minus; break;
  case tok::tilde:        Opc = UnaryOperator::Not; break;
  case tok::exclaim:      Opc = UnaryOperator::LNot; break;
  case tok::kw_sizeof:    Opc = UnaryOperator::SizeOf; break;
  case tok::kw___alignof: Opc = UnaryOperator::AlignOf; break;
  case tok::kw___real:    Opc = UnaryOperator::Real; break;
  case tok::kw___imag:    Opc = UnaryOperator::Imag; break;
  case tok::ampamp:       Opc = UnaryOperator::AddrLabel; break;
  case tok::kw___extension__: 
    return Input;
    //Opc = UnaryOperator::Extension;
    //break;
  }

  return new UnaryOperator((Expr*)Input, Opc);
}

Action::ExprResult Sema::
ParseSizeOfAlignOfTypeExpr(SourceLocation OpLoc, bool isSizeof, 
                           SourceLocation LParenLoc, TypeTy *Ty,
                           SourceLocation RParenLoc) {
  // If error parsing type, ignore.
  if (Ty == 0) return true;
  
  // Verify that this is a valid expression.
  TypeRef ArgTy = TypeRef::getFromOpaquePtr(Ty);
  
  if (isa<FunctionType>(ArgTy) && isSizeof) {
    // alignof(function) is allowed.
    Diag(OpLoc, diag::ext_sizeof_function_type);
    return new IntegerLiteral(/*1*/);
  } else if (ArgTy->isVoidType()) {
    Diag(OpLoc, diag::ext_sizeof_void_type, isSizeof ? "sizeof" : "__alignof");
  } else if (ArgTy->isIncompleteType()) {
    std::string TypeName;
    ArgTy->getAsString(TypeName);
    Diag(OpLoc, isSizeof ? diag::err_sizeof_incomplete_type : 
         diag::err_alignof_incomplete_type, TypeName);
    return new IntegerLiteral(/*0*/);
  }
  
  return new SizeOfAlignOfTypeExpr(isSizeof, ArgTy);
}


Action::ExprResult Sema::ParsePostfixUnaryOp(SourceLocation OpLoc, 
                                             tok::TokenKind Kind,
                                             ExprTy *Input) {
  UnaryOperator::Opcode Opc;
  switch (Kind) {
  default: assert(0 && "Unknown unary op!");
  case tok::plusplus:   Opc = UnaryOperator::PostInc; break;
  case tok::minusminus: Opc = UnaryOperator::PostDec; break;
  }
  
  return new UnaryOperator((Expr*)Input, Opc);
}

Action::ExprResult Sema::
ParseArraySubscriptExpr(ExprTy *Base, SourceLocation LLoc,
                        ExprTy *Idx, SourceLocation RLoc) {
  return new ArraySubscriptExpr((Expr*)Base, (Expr*)Idx);
}

Action::ExprResult Sema::
ParseMemberReferenceExpr(ExprTy *Base, SourceLocation OpLoc,
                         tok::TokenKind OpKind, SourceLocation MemberLoc,
                         IdentifierInfo &Member) {
  Decl *MemberDecl = 0;
  // TODO: Look up MemberDecl.
  return new MemberExpr((Expr*)Base, OpKind == tok::arrow, MemberDecl);
}

/// ParseCallExpr - Handle a call to Fn with the specified array of arguments.
/// This provides the location of the left/right parens and a list of comma
/// locations.
Action::ExprResult Sema::
ParseCallExpr(ExprTy *Fn, SourceLocation LParenLoc,
              ExprTy **Args, unsigned NumArgs,
              SourceLocation *CommaLocs, SourceLocation RParenLoc) {
  return new CallExpr((Expr*)Fn, (Expr**)Args, NumArgs);
}

Action::ExprResult Sema::
ParseCastExpr(SourceLocation LParenLoc, TypeTy *Ty,
              SourceLocation RParenLoc, ExprTy *Op) {
  // If error parsing type, ignore.
  if (Ty == 0) return true;
  return new CastExpr(TypeRef::getFromOpaquePtr(Ty), (Expr*)Op);
}



// Binary Operators.  'Tok' is the token for the operator.
Action::ExprResult Sema::ParseBinOp(SourceLocation TokLoc, tok::TokenKind Kind,
                                    ExprTy *LHS, ExprTy *RHS) {
  BinaryOperator::Opcode Opc;
  switch (Kind) {
  default: assert(0 && "Unknown binop!");
  case tok::star:                 Opc = BinaryOperator::Mul; break;
  case tok::slash:                Opc = BinaryOperator::Div; break;
  case tok::percent:              Opc = BinaryOperator::Rem; break;
  case tok::plus:                 Opc = BinaryOperator::Add; break;
  case tok::minus:                Opc = BinaryOperator::Sub; break;
  case tok::lessless:             Opc = BinaryOperator::Shl; break;
  case tok::greatergreater:       Opc = BinaryOperator::Shr; break;
  case tok::lessequal:            Opc = BinaryOperator::LE; break;
  case tok::less:                 Opc = BinaryOperator::LT; break;
  case tok::greaterequal:         Opc = BinaryOperator::GE; break;
  case tok::greater:              Opc = BinaryOperator::GT; break;
  case tok::exclaimequal:         Opc = BinaryOperator::NE; break;
  case tok::equalequal:           Opc = BinaryOperator::EQ; break;
  case tok::amp:                  Opc = BinaryOperator::And; break;
  case tok::caret:                Opc = BinaryOperator::Xor; break;
  case tok::pipe:                 Opc = BinaryOperator::Or; break;
  case tok::ampamp:               Opc = BinaryOperator::LAnd; break;
  case tok::pipepipe:             Opc = BinaryOperator::LOr; break;
  case tok::equal:                Opc = BinaryOperator::Assign; break;
  case tok::starequal:            Opc = BinaryOperator::MulAssign; break;
  case tok::slashequal:           Opc = BinaryOperator::DivAssign; break;
  case tok::percentequal:         Opc = BinaryOperator::RemAssign; break;
  case tok::plusequal:            Opc = BinaryOperator::AddAssign; break;
  case tok::minusequal:           Opc = BinaryOperator::SubAssign; break;
  case tok::lesslessequal:        Opc = BinaryOperator::ShlAssign; break;
  case tok::greatergreaterequal:  Opc = BinaryOperator::ShrAssign; break;
  case tok::ampequal:             Opc = BinaryOperator::AndAssign; break;
  case tok::caretequal:           Opc = BinaryOperator::XorAssign; break;
  case tok::pipeequal:            Opc = BinaryOperator::OrAssign; break;
  case tok::comma:                Opc = BinaryOperator::Comma; break;
  }
  
  return new BinaryOperator((Expr*)LHS, (Expr*)RHS, Opc);
}

/// ParseConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
/// in the case of a the GNU conditional expr extension.
Action::ExprResult Sema::ParseConditionalOp(SourceLocation QuestionLoc, 
                                            SourceLocation ColonLoc,
                                            ExprTy *Cond, ExprTy *LHS,
                                            ExprTy *RHS) {
  return new ConditionalOperator((Expr*)Cond, (Expr*)LHS, (Expr*)RHS);
}

