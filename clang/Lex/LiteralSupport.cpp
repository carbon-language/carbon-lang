//===--- LiteralSupport.cpp - Code to parse and process literals-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Steve Naroff and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the NumericLiteralParser interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;
using namespace clang;

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

NumericLiteralParser::
NumericLiteralParser(const char *begin, const char *end,
                     SourceLocation TokLoc, Preprocessor &pp) :
  PP(pp), ThisTokBegin(begin), ThisTokEnd(end)
{
  s = DigitsBegin = begin;
  saw_exponent = false;
  saw_period = false;
  saw_float_suffix = false;
  isLong = false;
  isUnsigned = false;
  isLongLong = false;
  hadError = false;
  
  if (*s == '0') { // parse radix
    s++;
    if ((*s == 'x' || *s == 'X') && (isxdigit(s[1]) || s[1] == '.')) {
      s++;
      radix = 16;
      DigitsBegin = s;
      s = SkipHexDigits(s);
      if (s == ThisTokEnd) {
      } else if (*s == '.') {
        s++;
        saw_period = true;
        s = SkipHexDigits(s);
      }
      // A binary exponent can appear with or with a '.'. If dotted, the
      // binary exponent is required. 
      if (*s == 'p' || *s == 'P') { 
        s++;
        saw_exponent = true;
        if (*s == '+' || *s == '-')  s++; // sign
        const char *first_non_digit = SkipDigits(s);
        if (first_non_digit == s) {
          Diag(TokLoc, diag::err_exponent_has_no_digits);
          return;
        } else {
          s = first_non_digit;
        }
      } else if (saw_period) {
        Diag(TokLoc, diag::err_hexconstant_requires_exponent);
        return;
      }
    } else {
      // For now, the radix is set to 8. If we discover that we have a
      // floating point constant, the radix will change to 10. Octal floating
      // point constants are not permitted (only decimal and hexadecimal). 
      radix = 8;
      DigitsBegin = s;
      s = SkipOctalDigits(s);
      if (s == ThisTokEnd) {
      } else if (*s == '.') {
        s++;
        radix = 10;
        saw_period = true;
        s = SkipDigits(s);
      }
      if (*s == 'e' || *s == 'E') { // exponent
        s++;
        radix = 10;
        saw_exponent = true;
        if (*s == '+' || *s == '-')  s++; // sign
        const char *first_non_digit = SkipDigits(s);
        if (first_non_digit == s) {
          Diag(TokLoc, diag::err_exponent_has_no_digits);
          return;
        } else {
          s = first_non_digit;
        }
      }
    }
  } else { // the first digit is non-zero
    radix = 10;
    s = SkipDigits(s);
    if (s == ThisTokEnd) {
    } else if (*s == '.') {
      s++;
      saw_period = true;
      s = SkipDigits(s);
    } 
    if (*s == 'e' || *s == 'E') { // exponent
      s++;
      saw_exponent = true;
      if (*s == '+' || *s == '-')  s++; // sign
      const char *first_non_digit = SkipDigits(s);
      if (first_non_digit == s) {
        Diag(TokLoc, diag::err_exponent_has_no_digits);
        return;
      } else {
        s = first_non_digit;
      }
    }
  }

  SuffixBegin = s;

  if (saw_period || saw_exponent) {    
    if (s < ThisTokEnd) { // parse size suffix (float, long double)
      if (*s == 'f' || *s == 'F') {
        saw_float_suffix = true;
        s++;
      } else if (*s == 'l' || *s == 'L') {
        isLong = true;
        s++;
      }
      if (s != ThisTokEnd) {
        Diag(TokLoc, diag::err_invalid_suffix_float_constant, 
             std::string(SuffixBegin, ThisTokEnd));
        return;
      }
    }
  } else {    
    if (s < ThisTokEnd) {
      // parse int suffix - they can appear in any order ("ul", "lu", "llu").
      if (*s == 'u' || *s == 'U') {
        s++;
        isUnsigned = true; // unsigned

        if ((s < ThisTokEnd) && (*s == 'l' || *s == 'L')) {
          s++;
          // handle "long long" type - l's need to be adjacent and same case.
          if ((s < ThisTokEnd) && (*s == *(s-1))) {
            isLongLong = true; // unsigned long long
            s++;
          } else {
            isLong = true; // unsigned long 
          }
        }
      } else if (*s == 'l' || *s == 'L') {
        s++;
        // handle "long long" types - l's need to be adjacent and same case.
        if ((s < ThisTokEnd) && (*s == *(s-1))) {
          s++;
          if ((s < ThisTokEnd) && (*s == 'u' || *s == 'U')) {
            isUnsigned = true; // unsigned long long
            s++;
          } else {
            isLongLong = true; // long long
          }
        } else { // handle "long" types
          if ((s < ThisTokEnd) && (*s == 'u' || *s == 'U')) {
            isUnsigned = true; // unsigned  long
            s++;
          } else {
            isLong = true; // long 
          }
        }
      } 
      if (s != ThisTokEnd) {
        Diag(TokLoc, diag::err_invalid_suffix_integer_constant, 
             std::string(SuffixBegin, ThisTokEnd));
        return;
      }
    }
  }
}

static unsigned HexLetterToVal(char c) {
  if (c >= '0' && c <= '9')
    return c - '0';
  else if (c >= 'A' && c <= 'F') 
    return c - 'A' - 10; 
  else
    assert(c >= 'a' && c <= 'f' && "Lexer scanning error");
  return c - 'a' - 10;
}

bool NumericLiteralParser::GetIntegerValue(uintmax_t &val) {
  uintmax_t max_value = UINTMAX_MAX / radix;
  unsigned max_digit = UINTMAX_MAX % radix;
  
  val = 0;
  s = DigitsBegin;
  while (s < SuffixBegin) {
    unsigned C = HexLetterToVal(*s++);
    
    if (val > max_value || (val == max_value && C > max_digit)) {
      return false; // Overflow!
    } else {
      val *= radix;
      val += C;
    }
  }
  return true;
}

bool NumericLiteralParser::GetIntegerValue(int &val) {
  intmax_t max_value = INT_MAX / radix;
  unsigned max_digit = INT_MAX % radix;
  
  val = 0;
  s = DigitsBegin;
  while (s < SuffixBegin) {
    unsigned C = HexLetterToVal(*s++);
    
    if (val > max_value || (val == max_value && C > max_digit)) {
      return false; // Overflow!
    } else {
      val *= radix;
      val += C;
    }
  }
  return true;
}

/// GetIntegerValue - Convert this numeric literal value to an APInt that
/// matches Val's input width.  If there is an overflow, saturate Val to zero
/// and return false.  Otherwise, set Val and return true.
bool NumericLiteralParser::GetIntegerValue(APInt &Val) {
  Val = 0;
  s = DigitsBegin;

  // FIXME: This doesn't handle sign right, doesn't autopromote to wider
  // integer, and is generally not conformant.
  APInt RadixVal(Val.getBitWidth(), radix);
  APInt CharVal(Val.getBitWidth(), 0);
  APInt OldVal = Val;
  while (s < SuffixBegin) {
    unsigned C = HexLetterToVal(*s++);
    
    // If this letter is out of bound for this radix, reject it.
    if (C >= radix) { Val = 0; return false; }
    
    CharVal = C;
    
    OldVal = Val;
    Val *= RadixVal;
    Val += CharVal;
    if (OldVal.ugt(Val))
      return false; // Overflow!
  }
  return true;
}


void NumericLiteralParser::Diag(SourceLocation Loc, unsigned DiagID, 
          const std::string &M) {
  PP.Diag(Loc, DiagID, M);
  hadError = true;
}

///       string-literal: [C99 6.4.5]
///          " [s-char-sequence] "
///         L" [s-char-sequence] "
///       s-char-sequence:
///         s-char
///         s-char-sequence s-char
///       s-char:
///         any source character except the double quote ",
///           backslash \, or newline character
///         escape-character
///         universal-character-name
///       escape-character: [C99 6.4.4.4]
///         \ escape-code
///         universal-character-name
///       escape-code:
///         character-escape-code
///         octal-escape-code
///         hex-escape-code
///       character-escape-code: one of
///         n t b r f v a
///         \ ' " ?
///       octal-escape-code:
///         octal-digit
///         octal-digit octal-digit
///         octal-digit octal-digit octal-digit
///       hex-escape-code:
///         x hex-digit
///         hex-escape-code hex-digit
///       universal-character-name:
///         \u hex-quad
///         \U hex-quad hex-quad
///       hex-quad:
///         hex-digit hex-digit hex-digit hex-digit

StringLiteralParser::
StringLiteralParser(const LexerToken *StringToks, unsigned NumStringToks,
                    Preprocessor &pp, TargetInfo &t) : 
  PP(pp), Target(t) 
{
  // Scan all of the string portions, remember the max individual token length,
  // computing a bound on the concatenated string length, and see whether any
  // piece is a wide-string.  If any of the string portions is a wide-string
  // literal, the result is a wide-string literal [C99 6.4.5p4].
  MaxTokenLength = StringToks[0].getLength();
  SizeBound = StringToks[0].getLength()-2;  // -2 for "".
  AnyWide = StringToks[0].getKind() == tok::wide_string_literal;
  
  hadError = false;
  
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
  wchar_tByteWidth = ~0U;
  if (AnyWide)
    wchar_tByteWidth = Target.getWCharWidth(StringToks[0].getLocation());
  
  // The output buffer size needs to be large enough to hold wide characters.
  // This is a worst-case assumption which basically corresponds to L"" "long".
  if (AnyWide)
    SizeBound *= wchar_tByteWidth;
  
  // Size the temporary buffer to hold the result string data.
  ResultBuf.resize(SizeBound);
  
  // Likewise, but for each string piece.
  SmallString<512> TokenBuf;
  TokenBuf.resize(MaxTokenLength);
  
  // Loop over all the strings, getting their spelling, and expanding them to
  // wide strings as appropriate.
  ResultPtr = &ResultBuf[0];   // Next byte to fill in.
  
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
}

void StringLiteralParser::Diag(SourceLocation Loc, unsigned DiagID, 
                               const std::string &M) {
  PP.Diag(Loc, DiagID, M);
  hadError = true;
}

