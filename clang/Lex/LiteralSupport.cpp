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

bool NumericLiteralParser::GetIntegerValue(uintmax_t &val) {
  uintmax_t cutoff = UINTMAX_MAX / radix;
  int cutlim = UINTMAX_MAX % radix;
  char c;
  
  val = 0;
  s = DigitsBegin;
  while (s < SuffixBegin) {
    c = *s++;
    if (c >= '0' && c <= '9')
      c -= '0';
    else if (c >= 'A' && c <= 'F') // 10...15
      c -= 'A' - 10;
    else if (c >= 'a' && c <= 'f') // 10...15
      c -= 'a' - 10;
    
    if (val > cutoff || (val == cutoff && c > cutlim)) {
      return false; // Overflow!
    } else {
      val *= radix;
      val += c;
    }
  }
  return true;
}

bool NumericLiteralParser::GetIntegerValue(int &val) {
  intmax_t cutoff = INT_MAX / radix;
  int cutlim = INT_MAX % radix;
  char c;
  
  val = 0;
  s = DigitsBegin;
  while (s < SuffixBegin) {
    c = *s++;
    if (c >= '0' && c <= '9')
      c -= '0';
    else if (c >= 'A' && c <= 'F') // 10...15
      c -= 'A' - 10;
    else if (c >= 'a' && c <= 'f') // 10...15
      c -= 'a' - 10;
    
    if (val > cutoff || (val == cutoff && c > cutlim)) {
      return false; // Overflow!
    } else {
      val *= radix;
      val += c;
    }
  }
  return true;
}
