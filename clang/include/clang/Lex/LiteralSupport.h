//===--- LiteralSupport.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the NumericLiteralParser, CharLiteralParser, and
// StringLiteralParser interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LITERALSUPPORT_H
#define CLANG_LITERALSUPPORT_H

#include <string>
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/System/DataTypes.h"

namespace clang {

class Diagnostic;
class Preprocessor;
class Token;
class SourceLocation;
class TargetInfo;

/// NumericLiteralParser - This performs strict semantic analysis of the content
/// of a ppnumber, classifying it as either integer, floating, or erroneous,
/// determines the radix of the value and can convert it to a useful value.
class NumericLiteralParser {
  Preprocessor &PP; // needed for diagnostics

  const char *const ThisTokBegin;
  const char *const ThisTokEnd;
  const char *DigitsBegin, *SuffixBegin; // markers
  const char *s; // cursor

  unsigned radix;

  bool saw_exponent, saw_period;

public:
  NumericLiteralParser(const char *begin, const char *end,
                       SourceLocation Loc, Preprocessor &PP);
  bool hadError;
  bool isUnsigned;
  bool isLong;        // This is *not* set for long long.
  bool isLongLong;
  bool isFloat;       // 1.0f
  bool isImaginary;   // 1.0i
  bool isMicrosoftInteger;  // Microsoft suffix extension i8, i16, i32, or i64.

  bool isIntegerLiteral() const {
    return !saw_period && !saw_exponent;
  }
  bool isFloatingLiteral() const {
    return saw_period || saw_exponent;
  }
  bool hasSuffix() const {
    return SuffixBegin != ThisTokEnd;
  }

  unsigned getRadix() const { return radix; }

  /// GetIntegerValue - Convert this numeric literal value to an APInt that
  /// matches Val's input width.  If there is an overflow (i.e., if the unsigned
  /// value read is larger than the APInt's bits will hold), set Val to the low
  /// bits of the result and return true.  Otherwise, return false.
  bool GetIntegerValue(llvm::APInt &Val);

  /// GetFloatValue - Convert this numeric literal to a floating value, using
  /// the specified APFloat fltSemantics (specifying float, double, etc).
  /// The optional bool isExact (passed-by-reference) has its value
  /// set to true if the returned APFloat can represent the number in the
  /// literal exactly, and false otherwise.
  llvm::APFloat::opStatus GetFloatValue(llvm::APFloat &Result);

private:

  void ParseNumberStartingWithZero(SourceLocation TokLoc);

  /// SkipHexDigits - Read and skip over any hex digits, up to End.
  /// Return a pointer to the first non-hex digit or End.
  const char *SkipHexDigits(const char *ptr) {
    while (ptr != ThisTokEnd && isxdigit(*ptr))
      ptr++;
    return ptr;
  }

  /// SkipOctalDigits - Read and skip over any octal digits, up to End.
  /// Return a pointer to the first non-hex digit or End.
  const char *SkipOctalDigits(const char *ptr) {
    while (ptr != ThisTokEnd && ((*ptr >= '0') && (*ptr <= '7')))
      ptr++;
    return ptr;
  }

  /// SkipDigits - Read and skip over any digits, up to End.
  /// Return a pointer to the first non-hex digit or End.
  const char *SkipDigits(const char *ptr) {
    while (ptr != ThisTokEnd && isdigit(*ptr))
      ptr++;
    return ptr;
  }

  /// SkipBinaryDigits - Read and skip over any binary digits, up to End.
  /// Return a pointer to the first non-binary digit or End.
  const char *SkipBinaryDigits(const char *ptr) {
    while (ptr != ThisTokEnd && (*ptr == '0' || *ptr == '1'))
      ptr++;
    return ptr;
  }

};

/// CharLiteralParser - Perform interpretation and semantic analysis of a
/// character literal.
class CharLiteralParser {
  uint64_t Value;
  bool IsWide;
  bool IsMultiChar;
  bool HadError;
public:
  CharLiteralParser(const char *begin, const char *end,
                    SourceLocation Loc, Preprocessor &PP);

  bool hadError() const { return HadError; }
  bool isWide() const { return IsWide; }
  bool isMultiChar() const { return IsMultiChar; }
  uint64_t getValue() const { return Value; }
};

/// StringLiteralParser - This decodes string escape characters and performs
/// wide string analysis and Translation Phase #6 (concatenation of string
/// literals) (C99 5.1.1.2p1).
class StringLiteralParser {
  Preprocessor &PP;

  unsigned MaxTokenLength;
  unsigned SizeBound;
  unsigned wchar_tByteWidth;
  llvm::SmallString<512> ResultBuf;
  char *ResultPtr; // cursor
public:
  StringLiteralParser(const Token *StringToks, unsigned NumStringToks,
                      Preprocessor &PP);
  bool hadError;
  bool AnyWide;
  bool Pascal;

  const char *GetString() { return &ResultBuf[0]; }
  unsigned GetStringLength() const { return ResultPtr-&ResultBuf[0]; }

  unsigned GetNumStringChars() const {
    if (AnyWide)
      return GetStringLength() / wchar_tByteWidth;
    return GetStringLength();
  }
  /// getOffsetOfStringByte - This function returns the offset of the
  /// specified byte of the string data represented by Token.  This handles
  /// advancing over escape sequences in the string.
  static unsigned getOffsetOfStringByte(const Token &TheTok, unsigned ByteNo,
                                        Preprocessor &PP);
};

}  // end namespace clang

#endif
