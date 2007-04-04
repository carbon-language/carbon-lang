//===--- LiteralSupport.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Steve Naroff and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the NumericLiteralParser interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LITERALSUPPORT_H
#define LLVM_CLANG_LITERALSUPPORT_H

#include <string>
#include "llvm/ADT/SmallString.h"

namespace llvm {
  class APInt;
namespace clang {

class Diagnostic;
class Preprocessor;
class LexerToken;
class SourceLocation;
class TargetInfo;
    
class NumericLiteralParser {
  Preprocessor &PP; // needed for diagnostics
  
  const char *const ThisTokBegin;
  const char *const ThisTokEnd;
  const char *DigitsBegin, *SuffixBegin; // markers
  const char *s; // cursor
  
  unsigned int radix;
  
  bool saw_exponent, saw_period;
  bool saw_float_suffix;
  
public:
  NumericLiteralParser(const char *begin, const char *end,
                       SourceLocation Loc, Preprocessor &PP);
  bool hadError;
  bool isUnsigned;
  bool isLong;
  bool isLongLong;
  
  bool isIntegerLiteral() const { 
    return !saw_period && !saw_exponent ? true : false;
  }
  bool isFloatingLiteral() const {
    return saw_period || saw_exponent ? true : false;
  }
  bool hasSuffix() const {
    return SuffixBegin != ThisTokEnd;
  }
  /// getIntegerValue - Convert the string into a number. At this point, we 
  /// know the digit characters are valid (0...9, a...f, A...F). We don't know
  /// how many bits are needed to store the number. Sizing of the integer
  /// type (int, unsigned, long, unsigned long, long long, unsigned long long) 
  /// will be done elsewhere - the size computation is target dependent. We  
  /// return true if the value fit into "val", false otherwise. 
  /// NOTE: The api of these returns an inverted value for 'overflow' than the
  /// version below does.
  bool GetIntegerValue(uintmax_t &val);
  bool GetIntegerValue(int &val);
  
  /// GetIntegerValue - Convert this numeric literal value to an APInt that
  /// matches Val's input width.  If there is an overflow, set Val to the low
  /// bits of the result and return true.  Otherwise, return false.
  bool GetIntegerValue(APInt &Val);

private:  
  void Diag(SourceLocation Loc, unsigned DiagID, 
            const std::string &M = std::string());
  
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
};

class StringLiteralParser {
  Preprocessor &PP;
  TargetInfo &Target;
  
  unsigned MaxTokenLength;
  unsigned SizeBound;
  unsigned wchar_tByteWidth;
  SmallString<512> ResultBuf;
  char *ResultPtr; // cursor
public:
  StringLiteralParser(const LexerToken *StringToks, unsigned NumStringToks,
                      Preprocessor &PP, TargetInfo &T);
  bool hadError;
  bool AnyWide;
  
  const char *GetString() { return &ResultBuf[0]; }
  unsigned GetStringLength() { return ResultPtr-&ResultBuf[0]; }
private:
  void Diag(SourceLocation Loc, unsigned DiagID, 
            const std::string &M = std::string());

  /// HexDigitValue - Return the value of the specified hex digit, or -1 if it's
  /// not valid.
  static int HexDigitValue(char C) {
    if (C >= '0' && C <= '9') return C-'0';
    if (C >= 'a' && C <= 'f') return C-'a'+10;
    if (C >= 'A' && C <= 'F') return C-'A'+10;
    return -1;
  }
};
  
}  // end namespace clang
}  // end namespace llvm

#endif