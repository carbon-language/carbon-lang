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
#include "clang/AST/Expr.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/Diagnostic.h"
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

/// ParseStringExpr - The specified tokens were lexed as pasted string
/// fragments (e.g. "foo" "bar" L"baz").  The result string has to handle string
/// concatenation ([C99 5.1.1.2, translation phase #6]), so it may come from
/// multiple tokens.  However, the common case is that StringToks points to one
/// string.
/// 
Action::ExprResult
Sema::ParseStringExpr(const LexerToken *StringToks, unsigned NumStringToks) {
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
    wchar_tByteWidth =
      PP.getTargetInfo().getWCharWidth(StringToks[0].getLocation());
  
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
  
  SmallVector<SourceLocation, 4> StringTokLocs;
  for (unsigned i = 0; i != NumStringToks; ++i)
    StringTokLocs.push_back(StringToks[i].getLocation());
  
  // FIXME: use factory.
  
  // Pass &StringTokLocs[0], StringTokLocs.size() to factory!
  return new StringExpr(&ResultBuf[0], ResultPtr-&ResultBuf[0], AnyWide);
}

