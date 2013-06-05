//===--- Encoding.h - Format C++ code -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Contains functions for text encoding manipulation. Supports UTF-8,
/// 8-bit encodings and escape sequences in C++ string literals.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FORMAT_ENCODING_H
#define LLVM_CLANG_FORMAT_ENCODING_H

#include "clang/Basic/LLVM.h"
#include "llvm/Support/ConvertUTF.h"

namespace clang {
namespace format {
namespace encoding {

enum Encoding {
  Encoding_UTF8,
  Encoding_Unknown // We treat all other encodings as 8-bit encodings.
};

/// \brief Detects encoding of the Text. If the Text can be decoded using UTF-8,
/// it is considered UTF8, otherwise we treat it as some 8-bit encoding.
inline Encoding detectEncoding(StringRef Text) {
  const UTF8 *Ptr = reinterpret_cast<const UTF8 *>(Text.begin());
  const UTF8 *BufEnd = reinterpret_cast<const UTF8 *>(Text.end());
  if (::isLegalUTF8String(&Ptr, BufEnd))
    return Encoding_UTF8;
  return Encoding_Unknown;
}

inline unsigned getCodePointCountUTF8(StringRef Text) {
  unsigned CodePoints = 0;
  for (size_t i = 0, e = Text.size(); i < e; i += getNumBytesForUTF8(Text[i])) {
    ++CodePoints;
  }
  return CodePoints;
}

/// \brief Gets the number of code points in the Text using the specified
/// Encoding.
inline unsigned getCodePointCount(StringRef Text, Encoding Encoding) {
  switch (Encoding) {
    case Encoding_UTF8:
      return getCodePointCountUTF8(Text);
    default:
      return Text.size();
  }
}

/// \brief Gets the number of bytes in a sequence representing a single
/// codepoint and starting with FirstChar in the specified Encoding.
inline unsigned getCodePointNumBytes(char FirstChar, Encoding Encoding) {
  switch (Encoding) {
    case Encoding_UTF8:
      return getNumBytesForUTF8(FirstChar);
    default:
      return 1;
  }
}

inline bool isOctDigit(char c) {
  return '0' <= c && c <= '7';
}

inline bool isHexDigit(char c) {
  return ('0' <= c && c <= '9') || ('a' <= c && c <= 'f') ||
         ('A' <= c && c <= 'F');
}

/// \brief Gets the length of an escape sequence inside a C++ string literal.
/// Text should span from the beginning of the escape sequence (starting with a
/// backslash) to the end of the string literal.
inline unsigned getEscapeSequenceLength(StringRef Text) {
  assert(Text[0] == '\\');
  if (Text.size() < 2)
    return 1;

  switch (Text[1]) {
  case 'u':
    return 6;
  case 'U':
    return 10;
  case 'x': {
    unsigned I = 2; // Point after '\x'.
    while (I < Text.size() && isHexDigit(Text[I]))
      ++I;
    return I;
  }
  default:
    if (isOctDigit(Text[1])) {
      unsigned I = 1;
      while (I < Text.size() && I < 4 && isOctDigit(Text[I]))
        ++I;
      return I;
    }
    return 2;
  }
}

} // namespace encoding
} // namespace format
} // namespace clang

#endif // LLVM_CLANG_FORMAT_ENCODING_H
