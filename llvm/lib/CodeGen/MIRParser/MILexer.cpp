//===- MILexer.cpp - Machine instructions lexer implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the lexing of machine instructions.
//
//===----------------------------------------------------------------------===//

#include "MILexer.h"
#include "llvm/ADT/Twine.h"
#include <cctype>

using namespace llvm;

namespace {

/// This class provides a way to iterate and get characters from the source
/// string.
class Cursor {
  const char *Ptr;
  const char *End;

public:
  explicit Cursor(StringRef Str) {
    Ptr = Str.data();
    End = Ptr + Str.size();
  }

  bool isEOF() const { return Ptr == End; }

  char peek() const { return isEOF() ? 0 : *Ptr; }

  void advance() { ++Ptr; }

  StringRef remaining() const { return StringRef(Ptr, End - Ptr); }

  StringRef upto(Cursor C) const {
    assert(C.Ptr >= Ptr && C.Ptr <= End);
    return StringRef(Ptr, C.Ptr - Ptr);
  }

  StringRef::iterator location() const { return Ptr; }
};

} // end anonymous namespace

/// Skip the leading whitespace characters and return the updated cursor.
static Cursor skipWhitespace(Cursor C) {
  while (isspace(C.peek()))
    C.advance();
  return C;
}

static bool isIdentifierChar(char C) {
  return isalpha(C) || isdigit(C) || C == '_' || C == '-' || C == '.';
}

static Cursor lexIdentifier(Cursor C, MIToken &Token) {
  auto Range = C;
  while (isIdentifierChar(C.peek()))
    C.advance();
  Token = MIToken(MIToken::Identifier, Range.upto(C));
  return C;
}

StringRef llvm::lexMIToken(
    StringRef Source, MIToken &Token,
    function_ref<void(StringRef::iterator Loc, const Twine &)> ErrorCallback) {
  auto C = skipWhitespace(Cursor(Source));
  if (C.isEOF()) {
    Token = MIToken(MIToken::Eof, C.remaining());
    return C.remaining();
  }

  auto Char = C.peek();
  if (isalpha(Char) || Char == '_')
    return lexIdentifier(C, Token).remaining();
  Token = MIToken(MIToken::Error, C.remaining());
  ErrorCallback(C.location(),
                Twine("unexpected character '") + Twine(Char) + "'");
  return C.remaining();
}
