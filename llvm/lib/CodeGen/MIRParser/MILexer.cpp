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

  char peek(int I = 0) const { return End - Ptr <= I ? 0 : Ptr[I]; }

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
  auto Identifier = Range.upto(C);
  Token = MIToken(Identifier == "_" ? MIToken::underscore : MIToken::Identifier,
                  Identifier);
  return C;
}

static Cursor lexPercent(Cursor C, MIToken &Token) {
  auto Range = C;
  C.advance(); // Skip '%'
  while (isIdentifierChar(C.peek()))
    C.advance();
  Token = MIToken(MIToken::NamedRegister, Range.upto(C));
  return C;
}

static Cursor lexIntegerLiteral(Cursor C, MIToken &Token) {
  auto Range = C;
  C.advance();
  while (isdigit(C.peek()))
    C.advance();
  StringRef StrVal = Range.upto(C);
  Token = MIToken(MIToken::IntegerLiteral, StrVal, APSInt(StrVal));
  return C;
}

static MIToken::TokenKind symbolToken(char C) {
  switch (C) {
  case ',':
    return MIToken::comma;
  case '=':
    return MIToken::equal;
  default:
    return MIToken::Error;
  }
}

static Cursor lexSymbol(Cursor C, MIToken::TokenKind Kind, MIToken &Token) {
  auto Range = C;
  C.advance();
  Token = MIToken(Kind, Range.upto(C));
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
  if (Char == '%')
    return lexPercent(C, Token).remaining();
  if (isdigit(Char) || (Char == '-' && isdigit(C.peek(1))))
    return lexIntegerLiteral(C, Token).remaining();
  MIToken::TokenKind Kind = symbolToken(Char);
  if (Kind != MIToken::Error)
    return lexSymbol(C, Kind, Token).remaining();
  Token = MIToken(MIToken::Error, C.remaining());
  ErrorCallback(C.location(),
                Twine("unexpected character '") + Twine(Char) + "'");
  return C.remaining();
}
