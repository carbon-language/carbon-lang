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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
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
  Cursor(NoneType) : Ptr(nullptr), End(nullptr) {}

  explicit Cursor(StringRef Str) {
    Ptr = Str.data();
    End = Ptr + Str.size();
  }

  bool isEOF() const { return Ptr == End; }

  char peek(int I = 0) const { return End - Ptr <= I ? 0 : Ptr[I]; }

  void advance(unsigned I = 1) { Ptr += I; }

  StringRef remaining() const { return StringRef(Ptr, End - Ptr); }

  StringRef upto(Cursor C) const {
    assert(C.Ptr >= Ptr && C.Ptr <= End);
    return StringRef(Ptr, C.Ptr - Ptr);
  }

  StringRef::iterator location() const { return Ptr; }

  operator bool() const { return Ptr != nullptr; }
};

} // end anonymous namespace

/// Skip the leading whitespace characters and return the updated cursor.
static Cursor skipWhitespace(Cursor C) {
  while (isspace(C.peek()))
    C.advance();
  return C;
}

/// Return true if the given character satisfies the following regular
/// expression: [-a-zA-Z$._0-9]
static bool isIdentifierChar(char C) {
  return isalpha(C) || isdigit(C) || C == '_' || C == '-' || C == '.' ||
         C == '$';
}

void MIToken::unescapeQuotedStringValue(std::string &Str) const {
  assert(isStringValueQuoted() && "String value isn't quoted");
  StringRef Value = Range.drop_front(StringOffset);
  assert(Value.front() == '"' && Value.back() == '"');
  Cursor C = Cursor(Value.substr(1, Value.size() - 2));

  Str.clear();
  Str.reserve(C.remaining().size());
  while (!C.isEOF()) {
    char Char = C.peek();
    if (Char == '\\') {
      if (C.peek(1) == '\\') {
        // Two '\' become one
        Str += '\\';
        C.advance(2);
        continue;
      }
      if (isxdigit(C.peek(1)) && isxdigit(C.peek(2))) {
        Str += hexDigitValue(C.peek(1)) * 16 + hexDigitValue(C.peek(2));
        C.advance(3);
        continue;
      }
    }
    Str += Char;
    C.advance();
  }
}

/// Lex a string constant using the following regular expression: \"[^\"]*\"
static Cursor lexStringConstant(
    Cursor C,
    function_ref<void(StringRef::iterator Loc, const Twine &)> ErrorCallback) {
  assert(C.peek() == '"');
  for (C.advance(); C.peek() != '"'; C.advance()) {
    if (C.isEOF()) {
      ErrorCallback(
          C.location(),
          "end of machine instruction reached before the closing '\"'");
      return None;
    }
  }
  C.advance();
  return C;
}

static MIToken::TokenKind getIdentifierKind(StringRef Identifier) {
  return StringSwitch<MIToken::TokenKind>(Identifier)
      .Case("_", MIToken::underscore)
      .Case("implicit", MIToken::kw_implicit)
      .Case("implicit-def", MIToken::kw_implicit_define)
      .Case("dead", MIToken::kw_dead)
      .Case("killed", MIToken::kw_killed)
      .Case("undef", MIToken::kw_undef)
      .Case("frame-setup", MIToken::kw_frame_setup)
      .Case("debug-location", MIToken::kw_debug_location)
      .Case(".cfi_offset", MIToken::kw_cfi_offset)
      .Case(".cfi_def_cfa_register", MIToken::kw_cfi_def_cfa_register)
      .Case(".cfi_def_cfa_offset", MIToken::kw_cfi_def_cfa_offset)
      .Default(MIToken::Identifier);
}

static Cursor maybeLexIdentifier(Cursor C, MIToken &Token) {
  if (!isalpha(C.peek()) && C.peek() != '_' && C.peek() != '.')
    return None;
  auto Range = C;
  while (isIdentifierChar(C.peek()))
    C.advance();
  auto Identifier = Range.upto(C);
  Token = MIToken(getIdentifierKind(Identifier), Identifier);
  return C;
}

static Cursor maybeLexMachineBasicBlock(
    Cursor C, MIToken &Token,
    function_ref<void(StringRef::iterator Loc, const Twine &)> ErrorCallback) {
  if (!C.remaining().startswith("%bb."))
    return None;
  auto Range = C;
  C.advance(4); // Skip '%bb.'
  if (!isdigit(C.peek())) {
    Token = MIToken(MIToken::Error, C.remaining());
    ErrorCallback(C.location(), "expected a number after '%bb.'");
    return C;
  }
  auto NumberRange = C;
  while (isdigit(C.peek()))
    C.advance();
  StringRef Number = NumberRange.upto(C);
  unsigned StringOffset = 4 + Number.size(); // Drop '%bb.<id>'
  if (C.peek() == '.') {
    C.advance(); // Skip '.'
    ++StringOffset;
    while (isIdentifierChar(C.peek()))
      C.advance();
  }
  Token = MIToken(MIToken::MachineBasicBlock, Range.upto(C), APSInt(Number),
                  StringOffset);
  return C;
}

static Cursor maybeLexIndex(Cursor C, MIToken &Token, StringRef Rule,
                            MIToken::TokenKind Kind) {
  if (!C.remaining().startswith(Rule) || !isdigit(C.peek(Rule.size())))
    return None;
  auto Range = C;
  C.advance(Rule.size());
  auto NumberRange = C;
  while (isdigit(C.peek()))
    C.advance();
  Token = MIToken(Kind, Range.upto(C), APSInt(NumberRange.upto(C)));
  return C;
}

static Cursor maybeLexIndexAndName(Cursor C, MIToken &Token, StringRef Rule,
                                   MIToken::TokenKind Kind) {
  if (!C.remaining().startswith(Rule) || !isdigit(C.peek(Rule.size())))
    return None;
  auto Range = C;
  C.advance(Rule.size());
  auto NumberRange = C;
  while (isdigit(C.peek()))
    C.advance();
  StringRef Number = NumberRange.upto(C);
  unsigned StringOffset = Rule.size() + Number.size();
  if (C.peek() == '.') {
    C.advance();
    ++StringOffset;
    while (isIdentifierChar(C.peek()))
      C.advance();
  }
  Token = MIToken(Kind, Range.upto(C), APSInt(Number), StringOffset);
  return C;
}

static Cursor maybeLexJumpTableIndex(Cursor C, MIToken &Token) {
  return maybeLexIndex(C, Token, "%jump-table.", MIToken::JumpTableIndex);
}

static Cursor maybeLexStackObject(Cursor C, MIToken &Token) {
  return maybeLexIndexAndName(C, Token, "%stack.", MIToken::StackObject);
}

static Cursor maybeLexFixedStackObject(Cursor C, MIToken &Token) {
  return maybeLexIndex(C, Token, "%fixed-stack.", MIToken::FixedStackObject);
}

static Cursor maybeLexConstantPoolItem(Cursor C, MIToken &Token) {
  return maybeLexIndex(C, Token, "%const.", MIToken::ConstantPoolItem);
}

static Cursor maybeLexIRBlock(Cursor C, MIToken &Token) {
  return maybeLexIndex(C, Token, "%ir-block.", MIToken::IRBlock);
}

static Cursor lexVirtualRegister(Cursor C, MIToken &Token) {
  auto Range = C;
  C.advance(); // Skip '%'
  auto NumberRange = C;
  while (isdigit(C.peek()))
    C.advance();
  Token = MIToken(MIToken::VirtualRegister, Range.upto(C),
                  APSInt(NumberRange.upto(C)));
  return C;
}

static Cursor maybeLexRegister(Cursor C, MIToken &Token) {
  if (C.peek() != '%')
    return None;
  if (isdigit(C.peek(1)))
    return lexVirtualRegister(C, Token);
  auto Range = C;
  C.advance(); // Skip '%'
  while (isIdentifierChar(C.peek()))
    C.advance();
  Token = MIToken(MIToken::NamedRegister, Range.upto(C),
                  /*StringOffset=*/1); // Drop the '%'
  return C;
}

static Cursor lexName(
    Cursor C, MIToken &Token, MIToken::TokenKind Type,
    MIToken::TokenKind QuotedType, unsigned PrefixLength,
    function_ref<void(StringRef::iterator Loc, const Twine &)> ErrorCallback) {
  auto Range = C;
  C.advance(PrefixLength);
  if (C.peek() == '"') {
    if (Cursor R = lexStringConstant(C, ErrorCallback)) {
      Token = MIToken(QuotedType, Range.upto(R), PrefixLength);
      return R;
    }
    Token = MIToken(MIToken::Error, Range.remaining());
    return Range;
  }
  while (isIdentifierChar(C.peek()))
    C.advance();
  Token = MIToken(Type, Range.upto(C), PrefixLength);
  return C;
}

static Cursor maybeLexGlobalValue(
    Cursor C, MIToken &Token,
    function_ref<void(StringRef::iterator Loc, const Twine &)> ErrorCallback) {
  if (C.peek() != '@')
    return None;
  if (!isdigit(C.peek(1)))
    return lexName(C, Token, MIToken::NamedGlobalValue,
                   MIToken::QuotedNamedGlobalValue, /*PrefixLength=*/1,
                   ErrorCallback);
  auto Range = C;
  C.advance(1); // Skip the '@'
  auto NumberRange = C;
  while (isdigit(C.peek()))
    C.advance();
  Token =
      MIToken(MIToken::GlobalValue, Range.upto(C), APSInt(NumberRange.upto(C)));
  return C;
}

static Cursor maybeLexExternalSymbol(
    Cursor C, MIToken &Token,
    function_ref<void(StringRef::iterator Loc, const Twine &)> ErrorCallback) {
  if (C.peek() != '$')
    return None;
  return lexName(C, Token, MIToken::ExternalSymbol,
                 MIToken::QuotedExternalSymbol,
                 /*PrefixLength=*/1, ErrorCallback);
}

static Cursor maybeLexIntegerLiteral(Cursor C, MIToken &Token) {
  if (!isdigit(C.peek()) && (C.peek() != '-' || !isdigit(C.peek(1))))
    return None;
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
  case ':':
    return MIToken::colon;
  case '!':
    return MIToken::exclaim;
  default:
    return MIToken::Error;
  }
}

static Cursor maybeLexSymbol(Cursor C, MIToken &Token) {
  auto Kind = symbolToken(C.peek());
  if (Kind == MIToken::Error)
    return None;
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

  if (Cursor R = maybeLexIdentifier(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexMachineBasicBlock(C, Token, ErrorCallback))
    return R.remaining();
  if (Cursor R = maybeLexJumpTableIndex(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexStackObject(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexFixedStackObject(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexConstantPoolItem(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexIRBlock(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexRegister(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexGlobalValue(C, Token, ErrorCallback))
    return R.remaining();
  if (Cursor R = maybeLexExternalSymbol(C, Token, ErrorCallback))
    return R.remaining();
  if (Cursor R = maybeLexIntegerLiteral(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexSymbol(C, Token))
    return R.remaining();

  Token = MIToken(MIToken::Error, C.remaining());
  ErrorCallback(C.location(),
                Twine("unexpected character '") + Twine(C.peek()) + "'");
  return C.remaining();
}
