//===- Token.h - MLIR Token Interface ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_PARSER_TOKEN_H
#define MLIR_LIB_PARSER_TOKEN_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

namespace mlir {

/// This represents a token in the MLIR syntax.
class Token {
public:
  enum Kind {
#define TOK_MARKER(NAME) NAME,
#define TOK_IDENTIFIER(NAME) NAME,
#define TOK_LITERAL(NAME) NAME,
#define TOK_PUNCTUATION(NAME, SPELLING) NAME,
#define TOK_KEYWORD(SPELLING) kw_##SPELLING,
#include "TokenKinds.def"
  };

  Token(Kind kind, StringRef spelling) : kind(kind), spelling(spelling) {}

  // Return the bytes that make up this token.
  StringRef getSpelling() const { return spelling; }

  // Token classification.
  Kind getKind() const { return kind; }
  bool is(Kind k) const { return kind == k; }

  bool isAny(Kind k1, Kind k2) const { return is(k1) || is(k2); }

  /// Return true if this token is one of the specified kinds.
  template <typename... T>
  bool isAny(Kind k1, Kind k2, Kind k3, T... others) const {
    if (is(k1))
      return true;
    return isAny(k2, k3, others...);
  }

  bool isNot(Kind k) const { return kind != k; }

  /// Return true if this token isn't one of the specified kinds.
  template <typename... T>
  bool isNot(Kind k1, Kind k2, T... others) const {
    return !isAny(k1, k2, others...);
  }

  /// Return true if this is one of the keyword token kinds (e.g. kw_if).
  bool isKeyword() const;

  // Helpers to decode specific sorts of tokens.

  /// For an integer token, return its value as an unsigned.  If it doesn't fit,
  /// return None.
  Optional<unsigned> getUnsignedIntegerValue() const;

  /// For an integer token, return its value as an uint64_t.  If it doesn't fit,
  /// return None.
  static Optional<uint64_t> getUInt64IntegerValue(StringRef spelling);
  Optional<uint64_t> getUInt64IntegerValue() const {
    return getUInt64IntegerValue(getSpelling());
  }

  /// For a floatliteral token, return its value as a double. Returns None in
  /// the case of underflow or overflow.
  Optional<double> getFloatingPointValue() const;

  /// For an inttype token, return its bitwidth.
  Optional<unsigned> getIntTypeBitwidth() const;

  /// For an inttype token, return its signedness semantics: llvm::None means no
  /// signedness semantics; true means signed integer type; false means unsigned
  /// integer type.
  Optional<bool> getIntTypeSignedness() const;

  /// Given a hash_identifier token like #123, try to parse the number out of
  /// the identifier, returning None if it is a named identifier like #x or
  /// if the integer doesn't fit.
  Optional<unsigned> getHashIdentifierNumber() const;

  /// Given a token containing a string literal, return its value, including
  /// removing the quote characters and unescaping the contents of the string.
  std::string getStringValue() const;

  /// Given a token containing a hex string literal, return its value or None if
  /// the token does not contain a valid hex string. A hex string literal is a
  /// string starting with `0x` and only containing hex digits.
  Optional<std::string> getHexStringValue() const;

  /// Given a token containing a symbol reference, return the unescaped string
  /// value.
  std::string getSymbolReference() const;

  // Location processing.
  llvm::SMLoc getLoc() const;
  llvm::SMLoc getEndLoc() const;
  llvm::SMRange getLocRange() const;

  /// Given a punctuation or keyword token kind, return the spelling of the
  /// token as a string.  Warning: This will abort on markers, identifiers and
  /// literal tokens since they have no fixed spelling.
  static StringRef getTokenSpelling(Kind kind);

private:
  /// Discriminator that indicates the sort of token this is.
  Kind kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef spelling;
};

} // namespace mlir

#endif // MLIR_LIB_PARSER_TOKEN_H
