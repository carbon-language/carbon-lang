// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEXER_STRING_LITERAL_H_
#define CARBON_TOOLCHAIN_LEXER_STRING_LITERAL_H_

#include <string>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon {

class LexedStringLiteral {
 public:
  // Extract a string literal token from the given text, if it has a suitable
  // form. Returning llvm::None indicates no string literal was found; returning
  // an invalid literal indicates a string prefix was found, but it's malformed
  // and is returning a partial string literal to assist error construction.
  static auto Lex(llvm::StringRef source_text)
      -> llvm::Optional<LexedStringLiteral>;

  // Expand any escape sequences in the given string literal and compute the
  // resulting value. This handles error recovery internally and cannot fail.
  auto ComputeValue(DiagnosticEmitter<const char*>& emitter) const
      -> std::string;

  // Get the text corresponding to this literal.
  [[nodiscard]] auto text() const -> llvm::StringRef { return text_; }

  // Determine whether this is a multi-line string literal.
  [[nodiscard]] auto is_multi_line() const -> bool { return multi_line_; }

  // Returns true if the string has a valid terminator.
  [[nodiscard]] auto is_terminated() const -> bool { return is_terminated_; }

 private:
  enum MultiLineKind { NotMultiLine, MultiLine, MultiLineWithDoubleQuotes };

  struct Introducer;

  LexedStringLiteral(llvm::StringRef text, llvm::StringRef content,
                     int hash_level, MultiLineKind multi_line,
                     bool is_terminated)
      : text_(text),
        content_(content),
        hash_level_(hash_level),
        multi_line_(multi_line),
        is_terminated_(is_terminated) {}

  // The complete text of the string literal.
  llvm::StringRef text_;
  // The content of the literal. For a multi-line literal, this begins
  // immediately after the newline following the file type indicator, and ends
  // at the start of the closing `"""`. Leading whitespace is not removed from
  // either end.
  llvm::StringRef content_;
  // The number of `#`s preceding the opening `"` or `"""`.
  int hash_level_;
  // Whether this was a multi-line string literal.
  MultiLineKind multi_line_;
  // Whether the literal is valid, or should only be used for errors.
  bool is_terminated_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_LEXER_STRING_LITERAL_H_
