// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_STRING_LITERAL_H_
#define CARBON_TOOLCHAIN_LEX_STRING_LITERAL_H_

#include <optional>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/token_info.h"

namespace Carbon::Lex {

class StringLiteral {
 public:
  // A string literal's kind.
  enum class Kind : int8_t {
    // A character literal is still handled through string literal lexing.
    Char,

    // A single-line string, `"<content>"`.
    SingleLine,

    // A multi-line string, `'''<content>'''`.
    MultiLine,

    // An incorrectly double-quoted multi-line string, `"""<content>"""`.
    MultiLineWithDoubleQuotes,
  };

  // Extract a string literal token from the given text, if it has a suitable
  // form. Returning std::nullopt indicates no string literal was found;
  // returning an invalid literal indicates a string prefix was found, but it's
  // malformed and is returning a partial string literal to assist error
  // construction.
  static auto Lex(llvm::StringRef source_text) -> std::optional<StringLiteral>;

  // Expand any escape sequences and compute the resulting character. This
  // handles error recovery internally, but can return nullopt for an invalid
  // character.
  auto ComputeCharLiteralValue(Diagnostics::Emitter<const char*>& emitter) const
      -> std::optional<CharLiteralValue>;

  // Expand any escape sequences in the given string literal and compute the
  // resulting value. This handles error recovery internally and cannot fail.
  //
  // When content_needs_validation_ is false and the string has no indent to
  // deal with, this can return the content directly. Otherwise, the allocator
  // will be used for the StringRef.
  auto ComputeStringValue(llvm::BumpPtrAllocator& allocator,
                          Diagnostics::Emitter<const char*>& emitter) const
      -> llvm::StringRef;

  // Get the text corresponding to this literal.
  auto text() const -> llvm::StringRef { return text_; }

  // Determine whether this is a multi-line string literal.
  auto kind() const -> Kind { return kind_; }

  // Returns true if the string has a valid terminator.
  auto is_terminated() const -> bool { return is_terminated_; }

 private:
  struct Introducer;

  explicit StringLiteral(llvm::StringRef text, llvm::StringRef content,
                         bool content_needs_validation, int hash_level,
                         Kind kind, bool is_terminated)
      : text_(text),
        content_(content),
        content_needs_validation_(content_needs_validation),
        hash_level_(hash_level),
        kind_(kind),
        is_terminated_(is_terminated) {}

  // The complete text of the string literal.
  llvm::StringRef text_;

  // The content of the literal. For a multi-line literal, this begins
  // immediately after the newline following the file type indicator, and ends
  // at the start of the closing `"""`. Leading whitespace is not removed from
  // either end.
  llvm::StringRef content_;

  // Whether content needs validation, in particular due to either an escape
  // (which needs modifications) or a tab character (which may cause a warning).
  bool content_needs_validation_;

  // The number of `#`s preceding the opening `"` or `"""`.
  int hash_level_;

  // Whether this was a single-line string literal, multi-line string literal,
  // or a char literal.
  Kind kind_;

  // Whether the literal is valid, or should only be used for errors.
  bool is_terminated_;
};

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_STRING_LITERAL_H_
