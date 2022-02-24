// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon {

class LexedStringLiteral {
 public:
  // Extract a string literal token from the given text, if it has a suitable
  // form.
  static auto Lex(llvm::StringRef source_text)
      -> llvm::Optional<LexedStringLiteral>;

  // Get the text corresponding to this literal.
  [[nodiscard]] auto Text() const -> llvm::StringRef { return text_; }

  // Determine whether this is a multi-line string literal.
  [[nodiscard]] auto IsMultiLine() const -> bool { return multi_line_; }

  // Expand any escape sequences in the given string literal and compute the
  // resulting value. This handles error recovery internally and cannot fail.
  auto ComputeValue(DiagnosticEmitter<const char*>& emitter) const
      -> std::string;

 private:
  LexedStringLiteral(llvm::StringRef text, llvm::StringRef content,
                     int hash_level, bool multi_line)
      : text_(text),
        content_(content),
        hash_level_(hash_level),
        multi_line_(multi_line) {}

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
  bool multi_line_;
};

}  // namespace Carbon
