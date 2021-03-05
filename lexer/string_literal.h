// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "diagnostics/diagnostic_emitter.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

class StringLiteralToken {
 public:
  // Get the text corresponding to this literal.
  auto Text() const -> llvm::StringRef { return text; }

  // Determine whether this is a multi-line string literal.
  auto IsMultiLine() const -> bool { return multi_line; }

  // Extract a string literal token from the given text, if it has a suitable
  // form.
  static auto Lex(llvm::StringRef source_text)
      -> llvm::Optional<StringLiteralToken>;

  // The leading whitespace in a multi-line string literal.
  struct Indent {
   public:
    // Get the indentation text: a sequence of horizontal whitespace
    // characters.
    auto Text() const -> llvm::StringRef { return indent; }

   private:
    Indent() : Indent("", false) {}
    Indent(llvm::StringRef indent, bool has_errors)
        : indent(indent), has_errors(has_errors) {}
    friend class StringLiteralToken;
    llvm::StringRef indent;
    bool has_errors;
  };

  // Check the literal is indented properly, if it's a multi-line litera.
  auto CheckIndent(DiagnosticEmitter& emitter) const -> Indent;

  // The result of expanding escape sequences in a string literal.
  struct ExpandedValue {
    std::string result;
    bool has_errors;
  };

  // Expand any escape sequences in the given string literal and compute the
  // resulting value.
  auto ComputeValue(DiagnosticEmitter& emitter, Indent indent) const
      -> ExpandedValue;

 private:
  StringLiteralToken(llvm::StringRef text, llvm::StringRef content,
                     int hash_level, bool multi_line)
      : text(text),
        content(content),
        hash_level(hash_level),
        multi_line(multi_line) {}

  // The complete text of the string literal.
  llvm::StringRef text;
  // The content of the literal. For a multi-line literal, this begins
  // immediately after the newline following the file type indicator, and ends
  // at the start of the closing `"""`. Leading whitespace is not removed from
  // either end.
  llvm::StringRef content;
  // The number of `#`s preceding the opening `"` or `"""`.
  int hash_level;
  // Whether this was a multi-line string literal.
  bool multi_line;
};


}  // namespace Carbon
