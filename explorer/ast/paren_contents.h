// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPLORER_AST_PAREN_CONTENTS_H_
#define EXPLORER_AST_PAREN_CONTENTS_H_

#include <optional>
#include <string>
#include <vector>

#include "explorer/ast/source_location.h"
#include "explorer/common/error.h"

namespace Carbon {

// Represents the syntactic contents of an expression or pattern delimited by
// parentheses. In those syntaxes, parentheses can be used either for grouping
// or for forming a tuple, depending on their context and the syntax of their
// contents; this class helps calling code resolve that ambiguity. Since that
// ambiguity is purely syntactic, this class should only be needed during
// parsing.
//
// `Term` is the type of the syntactic grouping being built, and the type of
// the individual syntactic units it's built from; typically it should be
// either `Expression` or `Pattern`.
template <typename Term>
struct ParenContents {
  // If this object represents a single term with no trailing comma, this
  // method returns that term. This typically means the parentheses can be
  // interpreted as grouping.
  auto SingleTerm() const -> std::optional<Nonnull<Term*>>;

  std::vector<Nonnull<Term*>> elements;
  bool has_trailing_comma;
};

// Implementation details only below here.

template <typename Term>
auto ParenContents<Term>::SingleTerm() const -> std::optional<Nonnull<Term*>> {
  if (elements.size() == 1 && !has_trailing_comma) {
    return elements.front();
  } else {
    return std::nullopt;
  }
}

}  // namespace Carbon

#endif  // EXPLORER_AST_PAREN_CONTENTS_H_
