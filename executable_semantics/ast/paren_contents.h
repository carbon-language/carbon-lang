// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_PAREN_CONTENTS_H_
#define EXECUTABLE_SEMANTICS_AST_PAREN_CONTENTS_H_

#include <optional>
#include <string>
#include <vector>

#include "executable_semantics/ast/source_location.h"
#include "executable_semantics/common/error.h"

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

  // Converts `elements` to std::vector<TupleElement>. TupleElement must
  // have a constructor that takes a std::string and a Nonnull<Term*>.
  //
  // TODO: Find a way to deduce TupleElement from Term.
  template <typename TupleElement>
  auto TupleElements(SourceLocation source_loc) const
      -> std::vector<TupleElement>;

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

template <typename Term>
template <typename TupleElement>
auto ParenContents<Term>::TupleElements(SourceLocation source_loc) const
    -> std::vector<TupleElement> {
  std::vector<TupleElement> result;
  int i = 0;
  for (auto element : elements) {
    result.push_back(TupleElement(std::to_string(i), element));
    ++i;
  }
  return result;
}

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_PAREN_CONTENTS_H_
