// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_SYNTAX_PAREN_CONTENTS_H_
#define EXECUTABLE_SEMANTICS_SYNTAX_PAREN_CONTENTS_H_

#include <list>

#include "executable_semantics/ast/expression.h"

namespace Carbon {

// Represents the syntactic contents of an expression delimited by
// parentheses. Such expressions can be interpreted as either tuples or
// arbitrary expressions, depending on their context and the syntax of their
// contents; this class helps calling code resolve that ambiguity. Since that
// ambiguity is purely syntactic, this class should only be needed during
// parsing.
class ParenContents {
 public:
  // Indicates whether the paren expression's contents end with a comma.
  enum class HasTrailingComma { Yes, No };

  // Constructs a ParenContents representing the contents of "()".
  ParenContents() : fields_({}), has_trailing_comma_(HasTrailingComma::No) {}

  // Constructs a ParenContents representing the given list of fields,
  // with or without a trailing comma.
  ParenContents(std::vector<FieldInitializer> fields,
                HasTrailingComma has_trailing_comma)
      : fields_(fields), has_trailing_comma_(has_trailing_comma) {}

  ParenContents(const ParenContents&) = default;
  ParenContents& operator=(const ParenContents&) = default;

  // Returns the paren expression, interpreted as a tuple.
  const Expression* AsTuple(int line_number) const;

  // Returns the paren expression, with no external constraints on what kind
  // of expression it represents.
  const Expression* AsExpression(int line_number) const;

 private:
  std::vector<FieldInitializer> fields_;
  HasTrailingComma has_trailing_comma_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_SYNTAX_PAREN_CONTENTS_H_
