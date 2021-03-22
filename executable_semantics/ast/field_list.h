// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_FIELD_LIST_H_
#define EXECUTABLE_SEMANTICS_AST_FIELD_LIST_H_

#include <list>

#include "executable_semantics/ast/expression.h"

namespace Carbon {

// A FieldInitializer represents the initialization of a single tuple field.
struct FieldInitializer {
  // The field name. An empty string indicates that this represents a
  // positional field.
  std::string name;

  // The expression that initializes the field.
  Expression* expression;
};

// A FieldList represents the syntactic contents of an expression delimited by
// parentheses. Such expressions can be interpreted as either tuples or
// arbitrary expressions, depending on their context and the syntax of their
// contents; this class helps calling code resolve that ambiguity. Since that
// ambiguity is purely syntactic, this class should only be needed during
// parsing.
//
// FIXME rename to ParenExpressionContents?
class FieldList {
 public:
  // Indicates whether the paren expression's contents end with a comma.
  enum class HasTrailingComma { kYes, kNo };

  // Constructs a FieldList representing the given contents, with or without a
  // trailing comma.
  FieldList(std::vector<FieldInitializer>* fields,
            HasTrailingComma has_trailing_comma)
      : fields_(fields), has_trailing_comma_(has_trailing_comma) {}

  // Returns the paren expression, interpreted as a tuple.
  Expression* AsTuple(int line_number) const;

  // Returns the paren expression, with no external constraints on what kind
  // of expression it represents.
  Expression* AsExpression(int line_number) const;

 private:
  std::vector<FieldInitializer>* fields_;
  HasTrailingComma has_trailing_comma_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_FIELD_LIST_H_
