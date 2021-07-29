// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/pattern.h"

#include <string>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/common/error.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

void Pattern::Print(llvm::raw_ostream& out) const {
  switch (Tag()) {
    case Kind::AutoPattern:
      out << "auto";
      break;
    case Kind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(*this);
      if (binding.Name().has_value()) {
        out << *binding.Name();
      } else {
        out << "_";
      }
      out << ": " << *binding.Type();
      break;
    }
    case Kind::TuplePattern: {
      const auto& tuple = cast<TuplePattern>(*this);
      out << "(";
      llvm::ListSeparator sep;
      for (const TuplePattern::Field& field : tuple.Fields()) {
        out << sep << field.name << " = " << field.pattern;
      }
      out << ")";
      break;
    }
    case Kind::AlternativePattern: {
      const auto& alternative = cast<AlternativePattern>(*this);
      out << alternative.ChoiceType() << "." << alternative.AlternativeName()
          << alternative.Arguments();
      break;
    }
    case Kind::ExpressionPattern:
      out << cast<ExpressionPattern>(*this).Expression();
      break;
  }
}

TuplePattern::TuplePattern(const Expression* tuple_literal)
    : Pattern(Kind::TuplePattern, tuple_literal->line_num) {
  const auto& tuple = tuple_literal->GetTupleLiteral();
  for (const FieldInitializer& init : tuple.fields) {
    fields.push_back(Field(init.name, new ExpressionPattern(init.expression)));
  }
}

auto AsPattern(int line_num, const ParenContents<Pattern>& paren_contents)
    -> const Pattern* {
  std::optional<const Pattern*> single_term = paren_contents.SingleTerm();
  if (single_term.has_value()) {
    return *single_term;
  } else {
    return AsTuplePattern(line_num, paren_contents);
  }
}

auto AsTuplePattern(int line_num, const ParenContents<Pattern>& paren_contents)
    -> const TuplePattern* {
  return new TuplePattern(
      line_num, paren_contents.TupleElements<TuplePattern::Field>(line_num));
}

AlternativePattern::AlternativePattern(int line_num,
                                       const Expression* alternative,
                                       const TuplePattern* arguments)
    : Pattern(Kind::AlternativePattern, line_num), arguments(arguments) {
  if (alternative->tag() != ExpressionKind::FieldAccessExpression) {
    FATAL_USER_ERROR(alternative->line_num)
        << "Alternative pattern must have the form of a field access.\n";
  }
  const auto& field_access = alternative->GetFieldAccessExpression();
  choice_type = field_access.aggregate;
  alternative_name = field_access.field;
}

auto ParenExpressionToParenPattern(const ParenContents<Expression>& contents)
    -> ParenContents<Pattern> {
  ParenContents<Pattern> result = {
      .elements = {}, .has_trailing_comma = contents.has_trailing_comma};
  for (const auto& element : contents.elements) {
    result.elements.push_back(
        {.name = element.name, .term = new ExpressionPattern(element.term)});
  }
  return result;
}

}  // namespace Carbon
