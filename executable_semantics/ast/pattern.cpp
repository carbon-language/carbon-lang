// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/pattern.h"

#include <string>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/common/arena.h"
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
        out << sep << field.name << " = " << *field.pattern;
      }
      out << ")";
      break;
    }
    case Kind::AlternativePattern: {
      const auto& alternative = cast<AlternativePattern>(*this);
      out << *alternative.ChoiceType() << "." << alternative.AlternativeName()
          << *alternative.Arguments();
      break;
    }
    case Kind::ExpressionPattern:
      out << *cast<ExpressionPattern>(*this).Expression();
      break;
  }
}

TuplePattern::TuplePattern(Nonnull<Arena*> arena,
                           Nonnull<const Expression*> tuple_literal)
    : Pattern(Kind::TuplePattern, tuple_literal->SourceLoc()) {
  const auto& tuple = cast<TupleLiteral>(*tuple_literal);
  for (const FieldInitializer& init : tuple.Fields()) {
    fields.push_back(
        Field(init.name, arena->New<ExpressionPattern>(init.expression)));
  }
}

auto PatternFromParenContents(Nonnull<Arena*> arena, SourceLocation loc,
                              const ParenContents<Pattern>& paren_contents)
    -> Nonnull<const Pattern*> {
  std::optional<Nonnull<const Pattern*>> single_term =
      paren_contents.SingleTerm();
  if (single_term.has_value()) {
    return *single_term;
  } else {
    return TuplePatternFromParenContents(arena, loc, paren_contents);
  }
}

auto TuplePatternFromParenContents(Nonnull<Arena*> arena, SourceLocation loc,
                                   const ParenContents<Pattern>& paren_contents)
    -> Nonnull<const TuplePattern*> {
  return arena->New<TuplePattern>(
      loc, paren_contents.TupleElements<TuplePattern::Field>(loc));
}

// Used by AlternativePattern for constructor initialization. Produces a helpful
// error for incorrect expressions, rather than letting a default cast error
// apply.
static const FieldAccessExpression& RequireFieldAccess(
    Nonnull<const Expression*> alternative) {
  if (alternative->Tag() != Expression::Kind::FieldAccessExpression) {
    FATAL_PROGRAM_ERROR(alternative->SourceLoc())
        << "Alternative pattern must have the form of a field access.";
  }
  return cast<FieldAccessExpression>(*alternative);
}

AlternativePattern::AlternativePattern(SourceLocation loc,
                                       Nonnull<const Expression*> alternative,
                                       Nonnull<const TuplePattern*> arguments)
    : Pattern(Kind::AlternativePattern, loc),
      choice_type(RequireFieldAccess(alternative).Aggregate()),
      alternative_name(RequireFieldAccess(alternative).Field()),
      arguments(arguments) {}

auto ParenExpressionToParenPattern(Nonnull<Arena*> arena,
                                   const ParenContents<Expression>& contents)
    -> ParenContents<Pattern> {
  ParenContents<Pattern> result = {
      .elements = {}, .has_trailing_comma = contents.has_trailing_comma};
  for (const auto& element : contents.elements) {
    result.elements.push_back(
        {.name = element.name,
         .term = arena->New<ExpressionPattern>(element.term)});
  }
  return result;
}

}  // namespace Carbon
