// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/pattern.h"

#include <string>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/unimplemented.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/error.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

void Pattern::Print(llvm::raw_ostream& out) const {
  switch (kind()) {
    case Kind::AutoPattern:
      out << "auto";
      break;
    case Kind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(*this);
      if (binding.name().has_value()) {
        out << *binding.name();
      } else {
        out << "_";
      }
      out << ": " << binding.type();
      break;
    }
    case Kind::TuplePattern: {
      const auto& tuple = cast<TuplePattern>(*this);
      out << "(";
      llvm::ListSeparator sep;
      for (Nonnull<const Pattern*> field : tuple.fields()) {
        out << sep << *field;
      }
      out << ")";
      break;
    }
    case Kind::AlternativePattern: {
      const auto& alternative = cast<AlternativePattern>(*this);
      out << alternative.choice_type() << "." << alternative.alternative_name()
          << alternative.arguments();
      break;
    }
    case Kind::ExpressionPattern:
      out << cast<ExpressionPattern>(*this).expression();
      break;
    case Kind::Unimplemented:
      out << cast<Unimplemented<Pattern>>(*this).printed_form();
  }
}

auto PatternFromParenContents(Nonnull<Arena*> arena, SourceLocation source_loc,
                              const ParenContents<Pattern>& paren_contents)
    -> Nonnull<Pattern*> {
  std::optional<Nonnull<Pattern*>> single_term = paren_contents.SingleTerm();
  if (single_term.has_value()) {
    return *single_term;
  } else {
    return TuplePatternFromParenContents(arena, source_loc, paren_contents);
  }
}

auto TuplePatternFromParenContents(Nonnull<Arena*> arena,
                                   SourceLocation source_loc,
                                   const ParenContents<Pattern>& paren_contents)
    -> Nonnull<TuplePattern*> {
  return arena->New<TuplePattern>(source_loc, paren_contents.elements);
}

// Used by AlternativePattern for constructor initialization. Produces a helpful
// error for incorrect expressions, rather than letting a default cast error
// apply.
static auto RequireFieldAccess(Nonnull<Expression*> alternative)
    -> FieldAccessExpression& {
  if (alternative->kind() != Expression::Kind::FieldAccessExpression) {
    FATAL_PROGRAM_ERROR(alternative->source_loc())
        << "Alternative pattern must have the form of a field access.";
  }
  return cast<FieldAccessExpression>(*alternative);
}

AlternativePattern::AlternativePattern(SourceLocation source_loc,
                                       Nonnull<Expression*> alternative,
                                       Nonnull<TuplePattern*> arguments)
    : Pattern(Kind::AlternativePattern, source_loc),
      choice_type_(&RequireFieldAccess(alternative).aggregate()),
      alternative_name_(RequireFieldAccess(alternative).field()),
      arguments_(arguments) {}

auto ParenExpressionToParenPattern(Nonnull<Arena*> arena,
                                   const ParenContents<Expression>& contents)
    -> ParenContents<Pattern> {
  ParenContents<Pattern> result = {
      .elements = {}, .has_trailing_comma = contents.has_trailing_comma};
  for (const auto& element : contents.elements) {
    result.elements.push_back(arena->New<ExpressionPattern>(element));
  }
  return result;
}

}  // namespace Carbon
