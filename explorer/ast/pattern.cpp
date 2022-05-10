// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/pattern.h"

#include <string>

#include "common/ostream.h"
#include "explorer/ast/expression.h"
#include "explorer/common/arena.h"
#include "explorer/common/error.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

Pattern::~Pattern() = default;

void Pattern::Print(llvm::raw_ostream& out) const {
  switch (kind()) {
    case PatternKind::AutoPattern:
      out << "auto";
      break;
    case PatternKind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(*this);
      out << binding.name() << ": " << binding.type();
      break;
    }
    case PatternKind::GenericBinding: {
      const auto& binding = cast<GenericBinding>(*this);
      out << binding.name() << ":! " << binding.type();
      break;
    }
    case PatternKind::TuplePattern: {
      const auto& tuple = cast<TuplePattern>(*this);
      out << "(";
      llvm::ListSeparator sep;
      for (Nonnull<const Pattern*> field : tuple.fields()) {
        out << sep << *field;
      }
      out << ")";
      break;
    }
    case PatternKind::AlternativePattern: {
      const auto& alternative = cast<AlternativePattern>(*this);
      out << alternative.choice_type() << "." << alternative.alternative_name()
          << alternative.arguments();
      break;
    }
    case PatternKind::ExpressionPattern:
      out << cast<ExpressionPattern>(*this).expression();
      break;
    case PatternKind::VarPattern:
      out << "var " << cast<VarPattern>(*this).pattern();
      break;
    case PatternKind::AddrBindingPattern:
      out << "addr " << cast<AddrBindingPattern>(*this).binding();
      break;
  }
}

void Pattern::PrintID(llvm::raw_ostream& out) const {
  switch (kind()) {
    case PatternKind::AutoPattern:
      out << "auto";
      break;
    case PatternKind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(*this);
      out << binding.name();
      break;
    }
    case PatternKind::GenericBinding: {
      const auto& binding = cast<GenericBinding>(*this);
      out << binding.name();
      break;
    }
    case PatternKind::TuplePattern: {
      out << "(...)";
      break;
    }
    case PatternKind::AlternativePattern: {
      const auto& alternative = cast<AlternativePattern>(*this);
      out << alternative.choice_type() << "." << alternative.alternative_name()
          << "(...)";
      break;
    }
    case PatternKind::VarPattern:
      out << "var ...";
      break;
    case PatternKind::AddrBindingPattern:
      out << "addr ...";
      break;
    case PatternKind::ExpressionPattern:
      out << "...";
      break;
  }
}

// Equivalent to `GetBindings`, but stores its output in `bindings` instead of
// returning it.
static void GetBindingsImpl(
    const Pattern& pattern,
    std::vector<Nonnull<const BindingPattern*>>& bindings) {
  switch (pattern.kind()) {
    case PatternKind::BindingPattern:
      bindings.push_back(&cast<BindingPattern>(pattern));
      return;
    case PatternKind::TuplePattern:
      for (const Pattern* field : cast<TuplePattern>(pattern).fields()) {
        GetBindingsImpl(*field, bindings);
      }
      return;
    case PatternKind::AlternativePattern:
      GetBindingsImpl(cast<AlternativePattern>(pattern).arguments(), bindings);
      return;
    case PatternKind::AutoPattern:
    case PatternKind::ExpressionPattern:
    case PatternKind::GenericBinding:
      return;
    case PatternKind::VarPattern:
      GetBindingsImpl(cast<VarPattern>(pattern).pattern(), bindings);
      return;
    case PatternKind::AddrBindingPattern:
      GetBindingsImpl(cast<AddrBindingPattern>(pattern).binding(), bindings);
      return;
  }
}

auto GetBindings(const Pattern& pattern)
    -> std::vector<Nonnull<const BindingPattern*>> {
  std::vector<Nonnull<const BindingPattern*>> result;
  GetBindingsImpl(pattern, result);
  return result;
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
auto AlternativePattern::RequireFieldAccess(Nonnull<Expression*> alternative)
    -> ErrorOr<Nonnull<FieldAccessExpression*>> {
  if (alternative->kind() != ExpressionKind::FieldAccessExpression) {
    return FATAL_PROGRAM_ERROR(alternative->source_loc())
           << "Alternative pattern must have the form of a field access.";
  }
  return &cast<FieldAccessExpression>(*alternative);
}

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
