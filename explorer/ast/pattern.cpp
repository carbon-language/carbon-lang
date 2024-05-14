// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/pattern.h"

#include <string>

#include "common/ostream.h"
#include "explorer/ast/expression.h"
#include "explorer/ast/impl_binding.h"
#include "explorer/ast/value.h"
#include "explorer/base/arena.h"
#include "explorer/base/error_builders.h"
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
      switch (binding.binding_kind()) {
        case GenericBinding::BindingKind::Checked:
          break;
        case GenericBinding::BindingKind::Template:
          out << "template ";
          break;
      }
      out << binding.name() << ":! " << binding.type();
      if (auto value = binding.constant_value()) {
        out << " = " << **value;
      }
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
      out << "var" << cast<VarPattern>(*this).pattern();
      break;
    case PatternKind::AddrPattern:
      out << "addr" << cast<AddrPattern>(*this).binding();
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
    case PatternKind::AddrPattern:
      out << "addr ...";
      break;
    case PatternKind::ExpressionPattern:
      out << "...";
      break;
  }
}

auto VisitNestedPatterns(const Pattern& pattern,
                         llvm::function_ref<bool(const Pattern&)> visitor)
    -> bool {
  if (!visitor(pattern)) {
    return false;
  }
  switch (pattern.kind()) {
    case PatternKind::TuplePattern:
      for (const Pattern* field : cast<TuplePattern>(pattern).fields()) {
        if (!VisitNestedPatterns(*field, visitor)) {
          return false;
        }
      }
      return true;
    case PatternKind::AlternativePattern:
      return VisitNestedPatterns(cast<AlternativePattern>(pattern).arguments(),
                                 visitor);
    case PatternKind::VarPattern:
      return VisitNestedPatterns(cast<VarPattern>(pattern).pattern(), visitor);
    case PatternKind::AddrPattern:
      return VisitNestedPatterns(cast<AddrPattern>(pattern).binding(), visitor);
    case PatternKind::BindingPattern:
    case PatternKind::AutoPattern:
    case PatternKind::ExpressionPattern:
    case PatternKind::GenericBinding:
      return true;
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
auto AlternativePattern::RequireSimpleMemberAccess(
    Nonnull<Expression*> alternative)
    -> ErrorOr<Nonnull<SimpleMemberAccessExpression*>> {
  if (alternative->kind() != ExpressionKind::SimpleMemberAccessExpression) {
    return ProgramError(alternative->source_loc())
           << "Alternative pattern must have the form of a field access.";
  }
  return &cast<SimpleMemberAccessExpression>(*alternative);
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

GenericBinding::GenericBinding(CloneContext& context,
                               const GenericBinding& other)
    : Pattern(context, other),
      name_(other.name_),
      type_(context.Clone(other.type_)),
      binding_kind_(other.binding_kind_),
      template_value_(context.Clone(other.template_value_)),
      symbolic_identity_(context.Clone(other.symbolic_identity_)),
      impl_binding_(context.Clone(other.impl_binding_)),
      original_(context.Remap(other.original_)),
      named_as_type_via_dot_self_(other.named_as_type_via_dot_self_) {}

}  // namespace Carbon
