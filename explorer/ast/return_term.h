// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_RETURN_TERM_H_
#define CARBON_EXPLORER_AST_RETURN_TERM_H_

#include <optional>
#include <utility>

#include "common/check.h"
#include "common/ostream.h"
#include "explorer/ast/clone_context.h"
#include "explorer/ast/expression.h"
#include "explorer/base/nonnull.h"
#include "explorer/base/source_location.h"

namespace Carbon {

class Value;

// The syntactic representation of a function declaration's return type.
// This syntax can take one of three forms:
// - An _explicit_ term consists of `->` followed by a type expression.
// - An _auto_ term consists of `-> auto`.
// - An _omitted_ term consists of no tokens at all.
// Each of these forms has a corresponding factory function.
class ReturnTerm : public Printable<ReturnTerm> {
 public:
  explicit ReturnTerm(CloneContext& context, const ReturnTerm& other)
      : kind_(other.kind_),
        type_expression_(context.Clone(other.type_expression_)),
        static_type_(context.Clone(other.static_type_)),
        source_loc_(other.source_loc_) {}

  ReturnTerm(const ReturnTerm&) = default;
  auto operator=(const ReturnTerm&) -> ReturnTerm& = default;

  // Represents an omitted return term at `source_loc`.
  static auto Omitted(SourceLocation source_loc) -> ReturnTerm {
    return ReturnTerm(ReturnKind::Omitted, source_loc);
  }

  // Represents an auto return term at `source_loc`.
  static auto Auto(SourceLocation source_loc) -> ReturnTerm {
    return ReturnTerm(ReturnKind::Auto, source_loc);
  }

  // Represents an explicit return term with the given type expression.
  static auto Explicit(Nonnull<Expression*> type_expression) -> ReturnTerm {
    return ReturnTerm(type_expression);
  }

  // Returns true if this represents an omitted return term.
  auto is_omitted() const -> bool { return kind_ == ReturnKind::Omitted; }

  // Returns true if this represents an auto return term.
  auto is_auto() const -> bool { return kind_ == ReturnKind::Auto; }

  // If this represents an explicit return term, returns the type expression.
  // Otherwise, returns nullopt.
  auto type_expression() const -> std::optional<Nonnull<const Expression*>> {
    return type_expression_;
  }
  auto type_expression() -> std::optional<Nonnull<Expression*>> {
    return type_expression_;
  }

  // The static return type this term resolves to. Cannot be called before
  // typechecking.
  auto static_type() const -> const Value& { return **static_type_; }

  // Sets the value of static_type(). Can only be called once, during
  // typechecking.
  void set_static_type(Nonnull<const Value*> type) {
    CARBON_CHECK(!static_type_.has_value());
    static_type_ = type;
  }

  auto source_loc() const -> SourceLocation { return source_loc_; }

  void Print(llvm::raw_ostream& out) const;

 private:
  enum class ReturnKind { Omitted, Auto, Expression };

  explicit ReturnTerm(ReturnKind kind, SourceLocation source_loc)
      : kind_(kind), source_loc_(source_loc) {
    CARBON_CHECK(kind != ReturnKind::Expression);
  }

  explicit ReturnTerm(Nonnull<Expression*> type_expression)
      : kind_(ReturnKind::Expression),
        type_expression_(type_expression),
        source_loc_(type_expression->source_loc()) {}

  ReturnKind kind_;
  std::optional<Nonnull<Expression*>> type_expression_;
  std::optional<Nonnull<const Value*>> static_type_;

  SourceLocation source_loc_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_RETURN_TERM_H_
