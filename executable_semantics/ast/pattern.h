// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_PATTERN_H_
#define EXECUTABLE_SEMANTICS_AST_PATTERN_H_

#include <optional>
#include <string>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/ast_node.h"
#include "executable_semantics/ast/ast_rtti.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/source_location.h"
#include "executable_semantics/ast/static_scope.h"
#include "llvm/ADT/ArrayRef.h"

namespace Carbon {

class Value;

// Abstract base class of all AST nodes representing patterns.
//
// Pattern and its derived classes support LLVM-style RTTI, including
// llvm::isa, llvm::cast, and llvm::dyn_cast. To support this, every
// class derived from Pattern must provide a `classof` operation, and
// every concrete derived class must have a corresponding enumerator
// in `Kind`; see https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html for
// details.
class Pattern : public AstNode {
 public:
  Pattern(const Pattern&) = delete;
  auto operator=(const Pattern&) -> Pattern& = delete;

  ~Pattern() override = 0;

  void Print(llvm::raw_ostream& out) const override;

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromPattern(node->kind());
  }

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> PatternKind {
    return static_cast<PatternKind>(root_kind());
  }

  // The static type of this pattern. Cannot be called before typechecking.
  auto static_type() const -> const Value& { return **static_type_; }

  // Sets the static type of this expression. Can only be called once, during
  // typechecking.
  void set_static_type(Nonnull<const Value*> type) { static_type_ = type; }

  // Returns whether the static type has been set. Should only be called
  // during typechecking: before typechecking it's guaranteed to be false,
  // and after typechecking it's guaranteed to be true.
  auto has_static_type() const -> bool { return static_type_.has_value(); }

  // The value of this pattern. Cannot be called before typechecking.
  auto value() const -> const Value& { return **value_; }

  // Sets the value of this pattern. Can only be called once, during
  // typechecking.
  void set_value(Nonnull<const Value*> value) { value_ = value; }

  // Returns whether the value has been set. Should only be called
  // during typechecking: before typechecking it's guaranteed to be false,
  // and after typechecking it's guaranteed to be true.
  auto has_value() const -> bool { return value_.has_value(); }

 protected:
  // Constructs a Pattern representing syntax at the given line number.
  // `kind` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Pattern(AstNodeKind kind, SourceLocation source_loc)
      : AstNode(kind, source_loc) {}

 private:
  std::optional<Nonnull<const Value*>> static_type_;
  std::optional<Nonnull<const Value*>> value_;
};

// A pattern consisting of the `auto` keyword.
class AutoPattern : public Pattern {
 public:
  explicit AutoPattern(SourceLocation source_loc)
      : Pattern(AstNodeKind::AutoPattern, source_loc) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromAutoPattern(node->kind());
  }
};

// A pattern that matches a value of a specified type, and optionally binds
// a name to it.
class BindingPattern : public Pattern {
 public:
  using ImplementsCarbonNamedEntity = void;

  BindingPattern(SourceLocation source_loc, std::optional<std::string> name,
                 Nonnull<Pattern*> type)
      : Pattern(AstNodeKind::BindingPattern, source_loc),
        name_(std::move(name)),
        type_(type) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromBindingPattern(node->kind());
  }

  // The name this pattern binds, if any.
  auto name() const -> const std::optional<std::string>& { return name_; }

  // The pattern specifying the type of values that this pattern matches.
  auto type() const -> const Pattern& { return *type_; }
  auto type() -> Pattern& { return *type_; }

 private:
  std::optional<std::string> name_;
  Nonnull<Pattern*> type_;
};

// A pattern that matches a tuple value field-wise.
class TuplePattern : public Pattern {
 public:
  TuplePattern(SourceLocation source_loc, std::vector<Nonnull<Pattern*>> fields)
      : Pattern(AstNodeKind::TuplePattern, source_loc),
        fields_(std::move(fields)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromTuplePattern(node->kind());
  }

  auto fields() const -> llvm::ArrayRef<Nonnull<const Pattern*>> {
    return fields_;
  }
  auto fields() -> llvm::ArrayRef<Nonnull<Pattern*>> { return fields_; }

 private:
  std::vector<Nonnull<Pattern*>> fields_;
};

// Converts paren_contents to a Pattern, interpreting the parentheses as
// grouping if their contents permit that interpretation, or as forming a
// tuple otherwise.
auto PatternFromParenContents(Nonnull<Arena*> arena, SourceLocation source_loc,
                              const ParenContents<Pattern>& paren_contents)
    -> Nonnull<Pattern*>;

// Converts paren_contents to a TuplePattern, interpreting the parentheses as
// forming a tuple.
auto TuplePatternFromParenContents(Nonnull<Arena*> arena,
                                   SourceLocation source_loc,
                                   const ParenContents<Pattern>& paren_contents)
    -> Nonnull<TuplePattern*>;

// Converts `contents` to ParenContents<Pattern> by replacing each Expression
// with an ExpressionPattern.
auto ParenExpressionToParenPattern(Nonnull<Arena*> arena,
                                   const ParenContents<Expression>& contents)
    -> ParenContents<Pattern>;

// A pattern that matches an alternative of a choice type.
class AlternativePattern : public Pattern {
 public:
  // Constructs an AlternativePattern that matches a value of the type
  // specified by choice_type if it represents an alternative named
  // alternative_name, and its arguments match `arguments`.
  AlternativePattern(SourceLocation source_loc,
                     Nonnull<Expression*> choice_type,
                     std::string alternative_name,
                     Nonnull<TuplePattern*> arguments)
      : Pattern(AstNodeKind::AlternativePattern, source_loc),
        choice_type_(choice_type),
        alternative_name_(std::move(alternative_name)),
        arguments_(arguments) {}

  // Constructs an AlternativePattern that matches the alternative specified
  // by `alternative`, if its arguments match `arguments`.
  AlternativePattern(SourceLocation source_loc,
                     Nonnull<Expression*> alternative,
                     Nonnull<TuplePattern*> arguments);

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromAlternativePattern(node->kind());
  }

  auto choice_type() const -> const Expression& { return *choice_type_; }
  auto choice_type() -> Expression& { return *choice_type_; }
  auto alternative_name() const -> const std::string& {
    return alternative_name_;
  }
  auto arguments() const -> const TuplePattern& { return *arguments_; }
  auto arguments() -> TuplePattern& { return *arguments_; }

 private:
  Nonnull<Expression*> choice_type_;
  std::string alternative_name_;
  Nonnull<TuplePattern*> arguments_;
};

// A pattern that matches a value if it is equal to the value of a given
// expression.
class ExpressionPattern : public Pattern {
 public:
  explicit ExpressionPattern(Nonnull<Expression*> expression)
      : Pattern(AstNodeKind::ExpressionPattern, expression->source_loc()),
        expression_(expression) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromExpressionPattern(node->kind());
  }

  auto expression() const -> const Expression& { return *expression_; }
  auto expression() -> Expression& { return *expression_; }

 private:
  Nonnull<Expression*> expression_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_PATTERN_H_
