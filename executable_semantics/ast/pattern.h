// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_PATTERN_H_
#define EXECUTABLE_SEMANTICS_AST_PATTERN_H_

#include <optional>
#include <string>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/source_location.h"
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
class Pattern {
 public:
  enum class Kind {
    AutoPattern,
    BindingPattern,
    TuplePattern,
    AlternativePattern,
    ExpressionPattern,
  };

  Pattern(const Pattern&) = delete;
  Pattern& operator=(const Pattern&) = delete;

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> Kind { return kind_; }

  auto source_loc() const -> SourceLocation { return source_loc_; }

  // The static type of this pattern. Cannot be called before typechecking.
  auto static_type() const -> Nonnull<const Value*> {
    CHECK(static_type_.has_value());  // FIXME drop this
    return *static_type_;
  }

 protected:
  // Constructs a Pattern representing syntax at the given line number.
  // `kind` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Pattern(Kind kind, SourceLocation source_loc)
      : kind_(kind), source_loc_(source_loc) {}

 private:
  // Defined in type_checker.cpp to avoid circular dependencies.
  friend void set_static_type(Nonnull<Pattern*> pattern,
                              Nonnull<const Value*> type);

  const Kind kind_;
  SourceLocation source_loc_;

  std::optional<Nonnull<const Value*>> static_type_;
};

// A pattern consisting of the `auto` keyword.
class AutoPattern : public Pattern {
 public:
  explicit AutoPattern(SourceLocation source_loc)
      : Pattern(Kind::AutoPattern, source_loc) {}

  static auto classof(const Pattern* pattern) -> bool {
    return pattern->kind() == Kind::AutoPattern;
  }
};

// A pattern that matches a value of a specified type, and optionally binds
// a name to it.
class BindingPattern : public Pattern {
 public:
  BindingPattern(SourceLocation source_loc, std::optional<std::string> name,
                 Nonnull<Pattern*> type)
      : Pattern(Kind::BindingPattern, source_loc),
        name(std::move(name)),
        type(type) {}

  static auto classof(const Pattern* pattern) -> bool {
    return pattern->kind() == Kind::BindingPattern;
  }

  // The name this pattern binds, if any.
  auto Name() const -> const std::optional<std::string>& { return name; }

  // The pattern specifying the type of values that this pattern matches.
  auto Type() const -> Nonnull<const Pattern*> { return type; }
  auto Type() -> Nonnull<Pattern*> { return type; }

 private:
  std::optional<std::string> name;
  Nonnull<Pattern*> type;
};

// A pattern that matches a tuple value field-wise.
class TuplePattern : public Pattern {
 public:
  // Represents a portion of a tuple pattern corresponding to a single field.
  struct Field {
    Field(std::string name, Nonnull<Pattern*> pattern)
        : name(std::move(name)), pattern(pattern) {}

    // The field name. Cannot be empty
    std::string name;

    // The pattern the field must match.
    Nonnull<Pattern*> pattern;
  };

  TuplePattern(SourceLocation source_loc, std::vector<Field> fields)
      : Pattern(Kind::TuplePattern, source_loc), fields(std::move(fields)) {}

  static auto classof(const Pattern* pattern) -> bool {
    return pattern->kind() == Kind::TuplePattern;
  }

  auto Fields() const -> llvm::ArrayRef<Field> { return fields; }
  auto Fields() -> llvm::MutableArrayRef<Field> { return fields; }

 private:
  std::vector<Field> fields;
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
      : Pattern(Kind::AlternativePattern, source_loc),
        choice_type(choice_type),
        alternative_name(std::move(alternative_name)),
        arguments(arguments) {}

  // Constructs an AlternativePattern that matches the alternative specified
  // by `alternative`, if its arguments match `arguments`.
  AlternativePattern(SourceLocation source_loc,
                     Nonnull<Expression*> alternative,
                     Nonnull<TuplePattern*> arguments);

  static auto classof(const Pattern* pattern) -> bool {
    return pattern->kind() == Kind::AlternativePattern;
  }

  auto ChoiceType() const -> Nonnull<const Expression*> { return choice_type; }
  auto ChoiceType() -> Nonnull<Expression*> { return choice_type; }
  auto AlternativeName() const -> const std::string& {
    return alternative_name;
  }
  auto Arguments() const -> Nonnull<const TuplePattern*> { return arguments; }
  auto Arguments() -> Nonnull<TuplePattern*> { return arguments; }

 private:
  Nonnull<Expression*> choice_type;
  std::string alternative_name;
  Nonnull<TuplePattern*> arguments;
};

// A pattern that matches a value if it is equal to the value of a given
// expression.
class ExpressionPattern : public Pattern {
 public:
  ExpressionPattern(Nonnull<Expression*> expression)
      : Pattern(Kind::ExpressionPattern, expression->source_loc()),
        expression(expression) {}

  static auto classof(const Pattern* pattern) -> bool {
    return pattern->kind() == Kind::ExpressionPattern;
  }

  auto Expression() const -> Nonnull<const Expression*> { return expression; }
  auto Expression() -> Nonnull<Carbon::Expression*> { return expression; }

 private:
  Nonnull<Carbon::Expression*> expression;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_PATTERN_H_
