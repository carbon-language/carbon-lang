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

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto Tag() const -> Kind { return kind; }

  auto SourceLoc() const -> SourceLocation { return loc; }

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

 protected:
  // Constructs a Pattern representing syntax at the given line number.
  // `tag` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Pattern(Kind kind, SourceLocation loc) : kind(kind), loc(loc) {}

 private:
  const Kind kind;
  SourceLocation loc;
};

// A pattern consisting of the `auto` keyword.
class AutoPattern : public Pattern {
 public:
  explicit AutoPattern(SourceLocation loc) : Pattern(Kind::AutoPattern, loc) {}

  static auto classof(const Pattern* pattern) -> bool {
    return pattern->Tag() == Kind::AutoPattern;
  }
};

// A pattern that matches a value of a specified type, and optionally binds
// a name to it.
class BindingPattern : public Pattern {
 public:
  BindingPattern(SourceLocation loc, std::optional<std::string> name,
                 Nonnull<Pattern*> type)
      : Pattern(Kind::BindingPattern, loc), name(std::move(name)), type(type) {}

  static auto classof(const Pattern* pattern) -> bool {
    return pattern->Tag() == Kind::BindingPattern;
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

  TuplePattern(SourceLocation loc, std::vector<Field> fields)
      : Pattern(Kind::TuplePattern, loc), fields(std::move(fields)) {}

  static auto classof(const Pattern* pattern) -> bool {
    return pattern->Tag() == Kind::TuplePattern;
  }

  auto Fields() const -> llvm::ArrayRef<Field> { return fields; }
  auto Fields() -> llvm::MutableArrayRef<Field> { return fields; }

 private:
  std::vector<Field> fields;
};

// Converts paren_contents to a Pattern, interpreting the parentheses as
// grouping if their contents permit that interpretation, or as forming a
// tuple otherwise.
auto PatternFromParenContents(Nonnull<Arena*> arena, SourceLocation loc,
                              const ParenContents<Pattern>& paren_contents)
    -> Nonnull<Pattern*>;

// Converts paren_contents to a TuplePattern, interpreting the parentheses as
// forming a tuple.
auto TuplePatternFromParenContents(Nonnull<Arena*> arena, SourceLocation loc,
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
  AlternativePattern(SourceLocation loc, Nonnull<Expression*> choice_type,
                     std::string alternative_name,
                     Nonnull<TuplePattern*> arguments)
      : Pattern(Kind::AlternativePattern, loc),
        choice_type(choice_type),
        alternative_name(std::move(alternative_name)),
        arguments(arguments) {}

  // Constructs an AlternativePattern that matches the alternative specified
  // by `alternative`, if its arguments match `arguments`.
  AlternativePattern(SourceLocation loc, Nonnull<Expression*> alternative,
                     Nonnull<TuplePattern*> arguments);

  static auto classof(const Pattern* pattern) -> bool {
    return pattern->Tag() == Kind::AlternativePattern;
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
      : Pattern(Kind::ExpressionPattern, expression->SourceLoc()),
        expression(expression) {}

  static auto classof(const Pattern* pattern) -> bool {
    return pattern->Tag() == Kind::ExpressionPattern;
  }

  auto Expression() const -> Nonnull<const Expression*> { return expression; }
  auto Expression() -> Nonnull<Carbon::Expression*> { return expression; }

 private:
  Nonnull<Carbon::Expression*> expression;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_PATTERN_H_
