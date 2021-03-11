// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
#define EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_

#include <optional>
#include <string>
#include <vector>

#include "executable_semantics/interpreter/vocabulary_types.h"

namespace Carbon {

class Value;
struct TCResult;
struct TCContext_;

// An existential AST expression satisfying the Expression concept.
class Expression {
 public:  // ValueSemantic concept API.
  Expression(const Expression& other) = default;
  Expression& operator=(const Expression& other) = default;

  // Constructs an instance equivalent to `d`, where `Model` satisfies the
  // Expression concept.
  template <class Model>
  Expression(Model d) : box(std::make_shared<Boxed<Model>>(d)) {}

 public:  // Expression concept API, in addition to ValueSemantic.
  auto Print() const -> void { box->Print(); }
  auto ToValue() const -> Value* { return box->ToValue(); }
  auto StepLvalue() const -> void { box->StepLvalue(); }
  auto StepExp() const -> void { box->StepExp(); }
  auto HandleValue() const -> void { box->HandleValue(); }
  auto HandleAction() const -> void { box->HandleAction(); }
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected, TCContext_ context)
      -> TCResult;
  auto SourceLocation() const -> int { return box->SourceLocation(); }

 public:  // Dynamic casting/unwrapping
  // Returns the instance of Concrete that *this notionally is, or nullopt if
  // *this notionally has some other concrete type.
  template <class Content>
  auto As() const -> std::optional<Content> {
    const auto* p = dynamic_cast<const Boxed<Content>*>(box.get());
    return p ? std::optional(p->content) : std::nullopt;
  }

 private:  // types
  /// A base class that erases the type of a `Boxed<Content>`, where `Content`
  /// satisfies the Expression concept.
  struct Box {
   protected:
    Box() {}

   public:
    Box(const Box& other) = delete;
    Box& operator=(const Box& other) = delete;

    virtual ~Box() {}
    virtual auto Print() const -> void = 0;
    virtual auto ToValue() const -> Value* = 0;
    virtual auto StepLvalue() const -> void = 0;
    virtual auto StepExp() const -> void = 0;
    virtual auto HandleValue() const -> void = 0;
    virtual auto HandleAction() const -> void = 0;
    virtual auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                           TCContext_ context) const -> TCResult = 0;
    virtual auto SourceLocation() const -> int = 0;
  };

  /// The derived class that holds an instance of `Content` satisfying the
  /// Expression concept.
  template <class Content>
  struct Boxed final : Box {
    const Content content;
    explicit Boxed(Content content) : Box(), content(content) {}

    auto Print() const -> void { content.Print(); }
    auto ToValue() const -> Value* { return content.ToValue(); }
    auto StepLvalue() const -> void { content.StepLvalue(); }
    auto StepExp() const -> void { content.StepExp(); }
    auto HandleValue() const -> void { content.HandleValue(); }
    auto HandleAction() const -> void { content.HandleAction(); }
    auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                   TCContext_ context) const -> TCResult;
    auto SourceLocation() const -> int { return content.sourceLocation; }
  };

 private:  // data members
  // Note: the pointee is const as long as we have no mutating methods. When
  std::shared_ptr<const Box> box;
};

// Base class of all expressions in the source code.
class ExpressionSource {
 public:
  // A location in the the source code.
  struct Location {
    // TODO: replace with yy::parser::location or a wrapper thereof.
    // TODO: this shouldn't be specific to expressions.
    explicit Location(int lineNumber) : lineNumber(lineNumber) {}

    int lineNumber;
  };

  Location location;

 protected:
  ExpressionSource(Location location) : location(location) {}
};

struct AutoTypeExpression : ExpressionSource {
  AutoTypeExpression(Location textualPlacement)
      : ExpressionSource(textualPlacement) {}
  auto Print() const -> void;
  auto ToValue() const -> Value*;
  auto StepLvalue() const -> void;
  auto StepExp() const -> void;
  auto HandleValue() const -> void;
  auto HandleAction() const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;
};

struct BoolTypeExpression : ExpressionSource {
  BoolTypeExpression(Location textualPlacement)
      : ExpressionSource(textualPlacement) {}
  auto Print() const -> void;
  auto ToValue() const -> Value*;
  auto StepLvalue() const -> void;
  auto StepExp() const -> void;
  auto HandleValue() const -> void;
  auto HandleAction() const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;
};

struct BooleanExpression : ExpressionSource {
  BooleanExpression(Location textualPlacement, bool value)
      : ExpressionSource(textualPlacement), value(value) {}
  auto Print() const -> void;
  auto ToValue() const -> Value*;
  auto StepLvalue() const -> void;
  auto StepExp() const -> void;
  auto HandleValue() const -> void;
  auto HandleAction() const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;

  bool value;
  int sourceLocation;
};

struct CallExpression : ExpressionSource {
  CallExpression(Location textualPlacement, Expression function,
                 Expression argumentTuple)
      : ExpressionSource(textualPlacement),
        function(function),
        argumentTuple(argumentTuple) {}

  auto Print() const -> void;
  auto ToValue() const -> Value*;
  auto StepLvalue() const -> void;
  auto StepExp() const -> void;
  auto HandleValue() const -> void;
  auto HandleAction() const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;

  Expression function;
  // If not a tuple expression, this is the sole argument to a unary call.
  Expression argumentTuple;
};

struct FunctionTypeExpression : ExpressionSource {
  FunctionTypeExpression(Location textualPlacement,
                         Expression parameterTupleType, Expression returnType)
      : ExpressionSource(textualPlacement),
        parameterTupleType(parameterTupleType),
        returnType(returnType) {}

  auto Print() const -> void;
  auto ToValue() const -> Value*;
  auto StepLvalue() const -> void;
  auto StepExp() const -> void;
  auto HandleValue() const -> void;
  auto HandleAction() const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;

  Expression parameterTupleType;
  Expression returnType;
};

struct GetFieldExpression : ExpressionSource {
  GetFieldExpression(Location textualPlacement, Expression aggregate,
                     std::string const& fieldName)
      : ExpressionSource(textualPlacement),
        aggregate(aggregate),
        fieldName(fieldName) {}

  auto Print() const -> void;
  auto ToValue() const -> Value*;
  auto StepLvalue() const -> void;
  auto StepExp() const -> void;
  auto HandleValue() const -> void;
  auto HandleAction() const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;

  Expression aggregate;
  std::string fieldName;
};

struct IndexExpression : ExpressionSource {
  IndexExpression(Location textualPlacement, Expression aggregate,
                  Expression offset)
      : ExpressionSource(textualPlacement),
        aggregate(aggregate),
        offset(offset) {}

  auto Print() const -> void;
  auto ToValue() const -> Value*;
  auto StepLvalue() const -> void;
  auto StepExp() const -> void;
  auto HandleValue() const -> void;
  auto HandleAction() const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;

  Expression aggregate;
  Expression offset;
};

struct IntTypeExpression : ExpressionSource {
  IntTypeExpression(Location textualPlacement)
      : ExpressionSource(textualPlacement) {}

  auto Print() const -> void;
  auto ToValue() const -> Value*;
  auto StepLvalue() const -> void;
  auto StepExp() const -> void;
  auto HandleValue() const -> void;
  auto HandleAction() const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;
};

struct IntegerExpression : ExpressionSource {
  IntegerExpression(Location textualPlacement, int value)
      : ExpressionSource(textualPlacement), value(value) {}

  auto Print() const -> void;
  auto ToValue() const -> Value*;
  auto StepLvalue() const -> void;
  auto StepExp() const -> void;
  auto HandleValue() const -> void;
  auto HandleAction() const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;

  int value;
};

struct PatternVariableExpression : ExpressionSource {
  PatternVariableExpression(Location textualPlacement, const std::string& name,
                            Expression type)
      : ExpressionSource(textualPlacement), name(name), type(type) {}

  auto Print() const -> void;
  auto ToValue() const -> Value*;
  auto StepLvalue() const -> void;
  auto StepExp() const -> void;
  auto HandleValue() const -> void;
  auto HandleAction() const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;

  std::string name;
  Expression type;
};

struct PrimitiveOperatorExpression : ExpressionSource {
  enum class Operation {
    Add,
    And,
    Eq,
    Neg,
    Not,
    Or,
    Sub,
  };

  // Creates a unary operator expression
  template <class... Arguments>
  PrimitiveOperatorExpression(Location textualPlacement, Operation operation,
                              Arguments... arguments);

  auto Print() const -> void;
  auto ToValue() const -> Value*;
  auto StepLvalue() const -> void;
  auto StepExp() const -> void;
  auto HandleValue() const -> void;
  auto HandleAction() const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;

  Operation operation;
  std::vector<Expression> arguments;

 private:
  // Creates an N-ary operator expression
  PrimitiveOperatorExpression(Location textualPlacement, Operation operation,
                              const std::vector<Expression>& arguments)
      : ExpressionSource(textualPlacement),
        operation(operation),
        arguments(arguments) {}
};

struct TupleExpression : ExpressionSource {
  // Creates an instance storing a copy of elements, except that each empty name
  // in is replaced by a string representation of its index.
  TupleExpression(Location textualPlacement,
                  std::vector<std::pair<std::string, Expression>> elements)
      : ExpressionSource(textualPlacement), elements(elements) {
    int i = 0;
    for (auto& e : this->elements) {
      if (e.first.empty()) {
        e.first = std::to_string(i);
        ++i;
      }
    }
  }

  auto Print() const -> void;
  auto ToValue() const -> Value*;
  auto StepLvalue() const -> void;
  auto StepExp() const -> void;
  auto HandleValue() const -> void;
  auto HandleAction() const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;

  std::vector<std::pair<std::string, Expression>> elements;
};

struct TypeTypeExpression : ExpressionSource {
  auto Print() const -> void;
  auto ToValue() const -> Value*;
  auto StepLvalue() const -> void;
  auto StepExp() const -> void;
  auto HandleValue() const -> void;
  auto HandleAction() const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;
};

struct VariableExpression : ExpressionSource {
  std::string name;

  VariableExpression(Location textualPlacement, const std::string& name)
      : ExpressionSource(textualPlacement), name(name) {}

  auto Print() const -> void;
  auto ToValue() const -> Value*;
  auto StepLvalue() const -> void;
  auto StepExp() const -> void;
  auto HandleValue() const -> void;
  auto HandleAction() const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
