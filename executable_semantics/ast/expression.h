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

struct Value;
struct TCResult;
struct TCContext_;
struct Action;
struct Frame;

// An existential AST expression satisfying the Expression concept.
class Expression {
 public:  // ValueSemantic concept API.
  // TODO: REMOVE THIS WHEN WE FIGURE OUT HOW TO GET BISON TO COOPERATE
  Expression() = default;
  Expression(const Expression& other) = default;
  Expression& operator=(const Expression& other) = default;

  // Constructs an instance equivalent to `e`, where `Model` satisfies the
  // Expression concept.
  template <class Model>
  Expression(Model e) : box(std::make_shared<Boxed<Model>>(e)) {}

  // Makes *this equivalent to `e`, where `Model` satisfies the
  // Expression concept.
  template <class Model>
  Expression& operator=(const Model& e) {
    box = std::make_shared<Boxed<Model>>(e);
    return *this;
  }

 public:  // Expression concept API, in addition to ValueSemantic.
  auto Print() const -> void { box->Print(); }
  auto StepLvalue(Action* act, Frame* frame) const -> void {
    box->StepLvalue(act, frame);
  }
  auto StepExp(Action* act, Frame* frame) const -> void {
    box->StepExp(act, frame);
  }
  auto LValAction(Action* act, Frame* frame) const -> void {
    box->LValAction(act, frame);
  }
  auto ExpressionAction(Action* act, Frame* frame) const -> void {
    box->ExpressionAction(act, frame);
  }

  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;
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
    virtual auto StepLvalue(Action* act, Frame* frame) const -> void = 0;
    virtual auto StepExp(Action* act, Frame* frame) const -> void = 0;
    virtual auto LValAction(Action* act, Frame* frame) const -> void = 0;
    virtual auto ExpressionAction(Action* act, Frame* frame) const -> void = 0;
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
    auto StepLvalue(Action* act, Frame* frame) const -> void {
      content.StepLvalue(act, frame);
    }
    auto StepExp(Action* act, Frame* frame) const -> void {
      content.StepExp(act, frame);
    }
    auto LValAction(Action* act, Frame* frame) const -> void {
      content.LValAction(act, frame);
    }
    auto ExpressionAction(Action* act, Frame* frame) const -> void {
      content.ExpressionAction(act, frame);
    }
    auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                   TCContext_ context) const -> TCResult;
    auto SourceLocation() const -> int { return content.location.line_number; }
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
    explicit Location(int line_number) : line_number(line_number) {}

    int line_number;
  };

  Location location;

 protected:
  explicit ExpressionSource(Location location) : location(location) {}
  auto FatalLValAction() const -> void;
  auto FatalBadExpressionContext() const -> void;
};

struct AutoTypeExpression : ExpressionSource {
  AutoTypeExpression(Location textual_placement)
      : ExpressionSource(textual_placement) {}
  auto Print() const -> void;
  auto StepLvalue(Action* act, Frame* frame) const -> void;
  auto StepExp(Action* act, Frame* frame) const -> void;
  auto LValAction(Action* act, Frame* frame) const -> void {
    FatalLValAction();
  }
  auto ExpressionAction(Action* act, Frame* frame) const -> void {
    FatalBadExpressionContext();
  }
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;
};

struct BoolTypeExpression : ExpressionSource {
  BoolTypeExpression(Location textual_placement)
      : ExpressionSource(textual_placement) {}
  auto Print() const -> void;
  auto StepLvalue(Action* act, Frame* frame) const -> void;
  auto StepExp(Action* act, Frame* frame) const -> void;
  auto LValAction(Action* act, Frame* frame) const -> void {
    FatalLValAction();
  }
  auto ExpressionAction(Action* act, Frame* frame) const -> void {
    FatalBadExpressionContext();
  }
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;
};

struct BooleanExpression : ExpressionSource {
  BooleanExpression(Location textual_placement, bool value)
      : ExpressionSource(textual_placement), value(value) {}
  auto Print() const -> void;
  auto StepLvalue(Action* act, Frame* frame) const -> void;
  auto StepExp(Action* act, Frame* frame) const -> void;
  auto LValAction(Action* act, Frame* frame) const -> void {
    FatalLValAction();
  }
  auto ExpressionAction(Action* act, Frame* frame) const -> void {
    FatalBadExpressionContext();
  }
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;

  bool value;
  int source_location;
};

struct CallExpression : ExpressionSource {
  CallExpression(Location textual_placement, Expression function,
                 Expression argument_tuple)
      : ExpressionSource(textual_placement),
        function(function),
        argument_tuple(argument_tuple) {}

  auto Print() const -> void;
  auto StepLvalue(Action* act, Frame* frame) const -> void;
  auto StepExp(Action* act, Frame* frame) const -> void;
  auto LValAction(Action* act, Frame* frame) const -> void {
    FatalLValAction();
  }
  auto ExpressionAction(Action* act, Frame* frame) const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;

  Expression function;
  // If not a tuple expression, this is the sole argument to a unary call.
  Expression argument_tuple;
};

struct FunctionTypeExpression : ExpressionSource {
  FunctionTypeExpression(Location textual_placement,
                         Expression parameter_tuple_type,
                         Expression return_type)
      : ExpressionSource(textual_placement),
        parameter_tuple_type(parameter_tuple_type),
        return_type(return_type) {}

  auto Print() const -> void;
  auto StepLvalue(Action* act, Frame* frame) const -> void;
  auto StepExp(Action* act, Frame* frame) const -> void;
  auto LValAction(Action* act, Frame* frame) const -> void {
    FatalLValAction();
  }
  auto ExpressionAction(Action* act, Frame* frame) const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;

  Expression parameter_tuple_type;
  Expression return_type;
};

struct GetFieldExpression : ExpressionSource {
  GetFieldExpression(Location textual_placement, Expression aggregate,
                     std::string const& field_name)
      : ExpressionSource(textual_placement),
        aggregate(aggregate),
        field_name(field_name) {}

  auto Print() const -> void;
  auto StepLvalue(Action* act, Frame* frame) const -> void;
  auto StepExp(Action* act, Frame* frame) const -> void;
  auto LValAction(Action* act, Frame* frame) const -> void;
  auto ExpressionAction(Action* act, Frame* frame) const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;

  Expression aggregate;
  std::string field_name;
};

struct IndexExpression : ExpressionSource {
  IndexExpression(Location textual_placement, Expression aggregate,
                  Expression offset)
      : ExpressionSource(textual_placement),
        aggregate(aggregate),
        offset(offset) {}

  auto Print() const -> void;
  auto StepLvalue(Action* act, Frame* frame) const -> void;
  auto StepExp(Action* act, Frame* frame) const -> void;
  auto LValAction(Action* act, Frame* frame) const -> void;
  auto ExpressionAction(Action* act, Frame* frame) const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;

  Expression aggregate;
  Expression offset;
};

struct IntTypeExpression : ExpressionSource {
  IntTypeExpression(Location textual_placement)
      : ExpressionSource(textual_placement) {}

  auto Print() const -> void;
  auto StepLvalue(Action* act, Frame* frame) const -> void;
  auto StepExp(Action* act, Frame* frame) const -> void;
  auto LValAction(Action* act, Frame* frame) const -> void {
    FatalLValAction();
  }
  auto ExpressionAction(Action* act, Frame* frame) const -> void {
    FatalBadExpressionContext();
  }
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;
};

struct IntegerExpression : ExpressionSource {
  IntegerExpression(Location textual_placement, int value)
      : ExpressionSource(textual_placement), value(value) {}

  auto Print() const -> void;
  auto StepLvalue(Action* act, Frame* frame) const -> void;
  auto StepExp(Action* act, Frame* frame) const -> void;
  auto LValAction(Action* act, Frame* frame) const -> void {
    FatalLValAction();
  }
  auto ExpressionAction(Action* act, Frame* frame) const -> void {
    FatalBadExpressionContext();
  }
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;

  int value;
};

struct PatternVariableExpression : ExpressionSource {
  PatternVariableExpression(Location textual_placement, const std::string& name,
                            Expression type)
      : ExpressionSource(textual_placement), name(name), type(type) {}

  auto Print() const -> void;
  auto StepLvalue(Action* act, Frame* frame) const -> void;
  auto StepExp(Action* act, Frame* frame) const -> void;
  auto LValAction(Action* act, Frame* frame) const -> void {
    FatalLValAction();
  }
  auto ExpressionAction(Action* act, Frame* frame) const -> void;
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

  PrimitiveOperatorExpression(Location textual_placement, Operation operation,
                              std::vector<Expression> arguments)
      : ExpressionSource(textual_placement),
        operation(operation),
        arguments(arguments) {}

  auto Print() const -> void;
  auto StepLvalue(Action* act, Frame* frame) const -> void;
  auto StepExp(Action* act, Frame* frame) const -> void;
  auto LValAction(Action* act, Frame* frame) const -> void {
    FatalLValAction();
  }
  auto ExpressionAction(Action* act, Frame* frame) const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;

  Operation operation;
  std::vector<Expression> arguments;
};

struct TupleExpression : ExpressionSource {
  // Creates an instance storing a copy of elements with each empty name
  // replaced by a string representation of its index.
  TupleExpression(Location textual_placement,
                  std::vector<std::pair<std::string, Expression>> elements)
      : ExpressionSource(textual_placement), elements(elements) {
    int i = 0;
    for (auto& e : this->elements) {
      if (e.first.empty()) {
        e.first = std::to_string(i);
        ++i;
      }
    }
  }

  auto Print() const -> void;
  auto StepLvalue(Action* act, Frame* frame) const -> void;
  auto StepExp(Action* act, Frame* frame) const -> void;
  auto LValAction(Action* act, Frame* frame) const -> void;
  auto ExpressionAction(Action* act, Frame* frame) const -> void;
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;

  std::vector<std::pair<std::string, Expression>> elements;
};

struct TypeTypeExpression : ExpressionSource {
  explicit TypeTypeExpression(Location location) : ExpressionSource(location) {}

  auto Print() const -> void;
  auto StepLvalue(Action* act, Frame* frame) const -> void;
  auto StepExp(Action* act, Frame* frame) const -> void;
  auto LValAction(Action* act, Frame* frame) const -> void {
    FatalLValAction();
  }
  auto ExpressionAction(Action* act, Frame* frame) const -> void {
    FatalBadExpressionContext();
  }
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;
};

struct VariableExpression : ExpressionSource {
  std::string name;

  VariableExpression(Location textual_placement, const std::string& name)
      : ExpressionSource(textual_placement), name(name) {}

  auto Print() const -> void;
  auto StepLvalue(Action* act, Frame* frame) const -> void;
  auto StepExp(Action* act, Frame* frame) const -> void;
  auto LValAction(Action* act, Frame* frame) const -> void {
    FatalLValAction();
  }
  auto ExpressionAction(Action* act, Frame* frame) const -> void {
    FatalBadExpressionContext();
  }
  auto TypeCheck(TypeEnv env, Env ct_env, Value* expected,
                 TCContext_ context) const -> TCResult;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
