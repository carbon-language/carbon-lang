// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
#define EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/paren_contents.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class Expression {
 public:
  enum class Kind {
    BoolTypeLiteral,
    BoolLiteral,
    CallExpression,
    FunctionTypeLiteral,
    FieldAccessExpression,
    IndexExpression,
    IntTypeLiteral,
    ContinuationTypeLiteral,  // The type of a continuation value.
    IntLiteral,
    PrimitiveOperatorExpression,
    StringLiteral,
    StringTypeLiteral,
    TupleLiteral,
    TypeTypeLiteral,
    IdentifierExpression,
    IntrinsicExpression,
  };

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto Tag() const -> Kind { return tag; }

  auto LineNumber() const -> int { return line_num; }

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

 protected:
  // Constructs an Expression representing syntax at the given line number.
  // `tag` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Expression(Kind tag, int line_num) : tag(tag), line_num(line_num) {}

 private:
  const Kind tag;
  int line_num;
};

// Converts paren_contents to an Expression, interpreting the parentheses as
// grouping if their contents permit that interpretation, or as forming a
// tuple otherwise.
auto ExpressionFromParenContents(
    int line_num, const ParenContents<Expression>& paren_contents)
    -> const Expression*;

// Converts paren_contents to an Expression, interpreting the parentheses as
// forming a tuple.
auto TupleExpressionFromParenContents(
    int line_num, const ParenContents<Expression>& paren_contents)
    -> const Expression*;

// A FieldInitializer represents the initialization of a single tuple field.
struct FieldInitializer {
  FieldInitializer(std::string name, const Expression* expression)
      : name(std::move(name)), expression(expression) {}

  // The field name. Cannot be empty.
  std::string name;

  // The expression that initializes the field.
  const Expression* expression;
};

enum class Operator {
  Add,
  And,
  Deref,
  Eq,
  Mul,
  Neg,
  Not,
  Or,
  Sub,
  Ptr,
};

class IdentifierExpression : public Expression {
 public:
  explicit IdentifierExpression(int line_num, std::string name)
      : Expression(Kind::IdentifierExpression, line_num),
        name(std::move(name)) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::IdentifierExpression;
  }

  auto Name() const -> const std::string& { return name; }

 private:
  std::string name;
};

class FieldAccessExpression : public Expression {
 public:
  explicit FieldAccessExpression(int line_num, const Expression* aggregate,
                                 std::string field)
      : Expression(Kind::FieldAccessExpression, line_num),
        aggregate(aggregate),
        field(std::move(field)) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::FieldAccessExpression;
  }

  auto Aggregate() const -> const Expression* { return aggregate; }
  auto Field() const -> const std::string& { return field; }

 private:
  const Expression* aggregate;
  std::string field;
};

class IndexExpression : public Expression {
 public:
  explicit IndexExpression(int line_num, const Expression* aggregate,
                           const Expression* offset)
      : Expression(Kind::IndexExpression, line_num),
        aggregate(aggregate),
        offset(offset) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::IndexExpression;
  }

  auto Aggregate() const -> const Expression* { return aggregate; }
  auto Offset() const -> const Expression* { return offset; }

 private:
  const Expression* aggregate;
  const Expression* offset;
};

class IntLiteral : public Expression {
 public:
  explicit IntLiteral(int line_num, int val)
      : Expression(Kind::IntLiteral, line_num), val(val) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::IntLiteral;
  }

  auto Val() const -> int { return val; }

 private:
  int val;
};

class BoolLiteral : public Expression {
 public:
  explicit BoolLiteral(int line_num, bool val)
      : Expression(Kind::BoolLiteral, line_num), val(val) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::BoolLiteral;
  }

  auto Val() const -> bool { return val; }

 private:
  bool val;
};

class StringLiteral : public Expression {
 public:
  explicit StringLiteral(int line_num, std::string val)
      : Expression(Kind::StringLiteral, line_num), val(std::move(val)) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::StringLiteral;
  }

  auto Val() const -> const std::string& { return val; }

 private:
  std::string val;
};

class StringTypeLiteral : public Expression {
 public:
  explicit StringTypeLiteral(int line_num)
      : Expression(Kind::StringTypeLiteral, line_num) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::StringTypeLiteral;
  }
};

class TupleLiteral : public Expression {
 public:
  explicit TupleLiteral(int line_num) : TupleLiteral(line_num, {}) {}

  explicit TupleLiteral(int line_num, std::vector<FieldInitializer> fields)
      : Expression(Kind::TupleLiteral, line_num), fields(std::move(fields)) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::TupleLiteral;
  }

  auto Fields() const -> const std::vector<FieldInitializer>& { return fields; }

 private:
  std::vector<FieldInitializer> fields;
};

class PrimitiveOperatorExpression : public Expression {
 public:
  explicit PrimitiveOperatorExpression(int line_num, Operator op,
                                       std::vector<const Expression*> arguments)
      : Expression(Kind::PrimitiveOperatorExpression, line_num),
        op(op),
        arguments(std::move(arguments)) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::PrimitiveOperatorExpression;
  }

  auto Op() const -> Operator { return op; }
  auto Arguments() const -> const std::vector<const Expression*>& {
    return arguments;
  }

 private:
  Operator op;
  std::vector<const Expression*> arguments;
};

class CallExpression : public Expression {
 public:
  explicit CallExpression(int line_num, const Expression* function,
                          const Expression* argument)
      : Expression(Kind::CallExpression, line_num),
        function(function),
        argument(argument) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::CallExpression;
  }

  auto Function() const -> const Expression* { return function; }
  auto Argument() const -> const Expression* { return argument; }

 private:
  const Expression* function;
  const Expression* argument;
};

class FunctionTypeLiteral : public Expression {
 public:
  explicit FunctionTypeLiteral(int line_num, const Expression* parameter,
                               const Expression* return_type,
                               bool is_omitted_return_type)
      : Expression(Kind::FunctionTypeLiteral, line_num),
        parameter(parameter),
        return_type(return_type),
        is_omitted_return_type(is_omitted_return_type) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::FunctionTypeLiteral;
  }

  auto Parameter() const -> const Expression* { return parameter; }
  auto ReturnType() const -> const Expression* { return return_type; }
  auto IsOmittedReturnType() const -> bool { return is_omitted_return_type; }

 private:
  const Expression* parameter;
  const Expression* return_type;
  bool is_omitted_return_type;
};

class BoolTypeLiteral : public Expression {
 public:
  explicit BoolTypeLiteral(int line_num)
      : Expression(Kind::BoolTypeLiteral, line_num) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::BoolTypeLiteral;
  }
};

class IntTypeLiteral : public Expression {
 public:
  explicit IntTypeLiteral(int line_num)
      : Expression(Kind::IntTypeLiteral, line_num) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::IntTypeLiteral;
  }
};

class ContinuationTypeLiteral : public Expression {
 public:
  explicit ContinuationTypeLiteral(int line_num)
      : Expression(Kind::ContinuationTypeLiteral, line_num) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::ContinuationTypeLiteral;
  }
};

class TypeTypeLiteral : public Expression {
 public:
  explicit TypeTypeLiteral(int line_num)
      : Expression(Kind::TypeTypeLiteral, line_num) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::TypeTypeLiteral;
  }
};

class IntrinsicExpression : public Expression {
 public:
  enum class IntrinsicKind {
    Print,
  };

  explicit IntrinsicExpression(IntrinsicKind intrinsic)
      : Expression(Kind::IntrinsicExpression, -1), intrinsic(intrinsic) {}

  static auto classof(const Expression* exp) -> bool {
    return exp->Tag() == Kind::IntrinsicExpression;
  }

  auto Intrinsic() const -> IntrinsicKind { return intrinsic; }

 private:
  IntrinsicKind intrinsic;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_EXPRESSION_H_
