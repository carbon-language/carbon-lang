// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FORTRAN_EVALUATE_EXPRESSION_H_
#define FORTRAN_EVALUATE_EXPRESSION_H_

#include "common.h"
#include "type.h"
#include <memory>
#include <ostream>
#include <variant>

namespace Fortran::evaluate {

// First, a statically polymorphic representation of expressions that's
// specialized across the type categories of Fortran.  Subexpression
// operands are implemented with owning pointers that should not be null
// unless an expression is being decomposed.
// Every Expression specialization has (at least) these type aliases:
//   using Result = Type<category, kind>
//   using Operand = std::unique_ptr<Expression<Result>>
//   using Constant = typename Result::Value  // e.g., value::Integer<BITS>
// nested declarations of wrapper structs for each operation, e.g.
//   struct Add { Operand x, y; };
// a data member to hold an instance of one of these structs:
//   std::variant<> u;
// and a formatting member function, Dump().
template<typename T> struct Expression;

template<typename T> using ExprOperand = std::unique_ptr<Expression<T>>;
template<typename T, typename... A> ExprOperand<T> Opd(A &&... args) {
  return std::make_unique<Expression<T>>(std::forward<A>(args)...);
}

// Dynamically polymorphic operands that can hold any supported kind.
struct IntegerOperand {
  template<int KIND> using Operand = ExprOperand<Type<Category::Integer, KIND>>;
  IntegerKindsVariant<Operand> u;
};
struct RealOperand {
  template<int KIND> using Operand = ExprOperand<Type<Category::Real, KIND>>;
  RealKindsVariant<Operand> u;
};
struct CharacterOperand {
  template<int KIND>
  using Operand = ExprOperand<Type<Category::Character, KIND>>;
  CharacterKindsVariant<Operand> u;
};

template<int KIND> struct Expression<Type<Category::Integer, KIND>> {
  static constexpr Category category{Category::Integer};
  static constexpr int kind{KIND};
  using Result = Type<category, kind>;
  using Operand = ExprOperand<Result>;
  using Constant = typename Result::Value;
  struct Convert {
    std::variant<IntegerOperand, RealOperand> u;
  };
  struct Unary {
    Operand x;
  };
  struct Parentheses : public Unary {};
  struct Negate : public Unary {};
  struct Binary {
    Operand x, y;
  };
  struct Add : public Binary {};
  struct Subtract : public Binary {};
  struct Multiply : public Binary {};
  struct Divide : public Binary {};
  struct Power : public Binary {};

  Expression() = delete;
  Expression(Expression &&) = default;
  Expression(const Constant &x) : u{x} {}
  Expression(std::int64_t n) : u{Constant{n}} {}
  Expression(int n) : u{Constant{n}} {}
  template<typename A> Expression(A &&x) : u{std::move(x)} {}
  std::ostream &Dump(std::ostream &) const;
  std::variant<Constant, Convert, Parentheses, Negate, Add, Subtract, Multiply,
      Divide, Power>
      u;
};

template<int KIND> struct Expression<Type<Category::Real, KIND>> {
  static constexpr Category category{Category::Real};
  static constexpr int kind{KIND};
  using Result = Type<category, kind>;
  using Operand = ExprOperand<Result>;
  using Constant = typename Result::Value;
  struct Convert {
    // N.B. Real->Complex and Complex->Real conversions are done with CMPLX
    // and part access operations (resp.).  Conversions between kinds of
    // Complex are done via decomposition to Real and reconstruction.
    std::variant<IntegerOperand, RealOperand> u;
  };
  struct Unary {
    Operand x;
  };
  struct Parentheses : public Unary {};
  struct Negate : public Unary {};
  struct Binary {
    Operand x, y;
  };
  struct Add : public Binary {};
  struct Subtract : public Binary {};
  struct Multiply : public Binary {};
  struct Divide : public Binary {};
  struct Power : public Binary {};
  struct IntegerPower {
    Operand x;
    IntegerOperand y;
  };
  struct FromComplex {
    ExprOperand<typename Result::Complex> x;
  };
  struct RealPart : public FromComplex {};
  struct AIMAG : public FromComplex {};

  Expression() = delete;
  Expression(Expression &&) = default;
  Expression(const Constant &x) : u{x} {}
  template<typename A> Expression(A &&x) : u{std::move(x)} {}
  std::ostream &Dump(std::ostream &) const;
  std::variant<Constant, Convert, Parentheses, Negate, Add, Subtract, Multiply,
      Divide, Power, IntegerPower, RealPart, AIMAG>
      u;
};

template<int KIND> struct Expression<Type<Category::Complex, KIND>> {
  static constexpr Category category{Category::Complex};
  static constexpr int kind{KIND};
  using Result = Type<category, kind>;
  using Operand = ExprOperand<Result>;
  using Constant = typename Result::Value;
  struct Unary {
    Operand x;
  };
  struct Parentheses : public Unary {};
  struct Negate : public Unary {};
  struct Binary {
    Operand x, y;
  };
  struct Add : public Binary {};
  struct Subtract : public Binary {};
  struct Multiply : public Binary {};
  struct Divide : public Binary {};
  struct Power : public Binary {};
  struct IntegerPower {
    Operand x;
    IntegerOperand y;
  };
  struct CMPLX {
    ExprOperand<typename Result::Part> re, im;
  };

  Expression() = delete;
  Expression(Expression &&) = default;
  Expression(const Constant &x) : u{x} {}
  template<typename A> Expression(A &&x) : u{std::move(x)} {}
  std::ostream &Dump(std::ostream &) const;
  std::variant<Constant, Parentheses, Negate, Add, Subtract, Multiply, Divide,
      Power, IntegerPower, CMPLX>
      u;
};

// No need to distinguish the various kinds of LOGICAL in expressions,
// only in memory.
using LogicalResult = Type<Category::Logical, 1>;

template<typename T> struct Comparison {
  using Result = LogicalResult;
  using Operand = ExprOperand<T>;
  struct Binary {
    Operand x, y;
  };
  struct LT : public Binary {};
  struct LE : public Binary {};
  struct EQ : public Binary {};
  struct NE : public Binary {};
  struct GE : public Binary {};
  struct GT : public Binary {};
  // TODO: .UN. extension?

  Comparison() = delete;
  Comparison(Comparison &&) = default;
  template<typename A> Comparison(A &&x) : u{std::move(x)} {}

  std::ostream &Dump(std::ostream &) const;

  std::variant<LT, LE, EQ, NE, GE, GT> u;
};

template<int KIND> struct Comparison<Type<Category::Complex, KIND>> {
  using Result = LogicalResult;
  using Operand = ExprOperand<Type<Category::Complex, KIND>>;
  struct Binary {
    Operand x, y;
  };
  struct EQ : public Binary {};
  struct NE : public Binary {};
  // TODO: .UN. extension?

  Comparison() = delete;
  Comparison(Comparison &&) = default;
  template<typename A> Comparison(A &&x) : u{std::move(x)} {}

  std::ostream &Dump(std::ostream &) const;

  std::variant<EQ, NE> u;
};

struct IntegerComparison {
  template<int KIND> using C = Comparison<Type<Category::Integer, KIND>>;
  IntegerKindsVariant<C> u;
};

struct RealComparison {
  template<int KIND> using C = Comparison<Type<Category::Real, KIND>>;
  RealKindsVariant<C> u;
};

struct ComplexComparison {
  template<int KIND> using C = Comparison<Type<Category::Complex, KIND>>;
  RealKindsVariant<C> u;
};

struct CharacterComparison {
  template<int KIND> using C = Comparison<Type<Category::Character, KIND>>;
  CharacterKindsVariant<C> u;
};

template<> struct Expression<LogicalResult> {
  using Result = LogicalResult;
  using Operand = ExprOperand<Result>;
  using Constant = typename Result::Value;
  struct Not {
    Operand x;
  };
  struct Binary {
    Operand x, y;
  };
  struct And : public Binary {};
  struct Or : public Binary {};
  struct Eqv : public Binary {};
  struct Neqv : public Binary {};

  Expression() = delete;
  Expression(Expression &&) = default;
  Expression(const Constant &x) : u{x} {}
  template<int KIND>
  Expression(Comparison<Type<Category::Integer, KIND>> &&x)
    : u{IntegerComparison{std::move(x)}} {}
  template<int KIND>
  Expression(Comparison<Type<Category::Real, KIND>> &&x)
    : u{RealComparison{std::move(x)}} {}
  template<int KIND>
  Expression(Comparison<Type<Category::Complex, KIND>> &&x)
    : u{ComplexComparison{std::move(x)}} {}
  template<int KIND>
  Expression(Comparison<Type<Category::Character, KIND>> &&x)
    : u{CharacterComparison{std::move(x)}} {}
  template<typename A> Expression(A &&x) : u{std::move(x)} {}

  std::ostream &Dump(std::ostream &) const;

  std::variant<Constant, Not, And, Or, Eqv, Neqv, IntegerComparison,
      RealComparison, ComplexComparison, CharacterComparison>
      u;
};

template<int KIND> struct Expression<Type<Category::Character, KIND>> {
  static constexpr Category category{Category::Character};
  static constexpr int kind{KIND};
  using Result = Type<category, kind>;
  using Operand = ExprOperand<Result>;
  using Constant = typename Result::Value;
  using Length = Expression<IntrinsicTypeParameterType>;
  struct Concat {
    Operand x, y;
  };

  Expression() = delete;
  Expression(Expression &&) = default;
  Expression(const Constant &x) : u{x} {}
  Expression(Expression &&a, Expression &&b)
    : u{Concat{Opd<Result>(std::move(a)), Opd<Result>(std::move(b))}} {}

  Length LEN() const;
  std::ostream &Dump(std::ostream &) const;

  std::variant<Constant, Concat> u;
};

// Convenience type aliases
template<int KIND> using IntExpr = Expression<Type<Category::Integer, KIND>>;
template<int KIND> using RealExpr = Expression<Type<Category::Real, KIND>>;
template<int KIND>
using ComplexExpr = Expression<Type<Category::Complex, KIND>>;
using LogicalExpr = Expression<LogicalResult>;
template<int KIND> using CharExpr = Expression<Type<Category::Character, KIND>>;
using DefaultIntExpr = Expression<DefaultInteger>;

// Dynamically polymorphic representation of expressions across kinds
struct IntegerExpression {
  IntegerKindsVariant<IntExpr> u;
};
struct RealExpression {
  RealKindsVariant<RealExpr> u;
};
struct ComplexExpression {
  ComplexKindsVariant<ComplexExpr> u;
};
struct CharacterExpression {
  CharacterKindsVariant<CharExpr> u;
};

// Dynamically polymorphic representation of expressions across categories
struct ArbitraryExpression {
  ArbitraryExpression() = delete;
  ArbitraryExpression(ArbitraryExpression &&) = default;
  template<int KIND>
  ArbitraryExpression(IntExpr<KIND> &&x) : u{IntegerExpression{std::move(x)}} {}
  template<int KIND>
  ArbitraryExpression(RealExpr<KIND> &&x) : u{RealExpression{std::move(x)}} {}
  template<int KIND>
  ArbitraryExpression(ComplexExpr<KIND> &&x)
    : u{ComplexExpression{std::move(x)}} {}
  template<int KIND>
  ArbitraryExpression(CharExpr<KIND> &&x)
    : u{CharacterExpression{std::move(x)}} {}
  template<typename A> ArbitraryExpression(A &&x) : u{std::move(x)} {}

  std::variant<IntegerExpression, RealExpression, ComplexExpression,
      LogicalExpr, CharacterExpression>
      u;
};

// Convenience functions and operator overloadings for expression construction.
template<typename A> Expression<A> Parentheses(Expression<A> &&x) {
  return {typename Expression<A>::Parentheses{{Opd<A>(std::move(x))}}};
}
template<typename A> Expression<A> operator-(Expression<A> &&x) {
  return {typename Expression<A>::Negate{{Opd<A>(std::move(x))}}};
}
template<typename A>
Expression<A> operator+(Expression<A> &&x, Expression<A> &&y) {
  return {typename Expression<A>::Add{
      {Opd<A>(std::move(x)), Opd<A>(std::move(y))}}};
}
template<typename A>
Expression<A> operator-(Expression<A> &&x, Expression<A> &&y) {
  return {typename Expression<A>::Subtract{
      {Opd<A>(std::move(x)), Opd<A>(std::move(y))}}};
}
template<typename A>
Expression<A> operator*(Expression<A> &&x, Expression<A> &&y) {
  return {typename Expression<A>::Multiply{
      {Opd<A>(std::move(x)), Opd<A>(std::move(y))}}};
}
template<typename A>
Expression<A> operator/(Expression<A> &&x, Expression<A> &&y) {
  return {typename Expression<A>::Divide{
      {Opd<A>(std::move(x)), Opd<A>(std::move(y))}}};
}
template<typename A> Expression<A> Power(Expression<A> &&x, Expression<A> &&y) {
  return {typename Expression<A>::Power{
      {Opd<A>(std::move(x)), Opd<A>(std::move(y))}}};
}

template<typename A>
Comparison<A> operator<(Expression<A> &&x, Expression<A> &&y) {
  return {
      typename Comparison<A>::LT{{Opd<A>(std::move(x)), Opd<A>(std::move(y))}}};
}
template<typename A>
Comparison<A> operator<=(Expression<A> &&x, Expression<A> &&y) {
  return {
      typename Comparison<A>::LE{{Opd<A>(std::move(x)), Opd<A>(std::move(y))}}};
}
template<typename A>
Comparison<A> operator==(Expression<A> &&x, Expression<A> &&y) {
  return {
      typename Comparison<A>::EQ{{Opd<A>(std::move(x)), Opd<A>(std::move(y))}}};
}
template<typename A>
Comparison<A> operator!=(Expression<A> &&x, Expression<A> &&y) {
  return {
      typename Comparison<A>::NE{{Opd<A>(std::move(x)), Opd<A>(std::move(y))}}};
}
template<typename A>
Comparison<A> operator>=(Expression<A> &&x, Expression<A> &&y) {
  return {
      typename Comparison<A>::GE{{Opd<A>(std::move(x)), Opd<A>(std::move(y))}}};
}
template<typename A>
Comparison<A> operator>(Expression<A> &&x, Expression<A> &&y) {
  return {
      typename Comparison<A>::GT{{Opd<A>(std::move(x)), Opd<A>(std::move(y))}}};
}

// External instantiations
extern template struct Expression<Type<Category::Integer, 1>>;
extern template struct Expression<Type<Category::Integer, 2>>;
extern template struct Expression<Type<Category::Integer, 4>>;
extern template struct Expression<Type<Category::Integer, 8>>;
extern template struct Expression<Type<Category::Integer, 16>>;
extern template struct Expression<Type<Category::Real, 2>>;
extern template struct Expression<Type<Category::Real, 4>>;
extern template struct Expression<Type<Category::Real, 8>>;
extern template struct Expression<Type<Category::Real, 10>>;
extern template struct Expression<Type<Category::Real, 16>>;
extern template struct Expression<Type<Category::Complex, 2>>;
extern template struct Expression<Type<Category::Complex, 4>>;
extern template struct Expression<Type<Category::Complex, 8>>;
extern template struct Expression<Type<Category::Complex, 10>>;
extern template struct Expression<Type<Category::Complex, 16>>;
extern template struct Expression<Type<Category::Logical, 1>>;
extern template struct Expression<Type<Category::Character, 1>>;
}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_EXPRESSION_H_
