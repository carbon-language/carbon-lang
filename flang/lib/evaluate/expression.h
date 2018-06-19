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
#include "../common/indirection.h"
#include <variant>

namespace Fortran::evaluate {

template<typename T> struct Expression;

template<typename T> struct ExprOperand {
  template<typename... ARGS> ExprOperand(const ARGS &... args) : v{args...} {}
  template<typename... ARGS>
  ExprOperand(ARGS &&... args) : v{std::forward<ARGS>(args)...} {}
  common::Indirection<Expression<T>> v;
};

struct IntegerOperand {
  std::variant<ExprOperand<Type<Category::Integer, 1>>,
      ExprOperand<Type<Category::Integer, 2>>,
      ExprOperand<Type<Category::Integer, 4>>,
      ExprOperand<Type<Category::Integer, 8>>,
      ExprOperand<Type<Category::Integer, 16>>>
      u;
};
struct RealOperand {
  std::variant<ExprOperand<Type<Category::Real, 2>>,
      ExprOperand<Type<Category::Real, 4>>,
      ExprOperand<Type<Category::Real, 8>>,
      ExprOperand<Type<Category::Real, 10>>,
      ExprOperand<Type<Category::Real, 16>>>
      u;
};
struct ComplexOperand {
  std::variant<ExprOperand<Type<Category::Complex, 2>>,
      ExprOperand<Type<Category::Complex, 4>>,
      ExprOperand<Type<Category::Complex, 8>>,
      ExprOperand<Type<Category::Complex, 10>>,
      ExprOperand<Type<Category::Complex, 16>>>
      u;
};
struct CharacterOperand {
  std::variant<ExprOperand<Type<Category::Character, 1>>> u;
};

struct FloatingOperand {
  std::variant<RealOperand, ComplexOperand> u;
};
struct NumericOperand {
  std::variant<IntegerOperand, FloatingOperand> u;
};

template<Category C, int KIND> struct NumericBase {
  static constexpr Category category{C};
  static constexpr int kind{KIND};
  using Result = Type<category, kind>;
  using Operand = ExprOperand<Result>;
  using Constant = typename Result::Value;
  struct Unary {
    enum class Operator { Parentheses, Negate } op;
    Operand x;
  };
  struct Binary {
    enum class Operator { Add, Subtract, Multiply, Divide, Power } op;
    Operand x, y;
  };
  struct Convert {
    NumericOperand x;
  };
};

template<int KIND>
struct Expression<Type<Category::Integer, KIND>>
  : public NumericBase<Category::Integer, KIND> {
  using Base = NumericBase<Category::Integer, KIND>;
  using Result = typename Base::Result;
  using Constant = typename Base::Constant;
  using Convert = typename Base::Convert;
  using Unary = typename Base::Unary;
  using Binary = typename Base::Binary;
  Expression() = delete;
  Expression(Expression &&) = default;
  Expression(const Constant &x) : u{x} {}
  Expression(Convert &&x) : u{std::move(x)} {}
  Expression(typename Unary::Operator o, Expression &&a)
    : u{Unary{o, std::move(a)}} {}
  Expression(typename Binary::Operator o, Expression &&a, Expression &&b)
    : u{Binary{o, std::move(a), std::move(b)}} {}
  std::variant<Constant, Convert, Unary, Binary> u;
};

template<Category C, int KIND>
struct FloatingBase : public NumericBase<C, KIND> {
  using Result = typename NumericBase<C, KIND>::Result;
  struct IntegerPower {
    Result x;
    IntegerOperand y;
  };
};

template<int KIND>
struct Expression<Type<Category::Real, KIND>>
  : public FloatingBase<Category::Real, KIND> {
  using Base = FloatingBase<Category::Real, KIND>;
  using Result = typename Base::Result;
  using Constant = typename Base::Constant;
  using Convert = typename Base::Convert;
  using Unary = typename Base::Unary;
  using Binary = typename Base::Binary;
  using IntegerPower = typename Base::IntegerPower;
  Expression() = delete;
  Expression(Expression &&) = default;
  Expression(const Constant &x) : u{x} {}
  Expression(Convert &&x) : u{std::move(x)} {}
  Expression(typename Unary::Operator o, Expression &&a)
    : u{Unary{o, std::move(a)}} {}
  Expression(typename Binary::Operator o, Expression &&a, Expression &&b)
    : u{Binary{o, std::move(a), std::move(b)}} {}
  std::variant<Constant, Convert, Unary, Binary, IntegerPower> u;
};

template<int KIND>
struct Expression<Type<Category::Complex, KIND>>
  : public FloatingBase<Category::Complex, KIND> {
  using Base = FloatingBase<Category::Complex, KIND>;
  using Result = typename Base::Result;
  using Constant = typename Base::Constant;
  using Convert = typename Base::Convert;
  using Unary = typename Base::Unary;
  using Binary = typename Base::Binary;
  using IntegerPower = typename Base::IntegerPower;
  Expression() = delete;
  Expression(Expression &&) = default;
  Expression(const Constant &x) : u{x} {}
  Expression(Convert &&x) : u{std::move(x)} {}
  Expression(typename Unary::Operator o, Expression &&a)
    : u{Unary{o, std::move(a)}} {}
  Expression(typename Binary::Operator o, Expression &&a, Expression &&b)
    : u{Binary{o, std::move(a), std::move(b)}} {}
  std::variant<Constant, Convert, Unary, Binary, IntegerPower> u;
};

template<> struct Expression<Type<Category::Logical, 1>> {
  // No need to distinguish the various kinds of LOGICAL in expressions.
  static constexpr Category category{Category::Logical};
  static constexpr int kind{1};
  using Result = Type<category, kind>;
  using Operand = ExprOperand<Result>;
  using Constant = typename Result::Value;

  struct Unary {
    enum class Operator { Not } op;
    Operand x;
  };
  struct Binary {
    enum class Operator { And, Or, Eqv, Neqv } op;
    Operand x, y;
  };

  enum class ComparisonOperator { LT, LE, EQ, NE, GE, GT };  // TODO: .UN.?
  template<typename T> struct Comparison {
    ComparisonOperator op;
    ExprOperand<T> x, y;
  };

  enum class EqualityOperator { EQ, NE };
  template<int KIND> struct ComplexComparison {
    EqualityOperator op;
    ExprOperand<Type<Category::Complex, KIND>> x, y;
  };

  Expression() = delete;
  Expression(Expression &&) = default;
  Expression(const Constant &x) : u{x} {}
  Expression(typename Unary::Operator o, Expression &&a)
    : u{Unary{o, std::move(a)}} {}
  Expression(typename Binary::Operator o, Expression &&a, Expression &&b)
    : u{Binary{o, std::move(a), std::move(b)}} {}
  template<typename T>
  Expression(ComparisonOperator o, Expression<T> &&a, Expression<T> &&b)
    : u{Comparison<T>{o, std::move(a), std::move(b)}} {}
  template<int KIND>
  Expression(EqualityOperator o, Expression<Type<Category::Complex, KIND>> &&a,
      Expression<Type<Category::Complex, KIND>> &&b)
    : u{ComplexComparison<KIND>{o, std::move(a), std::move(b)}} {}
  std::variant<Constant, Unary, Binary, Comparison<Type<Category::Integer, 1>>,
      Comparison<Type<Category::Integer, 2>>,
      Comparison<Type<Category::Integer, 4>>,
      Comparison<Type<Category::Integer, 8>>,
      Comparison<Type<Category::Integer, 16>>,
      Comparison<Type<Category::Character, 1>>,
      Comparison<Type<Category::Real, 2>>, Comparison<Type<Category::Real, 4>>,
      Comparison<Type<Category::Real, 8>>, Comparison<Type<Category::Real, 10>>,
      Comparison<Type<Category::Real, 16>>, ComplexComparison<2>,
      ComplexComparison<4>, ComplexComparison<8>, ComplexComparison<10>,
      ComplexComparison<16>>
      u;
};

template<int KIND> struct Expression<Type<Category::Character, KIND>> {
  static constexpr Category category{Category::Character};
  static constexpr int kind{KIND};
  using Result = Type<category, kind>;
  using Constant = typename Result::Value;
  struct Concat {
    ExprOperand<Result> x, y;
  };
  Expression() = delete;
  Expression(Expression &&) = default;
  Expression(const Constant &x) : u{x} {}
  Expression(Expression &&a, Expression &&b)
    : u{Concat{std::move(a), std::move(b)}} {}
  std::variant<Constant, Concat> u;
  // TODO: length
};
}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_EXPRESSION_H_
