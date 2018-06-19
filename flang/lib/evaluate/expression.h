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

template<template<int> class T> using IntegerKindsVariant =
  std::variant<T<1>, T<2>, T<4>, T<8>, T<16>>;
template<template<int> class T> using RealKindsVariant =
  std::variant<T<2>, T<4>, T<8>, T<10>, T<16>>;
template<template<int> class T> using CharacterKindsVariant =
  std::variant<T<1>>;  // TODO larger CHARACTER kinds, incl. Kanji

struct IntegerOperand {
  template<int KIND> using Operand = ExprOperand<Type<Category::Integer, KIND>>;
  IntegerKindsVariant<Operand> u;
};
struct RealOperand {
  template<int KIND> using Operand = ExprOperand<Type<Category::Real, KIND>>;
  RealKindsVariant<Operand> u;
};
struct CharacterOperand {
  template<int KIND> using Operand = ExprOperand<Type<Category::Character, KIND>>;
  CharacterKindsVariant<Operand> u;
};

template<Category C, int KIND> struct NumericBase {
  static constexpr Category category{C};
  static constexpr int kind{KIND};
  using Result = Type<category, kind>;
  using Operand = ExprOperand<Result>;
  using Constant = typename Result::Value;
  struct Convert {
    // N.B. Conversions to/from Complex are done with CMPLX and part access
    // operations (resp.).  Conversions between kinds of Complex are done
    // via decomposition and reconstruction.
    std::variant<IntegerOperand, RealOperand> u;
  };
  struct Parentheses { Operand x; };
  struct Negate { Operand x; };
  struct Add { Operand x, y; };
  struct Subtract { Operand x, y; };
  struct Multiply { Operand x, y; };
  struct Divide { Operand x, y; };
  struct Power { Operand x, y; };
};

template<int KIND>
struct Expression<Type<Category::Integer, KIND>>
  : public NumericBase<Category::Integer, KIND> {
  using Base = NumericBase<Category::Integer, KIND>;
  using Result = typename Base::Result;
  using Convert = typename Base::Convert;
  using Constant = typename Base::Constant;
  using Parentheses = typename Base::Parentheses;
  using Negate = typename Base::Negate;
  using Add = typename Base::Add;
  using Subtract = typename Base::Subtract;
  using Multiply = typename Base::Multiply;
  using Divide = typename Base::Divide;
  using Power = typename Base::Power;
  Expression() = delete;
  Expression(Expression &&) = default;
  Expression(const typename Base::Constant &x) : u{x} {}
  template<typename A> Expression(A &&x) : u{std::move(x)} {}
  std::variant<Constant, Convert, Parentheses, Negate, Add, Subtract,
               Multiply, Divide, Power> u;
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
  using Parentheses = typename Base::Parentheses;
  using Negate = typename Base::Negate;
  using Add = typename Base::Add;
  using Subtract = typename Base::Subtract;
  using Multiply = typename Base::Multiply;
  using Divide = typename Base::Divide;
  using Power = typename Base::Power;
  using IntegerPower = typename Base::IntegerPower;
  struct RealPart { ExprOperand<typename Result::Complex> x; };
  struct AIMAG { ExprOperand<typename Result::Complex> x; };
  Expression() = delete;
  Expression(Expression &&) = default;
  Expression(const Constant &x) : u{x} {}
  template<typename A> Expression(A &&x) : u{std::move(x)} {}
  std::variant<Constant, Convert, Parentheses, Negate, Add, Subtract,
               Multiply, Divide, Power, IntegerPower,
               RealPart, AIMAG> u;
};

template<int KIND>
struct Expression<Type<Category::Complex, KIND>>
  : public FloatingBase<Category::Complex, KIND> {
  using Base = FloatingBase<Category::Complex, KIND>;
  using Result = typename Base::Result;
  using Constant = typename Base::Constant;
  using Parentheses = typename Base::Parentheses;
  using Negate = typename Base::Negate;
  using Add = typename Base::Add;
  using Subtract = typename Base::Subtract;
  using Multiply = typename Base::Multiply;
  using Divide = typename Base::Divide;
  using Power = typename Base::Power;
  using IntegerPower = typename Base::IntegerPower;
  struct CMPLX { ExprOperand<typename Result::Part> re, im; };
  Expression() = delete;
  Expression(Expression &&) = default;
  Expression(const Constant &x) : u{x} {}
  template<typename A> Expression(A &&x) : u{std::move(x)} {}
  std::variant<Constant, Parentheses, Negate, Add, Subtract,
               Multiply, Divide, Power, IntegerPower, CMPLX> u;
};

template<> struct Expression<Type<Category::Logical, 1>> {
  // No need to distinguish the various kinds of LOGICAL in expressions.
  static constexpr Category category{Category::Logical};
  static constexpr int kind{1};
  using Result = Type<category, kind>;
  using Operand = ExprOperand<Result>;
  using Constant = typename Result::Value;
  struct Not { Operand x; };
  struct And { Operand x, y; };
  struct Or { Operand x, y; };
  struct Eqv { Operand x, y; };
  struct Neqv { Operand x, y; };

  template<typename T> struct Comparison {
    using Operand = ExprOperand<T>;
    struct LT { Operand x, y; };
    struct LE { Operand x, y; };
    struct EQ { Operand x, y; };
    struct NE { Operand x, y; };
    struct GE { Operand x, y; };
    struct GT { Operand x, y; };
    std::variant<LT, LE, EQ, NE, GE, GT> u;  // TODO: .UN.?
  };
  template<int KIND> struct Comparison<Type<Category::Complex, KIND>> {
    using Operand = ExprOperand<Type<Category::Complex, KIND>>;
    struct EQ { Operand x, y; };
    struct NE { Operand x, y; };
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

  Expression() = delete;
  Expression(Expression &&) = default;
  Expression(const Constant &x) : u{x} {}
  template<typename A> Expression(A &&x) : u{std::move(x)} {}

  template<typename T> static Comparison<T> LT(Expression<T> &&x, Expression<T> &&y) {
    return {typename Comparison<T>::LT{std::move(x), std::move(y)}};
  }
  template<typename T> static Comparison<T> LE(Expression<T> &&x, Expression<T> &&y) {
    return {typename Comparison<T>::LE{std::move(x), std::move(y)}};
  }
  template<typename T> static Comparison<T> EQ(Expression<T> &&x, Expression<T> &&y) {
    return {typename Comparison<T>::EQ{std::move(x), std::move(y)}};
  }
  template<typename T> static Comparison<T> NE(Expression<T> &&x, Expression<T> &&y) {
    return {typename Comparison<T>::NE{std::move(x), std::move(y)}};
  }
  template<typename T> static Comparison<T> GE(Expression<T> &&x, Expression<T> &&y) {
    return {typename Comparison<T>::GE{std::move(x), std::move(y)}};
  }
  template<typename T> static Comparison<T> GT(Expression<T> &&x, Expression<T> &&y) {
    return {typename Comparison<T>::GT{std::move(x), std::move(y)}};
  }

  std::variant<Constant, Not, And, Or, Eqv, Neqv,
      Comparison<Type<Category::Integer, 1>>,
      Comparison<Type<Category::Integer, 2>>,
      Comparison<Type<Category::Integer, 4>>,
      Comparison<Type<Category::Integer, 8>>,
      Comparison<Type<Category::Integer, 16>>,
      Comparison<Type<Category::Character, 1>>,
      Comparison<Type<Category::Real, 2>>,
      Comparison<Type<Category::Real, 4>>,
      Comparison<Type<Category::Real, 8>>,
      Comparison<Type<Category::Real, 10>>,
      Comparison<Type<Category::Real, 16>>,
      Comparison<Type<Category::Complex, 2>>,
      Comparison<Type<Category::Complex, 4>>,
      Comparison<Type<Category::Complex, 8>>,
      Comparison<Type<Category::Complex, 10>>,
      Comparison<Type<Category::Complex, 16>>> u;
};

template<int KIND> struct Expression<Type<Category::Character, KIND>> {
  static constexpr Category category{Category::Character};
  static constexpr int kind{KIND};
  using Result = Type<category, kind>;
  using Constant = typename Result::Value;
  struct Concat { ExprOperand<Result> x, y; };
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
