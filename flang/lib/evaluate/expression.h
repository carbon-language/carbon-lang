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
#include <ostream>
#include <variant>

namespace Fortran::evaluate {

// First, a statically polymorphic representation of expressions that's
// specialized across the type categories of Fortran.  Subexpression
// operands are implemented with non-nullable owning pointers.
// Every Expression specialization has (at least) these type aliases:
//   using Result = Type<category, kind>
//   using Operand = ExprOperand<Result>
//   using Constant = typename Result::Value  // e.g., value::Integer<BITS>
// nested declarations of wrapper structs for each operation, e.g.
//   struct Add { Operand x, y; };
// a data member to hold an instance of one of these structs:
//   std::variant<> u;
// and a formatting member function, dump().
template<typename T> struct Expression;

template<typename T> struct ExprOperand {
  template<typename... ARGS> ExprOperand(const ARGS &... args) : v{args...} {}
  template<typename... ARGS>
  ExprOperand(ARGS &&... args) : v{std::forward<ARGS>(args)...} {}
  Expression<T> &operator*() { return *v; }
  const Expression<T> &operator*() const { return *v; }
  Expression<T> *operator->() { return &*v; }
  const Expression<T> *operator->() const { return &*v; }
  common::Indirection<Expression<T>> v;
};

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

// Shared by Integer, Real, and Complex
template<Category C, int KIND> struct NumericBase {
  static constexpr Category category{C};
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
  struct Parentheses {
    Operand x;
  };
  struct Negate {
    Operand x;
  };
  struct Add {
    Operand x, y;
  };
  struct Subtract {
    Operand x, y;
  };
  struct Multiply {
    Operand x, y;
  };
  struct Divide {
    Operand x, y;
  };
  struct Power {
    Operand x, y;
  };

  void dump(std::ostream &) const;
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
  std::variant<Constant, Convert, Parentheses, Negate, Add, Subtract, Multiply,
      Divide, Power>
      u;
};

// Shared by Real and Complex, which need to allow and distinguish
// exponentiation by integer powers.
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
  struct RealPart {
    ExprOperand<typename Result::Complex> x;
  };
  struct AIMAG {
    ExprOperand<typename Result::Complex> x;
  };
  Expression() = delete;
  Expression(Expression &&) = default;
  Expression(const Constant &x) : u{x} {}
  template<typename A> Expression(A &&x) : u{std::move(x)} {}
  std::variant<Constant, Convert, Parentheses, Negate, Add, Subtract, Multiply,
      Divide, Power, IntegerPower, RealPart, AIMAG>
      u;
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
  struct CMPLX {
    ExprOperand<typename Result::Part> re, im;
  };
  Expression() = delete;
  Expression(Expression &&) = default;
  Expression(const Constant &x) : u{x} {}
  template<typename A> Expression(A &&x) : u{std::move(x)} {}
  std::variant<Constant, Parentheses, Negate, Add, Subtract, Multiply, Divide,
      Power, IntegerPower, CMPLX>
      u;
};

template<> struct Expression<Type<Category::Logical, 1>> {
  // No need to distinguish the various kinds of LOGICAL in expressions,
  // only in memory.
  static constexpr Category category{Category::Logical};
  static constexpr int kind{1};
  using Result = Type<category, kind>;
  using Operand = ExprOperand<Result>;
  using Constant = typename Result::Value;
  struct Not {
    Operand x;
  };
  struct And {
    Operand x, y;
  };
  struct Or {
    Operand x, y;
  };
  struct Eqv {
    Operand x, y;
  };
  struct Neqv {
    Operand x, y;
  };

  template<typename T> struct Comparison {
    using Operand = ExprOperand<T>;
    struct LT {
      Operand x, y;
    };
    struct LE {
      Operand x, y;
    };
    struct EQ {
      Operand x, y;
    };
    struct NE {
      Operand x, y;
    };
    struct GE {
      Operand x, y;
    };
    struct GT {
      Operand x, y;
    };
    std::variant<LT, LE, EQ, NE, GE, GT> u;  // TODO: .UN. extension?
  };
  template<int KIND> struct Comparison<Type<Category::Complex, KIND>> {
    using Operand = ExprOperand<Type<Category::Complex, KIND>>;
    struct EQ {
      Operand x, y;
    };
    struct NE {
      Operand x, y;
    };
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

  template<typename T>
  static Comparison<T> LT(Expression<T> &&x, Expression<T> &&y) {
    return {typename Comparison<T>::LT{std::move(x), std::move(y)}};
  }
  template<typename T>
  static Comparison<T> LE(Expression<T> &&x, Expression<T> &&y) {
    return {typename Comparison<T>::LE{std::move(x), std::move(y)}};
  }
  template<typename T>
  static Comparison<T> EQ(Expression<T> &&x, Expression<T> &&y) {
    return {typename Comparison<T>::EQ{std::move(x), std::move(y)}};
  }
  template<typename T>
  static Comparison<T> NE(Expression<T> &&x, Expression<T> &&y) {
    return {typename Comparison<T>::NE{std::move(x), std::move(y)}};
  }
  template<typename T>
  static Comparison<T> GE(Expression<T> &&x, Expression<T> &&y) {
    return {typename Comparison<T>::GE{std::move(x), std::move(y)}};
  }
  template<typename T>
  static Comparison<T> GT(Expression<T> &&x, Expression<T> &&y) {
    return {typename Comparison<T>::GT{std::move(x), std::move(y)}};
  }

  std::variant<Constant, Not, And, Or, Eqv, Neqv, IntegerComparison,
      RealComparison, ComplexComparison, CharacterComparison>
      u;
};

template<int KIND> struct Expression<Type<Category::Character, KIND>> {
  static constexpr Category category{Category::Character};
  static constexpr int kind{KIND};
  using Result = Type<category, kind>;
  using Constant = typename Result::Value;
  using Length = Expression<IntrinsicTypeParameterType>;
  struct Concat {
    ExprOperand<Result> x, y;
  };
  Expression() = delete;
  Expression(Expression &&) = default;
  Expression(const Constant &x) : u{x} {}
  Expression(Expression &&a, Expression &&b)
    : u{Concat{std::move(a), std::move(b)}} {}
  Length LEN() const;
  std::variant<Constant, Concat> u;
};

// Convenience type aliases
template<int KIND> using IntExpr = Expression<Type<Category::Integer, KIND>>;
template<int KIND> using RealExpr = Expression<Type<Category::Real, KIND>>;
template<int KIND>
using ComplexExpr = Expression<Type<Category::Complex, KIND>>;
using LogicalExpr = Expression<Type<Category::Logical, 1>>;
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

// Convenience operator overloadings for expression construction.
template<typename A> Expression<A> operator-(Expression<A> &&x) {
  return {typename Expression<A>::Negate{std::move(x)}};
}
template<typename A>
Expression<A> operator+(Expression<A> &&x, Expression<A> &&y) {
  return {typename Expression<A>::Add{std::move(x), std::move(y)}};
}
template<typename A>
Expression<A> operator-(Expression<A> &&x, Expression<A> &&y) {
  return {typename Expression<A>::Subtract{std::move(x), std::move(y)}};
}
template<typename A>
Expression<A> operator*(Expression<A> &&x, Expression<A> &&y) {
  return {typename Expression<A>::Multiply{std::move(x), std::move(y)}};
}
template<typename A>
Expression<A> operator/(Expression<A> &&x, Expression<A> &&y) {
  return {typename Expression<A>::Divide{std::move(x), std::move(y)}};
}

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
