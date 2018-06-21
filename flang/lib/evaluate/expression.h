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

// Some forward definitions
template<int KIND> struct IntegerExpr;
template<int KIND> struct RealExpr;
template<int KIND> struct ComplexExpr;
template<int KIND> struct CharacterExpr;
struct AnyIntegerExpr;
struct AnyRealExpr;
struct AnyComplexExpr;
struct AnyCharacterExpr;
struct AnyIntegerOrRealExpr;

template<int KIND> struct IntegerExpr {
  using Result = Type<Category::Integer, KIND>;
  using Operand = std::unique_ptr<IntegerExpr>;
  using Constant = typename Result::Value;
  struct Convert {
    std::unique_ptr<AnyIntegerOrRealExpr> x;
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

  IntegerExpr() = delete;
  IntegerExpr(IntegerExpr &&) = default;
  IntegerExpr(const Constant &x) : u{x} {}
  IntegerExpr(std::int64_t n) : u{Constant{n}} {}
  IntegerExpr(int n) : u{Constant{n}} {}
  template<typename A> IntegerExpr(A &&x) : u{std::move(x)} {}

  std::ostream &Dump(std::ostream &) const;

  std::variant<Constant, Convert, Parentheses, Negate, Add, Subtract, Multiply,
      Divide, Power>
      u;
};

using DefaultIntegerExpr = IntegerExpr<DefaultInteger::kind>;

template<int KIND> struct RealExpr {
  using Result = Type<Category::Real, KIND>;
  using Operand = std::unique_ptr<RealExpr>;
  using Constant = typename Result::Value;
  struct Convert {
    // N.B. Real->Complex and Complex->Real conversions are done with CMPLX
    // and part access operations (resp.).  Conversions between kinds of
    // Complex are done via decomposition to Real and reconstruction.
    std::unique_ptr<AnyIntegerOrRealExpr> x;
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
  struct IntPower {
    Operand x;
    std::unique_ptr<AnyIntegerExpr> y;
  };
  struct FromComplex {
    std::unique_ptr<ComplexExpr<KIND>> z;
  };
  struct RealPart : public FromComplex {};
  struct AIMAG : public FromComplex {};

  RealExpr() = delete;
  RealExpr(RealExpr &&) = default;
  RealExpr(const Constant &x) : u{x} {}
  template<typename A> RealExpr(A &&x) : u{std::move(x)} {}

  std::ostream &Dump(std::ostream &) const;

  std::variant<Constant, Convert, Parentheses, Negate, Add, Subtract, Multiply,
      Divide, Power, IntPower, RealPart, AIMAG>
      u;
};

template<int KIND> struct ComplexExpr {
  using Result = Type<Category::Complex, KIND>;
  using Operand = std::unique_ptr<ComplexExpr>;
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
  struct IntPower {
    Operand x;
    std::unique_ptr<AnyIntegerExpr> y;
  };
  struct CMPLX {
    std::unique_ptr<RealExpr<KIND>> re, im;
  };

  ComplexExpr() = delete;
  ComplexExpr(ComplexExpr &&) = default;
  ComplexExpr(const Constant &x) : u{x} {}
  template<typename A> ComplexExpr(A &&x) : u{std::move(x)} {}

  std::ostream &Dump(std::ostream &) const;

  std::variant<Constant, Parentheses, Negate, Add, Subtract, Multiply, Divide,
      Power, IntPower, CMPLX>
      u;
};

template<int KIND> struct CharacterExpr {
  using Result = Type<Category::Character, KIND>;
  using Operand = std::unique_ptr<CharacterExpr>;
  using Constant = typename Result::Value;
  using Length = IntegerExpr<IntrinsicTypeParameterType::kind>;
  struct Concat {
    Operand x, y;
  };

  CharacterExpr() = delete;
  CharacterExpr(CharacterExpr &&) = default;
  CharacterExpr(const Constant &x) : u{x} {}
  CharacterExpr(CharacterExpr &&a, CharacterExpr &&b)
    : u{Concat{std::make_unique<CharacterExpr>(std::move(a)),
          std::make_unique<CharacterExpr>(std::move(b))}} {}

  Length LEN() const;
  std::ostream &Dump(std::ostream &) const;

  std::variant<Constant, Concat> u;
};

// The Comparison class template is a helper for constructing logical
// expressions with polymorphism over all of the possible categories and
// kinds of comparable operands.
template<typename T> struct Comparison {
  struct Binary {
    std::unique_ptr<T> x, y;
  };
  struct LT : public Binary {};
  struct LE : public Binary {};
  struct EQ : public Binary {};
  struct NE : public Binary {};
  struct GE : public Binary {};
  struct GT : public Binary {};

  Comparison() = delete;
  Comparison(Comparison &&) = default;
  template<typename A> Comparison(A &&x) : u{std::move(x)} {}

  std::ostream &Dump(std::ostream &) const;

  std::variant<LT, LE, EQ, NE, GE, GT> u;
};

// COMPLEX admits only .EQ. and .NE. comparisons.
template<int KIND> struct Comparison<ComplexExpr<KIND>> {
  struct Binary {
    std::unique_ptr<ComplexExpr<KIND>> x, y;
  };
  struct EQ : public Binary {};
  struct NE : public Binary {};

  Comparison() = delete;
  Comparison(Comparison &&) = default;
  template<typename A> Comparison(A &&x) : u{std::move(x)} {}

  std::ostream &Dump(std::ostream &) const;

  std::variant<EQ, NE> u;
};

struct IntegerComparison {
  IntegerComparison() = delete;
  IntegerComparison(IntegerComparison &&) = default;
  template<typename A> IntegerComparison(A &&x) : u{std::move(x)} {}
  std::ostream &Dump(std::ostream &) const;
  template<int KIND> using C = Comparison<IntegerExpr<KIND>>;
  IntegerKindsVariant<C> u;
};

struct RealComparison {
  RealComparison() = delete;
  RealComparison(RealComparison &&) = default;
  template<typename A> RealComparison(A &&x) : u{std::move(x)} {}
  std::ostream &Dump(std::ostream &) const;
  template<int KIND> using C = Comparison<RealExpr<KIND>>;
  RealKindsVariant<C> u;
};

struct ComplexComparison {
  ComplexComparison() = delete;
  ComplexComparison(ComplexComparison &&) = default;
  template<typename A> ComplexComparison(A &&x) : u{std::move(x)} {}
  std::ostream &Dump(std::ostream &) const;
  template<int KIND> using C = Comparison<ComplexExpr<KIND>>;
  ComplexKindsVariant<C> u;
};

struct CharacterComparison {
  CharacterComparison() = delete;
  CharacterComparison(CharacterComparison &&) = default;
  template<typename A> CharacterComparison(A &&x) : u{std::move(x)} {}
  std::ostream &Dump(std::ostream &) const;
  template<int KIND> using C = Comparison<CharacterExpr<KIND>>;
  CharacterKindsVariant<C> u;
};

// No need to distinguish the various kinds of LOGICAL expression results.
struct LogicalExpr {
  using Operand = std::unique_ptr<LogicalExpr>;
  using Constant = bool;
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

  LogicalExpr() = delete;
  LogicalExpr(LogicalExpr &&) = default;
  LogicalExpr(const Constant &x) : u{x} {}
  template<int KIND>
  LogicalExpr(Comparison<IntegerExpr<KIND>> &&x)
    : u{IntegerComparison{std::move(x)}} {}
  template<int KIND>
  LogicalExpr(Comparison<RealExpr<KIND>> &&x)
    : u{RealComparison{std::move(x)}} {}
  template<int KIND>
  LogicalExpr(Comparison<ComplexExpr<KIND>> &&x)
    : u{ComplexComparison{std::move(x)}} {}
  template<int KIND>
  LogicalExpr(Comparison<CharacterExpr<KIND>> &&x)
    : u{CharacterComparison{std::move(x)}} {}
  template<typename A> LogicalExpr(A &&x) : u{std::move(x)} {}

  std::ostream &Dump(std::ostream &) const;

  std::variant<Constant, Not, And, Or, Eqv, Neqv, IntegerComparison,
      RealComparison, ComplexComparison, CharacterComparison>
      u;
};

// Dynamically polymorphic expressions that can hold any supported kind.
struct AnyIntegerExpr {
  AnyIntegerExpr() = delete;
  AnyIntegerExpr(AnyIntegerExpr &&) = default;
  template<int KIND> AnyIntegerExpr(IntegerExpr<KIND> &&x) : u{x} {}
  std::ostream &Dump(std::ostream &) const;
  IntegerKindsVariant<IntegerExpr> u;
};
struct AnyRealExpr {
  AnyRealExpr() = delete;
  AnyRealExpr(AnyRealExpr &&) = default;
  template<int KIND> AnyRealExpr(RealExpr<KIND> &&x) : u{x} {}
  std::ostream &Dump(std::ostream &) const;
  RealKindsVariant<RealExpr> u;
};
struct AnyComplexExpr {
  AnyComplexExpr() = delete;
  AnyComplexExpr(AnyComplexExpr &&) = default;
  template<int KIND> AnyComplexExpr(ComplexExpr<KIND> &&x) : u{x} {}
  std::ostream &Dump(std::ostream &) const;
  ComplexKindsVariant<ComplexExpr> u;
};
struct AnyCharacterExpr {
  AnyCharacterExpr() = delete;
  AnyCharacterExpr(AnyCharacterExpr &&) = default;
  template<int KIND> AnyCharacterExpr(CharacterExpr<KIND> &&x) : u{x} {}
  std::ostream &Dump(std::ostream &) const;
  CharacterKindsVariant<CharacterExpr> u;
};

struct AnyIntegerOrRealExpr {
  AnyIntegerOrRealExpr() = delete;
  AnyIntegerOrRealExpr(AnyIntegerOrRealExpr &&) = default;
  template<int KIND>
  AnyIntegerOrRealExpr(IntegerExpr<KIND> &&x)
    : u{AnyIntegerExpr{std::move(x)}} {}
  template<int KIND>
  AnyIntegerOrRealExpr(RealExpr<KIND> &&x) : u{AnyRealExpr{std::move(x)}} {}
  template<typename A> AnyIntegerOrRealExpr(A &&x) : u{std::move(x)} {}
  std::ostream &Dump(std::ostream &) const;
  std::variant<AnyIntegerExpr, AnyRealExpr> u;
};

// Convenience functions and operator overloadings for expression construction.
template<typename A> A Parentheses(A &&x) {
  return {typename A::Parentheses{{std::make_unique<A>(std::move(x))}}};
}
template<typename A> A operator-(A &&x) {
  return {typename A::Negate{{std::make_unique<A>(std::move(x))}}};
}
template<typename A> A operator+(A &&x, A &&y) {
  return {typename A::Add{
      {std::make_unique<A>(std::move(x)), std::make_unique<A>(std::move(y))}}};
}
template<typename A> A operator-(A &&x, A &&y) {
  return {typename A::Subtract{
      {std::make_unique<A>(std::move(x)), std::make_unique<A>(std::move(y))}}};
}
template<typename A> A operator*(A &&x, A &&y) {
  return {typename A::Multiply{
      {std::make_unique<A>(std::move(x)), std::make_unique<A>(std::move(y))}}};
}
template<typename A> A operator/(A &&x, A &&y) {
  return {typename A::Divide{
      {std::make_unique<A>(std::move(x)), std::make_unique<A>(std::move(y))}}};
}
template<typename A> A Power(A &&x, A &&y) {
  return {typename A::Power{
      {std::make_unique<A>(std::move(x)), std::make_unique<A>(std::move(y))}}};
}

template<typename A> Comparison<A> operator<(A &&x, A &&y) {
  return {typename Comparison<A>::LT{
      {std::make_unique<A>(std::move(x)), std::make_unique<A>(std::move(y))}}};
}
template<typename A> Comparison<A> operator<=(A &&x, A &&y) {
  return {typename Comparison<A>::LE{
      {std::make_unique<A>(std::move(x)), std::make_unique<A>(std::move(y))}}};
}
template<typename A> Comparison<A> operator==(A &&x, A &&y) {
  return {typename Comparison<A>::EQ{
      {std::make_unique<A>(std::move(x)), std::make_unique<A>(std::move(y))}}};
}
template<typename A> Comparison<A> operator!=(A &&x, A &&y) {
  return {typename Comparison<A>::NE{
      {std::make_unique<A>(std::move(x)), std::make_unique<A>(std::move(y))}}};
}
template<typename A> Comparison<A> operator>=(A &&x, A &&y) {
  return {typename Comparison<A>::GE{
      {std::make_unique<A>(std::move(x)), std::make_unique<A>(std::move(y))}}};
}
template<typename A> Comparison<A> operator>(A &&x, A &&y) {
  return {typename Comparison<A>::GT{
      {std::make_unique<A>(std::move(x)), std::make_unique<A>(std::move(y))}}};
}

// External instantiations
extern template struct IntegerExpr<1>;
extern template struct IntegerExpr<2>;
extern template struct IntegerExpr<4>;
extern template struct IntegerExpr<8>;
extern template struct IntegerExpr<16>;
extern template struct RealExpr<2>;
extern template struct RealExpr<4>;
extern template struct RealExpr<8>;
extern template struct RealExpr<10>;
extern template struct RealExpr<16>;
extern template struct ComplexExpr<2>;
extern template struct ComplexExpr<4>;
extern template struct ComplexExpr<8>;
extern template struct ComplexExpr<10>;
extern template struct ComplexExpr<16>;
extern template struct CharacterExpr<1>;
}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_EXPRESSION_H_
