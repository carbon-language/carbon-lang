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

// Represent Fortran expressions in a type-safe manner.
// Expressions are the sole owners of their constituents; there is no
// context-independent hash table or sharing of common subexpressions.
// Both deep copy and move semantics are supported for expression construction.
// TODO: variable and function references

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

// Helper classes to manage subexpressions.
template<typename A> struct Unary {
  Unary(const A &a) : x{std::make_unique<A>(a)} {}
  Unary(std::unique_ptr<const A> &&a) : x{std::move(a)} {}
  Unary(A &&a) : x{std::make_unique<A>(std::move(a))} {}
  Unary(const Unary &that) : x{std::make_unique<A>(A{*that.x})} {}
  Unary(Unary &&) = default;
  Unary &operator=(const Unary &that) {
    *x = *that.x;
    return *this;
  }
  Unary &operator=(Unary &&) = default;
  std::unique_ptr<const A> x;
};

template<typename A, typename B> struct Binary {
  Binary(const A &a, const B &b)
    : x{std::make_unique<A>(a)}, y{std::make_unique<B>(b)} {}
  Binary(std::unique_ptr<const A> &&a, std::unique_ptr<const B> &&b)
    : x{std::move(a)}, y{std::move(b)} {}
  Binary(A &&a, B &&b)
    : x{std::make_unique<A>(std::move(a))}, y{std::make_unique<B>(
                                                std::move(b))} {}
  Binary(const Binary &that)
    : x{std::make_unique<A>(A{*that.x})}, y{std::make_unique<B>(B{*that.y})} {}
  Binary(Binary &&) = default;
  Binary &operator=(const Binary &that) {
    *x = *that.x;
    *y = *that.y;
    return *this;
  }
  Binary &operator=(Binary &&) = default;
  std::unique_ptr<const A> x;
  std::unique_ptr<const B> y;
};

template<int KIND> struct IntegerExpr {
  using Result = Type<Category::Integer, KIND>;
  using Constant = typename Result::Value;
  struct Convert : Unary<AnyIntegerOrRealExpr> {
    using Unary<AnyIntegerOrRealExpr>::Unary;
  };
  using Un = Unary<IntegerExpr>;
  using Bin = Binary<IntegerExpr, IntegerExpr>;
  struct Parentheses : public Un {
    using Un::Un;
  };
  struct Negate : public Un {
    using Un::Un;
  };
  struct Add : public Bin {
    using Bin::Bin;
  };
  struct Subtract : public Bin {
    using Bin::Bin;
  };
  struct Multiply : public Bin {
    using Bin::Bin;
  };
  struct Divide : public Bin {
    using Bin::Bin;
  };
  struct Power : public Bin {
    using Bin::Bin;
  };

  IntegerExpr() = delete;
  IntegerExpr(const IntegerExpr &) = default;
  IntegerExpr(IntegerExpr &&) = default;
  IntegerExpr(const Constant &x) : u{x} {}
  IntegerExpr(std::int64_t n) : u{Constant{n}} {}
  IntegerExpr(int n) : u{Constant{n}} {}
  template<typename A> IntegerExpr(A &&x) : u{std::move(x)} {}
  IntegerExpr &operator=(const IntegerExpr &) = default;
  IntegerExpr &operator=(IntegerExpr &&) = default;

  std::ostream &Dump(std::ostream &) const;

  std::variant<Constant, Convert, Parentheses, Negate, Add, Subtract, Multiply,
      Divide, Power>
      u;
};

using DefaultIntegerExpr = IntegerExpr<DefaultInteger::kind>;

template<int KIND> struct RealExpr {
  using Result = Type<Category::Real, KIND>;
  using Constant = typename Result::Value;
  // N.B. Real->Complex and Complex->Real conversions are done with CMPLX
  // and part access operations (resp.).  Conversions between kinds of
  // Complex are done via decomposition to Real and reconstruction.
  struct Convert : Unary<AnyIntegerOrRealExpr> {
    using Unary<AnyIntegerOrRealExpr>::Unary;
  };
  using Un = Unary<RealExpr>;
  using Bin = Binary<RealExpr, RealExpr>;
  struct Parentheses : public Un {
    using Un::Un;
  };
  struct Negate : public Un {
    using Un::Un;
  };
  struct Add : public Bin {
    using Bin::Bin;
  };
  struct Subtract : public Bin {
    using Bin::Bin;
  };
  struct Multiply : public Bin {
    using Bin::Bin;
  };
  struct Divide : public Bin {
    using Bin::Bin;
  };
  struct Power : public Bin {
    using Bin::Bin;
  };
  struct IntPower : public Binary<RealExpr, AnyIntegerExpr> {
    using Binary<RealExpr, AnyIntegerExpr>::Binary;
  };
  using CplxUn = Unary<ComplexExpr<KIND>>;
  struct RealPart : public CplxUn {
    using CplxUn::CplxUn;
  };
  struct AIMAG : public CplxUn {
    using CplxUn::CplxUn;
  };

  RealExpr() = delete;
  RealExpr(const RealExpr &) = default;
  RealExpr(RealExpr &&) = default;
  RealExpr(const Constant &x) : u{x} {}
  template<typename A> RealExpr(A &&x) : u{std::move(x)} {}
  RealExpr &operator=(const RealExpr &) = default;
  RealExpr &operator=(RealExpr &&) = default;

  std::ostream &Dump(std::ostream &) const;

  std::variant<Constant, Convert, Parentheses, Negate, Add, Subtract, Multiply,
      Divide, Power, IntPower, RealPart, AIMAG>
      u;
};

template<int KIND> struct ComplexExpr {
  using Result = Type<Category::Complex, KIND>;
  using Constant = typename Result::Value;
  using Un = Unary<ComplexExpr>;
  using Bin = Binary<ComplexExpr, ComplexExpr>;
  struct Parentheses : public Un {
    using Un::Un;
  };
  struct Negate : public Un {
    using Un::Un;
  };
  struct Add : public Bin {
    using Bin::Bin;
  };
  struct Subtract : public Bin {
    using Bin::Bin;
  };
  struct Multiply : public Bin {
    using Bin::Bin;
  };
  struct Divide : public Bin {
    using Bin::Bin;
  };
  struct Power : public Bin {
    using Bin::Bin;
  };
  struct IntPower : public Binary<ComplexExpr, AnyIntegerExpr> {
    using Binary<ComplexExpr, AnyIntegerExpr>::Binary;
  };
  struct CMPLX : public Binary<RealExpr<KIND>, RealExpr<KIND>> {
    using Binary<RealExpr<KIND>, RealExpr<KIND>>::Binary;
  };

  ComplexExpr() = delete;
  ComplexExpr(const ComplexExpr &) = default;
  ComplexExpr(ComplexExpr &&) = default;
  ComplexExpr(const Constant &x) : u{x} {}
  template<typename A> ComplexExpr(A &&x) : u{std::move(x)} {}
  ComplexExpr &operator=(const ComplexExpr &) = default;
  ComplexExpr &operator=(ComplexExpr &&) = default;

  std::ostream &Dump(std::ostream &) const;

  std::variant<Constant, Parentheses, Negate, Add, Subtract, Multiply, Divide,
      Power, IntPower, CMPLX>
      u;
};

template<int KIND> struct CharacterExpr {
  using Result = Type<Category::Character, KIND>;
  using Constant = typename Result::Value;
  using LengthExpr = IntegerExpr<IntrinsicTypeParameterType::kind>;
  struct Concat : public Binary<CharacterExpr, CharacterExpr> {
    using Binary<CharacterExpr, CharacterExpr>::Binary;
  };

  CharacterExpr() = delete;
  CharacterExpr(const CharacterExpr &) = default;
  CharacterExpr(CharacterExpr &&) = default;
  CharacterExpr(const Constant &x) : u{x} {}
  CharacterExpr(Constant &&x) : u{std::move(x)} {}
  CharacterExpr(Concat &&x) : u{std::move(x)} {}
  CharacterExpr &operator=(const CharacterExpr &) = default;
  CharacterExpr &operator=(CharacterExpr &&) = default;

  IntegerExpr<IntrinsicTypeParameterType::kind> LEN() const;
  std::ostream &Dump(std::ostream &) const;

  std::variant<Constant, Concat> u;
};

// The Comparison class template is a helper for constructing logical
// expressions with polymorphism over all of the possible categories and
// kinds of comparable operands.
template<typename T> struct Comparison {
  struct LT : public Binary<T, T> {
    using Binary<T, T>::Binary;
  };
  struct LE : public Binary<T, T> {
    using Binary<T, T>::Binary;
  };
  struct EQ : public Binary<T, T> {
    using Binary<T, T>::Binary;
  };
  struct NE : public Binary<T, T> {
    using Binary<T, T>::Binary;
  };
  struct GE : public Binary<T, T> {
    using Binary<T, T>::Binary;
  };
  struct GT : public Binary<T, T> {
    using Binary<T, T>::Binary;
  };

  Comparison() = delete;
  Comparison(const Comparison &) = default;
  Comparison(Comparison &&) = default;
  template<typename A> Comparison(A &&x) : u{std::move(x)} {}
  Comparison &operator=(const Comparison &) = default;
  Comparison &operator=(Comparison &&) = default;

  std::ostream &Dump(std::ostream &) const;

  std::variant<LT, LE, EQ, NE, GE, GT> u;
};

// COMPLEX admits only .EQ. and .NE. comparisons.
template<int KIND> struct Comparison<ComplexExpr<KIND>> {
  using Bin = Binary<ComplexExpr<KIND>, ComplexExpr<KIND>>;
  struct EQ : public Bin {
    using Bin::Bin;
  };
  struct NE : public Bin {
    using Bin::Bin;
  };

  Comparison() = delete;
  Comparison(const Comparison &) = default;
  Comparison(Comparison &&) = default;
  template<typename A> Comparison(A &&x) : u{std::move(x)} {}
  Comparison &operator=(const Comparison &) = default;
  Comparison &operator=(Comparison &&) = default;

  std::ostream &Dump(std::ostream &) const;

  std::variant<EQ, NE> u;
};

struct IntegerComparison {
  IntegerComparison() = delete;
  IntegerComparison(const IntegerComparison &) = default;
  IntegerComparison(IntegerComparison &&) = default;
  template<typename A> IntegerComparison(A &&x) : u{std::move(x)} {}
  IntegerComparison &operator=(const IntegerComparison &) = default;
  IntegerComparison &operator=(IntegerComparison &&) = default;
  std::ostream &Dump(std::ostream &) const;
  template<int KIND> using C = Comparison<IntegerExpr<KIND>>;
  IntegerKindsVariant<C> u;
};

struct RealComparison {
  RealComparison() = delete;
  RealComparison(const RealComparison &) = default;
  RealComparison(RealComparison &&) = default;
  template<typename A> RealComparison(A &&x) : u{std::move(x)} {}
  RealComparison &operator=(const RealComparison &) = default;
  RealComparison &operator=(RealComparison &&) = default;
  std::ostream &Dump(std::ostream &) const;
  template<int KIND> using C = Comparison<RealExpr<KIND>>;
  RealKindsVariant<C> u;
};

struct ComplexComparison {
  ComplexComparison() = delete;
  ComplexComparison(ComplexComparison &&) = default;
  template<typename A> ComplexComparison(A &&x) : u{std::move(x)} {}
  ComplexComparison &operator=(const ComplexComparison &) = default;
  ComplexComparison &operator=(ComplexComparison &&) = default;
  std::ostream &Dump(std::ostream &) const;
  template<int KIND> using C = Comparison<ComplexExpr<KIND>>;
  ComplexKindsVariant<C> u;
};

struct CharacterComparison {
  CharacterComparison() = delete;
  CharacterComparison(const CharacterComparison &) = default;
  CharacterComparison(CharacterComparison &&) = default;
  template<typename A> CharacterComparison(A &&x) : u{std::move(x)} {}
  CharacterComparison &operator=(const CharacterComparison &) = default;
  CharacterComparison &operator=(CharacterComparison &&) = default;
  std::ostream &Dump(std::ostream &) const;
  template<int KIND> using C = Comparison<CharacterExpr<KIND>>;
  CharacterKindsVariant<C> u;
};

// No need to distinguish the various kinds of LOGICAL expression results.
struct LogicalExpr {
  using Constant = bool;
  struct Not : Unary<LogicalExpr> {
    using Unary<LogicalExpr>::Unary;
  };
  using Bin = Binary<LogicalExpr, LogicalExpr>;
  struct And : public Bin {
    using Bin::Bin;
  };
  struct Or : public Bin {
    using Bin::Bin;
  };
  struct Eqv : public Bin {
    using Bin::Bin;
  };
  struct Neqv : public Bin {
    using Bin::Bin;
  };

  LogicalExpr() = delete;
  LogicalExpr(const LogicalExpr &) = default;
  LogicalExpr(LogicalExpr &&) = default;
  LogicalExpr(Constant x) : u{x} {}
  template<int KIND>
  LogicalExpr(const Comparison<IntegerExpr<KIND>> &x)
    : u{IntegerComparison{x}} {}
  template<int KIND>
  LogicalExpr(Comparison<IntegerExpr<KIND>> &&x)
    : u{IntegerComparison{std::move(x)}} {}
  template<int KIND>
  LogicalExpr(const Comparison<RealExpr<KIND>> &x) : u{RealComparison{x}} {}
  template<int KIND>
  LogicalExpr(Comparison<RealExpr<KIND>> &&x)
    : u{RealComparison{std::move(x)}} {}
  template<int KIND>
  LogicalExpr(const Comparison<ComplexExpr<KIND>> &x)
    : u{ComplexComparison{x}} {}
  template<int KIND>
  LogicalExpr(Comparison<ComplexExpr<KIND>> &&x)
    : u{ComplexComparison{std::move(x)}} {}
  template<int KIND>
  LogicalExpr(const Comparison<CharacterExpr<KIND>> &x)
    : u{CharacterComparison{x}} {}
  template<int KIND>
  LogicalExpr(Comparison<CharacterExpr<KIND>> &&x)
    : u{CharacterComparison{std::move(x)}} {}
  template<typename A> LogicalExpr(A &&x) : u{std::move(x)} {}
  LogicalExpr &operator=(const LogicalExpr &) = default;
  LogicalExpr &operator=(LogicalExpr &&) = default;

  std::ostream &Dump(std::ostream &) const;

  std::variant<Constant, Not, And, Or, Eqv, Neqv, IntegerComparison,
      RealComparison, ComplexComparison, CharacterComparison>
      u;
};

// Dynamically polymorphic expressions that can hold any supported kind.
struct AnyIntegerExpr {
  AnyIntegerExpr() = delete;
  AnyIntegerExpr(const AnyIntegerExpr &) = default;
  AnyIntegerExpr(AnyIntegerExpr &&) = default;
  template<int KIND> AnyIntegerExpr(const IntegerExpr<KIND> &x) : u{x} {}
  template<int KIND> AnyIntegerExpr(IntegerExpr<KIND> &&x) : u{std::move(x)} {}
  AnyIntegerExpr &operator=(const AnyIntegerExpr &) = default;
  AnyIntegerExpr &operator=(AnyIntegerExpr &&) = default;
  std::ostream &Dump(std::ostream &) const;
  IntegerKindsVariant<IntegerExpr> u;
};
struct AnyRealExpr {
  AnyRealExpr() = delete;
  AnyRealExpr(const AnyRealExpr &) = default;
  AnyRealExpr(AnyRealExpr &&) = default;
  template<int KIND> AnyRealExpr(const RealExpr<KIND> &x) : u{x} {}
  template<int KIND> AnyRealExpr(RealExpr<KIND> &&x) : u{std::move(x)} {}
  AnyRealExpr &operator=(const AnyRealExpr &) = default;
  AnyRealExpr &operator=(AnyRealExpr &&) = default;
  std::ostream &Dump(std::ostream &) const;
  RealKindsVariant<RealExpr> u;
};
struct AnyComplexExpr {
  AnyComplexExpr() = delete;
  AnyComplexExpr(const AnyComplexExpr &) = default;
  AnyComplexExpr(AnyComplexExpr &&) = default;
  template<int KIND> AnyComplexExpr(const ComplexExpr<KIND> &x) : u{x} {}
  template<int KIND> AnyComplexExpr(ComplexExpr<KIND> &&x) : u{std::move(x)} {}
  AnyComplexExpr &operator=(const AnyComplexExpr &) = default;
  AnyComplexExpr &operator=(AnyComplexExpr &&) = default;
  std::ostream &Dump(std::ostream &) const;
  ComplexKindsVariant<ComplexExpr> u;
};
struct AnyCharacterExpr {
  AnyCharacterExpr() = delete;
  AnyCharacterExpr(const AnyCharacterExpr &) = default;
  AnyCharacterExpr(AnyCharacterExpr &&) = default;
  template<int KIND> AnyCharacterExpr(const CharacterExpr<KIND> &x) : u{x} {}
  template<int KIND>
  AnyCharacterExpr(CharacterExpr<KIND> &&x) : u{std::move(x)} {}
  AnyCharacterExpr &operator=(const AnyCharacterExpr &) = default;
  AnyCharacterExpr &operator=(AnyCharacterExpr &&) = default;
  std::ostream &Dump(std::ostream &) const;
  CharacterKindsVariant<CharacterExpr> u;
};

struct AnyIntegerOrRealExpr {
  AnyIntegerOrRealExpr() = delete;
  AnyIntegerOrRealExpr(const AnyIntegerOrRealExpr &) = default;
  AnyIntegerOrRealExpr(AnyIntegerOrRealExpr &&) = default;
  template<int KIND>
  AnyIntegerOrRealExpr(const IntegerExpr<KIND> &x) : u{AnyIntegerExpr{x}} {}
  template<int KIND>
  AnyIntegerOrRealExpr(IntegerExpr<KIND> &&x)
    : u{AnyIntegerExpr{std::move(x)}} {}
  template<int KIND>
  AnyIntegerOrRealExpr(const RealExpr<KIND> &x) : u{AnyRealExpr{x}} {}
  template<int KIND>
  AnyIntegerOrRealExpr(RealExpr<KIND> &&x) : u{AnyRealExpr{std::move(x)}} {}
  AnyIntegerOrRealExpr(const AnyIntegerExpr &x) : u{x} {}
  AnyIntegerOrRealExpr(AnyIntegerExpr &&x) : u{std::move(x)} {}
  AnyIntegerOrRealExpr(const AnyRealExpr &x) : u{x} {}
  AnyIntegerOrRealExpr(AnyRealExpr &&x) : u{std::move(x)} {}
  AnyIntegerOrRealExpr &operator=(const AnyIntegerOrRealExpr &) = default;
  AnyIntegerOrRealExpr &operator=(AnyIntegerOrRealExpr &&) = default;
  std::ostream &Dump(std::ostream &) const;
  std::variant<AnyIntegerExpr, AnyRealExpr> u;
};

// Convenience functions and operator overloadings for expression construction.
#define UNARY(FUNC, CONSTR) \
  template<typename A> A FUNC(const A &x) { return {typename A::CONSTR{x}}; }
UNARY(Parentheses, Parentheses)
UNARY(operator-, Negate)
#undef UNARY

#define BINARY(FUNC, CONSTR) \
  template<typename A> A FUNC(const A &x, const A &y) { \
    return {typename A::CONSTR{x, y}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, A> FUNC(const A &x, A &&y) { \
    return {typename A::CONSTR{A{x}, std::move(y)}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, A> FUNC(A &&x, const A &y) { \
    return {typename A::CONSTR{std::move(x), A{y}}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, A> FUNC(A &&x, A &&y) { \
    return {typename A::CONSTR{std::move(x), std::move(y)}}; \
  }

BINARY(operator+, Add)
BINARY(operator-, Subtract)
BINARY(operator*, Multiply)
BINARY(operator/, Divide)
BINARY(Power, Power)
#undef BINARY

#define BINARY(FUNC, CONSTR) \
  template<typename A> LogicalExpr FUNC(const A &x, const A &y) { \
    return {Comparison<A>{typename Comparison<A>::CONSTR{x, y}}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, LogicalExpr> FUNC( \
      const A &x, A &&y) { \
    return { \
        Comparison<A>{typename Comparison<A>::CONSTR{A{x}, std::move(y)}}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, LogicalExpr> FUNC( \
      A &&x, const A &y) { \
    return { \
        Comparison<A>{typename Comparison<A>::CONSTR{std::move(x), A{y}}}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, LogicalExpr> FUNC(A &&x, A &&y) { \
    return {Comparison<A>{ \
        typename Comparison<A>::CONSTR{std::move(x), std::move(y)}}}; \
  }

BINARY(operator<, LT)
BINARY(operator<=, LE)
BINARY(operator==, EQ)
BINARY(operator!=, NE)
BINARY(operator>=, GE)
BINARY(operator>, GT)
#undef BINARY

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
