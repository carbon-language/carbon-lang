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
// Expressions are the sole owners of their constituents; i.e., there is no
// context-independent hash table or sharing of common subexpressions.
// Both deep copy and move semantics are supported for expression construction
// and manipulation in place.
// TODO: variable and function references
// TODO: elevate some intrinsics to operations
// TODO: convenience wrappers for constructing conversions

#include "common.h"
#include "type.h"
#include "../lib/common/idioms.h"
#include "../lib/parser/char-block.h"
#include "../lib/parser/message.h"
#include <memory>
#include <ostream>
#include <variant>

namespace Fortran::evaluate {

// Some forward definitions
template<Category CAT, int KIND> struct Expr;
template<Category CAT> struct AnyKindExpr;

template<int KIND> using IntegerExpr = Expr<Category::Integer, KIND>;
using DefaultIntegerExpr = IntegerExpr<DefaultInteger::kind>;
template<int KIND> using RealExpr = Expr<Category::Real, KIND>;
template<int KIND> using ComplexExpr = Expr<Category::Complex, KIND>;
template<int KIND> using CharacterExpr = Expr<Category::Character, KIND>;
using LogicalExpr = Expr<Category::Logical, 1>;
using AnyKindIntegerExpr = AnyKindExpr<Category::Integer>;
using AnyKindRealExpr = AnyKindExpr<Category::Real>;
using AnyKindComplexExpr = AnyKindExpr<Category::Complex>;
using AnyKindCharacterExpr = AnyKindExpr<Category::Character>;

struct FoldingContext {
  const parser::CharBlock &at;
  parser::Messages *messages;
};

// Helper base classes to manage subexpressions, which are known as data
// members named 'x' and, for binary operations, 'y'.
template<typename A> struct Unary {
  Unary(const A &a) : x{std::make_unique<A>(a)} {}
  Unary(std::unique_ptr<A> &&a) : x{std::move(a)} {}
  Unary(A &&a) : x{std::make_unique<A>(std::move(a))} {}
  Unary(const Unary &that) : x{std::make_unique<A>(*that.x)} {}
  Unary(Unary &&) = default;
  Unary &operator=(const Unary &that) {
    x = std::make_unique<A>(*that.x);
    return *this;
  }
  Unary &operator=(Unary &&) = default;
  std::ostream &Dump(std::ostream &, const char *opr) const;
  std::unique_ptr<A> x;
};

template<typename A, typename B = A> struct Binary {
  Binary(const A &a, const B &b)
    : x{std::make_unique<A>(a)}, y{std::make_unique<B>(b)} {}
  Binary(std::unique_ptr<const A> &&a, std::unique_ptr<const B> &&b)
    : x{std::move(a)}, y{std::move(b)} {}
  Binary(A &&a, B &&b)
    : x{std::make_unique<A>(std::move(a))}, y{std::make_unique<B>(
                                                std::move(b))} {}
  Binary(const Binary &that)
    : x{std::make_unique<A>(*that.x)}, y{std::make_unique<B>(*that.y)} {}
  Binary(Binary &&) = default;
  Binary &operator=(const Binary &that) {
    x = std::make_unique<A>(*that.x);
    y = std::make_unique<B>(*that.y);
    return *this;
  }
  Binary &operator=(Binary &&) = default;
  std::ostream &Dump(std::ostream &, const char *opr) const;
  std::unique_ptr<A> x;
  std::unique_ptr<B> y;
};

template<int KIND> struct Expr<Category::Integer, KIND> {
  using Result = Type<Category::Integer, KIND>;
  using Constant = typename Result::Value;
  template<typename A> struct Convert : Unary<A> { using Unary<A>::Unary; };
  using Un = Unary<Expr>;
  using Bin = Binary<Expr>;
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

  Expr() = delete;
  Expr(const Expr &) = default;
  Expr(Expr &&) = default;
  Expr(const Constant &x) : u{x} {}
  Expr(std::int64_t n) : u{Constant{n}} {}
  Expr(int n) : u{Constant{n}} {}
  template<Category CAT, int K>
  Expr(const Expr<CAT, K> &x)
    : u{Convert<AnyKindExpr<CAT>>{AnyKindExpr<CAT>{x}}} {}
  template<Category CAT, int K>
  Expr(Expr<CAT, K> &&x)
    : u{Convert<AnyKindExpr<CAT>>{AnyKindExpr<CAT>{std::move(x)}}} {}
  template<typename A> Expr(const A &x) : u{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u(std::move(x)) {}
  Expr &operator=(const Expr &) = default;
  Expr &operator=(Expr &&) = default;

  std::ostream &Dump(std::ostream &) const;
  void Fold(FoldingContext &);

  std::variant<Constant, Convert<AnyKindIntegerExpr>, Convert<AnyKindRealExpr>,
      Parentheses, Negate, Add, Subtract, Multiply, Divide, Power>
      u;
};

template<int KIND> struct Expr<Category::Real, KIND> {
  using Result = Type<Category::Real, KIND>;
  using Constant = typename Result::Value;
  // N.B. Real->Complex and Complex->Real conversions are done with CMPLX
  // and part access operations (resp.).  Conversions between kinds of
  // Complex are done via decomposition to Real and reconstruction.
  template<typename A> struct Convert : Unary<A> { using Unary<A>::Unary; };
  using Un = Unary<Expr>;
  using Bin = Binary<Expr>;
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
  struct IntPower : public Binary<Expr, AnyKindIntegerExpr> {
    using Binary<Expr, AnyKindIntegerExpr>::Binary;
  };
  using CplxUn = Unary<ComplexExpr<KIND>>;
  struct RealPart : public CplxUn {
    using CplxUn::CplxUn;
  };
  struct AIMAG : public CplxUn {
    using CplxUn::CplxUn;
  };

  Expr() = delete;
  Expr(const Expr &) = default;
  Expr(Expr &&) = default;
  Expr(const Constant &x) : u{x} {}
  template<Category CAT, int K>
  Expr(const Expr<CAT, K> &x)
    : u{Convert<AnyKindExpr<CAT>>{AnyKindExpr<CAT>{x}}} {}
  template<Category CAT, int K>
  Expr(Expr<CAT, K> &&x)
    : u{Convert<AnyKindExpr<CAT>>{AnyKindExpr<CAT>{std::move(x)}}} {}
  template<typename A> Expr(const A &x) : u{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u{std::move(x)} {}
  Expr &operator=(const Expr &) = default;
  Expr &operator=(Expr &&) = default;

  std::ostream &Dump(std::ostream &) const;

  std::variant<Constant, Convert<AnyKindIntegerExpr>, Convert<AnyKindRealExpr>,
      Parentheses, Negate, Add, Subtract, Multiply, Divide, Power, IntPower,
      RealPart, AIMAG>
      u;
};

template<int KIND> struct Expr<Category::Complex, KIND> {
  using Result = Type<Category::Complex, KIND>;
  using Constant = typename Result::Value;
  using Un = Unary<Expr>;
  using Bin = Binary<Expr>;
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
  struct IntPower : public Binary<Expr, AnyKindIntegerExpr> {
    using Binary<Expr, AnyKindIntegerExpr>::Binary;
  };
  struct CMPLX : public Binary<RealExpr<KIND>> {
    using Binary<RealExpr<KIND>>::Binary;
  };

  Expr() = delete;
  Expr(const Expr &) = default;
  Expr(Expr &&) = default;
  Expr(const Constant &x) : u{x} {}
  template<typename A> Expr(const A &x) : u{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u{std::move(x)} {}
  Expr &operator=(const Expr &) = default;
  Expr &operator=(Expr &&) = default;

  std::ostream &Dump(std::ostream &) const;

  std::variant<Constant, Parentheses, Negate, Add, Subtract, Multiply, Divide,
      Power, IntPower, CMPLX>
      u;
};

template<int KIND> struct Expr<Category::Character, KIND> {
  using Result = Type<Category::Character, KIND>;
  using Constant = typename Result::Value;
  using LengthExpr = IntegerExpr<IntrinsicTypeParameterType::kind>;
  struct Concat : public Binary<Expr> {
    using Binary<Expr>::Binary;
  };

  Expr() = delete;
  Expr(const Expr &) = default;
  Expr(Expr &&) = default;
  Expr(const Constant &x) : u{x} {}
  Expr(Constant &&x) : u{std::move(x)} {}
  Expr(const Concat &x) : u{x} {}
  Expr(Concat &&x) : u{std::move(x)} {}
  Expr &operator=(const Expr &) = default;
  Expr &operator=(Expr &&) = default;

  LengthExpr LEN() const;
  std::ostream &Dump(std::ostream &) const;

  std::variant<Constant, Concat> u;
};

// The Comparison class template is a helper for constructing logical
// expressions with polymorphism over the cross product of the possible
// categories and kinds of comparable operands.
ENUM_CLASS(RelationalOperator, LT, LE, EQ, NE, GE, GT)

template<typename EXPR> struct Comparison : Binary<EXPR> {
  Comparison(const Comparison &) = default;
  Comparison(Comparison &&) = default;
  Comparison(RelationalOperator r, const EXPR &a, const EXPR &b)
    : Binary<EXPR>{a, b}, opr{r} {}
  Comparison(RelationalOperator r, EXPR &&a, EXPR &&b)
    : Binary<EXPR>{std::move(a), std::move(b)}, opr{r} {}
  Comparison &operator=(const Comparison &) = default;
  Comparison &operator=(Comparison &&) = default;

  std::ostream &Dump(std::ostream &) const;
  RelationalOperator opr;
};

extern template struct Comparison<IntegerExpr<1>>;
extern template struct Comparison<IntegerExpr<2>>;
extern template struct Comparison<IntegerExpr<4>>;
extern template struct Comparison<IntegerExpr<8>>;
extern template struct Comparison<IntegerExpr<16>>;
extern template struct Comparison<RealExpr<2>>;
extern template struct Comparison<RealExpr<4>>;
extern template struct Comparison<RealExpr<8>>;
extern template struct Comparison<RealExpr<10>>;
extern template struct Comparison<RealExpr<16>>;
extern template struct Comparison<ComplexExpr<2>>;
extern template struct Comparison<ComplexExpr<4>>;
extern template struct Comparison<ComplexExpr<8>>;
extern template struct Comparison<ComplexExpr<10>>;
extern template struct Comparison<ComplexExpr<16>>;
extern template struct Comparison<CharacterExpr<1>>;

// Dynamically polymorphic comparisonsq that can hold any supported kind
// of a category.
template<Category CAT> struct AnyKindComparison {
  AnyKindComparison() = delete;
  AnyKindComparison(const AnyKindComparison &) = default;
  AnyKindComparison(AnyKindComparison &&) = default;
  template<int KIND>
  AnyKindComparison(const Comparison<Expr<CAT, KIND>> &x) : u{x} {}
  template<int KIND>
  AnyKindComparison(Comparison<Expr<CAT, KIND>> &&x) : u{std::move(x)} {}
  AnyKindComparison &operator=(const AnyKindComparison &) = default;
  AnyKindComparison &operator=(AnyKindComparison &&) = default;
  std::ostream &Dump(std::ostream &) const;
  template<int K> using KindComparison = Comparison<Expr<CAT, K>>;
  typename KindsVariant<CAT, KindComparison>::type u;
};

// No need to distinguish the various kinds of LOGICAL expression results.
template<> struct Expr<Category::Logical, 1> {
  using Constant = bool;
  struct Not : Unary<Expr> {
    using Unary<Expr>::Unary;
  };
  using Bin = Binary<Expr, Expr>;
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

  Expr() = delete;
  Expr(const Expr &) = default;
  Expr(Expr &&) = default;
  Expr(Constant x) : u{x} {}
  template<Category CAT, int KIND>
  Expr(const Comparison<Expr<CAT, KIND>> &x) : u{AnyKindComparison<CAT>{x}} {}
  template<Category CAT, int KIND>
  Expr(Comparison<Expr<CAT, KIND>> &&x)
    : u{AnyKindComparison<CAT>{std::move(x)}} {}
  template<typename A> Expr(const A &x) : u{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u{std::move(x)} {}
  Expr &operator=(const Expr &) = default;
  Expr &operator=(Expr &&) = default;

  std::ostream &Dump(std::ostream &) const;

  std::variant<Constant, Not, And, Or, Eqv, Neqv,
      AnyKindComparison<Category::Integer>, AnyKindComparison<Category::Real>,
      AnyKindComparison<Category::Complex>,
      AnyKindComparison<Category::Character>>
      u;
};

extern template struct Expr<Category::Integer, 1>;
extern template struct Expr<Category::Integer, 2>;
extern template struct Expr<Category::Integer, 4>;
extern template struct Expr<Category::Integer, 8>;
extern template struct Expr<Category::Integer, 16>;
extern template struct Expr<Category::Real, 2>;
extern template struct Expr<Category::Real, 4>;
extern template struct Expr<Category::Real, 8>;
extern template struct Expr<Category::Real, 10>;
extern template struct Expr<Category::Real, 16>;
extern template struct Expr<Category::Complex, 2>;
extern template struct Expr<Category::Complex, 4>;
extern template struct Expr<Category::Complex, 8>;
extern template struct Expr<Category::Complex, 10>;
extern template struct Expr<Category::Complex, 16>;
extern template struct Expr<Category::Character, 1>;
extern template struct Expr<Category::Logical, 1>;

// Dynamically polymorphic expressions that can hold any supported kind
// of a category.
template<Category CAT> struct AnyKindExpr {
  AnyKindExpr() = delete;
  AnyKindExpr(const AnyKindExpr &) = default;
  AnyKindExpr(AnyKindExpr &&) = default;
  template<int KIND> AnyKindExpr(const Expr<CAT, KIND> &x) : u{x} {}
  template<int KIND> AnyKindExpr(Expr<CAT, KIND> &&x) : u{std::move(x)} {}
  AnyKindExpr &operator=(const AnyKindExpr &) = default;
  AnyKindExpr &operator=(AnyKindExpr &&) = default;
  std::ostream &Dump(std::ostream &) const;
  template<int K> using KindExpr = Expr<CAT, K>;
  typename KindsVariant<CAT, KindExpr>::type u;
};

struct AnyExpr {
  AnyExpr() = delete;
  AnyExpr(const AnyExpr &) = default;
  AnyExpr(AnyExpr &&) = default;
  template<Category CAT, int KIND>
  AnyExpr(const Expr<CAT, KIND> &x) : u{AnyKindExpr<CAT>{x}} {}
  template<Category CAT, int KIND>
  AnyExpr(Expr<CAT, KIND> &&x) : u{AnyKindExpr<CAT>{std::move(x)}} {}
  template<typename A> AnyExpr(const A &x) : u{x} {}
  template<typename A>
  AnyExpr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u{std::move(x)} {}
  AnyExpr &operator=(const AnyExpr &) = default;
  AnyExpr &operator=(AnyExpr &&) = default;
  std::ostream &Dump(std::ostream &) const;
  std::variant<AnyKindIntegerExpr, AnyKindRealExpr, AnyKindComplexExpr,
      AnyKindCharacterExpr, LogicalExpr>
      u;
};

// Convenience functions and operator overloadings for expression construction.
// These definitions are created with temporary helper macros to reduce
// C++ boilerplate.  All combinations of lvalue and rvalue references are
// allowed for operands.
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

#define BINARY(FUNC, OP) \
  template<typename A> LogicalExpr FUNC(const A &x, const A &y) { \
    return {Comparison<A>{OP, x, y}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, LogicalExpr> FUNC( \
      const A &x, A &&y) { \
    return {Comparison<A>{OP, x, std::move(y)}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, LogicalExpr> FUNC( \
      A &&x, const A &y) { \
    return {Comparison<A>{OP, std::move(x), y}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, LogicalExpr> FUNC(A &&x, A &&y) { \
    return {Comparison<A>{OP, std::move(x), std::move(y)}}; \
  }

BINARY(operator<, RelationalOperator::LT)
BINARY(operator<=, RelationalOperator::LE)
BINARY(operator==, RelationalOperator::EQ)
BINARY(operator!=, RelationalOperator::NE)
BINARY(operator>=, RelationalOperator::GE)
BINARY(operator>, RelationalOperator::GT)
#undef BINARY
}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_EXPRESSION_H_
