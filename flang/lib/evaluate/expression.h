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
// TODO: convenience wrappers for constructing conversions

#include "common.h"
#include "expression-forward.h"
#include "type.h"
#include "variable.h"
#include "../lib/common/idioms.h"
#include "../lib/parser/char-block.h"
#include "../lib/parser/message.h"
#include <ostream>
#include <variant>

namespace Fortran::evaluate {

struct FoldingContext {
  const parser::CharBlock &at;
  parser::Messages *messages;
};

// Helper base classes for packaging subexpressions, which are known as data
// members named 'x' and, for binary operations, 'y'.
template<typename A> struct Unary {
  CLASS_BOILERPLATE(Unary)
  Unary(const A &a) : x{a} {}
  Unary(CopyableIndirection<A> &&a) : x{std::move(a)} {}
  Unary(A &&a) : x{std::move(a)} {}
  std::ostream &Dump(std::ostream &, const char *opr) const;
  CopyableIndirection<A> x;
};

template<typename A, typename B = A> struct Binary {
  CLASS_BOILERPLATE(Binary)
  Binary(const A &a, const B &b) : x{a}, y{b} {}
  Binary(CopyableIndirection<const A> &&a, CopyableIndirection<const B> &&b)
    : x{std::move(a)}, y{std::move(b)} {}
  Binary(A &&a, B &&b) : x{std::move(a)}, y{std::move(b)} {}
  std::ostream &Dump(
      std::ostream &, const char *opr, const char *before = "(") const;
  CopyableIndirection<A> x;
  CopyableIndirection<B> y;
};

template<int KIND> struct Expr<Category::Integer, KIND> {
  using Result = Type<Category::Integer, KIND>;
  using Constant = typename Result::Value;
  template<typename A> struct Convert : public Unary<A> {
    using Unary<A>::Unary;
  };
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
  struct Max : public Bin {
    using Bin::Bin;
  };
  struct Min : public Bin {
    using Bin::Bin;
  };
  // TODO: R916 type-param-inquiry

  CLASS_BOILERPLATE(Expr)
  Expr(const Constant &x) : u{x} {}
  Expr(std::int64_t n) : u{Constant{n}} {}
  Expr(std::uint64_t n) : u{Constant{n}} {}
  Expr(int n) : u{Constant{n}} {}
  template<Category CAT, int K>
  Expr(const Expr<CAT, K> &x)
    : u{Convert<CategoryExpr<CAT>>{CategoryExpr<CAT>{x}}} {}
  template<Category CAT, int K>
  Expr(Expr<CAT, K> &&x)
    : u{Convert<CategoryExpr<CAT>>{CategoryExpr<CAT>{std::move(x)}}} {}
  template<typename A> Expr(const A &x) : u{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A> &&
          (std::is_base_of_v<Un, A> || std::is_base_of_v<Bin, A>),
      A> &&x)
    : u(std::move(x)) {}
  template<typename A> Expr(CopyableIndirection<A> &&x) : u{std::move(x)} {}

  void Fold(FoldingContext &);

  std::variant<Constant, CopyableIndirection<DataRef>,
      CopyableIndirection<FunctionRef>, Convert<GenericIntegerExpr>,
      Convert<GenericRealExpr>, Parentheses, Negate, Add, Subtract, Multiply,
      Divide, Power, Max, Min>
      u;
};

template<int KIND> struct Expr<Category::Real, KIND> {
  using Result = Type<Category::Real, KIND>;
  using Constant = typename Result::Value;
  // N.B. Real->Complex and Complex->Real conversions are done with CMPLX
  // and part access operations (resp.).  Conversions between kinds of
  // Complex are done via decomposition to Real and reconstruction.
  template<typename A> struct Convert : public Unary<A> {
    using Unary<A>::Unary;
  };
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
  struct IntPower : public Binary<Expr, GenericIntegerExpr> {
    using Binary<Expr, GenericIntegerExpr>::Binary;
  };
  struct Max : public Bin {
    using Bin::Bin;
  };
  struct Min : public Bin {
    using Bin::Bin;
  };
  using CplxUn = Unary<ComplexExpr<KIND>>;
  struct RealPart : public CplxUn {
    using CplxUn::CplxUn;
  };
  struct AIMAG : public CplxUn {
    using CplxUn::CplxUn;
  };

  CLASS_BOILERPLATE(Expr)
  Expr(const Constant &x) : u{x} {}
  template<Category CAT, int K>
  Expr(const Expr<CAT, K> &x)
    : u{Convert<CategoryExpr<CAT>>{CategoryExpr<CAT>{x}}} {}
  template<Category CAT, int K>
  Expr(Expr<CAT, K> &&x)
    : u{Convert<CategoryExpr<CAT>>{CategoryExpr<CAT>{std::move(x)}}} {}
  template<typename A> Expr(const A &x) : u{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u{std::move(x)} {}
  template<typename A> Expr(CopyableIndirection<A> &&x) : u{std::move(x)} {}

  std::variant<Constant, CopyableIndirection<DataRef>,
      CopyableIndirection<ComplexPart>, CopyableIndirection<FunctionRef>,
      Convert<GenericIntegerExpr>, Convert<GenericRealExpr>, Parentheses,
      Negate, Add, Subtract, Multiply, Divide, Power, IntPower, Max, Min,
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
  struct IntPower : public Binary<Expr, GenericIntegerExpr> {
    using Binary<Expr, GenericIntegerExpr>::Binary;
  };
  struct CMPLX : public Binary<RealExpr<KIND>> {
    using Binary<RealExpr<KIND>>::Binary;
  };

  CLASS_BOILERPLATE(Expr)
  Expr(const Constant &x) : u{x} {}
  template<typename A> Expr(const A &x) : u{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u{std::move(x)} {}
  template<typename A> Expr(CopyableIndirection<A> &&x) : u{std::move(x)} {}

  std::variant<Constant, CopyableIndirection<DataRef>,
      CopyableIndirection<FunctionRef>, Parentheses, Negate, Add, Subtract,
      Multiply, Divide, Power, IntPower, CMPLX>
      u;
};

template<int KIND> struct Expr<Category::Character, KIND> {
  using Result = Type<Category::Character, KIND>;
  using Constant = typename Result::Value;
  using Bin = Binary<Expr>;
  struct Concat : public Bin {
    using Bin::Bin;
  };
  struct Max : public Bin {
    using Bin::Bin;
  };
  struct Min : public Bin {
    using Bin::Bin;
  };

  CLASS_BOILERPLATE(Expr)
  Expr(const Constant &x) : u{x} {}
  Expr(Constant &&x) : u{std::move(x)} {}
  template<typename A> Expr(const A &x) : u{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u{std::move(x)} {}
  template<typename A> Expr(CopyableIndirection<A> &&x) : u{std::move(x)} {}

  SubscriptIntegerExpr LEN() const;

  std::variant<Constant, CopyableIndirection<DataRef>,
      CopyableIndirection<Substring>, CopyableIndirection<FunctionRef>, Concat,
      Max, Min>
      u;
};

// The Comparison class template is a helper for constructing logical
// expressions with polymorphism over the cross product of the possible
// categories and kinds of comparable operands.
ENUM_CLASS(RelationalOperator, LT, LE, EQ, NE, GE, GT)

template<typename EXPR> struct Comparison : Binary<EXPR> {
  CLASS_BOILERPLATE(Comparison)
  Comparison(RelationalOperator r, const EXPR &a, const EXPR &b)
    : Binary<EXPR>{a, b}, opr{r} {}
  Comparison(RelationalOperator r, EXPR &&a, EXPR &&b)
    : Binary<EXPR>{std::move(a), std::move(b)}, opr{r} {}
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

// Dynamically polymorphic comparisons that can hold any supported kind
// of a specific category.
template<Category CAT> struct CategoryComparison {
  CLASS_BOILERPLATE(CategoryComparison)
  template<int KIND>
  CategoryComparison(const Comparison<Expr<CAT, KIND>> &x) : u{x} {}
  template<int KIND>
  CategoryComparison(Comparison<Expr<CAT, KIND>> &&x) : u{std::move(x)} {}
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

  CLASS_BOILERPLATE(Expr)
  Expr(Constant x) : u{x} {}
  template<Category CAT, int KIND>
  Expr(const Comparison<Expr<CAT, KIND>> &x) : u{CategoryComparison<CAT>{x}} {}
  template<Category CAT, int KIND>
  Expr(Comparison<Expr<CAT, KIND>> &&x)
    : u{CategoryComparison<CAT>{std::move(x)}} {}
  template<typename A> Expr(const A &x) : u(x) {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u{std::move(x)} {}
  template<typename A> Expr(CopyableIndirection<A> &&x) : u{std::move(x)} {}

  std::variant<Constant, CopyableIndirection<DataRef>,
      CopyableIndirection<FunctionRef>, Not, And, Or, Eqv, Neqv,
      CategoryComparison<Category::Integer>, CategoryComparison<Category::Real>,
      CategoryComparison<Category::Complex>,
      CategoryComparison<Category::Character>>
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
// of a specific category.
template<Category CAT> struct CategoryExpr {
  CLASS_BOILERPLATE(CategoryExpr)
  template<int KIND> CategoryExpr(const Expr<CAT, KIND> &x) : u{x} {}
  template<int KIND> CategoryExpr(Expr<CAT, KIND> &&x) : u{std::move(x)} {}
  template<int K> using KindExpr = Expr<CAT, K>;
  typename KindsVariant<CAT, KindExpr>::type u;
};

// A completely generic expression, polymorphic across the type categories.
struct GenericExpr {
  CLASS_BOILERPLATE(GenericExpr)
  template<Category CAT, int KIND>
  GenericExpr(const Expr<CAT, KIND> &x) : u{CategoryExpr<CAT>{x}} {}
  template<Category CAT, int KIND>
  GenericExpr(Expr<CAT, KIND> &&x) : u{CategoryExpr<CAT>{std::move(x)}} {}
  template<typename A> GenericExpr(const A &x) : u{x} {}
  template<typename A>
  GenericExpr(std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : u{std::move(x)} {}
  std::variant<GenericIntegerExpr, GenericRealExpr, GenericComplexExpr,
      GenericCharacterExpr, LogicalExpr>
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
