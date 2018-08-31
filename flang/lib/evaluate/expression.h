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
// Both deep copy and move semantics are supported for expression construction.

#include "common.h"
#include "type.h"
#include "variable.h"
#include "../lib/common/fortran.h"
#include "../lib/common/idioms.h"
#include "../lib/parser/char-block.h"
#include "../lib/parser/message.h"
#include <ostream>
#include <tuple>
#include <type_traits>
#include <variant>

namespace Fortran::evaluate {

using common::RelationalOperator;

// Expressions are represented by specializations of Expr.
// Each of these specializations wraps a single data member "u" that
// is a std::variant<> discriminated union over the representational
// types for the constants, variables, operations, and other entities that
// can be valid expressions in that context:
// - Expr<Type<CATEGORY, KIND>> is an expression whose result is of a
//   specific intrinsic type category and kind, e.g. Type<TypeCategory::Real, 4>
// - Expr<SomeKind<CATEGORY>> is a union of Expr<Type<CATEGORY, K>> for each
//   kind type parameter value K in that intrinsic type category
// - Expr<SomeType> is a union of Expr<SomeKind<CATEGORY>> over the five
//   intrinsic type categories of Fortran.
template<typename A> class Expr;

// Everything that can appear in, or as, a valid Fortran expression must be
// represented with an instance of some class containing a Result typedef that
// maps to some instantiation of Type<CATEGORY, KIND>, SomeKind<CATEGORY>,
// or SomeType.
template<typename A> using ResultType = typename std::decay_t<A>::Result;

// Wraps a constant value in a class to make its type clear.
template<typename T> struct Constant {
  using Result = T;
  using Value = Scalar<Result>;  // TODO rank > 0
  CLASS_BOILERPLATE(Constant)
  template<typename A> Constant(const A &x) : value{x} {}
  template<typename A>
  Constant(std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : value(std::move(x)) {}
  std::ostream &Dump(std::ostream &) const;
  Value value;
};

// BOZ literal constants need to be wide enough to hold an integer or real
// value of any supported kind.  They also need to be distinguishable from
// other integer constants, since they are permitted to be used in only a
// few situations.
using BOZLiteralConstant = value::Integer<128>;

// "Typeless" operands to INTEGER and REAL operations.
template<typename T> struct BOZConstant {
  using Result = T;
  using Value = BOZLiteralConstant;
  CLASS_BOILERPLATE(BOZConstant)
  BOZConstant(const BOZLiteralConstant &x) : value{x} {}
  BOZConstant(BOZLiteralConstant &&x) : value{std::move(x)} {}
  std::ostream &Dump(std::ostream &) const;
  Value value;
};

// These wrappers around data and function references expose their resolved
// types.
template<typename T> struct DataReference {
  using Result = T;
  CopyableIndirection<DataRef> reference;
};

template<typename T> struct FunctionReference {
  using Result = T;
  CopyableIndirection<FunctionRef> reference;
};

// Abstract Operation<> base class. The first type parameter is a "CRTP"
// reference to the specific operation class; e.g., Add is defined with
// struct Add : public Operation<Add, ...>.
template<typename DERIVED, typename RESULT, typename... OPERANDS>
class Operation {
  using OperandTypes = std::tuple<OPERANDS...>;
  static_assert(RESULT::kind > 0 || !"bad result Type");

public:
  using Derived = DERIVED;
  using Result = RESULT;
  static constexpr auto operands() { return std::tuple_size_v<OperandTypes>; }
  template<int J> using Operand = std::tuple_element_t<J, OperandTypes>;
  using IsFoldableTrait = std::true_type;

  // Unary operations wrap a single Expr with a CopyableIndirection.
  // Binary operations wrap a tuple of CopyableIndirections to Exprs.
private:
  using Container =
      std::conditional_t<operands() == 1, CopyableIndirection<Expr<Operand<0>>>,
          std::tuple<CopyableIndirection<Expr<OPERANDS>>...>>;

public:
  CLASS_BOILERPLATE(Operation)
  Operation(const Expr<OPERANDS> &... x) : operand_{x...} {}
  Operation(Expr<OPERANDS> &&... x)
    : operand_{std::forward<Expr<OPERANDS>>(x)...} {}

  Derived &derived() { return *static_cast<Derived *>(this); }
  const Derived &derived() const { return *static_cast<const Derived *>(this); }

  template<int J> Expr<Operand<J>> &operand() {
    if constexpr (operands() == 1) {
      static_assert(J == 0);
      return *operand_;
    } else {
      return *std::get<J>(operand_);
    }
  }
  template<int J> const Expr<Operand<J>> &operand() const {
    if constexpr (operands() == 1) {
      static_assert(J == 0);
      return *operand_;
    } else {
      return *std::get<J>(operand_);
    }
  }

  std::ostream &Dump(std::ostream &) const;
  std::optional<Constant<Result>> Fold(FoldingContext &);

protected:
  // Overridable string functions for Dump()
  static const char *prefix() { return "("; }
  static const char *infix() { return ","; }
  static const char *suffix() { return ")"; }

private:
  Container operand_;
};

// Unary operations

template<typename TO, TypeCategory FROMCAT>
struct Convert : public Operation<Convert<TO, FROMCAT>, TO, SomeKind<FROMCAT>> {
  using Result = TO;
  using Operand = SomeKind<FROMCAT>;
  using Base = Operation<Convert, Result, Operand>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &);
};

template<typename A>
struct Parentheses : public Operation<Parentheses<A>, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Parentheses, A, A>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &x) {
    return {x};
  }
};

template<typename A> struct Negate : public Operation<Negate<A>, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Negate, A, A>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &);
  static const char *prefix() { return "(-"; }
};

template<int KIND>
struct ComplexComponent
  : public Operation<ComplexComponent<KIND>, Type<TypeCategory::Real, KIND>,
        Type<TypeCategory::Complex, KIND>> {
  using Result = Type<TypeCategory::Real, KIND>;
  using Operand = Type<TypeCategory::Complex, KIND>;
  using Base = Operation<ComplexComponent, Result, Operand>;
  CLASS_BOILERPLATE(ComplexComponent)
  ComplexComponent(bool isImaginary, const Expr<Operand> &x)
    : Base{x}, isImaginaryPart{isImaginary} {}
  ComplexComponent(bool isImaginary, Expr<Operand> &&x)
    : Base{std::move(x)}, isImaginaryPart{isImaginary} {}

  std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &) const;
  const char *suffix() const { return isImaginaryPart ? "%IM)" : "%RE)"; }

  bool isImaginaryPart{true};
};

template<int KIND>
struct Not : public Operation<Not<KIND>, Type<TypeCategory::Logical, KIND>,
                 Type<TypeCategory::Logical, KIND>> {
  using Result = Type<TypeCategory::Logical, KIND>;
  using Operand = Result;
  using Base = Operation<Not, Result, Operand>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &);
  static const char *prefix() { return "(.NOT."; }
};

// Binary operations

template<typename A> struct Add : public Operation<Add<A>, A, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Add, A, A, A>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
  static constexpr const char *infix() { return "+"; }
};

template<typename A> struct Subtract : public Operation<Subtract<A>, A, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Subtract, A, A, A>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
  static constexpr const char *infix() { return "-"; }
};

template<typename A> struct Multiply : public Operation<Multiply<A>, A, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Multiply, A, A, A>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
  static constexpr const char *infix() { return "*"; }
};

template<typename A> struct Divide : public Operation<Divide<A>, A, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Divide, A, A, A>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
  static constexpr const char *infix() { return "/"; }
};

template<typename A> struct Power : public Operation<Power<A>, A, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Power, A, A, A>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
  static constexpr const char *infix() { return "**"; }
};

template<typename A>
struct RealToIntPower : public Operation<RealToIntPower<A>, A, A, SomeInteger> {
  using Base = Operation<RealToIntPower, A, A, SomeInteger>;
  using Result = A;
  using BaseOperand = A;
  using ExponentOperand = SomeInteger;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(FoldingContext &,
      const Scalar<BaseOperand> &, const Scalar<ExponentOperand> &);
  static constexpr const char *infix() { return "**"; }
};

template<typename A> struct Extremum : public Operation<Extremum<A>, A, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Extremum, A, A, A>;
  CLASS_BOILERPLATE(Extremum)
  Extremum(const Expr<Operand> &x, const Expr<Operand> &y,
      Ordering ord = Ordering::Greater)
    : Base{x, y}, ordering{ord} {}
  Extremum(
      Expr<Operand> &&x, Expr<Operand> &&y, Ordering ord = Ordering::Greater)
    : Base{std::move(x), std::move(y)}, ordering{ord} {}

  std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &) const;
  const char *prefix() const {
    return ordering == Ordering::Less ? "MIN(" : "MAX(";
  }

  Ordering ordering{Ordering::Greater};
};

template<int KIND>
struct ComplexConstructor
  : public Operation<ComplexConstructor<KIND>,
        Type<TypeCategory::Complex, KIND>, Type<TypeCategory::Real, KIND>,
        Type<TypeCategory::Real, KIND>> {
  using Result = Type<TypeCategory::Complex, KIND>;
  using Operand = Type<TypeCategory::Real, KIND>;
  using Base = Operation<ComplexConstructor, Result, Operand, Operand>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
};

template<int KIND>
struct Concat
  : public Operation<Concat<KIND>, Type<TypeCategory::Character, KIND>,
        Type<TypeCategory::Character, KIND>,
        Type<TypeCategory::Character, KIND>> {
  using Result = Type<TypeCategory::Character, KIND>;
  using Operand = Result;
  using Base = Operation<Concat, Result, Operand, Operand>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
  static constexpr const char *infix() { return "//"; }
};

ENUM_CLASS(LogicalOperator, And, Or, Eqv, Neqv)

template<int KIND>
struct LogicalOperation
  : public Operation<LogicalOperation<KIND>, Type<TypeCategory::Logical, KIND>,
        Type<TypeCategory::Logical, KIND>, Type<TypeCategory::Logical, KIND>> {
  using Result = Type<TypeCategory::Logical, KIND>;
  using Operand = Result;
  using Base = Operation<LogicalOperation, Result, Operand, Operand>;
  CLASS_BOILERPLATE(LogicalOperation)
  LogicalOperation(
      const Expr<Operand> &x, const Expr<Operand> &y, LogicalOperator opr)
    : Base{x, y}, logicalOperator{opr} {}
  LogicalOperation(Expr<Operand> &&x, Expr<Operand> &&y, LogicalOperator opr)
    : Base{std::move(x), std::move(y)}, logicalOperator{opr} {}

  std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &) const;
  const char *infix() const;

  LogicalOperator logicalOperator;
};

// Per-category expressions

// Common Expr<> behaviors
template<typename RESULT> struct ExpressionBase {
  using Result = RESULT;
  using Derived = Expr<Result>;

  Derived &derived() { return *static_cast<Derived *>(this); }
  const Derived &derived() const { return *static_cast<const Derived *>(this); }

  int Rank() const { return 0; }  // TODO

  template<typename A> Derived &operator=(const A &x) {
    Derived &d{derived()};
    d.u = x;
    return d;
  }

  template<typename A>
  Derived &operator=(std::enable_if_t<!std::is_reference_v<A>, A> &&x) {
    Derived &d{derived()};
    d.u = std::move(x);
    return d;
  }

  std::ostream &Dump(std::ostream &) const;
  std::optional<Constant<Result>> Fold(FoldingContext &c);
  std::optional<Scalar<Result>> ScalarValue() const;
};

template<int KIND>
class Expr<Type<TypeCategory::Integer, KIND>>
  : public ExpressionBase<Type<TypeCategory::Integer, KIND>> {
public:
  using Result = Type<TypeCategory::Integer, KIND>;
  using IsFoldableTrait = std::true_type;
  // TODO: R916 type-param-inquiry

  CLASS_BOILERPLATE(Expr)
  Expr(const Scalar<Result> &x) : u{Constant<Result>{x}} {}
  Expr(std::int64_t n) : u{Constant<Result>{n}} {}
  Expr(std::uint64_t n) : u{Constant<Result>{n}} {}
  Expr(int n) : u{Constant<Result>{n}} {}
  template<typename A> Expr(const A &x) : u{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u(std::move(x)) {}
  Expr(const DataRef &x) : u{DataReference<Result>{x}} {}
  Expr(const FunctionRef &x) : u{FunctionReference<Result>{x}} {}

private:
  using Conversions = std::variant<Convert<Result, TypeCategory::Integer>,
      Convert<Result, TypeCategory::Real>>;
  using Operations = std::variant<Parentheses<Result>, Negate<Result>,
      Add<Result>, Subtract<Result>, Multiply<Result>, Divide<Result>,
      Power<Result>, Extremum<Result>>;
  using Others = std::variant<Constant<Result>, BOZConstant<Result>,
      DataReference<Result>, FunctionReference<Result>>;

public:
  common::CombineVariants<Operations, Conversions, Others> u;
};

template<int KIND>
class Expr<Type<TypeCategory::Real, KIND>>
  : public ExpressionBase<Type<TypeCategory::Real, KIND>> {
public:
  using Result = Type<TypeCategory::Real, KIND>;
  using IsFoldableTrait = std::true_type;

  CLASS_BOILERPLATE(Expr)
  Expr(const Scalar<Result> &x) : u{Constant<Result>{x}} {}
  template<typename A> Expr(const A &x) : u{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u{std::move(x)} {}
  Expr(const DataRef &x) : u{DataReference<Result>{x}} {}
  Expr(const FunctionRef &x) : u{FunctionReference<Result>{x}} {}

private:
  // N.B. Real->Complex and Complex->Real conversions are done with CMPLX
  // and part access operations (resp.).  Conversions between kinds of
  // Complex are done via decomposition to Real and reconstruction.
  using Conversions = std::variant<Convert<Result, TypeCategory::Integer>,
      Convert<Result, TypeCategory::Real>>;
  using Operations = std::variant<ComplexComponent<KIND>, Parentheses<Result>,
      Negate<Result>, Add<Result>, Subtract<Result>, Multiply<Result>,
      Divide<Result>, Power<Result>, RealToIntPower<Result>, Extremum<Result>>;
  using Others = std::variant<Constant<Result>, BOZConstant<Result>,
      DataReference<Result>, FunctionReference<Result>>;

public:
  common::CombineVariants<Operations, Conversions, Others> u;
};

template<int KIND>
class Expr<Type<TypeCategory::Complex, KIND>>
  : public ExpressionBase<Type<TypeCategory::Complex, KIND>> {
public:
  using Result = Type<TypeCategory::Complex, KIND>;
  using IsFoldableTrait = std::true_type;
  CLASS_BOILERPLATE(Expr)
  Expr(const Scalar<Result> &x) : u{Constant<Result>{x}} {}
  template<typename A> Expr(const A &x) : u{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u{std::move(x)} {}
  Expr(const DataRef &x) : u{DataReference<Result>{x}} {}
  Expr(const FunctionRef &x) : u{FunctionReference<Result>{x}} {}

  using Operations = std::variant<Parentheses<Result>, Negate<Result>,
      Add<Result>, Subtract<Result>, Multiply<Result>, Divide<Result>,
      Power<Result>, RealToIntPower<Result>, ComplexConstructor<KIND>>;
  using Others = std::variant<Constant<Result>, DataReference<Result>,
      FunctionReference<Result>>;

public:
  common::CombineVariants<Operations, Others> u;
};

extern template class Expr<Type<TypeCategory::Integer, 1>>;
extern template class Expr<Type<TypeCategory::Integer, 2>>;
extern template class Expr<Type<TypeCategory::Integer, 4>>;
extern template class Expr<Type<TypeCategory::Integer, 8>>;
extern template class Expr<Type<TypeCategory::Integer, 16>>;
extern template class Expr<Type<TypeCategory::Real, 2>>;
extern template class Expr<Type<TypeCategory::Real, 4>>;
extern template class Expr<Type<TypeCategory::Real, 8>>;
extern template class Expr<Type<TypeCategory::Real, 10>>;
extern template class Expr<Type<TypeCategory::Real, 16>>;
extern template class Expr<Type<TypeCategory::Complex, 2>>;
extern template class Expr<Type<TypeCategory::Complex, 4>>;
extern template class Expr<Type<TypeCategory::Complex, 8>>;
extern template class Expr<Type<TypeCategory::Complex, 10>>;
extern template class Expr<Type<TypeCategory::Complex, 16>>;

template<int KIND>
class Expr<Type<TypeCategory::Character, KIND>>
  : public ExpressionBase<Type<TypeCategory::Character, KIND>> {
public:
  using Result = Type<TypeCategory::Character, KIND>;
  using IsFoldableTrait = std::true_type;
  CLASS_BOILERPLATE(Expr)
  Expr(const Scalar<Result> &x) : u{Constant<Result>{x}} {}
  Expr(Scalar<Result> &&x) : u{Constant<Result>{std::move(x)}} {}
  template<typename A> Expr(const A &x) : u{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u{std::move(x)} {}
  Expr(const DataRef &x) : u{DataReference<Result>{x}} {}
  Expr(const FunctionRef &x) : u{FunctionReference<Result>{x}} {}
  template<typename A> Expr(CopyableIndirection<A> &&x) : u{std::move(x)} {}

  Expr<SubscriptInteger> LEN() const;

  std::variant<Constant<Result>, DataReference<Result>,
      CopyableIndirection<Substring>, FunctionReference<Result>,
      // TODO Parentheses<Result>,
      Concat<KIND>, Extremum<Result>>
      u;
};

extern template class Expr<Type<TypeCategory::Character, 1>>;  // TODO more

// The Relational class template is a helper for constructing logical
// expressions with polymorphism over the cross product of the possible
// categories and kinds of comparable operands.
// Fortran defines a numeric relation with distinct types or kinds as
// undergoing the same operand conversions that occur with the addition
// intrinsic operator first.  Character relations must have the same kind.
// There are no relations between LOGICAL values.

template<typename A>
struct Relational : public Operation<Relational<A>, LogicalResult, A, A> {
  using Base = Operation<Relational, LogicalResult, A, A>;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
  CLASS_BOILERPLATE(Relational)
  Relational(
      RelationalOperator r, const Expr<Operand> &a, const Expr<Operand> &b)
    : Base{a, b}, opr{r} {}
  Relational(RelationalOperator r, Expr<Operand> &&a, Expr<Operand> &&b)
    : Base{std::move(a), std::move(b)}, opr{r} {}

  std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &c, const Scalar<Operand> &, const Scalar<Operand> &);
  std::string infix() const;

  RelationalOperator opr;
};

template<> struct Relational<SomeType> {
  using Result = LogicalResult;
  CLASS_BOILERPLATE(Relational)
  template<typename A> Relational(const A &x) : u(x) {}
  template<typename A>
  Relational(std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : u{std::move(x)} {}
  std::ostream &Dump(std::ostream &o) const;
  common::MapTemplate<Relational, RelationalTypes> u;
};

extern template struct Relational<Type<TypeCategory::Integer, 1>>;
extern template struct Relational<Type<TypeCategory::Integer, 2>>;
extern template struct Relational<Type<TypeCategory::Integer, 4>>;
extern template struct Relational<Type<TypeCategory::Integer, 8>>;
extern template struct Relational<Type<TypeCategory::Integer, 16>>;
extern template struct Relational<Type<TypeCategory::Real, 2>>;
extern template struct Relational<Type<TypeCategory::Real, 4>>;
extern template struct Relational<Type<TypeCategory::Real, 8>>;
extern template struct Relational<Type<TypeCategory::Real, 10>>;
extern template struct Relational<Type<TypeCategory::Real, 16>>;
extern template struct Relational<Type<TypeCategory::Complex, 2>>;
extern template struct Relational<Type<TypeCategory::Complex, 4>>;
extern template struct Relational<Type<TypeCategory::Complex, 8>>;
extern template struct Relational<Type<TypeCategory::Complex, 10>>;
extern template struct Relational<Type<TypeCategory::Complex, 16>>;
extern template struct Relational<Type<TypeCategory::Character, 1>>;  // TODO
                                                                      // more
extern template struct Relational<SomeType>;

template<int KIND>
class Expr<Type<TypeCategory::Logical, KIND>>
  : public ExpressionBase<Type<TypeCategory::Logical, KIND>> {
public:
  using Result = Type<TypeCategory::Logical, KIND>;
  using IsFoldableTrait = std::true_type;
  CLASS_BOILERPLATE(Expr)
  Expr(const Scalar<Result> &x) : u{Constant<Result>{x}} {}
  Expr(bool x) : u{Constant<Result>{x}} {}
  template<typename A> Expr(const A &x) : u(x) {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u{std::move(x)} {}
  Expr(const DataRef &x) : u{DataReference<Result>{x}} {}
  Expr(const FunctionRef &x) : u{FunctionReference<Result>{x}} {}

private:
  using Operations =
      std::variant<Not<KIND>, LogicalOperation<KIND>, Relational<SomeType>>;
  using Others = std::variant<Constant<Result>, DataReference<Result>,
      FunctionReference<Result>>;

public:
  common::CombineVariants<Operations, Others> u;
};

extern template class Expr<Type<TypeCategory::Logical, 1>>;
extern template class Expr<Type<TypeCategory::Logical, 2>>;
extern template class Expr<Type<TypeCategory::Logical, 4>>;
extern template class Expr<Type<TypeCategory::Logical, 8>>;

// A polymorphic expression of known intrinsic type category, but dynamic
// kind, represented as a discriminated union over Expr<Type<CAT, K>>
// for each supported kind K in the category.
template<TypeCategory CAT>
class Expr<SomeKind<CAT>> : public ExpressionBase<SomeKind<CAT>> {
public:
  using Result = SomeKind<CAT>;
  using IsFoldableTrait = std::true_type;
  CLASS_BOILERPLATE(Expr)

  template<typename A> Expr(const A &x) : u{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u{std::move(x)} {}

  common::MapTemplate<Expr, CategoryTypes<CAT>> u;
};

// A completely generic expression, polymorphic across all of the intrinsic type
// categories and each of their kinds.
template<> class Expr<SomeType> : public ExpressionBase<SomeType> {
public:
  using Result = SomeType;
  using IsFoldableTrait = std::true_type;
  CLASS_BOILERPLATE(Expr)

  template<typename A> Expr(const A &x) : u{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u{std::move(x)} {}

  template<TypeCategory CAT, int KIND>
  Expr(const Expr<Type<CAT, KIND>> &x) : u{Expr<SomeKind<CAT>>{x}} {}

  template<TypeCategory CAT, int KIND>
  Expr(Expr<Type<CAT, KIND>> &&x) : u{Expr<SomeKind<CAT>>{std::move(x)}} {}

  template<TypeCategory CAT, int KIND>
  Expr &operator=(const Expr<Type<CAT, KIND>> &x) {
    u = Expr<SomeKind<CAT>>{x};
    return *this;
  }

  template<TypeCategory CAT, int KIND>
  Expr &operator=(Expr<Type<CAT, KIND>> &&x) {
    u = Expr<SomeKind<CAT>>{std::move(x)};
    return *this;
  }

  using Others = std::variant<BOZLiteralConstant>;
  using Categories = common::MapTemplate<Expr, SomeCategory>;
  common::CombineVariants<Others, Categories> u;
};

extern template class Expr<SomeInteger>;
extern template class Expr<SomeReal>;
extern template class Expr<SomeComplex>;
extern template class Expr<SomeCharacter>;
extern template class Expr<SomeLogical>;
extern template class Expr<SomeType>;

extern template struct ExpressionBase<Type<TypeCategory::Integer, 1>>;
extern template struct ExpressionBase<Type<TypeCategory::Integer, 2>>;
extern template struct ExpressionBase<Type<TypeCategory::Integer, 4>>;
extern template struct ExpressionBase<Type<TypeCategory::Integer, 8>>;
extern template struct ExpressionBase<Type<TypeCategory::Integer, 16>>;
extern template struct ExpressionBase<Type<TypeCategory::Real, 2>>;
extern template struct ExpressionBase<Type<TypeCategory::Real, 4>>;
extern template struct ExpressionBase<Type<TypeCategory::Real, 8>>;
extern template struct ExpressionBase<Type<TypeCategory::Real, 10>>;
extern template struct ExpressionBase<Type<TypeCategory::Real, 16>>;
extern template struct ExpressionBase<Type<TypeCategory::Complex, 2>>;
extern template struct ExpressionBase<Type<TypeCategory::Complex, 4>>;
extern template struct ExpressionBase<Type<TypeCategory::Complex, 8>>;
extern template struct ExpressionBase<Type<TypeCategory::Complex, 10>>;
extern template struct ExpressionBase<Type<TypeCategory::Complex, 16>>;
extern template struct ExpressionBase<Type<TypeCategory::Character, 1>>;
extern template struct ExpressionBase<Type<TypeCategory::Logical, 1>>;
extern template struct ExpressionBase<Type<TypeCategory::Logical, 2>>;
extern template struct ExpressionBase<Type<TypeCategory::Logical, 4>>;
extern template struct ExpressionBase<Type<TypeCategory::Logical, 8>>;
extern template struct ExpressionBase<SomeInteger>;
extern template struct ExpressionBase<SomeReal>;
extern template struct ExpressionBase<SomeComplex>;
extern template struct ExpressionBase<SomeCharacter>;
extern template struct ExpressionBase<SomeLogical>;
extern template struct ExpressionBase<SomeType>;

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_EXPRESSION_H_
