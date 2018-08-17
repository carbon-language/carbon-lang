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

#include "common.h"
#include "type.h"
#include "variable.h"
#include "../lib/common/fortran.h"
#include "../lib/common/idioms.h"
#include "../lib/parser/char-block.h"
#include "../lib/parser/message.h"
#include <ostream>
#include <tuple>
#include <variant>

namespace Fortran::evaluate {

using common::RelationalOperator;

// Expr<A> represents an expression whose result is the Fortran type A,
// which can be specific, SomeKind<C> for a type category C, or
// Expr<SomeType> for a wholly generic expression.  Instances of Expr<>
// wrap discriminated unions.
template<typename A> class Expr;

template<typename DERIVED, typename RESULT, typename... OPERAND>
class Operation {
public:
  using Derived = DERIVED;
  using Result = RESULT;
  using OperandTypes = std::tuple<OPERAND...>;
  using OperandTuple = std::tuple<Expr<OPERAND>...>;
  template<int J> using Operand = std::tuple_element_t<J, OperandTypes>;
  using FoldableTrait = std::true_type;

  static_assert(Result::kind > 0);  // Operations have specific Result types

  CLASS_BOILERPLATE(Operation)
  Operation(const Expr<OPERAND> &... x) : operand_{OperandTuple{x...}} {}
  Operation(Expr<OPERAND> &&... x)
    : operand_{OperandTuple{std::forward<Expr<OPERAND>>(x)...}} {}

  DERIVED &derived() { return *static_cast<DERIVED *>(this); }
  const DERIVED &derived() const { return *static_cast<const DERIVED *>(this); }

  static constexpr auto operands() { return std::tuple_size_v<OperandTypes>; }
  template<int J> Expr<Operand<J>> &operand() { return std::get<J>(*operand_); }
  template<int J> const Expr<Operand<J>> &operand() const {
    return std::get<J>(*operand_);
  }

  std::ostream &Dump(std::ostream &) const;
  std::optional<Scalar<Result>> Fold(FoldingContext &);  // TODO rank > 0

protected:
  // Overridable string functions for Dump()
  static const char *prefix() { return "("; }
  static const char *infix() { return ","; }
  static const char *suffix() { return ")"; }

private:
  CopyableIndirection<OperandTuple> operand_;
};

// Unary operations

template<typename TO, typename FROM>
struct Convert : public Operation<Convert<TO, FROM>, TO, FROM> {
  using Base = Operation<Convert<TO, FROM>, TO, FROM>;
  using Base::Base;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &);
};

template<typename A>
struct Parentheses : public Operation<Parentheses<A>, A, A> {
  using Base = Operation<Parentheses, A, A>;
  using Base::Base;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &x) {
    return {x};
  }
};

template<typename A> struct Negate : public Operation<Negate<A>, A, A> {
  using Base = Operation<Negate, A, A>;
  using Base::Base;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &);
  static const char *prefix() { return "(-"; }
};

template<int KIND>
struct ComplexComponent
  : public Operation<ComplexComponent<KIND>, Type<TypeCategory::Real, KIND>,
        Type<TypeCategory::Complex, KIND>> {
  using Base = Operation<ComplexComponent, Type<TypeCategory::Real, KIND>,
      Type<TypeCategory::Complex, KIND>>;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
  CLASS_BOILERPLATE(ComplexComponent)
  ComplexComponent(bool isReal, const Expr<Operand> &x)
    : Base{x}, isRealPart{isReal} {}
  ComplexComponent(bool isReal, Expr<Operand> &&x)
    : Base{std::move(x)}, isRealPart{isReal} {}

  std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &) const;
  const char *suffix() const { return isRealPart ? "%RE)" : "%IM)"; }

  bool isRealPart{true};
};

template<int KIND>
struct Not : public Operation<Not<KIND>, Type<TypeCategory::Logical, KIND>,
                 Type<TypeCategory::Logical, KIND>> {
  using Base = Operation<Not, Type<TypeCategory::Logical, KIND>,
      Type<TypeCategory::Logical, KIND>>;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &);
  static const char *prefix() { return "(.NOT."; }
};

// Binary operations

template<typename A> struct Add : public Operation<Add<A>, A, A, A> {
  using Base = Operation<Add, A, A, A>;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
  static constexpr const char *infix() { return "+"; }
};

template<typename A> struct Subtract : public Operation<Subtract<A>, A, A, A> {
  using Base = Operation<Subtract, A, A, A>;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
  static constexpr const char *infix() { return "-"; }
};

template<typename A> struct Multiply : public Operation<Multiply<A>, A, A, A> {
  using Base = Operation<Multiply, A, A, A>;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
  static constexpr const char *infix() { return "*"; }
};

template<typename A> struct Divide : public Operation<Divide<A>, A, A, A> {
  using Base = Operation<Divide, A, A, A>;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
  static constexpr const char *infix() { return "/"; }
};

template<typename A> struct Power : public Operation<Power<A>, A, A, A> {
  using Base = Operation<Power, A, A, A>;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
  static constexpr const char *infix() { return "**"; }
};

template<typename A, typename B>
struct RealToIntPower : public Operation<RealToIntPower<A, B>, A, A, B> {
  using Base = Operation<RealToIntPower, A, A, B>;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
  using ExponentOperand = typename Base::template Operand<1>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(FoldingContext &,
      const Scalar<Operand> &, const Scalar<ExponentOperand> &);
  static constexpr const char *infix() { return "**"; }
};

template<typename A> struct Extremum : public Operation<Extremum<A>, A, A, A> {
  using Base = Operation<Extremum, A, A, A>;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
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
  using Base = Operation<ComplexConstructor, Type<TypeCategory::Complex, KIND>,
      Type<TypeCategory::Real, KIND>, Type<TypeCategory::Real, KIND>>;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
};

template<int KIND>
struct Concat
  : public Operation<Concat<KIND>, Type<TypeCategory::Character, KIND>,
        Type<TypeCategory::Character, KIND>,
        Type<TypeCategory::Character, KIND>> {
  using Base = Operation<Concat, Type<TypeCategory::Character, KIND>,
      Type<TypeCategory::Character, KIND>, Type<TypeCategory::Character, KIND>>;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
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
  using Base = Operation<LogicalOperation, Type<TypeCategory::Logical, KIND>,
      Type<TypeCategory::Logical, KIND>, Type<TypeCategory::Logical, KIND>>;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
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

template<int KIND> class Expr<Type<TypeCategory::Integer, KIND>> {
public:
  using Result = Type<TypeCategory::Integer, KIND>;
  using FoldableTrait = std::true_type;
  // TODO: R916 type-param-inquiry

  CLASS_BOILERPLATE(Expr)
  Expr(const Scalar<Result> &x) : u_{x} {}
  Expr(std::int64_t n) : u_{Scalar<Result>{n}} {}
  Expr(std::uint64_t n) : u_{Scalar<Result>{n}} {}
  Expr(int n) : u_{Scalar<Result>{n}} {}
  Expr(const Expr<SomeInteger> &x) : u_{Convert<Result, SomeInteger>{x}} {}
  Expr(Expr<SomeInteger> &&x)
    : u_{Convert<Result, SomeInteger>{std::move(x)}} {}
  template<int K>
  Expr(const Expr<Type<TypeCategory::Integer, K>> &x)
    : u_{Convert<Result, SomeInteger>{Expr<SomeInteger>{x}}} {}
  template<int K>
  Expr(Expr<Type<TypeCategory::Integer, K>> &&x)
    : u_{Convert<Result, SomeInteger>{Expr<SomeInteger>{std::move(x)}}} {}
  Expr(const Expr<SomeReal> &x) : u_{Convert<Result, SomeReal>{x}} {}
  Expr(Expr<SomeReal> &&x) : u_{Convert<Result, SomeReal>{std::move(x)}} {}
  template<int K>
  Expr(const Expr<Type<TypeCategory::Real, K>> &x)
    : u_{Convert<Result, SomeReal>{Expr<SomeReal>{x}}} {}
  template<int K>
  Expr(Expr<Type<TypeCategory::Real, K>> &&x)
    : u_{Convert<Result, SomeReal>{Expr<SomeReal>{std::move(x)}}} {}
  template<typename A> Expr(const A &x) : u_{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u_(std::move(x)) {}
  template<typename A> Expr(CopyableIndirection<A> &&x) : u_{std::move(x)} {}

  std::optional<Scalar<Result>> ScalarValue() const {
    // TODO: Also succeed when parenthesized constant
    return common::GetIf<Scalar<Result>>(u_);
  }
  std::ostream &Dump(std::ostream &) const;
  std::optional<Scalar<Result>> Fold(FoldingContext &c);
  int Rank() const { return 1; }  // TODO

private:
  std::variant<Scalar<Result>, CopyableIndirection<DataRef>,
      CopyableIndirection<FunctionRef>, Convert<Result, SomeInteger>,
      Convert<Result, SomeReal>, Parentheses<Result>, Negate<Result>,
      Add<Result>, Subtract<Result>, Multiply<Result>, Divide<Result>,
      Power<Result>, Extremum<Result>>
      u_;
};

template<int KIND> class Expr<Type<TypeCategory::Real, KIND>> {
public:
  using Result = Type<TypeCategory::Real, KIND>;
  using FoldableTrait = std::true_type;

  // N.B. Real->Complex and Complex->Real conversions are done with CMPLX
  // and part access operations (resp.).  Conversions between kinds of
  // Complex are done via decomposition to Real and reconstruction.

  CLASS_BOILERPLATE(Expr)
  Expr(const Scalar<Result> &x) : u_{x} {}
  Expr(const Expr<SomeInteger> &x) : u_{Convert<Result, SomeInteger>{x}} {}
  Expr(Expr<SomeInteger> &&x)
    : u_{Convert<Result, SomeInteger>{std::move(x)}} {}
  template<int K>
  Expr(const Expr<Type<TypeCategory::Integer, K>> &x)
    : u_{Convert<Result, SomeInteger>{Expr<SomeInteger>{x}}} {}
  template<int K>
  Expr(Expr<Type<TypeCategory::Integer, K>> &&x)
    : u_{Convert<Result, SomeInteger>{Expr<SomeInteger>{std::move(x)}}} {}
  Expr(const Expr<SomeReal> &x) : u_{Convert<Result, SomeReal>{x}} {}
  Expr(Expr<SomeReal> &&x) : u_{Convert<Result, SomeReal>{std::move(x)}} {}
  template<int K>
  Expr(const Expr<Type<TypeCategory::Real, K>> &x)
    : u_{Convert<Result, SomeReal>{Expr<SomeReal>{x}}} {}
  template<int K>
  Expr(Expr<Type<TypeCategory::Real, K>> &&x)
    : u_{Convert<Result, SomeReal>{Expr<SomeReal>{std::move(x)}}} {}
  template<typename A> Expr(const A &x) : u_{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u_{std::move(x)} {}
  template<typename A> Expr(CopyableIndirection<A> &&x) : u_{std::move(x)} {}

  std::optional<Scalar<Result>> ScalarValue() const {
    // TODO: parenthesized constants too
    return common::GetIf<Scalar<Result>>(u_);
  }
  std::ostream &Dump(std::ostream &) const;
  std::optional<Scalar<Result>> Fold(FoldingContext &c);
  int Rank() const { return 1; }  // TODO

private:
  std::variant<Scalar<Result>, CopyableIndirection<DataRef>,
      CopyableIndirection<ComplexPart>, CopyableIndirection<FunctionRef>,
      Convert<Result, SomeInteger>, Convert<Result, SomeReal>,
      ComplexComponent<KIND>, Parentheses<Result>, Negate<Result>, Add<Result>,
      Subtract<Result>, Multiply<Result>, Divide<Result>, Power<Result>,
      RealToIntPower<Result, SomeInteger>, Extremum<Result>>
      u_;
};

template<int KIND> class Expr<Type<TypeCategory::Complex, KIND>> {
public:
  using Result = Type<TypeCategory::Complex, KIND>;
  using FoldableTrait = std::true_type;
  CLASS_BOILERPLATE(Expr)
  Expr(const Scalar<Result> &x) : u_{x} {}
  template<typename A> Expr(const A &x) : u_{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u_{std::move(x)} {}
  template<typename A> Expr(CopyableIndirection<A> &&x) : u_{std::move(x)} {}

  std::optional<Scalar<Result>> ScalarValue() const {
    // TODO: parenthesized constants too
    return common::GetIf<Scalar<Result>>(u_);
  }
  std::ostream &Dump(std::ostream &) const;
  std::optional<Scalar<Result>> Fold(FoldingContext &c);
  int Rank() const { return 1; }  // TODO

private:
  std::variant<Scalar<Result>, CopyableIndirection<DataRef>,
      CopyableIndirection<FunctionRef>, Parentheses<Result>, Negate<Result>,
      Add<Result>, Subtract<Result>, Multiply<Result>, Divide<Result>,
      Power<Result>, RealToIntPower<Result, SomeInteger>,
      ComplexConstructor<KIND>>
      u_;
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

template<int KIND> class Expr<Type<TypeCategory::Character, KIND>> {
public:
  using Result = Type<TypeCategory::Character, KIND>;
  using FoldableTrait = std::true_type;
  CLASS_BOILERPLATE(Expr)
  Expr(const Scalar<Result> &x) : u_{x} {}
  Expr(Scalar<Result> &&x) : u_{std::move(x)} {}
  template<typename A> Expr(const A &x) : u_{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u_{std::move(x)} {}
  template<typename A> Expr(CopyableIndirection<A> &&x) : u_{std::move(x)} {}

  std::optional<Scalar<Result>> ScalarValue() const {
    // TODO: parenthesized constants too
    return common::GetIf<Scalar<Result>>(u_);
  }
  std::ostream &Dump(std::ostream &) const;
  std::optional<Scalar<Result>> Fold(FoldingContext &c);
  int Rank() const { return 1; }  // TODO
  Expr<SubscriptInteger> LEN() const;

private:
  std::variant<Scalar<Result>, CopyableIndirection<DataRef>,
      CopyableIndirection<Substring>, CopyableIndirection<FunctionRef>,
      // TODO Parentheses<Result>,
      Concat<KIND>, Extremum<Result>>
      u_;
};

// The Relation class template is a helper for constructing logical
// expressions with polymorphism over the cross product of the possible
// categories and kinds of comparable operands.
// Fortran defines a numeric relation with distinct types or kinds as
// undergoing the same operand conversions that occur with the addition
// intrinsic operator first.  Character relations must have the same kind.
// There are no relations between logicals.

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

// A generic relation between two operands of the same kind in some intrinsic
// type category (except LOGICAL).
template<TypeCategory CAT> struct Relational<SomeKind<CAT>> {
  using Result = LogicalResult;
  using Operand = SomeKind<CAT>;
  template<int KIND> using KindRelational = Relational<Type<CAT, KIND>>;

  CLASS_BOILERPLATE(Relational)
  template<int KIND> Relational(const KindRelational<KIND> &x) : u{x} {}
  template<int KIND> Relational(KindRelational<KIND> &&x) : u{std::move(x)} {}

  std::optional<Scalar<Result>> Fold(FoldingContext &);
  std::ostream &Dump(std::ostream &) const;

  KindsVariant<CAT, KindRelational> u;
};

template<int KIND> class Expr<Type<TypeCategory::Logical, KIND>> {
public:
  using Result = Type<TypeCategory::Logical, KIND>;
  using FoldableTrait = std::true_type;
  CLASS_BOILERPLATE(Expr)
  Expr(const Scalar<Result> &x) : u_{x} {}
  Expr(bool x) : u_{Scalar<Result>{x}} {}
  template<TypeCategory CAT, int K>
  Expr(const Relational<Type<CAT, K>> &x) : u_{Relational<SomeKind<CAT>>{x}} {}
  template<TypeCategory CAT, int K>
  Expr(Relational<Type<CAT, K>> &&x)
    : u_{Relational<SomeKind<CAT>>{std::move(x)}} {}
  template<typename A> Expr(const A &x) : u_(x) {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u_{std::move(x)} {}
  template<typename A> Expr(CopyableIndirection<A> &&x) : u_{std::move(x)} {}

  std::optional<Scalar<Result>> ScalarValue() const {
    // TODO: parenthesized constants too
    return common::GetIf<Scalar<Result>>(u_);
  }
  std::ostream &Dump(std::ostream &) const;
  std::optional<Scalar<Result>> Fold(FoldingContext &c);
  int Rank() const { return 1; }  // TODO

private:
  std::variant<Scalar<Result>, CopyableIndirection<DataRef>,
      CopyableIndirection<FunctionRef>,
      // TODO Parentheses<Result>,
      Not<KIND>, LogicalOperation<KIND>,
      Relational<SomeKind<TypeCategory::Integer>>,
      Relational<SomeKind<TypeCategory::Real>>,
      Relational<SomeKind<TypeCategory::Complex>>,
      Relational<SomeKind<TypeCategory::Character>>>
      u_;
};

// Dynamically polymorphic expressions that can hold any supported kind
// of a specific intrinsic type category.
template<TypeCategory CAT> class Expr<SomeKind<CAT>> {
public:
  using Result = SomeKind<CAT>;
  using FoldableTrait = std::true_type;
  CLASS_BOILERPLATE(Expr)

  template<int KIND> using KindExpr = Expr<Type<CAT, KIND>>;
  template<int KIND> Expr(const KindExpr<KIND> &x) : u{x} {}
  template<int KIND> Expr(KindExpr<KIND> &&x) : u{std::move(x)} {}
  std::optional<Scalar<Result>> ScalarValue() const;
  std::ostream &Dump(std::ostream &) const;
  std::optional<Scalar<Result>> Fold(FoldingContext &);
  int Rank() const;

  KindsVariant<CAT, KindExpr> u;
};

// BOZ literal constants need to be wide enough to hold an integer or real
// value of any supported kind.  They also need to be distinguishable from
// other integer constants, since they are permitted to be used in only a
// few situations.
using BOZLiteralConstant = value::Integer<128>;

// A completely generic expression, polymorphic across the intrinsic type
// categories and each of their kinds.
template<> class Expr<SomeType> {
public:
  using Result = SomeType;
  using FoldableTrait = std::true_type;
  CLASS_BOILERPLATE(Expr)

  template<typename A> Expr(const A &x) : u{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u{std::move(x)} {}

  template<TypeCategory CAT, int KIND>
  Expr(const Expr<Type<CAT, KIND>> &x) : u{Expr<SomeKind<CAT>>{x}} {}

  template<TypeCategory CAT, int KIND>
  Expr(Expr<Type<CAT, KIND>> &&x) : u{Expr<SomeKind<CAT>>{std::move(x)}} {}

  std::optional<Scalar<Result>> ScalarValue() const;
  std::ostream &Dump(std::ostream &) const;
  std::optional<Scalar<Result>> Fold(FoldingContext &);
  int Rank() const;

  std::variant<Expr<SomeInteger>, Expr<SomeReal>, Expr<SomeComplex>,
      Expr<SomeCharacter>, Expr<SomeLogical>, BOZLiteralConstant>
      u;
};

using GenericExpr = Expr<SomeType>;  // TODO: delete name?

template<typename A> using ResultType = typename std::decay_t<A>::Result;

// Convenience functions and operator overloadings for expression construction.

template<TypeCategory C, int K>
Expr<Type<C, K>> operator-(const Expr<Type<C, K>> &x) {
  return {Negate<Type<C, K>>{x}};
}
template<TypeCategory C>
Expr<SomeKind<C>> operator-(const Expr<SomeKind<C>> &x) {
  return std::visit(
      [](const auto &y) -> Expr<SomeKind<C>> { return {-y}; }, x.u);
}

#define BINARY(op, CONSTR) \
  template<TypeCategory C, int K> \
  Expr<Type<C, K>> operator op( \
      const Expr<Type<C, K>> &x, const Expr<Type<C, K>> &y) { \
    return {CONSTR<Type<C, K>>{x, y}}; \
  } \
  template<TypeCategory C> \
  Expr<SomeKind<C>> operator op( \
      const Expr<SomeKind<C>> &x, const Expr<SomeKind<C>> &y) { \
    return std::visit( \
        [](const auto &xk, const auto &yk) -> Expr<SomeKind<C>> { \
          return {xk op yk}; \
        }, \
        x.u, y.u); \
  }

BINARY(+, Add)
BINARY(-, Subtract)
BINARY(*, Multiply)
BINARY(/, Divide)
#undef BINARY

#if 0
#define OLDBINARY(FUNC, CONSTR) \
  template<typename A> A FUNC(const A &x, const A &y) { \
    return {CONSTR<typename A::Result>{x, y}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, A> FUNC(const A &x, A &&y) { \
    return {CONSTR<typename A::Result>{A{x}, std::move(y)}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, A> FUNC(A &&x, const A &y) { \
    return {CONSTR<typename A::Result>{std::move(x), A{y}}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, A> FUNC(A &&x, A &&y) { \
    return {CONSTR<typename A::Result>{std::move(x), std::move(y)}}; \
  }
#undef OLDBINARY
#endif

#define BINARY(FUNC, OP) \
  template<typename A> Expr<LogicalResult> FUNC(const A &x, const A &y) { \
    return {Relational<ResultType<A>>{OP, x, y}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, Expr<LogicalResult>> FUNC( \
      const A &x, A &&y) { \
    return {Relational<ResultType<A>>{OP, x, std::move(y)}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, Expr<LogicalResult>> FUNC( \
      A &&x, const A &y) { \
    return {Relational<ResultType<A>>{OP, std::move(x), y}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, Expr<LogicalResult>> FUNC( \
      A &&x, A &&y) { \
    return {Relational<ResultType<A>>{OP, std::move(x), std::move(y)}}; \
  }

BINARY(operator<, RelationalOperator::LT)
BINARY(operator<=, RelationalOperator::LE)
BINARY(operator==, RelationalOperator::EQ)
BINARY(operator!=, RelationalOperator::NE)
BINARY(operator>=, RelationalOperator::GE)
BINARY(operator>, RelationalOperator::GT)
#undef BINARY

extern template class Expr<Type<TypeCategory::Character, 1>>;  // TODO others
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
extern template class Expr<Type<TypeCategory::Logical, 1>>;
extern template class Expr<Type<TypeCategory::Logical, 2>>;
extern template class Expr<Type<TypeCategory::Logical, 4>>;
extern template class Expr<Type<TypeCategory::Logical, 8>>;
extern template class Expr<SomeInteger>;
extern template class Expr<SomeReal>;
extern template class Expr<SomeComplex>;
extern template class Expr<SomeCharacter>;
extern template class Expr<SomeLogical>;
extern template class Expr<SomeType>;

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_EXPRESSION_H_
