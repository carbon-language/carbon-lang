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
// context-independent hash table or sharing of common subexpressions, and
// thus these are trees, not DAGs.  Both deep copy and move semantics are
// supported for expression construction.

#include "common.h"
#include "type.h"
#include "variable.h"
#include "../lib/common/fortran.h"
#include "../lib/common/idioms.h"
#include "../lib/common/template.h"
#include "../lib/parser/char-block.h"
#include "../lib/parser/message.h"
#include <ostream>
#include <tuple>
#include <type_traits>
#include <variant>

namespace Fortran::evaluate {

using common::RelationalOperator;

// Expressions are represented by specializations of the class template Expr.
// Each of these specializations wraps a single data member "u" that
// is a std::variant<> discriminated union over all of the representational
// types for the constants, variables, operations, and other entities that
// can be valid expressions in that context:
// - Expr<Type<CATEGORY, KIND>> represents an expression whose result is of a
//   specific intrinsic type category and kind, e.g. Type<TypeCategory::Real, 4>
// - Expr<SomeKind<CATEGORY>> is a union of Expr<Type<CATEGORY, K>> for each
//   kind type parameter value K in that intrinsic type category.  It represents
//   an expression with known category and any kind.
// - Expr<SomeType> is a union of Expr<SomeKind<CATEGORY>> over the five
//   intrinsic type categories of Fortran.  It represents any valid expression.
template<typename A> class Expr;

// Everything that can appear in, or as, a valid Fortran expression must be
// represented with an instance of some class containing a Result typedef that
// maps to some instantiation of Type<CATEGORY, KIND>, SomeKind<CATEGORY>,
// or SomeType.
template<typename A> using ResultType = typename std::decay_t<A>::Result;

// Wraps a constant value in a class with its resolved type.
template<typename T> struct Constant {
  using Result = T;
  using Value = Scalar<Result>;
  CLASS_BOILERPLATE(Constant)
  template<typename A> Constant(const A &x) : value{x} {}
  template<typename A>
  Constant(std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : value(std::move(x)) {}
  int Rank() const { return 0; }
  std::ostream &Dump(std::ostream &) const;
  Value value;
};

// BOZ literal "typeless" constants must be wide enough to hold a numeric
// value of any supported kind of INTEGER or REAL.  They must also be
// distinguishable from other integer constants, since they are permitted
// to be used in only a few situations.
using BOZLiteralConstant = typename LargestReal::Scalar::Word;

template<typename T> struct FunctionReference {
  using Result = T;
  static_assert(Result::isSpecificType);
  int Rank() const { return reference->Rank(); }
  CopyableIndirection<FunctionRef> reference;
};

// Operations always have specific Fortran result types (i.e., with known
// intrinsic type category and kind parameter value).  The classes that
// represent the operations all inherit from this Operation<> base class
// template.  Note that Operation has as its first type parameter (DERIVED) a
// "curiously reoccurring template pattern (CRTP)" reference to the specific
// operation class being derived from Operation; e.g., Add is defined with
// struct Add : public Operation<Add, ...>.  Uses of instances of Operation<>,
// including its own member functions, can access each specific class derived
// from it via its derived() member function with compile-time type safety.
template<typename DERIVED, typename RESULT, typename... OPERANDS>
class Operation {
  static_assert(RESULT::isSpecificType);
  // The extra "int" member is a dummy that allows a safe unused reference
  // to element 1 to arise indirectly in the definition of "right()" below
  // when the operation has but a single operand.
  using OperandTypes = std::tuple<OPERANDS..., int>;

public:
  using Derived = DERIVED;
  using Result = RESULT;
  static constexpr std::size_t operands{sizeof...(OPERANDS)};
  template<int J> using Operand = std::tuple_element_t<J, OperandTypes>;
  using IsFoldableTrait = std::true_type;

  // Unary operations wrap a single Expr with a CopyableIndirection.
  // Binary operations wrap a tuple of CopyableIndirections to Exprs.
private:
  using Container =
      std::conditional_t<operands == 1, CopyableIndirection<Expr<Operand<0>>>,
          std::tuple<CopyableIndirection<Expr<OPERANDS>>...>>;

public:
  CLASS_BOILERPLATE(Operation)
  explicit Operation(const Expr<OPERANDS> &... x) : operand_{x...} {}
  explicit Operation(Expr<OPERANDS> &&... x)
    : operand_{std::forward<Expr<OPERANDS>>(x)...} {}

  Derived &derived() { return *static_cast<Derived *>(this); }
  const Derived &derived() const { return *static_cast<const Derived *>(this); }

  // References to operand expressions from member functions of derived
  // classes for specific operators can be made by index, e.g. operand<0>(),
  // which must be spelled like "this->template operand<0>()" when
  // inherited in a derived class template.  There are convenience aliases
  // left() and right() that are not templates.
  template<int J> Expr<Operand<J>> &operand() {
    if constexpr (operands == 1) {
      static_assert(J == 0);
      return *operand_;
    } else {
      return *std::get<J>(operand_);
    }
  }
  template<int J> const Expr<Operand<J>> &operand() const {
    if constexpr (operands == 1) {
      static_assert(J == 0);
      return *operand_;
    } else {
      return *std::get<J>(operand_);
    }
  }

  Expr<Operand<0>> &left() { return operand<0>(); }
  const Expr<Operand<0>> &left() const { return operand<0>(); }

  std::conditional_t<(operands > 1), Expr<Operand<1>> &, void> right() {
    if constexpr (operands > 1) {
      return operand<1>();
    }
  }
  std::conditional_t<(operands > 1), const Expr<Operand<1>> &, void>
  right() const {
    if constexpr (operands > 1) {
      return operand<1>();
    }
  }

  int Rank() const {
    int rank{left().Rank()};
    if constexpr (operands > 1) {
      int rightRank{right().Rank()};
      if (rightRank > rank) {
        rank = rightRank;
      }
    }
    return rank;
  }

  std::ostream &Dump(std::ostream &) const;
  std::optional<Constant<Result>> Fold(FoldingContext &);

protected:
  // Overridable functions for Dump()
  static std::ostream &Prefix(std::ostream &o) { return o << '('; }
  static std::ostream &Infix(std::ostream &o) { return o << ','; }
  static std::ostream &Suffix(std::ostream &o) { return o << ')'; }

private:
  Container operand_;
};

// Unary operations

// Conversions to specific types from expressions of known category and
// dynamic kind.
template<typename TO, TypeCategory FROMCAT>
struct Convert : public Operation<Convert<TO, FROMCAT>, TO, SomeKind<FROMCAT>> {
  // Fortran doesn't have conversions between kinds of CHARACTER.
  // Conversions between kinds of COMPLEX are represented piecewise.
  static_assert(((TO::category == TypeCategory::Integer ||
                     TO::category == TypeCategory::Real) &&
                    (FROMCAT == TypeCategory::Integer ||
                        FROMCAT == TypeCategory::Real)) ||
      (TO::category == TypeCategory::Logical &&
          FROMCAT == TypeCategory::Logical));
  using Result = TO;
  using Operand = SomeKind<FROMCAT>;
  using Base = Operation<Convert, Result, Operand>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &);
  std::ostream &Dump(std::ostream &) const;
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
  static std::ostream &Prefix(std::ostream &o) { return o << "(-"; }
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
  std::ostream &Suffix(std::ostream &o) const {
    return o << (isImaginaryPart ? "%IM)" : "%RE)");
  }

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
  static std::ostream &Prefix(std::ostream &o) { return o << "(.NOT."; }
};

// Binary operations

template<typename A> struct Add : public Operation<Add<A>, A, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Add, A, A, A>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
  static std::ostream &Infix(std::ostream &o) { return o << '+'; }
};

template<typename A> struct Subtract : public Operation<Subtract<A>, A, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Subtract, A, A, A>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
  static std::ostream &Infix(std::ostream &o) { return o << '-'; }
};

template<typename A> struct Multiply : public Operation<Multiply<A>, A, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Multiply, A, A, A>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
  static std::ostream &Infix(std::ostream &o) { return o << '*'; }
};

template<typename A> struct Divide : public Operation<Divide<A>, A, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Divide, A, A, A>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
  static std::ostream &Infix(std::ostream &o) { return o << '/'; }
};

template<typename A> struct Power : public Operation<Power<A>, A, A, A> {
  using Result = A;
  using Operand = A;
  using Base = Operation<Power, A, A, A>;
  using Base::Base;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
  static std::ostream &Infix(std::ostream &o) { return o << "**"; }
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
  static std::ostream &Infix(std::ostream &o) { return o << "**"; }
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
  std::ostream &Prefix(std::ostream &o) const {
    return o << (ordering == Ordering::Less ? "MIN(" : "MAX(");
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
  static std::ostream &Infix(std::ostream &o) { return o << "//"; }
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
      LogicalOperator opr, const Expr<Operand> &x, const Expr<Operand> &y)
    : Base{x, y}, logicalOperator{opr} {}
  LogicalOperation(LogicalOperator opr, Expr<Operand> &&x, Expr<Operand> &&y)
    : Base{std::move(x), std::move(y)}, logicalOperator{opr} {}

  std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &) const;
  std::ostream &Infix(std::ostream &) const;

  LogicalOperator logicalOperator;
};

// Per-category expression representations

// Common Expr<> behaviors
template<typename RESULT> struct ExpressionBase {
  using Result = RESULT;
  using Derived = Expr<Result>;

  Derived &derived() { return *static_cast<Derived *>(this); }
  const Derived &derived() const { return *static_cast<const Derived *>(this); }

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

  int Rank() const;
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

  EVALUATE_UNION_CLASS_BOILERPLATE(Expr)
  explicit Expr(const Scalar<Result> &x) : u{Constant<Result>{x}} {}
  explicit Expr(std::int64_t n) : u{Constant<Result>{n}} {}
  explicit Expr(std::uint64_t n) : u{Constant<Result>{n}} {}
  explicit Expr(int n) : u{Constant<Result>{n}} {}

private:
  using Conversions = std::variant<Convert<Result, TypeCategory::Integer>,
      Convert<Result, TypeCategory::Real>>;
  using Operations = std::variant<Parentheses<Result>, Negate<Result>,
      Add<Result>, Subtract<Result>, Multiply<Result>, Divide<Result>,
      Power<Result>, Extremum<Result>>;
  using Others = std::variant<Constant<Result>, Designator<Result>,
      FunctionReference<Result>>;

public:
  common::CombineVariants<Operations, Conversions, Others> u;
};

template<int KIND>
class Expr<Type<TypeCategory::Real, KIND>>
  : public ExpressionBase<Type<TypeCategory::Real, KIND>> {
public:
  using Result = Type<TypeCategory::Real, KIND>;
  using IsFoldableTrait = std::true_type;

  EVALUATE_UNION_CLASS_BOILERPLATE(Expr)
  explicit Expr(const Scalar<Result> &x) : u{Constant<Result>{x}} {}

private:
  // N.B. Real->Complex and Complex->Real conversions are done with CMPLX
  // and part access operations (resp.).  Conversions between kinds of
  // Complex are done via decomposition to Real and reconstruction.
  using Conversions = std::variant<Convert<Result, TypeCategory::Integer>,
      Convert<Result, TypeCategory::Real>>;
  using Operations = std::variant<ComplexComponent<KIND>, Parentheses<Result>,
      Negate<Result>, Add<Result>, Subtract<Result>, Multiply<Result>,
      Divide<Result>, Power<Result>, RealToIntPower<Result>, Extremum<Result>>;
  using Others = std::variant<Constant<Result>, Designator<Result>,
      FunctionReference<Result>>;

public:
  common::CombineVariants<Operations, Conversions, Others> u;
};

template<int KIND>
class Expr<Type<TypeCategory::Complex, KIND>>
  : public ExpressionBase<Type<TypeCategory::Complex, KIND>> {
public:
  using Result = Type<TypeCategory::Complex, KIND>;
  using IsFoldableTrait = std::true_type;
  EVALUATE_UNION_CLASS_BOILERPLATE(Expr)
  explicit Expr(const Scalar<Result> &x) : u{Constant<Result>{x}} {}

  // Note that many COMPLEX operations are represented as REAL operations
  // over their components (viz., conversions, negation, add, and subtract).
  using Operations =
      std::variant<Parentheses<Result>, Multiply<Result>, Divide<Result>,
          Power<Result>, RealToIntPower<Result>, ComplexConstructor<KIND>>;
  using Others = std::variant<Constant<Result>, Designator<Result>,
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
  EVALUATE_UNION_CLASS_BOILERPLATE(Expr)
  explicit Expr(const Scalar<Result> &x) : u{Constant<Result>{x}} {}
  explicit Expr(Scalar<Result> &&x) : u{Constant<Result>{std::move(x)}} {}

  Expr<SubscriptInteger> LEN() const;

  std::variant<Constant<Result>, Designator<Result>, FunctionReference<Result>,
      Parentheses<Result>, Concat<KIND>, Extremum<Result>>
      u;
};

extern template class Expr<Type<TypeCategory::Character, 1>>;
extern template class Expr<Type<TypeCategory::Character, 2>>;
extern template class Expr<Type<TypeCategory::Character, 4>>;

// The Relational class template is a helper for constructing logical
// expressions with polymorphism over the cross product of the possible
// categories and kinds of comparable operands.
// Fortran defines a numeric relation with distinct types or kinds as
// first undergoing the same operand conversions that occur with the intrinsic
// addition operator.  Character relations must have the same kind.
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
  std::ostream &Infix(std::ostream &) const;

  RelationalOperator opr;
};

template<> class Relational<SomeType> {
  // COMPLEX data are compared piecewise.
  using DirectlyComparableTypes =
      common::CombineTuples<IntegerTypes, RealTypes, CharacterTypes>;

public:
  using Result = LogicalResult;
  EVALUATE_UNION_CLASS_BOILERPLATE(Relational)
  int Rank() const {
    return std::visit([](const auto &x) { return x.Rank(); }, u);
  }
  std::ostream &Dump(std::ostream &o) const;
  common::MapTemplate<Relational, DirectlyComparableTypes> u;
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
extern template struct Relational<Type<TypeCategory::Character, 1>>;
extern template struct Relational<Type<TypeCategory::Character, 2>>;
extern template struct Relational<Type<TypeCategory::Character, 4>>;
extern template struct Relational<SomeType>;

template<int KIND>
class Expr<Type<TypeCategory::Logical, KIND>>
  : public ExpressionBase<Type<TypeCategory::Logical, KIND>> {
public:
  using Result = Type<TypeCategory::Logical, KIND>;
  using IsFoldableTrait = std::true_type;
  EVALUATE_UNION_CLASS_BOILERPLATE(Expr)
  explicit Expr(const Scalar<Result> &x) : u{Constant<Result>{x}} {}
  explicit Expr(bool x) : u{Constant<Result>{x}} {}

private:
  using Operations =
      std::variant<Convert<Result, TypeCategory::Logical>, Parentheses<Result>,
          Not<KIND>, LogicalOperation<KIND>, Relational<SomeType>>;
  using Others = std::variant<Constant<Result>, Designator<Result>,
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
  EVALUATE_UNION_CLASS_BOILERPLATE(Expr)
  common::MapTemplate<Expr, CategoryTypes<CAT>> u;
};

template<> class Expr<SomeDerived> : public ExpressionBase<SomeDerived> {
public:
  using Result = SomeDerived;
  using IsFoldableTrait = std::false_type;
  CLASS_BOILERPLATE(Expr)

  template<typename A>
  explicit Expr(const semantics::DerivedTypeSpec &dts, const A &x)
    : result{dts}, u{x} {}
  template<typename A>
  explicit Expr(Result &&r, std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : result{std::move(r)}, u{std::move(x)} {}

  Result result;
  std::variant<Designator<Result>, FunctionReference<Result>> u;
};

// A completely generic expression, polymorphic across all of the intrinsic type
// categories and each of their kinds.
template<> class Expr<SomeType> : public ExpressionBase<SomeType> {
public:
  using Result = SomeType;
  using IsFoldableTrait = std::true_type;
  EVALUATE_UNION_CLASS_BOILERPLATE(Expr)

  // Owning references to these generic expressions can appear in other
  // compiler data structures (viz., the parse tree and symbol table), so
  // its destructor is externalized to reduce redundant default instances.
  ~Expr();

  template<TypeCategory CAT, int KIND>
  explicit Expr(const Expr<Type<CAT, KIND>> &x) : u{Expr<SomeKind<CAT>>{x}} {}

  template<TypeCategory CAT, int KIND>
  explicit Expr(Expr<Type<CAT, KIND>> &&x)
    : u{Expr<SomeKind<CAT>>{std::move(x)}} {}

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

private:
  using Others = std::variant<BOZLiteralConstant>;
  using Categories = common::MapTemplate<Expr, SomeCategory>;

public:
  common::CombineVariants<Others, Categories> u;
};

// This wrapper class is used, by means of a forward reference with
// OwningPointer, to implement owning pointers to analyzed expressions
// from parse tree nodes.
struct GenericExprWrapper {
  GenericExprWrapper(Expr<SomeType> &&x) : v{std::move(x)} {}
  Expr<SomeType> v;
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
extern template struct ExpressionBase<Type<TypeCategory::Character, 2>>;
extern template struct ExpressionBase<Type<TypeCategory::Character, 4>>;
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
