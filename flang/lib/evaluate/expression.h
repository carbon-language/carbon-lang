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
  static const char *prefix() { return "(("; }
  const char *suffix() const { return isRealPart ? "%RE)" : "%IM)"; }

  bool isRealPart{true};
};

template<int KIND>
struct Not : public Operation<Not<KIND>, Type<TypeCategory::Logical, KIND>,
                 Type<TypeCategory::Logical, KIND>> {
  using Base = Operation<Not, Type<TypeCategory::Logical, KIND>,
      Type<TypeCategory::Logical, KIND>>;
  using Base::Base;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &);
  static const char *prefix() { return "(.NOT."; }
};

// Binary operations

template<typename A> struct Add : public Operation<Add<A>, A, A, A> {
  using Base = Operation<Add, A, A, A>;
  using Base::Base;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
  static std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &, const Scalar<Operand> &, const Scalar<Operand> &);
  static const char *infix() { return "+"; }
};

template<typename CRTP, typename RESULT, typename A = RESULT, typename B = A>
class Binary {
public:
  using Result = RESULT;
  using Left = A;
  using Right = B;
  using FoldableTrait = std::true_type;
  CLASS_BOILERPLATE(Binary)
  Binary(const Expr<Left> &a, const Expr<Right> &b) : left_{a}, right_{b} {}
  Binary(Expr<Left> &&a, Expr<Right> &&b)
    : left_{std::move(a)}, right_{std::move(b)} {}
  Binary(CopyableIndirection<const Expr<Left>> &&a,
      CopyableIndirection<const Expr<Right>> &&b)
    : left_{std::move(a)}, right_{std::move(b)} {}
  const Expr<Left> &left() const { return *left_; }
  Expr<Left> &left() { return *left_; }
  const Expr<Right> &right() const { return *right_; }
  Expr<Right> &right() { return *right_; }
  std::ostream &Dump(
      std::ostream &, const char *opr, const char *before = "(") const;
  int Rank() const;
  std::optional<Scalar<Result>> Fold(FoldingContext &);

private:
  CopyableIndirection<Expr<Left>> left_;
  CopyableIndirection<Expr<Right>> right_;
};

// Per-category expressions

template<int KIND> class Expr<Type<TypeCategory::Integer, KIND>> {
public:
  using Result = Type<TypeCategory::Integer, KIND>;
  using FoldableTrait = std::true_type;

  template<typename CRTP> using Bin = Binary<CRTP, Result>;
  struct Subtract : public Bin<Subtract> {
    using Bin<Subtract>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct Multiply : public Bin<Multiply> {
    using Bin<Multiply>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct Divide : public Bin<Divide> {
    using Bin<Divide>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct Power : public Bin<Power> {
    using Bin<Power>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct Max : public Bin<Max> {
    using Bin<Max>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct Min : public Bin<Min> {
    using Bin<Min>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
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
      Add<Result>, Subtract, Multiply, Divide, Power, Max, Min>
      u_;
};

template<int KIND> class Expr<Type<TypeCategory::Real, KIND>> {
public:
  using Result = Type<TypeCategory::Real, KIND>;
  using FoldableTrait = std::true_type;

  // N.B. Real->Complex and Complex->Real conversions are done with CMPLX
  // and part access operations (resp.).  Conversions between kinds of
  // Complex are done via decomposition to Real and reconstruction.

  template<typename CRTP> using Bin = Binary<CRTP, Result>;
  struct Subtract : public Bin<Subtract> {
    using Bin<Subtract>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct Multiply : public Bin<Multiply> {
    using Bin<Multiply>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct Divide : public Bin<Divide> {
    using Bin<Divide>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct Power : public Bin<Power> {
    using Bin<Power>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct IntPower : public Binary<IntPower, Result, Result, SomeInteger> {
    using Binary<IntPower, Result, Result, SomeInteger>::Binary;
    static std::optional<Scalar<Result>> FoldScalar(FoldingContext &,
        const Scalar<Result> &, const SomeKindScalar<TypeCategory::Integer> &);
  };
  struct Max : public Bin<Max> {
    using Bin<Max>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct Min : public Bin<Min> {
    using Bin<Min>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };

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
      Subtract, Multiply, Divide, Power, IntPower, Max, Min>
      u_;
};

template<int KIND> class Expr<Type<TypeCategory::Complex, KIND>> {
public:
  using Result = Type<TypeCategory::Complex, KIND>;
  using FoldableTrait = std::true_type;

  template<typename CRTP> using Bin = Binary<CRTP, Result>;
  struct Subtract : public Bin<Subtract> {
    using Bin<Subtract>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct Multiply : public Bin<Multiply> {
    using Bin<Multiply>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct Divide : public Bin<Divide> {
    using Bin<Divide>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct Power : public Bin<Power> {
    using Bin<Power>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct IntPower : public Binary<IntPower, Result, Result, SomeInteger> {
    using Binary<IntPower, Result, Result, SomeInteger>::Binary;
    static std::optional<Scalar<Result>> FoldScalar(FoldingContext &,
        const Scalar<Result> &, const SomeKindScalar<TypeCategory::Integer> &);
  };
  struct CMPLX
    : public Binary<CMPLX, Result, SameKind<TypeCategory::Real, Result>> {
    using Binary<CMPLX, Result, SameKind<TypeCategory::Real, Result>>::Binary;
    static std::optional<Scalar<Result>> FoldScalar(FoldingContext &,
        const Scalar<SameKind<TypeCategory::Real, Result>> &,
        const Scalar<SameKind<TypeCategory::Real, Result>> &);
  };

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
      Add<Result>, Subtract, Multiply, Divide, Power, IntPower, CMPLX>
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
  template<typename CRTP> using Bin = Binary<CRTP, Result>;
  struct Concat : public Bin<Concat> {
    using Bin<Concat>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct Max : public Bin<Max> {
    using Bin<Max>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct Min : public Bin<Min> {
    using Bin<Min>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };

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
      //      Parentheses<Result>,
      Concat, Max, Min>
      u_;
};

// The Comparison class template is a helper for constructing logical
// expressions with polymorphism over the cross product of the possible
// categories and kinds of comparable operands.

template<typename A>
struct Comparison : public Operation<Comparison<A>, LogicalResult, A, A> {
  using Base = Operation<Comparison, LogicalResult, A, A>;
  using typename Base::Result;
  using Operand = typename Base::template Operand<0>;
  CLASS_BOILERPLATE(Comparison)
  Comparison(
      RelationalOperator r, const Expr<Operand> &a, const Expr<Operand> &b)
    : Base{a, b}, opr{r} {}
  Comparison(RelationalOperator r, Expr<Operand> &&a, Expr<Operand> &&b)
    : Base{std::move(a), std::move(b)}, opr{r} {}

  std::optional<Scalar<Result>> FoldScalar(
      FoldingContext &c, const Scalar<Operand> &, const Scalar<Operand> &);
  std::string infix() const;

  RelationalOperator opr;
};

// Dynamically polymorphic comparisons whose operands are expressions of
// the same supported kind of a particular type category.
template<TypeCategory CAT> struct CategoryComparison {
  using Result = LogicalResult;
  CLASS_BOILERPLATE(CategoryComparison)
  template<int KIND> using KindComparison = Comparison<Type<CAT, KIND>>;
  template<int KIND> CategoryComparison(const KindComparison<KIND> &x) : u{x} {}
  template<int KIND>
  CategoryComparison(KindComparison<KIND> &&x) : u{std::move(x)} {}
  std::ostream &Dump(std::ostream &) const;
  std::optional<Scalar<Result>> Fold(FoldingContext &c);
  int Rank() const { return 1; }  // TODO

  KindsVariant<CAT, KindComparison> u;
};

template<int KIND> class Expr<Type<TypeCategory::Logical, KIND>> {
public:
  using Result = Type<TypeCategory::Logical, KIND>;
  using FoldableTrait = std::true_type;
  template<typename CRTP> using Bin = Binary<CRTP, Result>;
  struct And : public Bin<And> {
    using Bin<And>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct Or : public Bin<Or> {
    using Bin<Or>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct Eqv : public Bin<Eqv> {
    using Bin<Eqv>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };
  struct Neqv : public Bin<Neqv> {
    using Bin<Neqv>::Bin;
    static std::optional<Scalar<Result>> FoldScalar(
        FoldingContext &, const Scalar<Result> &, const Scalar<Result> &);
  };

  CLASS_BOILERPLATE(Expr)
  Expr(const Scalar<Result> &x) : u_{x} {}
  Expr(bool x) : u_{Scalar<Result>{x}} {}
  template<TypeCategory CAT, int K>
  Expr(const Comparison<Type<CAT, K>> &x) : u_{CategoryComparison<CAT>{x}} {}
  template<TypeCategory CAT, int K>
  Expr(Comparison<Type<CAT, K>> &&x)
    : u_{CategoryComparison<CAT>{std::move(x)}} {}
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
      //      Parentheses<Result>,
      Not<KIND>, And, Or, Eqv, Neqv, CategoryComparison<TypeCategory::Integer>,
      CategoryComparison<TypeCategory::Real>,
      CategoryComparison<TypeCategory::Complex>,
      CategoryComparison<TypeCategory::Character>>
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
#undef BINARY

#define OLDBINARY(FUNC, CONSTR) \
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

OLDBINARY(operator-, Subtract)
OLDBINARY(operator*, Multiply)
OLDBINARY(operator/, Divide)
OLDBINARY(Power, Power)
#undef OLDBINARY

#define BINARY(FUNC, OP) \
  template<typename A> Expr<LogicalResult> FUNC(const A &x, const A &y) { \
    return {Comparison<ResultType<A>>{OP, x, y}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, Expr<LogicalResult>> FUNC( \
      const A &x, A &&y) { \
    return {Comparison<ResultType<A>>{OP, x, std::move(y)}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, Expr<LogicalResult>> FUNC( \
      A &&x, const A &y) { \
    return {Comparison<ResultType<A>>{OP, std::move(x), y}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, Expr<LogicalResult>> FUNC( \
      A &&x, A &&y) { \
    return {Comparison<ResultType<A>>{OP, std::move(x), std::move(y)}}; \
  }

BINARY(operator<, RelationalOperator::LT)
BINARY(operator<=, RelationalOperator::LE)
BINARY(operator==, RelationalOperator::EQ)
BINARY(operator!=, RelationalOperator::NE)
BINARY(operator>=, RelationalOperator::GE)
BINARY(operator>, RelationalOperator::GT)
#undef BINARY

extern template class Expr<Type<TypeCategory::Character, 1>>;  // TODO others
extern template struct Comparison<Type<TypeCategory::Integer, 1>>;
extern template struct Comparison<Type<TypeCategory::Integer, 2>>;
extern template struct Comparison<Type<TypeCategory::Integer, 4>>;
extern template struct Comparison<Type<TypeCategory::Integer, 8>>;
extern template struct Comparison<Type<TypeCategory::Integer, 16>>;
extern template struct Comparison<Type<TypeCategory::Real, 2>>;
extern template struct Comparison<Type<TypeCategory::Real, 4>>;
extern template struct Comparison<Type<TypeCategory::Real, 8>>;
extern template struct Comparison<Type<TypeCategory::Real, 10>>;
extern template struct Comparison<Type<TypeCategory::Real, 16>>;
extern template struct Comparison<Type<TypeCategory::Complex, 2>>;
extern template struct Comparison<Type<TypeCategory::Complex, 4>>;
extern template struct Comparison<Type<TypeCategory::Complex, 8>>;
extern template struct Comparison<Type<TypeCategory::Complex, 10>>;
extern template struct Comparison<Type<TypeCategory::Complex, 16>>;
extern template struct Comparison<Type<TypeCategory::Character, 1>>;  // TODO
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
