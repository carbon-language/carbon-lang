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
#include "expression-forward.h"
#include "type.h"
#include "variable.h"
#include "../lib/common/idioms.h"
#include "../lib/parser/char-block.h"
#include "../lib/parser/message.h"
#include <ostream>
#include <variant>

namespace Fortran::evaluate {

// Helper base classes for packaging subexpressions.
template<typename CRTP, typename RESULT, typename A = RESULT> class Unary {
protected:
  using OperandType = A;
  using Operand = Expr<OperandType>;
  using OperandScalarConstant = Scalar<OperandType>;

public:
  using Result = RESULT;
  using Scalar = Scalar<Result>;
  using FoldableTrait = std::true_type;
  CLASS_BOILERPLATE(Unary)
  Unary(const Operand &a) : operand_{a} {}
  Unary(Operand &&a) : operand_{std::move(a)} {}
  Unary(CopyableIndirection<Operand> &&a) : operand_{std::move(a)} {}
  const Operand &operand() const { return *operand_; }
  Operand &operand() { return *operand_; }
  std::ostream &Dump(std::ostream &, const char *opr) const;
  int Rank() const { return operand_.Rank(); }
  std::optional<Scalar> Fold(FoldingContext &);  // TODO: array result
private:
  CopyableIndirection<Operand> operand_;
};

template<typename CRTP, typename RESULT, typename A = RESULT, typename B = A>
class Binary {
protected:
  using LeftType = A;
  using Left = Expr<LeftType>;
  using LeftScalar = Scalar<LeftType>;
  using RightType = B;
  using Right = Expr<RightType>;
  using RightScalar = Scalar<RightType>;

public:
  using Result = RESULT;
  using Scalar = Scalar<Result>;
  using FoldableTrait = std::true_type;
  CLASS_BOILERPLATE(Binary)
  Binary(const Left &a, const Right &b) : left_{a}, right_{b} {}
  Binary(Left &&a, Right &&b) : left_{std::move(a)}, right_{std::move(b)} {}
  Binary(
      CopyableIndirection<const Left> &&a, CopyableIndirection<const Right> &&b)
    : left_{std::move(a)}, right_{std::move(b)} {}
  const Left &left() const { return *left_; }
  Left &left() { return *left_; }
  const Right &right() const { return *right_; }
  Right &right() { return *right_; }
  std::ostream &Dump(
      std::ostream &, const char *opr, const char *before = "(") const;
  int Rank() const;
  std::optional<Scalar> Fold(FoldingContext &);

private:
  CopyableIndirection<Left> left_;
  CopyableIndirection<Right> right_;
};

// Per-category expressions

template<int KIND> class Expr<Type<TypeCategory::Integer, KIND>> {
public:
  using Result = Type<TypeCategory::Integer, KIND>;
  using Scalar = Scalar<Result>;
  using FoldableTrait = std::true_type;

  struct ConvertInteger
    : public Unary<ConvertInteger, Result, SomeKind<TypeCategory::Integer>> {
    using Unary<ConvertInteger, Result, SomeKind<TypeCategory::Integer>>::Unary;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const SomeKindScalar<TypeCategory::Integer> &);
  };

  struct ConvertReal
    : public Unary<ConvertReal, Result, SomeKind<TypeCategory::Real>> {
    using Unary<ConvertReal, Result, SomeKind<TypeCategory::Real>>::Unary;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const SomeKindScalar<TypeCategory::Real> &);
  };

  template<typename CRTP> using Un = Unary<CRTP, Result>;
  template<typename CRTP> using Bin = Binary<CRTP, Result>;
  struct Parentheses : public Un<Parentheses> {
    using Un<Parentheses>::Un;
    static std::optional<Scalar> FoldScalar(FoldingContext &, const Scalar &x) {
      return {x};
    }
  };
  struct Negate : public Un<Negate> {
    using Un<Negate>::Un;
    static std::optional<Scalar> FoldScalar(FoldingContext &, const Scalar &);
  };
  struct Add : public Bin<Add> {
    using Bin<Add>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Subtract : public Bin<Subtract> {
    using Bin<Subtract>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Multiply : public Bin<Multiply> {
    using Bin<Multiply>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Divide : public Bin<Divide> {
    using Bin<Divide>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Power : public Bin<Power> {
    using Bin<Power>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Max : public Bin<Max> {
    using Bin<Max>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Min : public Bin<Min> {
    using Bin<Min>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  // TODO: R916 type-param-inquiry

  CLASS_BOILERPLATE(Expr)
  Expr(const Scalar &x) : u_{x} {}
  Expr(std::int64_t n) : u_{Scalar{n}} {}
  Expr(std::uint64_t n) : u_{Scalar{n}} {}
  Expr(int n) : u_{Scalar{n}} {}
  Expr(const SomeKindIntegerExpr &x) : u_{ConvertInteger{x}} {}
  Expr(SomeKindIntegerExpr &&x) : u_{ConvertInteger{std::move(x)}} {}
  template<int K>
  Expr(const IntegerExpr<K> &x) : u_{ConvertInteger{SomeKindIntegerExpr{x}}} {}
  template<int K>
  Expr(IntegerExpr<K> &&x)
    : u_{ConvertInteger{SomeKindIntegerExpr{std::move(x)}}} {}
  Expr(const SomeKindRealExpr &x) : u_{ConvertReal{x}} {}
  Expr(SomeKindRealExpr &&x) : u_{ConvertReal{std::move(x)}} {}
  template<int K>
  Expr(const RealExpr<K> &x) : u_{ConvertReal{SomeKindRealExpr{x}}} {}
  template<int K>
  Expr(RealExpr<K> &&x) : u_{ConvertReal{SomeKindRealExpr{std::move(x)}}} {}
  template<typename A> Expr(const A &x) : u_{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A> &&
          (std::is_base_of_v<Un, A> || std::is_base_of_v<Bin, A>),
      A> &&x)
    : u_(std::move(x)) {}
  template<typename A> Expr(CopyableIndirection<A> &&x) : u_{std::move(x)} {}

  std::optional<Scalar> ScalarValue() const {
    return common::GetIf<Scalar>(u_);
  }
  std::optional<Scalar> Fold(FoldingContext &c);

private:
  std::variant<Scalar, CopyableIndirection<DataRef>,
      CopyableIndirection<FunctionRef>, ConvertInteger, ConvertReal,
      Parentheses, Negate, Add, Subtract, Multiply, Divide, Power, Max, Min>
      u_;
};

template<int KIND> class Expr<Type<TypeCategory::Real, KIND>> {
public:
  using Result = Type<TypeCategory::Real, KIND>;
  using Scalar = Scalar<Result>;
  using FoldableTrait = std::true_type;
  using Complex = typename Result::Complex;

  // N.B. Real->Complex and Complex->Real conversions are done with CMPLX
  // and part access operations (resp.).  Conversions between kinds of
  // Complex are done via decomposition to Real and reconstruction.
  struct ConvertInteger
    : public Unary<ConvertInteger, Result, SomeKind<TypeCategory::Integer>> {
    using Unary<ConvertInteger, Result, SomeKind<TypeCategory::Integer>>::Unary;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const SomeKindScalar<TypeCategory::Integer> &);
  };
  struct ConvertReal
    : public Unary<ConvertReal, Result, SomeKind<TypeCategory::Real>> {
    using Unary<ConvertReal, Result, SomeKind<TypeCategory::Real>>::Unary;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const SomeKindScalar<TypeCategory::Real> &);
  };
  template<typename CRTP> using Un = Unary<CRTP, Result>;
  template<typename CRTP> using Bin = Binary<CRTP, Result>;
  struct Parentheses : public Un<Parentheses> {
    using Un<Parentheses>::Un;
    static std::optional<Scalar> FoldScalar(FoldingContext &, const Scalar &x) {
      return {x};
    }
  };
  struct Negate : public Un<Negate> {
    using Un<Negate>::Un;
    static std::optional<Scalar> FoldScalar(FoldingContext &, const Scalar &);
  };
  struct Add : public Bin<Add> {
    using Bin<Add>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Subtract : public Bin<Subtract> {
    using Bin<Subtract>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Multiply : public Bin<Multiply> {
    using Bin<Multiply>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Divide : public Bin<Divide> {
    using Bin<Divide>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Power : public Bin<Power> {
    using Bin<Power>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct IntPower
    : public Binary<IntPower, Result, Result, SomeKind<TypeCategory::Integer>> {
    using Binary<IntPower, Result, Result,
        SomeKind<TypeCategory::Integer>>::Binary;
    static std::optional<Scalar> FoldScalar(FoldingContext &, const Scalar &,
        const SomeKindScalar<TypeCategory::Integer> &);
  };
  struct Max : public Bin<Max> {
    using Bin<Max>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Min : public Bin<Min> {
    using Bin<Min>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  template<typename CRTP> using ComplexUn = Unary<CRTP, Result, Complex>;
  struct RealPart : public ComplexUn<RealPart> {
    using ComplexUn<RealPart>::ComplexUn;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const evaluate::Scalar<Complex> &);
  };
  struct AIMAG : public ComplexUn<AIMAG> {
    using ComplexUn<AIMAG>::ComplexUn;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const evaluate::Scalar<Complex> &);
  };

  CLASS_BOILERPLATE(Expr)
  Expr(const Scalar &x) : u_{x} {}
  Expr(const SomeKindIntegerExpr &x) : u_{ConvertInteger{x}} {}
  Expr(SomeKindIntegerExpr &&x) : u_{ConvertInteger{std::move(x)}} {}
  template<int K>
  Expr(const IntegerExpr<K> &x) : u_{ConvertInteger{SomeKindIntegerExpr{x}}} {}
  template<int K>
  Expr(IntegerExpr<K> &&x)
    : u_{ConvertInteger{SomeKindIntegerExpr{std::move(x)}}} {}
  Expr(const SomeKindRealExpr &x) : u_{ConvertReal{x}} {}
  Expr(SomeKindRealExpr &&x) : u_{ConvertReal{std::move(x)}} {}
  template<int K>
  Expr(const RealExpr<K> &x) : u_{ConvertReal{SomeKindRealExpr{x}}} {}
  template<int K>
  Expr(RealExpr<K> &&x) : u_{ConvertReal{SomeKindRealExpr{std::move(x)}}} {}
  template<typename A> Expr(const A &x) : u_{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u_{std::move(x)} {}
  template<typename A> Expr(CopyableIndirection<A> &&x) : u_{std::move(x)} {}

  std::optional<Scalar> ScalarValue() const {
    return common::GetIf<Scalar>(u_);
  }
  std::optional<Scalar> Fold(FoldingContext &c);

private:
  std::variant<Scalar, CopyableIndirection<DataRef>,
      CopyableIndirection<ComplexPart>, CopyableIndirection<FunctionRef>,
      ConvertInteger, ConvertReal, Parentheses, Negate, Add, Subtract, Multiply,
      Divide, Power, IntPower, Max, Min, RealPart, AIMAG>
      u_;
};

template<int KIND> class Expr<Type<TypeCategory::Complex, KIND>> {
public:
  using Result = Type<TypeCategory::Complex, KIND>;
  using Scalar = Scalar<Result>;
  using Part = typename Result::Part;
  using FoldableTrait = std::true_type;

  template<typename CRTP> using Un = Unary<CRTP, Result>;
  template<typename CRTP> using Bin = Binary<CRTP, Result>;
  struct Parentheses : public Un<Parentheses> {
    using Un<Parentheses>::Un;
    static std::optional<Scalar> FoldScalar(FoldingContext &, const Scalar &x) {
      return {x};
    }
  };
  struct Negate : public Un<Negate> {
    using Un<Negate>::Un;
    static std::optional<Scalar> FoldScalar(FoldingContext &, const Scalar &);
  };
  struct Add : public Bin<Add> {
    using Bin<Add>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Subtract : public Bin<Subtract> {
    using Bin<Subtract>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Multiply : public Bin<Multiply> {
    using Bin<Multiply>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Divide : public Bin<Divide> {
    using Bin<Divide>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Power : public Bin<Power> {
    using Bin<Power>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct IntPower
    : public Binary<IntPower, Result, Result, SomeKind<TypeCategory::Integer>> {
    using Binary<IntPower, Result, Result,
        SomeKind<TypeCategory::Integer>>::Binary;
    static std::optional<Scalar> FoldScalar(FoldingContext &, const Scalar &,
        const SomeKindScalar<TypeCategory::Integer> &);
  };
  struct CMPLX : public Binary<CMPLX, Result, Part> {
    using Binary<CMPLX, Result, Part>::Binary;
    static std::optional<Scalar> FoldScalar(FoldingContext &,
        const evaluate::Scalar<Part> &, const evaluate::Scalar<Part> &);
  };

  CLASS_BOILERPLATE(Expr)
  Expr(const Scalar &x) : u_{x} {}
  template<typename A> Expr(const A &x) : u_{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u_{std::move(x)} {}
  template<typename A> Expr(CopyableIndirection<A> &&x) : u_{std::move(x)} {}

  std::optional<Scalar> ScalarValue() const {
    return common::GetIf<Scalar>(u_);
  }
  std::optional<Scalar> Fold(FoldingContext &c);

private:
  std::variant<Scalar, CopyableIndirection<DataRef>,
      CopyableIndirection<FunctionRef>, Parentheses, Negate, Add, Subtract,
      Multiply, Divide, Power, IntPower, CMPLX>
      u_;
};

template<int KIND> class Expr<Type<TypeCategory::Character, KIND>> {
public:
  using Result = Type<TypeCategory::Character, KIND>;
  using Scalar = Scalar<Result>;
  using FoldableTrait = std::true_type;
  template<typename CRTP> using Bin = Binary<CRTP, Result>;
  struct Concat : public Bin<Concat> {
    using Bin<Concat>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Max : public Bin<Max> {
    using Bin<Max>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Min : public Bin<Min> {
    using Bin<Min>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };

  CLASS_BOILERPLATE(Expr)
  Expr(const Scalar &x) : u_{x} {}
  Expr(Scalar &&x) : u_{std::move(x)} {}
  template<typename A> Expr(const A &x) : u_{x} {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u_{std::move(x)} {}
  template<typename A> Expr(CopyableIndirection<A> &&x) : u_{std::move(x)} {}

  std::optional<Scalar> ScalarValue() const {
    return common::GetIf<Scalar>(u_);
  }
  std::optional<Scalar> Fold(FoldingContext &c);
  SubscriptIntegerExpr LEN() const;

private:
  std::variant<Scalar, CopyableIndirection<DataRef>,
      CopyableIndirection<Substring>, CopyableIndirection<FunctionRef>, Concat,
      Max, Min>
      u_;
};

// The Comparison class template is a helper for constructing logical
// expressions with polymorphism over the cross product of the possible
// categories and kinds of comparable operands.
ENUM_CLASS(RelationalOperator, LT, LE, EQ, NE, GE, GT)

template<typename A>
struct Comparison
  : public Binary<Comparison<A>, Type<TypeCategory::Logical, 1>, A> {
  using Base = Binary<Comparison, Type<TypeCategory::Logical, 1>, A>;
  using typename Base::Scalar;
  using OperandScalarConstant = typename Base::LeftScalar;
  CLASS_BOILERPLATE(Comparison)
  Comparison(RelationalOperator r, const Expr<A> &a, const Expr<A> &b)
    : Base{a, b}, opr{r} {}
  Comparison(RelationalOperator r, Expr<A> &&a, Expr<A> &&b)
    : Base{std::move(a), std::move(b)}, opr{r} {}
  std::optional<Scalar> FoldScalar(FoldingContext &c,
      const OperandScalarConstant &, const OperandScalarConstant &);
  RelationalOperator opr;
};

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
extern template struct Comparison<Type<TypeCategory::Character, 1>>;

// Dynamically polymorphic comparisons whose operands are expressions of
// the same supported kind of a particular type category.
template<TypeCategory CAT> struct CategoryComparison {
  using Scalar = Scalar<Type<TypeCategory::Logical, 1>>;
  CLASS_BOILERPLATE(CategoryComparison)
  template<int KIND> using KindComparison = Comparison<Type<CAT, KIND>>;
  template<int KIND> CategoryComparison(const KindComparison<KIND> &x) : u{x} {}
  template<int KIND>
  CategoryComparison(KindComparison<KIND> &&x) : u{std::move(x)} {}
  std::optional<Scalar> Fold(FoldingContext &c);
  typename KindsVariant<CAT, KindComparison>::type u;
};

template<int KIND> class Expr<Type<TypeCategory::Logical, KIND>> {
public:
  using Result = Type<TypeCategory::Logical, KIND>;
  using Scalar = Scalar<Result>;
  using FoldableTrait = std::true_type;
  struct Not : Unary<Not, Result> {
    using Unary<Not, Result>::Unary;
    static std::optional<Scalar> FoldScalar(FoldingContext &, const Scalar &);
  };
  template<typename CRTP> using Bin = Binary<CRTP, Result>;
  struct And : public Bin<And> {
    using Bin<And>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Or : public Bin<Or> {
    using Bin<Or>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Eqv : public Bin<Eqv> {
    using Bin<Eqv>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };
  struct Neqv : public Bin<Neqv> {
    using Bin<Neqv>::Bin;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const Scalar &, const Scalar &);
  };

  CLASS_BOILERPLATE(Expr)
  Expr(const Scalar &x) : u_{x} {}
  Expr(bool x) : u_{Scalar{x}} {}
  template<TypeCategory CAT, int K>
  Expr(const Comparison<Type<CAT, K>> &x) : u_{CategoryComparison<CAT>{x}} {}
  template<TypeCategory CAT, int K>
  Expr(Comparison<Type<CAT, K>> &&x)
    : u_{CategoryComparison<CAT>{std::move(x)}} {}
  template<typename A> Expr(const A &x) : u_(x) {}
  template<typename A>
  Expr(std::enable_if_t<!std::is_reference_v<A>, A> &&x) : u_{std::move(x)} {}
  template<typename A> Expr(CopyableIndirection<A> &&x) : u_{std::move(x)} {}

  std::optional<Scalar> ScalarValue() const {
    return common::GetIf<Scalar>(u_);
  }
  std::optional<Scalar> Fold(FoldingContext &c);

private:
  std::variant<Scalar, CopyableIndirection<DataRef>,
      CopyableIndirection<FunctionRef>, Not, And, Or, Eqv, Neqv,
      CategoryComparison<TypeCategory::Integer>,
      CategoryComparison<TypeCategory::Real>,
      CategoryComparison<TypeCategory::Complex>,
      CategoryComparison<TypeCategory::Character>>
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
extern template class Expr<Type<TypeCategory::Character, 1>>;
extern template class Expr<Type<TypeCategory::Logical, 1>>;
extern template class Expr<Type<TypeCategory::Logical, 2>>;
extern template class Expr<Type<TypeCategory::Logical, 4>>;
extern template class Expr<Type<TypeCategory::Logical, 8>>;

// Dynamically polymorphic expressions that can hold any supported kind
// of a specific intrinsic type category.
template<TypeCategory CAT> class Expr<SomeKind<CAT>> {
public:
  using Result = SomeKind<CAT>;
  using Scalar = Scalar<Result>;
  using FoldableTrait = std::true_type;
  CLASS_BOILERPLATE(Expr)
  template<int KIND> using KindExpr = Expr<Type<CAT, KIND>>;
  template<int KIND> Expr(const KindExpr<KIND> &x) : u{x} {}
  template<int KIND> Expr(KindExpr<KIND> &&x) : u{std::move(x)} {}
  std::optional<Scalar> ScalarValue() const;
  std::optional<Scalar> Fold(FoldingContext &);
  typename KindsVariant<CAT, KindExpr>::type u;
};

extern template class Expr<SomeKind<TypeCategory::Integer>>;
extern template class Expr<SomeKind<TypeCategory::Real>>;
extern template class Expr<SomeKind<TypeCategory::Complex>>;
extern template class Expr<SomeKind<TypeCategory::Character>>;
extern template class Expr<SomeKind<TypeCategory::Logical>>;

// BOZ literal constants need to be wide enough to hold an integer or real
// value of any supported kind.  They also need to be distinguishable from
// other integer constants, since they are permitted to be used in only a
// few situations.
using BOZLiteralConstant = value::Integer<128>;

// A completely generic expression, polymorphic across the intrinsic type
// categories and each of their kinds.
struct GenericExpr {
  using Scalar = GenericScalar;
  using FoldableTrait = std::true_type;
  CLASS_BOILERPLATE(GenericExpr)

  template<typename A> GenericExpr(const A &x) : u{x} {}
  template<typename A>
  GenericExpr(std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : u{std::move(x)} {}

  template<TypeCategory CAT, int KIND>
  GenericExpr(const Expr<Type<CAT, KIND>> &x) : u{Expr<SomeKind<CAT>>{x}} {}

  template<TypeCategory CAT, int KIND>
  GenericExpr(Expr<Type<CAT, KIND>> &&x)
    : u{Expr<SomeKind<CAT>>{std::move(x)}} {}

  std::optional<Scalar> ScalarValue() const;
  std::optional<Scalar> Fold(FoldingContext &);
  int Rank() const { return 1; }  // TODO

  std::variant<SomeKindIntegerExpr, SomeKindRealExpr, SomeKindComplexExpr,
      SomeKindCharacterExpr, SomeKindLogicalExpr, BOZLiteralConstant>
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
  template<typename A> LogicalExpr<1> FUNC(const A &x, const A &y) { \
    return {Comparison<typename A::Result>{OP, x, y}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, LogicalExpr<1>> FUNC( \
      const A &x, A &&y) { \
    return {Comparison<typename A::Result>{OP, x, std::move(y)}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, LogicalExpr<1>> FUNC( \
      A &&x, const A &y) { \
    return {Comparison<typename A::Result>{OP, std::move(x), y}}; \
  } \
  template<typename A> \
  std::enable_if_t<!std::is_reference_v<A>, LogicalExpr<1>> FUNC( \
      A &&x, A &&y) { \
    return {Comparison<typename A::Result>{OP, std::move(x), std::move(y)}}; \
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
