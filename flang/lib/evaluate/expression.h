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

CLASS_TRAIT(FoldableTrait);
struct FoldingContext {
  const parser::CharBlock &at;
  parser::Messages *messages;
};

// Helper base classes for packaging subexpressions.
template<typename CRTP, typename RESULT, typename A = RESULT> class Unary {
protected:
  using OperandType = A;
  using Operand = Expr<OperandType>;
  using OperandScalarConstant = typename OperandType::Value;

public:
  using Result = RESULT;
  using Scalar = typename Result::Value;
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
  using LeftScalar = typename LeftType::Value;
  using RightType = B;
  using Right = Expr<RightType>;
  using RightScalar = typename RightType::Value;

public:
  using Result = RESULT;
  using Scalar = typename Result::Value;
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

template<int KIND> class Expr<Type<Category::Integer, KIND>> {
public:
  using Result = Type<Category::Integer, KIND>;
  using Scalar = typename Result::Value;
  using FoldableTrait = std::true_type;

  struct ConvertInteger
    : public Unary<ConvertInteger, Result, AnyKindType<Category::Integer>> {
    using Unary<ConvertInteger, Result, AnyKindType<Category::Integer>>::Unary;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const ScalarConstant<Category::Integer> &);
  };

  struct ConvertReal
    : public Unary<ConvertReal, Result, AnyKindType<Category::Real>> {
    using Unary<ConvertReal, Result, AnyKindType<Category::Real>>::Unary;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const ScalarConstant<Category::Real> &);
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
  template<int K>
  Expr(const IntegerExpr<K> &x) : u_{ConvertInteger{AnyKindIntegerExpr{x}}} {}
  template<int K>
  Expr(IntegerExpr<K> &&x)
    : u_{ConvertInteger{AnyKindIntegerExpr{std::move(x)}}} {}
  template<int K>
  Expr(const RealExpr<K> &x) : u_{ConvertReal{AnyKindRealExpr{x}}} {}
  template<int K>
  Expr(RealExpr<K> &&x) : u_{ConvertReal{AnyKindRealExpr{std::move(x)}}} {}
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

template<int KIND> class Expr<Type<Category::Real, KIND>> {
public:
  using Result = Type<Category::Real, KIND>;
  using Scalar = typename Result::Value;
  using FoldableTrait = std::true_type;

  // N.B. Real->Complex and Complex->Real conversions are done with CMPLX
  // and part access operations (resp.).  Conversions between kinds of
  // Complex are done via decomposition to Real and reconstruction.
  struct ConvertInteger
    : public Unary<ConvertInteger, Result, AnyKindType<Category::Integer>> {
    using Unary<ConvertInteger, Result, AnyKindType<Category::Integer>>::Unary;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const ScalarConstant<Category::Integer> &);
  };
  struct ConvertReal
    : public Unary<ConvertReal, Result, AnyKindType<Category::Real>> {
    using Unary<ConvertReal, Result, AnyKindType<Category::Real>>::Unary;
    static std::optional<Scalar> FoldScalar(
        FoldingContext &, const ScalarConstant<Category::Real> &);
  };
  template<typename CRTP> using Un = Unary<CRTP, Result>;
  template<typename CRTP> using Bin = Binary<CRTP, Result>;
  struct Parentheses : public Un<Parentheses> {
    using Un<Parentheses>::Un;
  };
  struct Negate : public Un<Negate> {
    using Un<Negate>::Un;
  };
  struct Add : public Bin<Add> {
    using Bin<Add>::Bin;
  };
  struct Subtract : public Bin<Subtract> {
    using Bin<Subtract>::Bin;
  };
  struct Multiply : public Bin<Multiply> {
    using Bin<Multiply>::Bin;
  };
  struct Divide : public Bin<Divide> {
    using Bin<Divide>::Bin;
  };
  struct Power : public Bin<Power> {
    using Bin<Power>::Bin;
  };
  struct IntPower
    : public Binary<IntPower, Result, Result, AnyKindType<Category::Integer>> {
    using Binary<IntPower, Result, Result,
        AnyKindType<Category::Integer>>::Binary;
  };
  struct Max : public Bin<Max> {
    using Bin<Max>::Bin;
  };
  struct Min : public Bin<Min> {
    using Bin<Min>::Bin;
  };
  template<typename CRTP>
  using CplxUn = Unary<CRTP, Result, Type<Category::Complex, KIND>>;
  struct RealPart : public CplxUn<RealPart> {
    using CplxUn<RealPart>::CplxUn;
  };
  struct AIMAG : public CplxUn<AIMAG> {
    using CplxUn<AIMAG>::CplxUn;
  };

  CLASS_BOILERPLATE(Expr)
  Expr(const Scalar &x) : u_{x} {}
  template<int K>
  Expr(const IntegerExpr<K> &x) : u_{ConvertInteger{AnyKindIntegerExpr{x}}} {}
  template<int K>
  Expr(IntegerExpr<K> &&x)
    : u_{ConvertInteger{AnyKindIntegerExpr{std::move(x)}}} {}
  template<int K>
  Expr(const RealExpr<K> &x) : u_{ConvertReal{AnyKindRealExpr{x}}} {}
  template<int K>
  Expr(RealExpr<K> &&x) : u_{ConvertReal{AnyKindRealExpr{std::move(x)}}} {}
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

template<int KIND> class Expr<Type<Category::Complex, KIND>> {
public:
  using Result = Type<Category::Complex, KIND>;
  using Scalar = typename Result::Value;
  using FoldableTrait = std::true_type;
  template<typename CRTP> using Un = Unary<CRTP, Result>;
  template<typename CRTP> using Bin = Binary<CRTP, Result>;
  struct Parentheses : public Un<Parentheses> {
    using Un<Parentheses>::Un;
  };
  struct Negate : public Un<Negate> {
    using Un<Negate>::Un;
  };
  struct Add : public Bin<Add> {
    using Bin<Add>::Bin;
  };
  struct Subtract : public Bin<Subtract> {
    using Bin<Subtract>::Bin;
  };
  struct Multiply : public Bin<Multiply> {
    using Bin<Multiply>::Bin;
  };
  struct Divide : public Bin<Divide> {
    using Bin<Divide>::Bin;
  };
  struct Power : public Bin<Power> {
    using Bin<Power>::Bin;
  };
  struct IntPower
    : public Binary<IntPower, Result, AnyKindType<Category::Integer>> {
    using Binary<IntPower, Result, AnyKindType<Category::Integer>>::Binary;
  };
  struct CMPLX : public Binary<CMPLX, Result, Type<Category::Real, KIND>> {
    using Binary<CMPLX, Result, Type<Category::Real, KIND>>::Binary;
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

template<int KIND> class Expr<Type<Category::Character, KIND>> {
public:
  using Result = Type<Category::Character, KIND>;
  using Scalar = typename Result::Value;
  using FoldableTrait = std::true_type;
  template<typename CRTP> using Bin = Binary<CRTP, Result>;
  struct Concat : public Bin<Concat> {
    using Bin<Concat>::Bin;
  };
  struct Max : public Bin<Max> {
    using Bin<Max>::Bin;
  };
  struct Min : public Bin<Min> {
    using Bin<Min>::Bin;
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
  : public Binary<Comparison<A>, Type<Category::Logical, 1>, A> {
  using Base = Binary<Comparison, Type<Category::Logical, 1>, A>;
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

extern template struct Comparison<Type<Category::Integer, 1>>;
extern template struct Comparison<Type<Category::Integer, 2>>;
extern template struct Comparison<Type<Category::Integer, 4>>;
extern template struct Comparison<Type<Category::Integer, 8>>;
extern template struct Comparison<Type<Category::Integer, 16>>;
extern template struct Comparison<Type<Category::Real, 2>>;
extern template struct Comparison<Type<Category::Real, 4>>;
extern template struct Comparison<Type<Category::Real, 8>>;
extern template struct Comparison<Type<Category::Real, 10>>;
extern template struct Comparison<Type<Category::Real, 16>>;
extern template struct Comparison<Type<Category::Complex, 2>>;
extern template struct Comparison<Type<Category::Complex, 4>>;
extern template struct Comparison<Type<Category::Complex, 8>>;
extern template struct Comparison<Type<Category::Complex, 10>>;
extern template struct Comparison<Type<Category::Complex, 16>>;
extern template struct Comparison<Type<Category::Character, 1>>;

// Dynamically polymorphic comparisons whose operands are expressions of
// the same supported kind of a particular type category.
template<Category CAT> struct CategoryComparison {
  using Scalar = typename Type<Category::Logical, 1>::Value;
  CLASS_BOILERPLATE(CategoryComparison)
  template<int KIND> using KindComparison = Comparison<Type<CAT, KIND>>;
  template<int KIND> CategoryComparison(const KindComparison<KIND> &x) : u{x} {}
  template<int KIND>
  CategoryComparison(KindComparison<KIND> &&x) : u{std::move(x)} {}
  std::optional<Scalar> Fold(FoldingContext &c);
  typename KindsVariant<CAT, KindComparison>::type u;
};

template<int KIND> class Expr<Type<Category::Logical, KIND>> {
public:
  using Result = Type<Category::Logical, KIND>;
  using Scalar = typename Result::Value;
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
  template<Category CAT, int K>
  Expr(const Comparison<Type<CAT, K>> &x) : u_{CategoryComparison<CAT>{x}} {}
  template<Category CAT, int K>
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
      CategoryComparison<Category::Integer>, CategoryComparison<Category::Real>,
      CategoryComparison<Category::Complex>,
      CategoryComparison<Category::Character>>
      u_;
};

extern template class Expr<Type<Category::Integer, 1>>;
extern template class Expr<Type<Category::Integer, 2>>;
extern template class Expr<Type<Category::Integer, 4>>;
extern template class Expr<Type<Category::Integer, 8>>;
extern template class Expr<Type<Category::Integer, 16>>;
extern template class Expr<Type<Category::Real, 2>>;
extern template class Expr<Type<Category::Real, 4>>;
extern template class Expr<Type<Category::Real, 8>>;
extern template class Expr<Type<Category::Real, 10>>;
extern template class Expr<Type<Category::Real, 16>>;
extern template class Expr<Type<Category::Complex, 2>>;
extern template class Expr<Type<Category::Complex, 4>>;
extern template class Expr<Type<Category::Complex, 8>>;
extern template class Expr<Type<Category::Complex, 10>>;
extern template class Expr<Type<Category::Complex, 16>>;
extern template class Expr<Type<Category::Character, 1>>;
extern template class Expr<Type<Category::Logical, 1>>;
extern template class Expr<Type<Category::Logical, 2>>;
extern template class Expr<Type<Category::Logical, 4>>;
extern template class Expr<Type<Category::Logical, 8>>;

// Dynamically polymorphic expressions that can hold any supported kind
// of a specific intrinsic type category.
template<Category CAT> class Expr<AnyKindType<CAT>> {
public:
  using Result = AnyKindType<CAT>;
  using Scalar = typename Result::Value;
  using FoldableTrait = std::true_type;
  CLASS_BOILERPLATE(Expr)
  template<int KIND> using KindExpr = Expr<Type<CAT, KIND>>;
  template<int KIND> Expr(const KindExpr<KIND> &x) : u{x} {}
  template<int KIND> Expr(KindExpr<KIND> &&x) : u{std::move(x)} {}
  std::optional<Scalar> ScalarValue() const;
  std::optional<Scalar> Fold(FoldingContext &);
  typename KindsVariant<CAT, KindExpr>::type u;
};

extern template class Expr<AnyKindType<Category::Integer>>;
extern template class Expr<AnyKindType<Category::Real>>;
extern template class Expr<AnyKindType<Category::Complex>>;
extern template class Expr<AnyKindType<Category::Character>>;
extern template class Expr<AnyKindType<Category::Logical>>;

// A completely generic expression, polymorphic across the intrinsic type
// categories and each of their kinds.
struct GenericExpr {
  using Scalar = GenericScalar;
  using FoldableTrait = std::true_type;
  CLASS_BOILERPLATE(GenericExpr)
  template<Category CAT, int KIND>
  GenericExpr(const Expr<Type<CAT, KIND>> &x) : u{Expr<AnyKindType<CAT>>{x}} {}
  template<Category CAT, int KIND>
  GenericExpr(Expr<Type<CAT, KIND>> &&x)
    : u{Expr<AnyKindType<CAT>>{std::move(x)}} {}
  template<typename A> GenericExpr(const A &x) : u{x} {}
  template<typename A>
  GenericExpr(std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : u{std::move(x)} {}
  std::optional<Scalar> ScalarValue() const;
  std::optional<Scalar> Fold(FoldingContext &);
  int Rank() const { return 1; }  // TODO
  std::variant<AnyKindIntegerExpr, AnyKindRealExpr, AnyKindComplexExpr,
      AnyKindCharacterExpr, AnyKindLogicalExpr>
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
