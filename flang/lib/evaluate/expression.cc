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

#include "expression.h"
#include "common.h"
#include "int-power.h"
#include "tools.h"
#include "variable.h"
#include "../common/idioms.h"
#include "../parser/characters.h"
#include "../parser/message.h"
#include <ostream>
#include <string>
#include <type_traits>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

// Fold

template<typename D, typename R, typename... O>
auto Operation<D, R, O...>::Fold(FoldingContext &context)
    -> std::optional<Constant<Result>> {
  auto c0{left().Fold(context)};
  if constexpr (operands == 1) {
    if (c0.has_value()) {
      if (auto scalar{derived().FoldScalar(context, c0->value)}) {
        return {Constant<Result>{std::move(*scalar)}};
      }
    }
  } else {
    static_assert(operands == 2);  // TODO: generalize to N operands?
    auto c1{right().Fold(context)};
    if (c0.has_value() && c1.has_value()) {
      if (auto scalar{derived().FoldScalar(context, c0->value, c1->value)}) {
        return {Constant<Result>{std::move(*scalar)}};
      }
    }
  }
  return std::nullopt;
}

template<typename RESULT>
auto ExpressionBase<RESULT>::Fold(FoldingContext &context)
    -> std::optional<Constant<Result>> {
  using Const = Constant<Result>;
  if constexpr (Result::isSpecificType) {
    // Folding an expression of known type category and kind.
    return std::visit(
        [&](auto &x) -> std::optional<Const> {
          using Thing = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<Thing, Const>) {
            return {x};
          }
          if constexpr (IsFoldableTrait<Thing>) {
            if (auto c{x.Fold(context)}) {
              static constexpr TypeCategory category{Result::category};
              if constexpr (category == TypeCategory::Real ||
                  category == TypeCategory::Complex) {
                if (context.flushDenormalsToZero) {
                  c->value = c->value.FlushDenormalToZero();
                }
              } else if constexpr (category == TypeCategory::Logical) {
                // Folding may have produced a constant of some
                // dissimilar LOGICAL kind.
                bool truth{c->value.IsTrue()};
                derived() = Derived{truth};
                return {Const{truth}};
              }
              if constexpr (std::is_same_v<Parentheses<Result>, Thing>) {
                // Preserve parentheses around constants.
                derived() = Derived{Thing{Derived{*c}}};
              } else {
                derived() = Derived{*c};
              }
              return {Const{c->value}};
            }
          }
          return std::nullopt;
        },
        derived().u);
  } else {
    // Folding a generic expression into a generic constant.
    return std::visit(
        [&](auto &x) -> std::optional<Const> {
          if constexpr (IsFoldableTrait<std::decay_t<decltype(x)>>) {
            if (auto c{x.Fold(context)}) {
              if constexpr (ResultType<decltype(*c)>::isSpecificType) {
                return {Const{c->value}};
              } else {
                return {Const{common::MoveVariant<GenericScalar>(c->value.u)}};
              }
            }
          }
          return std::nullopt;
        },
        derived().u);
  }
}

// FoldScalar

template<typename TO, TypeCategory FROMCAT>
auto Convert<TO, FROMCAT>::FoldScalar(FoldingContext &context,
    const Scalar<Operand> &x) -> std::optional<Scalar<Result>> {
  return std::visit(
      [&](const auto &c) -> std::optional<Scalar<Result>> {
        if constexpr (Result::category == TypeCategory::Integer) {
          if constexpr (Operand::category == TypeCategory::Integer) {
            auto converted{Scalar<Result>::ConvertSigned(c)};
            if (converted.overflow) {
              context.messages.Say(
                  "INTEGER to INTEGER conversion overflowed"_en_US);
            } else {
              return {std::move(converted.value)};
            }
          } else if constexpr (Operand::category == TypeCategory::Real) {
            auto converted{c.template ToInteger<Scalar<Result>>()};
            if (converted.flags.test(RealFlag::InvalidArgument)) {
              context.messages.Say(
                  "REAL to INTEGER conversion: invalid argument"_en_US);
            } else if (converted.flags.test(RealFlag::Overflow)) {
              context.messages.Say(
                  "REAL to INTEGER conversion overflowed"_en_US);
            } else {
              return {std::move(converted.value)};
            }
          }
        } else if constexpr (Result::category == TypeCategory::Real) {
          if constexpr (Operand::category == TypeCategory::Integer) {
            auto converted{Scalar<Result>::FromInteger(c)};
            RealFlagWarnings(
                context, converted.flags, "INTEGER to REAL conversion");
            return {std::move(converted.value)};
          } else if constexpr (Operand::category == TypeCategory::Real) {
            auto converted{Scalar<Result>::Convert(c)};
            RealFlagWarnings(
                context, converted.flags, "REAL to REAL conversion");
            return {std::move(converted.value)};
          }
        }
        return std::nullopt;
      },
      x.u);
}

template<typename A>
auto Negate<A>::FoldScalar(FoldingContext &context, const Scalar<Operand> &c)
    -> std::optional<Scalar<Result>> {
  if constexpr (Result::category == TypeCategory::Integer) {
    auto negated{c.Negate()};
    if (negated.overflow) {
      context.messages.Say("INTEGER negation overflowed"_en_US);
    } else {
      return {std::move(negated.value)};
    }
  } else {
    return {c.Negate()};  // REAL & COMPLEX: no exceptions possible
  }
  return std::nullopt;
}

template<int KIND>
auto ComplexComponent<KIND>::FoldScalar(FoldingContext &context,
    const Scalar<Operand> &z) const -> std::optional<Scalar<Result>> {
  return {isImaginaryPart ? z.AIMAG() : z.REAL()};
}

template<int KIND>
auto Not<KIND>::FoldScalar(FoldingContext &context, const Scalar<Operand> &x)
    -> std::optional<Scalar<Result>> {
  return {Scalar<Result>{!x.IsTrue()}};
}

template<typename A>
auto Add<A>::FoldScalar(FoldingContext &context, const Scalar<Operand> &x,
    const Scalar<Operand> &y) -> std::optional<Scalar<Result>> {
  if constexpr (Result::category == TypeCategory::Integer) {
    auto sum{x.AddSigned(y)};
    if (sum.overflow) {
      context.messages.Say(
          "INTEGER(KIND=%d) addition overflowed"_en_US, Result::kind);
      return std::nullopt;
    }
    return {std::move(sum.value)};
  } else {
    auto sum{x.Add(y, context.rounding)};
    RealFlagWarnings(context, sum.flags, "addition");
    return {std::move(sum.value)};
  }
}

template<typename A>
auto Subtract<A>::FoldScalar(FoldingContext &context, const Scalar<Operand> &x,
    const Scalar<Operand> &y) -> std::optional<Scalar<Result>> {
  if constexpr (Result::category == TypeCategory::Integer) {
    auto diff{x.SubtractSigned(y)};
    if (diff.overflow) {
      context.messages.Say(
          "INTEGER(KIND=%d) subtraction overflowed"_en_US, Result::kind);
      return std::nullopt;
    }
    return {std::move(diff.value)};
  } else {
    auto difference{x.Subtract(y, context.rounding)};
    RealFlagWarnings(context, difference.flags, "subtraction");
    return {std::move(difference.value)};
  }
}

template<typename A>
auto Multiply<A>::FoldScalar(FoldingContext &context, const Scalar<Operand> &x,
    const Scalar<Operand> &y) -> std::optional<Scalar<Result>> {
  if constexpr (Result::category == TypeCategory::Integer) {
    auto product{x.MultiplySigned(y)};
    if (product.SignedMultiplicationOverflowed()) {
      context.messages.Say(
          "INTEGER(KIND=%d) multiplication overflowed"_en_US, Result::kind);
      return std::nullopt;
    }
    return {std::move(product.lower)};
  } else {
    auto product{x.Multiply(y, context.rounding)};
    RealFlagWarnings(context, product.flags, "multiplication");
    return {std::move(product.value)};
  }
}

template<typename A>
auto Divide<A>::FoldScalar(FoldingContext &context, const Scalar<Operand> &x,
    const Scalar<Operand> &y) -> std::optional<Scalar<Result>> {
  if constexpr (Result::category == TypeCategory::Integer) {
    auto qr{x.DivideSigned(y)};
    if (qr.divisionByZero) {
      context.messages.Say("INTEGER division by zero"_en_US);
      return std::nullopt;
    }
    if (qr.overflow) {
      context.messages.Say(
          "INTEGER(KIND=%d) division overflowed"_en_US, Result::kind);
      return std::nullopt;
    }
    return {std::move(qr.quotient)};
  } else {
    auto quotient{x.Divide(y, context.rounding)};
    RealFlagWarnings(context, quotient.flags, "division");
    return {std::move(quotient.value)};
  }
}

template<typename A>
auto Power<A>::FoldScalar(FoldingContext &context, const Scalar<Operand> &x,
    const Scalar<Operand> &y) -> std::optional<Scalar<Result>> {
  if constexpr (Result::category == TypeCategory::Integer) {
    typename Scalar<Result>::PowerWithErrors power{x.Power(y)};
    if (power.divisionByZero) {
      context.messages.Say("zero to negative power"_en_US);
    } else if (power.overflow) {
      context.messages.Say(
          "INTEGER(KIND=%d) power overflowed"_en_US, Result::kind);
    } else if (power.zeroToZero) {
      context.messages.Say("INTEGER 0**0 is not defined"_en_US);
    } else {
      return {std::move(power.power)};
    }
  } else {
    // TODO: real and complex exponentiation to non-integer powers
  }
  return std::nullopt;
}

template<typename A>
auto RealToIntPower<A>::FoldScalar(FoldingContext &context,
    const Scalar<BaseOperand> &x, const Scalar<ExponentOperand> &y)
    -> std::optional<Scalar<Result>> {
  return std::visit(
      [&](const auto &pow) -> std::optional<Scalar<Result>> {
        auto power{evaluate::IntPower(x, pow)};
        RealFlagWarnings(context, power.flags, "raising to INTEGER power");
        return {std::move(power.value)};
      },
      y.u);
}

template<typename A>
auto Extremum<A>::FoldScalar(FoldingContext &context, const Scalar<Operand> &x,
    const Scalar<Operand> &y) const -> std::optional<Scalar<Result>> {
  if constexpr (Operand::category == TypeCategory::Integer) {
    if (ordering == x.CompareSigned(y)) {
      return {x};
    }
  } else if constexpr (Operand::category == TypeCategory::Real) {
    if (x.IsNotANumber() ||
        (x.Compare(y) == Relation::Less) == (ordering == Ordering::Less)) {
      return {x};
    }
  } else {
    if (ordering == Compare(x, y)) {
      return {x};
    }
  }
  return {y};
}

template<int KIND>
auto ComplexConstructor<KIND>::FoldScalar(
    FoldingContext &context, const Scalar<Operand> &x, const Scalar<Operand> &y)
    -> std::optional<Scalar<Result>> {
  return {Scalar<Result>{x, y}};
}

template<int KIND>
auto Concat<KIND>::FoldScalar(FoldingContext &context, const Scalar<Operand> &x,
    const Scalar<Operand> &y) -> std::optional<Scalar<Result>> {
  if constexpr (KIND == 1) {
    return {x + y};
  }
  return std::nullopt;
}

template<typename A>
auto Relational<A>::FoldScalar(FoldingContext &c, const Scalar<Operand> &a,
    const Scalar<Operand> &b) -> std::optional<Scalar<Result>> {
  if constexpr (A::category == TypeCategory::Integer) {
    switch (a.CompareSigned(b)) {
    case Ordering::Less:
      return {opr == RelationalOperator::LE || opr == RelationalOperator::LE ||
          opr == RelationalOperator::NE};
    case Ordering::Equal:
      return {opr == RelationalOperator::LE || opr == RelationalOperator::EQ ||
          opr == RelationalOperator::GE};
    case Ordering::Greater:
      return {opr == RelationalOperator::NE || opr == RelationalOperator::GE ||
          opr == RelationalOperator::GT};
    }
  }
  if constexpr (A::category == TypeCategory::Real) {
    switch (a.Compare(b)) {
    case Relation::Less:
      return {opr == RelationalOperator::LE || opr == RelationalOperator::LE ||
          opr == RelationalOperator::NE};
    case Relation::Equal:
      return {opr == RelationalOperator::LE || opr == RelationalOperator::EQ ||
          opr == RelationalOperator::GE};
    case Relation::Greater:
      return {opr == RelationalOperator::NE || opr == RelationalOperator::GE ||
          opr == RelationalOperator::GT};
    case Relation::Unordered: return std::nullopt;
    }
  }
  if constexpr (A::category == TypeCategory::Character) {
    switch (Compare(a, b)) {
    case Ordering::Less:
      return {opr == RelationalOperator::LE || opr == RelationalOperator::LE ||
          opr == RelationalOperator::NE};
    case Ordering::Equal:
      return {opr == RelationalOperator::LE || opr == RelationalOperator::EQ ||
          opr == RelationalOperator::GE};
    case Ordering::Greater:
      return {opr == RelationalOperator::NE || opr == RelationalOperator::GE ||
          opr == RelationalOperator::GT};
    }
  }
  return std::nullopt;
}

template<int KIND>
auto LogicalOperation<KIND>::FoldScalar(FoldingContext &context,
    const Scalar<Operand> &x, const Scalar<Operand> &y) const
    -> std::optional<Scalar<Result>> {
  bool xt{x.IsTrue()}, yt{y.IsTrue()};
  switch (logicalOperator) {
  case LogicalOperator::And: return {Scalar<Result>{xt && yt}};
  case LogicalOperator::Or: return {Scalar<Result>{xt || yt}};
  case LogicalOperator::Eqv: return {Scalar<Result>{xt == yt}};
  case LogicalOperator::Neqv: return {Scalar<Result>{xt != yt}};
  }
  return std::nullopt;
}

// Dump

template<typename D, typename R, typename... O>
std::ostream &Operation<D, R, O...>::Dump(std::ostream &o) const {
  left().Dump(derived().Prefix(o));
  if constexpr (operands > 1) {
    right().Dump(derived().Infix(o));
  }
  return derived().Suffix(o);
}

template<typename TO, TypeCategory FROMCAT>
std::ostream &Convert<TO, FROMCAT>::Dump(std::ostream &o) const {
  static_assert(TO::category == TypeCategory::Integer ||
      TO::category == TypeCategory::Real ||
      TO::category == TypeCategory::Logical || !"Convert<> to bad category!");
  if constexpr (TO::category == TypeCategory::Integer) {
    o << "INT";
  } else if constexpr (TO::category == TypeCategory::Real) {
    o << "REAL";
  } else if constexpr (TO::category == TypeCategory::Logical) {
    o << "LOGICAL";
  }
  return this->left().Dump(o << '(') << ",KIND=" << TO::kind << ')';
}

template<typename A> std::ostream &Relational<A>::Infix(std::ostream &o) const {
  return o << '.' << EnumToString(opr) << '.';
}

std::ostream &Relational<SomeType>::Dump(std::ostream &o) const {
  std::visit([&](const auto &rel) { rel.Dump(o); }, u);
  return o;
}

template<int KIND>
std::ostream &LogicalOperation<KIND>::Infix(std::ostream &o) const {
  switch (logicalOperator) {
  case LogicalOperator::And: o << ".AND."; break;
  case LogicalOperator::Or: o << ".OR."; break;
  case LogicalOperator::Eqv: o << ".EQV."; break;
  case LogicalOperator::Neqv: o << ".NEQV."; break;
  }
  return o;
}

template<typename T> std::ostream &Constant<T>::Dump(std::ostream &o) const {
  if constexpr (T::category == TypeCategory::Integer) {
    return o << value.SignedDecimal() << '_' << T::kind;
  } else if constexpr (T::category == TypeCategory::Real ||
      T::category == TypeCategory::Complex) {
    return o << value.DumpHexadecimal() << '_' << T::kind;
  } else if constexpr (T::category == TypeCategory::Character) {
    if constexpr (T::kind == 1) {
      return o << T::kind << '_' << parser::QuoteCharacterLiteral(value);
    } else {
      return o << T::kind
               << "_'(wide character dumping unimplemented)'";  // TODO
    }
  } else if constexpr (T::category == TypeCategory::Logical) {
    if (value.IsTrue()) {
      o << ".TRUE.";
    } else {
      o << ".FALSE.";
    }
    return o << '_' << Result::kind;
  } else {
    return value.u.Dump(o);
  }
}

template<typename RESULT>
std::ostream &ExpressionBase<RESULT>::Dump(std::ostream &o) const {
  std::visit(common::visitors{[&](const BOZLiteralConstant &x) {
                                o << "Z'" << x.Hexadecimal() << "'";
                              },
                 [&](const CopyableIndirection<Substring> &s) { s->Dump(o); },
                 [&](const auto &x) { x.Dump(o); }},
      derived().u);
  return o;
}

template<int KIND>
Expr<SubscriptInteger> Expr<Type<TypeCategory::Character, KIND>>::LEN() const {
  return std::visit(
      common::visitors{[](const Constant<Result> &c) {
                         // std::string::size_type isn't convertible to uint64_t
                         // on Darwin
                         return AsExpr(Constant<SubscriptInteger>{
                             static_cast<std::uint64_t>(c.value.size())});
                       },
          [](const Parentheses<Result> &x) { return x.left().LEN(); },
          [](const Concat<KIND> &c) {
            return c.left().LEN() + c.right().LEN();
          },
          [](const Extremum<Result> &c) {
            return Expr<SubscriptInteger>{
                Extremum<SubscriptInteger>{c.left().LEN(), c.right().LEN()}};
          },
          [](const Designator<Result> &dr) { return dr.LEN(); },
          [](const FunctionRef<Result> &fr) { return fr.LEN(); }},
      u);
}

template<typename RESULT>
auto ExpressionBase<RESULT>::ScalarValue() const
    -> std::optional<Scalar<Result>> {
  if constexpr (Result::isSpecificType) {
    if (auto *c{std::get_if<Constant<Result>>(&derived().u)}) {
      return {c->value};
    }
    if (auto *p{std::get_if<Parentheses<Result>>(&derived().u)}) {
      return p->left().ScalarValue();
    }
  } else if constexpr (std::is_same_v<Result, SomeType>) {
    return std::visit(
        common::visitors{
            [](const BOZLiteralConstant &) -> std::optional<Scalar<Result>> {
              return std::nullopt;
            },
            [](const Expr<SomeDerived> &) -> std::optional<Scalar<Result>> {
              return std::nullopt;
            },
            [](const auto &catEx) -> std::optional<Scalar<Result>> {
              if (auto cv{catEx.ScalarValue()}) {
                // *cv is SomeKindScalar<CAT> for some category; rewrap it.
                return {common::MoveVariant<GenericScalar>(std::move(cv->u))};
              }
              return std::nullopt;
            }},
        derived().u);
  } else {
    return std::visit(
        [](const auto &kindEx) -> std::optional<Scalar<Result>> {
          if (auto sv{kindEx.ScalarValue()}) {
            return {SomeKindScalar<Result::category>{*sv}};
          }
          return std::nullopt;
        },
        derived().u);
  }
  return std::nullopt;
}

Expr<SomeType>::~Expr() {}

// Rank()
template<typename A> int ExpressionBase<A>::Rank() const {
  return std::visit(
      common::visitors{[](const BOZLiteralConstant &) { return 0; },
          [](const auto &x) { return x.Rank(); }},
      derived().u);
}

// Template instantiations to resolve the "extern template" declarations
// that appear in expression.h.

FOR_EACH_INTRINSIC_KIND(template class Expr)
FOR_EACH_CATEGORY_TYPE(template class Expr)
FOR_EACH_INTEGER_KIND(template struct Relational)
FOR_EACH_REAL_KIND(template struct Relational)
FOR_EACH_CHARACTER_KIND(template struct Relational)
template struct Relational<SomeType>;
FOR_EACH_INTRINSIC_KIND(template struct ExpressionBase)
FOR_EACH_CATEGORY_TYPE(template struct ExpressionBase)

}  // namespace Fortran::evaluate

// For reclamation of analyzed expressions to which owning pointers have
// been embedded in the parse tree.  This destructor appears here, where
// definitions for all the necessary types are available, to obviate a
// need to include lib/evaluate/*.h headers in the parser proper.
namespace Fortran::common {
template<> OwningPointer<evaluate::GenericExprWrapper>::~OwningPointer() {
  delete p_;
  p_ = nullptr;
}
template class OwningPointer<evaluate::GenericExprWrapper>;
}  // namespace Fortran::common
