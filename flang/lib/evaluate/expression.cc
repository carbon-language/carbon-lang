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

template<typename T>
std::ostream &Emit(std::ostream &o, const CopyableIndirection<Expr<T>> &expr) {
  return expr->Dump(o);
}
template<typename T>
std::ostream &Emit(std::ostream &, const ArrayConstructorValues<T> &);

template<typename ITEM, typename INT>
std::ostream &Emit(std::ostream &o, const ImpliedDo<ITEM, INT> &implDo) {
  o << '(';
  Emit(o, *implDo.values);
  o << ',' << INT::Dump() << "::";
  o << implDo.controlVariableName.ToString();
  o << '=';
  implDo.lower->Dump(o) << ',';
  implDo.upper->Dump(o) << ',';
  implDo.stride->Dump(o) << ')';
  return o;
}

template<typename T>
std::ostream &Emit(std::ostream &o, const ArrayConstructorValues<T> &values) {
  const char *sep{""};
  for (const auto &value : values.values) {
    o << sep;
    std::visit([&](const auto &x) { Emit(o, x); }, value.u);
    sep = ",";
  }
  return o;
}

template<typename T>
std::ostream &ArrayConstructor<T>::Dump(std::ostream &o) const {
  o << '[' << Result::Dump() << "::";
  Emit(o, *this);
  return o << ']';
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
                         return AsExpr(
                             Constant<SubscriptInteger>{c.value.size()});
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

Expr<SomeType>::~Expr() {}

template<typename A>
std::optional<DynamicType> ExpressionBase<A>::GetType() const {
  if constexpr (Result::isSpecificIntrinsicType) {
    return Result::GetType();
  } else {
    return std::visit(
        [](const auto &x) -> std::optional<DynamicType> {
          if constexpr (!std::is_same_v<std::decay_t<decltype(x)>,
                            BOZLiteralConstant>) {
            return x.GetType();
          }
          return std::nullopt;  // typeless -> no type
        },
        derived().u);
  }
}

template<typename A> int ExpressionBase<A>::Rank() const {
  return std::visit(
      [](const auto &x) {
        if constexpr (std::is_same_v<std::decay_t<decltype(x)>,
                          BOZLiteralConstant>) {
          return 0;
        } else {
          return x.Rank();
        }
      },
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
FOR_EACH_TYPE_AND_KIND(template class ExpressionBase)
}

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
}
