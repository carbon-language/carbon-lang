// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "formatting.h"
#include "call.h"
#include "constant.h"
#include "expression.h"
#include "../parser/characters.h"
#include "../semantics/symbol.h"

namespace Fortran::evaluate {

static void ShapeAsFortran(
    std::ostream &o, const std::vector<std::int64_t> &shape) {
  if (shape.size() > 1) {
    o << ",shape=";
    char ch{'['};
    for (auto dim : shape) {
      o << ch << dim;
      ch = ',';
    }
    o << "])";
  }
}

template<typename RESULT, typename VALUE>
std::ostream &ConstantBase<RESULT, VALUE>::AsFortran(std::ostream &o) const {
  if (Rank() > 1) {
    o << "reshape(";
  }
  if (Rank() > 0) {
    o << '[' << GetType().AsFortran() << "::";
  }
  bool first{true};
  for (const auto &value : values_) {
    if (first) {
      first = false;
    } else {
      o << ',';
    }
    if constexpr (Result::category == TypeCategory::Integer) {
      o << value.SignedDecimal() << '_' << Result::kind;
    } else if constexpr (Result::category == TypeCategory::Real ||
        Result::category == TypeCategory::Complex) {
      value.AsFortran(o, Result::kind);
    } else if constexpr (Result::category == TypeCategory::Character) {
      o << Result::kind << '_' << parser::QuoteCharacterLiteral(value);
    } else if constexpr (Result::category == TypeCategory::Logical) {
      if (value.IsTrue()) {
        o << ".true.";
      } else {
        o << ".false.";
      }
      o << '_' << Result::kind;
    } else {
      StructureConstructor{AsConstant().derivedTypeSpec(), value}.AsFortran(o);
    }
  }
  if (Rank() > 0) {
    o << ']';
  }
  ShapeAsFortran(o, shape_);
  return o;
}

template<int KIND>
std::ostream &Constant<Type<TypeCategory::Character, KIND>>::AsFortran(
    std::ostream &o) const {
  if (Rank() > 1) {
    o << "reshape(";
  }
  if (Rank() > 0) {
    o << '[' << GetType().AsFortran(std::to_string(length_)) << "::";
  }
  auto total{static_cast<std::int64_t>(size())};
  for (std::int64_t j{0}; j < total; ++j) {
    ScalarValue value{values_.substr(j * length_, length_)};
    if (j > 0) {
      o << ',';
    } else if (Rank() == 0) {
      o << Result::kind << '_';
    }
    o << parser::QuoteCharacterLiteral(value);
  }
  if (Rank() > 0) {
    o << ']';
  }
  ShapeAsFortran(o, shape_);
  return o;
}

std::ostream &ActualArgument::AsFortran(std::ostream &o) const {
  if (keyword.has_value()) {
    o << keyword->ToString() << '=';
  }
  if (isAlternateReturn) {
    o << '*';
  }
  return value().AsFortran(o);
}

std::ostream &SpecificIntrinsic::AsFortran(std::ostream &o) const {
  return o << name;
}

std::ostream &ProcedureRef::AsFortran(std::ostream &o) const {
  proc_.AsFortran(o);
  char separator{'('};
  for (const auto &arg : arguments_) {
    if (arg.has_value()) {
      arg->AsFortran(o << separator);
      separator = ',';
    }
  }
  if (separator == '(') {
    o << '(';
  }
  return o << ')';
}

// Operator precedence formatting; insert parentheses around operands
// only when necessary.

enum class Precedence {
  Primary,  // don't parenthesize
  Parenthesize,  // (x), (real, imaginary)
  DefinedUnary,
  Negate,
  Power,  // ** which is right-associative
  Multiplicative,  // *, /
  Additive,  // +, -, //
  Relational,
  Logical,  // .OR., .AND., .EQV., .NEQV.
  NOT,  // yes, this binds less tightly in Fortran than .OR./.AND./&c. do
  DefinedBinary
};

template<typename A> constexpr Precedence ToPrecedence{Precedence::Primary};

template<typename T>
constexpr Precedence ToPrecedence<Parentheses<T>>{Precedence::Parenthesize};
template<int KIND>
constexpr Precedence ToPrecedence<ComplexConstructor<KIND>>{
    Precedence::Parenthesize};
template<typename T>
constexpr Precedence ToPrecedence<Negate<T>>{Precedence::Negate};
template<typename T>
constexpr Precedence ToPrecedence<Power<T>>{Precedence::Power};
template<typename T>
constexpr Precedence ToPrecedence<RealToIntPower<T>>{Precedence::Power};
template<typename T>
constexpr Precedence ToPrecedence<Multiply<T>>{Precedence::Multiplicative};
template<typename T>
constexpr Precedence ToPrecedence<Divide<T>>{Precedence::Multiplicative};
template<typename T>
constexpr Precedence ToPrecedence<Add<T>>{Precedence::Additive};
template<typename T>
constexpr Precedence ToPrecedence<Subtract<T>>{Precedence::Additive};
template<int KIND>
constexpr Precedence ToPrecedence<Concat<KIND>>{Precedence::Additive};
template<typename T>
constexpr Precedence ToPrecedence<Relational<T>>{Precedence::Relational};
template<int KIND>
constexpr Precedence ToPrecedence<LogicalOperation<KIND>>{Precedence::Logical};
template<int KIND>
constexpr Precedence ToPrecedence<Not<KIND>>{Precedence::NOT};

template<typename T>
static constexpr Precedence GetPrecedence(const Expr<T> &expr) {
  return std::visit(
      [](const auto &x) { return ToPrecedence<std::decay_t<decltype(x)>>; },
      expr.u);
}
template<TypeCategory CAT>
static constexpr Precedence GetPrecedence(const Expr<SomeKind<CAT>> &expr) {
  return std::visit([](const auto &x) { return GetPrecedence(x); }, expr.u);
}
static constexpr Precedence GetPrecedence(const Expr<SomeDerived> &expr) {
  return std::visit(
      [](const auto &x) { return ToPrecedence<std::decay_t<decltype(x)>>; },
      expr.u);
}
static constexpr Precedence GetPrecedence(const Expr<SomeType> &expr) {
  return std::visit(
      common::visitors{
          [](const BOZLiteralConstant &) { return Precedence::Primary; },
          [](const NullPointer &) { return Precedence::Primary; },
          [](const auto &x) { return GetPrecedence(x); },
      },
      expr.u);
}

template<typename D, typename R, typename... O>
std::ostream &Operation<D, R, O...>::AsFortran(std::ostream &o) const {
  static constexpr Precedence lhsPrec{ToPrecedence<Operand<0>>};
  o << derived().Prefix();
  if constexpr (operands == 1) {
    bool parens{lhsPrec != Precedence::Primary};
    if (parens) {
      o << '(';
    }
    o << left();
    if (parens) {
      o << ')';
    }
  } else {
    static constexpr Precedence thisPrec{ToPrecedence<D>};
    bool lhsParens{lhsPrec == Precedence::Parenthesize || lhsPrec > thisPrec ||
        (lhsPrec == thisPrec && lhsPrec == Precedence::Power)};
    if (lhsParens) {
      o << '(';
    }
    o << left();
    if (lhsParens) {
      o << ')';
    }
    static constexpr Precedence rhsPrec{ToPrecedence<Operand<1>>};
    bool rhsParens{rhsPrec == Precedence::Parenthesize || rhsPrec > thisPrec};
    if (rhsParens) {
      o << '(';
    }
    o << derived().Infix() << right();
    if (rhsParens) {
      o << ')';
    }
  }
  return o << derived().Suffix();
}

template<typename TO, TypeCategory FROMCAT>
std::ostream &Convert<TO, FROMCAT>::AsFortran(std::ostream &o) const {
  static_assert(TO::category == TypeCategory::Integer ||
      TO::category == TypeCategory::Real ||
      TO::category == TypeCategory::Character ||
      TO::category == TypeCategory::Logical || !"Convert<> to bad category!");
  if constexpr (TO::category == TypeCategory::Character) {
    this->left().AsFortran(o << "achar(iachar(") << ')';
  } else if constexpr (TO::category == TypeCategory::Integer) {
    this->left().AsFortran(o << "int(");
  } else if constexpr (TO::category == TypeCategory::Real) {
    this->left().AsFortran(o << "real(");
  } else {
    this->left().AsFortran(o << "logical(");
  }
  return o << ",kind=" << TO::kind << ')';
}

template<typename A> const char *Relational<A>::Infix() const {
  switch (opr) {
  case RelationalOperator::LT: return "<";
  case RelationalOperator::LE: return "<=";
  case RelationalOperator::EQ: return "==";
  case RelationalOperator::NE: return "/=";
  case RelationalOperator::GE: return ">=";
  case RelationalOperator::GT: return ">";
  }
  return nullptr;
}

std::ostream &Relational<SomeType>::AsFortran(std::ostream &o) const {
  std::visit([&](const auto &rel) { rel.AsFortran(o); }, u);
  return o;
}

template<int KIND> const char *LogicalOperation<KIND>::Infix() const {
  switch (logicalOperator) {
  case LogicalOperator::And: return ".and.";
  case LogicalOperator::Or: return ".or.";
  case LogicalOperator::Eqv: return ".eqv.";
  case LogicalOperator::Neqv: return ".neqv.";
  }
  return nullptr;
}

template<typename T>
std::ostream &Emit(
    std::ostream &o, const common::CopyableIndirection<Expr<T>> &expr) {
  return expr.value().AsFortran(o);
}

template<typename T>
std::ostream &Emit(std::ostream &, const ArrayConstructorValues<T> &);

template<typename T>
std::ostream &Emit(std::ostream &o, const ImpliedDo<T> &implDo) {
  o << '(';
  Emit(o, implDo.values());
  o << ',' << ImpliedDoIndex::Result::AsFortran()
    << "::" << implDo.name().ToString() << '=';
  implDo.lower().AsFortran(o) << ',';
  implDo.upper().AsFortran(o) << ',';
  implDo.stride().AsFortran(o) << ')';
  return o;
}

template<typename T>
std::ostream &Emit(std::ostream &o, const ArrayConstructorValues<T> &values) {
  const char *sep{""};
  for (const auto &value : values.values()) {
    o << sep;
    std::visit([&](const auto &x) { Emit(o, x); }, value.u);
    sep = ",";
  }
  return o;
}

template<typename T>
std::ostream &ArrayConstructor<T>::AsFortran(std::ostream &o) const {
  o << '[' << GetType().AsFortran() << "::";
  Emit(o, *this);
  return o << ']';
}

template<int KIND>
std::ostream &ArrayConstructor<Type<TypeCategory::Character, KIND>>::AsFortran(
    std::ostream &o) const {
  std::stringstream len;
  LEN().AsFortran(len);
  o << '[' << GetType().AsFortran(len.str()) << "::";
  Emit(o, *this);
  return o << ']';
}

std::ostream &ArrayConstructor<SomeDerived>::AsFortran(std::ostream &o) const {
  o << '[' << GetType().AsFortran() << "::";
  Emit(o, *this);
  return o << ']';
}

template<typename RESULT>
std::ostream &ExpressionBase<RESULT>::AsFortran(std::ostream &o) const {
  std::visit(
      common::visitors{
          [&](const BOZLiteralConstant &x) {
            o << "z'" << x.Hexadecimal() << "'";
          },
          [&](const NullPointer &) { o << "NULL()"; },
          [&](const common::CopyableIndirection<Substring> &s) {
            s.value().AsFortran(o);
          },
          [&](const ImpliedDoIndex &i) { o << i.name.ToString(); },
          [&](const auto &x) { x.AsFortran(o); },
      },
      derived().u);
  return o;
}

std::ostream &StructureConstructor::AsFortran(std::ostream &o) const {
  DerivedTypeSpecAsFortran(o, *derivedTypeSpec_);
  if (values_.empty()) {
    o << '(';
  } else {
    char ch{'('};
    for (const auto &[symbol, value] : values_) {
      value.value().AsFortran(o << ch << symbol->name().ToString() << '=');
      ch = ',';
    }
  }
  return o << ')';
}

std::ostream &DerivedTypeSpecAsFortran(
    std::ostream &o, const semantics::DerivedTypeSpec &spec) {
  o << spec.typeSymbol().name().ToString();
  if (!spec.parameters().empty()) {
    char ch{'('};
    for (const auto &[name, value] : spec.parameters()) {
      value.GetExplicit()->AsFortran(o << ch << name.ToString() << '=');
      ch = ',';
    }
    o << ')';
  }
  return o;
}

INSTANTIATE_CONSTANT_TEMPLATES
INSTANTIATE_EXPRESSION_TEMPLATES
// TODO variable templates and call templates?
}
