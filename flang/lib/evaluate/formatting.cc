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
#include "fold.h"
#include "tools.h"
#include "../parser/characters.h"
#include "../semantics/symbol.h"

namespace Fortran::evaluate {

bool formatForPGF90{false};

static void ShapeAsFortran(std::ostream &o, const ConstantSubscripts &shape) {
  if (GetRank(shape) > 1) {
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
      o << Result::kind << '_' << parser::QuoteCharacterLiteral(value, true);
    } else if constexpr (Result::category == TypeCategory::Logical) {
      if (value.IsTrue()) {
        o << ".true.";
      } else {
        o << ".false.";
      }
      o << '_' << Result::kind;
    } else {
      StructureConstructor{result_.derivedTypeSpec(), value}.AsFortran(o);
    }
  }
  if (Rank() > 0) {
    o << ']';
  }
  ShapeAsFortran(o, shape());
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
  auto total{static_cast<ConstantSubscript>(size())};
  for (ConstantSubscript j{0}; j < total; ++j) {
    Scalar<Result> value{values_.substr(j * length_, length_)};
    if (j > 0) {
      o << ',';
    }
    if (Result::kind != 1 || !formatForPGF90) {
      o << Result::kind << '_';
    }
    o << parser::QuoteCharacterLiteral(value);
  }
  if (Rank() > 0) {
    o << ']';
  }
  ShapeAsFortran(o, shape());
  return o;
}

std::ostream &ActualArgument::AssumedType::AsFortran(std::ostream &o) const {
  return o << symbol_->name().ToString();
}

std::ostream &ActualArgument::AsFortran(std::ostream &o) const {
  if (keyword.has_value()) {
    o << keyword->ToString() << '=';
  }
  if (isAlternateReturn) {
    o << '*';
  }
  if (const auto *expr{UnwrapExpr()}) {
    return expr->AsFortran(o);
  } else {
    return std::get<AssumedType>(u_).AsFortran(o);
  }
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

enum class Precedence {  // in increasing order for sane comparisons
  DefinedBinary,
  Or,
  And,
  Equivalence,  // .EQV., .NEQV.
  Not,  // which binds *less* tightly in Fortran than relations
  Relational,
  Additive,  // +, -, and (arbitrarily) //
  Negate,  // which binds *less* tightly than *, /, **
  Multiplicative,  // *, /
  Power,  // **, which is right-associative unlike the other dyadic operators
  DefinedUnary,
  Parenthesize,  // (x), (real, imaginary)
  Constant,  // parenthesize if negative integer/real operand
  Primary,  // don't parenthesize
};

template<typename A> constexpr Precedence ToPrecedence{Precedence::Primary};

template<int KIND>
constexpr Precedence ToPrecedence<LogicalOperation<KIND>>{Precedence::Or};
template<int KIND>
constexpr Precedence ToPrecedence<Not<KIND>>{Precedence::Not};
template<typename T>
constexpr Precedence ToPrecedence<Relational<T>>{Precedence::Relational};
template<typename T>
constexpr Precedence ToPrecedence<Add<T>>{Precedence::Additive};
template<typename T>
constexpr Precedence ToPrecedence<Subtract<T>>{Precedence::Additive};
template<int KIND>
constexpr Precedence ToPrecedence<Concat<KIND>>{Precedence::Additive};
template<typename T>
constexpr Precedence ToPrecedence<Negate<T>>{Precedence::Negate};
template<typename T>
constexpr Precedence ToPrecedence<Multiply<T>>{Precedence::Multiplicative};
template<typename T>
constexpr Precedence ToPrecedence<Divide<T>>{Precedence::Multiplicative};
template<typename T>
constexpr Precedence ToPrecedence<Power<T>>{Precedence::Power};
template<typename T>
constexpr Precedence ToPrecedence<RealToIntPower<T>>{Precedence::Power};
template<typename T>
constexpr Precedence ToPrecedence<Constant<T>>{Precedence::Constant};
template<int KIND>
constexpr Precedence ToPrecedence<SetLength<KIND>>{Precedence::Constant};
template<typename T>
constexpr Precedence ToPrecedence<Parentheses<T>>{Precedence::Parenthesize};
template<int KIND>
constexpr Precedence ToPrecedence<ComplexConstructor<KIND>>{
    Precedence::Parenthesize};

template<typename T>
static constexpr Precedence GetPrecedence(const Expr<T> &expr) {
  return std::visit(
      [](const auto &x) {
        static constexpr Precedence prec{
            ToPrecedence<std::decay_t<decltype(x)>>};
        if constexpr (prec == Precedence::Or) {
          // Distinguish the four logical binary operations.
          switch (x.logicalOperator) {
          case LogicalOperator::And: return Precedence::And;
          case LogicalOperator::Or: return Precedence::Or;
          case LogicalOperator::Eqv:
          case LogicalOperator::Neqv:
            return Precedence::Equivalence;
            CRASH_NO_CASE;
          }
        }
        return prec;
      },
      expr.u);
}
template<TypeCategory CAT>
static constexpr Precedence GetPrecedence(const Expr<SomeKind<CAT>> &expr) {
  return std::visit([](const auto &x) { return GetPrecedence(x); }, expr.u);
}

template<typename T> static bool IsNegatedScalarConstant(const Expr<T> &expr) {
  static constexpr TypeCategory cat{T::category};
  if constexpr (cat == TypeCategory::Integer || cat == TypeCategory::Real) {
    if (auto n{GetScalarConstantValue<T>(expr)}) {
      return n->IsNegative();
    }
  }
  return false;
}

template<TypeCategory CAT>
static bool IsNegatedScalarConstant(const Expr<SomeKind<CAT>> &expr) {
  return std::visit(
      [](const auto &x) { return IsNegatedScalarConstant(x); }, expr.u);
}

template<typename D, typename R, typename... O>
std::ostream &Operation<D, R, O...>::AsFortran(std::ostream &o) const {
  Precedence lhsPrec{GetPrecedence(left())};
  o << derived().Prefix();
  static constexpr Precedence thisPrec{ToPrecedence<D>};
  if constexpr (operands == 1) {
    bool parens{lhsPrec < Precedence::Constant &&
        !(thisPrec == Precedence::Not && lhsPrec == Precedence::Relational)};
    o << (parens ? "(" : "") << left() << (parens ? ")" : "");
  } else {
    bool lhsParens{lhsPrec == Precedence::Parenthesize || lhsPrec < thisPrec ||
        (lhsPrec == thisPrec && lhsPrec == Precedence::Power) ||
        (thisPrec != Precedence::Additive && lhsPrec == Precedence::Constant &&
            IsNegatedScalarConstant(left()))};
    o << (lhsParens ? "(" : "") << left() << (lhsParens ? ")" : "");
    o << derived().Infix();
    Precedence rhsPrec{GetPrecedence(right())};
    bool rhsParens{rhsPrec == Precedence::Parenthesize ||
        rhsPrec == Precedence::Negate || rhsPrec < thisPrec ||
        (rhsPrec == Precedence::Constant && IsNegatedScalarConstant(right()))};
    o << (rhsParens ? "(" : "") << right() << (rhsParens ? ")" : "");
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
std::ostream &EmitArray(std::ostream &o, const Expr<T> &expr) {
  return expr.AsFortran(o);
}

template<typename T>
std::ostream &EmitArray(std::ostream &, const ArrayConstructorValues<T> &);

template<typename T>
std::ostream &EmitArray(std::ostream &o, const ImpliedDo<T> &implDo) {
  o << '(';
  EmitArray(o, implDo.values());
  o << ',' << ImpliedDoIndex::Result::AsFortran()
    << "::" << implDo.name().ToString() << '=';
  implDo.lower().AsFortran(o) << ',';
  implDo.upper().AsFortran(o) << ',';
  implDo.stride().AsFortran(o) << ')';
  return o;
}

template<typename T>
std::ostream &EmitArray(
    std::ostream &o, const ArrayConstructorValues<T> &values) {
  const char *sep{""};
  for (const auto &value : values) {
    o << sep;
    std::visit([&](const auto &x) { EmitArray(o, x); }, value.u);
    sep = ",";
  }
  return o;
}

template<typename T>
std::ostream &ArrayConstructor<T>::AsFortran(std::ostream &o) const {
  o << '[' << GetType().AsFortran() << "::";
  EmitArray(o, *this);
  return o << ']';
}

template<int KIND>
std::ostream &ArrayConstructor<Type<TypeCategory::Character, KIND>>::AsFortran(
    std::ostream &o) const {
  std::stringstream len;
  LEN().AsFortran(len);
  o << '[' << GetType().AsFortran(len.str()) << "::";
  EmitArray(o, *this);
  return o << ']';
}

std::ostream &ArrayConstructor<SomeDerived>::AsFortran(std::ostream &o) const {
  o << '[' << GetType().AsFortran() << "::";
  EmitArray(o, *this);
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
  o << DerivedTypeSpecAsFortran(result_.derivedTypeSpec());
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

std::string DynamicType::AsFortran() const {
  if (derived_ != nullptr) {
    CHECK(category_ == TypeCategory::Derived);
    return DerivedTypeSpecAsFortran(*derived_);
  } else if (charLength_ != nullptr) {
    std::string result{"CHARACTER(KIND="s + std::to_string(kind_) + ",LEN="};
    if (charLength_->isAssumed()) {
      result += '*';
    } else if (charLength_->isDeferred()) {
      result += ':';
    } else if (const auto &length{charLength_->GetExplicit()}) {
      std::stringstream ss;
      length->AsFortran(ss);
      result += ss.str();
    }
    return result + ')';
  } else if (IsUnlimitedPolymorphic()) {
    return "CLASS(*)";
  } else if (IsAssumedType()) {
    return "TYPE(*)";
  } else if (kind_ == 0) {
    return "(typeless intrinsic function argument)";
  } else {
    return EnumToString(category_) + '(' + std::to_string(kind_) + ')';
  }
}

std::string DynamicType::AsFortran(std::string &&charLenExpr) const {
  if (!charLenExpr.empty() && category_ == TypeCategory::Character) {
    return "CHARACTER(KIND=" + std::to_string(kind_) +
        ",LEN=" + std::move(charLenExpr) + ')';
  } else {
    return AsFortran();
  }
}

std::string SomeDerived::AsFortran() const {
  if (IsUnlimitedPolymorphic()) {
    return "CLASS(*)";
  } else {
    return "TYPE("s + DerivedTypeSpecAsFortran(derivedTypeSpec()) + ')';
  }
}

std::string DerivedTypeSpecAsFortran(const semantics::DerivedTypeSpec &spec) {
  if (spec.HasActualParameters()) {
    std::stringstream ss;
    ss << spec.typeSymbol().name().ToString();
    char ch{'('};
    for (const auto &[name, value] : spec.parameters()) {
      ss << ch << name.ToString() << '=';
      ch = ',';
      if (value.isAssumed()) {
        ss << '*';
      } else if (value.isDeferred()) {
        ss << ':';
      } else {
        value.GetExplicit()->AsFortran(ss);
      }
    }
    ss << ')';
    return ss.str();
  } else {
    return spec.typeSymbol().name().ToString();
  }
}

std::ostream &EmitVar(std::ostream &o, const Symbol &symbol) {
  return o << symbol.name().ToString();
}

std::ostream &EmitVar(std::ostream &o, const std::string &lit) {
  return o << parser::QuoteCharacterLiteral(lit);
}

std::ostream &EmitVar(std::ostream &o, const std::u16string &lit) {
  return o << parser::QuoteCharacterLiteral(lit);
}

std::ostream &EmitVar(std::ostream &o, const std::u32string &lit) {
  return o << parser::QuoteCharacterLiteral(lit);
}

template<typename A> std::ostream &EmitVar(std::ostream &o, const A &x) {
  return x.AsFortran(o);
}

template<typename A>
std::ostream &EmitVar(std::ostream &o, const A *p, const char *kw = nullptr) {
  if (p != nullptr) {
    if (kw != nullptr) {
      o << kw;
    }
    EmitVar(o, *p);
  }
  return o;
}

template<typename A>
std::ostream &EmitVar(
    std::ostream &o, const std::optional<A> &x, const char *kw = nullptr) {
  if (x.has_value()) {
    if (kw != nullptr) {
      o << kw;
    }
    EmitVar(o, *x);
  }
  return o;
}

template<typename A, bool COPY>
std::ostream &EmitVar(std::ostream &o, const common::Indirection<A, COPY> &p,
    const char *kw = nullptr) {
  if (kw != nullptr) {
    o << kw;
  }
  EmitVar(o, p.value());
  return o;
}

template<typename A>
std::ostream &EmitVar(std::ostream &o, const std::shared_ptr<A> &p) {
  CHECK(p != nullptr);
  return EmitVar(o, *p);
}

template<typename... A>
std::ostream &EmitVar(std::ostream &o, const std::variant<A...> &u) {
  std::visit([&](const auto &x) { EmitVar(o, x); }, u);
  return o;
}

std::ostream &BaseObject::AsFortran(std::ostream &o) const {
  return EmitVar(o, u);
}

template<int KIND>
std::ostream &TypeParamInquiry<KIND>::AsFortran(std::ostream &o) const {
  if (base_.has_value()) {
    return base_->AsFortran(o) << '%';
  }
  return EmitVar(o, *parameter_);
}

std::ostream &Component::AsFortran(std::ostream &o) const {
  base_.value().AsFortran(o);
  return EmitVar(o << '%', *symbol_);
}

std::ostream &NamedEntity::AsFortran(std::ostream &o) const {
  std::visit(
      common::visitors{
          [&](const Symbol *s) { EmitVar(o, *s); },
          [&](const Component &c) { c.AsFortran(o); },
      },
      u_);
  return o;
}

std::ostream &Triplet::AsFortran(std::ostream &o) const {
  EmitVar(o, lower_) << ':';
  EmitVar(o, upper_);
  EmitVar(o << ':', stride_.value());
  return o;
}

std::ostream &Subscript::AsFortran(std::ostream &o) const {
  return EmitVar(o, u);
}

std::ostream &ArrayRef::AsFortran(std::ostream &o) const {
  base_.AsFortran(o);
  char separator{'('};
  for (const Subscript &ss : subscript_) {
    ss.AsFortran(o << separator);
    separator = ',';
  }
  return o << ')';
}

std::ostream &CoarrayRef::AsFortran(std::ostream &o) const {
  bool first{true};
  for (const Symbol *part : base_) {
    if (first) {
      first = false;
    } else {
      o << '%';
    }
    EmitVar(o, *part);
  }
  char separator{'('};
  for (const auto &sscript : subscript_) {
    EmitVar(o << separator, sscript);
    separator = ',';
  }
  if (separator == ',') {
    o << ')';
  }
  separator = '[';
  for (const auto &css : cosubscript_) {
    EmitVar(o << separator, css);
    separator = ',';
  }
  if (stat_.has_value()) {
    EmitVar(o << separator, stat_, "STAT=");
    separator = ',';
  }
  if (team_.has_value()) {
    EmitVar(
        o << separator, team_, teamIsTeamNumber_ ? "TEAM_NUMBER=" : "TEAM=");
  }
  return o << ']';
}

std::ostream &DataRef::AsFortran(std::ostream &o) const {
  return EmitVar(o, u);
}

std::ostream &Substring::AsFortran(std::ostream &o) const {
  EmitVar(o, parent_) << '(';
  EmitVar(o, lower_) << ':';
  return EmitVar(o, upper_) << ')';
}

std::ostream &ComplexPart::AsFortran(std::ostream &o) const {
  return complex_.AsFortran(o) << '%' << EnumToString(part_);
}

std::ostream &ProcedureDesignator::AsFortran(std::ostream &o) const {
  return EmitVar(o, u);
}

template<typename T>
std::ostream &Designator<T>::AsFortran(std::ostream &o) const {
  std::visit(
      common::visitors{
          [&](const Symbol *sym) { EmitVar(o, *sym); },
          [&](const auto &x) { x.AsFortran(o); },
      },
      u);
  return o;
}

std::ostream &DescriptorInquiry::AsFortran(std::ostream &o) const {
  switch (field_) {
  case Field::LowerBound: o << "lbound("; break;
  case Field::Extent: o << "size("; break;
  case Field::Stride: o << "%STRIDE("; break;
  case Field::Rank: o << "rank("; break;
  }
  base_.AsFortran(o);
  if (dimension_ >= 0) {
    o << ",dim=" << (dimension_ + 1);
  }
  return o << ')';
}

INSTANTIATE_CONSTANT_TEMPLATES
INSTANTIATE_EXPRESSION_TEMPLATES
INSTANTIATE_VARIABLE_TEMPLATES
}
