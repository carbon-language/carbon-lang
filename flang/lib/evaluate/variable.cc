// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "variable.h"
#include "fold.h"
#include "tools.h"
#include "../common/idioms.h"
#include "../parser/char-block.h"
#include "../parser/characters.h"
#include "../parser/message.h"
#include "../semantics/symbol.h"
#include <ostream>
#include <type_traits>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

// Constructors, accessors, mutators

Triplet::Triplet() : stride_{Expr<SubscriptInteger>{1}} {}

Triplet::Triplet(std::optional<Expr<SubscriptInteger>> &&l,
    std::optional<Expr<SubscriptInteger>> &&u,
    std::optional<Expr<SubscriptInteger>> &&s)
  : stride_{s.has_value() ? std::move(*s) : Expr<SubscriptInteger>{1}} {
  if (l.has_value()) {
    lower_.emplace(std::move(*l));
  }
  if (u.has_value()) {
    upper_.emplace(std::move(*u));
  }
}

std::optional<Expr<SubscriptInteger>> Triplet::lower() const {
  if (lower_) {
    return {**lower_};
  }
  return std::nullopt;
}

std::optional<Expr<SubscriptInteger>> Triplet::upper() const {
  if (upper_) {
    return {**upper_};
  }
  return std::nullopt;
}

const Expr<SubscriptInteger> &Triplet::stride() const { return *stride_; }

bool Triplet::IsStrideOne() const {
  if (auto stride{ToInt64(*stride_)}) {
    return stride == 1;
  } else {
    return false;
  }
}

CoarrayRef::CoarrayRef(std::vector<const Symbol *> &&c,
    std::vector<Expr<SubscriptInteger>> &&ss,
    std::vector<Expr<SubscriptInteger>> &&css)
  : base_(std::move(c)), subscript_(std::move(ss)),
    cosubscript_(std::move(css)) {
  CHECK(!base_.empty());
}

std::optional<Expr<SomeInteger>> CoarrayRef::stat() const {
  if (stat_.has_value()) {
    return {**stat_};
  } else {
    return std::nullopt;
  }
}

std::optional<Expr<SomeInteger>> CoarrayRef::team() const {
  if (team_.has_value()) {
    return {**team_};
  } else {
    return std::nullopt;
  }
}

CoarrayRef &CoarrayRef::set_stat(Expr<SomeInteger> &&v) {
  CHECK(IsVariable(v));
  stat_.emplace(std::move(v));
  return *this;
}

CoarrayRef &CoarrayRef::set_team(Expr<SomeInteger> &&v, bool isTeamNumber) {
  CHECK(IsVariable(v));
  team_.emplace(std::move(v));
  teamIsTeamNumber_ = isTeamNumber;
  return *this;
}

void Substring::SetBounds(std::optional<Expr<SubscriptInteger>> &lower,
    std::optional<Expr<SubscriptInteger>> &upper) {
  if (lower.has_value()) {
    lower_.emplace(std::move(*lower));
  }
  if (upper.has_value()) {
    upper_.emplace(std::move(*upper));
  }
}

Expr<SubscriptInteger> Substring::lower() const {
  if (lower_.has_value()) {
    return **lower_;
  } else {
    return AsExpr(Constant<SubscriptInteger>{1});
  }
}

Expr<SubscriptInteger> Substring::upper() const {
  if (upper_.has_value()) {
    return **upper_;
  } else {
    return std::visit(
        common::visitors{
            [](const DataRef &dataRef) { return dataRef.LEN(); },
            [](const StaticDataObject::Pointer &object) {
              return AsExpr(Constant<SubscriptInteger>{object->data().size()});
            },
        },
        parent_);
  }
}

std::optional<Expr<SomeCharacter>> Substring::Fold(FoldingContext &context) {
  if (!lower_.has_value()) {
    lower_ = AsExpr(Constant<SubscriptInteger>{1});
  }
  *lower_ = evaluate::Fold(context, std::move(**lower_));
  std::optional<std::int64_t> lbi{ToInt64(**lower_)};
  if (lbi.has_value() && *lbi < 1) {
    context.messages().Say(
        "lower bound (%jd) on substring is less than one"_en_US,
        static_cast<std::intmax_t>(*lbi));
    *lbi = 1;
    lower_ = AsExpr(Constant<SubscriptInteger>{1});
  }
  if (!upper_.has_value()) {
    upper_ = upper();
  }
  *upper_ = evaluate::Fold(context, std::move(**upper_));
  if (std::optional<std::int64_t> ubi{ToInt64(**upper_)}) {
    auto *literal{std::get_if<StaticDataObject::Pointer>(&parent_)};
    std::optional<std::int64_t> length;
    if (literal != nullptr) {
      length = (*literal)->data().size();
    } else if (const Symbol * symbol{GetLastSymbol()}) {
      if (const semantics::DeclTypeSpec * type{symbol->GetType()}) {
        if (type->category() == semantics::DeclTypeSpec::Character) {
          length = ToInt64(type->characterTypeSpec().length().GetExplicit());
        }
      }
    }
    if (*ubi < 1 || (lbi.has_value() && *ubi < *lbi)) {
      // Zero-length string: canonicalize
      *lbi = 1, *ubi = 0;
      lower_ = AsExpr(Constant<SubscriptInteger>{*lbi});
      upper_ = AsExpr(Constant<SubscriptInteger>{*ubi});
    } else if (length.has_value() && *ubi > *length) {
      context.messages().Say("upper bound (&jd) on substring is greater "
                             "than character length (%jd)"_en_US,
          static_cast<std::intmax_t>(*ubi), static_cast<std::int64_t>(*length));
      *ubi = *length;
    }
    if (lbi.has_value()) {
      if (literal != nullptr || *ubi < *lbi) {
        auto newStaticData{StaticDataObject::Create()};
        auto items{*ubi - *lbi + 1};
        auto width{(*literal)->itemBytes()};
        auto bytes{items * width};
        auto startByte{(*lbi - 1) * width};
        const auto *from{&(*literal)->data()[0] + startByte};
        for (auto j{0}; j < bytes; ++j) {
          newStaticData->data().push_back(from[j]);
        }
        parent_ = newStaticData;
        lower_ = AsExpr(Constant<SubscriptInteger>{1});
        std::int64_t length = newStaticData->data().size();
        upper_ = AsExpr(Constant<SubscriptInteger>{length});
        switch (width) {
        case 1:
          return {
              AsCategoryExpr(AsExpr(Constant<Type<TypeCategory::Character, 1>>{
                  *newStaticData->AsString()}))};
        case 2:
          return {AsCategoryExpr(Constant<Type<TypeCategory::Character, 2>>{
              *newStaticData->AsU16String()})};
        case 4:
          return {AsCategoryExpr(Constant<Type<TypeCategory::Character, 4>>{
              *newStaticData->AsU32String()})};
        default: CRASH_NO_CASE;
        }
      }
    }
  }
  return std::nullopt;
}

// Variable formatting

std::ostream &Emit(std::ostream &o, const Symbol &symbol) {
  return o << symbol.name().ToString();
}

std::ostream &Emit(std::ostream &o, const std::string &lit) {
  return o << parser::QuoteCharacterLiteral(lit);
}

std::ostream &Emit(std::ostream &o, const std::u16string &lit) {
  return o << parser::QuoteCharacterLiteral(lit);
}

std::ostream &Emit(std::ostream &o, const std::u32string &lit) {
  return o << parser::QuoteCharacterLiteral(lit);
}

template<typename A> std::ostream &Emit(std::ostream &o, const A &x) {
  return x.AsFortran(o);
}

template<typename A>
std::ostream &Emit(std::ostream &o, const A *p, const char *kw = nullptr) {
  if (p != nullptr) {
    if (kw != nullptr) {
      o << kw;
    }
    Emit(o, *p);
  }
  return o;
}

template<typename A>
std::ostream &Emit(
    std::ostream &o, const std::optional<A> &x, const char *kw = nullptr) {
  if (x.has_value()) {
    if (kw != nullptr) {
      o << kw;
    }
    Emit(o, *x);
  }
  return o;
}

template<typename A>
std::ostream &Emit(std::ostream &o, const CopyableIndirection<A> &p,
    const char *kw = nullptr) {
  if (kw != nullptr) {
    o << kw;
  }
  Emit(o, *p);
  return o;
}

template<typename A>
std::ostream &Emit(std::ostream &o, const std::shared_ptr<A> &p) {
  CHECK(p != nullptr);
  return Emit(o, *p);
}

template<typename... A>
std::ostream &Emit(std::ostream &o, const std::variant<A...> &u) {
  std::visit([&](const auto &x) { Emit(o, x); }, u);
  return o;
}

std::ostream &BaseObject::AsFortran(std::ostream &o) const {
  return Emit(o, u);
}

template<int KIND>
std::ostream &TypeParamInquiry<KIND>::AsFortran(std::ostream &o) const {
  std::visit(
      common::visitors{
          [&](const Symbol *sym) {
            if (sym != nullptr) {
              Emit(o, *sym) << '%';
            }
          },
          [&](const Component &comp) { Emit(o, comp) << '%'; },
      },
      base_);
  return Emit(o, *parameter_);
}

std::ostream &Component::AsFortran(std::ostream &o) const {
  base_->AsFortran(o);
  return Emit(o << '%', *symbol_);
}

std::ostream &Triplet::AsFortran(std::ostream &o) const {
  Emit(o, lower_) << ':';
  Emit(o, upper_);
  Emit(o << ':', *stride_);
  return o;
}

std::ostream &Subscript::AsFortran(std::ostream &o) const { return Emit(o, u); }

std::ostream &ArrayRef::AsFortran(std::ostream &o) const {
  Emit(o, base_);
  char separator{'('};
  for (const Subscript &ss : subscript_) {
    ss.AsFortran(o << separator);
    separator = ',';
  }
  return o << ')';
}

std::ostream &CoarrayRef::AsFortran(std::ostream &o) const {
  for (const Symbol *sym : base_) {
    Emit(o, *sym);
  }
  char separator{'('};
  for (const auto &ss : subscript_) {
    Emit(o << separator, ss);
    separator = ',';
  }
  if (separator == ',') {
    o << ')';
  }
  separator = '[';
  for (const auto &css : cosubscript_) {
    Emit(o << separator, css);
    separator = ',';
  }
  if (stat_.has_value()) {
    Emit(o << separator, stat_, "STAT=");
    separator = ',';
  }
  if (team_.has_value()) {
    Emit(o << separator, team_, teamIsTeamNumber_ ? "TEAM_NUMBER=" : "TEAM=");
  }
  return o << ']';
}

std::ostream &DataRef::AsFortran(std::ostream &o) const { return Emit(o, u); }

std::ostream &Substring::AsFortran(std::ostream &o) const {
  Emit(o, parent_) << '(';
  Emit(o, lower_) << ':';
  return Emit(o, upper_);
}

std::ostream &ComplexPart::AsFortran(std::ostream &o) const {
  return complex_.AsFortran(o) << '%' << EnumToString(part_);
}

std::ostream &ProcedureDesignator::AsFortran(std::ostream &o) const {
  return Emit(o, u);
}

template<typename T>
std::ostream &Designator<T>::AsFortran(std::ostream &o) const {
  std::visit(
      common::visitors{
          [&](const Symbol *sym) { Emit(o, *sym); },
          [&](const auto &x) { x.AsFortran(o); },
      },
      u);
  return o;
}

// LEN()
static Expr<SubscriptInteger> SymbolLEN(const Symbol &sym) {
  return AsExpr(Constant<SubscriptInteger>{0});  // TODO
}

Expr<SubscriptInteger> BaseObject::LEN() const {
  return std::visit(
      common::visitors{
          [](const Symbol *symbol) { return SymbolLEN(*symbol); },
          [](const StaticDataObject::Pointer &object) {
            return AsExpr(Constant<SubscriptInteger>{object->data().size()});
          },
      },
      u);
}

Expr<SubscriptInteger> Component::LEN() const {
  return SymbolLEN(GetLastSymbol());
}

Expr<SubscriptInteger> ArrayRef::LEN() const {
  return std::visit(
      common::visitors{
          [](const Symbol *symbol) { return SymbolLEN(*symbol); },
          [](const Component &component) { return component.LEN(); },
      },
      base_);
}

Expr<SubscriptInteger> CoarrayRef::LEN() const {
  return SymbolLEN(*base_.back());
}

Expr<SubscriptInteger> DataRef::LEN() const {
  return std::visit(
      common::visitors{
          [](const Symbol *s) { return SymbolLEN(*s); },
          [](const auto &x) { return x.LEN(); },
      },
      u);
}

Expr<SubscriptInteger> Substring::LEN() const {
  return AsExpr(
      Extremum<SubscriptInteger>{AsExpr(Constant<SubscriptInteger>{0}),
          upper() - lower() + AsExpr(Constant<SubscriptInteger>{1})});
}

template<typename T> Expr<SubscriptInteger> Designator<T>::LEN() const {
  if constexpr (Result::category == TypeCategory::Character) {
    return std::visit(
        common::visitors{
            [](const Symbol *s) { return SymbolLEN(*s); },
            [](const Component &c) { return c.LEN(); },
            [](const auto &x) { return x.LEN(); },
        },
        u);
  } else {
    CHECK(!"LEN() on non-character Designator");
    return AsExpr(Constant<SubscriptInteger>{0});
  }
}

Expr<SubscriptInteger> ProcedureDesignator::LEN() const {
  return std::visit(
      common::visitors{
          [](const Symbol *s) { return SymbolLEN(*s); },
          [](const Component &c) { return c.LEN(); },
          [](const auto &) {
            CRASH_NO_CASE;
            return AsExpr(Constant<SubscriptInteger>{0});
          },
      },
      u);
}

// Rank()
int BaseObject::Rank() const {
  return std::visit(
      common::visitors{
          [](const Symbol *symbol) { return symbol->Rank(); },
          [](const StaticDataObject::Pointer &) { return 0; },
      },
      u);
}

int Component::Rank() const {
  if (int rank{symbol_->Rank()}; rank > 0) {
    return rank;
  }
  return base_->Rank();
}

int Subscript::Rank() const {
  return std::visit(
      common::visitors{
          [](const IndirectSubscriptIntegerExpr &x) { return x->Rank(); },
          [](const Triplet &) { return 1; },
      },
      u);
}

int ArrayRef::Rank() const {
  int rank{0};
  for (const auto &expr : subscript_) {
    rank += expr.Rank();
  }
  if (rank > 0) {
    return rank;
  }
  return std::visit(
      common::visitors{
          [=](const Symbol *s) { return 0; },
          [=](const Component &c) { return c.Rank(); },
      },
      base_);
}

int CoarrayRef::Rank() const {
  int rank{0};
  for (const auto &expr : subscript_) {
    rank += expr.Rank();
  }
  return rank;
}

int DataRef::Rank() const {
  return std::visit(
      // g++ 7.2 emits bogus warnings here and below when common::visitors{}
      // is used with a "const auto &" catch-all member, so a constexpr type
      // test has to be used instead.
      [](const auto &x) {
        if constexpr (std::is_same_v<std::decay_t<decltype(x)>,
                          const Symbol *>) {
          return x->Rank();
        } else {
          return x.Rank();
        }
      },
      u);
}

int Substring::Rank() const {
  return std::visit(
      common::visitors{
          [](const DataRef &dataRef) { return dataRef.Rank(); },
          [](const StaticDataObject::Pointer &) { return 0; },
      },
      parent_);
}

int ComplexPart::Rank() const { return complex_.Rank(); }
template<typename T> int Designator<T>::Rank() const {
  return std::visit(
      common::visitors{
          [](const Symbol *sym) { return sym->Rank(); },
          [](const auto &x) { return x.Rank(); },
      },
      u);
}

// GetBaseObject(), GetFirstSymbol(), & GetLastSymbol()
const Symbol &Component::GetFirstSymbol() const {
  return base_->GetFirstSymbol();
}

const Symbol &ArrayRef::GetFirstSymbol() const {
  return *std::visit(
      common::visitors{
          [](const Symbol *symbol) { return symbol; },
          [=](const Component &component) {
            return &component.GetFirstSymbol();
          },
      },
      base_);
}

const Symbol &ArrayRef::GetLastSymbol() const {
  return *std::visit(
      common::visitors{
          [](const Symbol *sym) { return sym; },
          [=](const Component &component) {
            return &component.GetLastSymbol();
          },
      },
      base_);
}

const Symbol &DataRef::GetFirstSymbol() const {
  return *std::visit(
      common::visitors{
          [](const Symbol *symbol) { return symbol; },
          [](const auto &x) { return &x.GetFirstSymbol(); },
      },
      u);
}

const Symbol &DataRef::GetLastSymbol() const {
  return *std::visit(
      common::visitors{
          [](const Symbol *symbol) { return symbol; },
          [](const auto &x) { return &x.GetLastSymbol(); },
      },
      u);
}

BaseObject Substring::GetBaseObject() const {
  return std::visit(
      common::visitors{
          [](const DataRef &dataRef) {
            return BaseObject{dataRef.GetFirstSymbol()};
          },
          [](StaticDataObject::Pointer pointer) {
            return BaseObject{std::move(pointer)};
          },
      },
      parent_);
}

const Symbol *Substring::GetLastSymbol() const {
  return std::visit(
      common::visitors{
          [](const DataRef &dataRef) { return &dataRef.GetLastSymbol(); },
          [](const auto &) -> const Symbol * { return nullptr; },
      },
      parent_);
}

template<typename T> BaseObject Designator<T>::GetBaseObject() const {
  return std::visit(
      common::visitors{
          [](const Symbol *symbol) { return BaseObject{*symbol}; },
          [](const auto &x) {
            if constexpr (std::is_same_v<std::decay_t<decltype(x)>,
                              Substring>) {
              return x.GetBaseObject();
            } else {
              return BaseObject{x.GetFirstSymbol()};
            }
          },
      },
      u);
}

template<typename T> const Symbol *Designator<T>::GetLastSymbol() const {
  return std::visit(
      common::visitors{
          [](const Symbol *symbol) { return symbol; },
          [](const auto &x) {
            if constexpr (std::is_same_v<std::decay_t<decltype(x)>,
                              Substring>) {
              return x.GetLastSymbol();
            } else {
              return &x.GetLastSymbol();
            }
          },
      },
      u);
}

template<typename T> std::optional<DynamicType> Designator<T>::GetType() const {
  if constexpr (IsLengthlessIntrinsicType<Result>) {
    return {Result::GetType()};
  } else {
    return GetSymbolType(GetLastSymbol());
  }
}

// Equality testing

bool BaseObject::operator==(const BaseObject &that) const {
  return u == that.u;
}
bool Component::operator==(const Component &that) const {
  return base_ == that.base_ && symbol_ == that.symbol_;
}
template<int KIND>
bool TypeParamInquiry<KIND>::operator==(
    const TypeParamInquiry<KIND> &that) const {
  return parameter_ == that.parameter_ && base_ == that.base_;
}
bool Triplet::operator==(const Triplet &that) const {
  return lower_ == that.lower_ && upper_ == that.upper_ &&
      *stride_ == *that.stride_;
}
bool ArrayRef::operator==(const ArrayRef &that) const {
  return base_ == that.base_ && subscript_ == that.subscript_;
}
bool CoarrayRef::operator==(const CoarrayRef &that) const {
  return base_ == that.base_ && subscript_ == that.subscript_ &&
      cosubscript_ == that.cosubscript_ && stat_ == that.stat_ &&
      team_ == that.team_ && teamIsTeamNumber_ == that.teamIsTeamNumber_;
}
bool Substring::operator==(const Substring &that) const {
  return parent_ == that.parent_ && lower_ == that.lower_ &&
      upper_ == that.upper_;
}
bool ComplexPart::operator==(const ComplexPart &that) const {
  return part_ == that.part_ && complex_ == that.complex_;
}
bool ProcedureRef::operator==(const ProcedureRef &that) const {
  return proc_ == that.proc_ && arguments_ == that.arguments_;
}

EXPAND_FOR_EACH_INTEGER_KIND(
    TEMPLATE_INSTANTIATION, template class TypeParamInquiry)
FOR_EACH_SPECIFIC_TYPE(template class Designator)
}
