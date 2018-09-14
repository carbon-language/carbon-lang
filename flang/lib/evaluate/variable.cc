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

#include "variable.h"
#include "tools.h"
#include "../common/idioms.h"
#include "../parser/char-block.h"
#include "../parser/characters.h"
#include "../parser/message.h"
#include "../semantics/symbol.h"
#include <ostream>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

// Constructors, accessors, mutators

Triplet::Triplet(std::optional<Expr<SubscriptInteger>> &&l,
    std::optional<Expr<SubscriptInteger>> &&u,
    std::optional<Expr<SubscriptInteger>> &&s) {
  if (l.has_value()) {
    lower_ = IndirectSubscriptIntegerExpr::Make(std::move(*l));
  }
  if (u.has_value()) {
    upper_ = IndirectSubscriptIntegerExpr::Make(std::move(*u));
  }
  if (s.has_value()) {
    stride_ = IndirectSubscriptIntegerExpr::Make(std::move(*s));
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

std::optional<Expr<SubscriptInteger>> Triplet::stride() const {
  if (stride_) {
    return {**stride_};
  }
  return std::nullopt;
}

CoarrayRef::CoarrayRef(std::vector<const Symbol *> &&c,
    std::vector<Expr<SubscriptInteger>> &&ss,
    std::vector<Expr<SubscriptInteger>> &&css)
  : base_(std::move(c)), subscript_(std::move(ss)),
    cosubscript_(std::move(css)) {
  CHECK(!base_.empty());
}

CoarrayRef &CoarrayRef::setStat(Variable &&v) {
  stat_ = CopyableIndirection<Variable>::Make(std::move(v));
  return *this;
}

CoarrayRef &CoarrayRef::setTeam(Variable &&v, bool isTeamNumber) {
  team_ = CopyableIndirection<Variable>::Make(std::move(v));
  teamIsTeamNumber_ = isTeamNumber;
  return *this;
}

Substring::Substring(DataRef &&d, std::optional<Expr<SubscriptInteger>> &&f,
    std::optional<Expr<SubscriptInteger>> &&l)
  : u_{std::move(d)} {
  if (f.has_value()) {
    first_ = IndirectSubscriptIntegerExpr::Make(std::move(*f));
  }
  if (l.has_value()) {
    last_ = IndirectSubscriptIntegerExpr::Make(std::move(*l));
  }
}

Substring::Substring(std::string &&s, std::optional<Expr<SubscriptInteger>> &&f,
    std::optional<Expr<SubscriptInteger>> &&l)
  : u_{std::move(s)} {
  if (f.has_value()) {
    first_ = IndirectSubscriptIntegerExpr::Make(std::move(*f));
  }
  if (l.has_value()) {
    last_ = IndirectSubscriptIntegerExpr::Make(std::move(*l));
  }
}

Expr<SubscriptInteger> Substring::first() const {
  if (first_.has_value()) {
    return **first_;
  }
  return AsExpr(Constant<SubscriptInteger>{1});
}

Expr<SubscriptInteger> Substring::last() const {
  if (last_.has_value()) {
    return **last_;
  }
  return std::visit(
      common::visitors{[](const std::string &s) {
                         // std::string::size_type isn't convertible to uint64_t
                         // on Darwin
                         return AsExpr(Constant<SubscriptInteger>{
                             static_cast<std::uint64_t>(s.size())});
                       },
          [](const DataRef &x) { return x.LEN(); }},
      u_);
}

std::optional<std::string> Substring::Fold(FoldingContext &context) {
  std::optional<Constant<SubscriptInteger>> lbConst{first().Fold(context)};
  if (lbConst.has_value()) {
    first_ = AsExpr(*lbConst);
  }
  std::optional<Constant<SubscriptInteger>> ubConst{last().Fold(context)};
  if (ubConst.has_value()) {
    last_ = AsExpr(*ubConst);
  }
  if (auto both{common::AllPresent(std::move(lbConst), std::move(ubConst))}) {
    std::int64_t lbi{std::get<0>(*both).value.ToInt64()};
    std::int64_t ubi{std::get<1>(*both).value.ToInt64()};
    if (ubi < lbi) {
      // These cases are well defined, and they produce zero-length results.
      u_ = ""s;
      first_ = AsExpr(Constant<SubscriptInteger>{1});
      last_ = AsExpr(Constant<SubscriptInteger>{0});
      return {""s};
    }
    if (lbi <= 0) {
      context.messages.Say(
          "lower bound on substring (%jd) is less than one"_en_US,
          static_cast<std::intmax_t>(lbi));
      lbi = 1;
      first_ = AsExpr(Constant<SubscriptInteger>{lbi});
    }
    if (ubi <= 0) {
      u_ = ""s;
      last_ = AsExpr(Constant<SubscriptInteger>{0});
      return {""s};
    }
    if (std::string * str{std::get_if<std::string>(&u_)}) {
      std::int64_t len = str->size();
      if (ubi > len) {
        context.messages.Say(
            "upper bound on substring (%jd) is greater than character length (%jd)"_en_US,
            static_cast<std::intmax_t>(ubi), static_cast<std::intmax_t>(len));
        ubi = len;
        last_ = AsExpr(Constant<SubscriptInteger>{ubi});
      }
      std::string result{str->substr(lbi - 1, ubi - lbi + 1)};
      u_ = result;
      return {result};
    }
  }
  return std::nullopt;
}

// Variable dumping

template<typename A> std::ostream &Emit(std::ostream &o, const A &x) {
  return x.Dump(o);
}

template<> std::ostream &Emit(std::ostream &o, const std::string &lit) {
  return o << parser::QuoteCharacterLiteral(lit);
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

template<typename... A>
std::ostream &Emit(std::ostream &o, const std::variant<A...> &u) {
  std::visit([&](const auto &x) { Emit(o, x); }, u);
  return o;
}

template<> std::ostream &Emit(std::ostream &o, const Symbol &symbol) {
  return o << symbol.name().ToString();
}

template<> std::ostream &Emit(std::ostream &o, const IntrinsicProcedure &p) {
  return o << EnumToString(p);
}

std::ostream &Component::Dump(std::ostream &o) const {
  base_->Dump(o);
  return Emit(o << '%', symbol_);
}

std::ostream &Triplet::Dump(std::ostream &o) const {
  Emit(o, lower_) << ':';
  Emit(o, upper_);
  if (stride_) {
    Emit(o << ':', stride_);
  }
  return o;
}

std::ostream &Subscript::Dump(std::ostream &o) const { return Emit(o, u); }

std::ostream &ArrayRef::Dump(std::ostream &o) const {
  Emit(o, u);
  char separator{'('};
  for (const Subscript &ss : subscript) {
    ss.Dump(o << separator);
    separator = ',';
  }
  return o << ')';
}

std::ostream &CoarrayRef::Dump(std::ostream &o) const {
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

std::ostream &DataRef::Dump(std::ostream &o) const { return Emit(o, u); }

std::ostream &Substring::Dump(std::ostream &o) const {
  Emit(o, u_) << '(';
  Emit(o, first_) << ':';
  return Emit(o, last_);
}

std::ostream &ComplexPart::Dump(std::ostream &o) const {
  return complex_.Dump(o) << '%' << EnumToString(part_);
}

std::ostream &ProcedureDesignator::Dump(std::ostream &o) const {
  return Emit(o, u);
}

template<typename ARG>
std::ostream &ProcedureRef<ARG>::Dump(std::ostream &o) const {
  Emit(o, proc_);
  char separator{'('};
  for (const auto &arg : argument_) {
    Emit(o << separator, arg);
    separator = ',';
  }
  if (separator == '(') {
    o << '(';
  }
  return o << ')';
}

std::ostream &Variable::Dump(std::ostream &o) const { return Emit(o, u); }

std::ostream &ActualFunctionArg::Dump(std::ostream &o) const {
  return Emit(o, u);
}
std::ostream &ActualSubroutineArg::Dump(std::ostream &o) const {
  return Emit(o, u);
}

std::ostream &Label::Dump(std::ostream &o) const {
  return o << '*' << std::dec << label;
}

// LEN()
static Expr<SubscriptInteger> SymbolLEN(const Symbol &sym) {
  return AsExpr(Constant<SubscriptInteger>{0});  // TODO
}
Expr<SubscriptInteger> Component::LEN() const { return SymbolLEN(symbol()); }
Expr<SubscriptInteger> ArrayRef::LEN() const {
  return std::visit(
      common::visitors{[](const Symbol *s) { return SymbolLEN(*s); },
          [](const Component &x) { return x.LEN(); }},
      u);
}
Expr<SubscriptInteger> CoarrayRef::LEN() const {
  return SymbolLEN(*base_.back());
}
Expr<SubscriptInteger> DataRef::LEN() const {
  return std::visit(
      common::visitors{[](const Symbol *s) { return SymbolLEN(*s); },
          [](const auto &x) { return x.LEN(); }},
      u);
}
Expr<SubscriptInteger> Substring::LEN() const {
  return AsExpr(
      Extremum<SubscriptInteger>{AsExpr(Constant<SubscriptInteger>{0}),
          last() - first() + AsExpr(Constant<SubscriptInteger>{1})});
}
template<typename A> Expr<SubscriptInteger> Designator<A>::LEN() const {
  return std::visit(
      common::visitors{[](const Symbol *s) { return SymbolLEN(*s); },
          [](const Component &c) { return c.LEN(); },
          [](const auto &x) { return x.LEN(); }},
      u);
}
Expr<SubscriptInteger> ProcedureDesignator::LEN() const {
  return std::visit(
      common::visitors{[](const Symbol *s) { return SymbolLEN(*s); },
          [](const Component &c) { return c.LEN(); },
          [](const auto &) {
            CRASH_NO_CASE;
            return AsExpr(Constant<SubscriptInteger>{0});
          }},
      u);
}

template class Designator<Type<TypeCategory::Character, 1>>;
template class Designator<Type<TypeCategory::Character, 2>>;
template class Designator<Type<TypeCategory::Character, 4>>;
}  // namespace Fortran::evaluate
