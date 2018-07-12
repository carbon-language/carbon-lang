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
#include "../common/idioms.h"
#include "../parser/char-block.h"
#include "../parser/characters.h"
#include "../semantics/symbol.h"
#include <ostream>

namespace Fortran::evaluate {

// Constructors, accessors, mutators

Triplet::Triplet(std::optional<SubscriptIntegerExpr> &&l,
    std::optional<SubscriptIntegerExpr> &&u,
    std::optional<SubscriptIntegerExpr> &&s) {
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

std::optional<SubscriptIntegerExpr> Triplet::lower() const {
  if (lower_) {
    return {**lower_};
  }
  return {};
}

std::optional<SubscriptIntegerExpr> Triplet::upper() const {
  if (upper_) {
    return {**upper_};
  }
  return {};
}

std::optional<SubscriptIntegerExpr> Triplet::stride() const {
  if (stride_) {
    return {**stride_};
  }
  return {};
}

CoarrayRef::CoarrayRef(std::vector<const Symbol *> &&c,
    std::vector<SubscriptIntegerExpr> &&ss,
    std::vector<SubscriptIntegerExpr> &&css)
  : base_(std::move(c)), subscript_(std::move(ss)), cosubscript_(std::move(css)) {
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

Substring::Substring(DataRef &&d, std::optional<SubscriptIntegerExpr> &&f,
    std::optional<SubscriptIntegerExpr> &&l)
  : u{std::move(d)} {
  if (f.has_value()) {
    first = IndirectSubscriptIntegerExpr::Make(std::move(*f));
  }
  if (l.has_value()) {
    last = IndirectSubscriptIntegerExpr::Make(std::move(*l));
  }
}

Substring::Substring(std::string &&s, std::optional<SubscriptIntegerExpr> &&f,
    std::optional<SubscriptIntegerExpr> &&l)
  : u{std::move(s)} {
  if (f.has_value()) {
    first = IndirectSubscriptIntegerExpr::Make(std::move(*f));
  }
  if (l.has_value()) {
    last = IndirectSubscriptIntegerExpr::Make(std::move(*l));
  }
}

SubscriptIntegerExpr Substring::First() const {
  if (first.has_value()) {
    return **first;
  }
  return {1};
}

SubscriptIntegerExpr Substring::Last() const {
  if (last.has_value()) {
    return **last;
  }
  return std::visit(common::visitors{[](const std::string &s) {
                                       return SubscriptIntegerExpr{s.size()};
                                     },
                        [](const DataRef &x) { return x.LEN(); }},
      u);
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

std::ostream &Subscript::Dump(std::ostream &o) const { return Emit(o, u_); }

std::ostream &ArrayRef::Dump(std::ostream &o) const {
  Emit(o, u_);
  char separator{'('};
  for (const Subscript &ss : subscript_) {
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
  Emit(o, u) << '(';
  Emit(o, first) << ':';
  return Emit(o, last);
}

std::ostream &ComplexPart::Dump(std::ostream &o) const {
  return complex.Dump(o) << '%' << EnumToString(part);
}

std::ostream &Designator::Dump(std::ostream &o) const { return Emit(o, u); }

std::ostream &ProcedureDesignator::Dump(std::ostream &o) const {
  return Emit(o, u);
}

template<typename ARG>
std::ostream &ProcedureRef<ARG>::Dump(std::ostream &o) const {
  Emit(o, proc);
  char separator{'('};
  for (const auto &arg : argument) {
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
static SubscriptIntegerExpr SymbolLEN(const Symbol &sym) {
  return SubscriptIntegerExpr{0};  // TODO
}
SubscriptIntegerExpr Component::LEN() const { return SymbolLEN(symbol()); }
SubscriptIntegerExpr ArrayRef::LEN() const {
  return std::visit(
      common::visitors{[](const Symbol *s) { return SymbolLEN(*s); },
          [](const Component &x) { return x.LEN(); }},
      u_);
}
SubscriptIntegerExpr CoarrayRef::LEN() const { return SymbolLEN(*base_.back()); }
SubscriptIntegerExpr DataRef::LEN() const {
  return std::visit(
      common::visitors{[](const Symbol *s) { return SymbolLEN(*s); },
          [](const auto &x) { return x.LEN(); }},
      u);
}
SubscriptIntegerExpr Substring::LEN() const {
  return SubscriptIntegerExpr::Max{
      SubscriptIntegerExpr{0}, Last() - First() + SubscriptIntegerExpr{1}};
}
SubscriptIntegerExpr ProcedureDesignator::LEN() const {
  return std::visit(
      common::visitors{[](const Symbol *s) { return SymbolLEN(*s); },
          [](const Component &c) { return c.LEN(); },
          [](const auto &) {
            CRASH_NO_CASE;
            return SubscriptIntegerExpr{0};
          }},
      u);
}

}  // namespace Fortran::evaluate
