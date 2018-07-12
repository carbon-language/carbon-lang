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

Triplet::Triplet(std::optional<SubscriptIntegerExpr> &&l,
    std::optional<SubscriptIntegerExpr> &&u,
    std::optional<SubscriptIntegerExpr> &&s) {
  if (l.has_value()) {
    lower = IndirectSubscriptIntegerExpr::Make(std::move(*l));
  }
  if (u.has_value()) {
    upper = IndirectSubscriptIntegerExpr::Make(std::move(*u));
  }
  if (s.has_value()) {
    stride = IndirectSubscriptIntegerExpr::Make(std::move(*s));
  }
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
  Emit(o, lower) << ':';
  Emit(o, upper);
  if (stride) {
    Emit(o << ':', stride);
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
  for (const Symbol *sym : base) {
    Emit(o, *sym);
  }
  char separator{'('};
  for (const auto &ss : subscript) {
    Emit(o << separator, ss);
    separator = ',';
  }
  if (separator == ',') {
    o << ')';
  }
  separator = '[';
  for (const auto &css : cosubscript) {
    Emit(o << separator, css);
    separator = ',';
  }
  if (stat.has_value()) {
    Emit(o << separator, stat, "STAT=");
    separator = ',';
  }
  if (team.has_value()) {
    Emit(o << separator, team, teamIsTeamNumber ? "TEAM_NUMBER=" : "TEAM=");
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

CoarrayRef::CoarrayRef(std::vector<const Symbol *> &&c,
    std::vector<SubscriptIntegerExpr> &&ss,
    std::vector<SubscriptIntegerExpr> &&css)
  : base(std::move(c)), subscript(std::move(ss)), cosubscript(std::move(css)) {
  CHECK(!base.empty());
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

// LEN()
static SubscriptIntegerExpr SymbolLEN(const Symbol &sym) {
  return SubscriptIntegerExpr{0};  // TODO
}
SubscriptIntegerExpr Component::LEN() const { return SymbolLEN(symbol()); }
SubscriptIntegerExpr ArrayRef::LEN() const {
  return std::visit(
      common::visitors{[](const Symbol *s) { return SymbolLEN(*s); },
          [](const Component &x) { return x.LEN(); }},
      u);
}
SubscriptIntegerExpr CoarrayRef::LEN() const { return SymbolLEN(*base.back()); }
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
