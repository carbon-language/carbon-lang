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
    lower = SubscriptIntegerExpr{std::move(*l)};
  }
  if (u.has_value()) {
    upper = SubscriptIntegerExpr{std::move(*u)};
  }
  if (s.has_value()) {
    stride = SubscriptIntegerExpr{std::move(*s)};
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

std::ostream &Component::Dump(std::ostream &o) const {
  base->Dump(o);
  return Emit(o << '%', sym);
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
  for (const SubscriptIntegerExpr &ss : subscript) {
    Emit(o << separator, ss);
    separator = ',';
  }
  if (separator == ',') {
    o << ')';
  }
  separator = '[';
  for (const SubscriptIntegerExpr &css : cosubscript) {
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
}  // namespace Fortran::evaluate
