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

#include "call.h"
#include "expression.h"
#include "../semantics/symbol.h"

namespace Fortran::evaluate {

std::optional<DynamicType> ActualArgument::GetType() const {
  return value->GetType();
}

int ActualArgument::Rank() const { return value->Rank(); }

bool ActualArgument::operator==(const ActualArgument &that) const {
  return keyword == that.keyword &&
      isAlternateReturn == that.isAlternateReturn && value == that.value;
}

std::ostream &ActualArgument::AsFortran(std::ostream &o) const {
  if (keyword.has_value()) {
    o << keyword->ToString() << '=';
  }
  if (isAlternateReturn) {
    o << '*';
  }
  return value->AsFortran(o);
}

std::optional<int> ActualArgument::VectorSize() const {
  if (Rank() != 1) {
    return std::nullopt;
  }
  // TODO: get shape vector of value, return its length
  return std::nullopt;
}

bool SpecificIntrinsic::operator==(const SpecificIntrinsic &that) const {
  return name == that.name && type == that.type && rank == that.rank &&
      attrs == that.attrs;
}

std::ostream &SpecificIntrinsic::AsFortran(std::ostream &o) const {
  return o << name;
}

std::optional<DynamicType> ProcedureDesignator::GetType() const {
  if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&u)}) {
    return intrinsic->type;
  } else {
    return GetSymbolType(GetSymbol());
  }
}

int ProcedureDesignator::Rank() const {
  if (const Symbol * symbol{GetSymbol()}) {
    return symbol->Rank();
  }
  if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&u)}) {
    return intrinsic->rank;
  }
  CHECK(!"ProcedureDesignator::Rank(): no case");
  return 0;
}

bool ProcedureDesignator::IsElemental() const {
  if (const Symbol * symbol{GetSymbol()}) {
    return symbol->attrs().test(semantics::Attr::ELEMENTAL);
  }
  if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&u)}) {
    return intrinsic->attrs.test(semantics::Attr::ELEMENTAL);
  }
  CHECK(!"ProcedureDesignator::IsElemental(): no case");
  return 0;
}

const Symbol *ProcedureDesignator::GetSymbol() const {
  return std::visit(
      common::visitors{
          [](const Symbol *sym) { return sym; },
          [](const Component &c) { return &c.GetLastSymbol(); },
          [](const auto &) -> const Symbol * { return nullptr; },
      },
      u);
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

Expr<SubscriptInteger> ProcedureRef::LEN() const {
  // TODO: the results of the intrinsic functions REPEAT and TRIM have
  // unpredictable lengths; maybe the concept of LEN() has to become dynamic
  return proc_.LEN();
}

FOR_EACH_SPECIFIC_TYPE(template struct FunctionRef)
}
