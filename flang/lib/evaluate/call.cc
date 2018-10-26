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

#include "call.h"
#include "expression.h"

namespace Fortran::evaluate {

std::optional<DynamicType> ActualArgument::GetType() const {
  return value->GetType();
}

int ActualArgument::Rank() const { return value->Rank(); }

std::ostream &ActualArgument::Dump(std::ostream &o) const {
  if (keyword.has_value()) {
    o << keyword->ToString() << '=';
  }
  if (isAlternateReturn) {
    o << '*';
  }
  return value->Dump(o);
}

std::optional<int> ActualArgument::VectorSize() const {
  if (Rank() != 1) {
    return std::nullopt;
  }
  // TODO: get shape vector of value, return its length
  return std::nullopt;
}

std::ostream &ProcedureRef::Dump(std::ostream &o) const {
  proc_.Dump(o);
  char separator{'('};
  for (const auto &arg : arguments_) {
    arg->Dump(o << separator);
    separator = ',';
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
