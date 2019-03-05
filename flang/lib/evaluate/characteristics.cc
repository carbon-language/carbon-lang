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

#include "characteristics.h"
#include <ostream>
#include <sstream>
#include <string>

using namespace std::literals::string_literals;

namespace Fortran::evaluate::characteristics {

bool DummyDataObject::operator==(const DummyDataObject &that) const {
  return attrs == that.attrs && intent == that.intent && type == that.type &&
      shape == that.shape && coshape == that.coshape;
}

std::ostream &DummyDataObject::Dump(std::ostream &o) const {
  attrs.Dump(o, EnumToString);
  if (intent != common::Intent::Default) {
    o << "INTENT(" << common::EnumToString(intent) << ')';
  }
  o << type.AsFortran();
  if (!shape.empty()) {
    char sep{'('};
    for (const auto &expr : shape) {
      o << sep;
      sep = ',';
      if (expr.has_value()) {
        expr->AsFortran(o);
      } else {
        o << ':';
      }
    }
    o << ')';
  }
  if (!coshape.empty()) {
    char sep{'['};
    for (const auto &expr : coshape) {
      expr.AsFortran(o << sep);
      sep = ',';
    }
  }
  return o;
}

bool DummyProcedure::operator==(const DummyProcedure &that) const {
  return attrs == that.attrs && explicitProcedure == that.explicitProcedure;
}

std::ostream &DummyProcedure::Dump(std::ostream &o) const {
  attrs.Dump(o, EnumToString);
  if (explicitProcedure.has_value()) {
    explicitProcedure.value().Dump(o);
  }
  return o;
}

std::ostream &AlternateReturn::Dump(std::ostream &o) const { return o << '*'; }

bool FunctionResult::operator==(const FunctionResult &that) const {
  return attrs == that.attrs && type == that.type && rank == that.rank;
}

std::ostream &FunctionResult::Dump(std::ostream &o) const {
  attrs.Dump(o, EnumToString);
  return o << type.AsFortran() << " rank " << rank;
}

bool Procedure::operator==(const Procedure &that) const {
  return attrs == that.attrs && dummyArguments == that.dummyArguments &&
      functionResult == that.functionResult;
}

std::ostream &Procedure::Dump(std::ostream &o) const {
  attrs.Dump(o, EnumToString);
  if (functionResult.has_value()) {
    functionResult->Dump(o << "TYPE(") << ") FUNCTION";
  } else {
    o << "SUBROUTINE";
  }
  char sep{'('};
  for (const auto &dummy : dummyArguments) {
    o << sep;
    sep = ',';
    std::visit([&](const auto &x) { x.Dump(o); }, dummy);
  }
  return o << (sep == '(' ? "()" : ")");
}
}

// Define OwningPointer special member functions
DEFINE_OWNING_SPECIAL_FUNCTIONS(
    OwningPointer, evaluate::characteristics::Procedure)
