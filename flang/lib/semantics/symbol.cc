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

#include "symbol.h"
#include "scope.h"
#include "../parser/idioms.h"
#include <memory>

namespace Fortran::semantics {

void EntityDetails::set_type(const DeclTypeSpec &type) {
  CHECK(!type_);
  type_ = type;
}

void EntityDetails::set_shape(const ArraySpec &shape) {
  CHECK(shape_.empty());
  for (const auto &shapeSpec : shape) {
    shape_.push_back(shapeSpec);
  }
}

// The name of the kind of details for this symbol.
// This is primarily for debugging.
const std::string Symbol::GetDetailsName() const {
  return std::visit(
      parser::visitors{
          [&](const UnknownDetails &x) { return "Unknown"; },
          [&](const MainProgramDetails &x) { return "MainProgram"; },
          [&](const ModuleDetails &x) { return "Module"; },
          [&](const SubprogramDetails &x) { return "Subprogram"; },
          [&](const EntityDetails &x) { return "Entity"; },
      },
      details_);
}

std::ostream &operator<<(std::ostream &os, const EntityDetails &x) {
  if (x.type()) {
    os << " type: " << *x.type();
  }
  if (!x.shape().empty()) {
    os << " shape:";
    for (const auto &s : x.shape()) {
      os << ' ' << s;
    }
  }
  return os;
}

static std::ostream &DumpType(std::ostream &os, const Symbol &symbol) {
  if (const auto *details = symbol.detailsIf<EntityDetails>()) {
    if (details->type()) {
      os << *details->type() << ' ';
    }
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const Symbol &sym) {
  os << sym.name().ToString();
  if (!sym.attrs().empty()) {
    os << ", " << sym.attrs();
  }
  os << ": " << sym.GetDetailsName();
  std::visit(
      parser::visitors{
          [&](const UnknownDetails &x) {},
          [&](const MainProgramDetails &x) {},
          [&](const ModuleDetails &x) {},
          [&](const SubprogramDetails &x) {
            os << " (";
            int n = 0;
            for (const auto &dummy : x.dummyArgs()) {
              if (n++ > 0) os << ", ";
              DumpType(os, *dummy);
              os << dummy->name().ToString();
            }
            os << ')';
            if (x.isFunction()) {
              os << " result(";
              DumpType(os, x.result());
              os << x.result().name().ToString() << ')';
            }
          },
          [&](const EntityDetails &x) { os << x; },
      },
      sym.details_);
  return os;
}

}  // namespace Fortran::semantics
