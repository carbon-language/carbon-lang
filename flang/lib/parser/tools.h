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

#ifndef FORTRAN_PARSER_TOOLS_H_
#define FORTRAN_PARSER_TOOLS_H_
#include "parse-tree.h"
namespace Fortran::parser {

// GetLastName() isolates and returns a reference to the rightmost Name
// in a variable (i.e., the Name whose symbol's type determines the type
// of the variable or expression).

const Name &GetLastName(const Name &x) { return x; }

const Name &GetLastName(const StructureComponent &x) {
  return GetLastName(x.component);
}

const Name &GetLastName(const DataRef &x) {
  return std::visit(
      common::visitors{
          [](const Name &name) { return GetLastName(name); },
          [](const common::Indirection<StructureComponent> &sc) {
            return GetLastName(sc.value());
          },
          [](const common::Indirection<ArrayElement> &sc) {
            return GetLastName(sc.value().base);
          },
          [](const common::Indirection<CoindexedNamedObject> &ci) {
            return GetLastName(ci.value().base);
          },
      },
      x.u);
}

const Name &GetLastName(const Substring &x) {
  return GetType(std::get<DataRef>(x.t));
}

const Name &GetLastName(const Designator &x) {
  return std::visit([](const auto &y) { return GetType(y); }, x.u);
}

const Name &GetLastName(const ProcComponentRef &x) {
  return GetType(x.v.thing);
}

const Name &GetLastName(const ProcedureDesignator &x) {
  return std::visit([](const auto &y) { return GetType(y); }, x.u);
}

const Name &GetLastName(const Call &x) {
  return GetType(std::get<ProcedureDesignator>(x.t));
}

const Name &GetLastName(const FunctionReference &x) { return GetType(x.v); }

const Name &GetLastName(const Variable &x) {
  return std::visit(
      [](const auto &indirection) { return GetType(indirection.value()); },
      x.u);
}

}
#endif  // FORTRAN_PARSER_TOOLS_H_
