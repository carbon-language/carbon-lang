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

#include "tools.h"

namespace Fortran::parser {

const Name &GetLastName(const Name &x) { return x; }

const Name &GetLastName(const StructureComponent &x) {
  return GetLastName(x.component);
}

const Name &GetLastName(const DataRef &x) {
  return std::visit(
      common::visitors{
          [](const Name &name) -> const Name & { return name; },
          [](const common::Indirection<StructureComponent> &sc)
              -> const Name & { return GetLastName(sc.value()); },
          [](const common::Indirection<ArrayElement> &sc) -> const Name & {
            return GetLastName(sc.value().base);
          },
          [](const common::Indirection<CoindexedNamedObject> &ci)
              -> const Name & { return GetLastName(ci.value().base); },
      },
      x.u);
}

const Name &GetLastName(const Substring &x) {
  return GetLastName(std::get<DataRef>(x.t));
}

const Name &GetLastName(const Designator &x) {
  return std::visit(
      [](const auto &y) -> const Name & { return GetLastName(y); }, x.u);
}

const Name &GetLastName(const ProcComponentRef &x) {
  return GetLastName(x.v.thing);
}

const Name &GetLastName(const ProcedureDesignator &x) {
  return std::visit(
      [](const auto &y) -> const Name & { return GetLastName(y); }, x.u);
}

const Name &GetLastName(const Call &x) {
  return GetLastName(std::get<ProcedureDesignator>(x.t));
}

const Name &GetLastName(const FunctionReference &x) { return GetLastName(x.v); }

const Name &GetLastName(const Variable &x) {
  return std::visit(
      [](const auto &indirection) -> const Name & {
        return GetLastName(indirection.value());
      },
      x.u);
}

const Name &GetLastName(const AllocateObject &x) {
  return std::visit(
      [](const auto &y) -> const Name & { return GetLastName(y); }, x.u);
}

const CoindexedNamedObject *GetCoindexedNamedObject(const DataRef &base) {
  return std::visit(
      common::visitors{
          [](const Name &) -> const CoindexedNamedObject * { return nullptr; },
          [](const common::Indirection<CoindexedNamedObject> &x)
              -> const CoindexedNamedObject * { return &x.value(); },
          [](const auto &x) -> const CoindexedNamedObject * {
            return GetCoindexedNamedObject(x.value().base);
          },
      },
      base.u);
}
const CoindexedNamedObject *GetCoindexedNamedObject(
    const AllocateObject &allocateObject) {
  return std::visit(
      common::visitors{
          [](const StructureComponent &x) -> const CoindexedNamedObject * {
            return GetCoindexedNamedObject(x.base);
          },
          [](const auto &) -> const CoindexedNamedObject * { return nullptr; },
      },
      allocateObject.u);
}
}
