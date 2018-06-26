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

#ifndef FORTRAN_EVALUATE_VARIABLE_H_
#define FORTRAN_EVALUATE_VARIABLE_H_

#include "expression.h"
#include <memory>
#include <optional>
#include <variant>

namespace Fortran::evaluate {

struct Designator;

struct DataObject {
  semantics::Symbol &object;
};

struct Component {
  semantics::Symbol &component;
  std::variant<std::unique_ptr<DataObject>, std::unique_ptr<Designator>>> u;
};

using SubscriptExpr = DefaultIntExpr;

struct Triplet {
  std::optional<DefaultIntExpr> lower, upper, stride;
};

struct Subscript {
  std::variant<DefaultIntExpr, Triplet>;
};

struct Subscripted {
  std::vector<std::unique_ptr<Subscript>> subscript;
  std::variant<std::unique_ptr<DataObject>, std::unique_ptr<Component>> u;
};

struct CoarrayRef {
  std::vector<std::unique_ptr<SubscriptExpr>>;
  // TODO R926 image selector specs
  std::variant<std::unique_ptr<Subscripted>, std::unique_ptr<Component>,
               std::unique_ptr<DataObject>> u;
};

struct Designator {
  Designator() = delete;
  std::variant<std::unique_ptr<Object>, std::unique_ptr<Component>,
               std::unique_ptr<Subscripted>, std::unique_ptr<CoarrayRef>> u;
};

struct Substring {
};

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_VARIABLE_H_
