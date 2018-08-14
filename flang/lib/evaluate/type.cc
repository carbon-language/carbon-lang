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

#include "type.h"
#include "expression.h"
#include "../common/idioms.h"
#include <cinttypes>
#include <optional>
#include <variant>

namespace Fortran::evaluate {

std::optional<std::int64_t> GenericScalar::ToInt64() const {
  if (const auto *j{std::get_if<SomeKindScalar<TypeCategory::Integer>>(&u)}) {
    return std::visit(
        [](const auto &k) { return std::optional<std::int64_t>{k.ToInt64()}; },
        j->u);
  }
  return std::nullopt;
}

std::optional<std::string> GenericScalar::ToString() const {
  if (const auto *c{std::get_if<SomeKindScalar<TypeCategory::Character>>(&u)}) {
    if (const std::string * s{std::get_if<std::string>(&c->u)}) {
      return std::optional<std::string>{*s};
    }
  }
  return std::nullopt;
}
}  // namespace Fortran::evaluate
