// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/field_path.h"

#include "common/ostream.h"

namespace Carbon {

auto FieldPath::DebugString() const -> std::string {
  return Carbon::DebugString(*this);
}

}  // namespace Carbon
