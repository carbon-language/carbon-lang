// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/bindings.h"

namespace Carbon {

auto Bindings::None() -> Nonnull<const Bindings*> {
  static Nonnull<const Bindings*> bindings = new Bindings({}, {});
  return bindings;
}

}  // namespace Carbon
