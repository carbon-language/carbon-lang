// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/type_info.h"

#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

auto GetValueRepr(const File& file, TypeId type_id) -> ValueRepr {
  return file.types().GetValueRepr(type_id);
}

auto GetInitRepr(const File& file, TypeId type_id) -> InitRepr {
  auto value_rep = GetValueRepr(file, type_id);
  switch (value_rep.kind) {
    case ValueRepr::None:
      return {.kind = InitRepr::None};

    case ValueRepr::Copy:
      // TODO: Use in-place initialization for types that have non-trivial
      // destructive move.
      return {.kind = InitRepr::ByCopy};

    case ValueRepr::Pointer:
    case ValueRepr::Custom:
      return {.kind = InitRepr::InPlace};

    case ValueRepr::Unknown:
      CARBON_FATAL()
          << "Attempting to perform initialization of incomplete type "
          << file.types().GetAsInst(type_id);
  }
}

}  // namespace Carbon::SemIR
