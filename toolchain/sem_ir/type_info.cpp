// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/type_info.h"

#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

auto ValueRepr::Print(llvm::raw_ostream& out) const -> void {
  out << "{kind: ";
  switch (kind) {
    case Unknown:
      out << "unknown";
      break;
    case None:
      out << "none";
      break;
    case Copy:
      out << "copy";
      break;
    case Pointer:
      out << "pointer";
      break;
    case Custom:
      out << "custom";
      break;
  }
  out << ", type: " << type_id << "}";
}

auto CompleteTypeInfo::Print(llvm::raw_ostream& out) const -> void {
  out << "{value_rep: " << value_repr << "}";
}

auto ValueRepr::ForType(const File& file, TypeId type_id) -> ValueRepr {
  return file.types().GetValueRepr(type_id);
}

auto InitRepr::ForType(const File& file, TypeId type_id) -> InitRepr {
  auto value_rep = ValueRepr::ForType(file, type_id);
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
      return {.kind = InitRepr::Incomplete};
  }
}

}  // namespace Carbon::SemIR
