// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/inst.h"

namespace Carbon::SemIR {

auto Inst::Print(llvm::raw_ostream& out) const -> void {
  out << "{kind: " << kind_;

  auto print_args = [&](auto info) {
    using Info = decltype(info);
    if constexpr (Info::NumArgs > 0) {
      out << ", arg0: " << FromRaw<typename Info::template ArgType<0>>(arg0_);
    }
    if constexpr (Info::NumArgs > 1) {
      out << ", arg1: " << FromRaw<typename Info::template ArgType<1>>(arg1_);
    }
  };

  // clang warns on unhandled enum values; clang-tidy is incorrect here.
  // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
  switch (kind_) {
#define CARBON_SEM_IR_INST_KIND(Name)      \
  case Name::Kind:                         \
    print_args(TypedInstArgsInfo<Name>()); \
    break;
#include "toolchain/sem_ir/inst_kind.def"
  }
  if (type_id_.is_valid()) {
    out << ", type: " << type_id_;
  }
  out << "}";
}

}  // namespace Carbon::SemIR
