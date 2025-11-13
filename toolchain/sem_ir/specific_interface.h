// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_SPECIFIC_INTERFACE_H_
#define CARBON_TOOLCHAIN_SEM_IR_SPECIFIC_INTERFACE_H_

#include "toolchain/base/canonical_value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// A pair of an interface and a specific for that interface.
struct SpecificInterface {
  using DiagnosticType = Diagnostics::TypeInfo<std::string>;

  static const SpecificInterface None;

  InterfaceId interface_id;
  SpecificId specific_id;

  friend auto operator==(const SpecificInterface& lhs,
                         const SpecificInterface& rhs) -> bool = default;
};

inline constexpr SpecificInterface SpecificInterface::None = {
    .interface_id = InterfaceId::None, .specific_id = SpecificId::None};

using SpecificInterfaceStore =
    CanonicalValueStore<SpecificInterfaceId, SpecificInterface>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_SPECIFIC_INTERFACE_H_
