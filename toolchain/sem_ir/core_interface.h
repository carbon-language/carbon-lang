// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_CORE_INTERFACE_H_
#define CARBON_TOOLCHAIN_SEM_IR_CORE_INTERFACE_H_

#include <cstdint>

#include "common/enum_base.h"
#include "llvm/ADT/ArrayRef.h"

namespace Carbon::SemIR {

CARBON_DEFINE_RAW_ENUM_CLASS(CoreInterface, std::uint8_t) {
#define CARBON_SEM_IR_CORE_INTERFACE_KIND(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "toolchain/sem_ir/core_interface_kind.def"
};

// Significant interfaces in `Core` which correspond to language features and
// can have custom witnesses.
class CoreInterface : public CARBON_ENUM_BASE(CoreInterface) {
 public:
#define CARBON_SEM_IR_CORE_INTERFACE_KIND(Name) CARBON_ENUM_CONSTANT_DECL(Name)
#include "toolchain/sem_ir/core_interface_kind.def"

  static const llvm::ArrayRef<CoreInterface> CoreInterfaces;
  using EnumBase::AsInt;
  using EnumBase::EnumBase;

 private:
  static const CoreInterface CoreInterfacesStorage[];
  static const llvm::StringLiteral Spelling[];
};

#define CARBON_SEM_IR_CORE_INTERFACE_KIND(Name) \
  CARBON_ENUM_CONSTANT_DEFINITION(CoreInterface, Name)
#include "toolchain/sem_ir/core_interface_kind.def"

inline constexpr CoreInterface CoreInterface::CoreInterfacesStorage[] = {
#define CARBON_SEM_IR_CORE_INTERFACE_KIND(Name) CoreInterface::Name,
#include "toolchain/sem_ir/core_interface_kind.def"
};
inline constexpr llvm::ArrayRef<CoreInterface> CoreInterface::CoreInterfaces =
    CoreInterfacesStorage;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CORE_INTERFACE_H_
