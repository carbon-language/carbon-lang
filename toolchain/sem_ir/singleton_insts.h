// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_SINGLETON_INSTS_H_
#define CARBON_TOOLCHAIN_SEM_IR_SINGLETON_INSTS_H_

#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// The canonical list of singleton kinds. The order of `TypeType` is
// significant because other singletons use it as a type.
static constexpr std::array SingletonInstKinds = {
    InstKind::TypeType,
    InstKind::AutoType,
    InstKind::BoolType,
    InstKind::BoundMethodType,
    InstKind::ErrorInst,
    InstKind::IntLiteralType,
    InstKind::LegacyFloatType,
    InstKind::NamespaceType,
    InstKind::SpecificFunctionType,
    InstKind::StringType,
    InstKind::VtableType,
    InstKind::WitnessType,
};

namespace Internal {

// Returns the index for a singleton instruction, or -1 if it's not a singleton.
static constexpr auto GetSingletonInstIndex(InstKind kind) -> int32_t {
  for (int32_t i = 0; i < static_cast<int32_t>(SingletonInstKinds.size());
       ++i) {
    if (SingletonInstKinds[i] == kind) {
      return i;
    }
  }
  return -1;
}

}  // namespace Internal

// Returns true if the InstKind is a singleton.
static constexpr auto IsSingletonInstKind(InstKind kind) -> bool {
  return Internal::GetSingletonInstIndex(kind) >= 0;
}

// Provides the InstId for singleton instructions. For example,
// `InstId error_inst_id = SingletonInstId<ErrorInst>;`.
template <typename InstT>
  requires(IsSingletonInstKind(InstT::Kind))
static constexpr InstId SingletonInstId =
    InstId(Internal::GetSingletonInstIndex(InstT::Kind));

// TODO: This verifies values match while working on removing
// `CARBON_SEM_IR_BUILTIN_INST_KIND`.
#define CARBON_SEM_IR_BUILTIN_INST_KIND(Name) \
  static_assert(InstId::Builtin##Name == SingletonInstId<Name>);
#include "toolchain/sem_ir/inst_kind.def"

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_SINGLETON_INSTS_H_
