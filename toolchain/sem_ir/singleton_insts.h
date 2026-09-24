// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_SINGLETON_INSTS_H_
#define CARBON_TOOLCHAIN_SEM_IR_SINGLETON_INSTS_H_

#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_kind.h"

namespace Carbon::SemIR {

// The canonical list of singleton kinds. The index of each in the array acts as
// a means to determine the InstId of the singleton inst for the kind.
static constexpr std::array SingletonInstKinds = {
    InstKind::AutoType,
    InstKind::BoolType,
    InstKind::BoundMethodType,
    InstKind::CharLiteralType,
    InstKind::ErrorInst,
    InstKind::FloatLiteralType,
    InstKind::FormType,
    InstKind::InstType,
    InstKind::IntLiteralType,
    InstKind::NamespaceType,
    InstKind::RequireSpecificDefinitionType,
    InstKind::SpecificFunctionType,
    InstKind::UnspecifiedValueType,
    InstKind::VtableType,
    InstKind::WitnessType,
};

// We have some builtin insts with fixed InstIds that are determined relative to
// the singletons InstIds.
// - TypeType::TypeInstId
// - Namespace::PackageInstId
constexpr auto NumBuiltinInstsBeforeSingletons = 1;
constexpr auto NumBuiltinInstsAfterSingletons = 1;

// The total number of InstIds that are fixed values. These are always the
// first InstIds in the file, and since they are fixed at compile-time of the
// toolchain, they are not tagged IDs.
constexpr auto NumBuiltinInsts = NumBuiltinInstsBeforeSingletons +
                                 static_cast<int>(SingletonInstKinds.size()) +
                                 NumBuiltinInstsAfterSingletons;

// Returns true if the InstKind is a singleton.
constexpr auto IsSingletonInstKind(InstKind kind) -> bool;

// Provides the TypeInstId for singleton instructions. These are exposed as
// `InstT::TypeInstId` in `typed_insts.h`.
template <InstKind::RawEnumType Kind>
  requires(IsSingletonInstKind(InstKind::Make(Kind)))
constexpr auto MakeSingletonTypeInstId() -> TypeInstId;

// Provides the TypeInstId for the `TypeType` inst. This is exposed as
// `TypeType::TypeInstId` in `typed_insts.h`. Its index is the very first index,
// before the singletons, so that they can refer to it.
constexpr auto MakeBuiltinTypeTypeInstId() -> TypeInstId {
  return TypeInstId(0);
}

// Provides the InstId for the `PackageInstId` inst. This is exposed as
// `Namespace::PackageInstId` in `typed_insts.h`. Its index is the first
// instruction after the singletons.
constexpr auto MakeBuiltinNamespacePackageInstId() -> InstId {
  return InstId(NumBuiltinInstsBeforeSingletons + SingletonInstKinds.size());
}

// Returns true if the InstId corresponds to a singleton inst.
constexpr auto IsSingletonInstId(InstId id) -> bool {
  auto index = id.index - NumBuiltinInstsBeforeSingletons;
  return index >= 0 && index < static_cast<int32_t>(SingletonInstKinds.size());
}

// Returns the InstKind for a singleton InstId.
constexpr auto GetSingletonInstKind(InstId id) -> InstKind {
  CARBON_CHECK(IsSingletonInstId(id));
  auto index = id.index - NumBuiltinInstsBeforeSingletons;
  return SingletonInstKinds[index];
}

// Only implementation details are below.

namespace Internal {

// Returns the InstId index for a singleton instruction, or -1 if it's not a
// singleton.
constexpr auto GetSingletonInstIndex(InstKind kind) -> int32_t {
  for (int32_t i = 0; i < static_cast<int32_t>(SingletonInstKinds.size());
       ++i) {
    if (SingletonInstKinds[i] == kind) {
      return i + NumBuiltinInstsBeforeSingletons;
    }
  }
  return -1;
}

}  // namespace Internal

constexpr auto IsSingletonInstKind(InstKind kind) -> bool {
  return Internal::GetSingletonInstIndex(kind) >= 0;
}

template <InstKind::RawEnumType Kind>
  requires(IsSingletonInstKind(InstKind::Make(Kind)))
constexpr auto MakeSingletonTypeInstId() -> TypeInstId {
  auto index = Internal::GetSingletonInstIndex(InstKind::Make(Kind));
  return TypeInstId(index);
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_SINGLETON_INSTS_H_
