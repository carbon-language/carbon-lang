// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_THUNK_H_
#define CARBON_TOOLCHAIN_SEM_IR_THUNK_H_

#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Metadata about a thunk.
struct ThunkInfo {
  // The callee wrapped by the thunk.
  InstId callee_id;

  // The signature of the thunk.
  FunctionId signature_id = FunctionId::None;

  // If `signature_id` is generic, the specific that this thunk is for.
  SpecificId specific_id = SpecificId::None;

  // If this is a thunk for a virtual function override, the type of the
  // `self` parameter of the override.
  TypeId override_self_type_id = TypeId::None;
};

using ThunkStore = ValueStore<ThunkId, ThunkInfo, Tag<CheckIRId>>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_THUNK_H_
