// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_THUNK_H_
#define CARBON_TOOLCHAIN_SEM_IR_THUNK_H_

#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Metadata tracking information about a thunk's signature, callee, and specific
// generic parameters.
struct ThunkInfo {
  InstId callee_id;
  FunctionId signature_id = FunctionId::None;
  SpecificId specific_id = SpecificId::None;
};

using ThunkStore = ValueStore<ThunkId, ThunkInfo, Tag<CheckIRId>>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_THUNK_H_
