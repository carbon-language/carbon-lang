// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/function_context.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Lower {

// For instructions that are always of type `type`, produce the trivial runtime
// representation of type `type`.
#define CARBON_SEM_IR_INST_KIND_TYPE_NEVER(...)
#define CARBON_SEM_IR_INST_KIND_TYPE_MAYBE(...)
#define CARBON_SEM_IR_INST_KIND_CONSTANT_ALWAYS(...)
#define CARBON_SEM_IR_INST_KIND(Name)                                \
  auto Handle##Name(FunctionContext& context, SemIR::InstId inst_id, \
                    SemIR::Name /*inst*/) -> void {                  \
    context.SetLocal(inst_id, context.GetTypeAsValue());             \
  }
#include "toolchain/sem_ir/inst_kind.def"

auto HandleFacetTypeAccess(FunctionContext& context, SemIR::InstId inst_id,
                           SemIR::FacetTypeAccess /*inst*/) -> void {
  context.SetLocal(inst_id, context.GetTypeAsValue());
}

}  // namespace Carbon::Lower
