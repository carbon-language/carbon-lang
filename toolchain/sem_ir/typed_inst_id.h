// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_TYPED_INST_ID_H_
#define CARBON_TOOLCHAIN_SEM_IR_TYPED_INST_ID_H_

#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_kind.h"

namespace Carbon::SemIR {

// Use a template on the InstKind to help provide unique InstId types
// per-instruction.
template <SemIR::InstKind::RawEnumType Kind>
class TypedInstId : public InstId {
 public:
  static const TypedInstId Invalid;
  using InstId::InstId;
};

template <SemIR::InstKind::RawEnumType Kind>
constexpr TypedInstId<Kind> TypedInstId<Kind>::Invalid =
    TypedInstId(InvalidIndex);

// Provide names for the per-instruction InstId types.
#define CARBON_SEM_IR_INST_KIND(Name) \
  using Name##InstId = TypedInstId<SemIR::InstKind::Name>;
#include "toolchain/sem_ir/inst_kind.def"

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_TYPED_INST_ID_H_
