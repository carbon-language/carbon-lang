// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_KIND_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_KIND_H_

#include <cstdint>

#include "common/ostream.h"

namespace Carbon {

#define CARBON_ENUM_BASE_NAME SemanticsNodeKindBase
#define CARBON_ENUM_DEF_PATH "toolchain/semantics/semantics_node_kind.def"
#include "toolchain/common/enum_base.def"

class SemanticsNodeKind : public SemanticsNodeKindBase<SemanticsNodeKind> {
  using SemanticsNodeKindBase::SemanticsNodeKindBase;
};

// We expect the node kind to fit compactly into 8 bits.
static_assert(sizeof(SemanticsNodeKind) == 1, "Kind objects include padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_KIND_H_
