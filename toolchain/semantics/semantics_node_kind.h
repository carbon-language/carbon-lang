// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_KIND_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_KIND_H_

#include "toolchain/common/enum_base.h"

namespace Carbon {

CARBON_ENUM_BASE_1_OF_7(SemanticsNodeKindBase)
#define CARBON_SEMANTICS_NODE_KIND(Name) CARBON_ENUM_BASE_2_OF_7_ITER(Name)
#include "toolchain/semantics/semantics_node_kind.def"
CARBON_ENUM_BASE_3_OF_7(SemanticsNodeKindBase)
#define CARBON_SEMANTICS_NODE_KIND(Name) CARBON_ENUM_BASE_4_OF_7_ITER(Name)
#include "toolchain/semantics/semantics_node_kind.def"
CARBON_ENUM_BASE_5_OF_7(SemanticsNodeKindBase)
#define CARBON_SEMANTICS_NODE_KIND(Name) CARBON_ENUM_BASE_6_OF_7_ITER(Name)
#include "toolchain/semantics/semantics_node_kind.def"
CARBON_ENUM_BASE_7_OF_7(SemanticsNodeKindBase)

class SemanticsNodeKind : public SemanticsNodeKindBase<SemanticsNodeKind> {
  using SemanticsNodeKindBase::SemanticsNodeKindBase;
};

// We expect the node kind to fit compactly into 8 bits.
static_assert(sizeof(SemanticsNodeKind) == 1, "Kind objects include padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_KIND_H_
