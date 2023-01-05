// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_KIND_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_KIND_H_

#include "toolchain/common/enum_base.h"

namespace Carbon {

CARBON_DEFINE_RAW_ENUM_CLASS(SemanticsNodeKind, uint8_t) {
#define CARBON_SEMANTICS_NODE_KIND(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "toolchain/semantics/semantics_node_kind.def"
};

class SemanticsNodeKind : public CARBON_ENUM_BASE(SemanticsNodeKind) {
 public:
#define CARBON_SEMANTICS_NODE_KIND(Name) CARBON_ENUM_CONSTANT_DECLARATION(Name)
#include "toolchain/semantics/semantics_node_kind.def"
};

#define CARBON_SEMANTICS_NODE_KIND(Name) \
  CARBON_ENUM_CONSTANT_DEFINITION(SemanticsNodeKind, Name)
#include "toolchain/semantics/semantics_node_kind.def"

// We expect the node kind to fit compactly into 8 bits.
static_assert(sizeof(SemanticsNodeKind) == 1, "Kind objects include padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_KIND_H_
