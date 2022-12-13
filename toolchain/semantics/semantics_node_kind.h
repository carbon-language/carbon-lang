// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_KIND_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_KIND_H_

#include "toolchain/common/enum_base.h"

namespace Carbon {

namespace Internal {
enum class SemanticsNodeKindEnum : uint8_t {
#define CARBON_SEMANTICS_NODE_KIND(Name) CARBON_ENUM_BASE_LITERAL(Name)
#include "toolchain/semantics/semantics_node_kind.def"
};
}  // namespace Internal

class SemanticsNodeKind
    : public EnumBase<SemanticsNodeKind, Internal::SemanticsNodeKindEnum> {
 public:
#define CARBON_SEMANTICS_NODE_KIND(Name) \
  CARBON_ENUM_BASE_FACTORY(SemanticsNodeKind, Name)
#include "toolchain/semantics/semantics_node_kind.def"

  // Gets a friendly name for the token for logging or debugging.
  [[nodiscard]] inline auto name() const -> llvm::StringRef {
    static constexpr llvm::StringLiteral Names[] = {
#define CARBON_SEMANTICS_NODE_KIND(Name) CARBON_ENUM_BASE_STRING(Name)
#include "toolchain/semantics/semantics_node_kind.def"
    };
    return Names[static_cast<int>(val_)];
  }

 private:
  using EnumBase::EnumBase;
};

// We expect the node kind to fit compactly into 8 bits.
static_assert(sizeof(SemanticsNodeKind) == 1, "Kind objects include padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_KIND_H_
