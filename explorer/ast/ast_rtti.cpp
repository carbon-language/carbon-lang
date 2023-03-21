// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/ast_rtti.h"

namespace Carbon {

#define IGNORE(C)

// Define kind names for the base enumeration type.
CARBON_DEFINE_ENUM_CLASS_NAMES(AstRttiNodeKind) = {
    CARBON_AST_RTTI_KINDS(IGNORE, CARBON_ENUM_CLASS_NAME_STRING)};

// For other kind enumerations, reuse the same table.
#define SPECIALIZE_NAME_FUNCTION(C)                                    \
  template <>                                                          \
  auto CARBON_ENUM_BASE(C##Kind)::name() const->llvm::StringRef {      \
    return AstRttiNodeKind(static_cast<const C##Kind&>(*this)).name(); \
  }
CARBON_AST_RTTI_KINDS(SPECIALIZE_NAME_FUNCTION, IGNORE)

}  // namespace Carbon
