// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/ast_rtti.h"

namespace Carbon {

// Define kind names for the base enumeration type.
CARBON_DEFINE_ENUM_CLASS_NAMES(AstRttiNodeKind) = {
    CARBON_AST_FOR_EACH_FINAL_CLASS(CARBON_ENUM_CLASS_NAME_STRING)};

// For other kind enumerations, reuse the same table.
#define DEFINE_NAME_FUNCTION(C)                                        \
  CARBON_ENUM_NAME_FUNCTION(C##Kind) {                                 \
    return AstRttiNodeKind(static_cast<const C##Kind&>(*this)).name(); \
  }

}  // namespace Carbon
