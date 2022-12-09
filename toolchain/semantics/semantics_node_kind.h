// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_KIND_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_KIND_H_

#include "toolchain/common/enum_base.h"

namespace Carbon {

// Enum values for SemanticsNodeKind, listing kinds of SemanticNodes.
//
// This is an X-macro; the argument should be a macro taking a single argument,
// the name.
#define CARBON_SEMANTICS_NODE_KINDS(X) \
  X(Invalid)                           \
  X(BinaryOperatorAdd)                 \
  X(BindName)                          \
  X(Builtin)                           \
  X(CodeBlock)                         \
  X(FunctionDeclaration)               \
  X(FunctionDefinition)                \
  X(IntegerLiteral)                    \
  X(RealLiteral)                       \
  X(Return)                            \
  X(ReturnExpression)

CARBON_ENUM_BASE(SemanticsNodeKindBase, CARBON_SEMANTICS_NODE_KINDS)

class SemanticsNodeKind : public SemanticsNodeKindBase<SemanticsNodeKind> {
  using SemanticsNodeKindBase::SemanticsNodeKindBase;
};

// We expect the node kind to fit compactly into 8 bits.
static_assert(sizeof(SemanticsNodeKind) == 1, "Kind objects include padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_KIND_H_
