// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_MODIFIERS_ALLOWED_H_
#define CARBON_TOOLCHAIN_CHECK_MODIFIERS_ALLOWED_H_

#include "toolchain/check/context.h"

namespace Carbon::Check {

// Reports a diagnostic (using `decl_name`) if access control modifiers on this
// are not allowed, and updates the declaration state in `context`.
auto CheckAccessModifiersOnDecl(Context& context, llvm::StringRef decl_name)
    -> void;

// Reports a diagnostic (using `decl_name`) if modifiers on this declaration are
// not in `allowed`. Returns modifiers that were both found and allowed, and
// updates the declaration state in `context`. Specify optional Parse::Node
// `context_node` to add a note to the diagnostic about the containing
// definition.
auto LimitModifiersOnDecl(Context& context, KeywordModifierSet allowed,
                          llvm::StringRef decl_name,
                          Parse::Node context_node = Parse::Node::Invalid)
    -> KeywordModifierSet;

// Like LimitModifiersOnDecl, except says which modifiers are forbidden,
// and a `context_string` specifying the context in which those modifiers are
// forbidden.
auto ForbidModifiersOnDecl(Context& context, KeywordModifierSet forbidden,
                           llvm::StringRef decl_name,
                           llvm::StringRef context_string,
                           Parse::Node context_node = Parse::Node::Invalid)
    -> void;

// Report a diagonostic if `default` and `final` modifiers are used on
// declarations where they are not allowed. Right now they are only allowed
// inside interfaces.
auto RequireDefaultFinalOnlyInInterfaces(Context& context,
                                         llvm::StringRef decl_name)
    -> KeywordModifierSet;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_MODIFIERS_ALLOWED_H_
