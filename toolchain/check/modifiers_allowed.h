// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_MODIFIERS_ALLOWED_H_
#define CARBON_TOOLCHAIN_CHECK_MODIFIERS_ALLOWED_H_

#include "toolchain/check/context.h"

namespace Carbon::Check {

// Reports a diagnostic (using `decl_name`) if modifiers on this declaration are
// not in `allowed`. Returns modifiers that were both found and allowed, and the
// parse node corresponding to the first token of the declaration.
// FIXME: remove Parse::Node return
// FIXME: add optional Parse::Node context which adds a note.
auto ModifiersAllowedOnDecl(Context& context, KeywordModifierSet allowed,
                            llvm::StringRef decl_name)
    -> std::pair<KeywordModifierSet, Parse::Node>;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_MODIFIERS_ALLOWED_H_
