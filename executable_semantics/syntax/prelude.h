// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_SYNTAX_PRELUDE_H_
#define EXECUTABLE_SEMANTICS_SYNTAX_PRELUDE_H_

#include <string_view>

#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/nonnull.h"

namespace Carbon {

// Adds the Carbon prelude to `declarations`.
void AddPrelude(std::string_view prelude_file_name, Nonnull<Arena*> arena,
                std::vector<Nonnull<Declaration*>>* declarations);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_SYNTAX_PRELUDE_H_
