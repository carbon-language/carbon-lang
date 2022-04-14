// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPLORER_SYNTAX_PRELUDE_H_
#define EXPLORER_SYNTAX_PRELUDE_H_

#include <string_view>

#include "explorer/ast/declaration.h"
#include "explorer/common/arena.h"
#include "explorer/common/nonnull.h"

namespace Carbon {

// Adds the Carbon prelude to `declarations`.
void AddPrelude(std::string_view prelude_file_name, Nonnull<Arena*> arena,
                std::vector<Nonnull<Declaration*>>* declarations);

}  // namespace Carbon

#endif  // EXPLORER_SYNTAX_PRELUDE_H_
