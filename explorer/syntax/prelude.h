// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_SYNTAX_PRELUDE_H_
#define CARBON_EXPLORER_SYNTAX_PRELUDE_H_

#include <string_view>

#include "explorer/ast/declaration.h"
#include "explorer/base/arena.h"
#include "explorer/base/nonnull.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace Carbon {

// Adds the Carbon prelude to `declarations`.
void AddPrelude(llvm::vfs::FileSystem& fs, std::string_view prelude_file_name,
                Nonnull<Arena*> arena,
                std::vector<Nonnull<Declaration*>>* declarations,
                int* num_prelude_declarations);

}  // namespace Carbon

#endif  // CARBON_EXPLORER_SYNTAX_PRELUDE_H_
