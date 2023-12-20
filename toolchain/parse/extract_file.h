// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_EXTRACT_FILE_H_
#define CARBON_TOOLCHAIN_PARSE_EXTRACT_FILE_H_

#include <tuple>
#include <utility>

#include "toolchain/parse/tree.h"
#include "toolchain/parse/typed_nodes.h"

namespace Carbon::Parse {

// A complete source file. Note that there is no corresponding parse node for
// the file. The file is instead the complete contents of the parse tree.
struct File {
  FileStartId start;
  llvm::SmallVector<AnyDecl> decls;
  FileEndId end;

  static auto Make(const Tree* tree) -> File {
    return tree->ExtractNodeFromChildren<File>(tree->roots());
  }
};

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_EXTRACT_FILE_H_
