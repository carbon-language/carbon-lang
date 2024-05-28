// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/exe_path.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

namespace Carbon {

auto FindExecutablePath(llvm::StringRef argv0) -> std::string {
  if (!llvm::sys::fs::exists(argv0)) {
    if (llvm::ErrorOr<std::string> path = llvm::sys::findProgramByName(argv0)) {
      return std::move(*path);
    }
  }

  return argv0.str();
}

}  // namespace Carbon
