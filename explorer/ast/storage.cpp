// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/storage.h"

namespace Carbon {
void Storage::Print(llvm::raw_ostream& out) const {
  // TODO: Add function name?
  out << "initializing expr storage";
}

void Storage::PrintID(llvm::raw_ostream& out) const {
  out << "initializing expr storage";
}

}  // namespace Carbon
