// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/frame.h"

#include "common/ostream.h"
#include "executable_semantics/interpreter/action.h"

namespace Carbon {

void Frame::Print(llvm::raw_ostream& out) const {
  out << name << "{";
  Action::PrintList(todo, out);
  out << "}";
}

}  // namespace Carbon
