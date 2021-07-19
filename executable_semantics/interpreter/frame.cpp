// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/frame.h"

#include <ostream>

#include "executable_semantics/interpreter/action.h"

namespace Carbon {

void PrintFrame(Frame* frame, std::ostream& out) {
  out << frame->name;
  out << "{";
  Action::PrintList(frame->todo, out);
  out << "}";
}

}  // namespace Carbon
