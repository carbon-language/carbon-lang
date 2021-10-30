// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/ostream.h"

#include <ostream>

namespace Carbon::Internal {

void PrintNullPointer(std::ostream &out) {
  out << "NULL";
}

}  // namespace Carbon::Internal
