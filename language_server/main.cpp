// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "language_server/language_server.h"

auto main(int /*argc*/, char** /*argv*/) -> int {
  Carbon::LS::LanguageServer::Start();
  return 0;
}
