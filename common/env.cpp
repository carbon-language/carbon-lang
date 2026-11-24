// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/env.h"

#include <cstdlib>

#include "common/check.h"

namespace Carbon {

auto SetEnv(const char* name, const char* value, bool overwrite) -> void {
#ifdef _WIN32
  if (!overwrite && std::getenv(name) != nullptr) {
    return;
  }
  CARBON_CHECK(_putenv_s(name, value) == 0,
               "Failed to set environment variable '{0}'", name);
#else
  CARBON_CHECK(setenv(name, value, overwrite ? 1 : 0) == 0,
               "Failed to set environment variable '{0}'", name);
#endif
}

auto UnsetEnv(const char* name) -> void {
#ifdef _WIN32
  CARBON_CHECK(_putenv_s(name, "") == 0,
               "Failed to unset environment variable '{0}'", name);
#else
  CARBON_CHECK(unsetenv(name) == 0,
               "Failed to unset environment variable '{0}'", name);
#endif
}

}  // namespace Carbon
