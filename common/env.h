// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_ENV_H_
#define CARBON_COMMON_ENV_H_

namespace Carbon {

// Sets the environment variable `name` to `value`. If `overwrite` is false and
// `name` is already set in the environment, leaves it unchanged.
//
// Note that on Windows, setting a variable to an empty string unsets it in the
// CRT environment block.
auto SetEnv(const char* name, const char* value, bool overwrite = true) -> void;

// Unsets the environment variable `name`.
auto UnsetEnv(const char* name) -> void;

}  // namespace Carbon

#endif  // CARBON_COMMON_ENV_H_
