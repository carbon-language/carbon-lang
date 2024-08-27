// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_MANGLER_H_
#define CARBON_TOOLCHAIN_LOWER_MANGLER_H_

#include <string>

#include "toolchain/sem_ir/file.h"

namespace Carbon::Lower {
class Mangler {
 public:
  explicit Mangler(const SemIR::File& sem_ir) : sem_ir_(sem_ir) {}
  auto Mangle(SemIR::FunctionId function_id) -> std::string;

 private:
  const SemIR::File& sem_ir_;
};
}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_MANGLER_H_
