// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_MANGLER_H_
#define CARBON_TOOLCHAIN_LOWER_MANGLER_H_

#include <string>

#include "toolchain/lower/file_context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Lower {
class Mangler {
 public:
  explicit Mangler(FileContext& file_context) : file_context_(file_context) {}
  auto Mangle(SemIR::FunctionId function_id) -> std::string;

 private:
  auto MangleInverseQualifiedNameScope(bool first_name_component,
                                       llvm::raw_ostream& os,
                                       SemIR::NameScopeId class_id) -> void;
  FileContext& file_context_;
};
}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_MANGLER_H_
