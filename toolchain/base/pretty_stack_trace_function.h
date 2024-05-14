// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_PRETTY_STACK_TRACE_FUNCTION_H_
#define CARBON_TOOLCHAIN_BASE_PRETTY_STACK_TRACE_FUNCTION_H_

#include <functional>

#include "llvm/Support/PrettyStackTrace.h"

namespace Carbon {

class PrettyStackTraceFunction : public llvm::PrettyStackTraceEntry {
 public:
  explicit PrettyStackTraceFunction(std::function<void(llvm::raw_ostream&)> fn)
      : fn_(std::move(fn)) {}
  ~PrettyStackTraceFunction() override = default;

  auto print(llvm::raw_ostream& output) const -> void override { fn_(output); }

 private:
  const std::function<void(llvm::raw_ostream&)> fn_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_PRETTY_STACK_TRACE_FUNCTION_H_
