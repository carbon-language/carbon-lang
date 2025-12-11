// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_TOOL_RUNNER_BASE_H_
#define CARBON_TOOLCHAIN_DRIVER_TOOL_RUNNER_BASE_H_

#include <optional>

#include "common/ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/base/install_paths.h"

namespace Carbon {

// Base  that factors out common utilities needed when implementing a runner of
// an external tool, especially tools part of the LLVM and Clang C++ toolchain
// that Carbon will end up wrapping.
//
// Note that this struct just collects common data and helper methods, and does
// not itself impose any invariants or form a meaningful API. It should be used
// as an implementation detail only.
class ToolRunnerBase {
 public:
  // Construct the tool runner bas.
  //
  // If `vlog_stream` is provided, it will be used for `CARBON_VLOG`s. If it is
  // also equal to `&llvm::errs()`, and so tied to stderr, that will be used by
  // verbose flag injection helpers in this class.
  explicit ToolRunnerBase(const InstallPaths* install_paths,
                          llvm::raw_ostream* vlog_stream = nullptr);

 protected:
  // We use protected members as this base is just factoring out common
  // implementation details of other runners.
  //
  // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
  const InstallPaths* installation_;
  llvm::raw_ostream* vlog_stream_;
  // NOLINTEND(misc-non-private-member-variables-in-classes)
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_TOOL_RUNNER_BASE_H_
