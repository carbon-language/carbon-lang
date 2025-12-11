// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/tool_runner_base.h"

#include <array>
#include <memory>
#include <optional>

#include "common/vlog.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

ToolRunnerBase::ToolRunnerBase(const InstallPaths* install_paths,
                               llvm::raw_ostream* vlog_stream)
    : installation_(install_paths), vlog_stream_(vlog_stream) {}

}  // namespace Carbon
