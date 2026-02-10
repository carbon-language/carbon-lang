// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/llvm_runner.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>
#include <string>

#include "common/string_helpers.h"
#include "common/vlog.h"
#include "lld/Common/Driver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"

namespace Carbon {

auto LLVMRunner::Run(LLVMTool tool, llvm::ArrayRef<llvm::StringRef> args)
    -> bool {
  std::string path = installation_->llvm_tool_path(tool);

  // Allocate one chunk of storage for the actual C-strings and a vector of
  // pointers into the storage.
  llvm::BumpPtrAllocator alloc;
  llvm::SmallVector<const char*, 64> cstr_args =
      BuildCStrArgs(path, args, alloc);

  CARBON_VLOG("Running LLVM's {0} tool with args:\n", tool.name());
  for (const char* cstr_arg : cstr_args) {
    CARBON_VLOG("    '{0}'\n", cstr_arg);
  }

  int exit_code = tool.main_fn()(
      cstr_args.size(), const_cast<char**>(cstr_args.data()),
      {.Path = path.c_str(), .PrependArg = nullptr, .NeedsPrependArg = false});

  // TODO: Should this be forwarding the full exit code?
  return exit_code == 0;
}

}  // namespace Carbon
