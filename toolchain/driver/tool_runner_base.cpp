// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/tool_runner_base.h"

#include <memory>

#include "common/vlog.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

ToolRunnerBase::ToolRunnerBase(const InstallPaths* install_paths,
                               llvm::raw_ostream* vlog_stream)
    : installation_(install_paths), vlog_stream_(vlog_stream) {}

auto ToolRunnerBase::BuildCStrArgs(llvm::StringRef tool_name,
                                   llvm::StringRef tool_path,
                                   std::optional<llvm::StringRef> verbose_flag,
                                   llvm::ArrayRef<llvm::StringRef> args,
                                   llvm::OwningArrayRef<char>& cstr_arg_storage)
    -> llvm::SmallVector<const char*, 64> {
  // TODO: Maybe handle response file expansion similar to the Clang CLI?

  // If we have a verbose logging stream, and that stream is the same as
  // `llvm::errs`, then add the `-v` flag so that the driver also prints verbose
  // information.
  bool inject_v_arg = verbose_flag.has_value() && vlog_stream_ == &llvm::errs();
  std::array<llvm::StringRef, 1> v_arg_storage;
  llvm::ArrayRef<llvm::StringRef> maybe_v_arg;
  if (inject_v_arg) {
    v_arg_storage[0] = *verbose_flag;
    maybe_v_arg = v_arg_storage;
  }

  CARBON_VLOG("Running {} driver with arguments:\n", tool_name);

  // Render the arguments into null-terminated C-strings. Command lines can get
  // quite long in build systems so this tries to minimize the memory allocation
  // overhead.

  // Provide the wrapped tool path as the synthetic `argv[0]`.
  std::array<llvm::StringRef, 1> exe_arg = {tool_path};
  auto args_range =
      llvm::concat<const llvm::StringRef>(exe_arg, maybe_v_arg, args);
  int total_size = 0;
  for (llvm::StringRef arg : args_range) {
    // Accumulate both the string size and a null terminator byte.
    total_size += arg.size() + 1;
  }

  // Allocate one chunk of storage for the actual C-strings and a vector of
  // pointers into the storage.
  cstr_arg_storage = llvm::OwningArrayRef<char>(total_size);
  llvm::SmallVector<const char*, 64> cstr_args;
  cstr_args.reserve(args.size() + inject_v_arg + 1);
  for (ssize_t i = 0; llvm::StringRef arg : args_range) {
    cstr_args.push_back(&cstr_arg_storage[i]);
    memcpy(&cstr_arg_storage[i], arg.data(), arg.size());
    i += arg.size();
    cstr_arg_storage[i] = '\0';
    ++i;
  }
  for (const char* cstr_arg : llvm::ArrayRef(cstr_args)) {
    CARBON_VLOG("    '{0}'\n", cstr_arg);
  }

  return cstr_args;
}

}  // namespace Carbon
