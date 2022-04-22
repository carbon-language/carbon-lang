// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/runfiles_util.h"

#include <filesystem>

#include "common/check.h"

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#include <sys/param.h>
#include <sys/stat.h>
#endif

namespace Carbon {

static auto GetProgramPath() -> std::string {
  std::string program_name;
#if defined(__APPLE__)
  char ch = '\0';
  uint32_t buf_size = 1;
  // First call is to determine required buffer size.
  CHECK(_NSGetExecutablePath(&ch, &buf_size) < 0);
  std::unique_ptr<char[]> buf(new char[buf_size]);
  CHECK(_NSGetExecutablePath(buf.get(), &buf_size) == 0);
  program_name = buf.get();
#else
  program_name = "/proc/self/exe";
#endif
  std::error_code error;
  program_name = std::filesystem::canonical(program_name, error);
  CHECK(error.value() == 0);
  llvm::errs() << "### program name=" << program_name << "\n";
  return program_name;
}

auto GetRunfilesDir() -> std::string {
  std::string runfiles_dir = GetProgramPath() + ".runfiles";
  llvm::errs() << "### runfiles dir=" << runfiles_dir << "\n";
  CHECK(std::filesystem::exists(runfiles_dir));
  return runfiles_dir;
}

}  // namespace Carbon
