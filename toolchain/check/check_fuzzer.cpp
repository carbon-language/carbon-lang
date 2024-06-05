// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/exe_path.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "testing/fuzzing/libfuzzer.h"
#include "toolchain/driver/driver.h"

namespace Carbon::Testing {

static InstallPaths install_paths;

// NOLINTNEXTLINE(readability-non-const-parameter): External API required types.
extern "C" auto LLVMFuzzerInitialize(int* argc, char*** argv) -> int {
  std::string exe_path;
  if (*argc >= 1) {
    exe_path = FindExecutablePath((*argv)[0]);
    install_paths = InstallPaths::MakeForBazelRunfiles(exe_path);
  }
  return 0;
}

// NOLINTNEXTLINE: Match the documented fuzzer entry point declaration style.
extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data,
                                      std::size_t size) {
  // Ignore large inputs.
  // TODO: See tokenized_buffer_fuzzer.cpp.
  if (size > 100000) {
    return 0;
  }

  static constexpr llvm::StringLiteral TestFileName = "test.carbon";
  llvm::vfs::InMemoryFileSystem fs;
  llvm::StringRef data_ref(reinterpret_cast<const char*>(data), size);
  CARBON_CHECK(fs.addFile(
      TestFileName, /*ModificationTime=*/0,
      llvm::MemoryBuffer::getMemBuffer(data_ref, /*BufferName=*/TestFileName,
                                       /*RequiresNullTerminator=*/false)));

  llvm::raw_null_ostream null_ostream;
  Driver driver(fs, &install_paths, null_ostream, null_ostream);

  // TODO: Get checking to a point where it can handle invalid parse trees
  // without crashing.
  if (!driver.RunCommand({"compile", "--phase=parse", TestFileName}).success) {
    return 0;
  }
  driver.RunCommand({"compile", "--phase=check", TestFileName});
  return 0;
}

}  // namespace Carbon::Testing
