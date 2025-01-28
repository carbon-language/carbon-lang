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

static const InstallPaths* install_paths = nullptr;

// NOLINTNEXTLINE(readability-non-const-parameter): External API required types.
extern "C" auto LLVMFuzzerInitialize(int* argc, char*** argv) -> int {
  CARBON_CHECK(*argc >= 1, "Need the `argv[0]` value to initialize!");
  install_paths = new InstallPaths(
      InstallPaths::MakeForBazelRunfiles(FindExecutablePath((*argv)[0])));
  return 0;
}

// NOLINTNEXTLINE: Match the documented fuzzer entry point declaration style.
extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data, size_t size) {
  // Ignore large inputs.
  // TODO: See tokenized_buffer_fuzzer.cpp.
  if (size > 100000) {
    return 0;
  }

  static constexpr llvm::StringLiteral TestFileName = "test.carbon";
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs =
      new llvm::vfs::InMemoryFileSystem;
  llvm::StringRef data_ref(reinterpret_cast<const char*>(data), size);
  CARBON_CHECK(fs->addFile(
      TestFileName, /*ModificationTime=*/0,
      llvm::MemoryBuffer::getMemBuffer(data_ref, /*BufferName=*/TestFileName,
                                       /*RequiresNullTerminator=*/false)));

  llvm::raw_null_ostream null_ostream;
  Driver driver(fs, install_paths, /*input_stream=*/nullptr, &null_ostream,
                &null_ostream, /*fuzzing=*/true);

  // TODO: Get checking to a point where it can handle invalid parse trees
  // without crashing.
  if (!driver.RunCommand({"compile", "--phase=parse", TestFileName}).success) {
    return 0;
  }
  driver.RunCommand({"compile", "--phase=check", TestFileName});
  return 0;
}

}  // namespace Carbon::Testing
