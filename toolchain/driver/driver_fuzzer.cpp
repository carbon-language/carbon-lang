// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <string>

#include "common/exe_path.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "testing/fuzzing/libfuzzer.h"
#include "toolchain/driver/driver.h"
#include "toolchain/install/install_paths.h"

namespace Carbon::Testing {

static const InstallPaths* install_paths = nullptr;

// NOLINTNEXTLINE(readability-non-const-parameter): External API required types.
extern "C" auto LLVMFuzzerInitialize(int* argc, char*** argv) -> int {
  CARBON_CHECK(*argc >= 1, "Need the `argv[0]` value to initialize!");
  install_paths = new InstallPaths(
      InstallPaths::MakeForBazelRunfiles(FindExecutablePath((*argv)[0])));
  return 0;
}

static auto Read(const unsigned char*& data, size_t& size, int& output)
    -> bool {
  if (size < sizeof(output)) {
    return false;
  }
  std::memcpy(&output, data, sizeof(output));
  size -= sizeof(output);
  data += sizeof(output);
  return true;
}

extern "C" auto LLVMFuzzerTestOneInput(const unsigned char* data, size_t size)
    -> int {
  // First use the data to compute the number of arguments. Note that for
  // scaling reasons we don't allow 2^31 arguments, even empty ones. Simply
  // creating the vector of those won't work. We limit this to 2^20 arguments
  // total.
  int num_args;
  if (!Read(data, size, num_args) || num_args < 0 || num_args > (1 << 20)) {
    return 0;
  }

  // Now use the data to compute the length of each argument. We don't want to
  // exhaust all memory, so bound the search space to using 2^17 bytes of
  // memory for the argument text itself.
  size_t arg_length_sum = 0;
  llvm::SmallVector<int> arg_lengths(num_args);
  for (int& arg_length : arg_lengths) {
    if (!Read(data, size, arg_length) || arg_length < 0) {
      return 0;
    }
    arg_length_sum += arg_length;
    if (arg_length_sum > (1 << 17)) {
      return 0;
    }
  }

  // Ensure we have enough data for all the arguments.
  if (size < arg_length_sum) {
    return 0;
  }

  // Lastly, read the contents of each argument out of the data.
  llvm::SmallVector<llvm::StringRef> args;
  args.reserve(num_args);
  for (int arg_length : arg_lengths) {
    args.push_back(
        llvm::StringRef(reinterpret_cast<const char*>(data), arg_length));
    data += arg_length;
    size -= arg_length;
  }

  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs =
      new llvm::vfs::InMemoryFileSystem;
  RawStringOstream error_stream;
  llvm::raw_null_ostream null_ostream;
  Driver driver(fs, install_paths, /*input_stream=*/nullptr, &null_ostream,
                &error_stream, /*fuzzing=*/true);
  if (!driver.RunCommand(args).success) {
    auto str = error_stream.TakeStr();
    if (llvm::StringRef(str).find("error:") == llvm::StringRef::npos) {
      llvm::errs() << "No error message on a failure!\n";
      return 1;
    }
  }
  return 0;
}
}  // namespace Carbon::Testing
