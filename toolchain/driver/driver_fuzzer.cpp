// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "testing/base/test_raw_ostream.h"
#include "toolchain/driver/driver.h"

namespace Carbon::Testing {

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

  llvm::vfs::InMemoryFileSystem fs;
  TestRawOstream error_stream;
  llvm::raw_null_ostream dest;
  Driver d(fs, "", dest, error_stream);
  if (!d.RunCommand(args).success) {
    if (error_stream.TakeStr().find("ERROR:") == std::string::npos) {
      llvm::errs() << "No error message on a failure!\n";
      return 1;
    }
  }
  return 0;
}
}  // namespace Carbon::Testing
