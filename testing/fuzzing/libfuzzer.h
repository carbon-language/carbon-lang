// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_FUZZING_LIBFUZZER_H_
#define CARBON_TESTING_FUZZING_LIBFUZZER_H_

#include <cstddef>

namespace Carbon::Testing {

// Declaration for the LLVM libfuzzer API that we implement in our fuzz tests.
// This is useful to ensure we get the API correct and avoid warnings about
// defining an undeclared extern function due to a Clang warning bug:
// https://github.com/llvm/llvm-project/issues/94138
// NOLINTNEXTLINE: Match the documented fuzzer entry point declaration style.
extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data,
                                      std::size_t size);

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_FUZZING_LIBFUZZER_H_
