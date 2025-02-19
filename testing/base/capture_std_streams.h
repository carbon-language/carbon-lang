// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_BASE_CAPTURE_STD_STREAMS_H_
#define CARBON_TESTING_BASE_CAPTURE_STD_STREAMS_H_

#include <string>

namespace Carbon::Testing {

// Implementation details.
namespace Internal {
auto BeginStdStreamCapture() -> void;
auto EndStdStreamCapture(std::string& out, std::string& err) -> void;
}  // namespace Internal

// Calls the provided function while capturing both `stdout` and `stderr`. The
// `out` and `err` strings are set to whatever is captured after the function
// returns.
//
// Note that any output that is captured will not be visible when running, which
// can make debugging tests that use this routine difficult. Make sure to either
// print back out or otherwise expose any of the contents of the captured output
// that are needed when debugging.
template <typename FnT>
static auto CallWithCapturedOutput(std::string& out, std::string& err,
                                   FnT function) {
  Internal::BeginStdStreamCapture();
  auto result = function();
  Internal::EndStdStreamCapture(out, err);
  return result;
}

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_BASE_CAPTURE_STD_STREAMS_H_
