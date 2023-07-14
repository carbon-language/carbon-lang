// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "explorer/common/trace_stream.h"

namespace Carbon::Testing {
namespace {

// TODO: write test cases to distinguish between file context of main file and
// an import once imports are supported.

TEST(SetFileContextRaiiTest, Simple) {
  TraceStream trace_stream;
  {
    SetFileContext set_file_ctx(trace_stream,
                                SourceLocation("example/main.carbon", 9));
    EXPECT_TRUE(trace_stream.file_context() == FileContext::Main);
  }

  // Considering the file context for a trace stream is FileContext::Unknown by
  // default, as the default value of source location in a trace stream is
  // std::nullopt.
  EXPECT_TRUE(trace_stream.file_context() == FileContext::Unknown);
}

TEST(SetFileContextRaiiTest, UpdateFileContext) {
  TraceStream trace_stream;
  {
    SetFileContext set_file_ctx(trace_stream,
                                SourceLocation("example/prelude.carbon", 9));
    EXPECT_TRUE(trace_stream.file_context() == FileContext::Prelude);
    set_file_ctx.update_source_loc(SourceLocation("example/main.carbon", 9));
    EXPECT_TRUE(trace_stream.file_context() == FileContext::Main);
  }

  // Considering the file context for a trace stream is FileContext::Unknown by
  // default, as the default value of source location in a trace stream is
  // std::nullopt.
  EXPECT_TRUE(trace_stream.file_context() == FileContext::Unknown);
}

}  // namespace
}  // namespace Carbon::Testing
