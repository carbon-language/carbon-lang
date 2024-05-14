// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "explorer/base/trace_stream.h"

namespace Carbon {
namespace {

TEST(SetFileContextRaiiTest, UpdateFileContext) {
  TraceStream trace_stream;
  trace_stream.set_stream(&llvm::nulls());
  trace_stream.set_allowed_phases({ProgramPhase::All});
  trace_stream.set_allowed_file_kinds({FileKind::Main});

  {
    SetFileContext set_file_ctx(
        trace_stream,
        SourceLocation("example/prelude.carbon", 9, FileKind::Prelude));
    EXPECT_FALSE(trace_stream.is_enabled());
    set_file_ctx.update_source_loc(
        SourceLocation("example/main.carbon", 9, FileKind::Main));
    EXPECT_TRUE(trace_stream.is_enabled());
    set_file_ctx.update_source_loc(
        SourceLocation("example/import.carbon", 9, FileKind::Import));
    EXPECT_FALSE(trace_stream.is_enabled());
  }

  // The trace stream should be enabled when we're not in any particular file.
  EXPECT_TRUE(trace_stream.is_enabled());
}

}  // namespace
}  // namespace Carbon
