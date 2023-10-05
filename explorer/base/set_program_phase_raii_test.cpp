// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "explorer/base/trace_stream.h"

namespace Carbon {
namespace {

TEST(SetProgramPhaseRaiiTest, Simple) {
  TraceStream trace_stream;
  trace_stream.set_current_phase(ProgramPhase::Unknown);
  {
    SetProgramPhase set_prog_phase(trace_stream, ProgramPhase::Execution);
    EXPECT_TRUE(trace_stream.current_phase() == ProgramPhase::Execution);
  }
  EXPECT_TRUE(trace_stream.current_phase() == ProgramPhase::Unknown);
}

TEST(SetProgramPhaseRaiiTest, UpdatePhase) {
  TraceStream trace_stream;
  trace_stream.set_current_phase(ProgramPhase::Unknown);
  {
    SetProgramPhase set_prog_phase(trace_stream, ProgramPhase::Execution);
    EXPECT_TRUE(trace_stream.current_phase() == ProgramPhase::Execution);
    set_prog_phase.update_phase(ProgramPhase::TypeChecking);
    EXPECT_TRUE(trace_stream.current_phase() == ProgramPhase::TypeChecking);
  }
  EXPECT_TRUE(trace_stream.current_phase() == ProgramPhase::Unknown);
}

}  // namespace
}  // namespace Carbon
