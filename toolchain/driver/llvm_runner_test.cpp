// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/llvm_runner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/ostream.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FormatVariadic.h"
#include "testing/base/capture_std_streams.h"
#include "testing/base/global_exe_path.h"

namespace Carbon {
namespace {

using ::testing::HasSubstr;
using ::testing::StrEq;

TEST(LLVMRunnerTest, Version) {
  RawStringOstream test_os;
  const auto install_paths =
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath());
  LLVMRunner runner(&install_paths, &test_os);

  std::string out;
  std::string err;

  for (LLVMTool tool : LLVMTool::Tools) {
    std::string test_flag = "--version";
    std::string expected_out = "LLVM version";

    // Handle any special requirements for specific tools.
    switch (tool) {
      case LLVMTool::Addr2Line:
      case LLVMTool::BitcodeStrip:
      case LLVMTool::Cgdata:
      case LLVMTool::DebuginfodFind:
      case LLVMTool::Dwp:
      case LLVMTool::Gsymutil:
      case LLVMTool::Ifs:
      case LLVMTool::InstallNameTool:
      case LLVMTool::Lipo:
      case LLVMTool::Objcopy:
      case LLVMTool::Profdata:
      case LLVMTool::Sancov:
      case LLVMTool::Strip:
      case LLVMTool::Symbolizer:
      case LLVMTool::Windres:
        // TODO: These tools are not well behaved when invoked as a library,
        // typically directly calling `exit` on some or all code paths. We
        // should see if they can be fixed upstream and re-enable testing.
        continue;

      case LLVMTool::Dlltool:
      case LLVMTool::Rc:
        // No good flags to generically test these tools.
        continue;

      case LLVMTool::Lib:
        test_flag = "/help";
        expected_out = "LLVM Lib";
        break;

      case LLVMTool::Ml:
        test_flag = "/help";
        expected_out = "LLVM MASM Assembler";
        break;

      case LLVMTool::Mt:
        test_flag = "/help";
        expected_out = "Manifest Tool";
        break;

      default:
        break;
    }

    EXPECT_TRUE(Testing::CallWithCapturedOutput(
        out, err, [&] { return runner.Run(tool, {test_flag}); }));

    // The arguments to the LLVM tool should be part of the verbose log.
    EXPECT_THAT(test_os.TakeStr(), HasSubstr(test_flag));

    // Nothing should print to stderr here.
    EXPECT_THAT(err, StrEq(""));

    EXPECT_THAT(out, HasSubstr(expected_out));
  }
}

}  // namespace
}  // namespace Carbon
