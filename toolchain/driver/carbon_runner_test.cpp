// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/carbon_runner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <utility>

#include "common/check.h"
#include "common/raw_string_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "testing/base/capture_std_streams.h"
#include "testing/base/global_exe_path.h"
#include "toolchain/driver/driver_env.h"
#include "toolchain/driver/llvm_runner.h"
#include "toolchain/install/install_paths.h"

namespace Carbon {
namespace {

using ::testing::HasSubstr;
using ::testing::StrEq;

class CarbonRunnerTest : public ::testing::Test {
 public:
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs_ =
      llvm::vfs::getRealFileSystem();
  RawStringOstream test_output_stream_;
  RawStringOstream test_error_stream_;

  InstallPaths install_paths_ =
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath());
  DriverEnv driver_env_ = DriverEnv(
      vfs_, &install_paths_,
      /*input_stream=*/nullptr, &test_output_stream_, &test_error_stream_,
      /*fuzzing*/ false, /*enable_leaking=*/false);

  Runtimes::Cache runtimes_cache_ =
      *Runtimes::Cache::MakeSystem(*driver_env_.installation);
};

TEST_F(CarbonRunnerTest, BuildCoreLibraries) {
  CarbonRunner runner(&driver_env_);

  // Note that we can't test arbitrary targets here as we need to be able
  // to compile the builtin functions for the target. We use the default
  // target as the most likely to pass.
  std::string target = llvm::sys::getDefaultTargetTriple();
  llvm::Triple target_triple(target);
  Runtimes::Cache::Features features = {.target = target};
  auto runtimes = *runtimes_cache_.Lookup(features);
  auto build_result = runner.BuildCoreLibraries(features, runtimes);
  ASSERT_TRUE(build_result.ok()) << build_result.error();
  std::filesystem::path archive_path = std::move(*build_result);

  // Check that the Core archive exists and contains a relevant symbol by
  // running the `llvm-nm` tool over it. Using `nm` rather than directly
  // inspecting the objects is a bit awkward, but lets us easily ignore the
  // wrapping in an archive file.
  LLVMRunner llvm_runner(driver_env_.installation, &llvm::errs());
  std::string out;
  std::string err;
  EXPECT_TRUE(Testing::CallWithCapturedOutput(out, err, [&] {
    return llvm_runner.Run(LLVMTool::Nm, {archive_path.native()});
  }));

  // Check that we found a symbol from the core libraries.
  EXPECT_THAT(out, HasSubstr(target_triple.isMacOSX() ? "T __CRange.Core\n"
                                                      : "T _CRange.Core\n"));

  EXPECT_THAT(test_output_stream_.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
}

}  // namespace
}  // namespace Carbon
