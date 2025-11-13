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

TEST_F(CarbonRunnerTest, BuildCoreLibrariesX8664Linux) {
  CarbonRunner runner(&driver_env_);

  // Build Core libs for x86_64-unknown-linux-gnu.
  std::string target = "x86_64-unknown-linux-gnu";
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
  EXPECT_THAT(out, HasSubstr("T _CRange.Core\n"));

  EXPECT_THAT(test_output_stream_.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
}

TEST_F(CarbonRunnerTest, BuildCoreLibrariesAarch64Darwin) {
  CarbonRunner runner(&driver_env_);

  // Build Core libs for aarch64-apple-darwin.
  std::string target = "aarch64-apple-darwin";
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
  EXPECT_THAT(out, HasSubstr("T __CRange.Core\n"));

  EXPECT_THAT(test_output_stream_.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
}

TEST_F(CarbonRunnerTest, RepeatedBuild) {
  // Repeated "builds" (one from scratch and one cached) should point to the
  // same file.
  std::filesystem::path archive_path1;
  std::filesystem::path archive_path2;

  // Two runners should be able to point to the same driver_env.
  {
    CarbonRunner runner(&driver_env_);

    std::string target = llvm::sys::getDefaultTargetTriple();
    llvm::Triple target_triple(target);
    Runtimes::Cache::Features features = {.target = target};
    auto runtimes = *runtimes_cache_.Lookup(features);
    auto build_result = runner.BuildCoreLibraries(features, runtimes);
    ASSERT_TRUE(build_result.ok()) << build_result.error();
    archive_path1 = std::move(*build_result);
  }
  {
    CarbonRunner runner(&driver_env_);

    std::string target = llvm::sys::getDefaultTargetTriple();
    llvm::Triple target_triple(target);
    Runtimes::Cache::Features features = {.target = target};
    auto runtimes = *runtimes_cache_.Lookup(features);
    auto build_result = runner.BuildCoreLibraries(features, runtimes);
    ASSERT_TRUE(build_result.ok()) << build_result.error();
    archive_path2 = std::move(*build_result);
  }

  ASSERT_TRUE(archive_path1 == archive_path2)
      << "Different paths: " << archive_path1 << ", " << archive_path2 << "\n";

  EXPECT_THAT(test_output_stream_.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
}

}  // namespace
}  // namespace Carbon
