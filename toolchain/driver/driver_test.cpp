// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "common/error_test_helpers.h"
#include "common/filesystem.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "testing/base/capture_std_streams.h"
#include "testing/base/file_helpers.h"
#include "testing/base/global_exe_path.h"
#include "toolchain/testing/yaml_test_helpers.h"

namespace Carbon {
namespace {

using ::testing::_;
using Testing::CallWithCapturedOutput;
using ::testing::ContainsRegex;
using ::testing::HasSubstr;
using Testing::IsSuccess;
using ::testing::Ne;
using ::testing::NotNull;
using ::testing::StrEq;

namespace Yaml = ::Carbon::Testing::Yaml;

class DriverTest : public testing::Test {
 public:
  DriverTest()
      : installation_(
            InstallPaths::MakeForBazelRunfiles(Testing::GetExePath())),
        driver_(fs_, &installation_, /*input_stream=*/nullptr,
                &test_output_stream_, &test_error_stream_) {
    test_tmpdir_ = Testing::GetTempDirectory();
  }

  auto MakeTestFile(llvm::StringRef text,
                    llvm::StringRef filename = "test_file.carbon")
      -> llvm::StringRef {
    fs_->addFile(filename, /*ModificationTime=*/0,
                 llvm::MemoryBuffer::getMemBuffer(text));
    return filename;
  }

  // Makes a temp directory and changes the working directory to it. Returns an
  // LLVM `scope_exit` that will restore the working directory and remove the
  // temporary directory (and everything it contains) when destroyed.
  auto ScopedTempWorkingDir() {
    // Save our current working directory.
    std::error_code ec;
    auto original_dir = std::filesystem::current_path(ec);
    CARBON_CHECK(!ec, "{0}", ec.message());
    Driver original_driver = std::move(driver_);

    const auto* unit_test = ::testing::UnitTest::GetInstance();
    const auto* test_info = unit_test->current_test_info();
    std::filesystem::path test_dir = test_tmpdir_.append(
        llvm::formatv("{0}_{1}", test_info->test_suite_name(),
                      test_info->name())
            .str());
    std::filesystem::create_directory(test_dir, ec);
    CARBON_CHECK(!ec, "Could not create test working dir '{0}': {1}", test_dir,
                 ec.message());
    std::filesystem::current_path(test_dir, ec);
    CARBON_CHECK(!ec, "Could not change the current working dir to '{0}': {1}",
                 test_dir, ec.message());

    // Build an overlay filesystem between the in-memory one and this new
    // directory.
    llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> overlay_fs =
        new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem());
    overlay_fs->pushOverlay(fs_);

    // Rebuild the driver around this filesystem.
    driver_ = Driver(overlay_fs, &installation_, /*input_stream=*/nullptr,
                     &test_output_stream_, &test_error_stream_);

    return llvm::scope_exit([this, original_dir, original_driver, test_dir] {
      std::error_code ec;
      std::filesystem::current_path(original_dir, ec);
      CARBON_CHECK(!ec,
                   "Could not change the current working dir to '{0}': {1}",
                   original_dir, ec.message());
      driver_ = original_driver;
      std::filesystem::remove_all(test_dir, ec);
      CARBON_CHECK(!ec, "Could not remove the test working dir '{0}': {1}",
                   test_dir, ec.message());
    });
  }

  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs_ =
      new llvm::vfs::InMemoryFileSystem;
  const InstallPaths installation_;
  RawStringOstream test_output_stream_;
  RawStringOstream test_error_stream_;

  // Some tests work directly with files in the test temporary directory.
  std::filesystem::path test_tmpdir_;

  Driver driver_;
};

TEST_F(DriverTest, BadCommandErrors) {
  EXPECT_FALSE(driver_.RunCommand({}).success);
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("error:"));

  EXPECT_FALSE(driver_.RunCommand({"foo"}).success);
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("error:"));

  EXPECT_FALSE(driver_.RunCommand({"foo --bar --baz"}).success);
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("error:"));
}

TEST_F(DriverTest, CompileCommandErrors) {
  // No input file. This error message is important so check all of it.
  EXPECT_FALSE(driver_.RunCommand({"compile"}).success);
  EXPECT_THAT(
      test_error_stream_.TakeStr(),
      StrEq("error: not all required positional arguments were provided; first "
            "missing and required positional argument: `FILE`\n"));

  // Pass non-existing file
  EXPECT_FALSE(driver_
                   .RunCommand({"compile", "--dump-mem-usage",
                                "non-existing-file.carbon"})
                   .success);
  EXPECT_THAT(
      test_error_stream_.TakeStr(),
      ContainsRegex("No such file or directory[\\n]*non-existing-file.carbon"));
  // Flush output stream, because it's ok that it's not empty here.
  test_output_stream_.TakeStr();

  // Invalid output filename. No reliably error message here.
  // TODO: Likely want a different filename on Windows.
  auto empty_file = MakeTestFile("");
  EXPECT_FALSE(driver_
                   .RunCommand({"compile", "--no-prelude-import",
                                "--output=/dev/empty", empty_file})
                   .success);
  EXPECT_THAT(test_error_stream_.TakeStr(),
              ContainsRegex("error: .*/dev/empty.*"));
}

TEST_F(DriverTest, DumpTokens) {
  auto file = MakeTestFile("Hello World");
  EXPECT_TRUE(driver_
                  .RunCommand({"compile", "--no-prelude-import", "--phase=lex",
                               "--dump-tokens", file})
                  .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  // Verify there is output without examining it.
  EXPECT_THAT(Yaml::Value::FromText(test_output_stream_.TakeStr()),
              Yaml::IsYaml(_));
}

TEST_F(DriverTest, DumpParseTree) {
  auto file = MakeTestFile("var v: () = ();");
  EXPECT_TRUE(driver_
                  .RunCommand({"compile", "--no-prelude-import",
                               "--phase=parse", "--dump-parse-tree",
                               "--parse-dump-format=yaml-postorder", file})
                  .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  // Verify there is output without examining it.
  EXPECT_THAT(Yaml::Value::FromText(test_output_stream_.TakeStr()),
              Yaml::IsYaml(_));
}

TEST_F(DriverTest, StdoutOutput) {
  // Use explicit filenames so we can look for those to validate output.
  MakeTestFile("fn Run() {}", "test.carbon");

  EXPECT_TRUE(driver_
                  .RunCommand({"compile", "--no-prelude-import", "--output=-",
                               "test.carbon"})
                  .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  // The default is textual assembly.
  EXPECT_THAT(test_output_stream_.TakeStr(), ContainsRegex("main:"));

  EXPECT_TRUE(driver_
                  .RunCommand({"compile", "--no-prelude-import", "--output=-",
                               "--force-obj-output", "test.carbon"})
                  .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  std::string output = test_output_stream_.TakeStr();
  auto result =
      llvm::object::createBinary(llvm::MemoryBufferRef(output, "test_output"));
  if (auto error = result.takeError()) {
    FAIL() << toString(std::move(error));
  }
  EXPECT_TRUE(result->get()->isObject());
}

TEST_F(DriverTest, LinkFileOutput) {
  auto scope = ScopedTempWorkingDir();

  // Use explicit filenames as the default output filename is computed from
  // this, and we can use this to validate output.
  MakeTestFile("fn Run() {}", "test.carbon");

  // Object output (the default) uses `.o`.
  // TODO: This should actually reflect the platform defaults.
  EXPECT_TRUE(
      driver_.RunCommand({"compile", "--no-prelude-import", "test.carbon"})
          .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  // Ensure we wrote an object file of some form with the correct name.
  auto result = llvm::object::createBinary("test.o");
  if (auto error = result.takeError()) {
    FAIL() << toString(std::move(error));
  }
  EXPECT_TRUE(result->getBinary()->isObject());

  // Assembly output uses `.s`.
  // TODO: This should actually reflect the platform defaults.
  EXPECT_TRUE(driver_
                  .RunCommand({"compile", "--no-prelude-import", "--asm-output",
                               "test.carbon"})
                  .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  // TODO: This may need to be tailored to other assembly formats.
  EXPECT_THAT(*Testing::ReadFile("test.s"), ContainsRegex("main:"));
}

TEST_F(DriverTest, Link) {
  auto scope = ScopedTempWorkingDir();

  // First compile a file to get a linkable object.
  MakeTestFile("fn Run() {}", "test.carbon");
  ASSERT_TRUE(
      driver_.RunCommand({"compile", "--no-prelude-import", "test.carbon"})
          .success)
      << test_error_stream_.TakeStr();

  // Now link this into a binary. Note that we suppress building runtimes on
  // demand here as no runtimes should be needed for the empty program. We also
  // pass some system library link flags through to the underlying Clang layer.
  EXPECT_TRUE(driver_
                  .RunCommand({"--no-build-runtimes", "link", "--output=test",
                               "test.o", "--", "-lc", "-lm"})
                  .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));

  // Ensure we wrote an executable file of some form with the correct name.
  // TODO: We may need to update this if we implicitly synthesize a
  // platform-specific `.exe` suffix or something similar.
  auto result = llvm::object::createBinary("test");
  if (auto error = result.takeError()) {
    FAIL() << toString(std::move(error));
  }
  // Executables are also classified as object files.
  EXPECT_TRUE(result->getBinary()->isObject());
}

TEST_F(DriverTest, LinkWithFlagLikeFiles) {
  auto scope = ScopedTempWorkingDir();

  // First compile a file to get a linkable object.
  MakeTestFile("fn Run() {}", "test.carbon");
  ASSERT_TRUE(
      driver_.RunCommand({"compile", "--no-prelude-import", "test.carbon"})
          .success)
      << test_error_stream_.TakeStr();

  // Rename it to a flag-like name.
  Filesystem::Cwd().Rename("test.o", Filesystem::Cwd(), "--test.o").Check();

  // Link this into a binary and pass flags to the Clang link invocation even
  // though we use `--` before the object file input list to handle weirdly
  // named objects.
  //
  // TODO: This works correctly in the Carbon link subcommand, but Clang itself
  // fails to pass object files to LLD in a way that supports flag-shaped object
  // file names. The last flag being `-Wl,--` tries to work around this by
  // passing a `--` to the linker before the object files. However, LLD in turn
  // appears to have bugs parsing command lines in this shape. We should get
  // these fixed and then can remove the `-Wl,--` hack and the test should
  // actually pass.
  std::string out;
  std::string err;
  EXPECT_FALSE(CallWithCapturedOutput(out, err, [&] {
                 return driver_.RunCommand({"--no-build-runtimes", "link",
                                            "--output=test", "--", "--test.o",
                                            "--", "-lc", "-lm", "-Wl,--"});
               }).success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  EXPECT_THAT(out, StrEq(""));
  // This error seems to stem from incorrectly handling `--` in the LLD command
  // line. See above; this should go away when the underlying bugs are fixed.
  EXPECT_THAT(err,
              HasSubstr("error: completed parsing all 1 configured positional "
                        "arguments, but found a subsequent `--`"));
}

TEST_F(DriverTest, ConfigJson) {
  // Use the real filesystem so that the installation digest can be read.
  auto cleanup = ScopedTempWorkingDir();
  EXPECT_TRUE(driver_.RunCommand({"config", "--json"}).success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));

  // Make sure the output parses as JSON.
  std::string output = test_output_stream_.TakeStr();
  llvm::Expected<llvm::json::Value> json_value = llvm::json::parse(output);
  if (auto error = json_value.takeError()) {
    FAIL() << "Unable to parse to JSON: " << toString(std::move(error))
           << "\nOriginal text:\n"
           << output << "\n";
  }
  llvm::json::Object* json_obj = json_value->getAsObject();
  ASSERT_THAT(json_obj, NotNull());

  // Check relevant paths in the output point to existing directories.
  std::optional<llvm::StringRef> install_root =
      json_obj->getString("INSTALL_ROOT");
  ASSERT_THAT(install_root, Ne(std::nullopt));
  EXPECT_THAT(Filesystem::Cwd().OpenDir(install_root->str()), IsSuccess(_));

  std::optional<llvm::StringRef> clang_sysroot =
      json_obj->getString("CLANG_SYSROOT");
  ASSERT_THAT(clang_sysroot, Ne(std::nullopt));
  EXPECT_THAT(Filesystem::Cwd().OpenDir(clang_sysroot->str()), IsSuccess(_));
}

TEST_F(DriverTest, BuildFileOutput) {
  auto scope = ScopedTempWorkingDir();

  MakeTestFile(R"""(
import Core library "io";

fn Run() {
  Core.PrintStr("Hello world!\n");
}
)""",
               "hello_world.carbon");

  // File should compile to a `hello_world` binary without error.
  EXPECT_TRUE(
      driver_
          .RunCommand({"--no-build-runtimes", "build", "--no-use-temp-dir",
                       "hello_world.carbon", "--", "--", "-lc"})
          .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));

  // Binary should read as valid to LLVM.
  auto result = llvm::object::createBinary("hello_world");
  if (auto error = result.takeError()) {
    FAIL() << toString(std::move(error));
  }

  // Executables are also classified as object files.
  EXPECT_TRUE(result->getBinary()->isObject());
}

TEST_F(DriverTest, TargetCpu) {
  MakeTestFile("fn Run() {}", "test.carbon");

  struct TargetCpuCase {
    llvm::StringRef target;
    llvm::StringRef cpu = "";
    llvm::StringRef tune_cpu = "";
    llvm::StringRef features = "";
    llvm::SmallVector<llvm::StringRef, 2> clang_args = {};
  };
  TargetCpuCase compile_cases[] = {
      {.target = "x86_64-unknown-linux-gnu", .cpu = "x86-64-v3"},
      {.target = "x86_64-unknown-linux-gnu",
       .cpu = "x86-64-v3",
       .tune_cpu = "znver4"},
      {.target = "x86_64-unknown-linux-gnu", .cpu = "znver4"},
      {.target = "x86_64-unknown-linux-gnu", .tune_cpu = "znver4"},
      {.target = "x86_64-unknown-linux-gnu",
       .cpu = "x86-64-v2",
       .tune_cpu = "znver4",
       .features = "+avx2,-fma,bmi2",
       .clang_args = {"--clang-arg=-march=x86-64", "--clang-arg=-mtune=generic",
                      "--clang-arg=-mfma"}},
      {.target = "aarch64-unknown-linux-gnu", .cpu = "armv9-a"},
      {.target = "aarch64-unknown-linux-gnu",
       .cpu = "armv9-a",
       .tune_cpu = "neoverse-v2"},
      {.target = "aarch64-unknown-linux-gnu", .cpu = "neoverse-v2"},
      {.target = "aarch64-unknown-linux-gnu", .tune_cpu = "neoverse-v2"},
      {.target = "aarch64-unknown-linux-gnu",
       .cpu = "armv8.2-a",
       .features = "sve,-crc",
       .clang_args = {"--clang-arg=-march=armv9-a", "--clang-arg=-mcrc"}},
  };

  for (const auto& tc : compile_cases) {
    SCOPED_TRACE(llvm::formatv("target={0}, cpu={1}, tune={2}, features={3}",
                               tc.target, tc.cpu, tc.tune_cpu, tc.features)
                     .str());
    std::string target_arg = llvm::formatv("--target={0}", tc.target).str();
    std::string cpu_arg = llvm::formatv("--target-cpu={0}", tc.cpu).str();
    std::string tune_arg =
        llvm::formatv("--target-cpu-tune={0}", tc.tune_cpu).str();
    std::string features_arg =
        llvm::formatv("--target-cpu-features={0}", tc.features).str();
    llvm::SmallVector<llvm::StringRef> args = {"compile", "--no-prelude-import",
                                               "--output=-", target_arg};
    if (!tc.cpu.empty()) {
      args.push_back(cpu_arg);
    }
    if (!tc.tune_cpu.empty()) {
      args.push_back(tune_arg);
    }
    if (!tc.features.empty()) {
      args.push_back(features_arg);
    }
    args.append(tc.clang_args.begin(), tc.clang_args.end());
    args.push_back("test.carbon");
    EXPECT_TRUE(driver_.RunCommand(args).success)
        << test_error_stream_.TakeStr();
    EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
    EXPECT_THAT(test_output_stream_.TakeStr(), ContainsRegex("main:"));
  }

  // Test `native` on the host target.
  std::string host_target = llvm::sys::getDefaultTargetTriple();
  llvm::Triple host_triple(host_target);
  if (host_triple.isX86() || host_triple.isAArch64()) {
    EXPECT_TRUE(driver_
                    .RunCommand({"compile", "--no-prelude-import", "--output=-",
                                 "--target-cpu=native",
                                 "--target-cpu-tune=native", "test.carbon"})
                    .success)
        << test_error_stream_.TakeStr();
    EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
    EXPECT_THAT(test_output_stream_.TakeStr(), ContainsRegex("main:"));
  }

  // Invalid target CPU, tune CPU, or target features should fail cleanly with a
  // diagnostic across `compile`, `link`, and `build-runtimes`.
  for (llvm::StringRef target :
       {"x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"}) {
    std::string target_arg = llvm::formatv("--target={0}", target).str();
    EXPECT_FALSE(
        driver_
            .RunCommand({"compile", "--no-prelude-import", "--output=-",
                         target_arg, "--target-cpu=not-a-valid-cpu",
                         "test.carbon"})
            .success);
    EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("not-a-valid-cpu"));
    test_output_stream_.TakeStr();

    EXPECT_FALSE(
        driver_
            .RunCommand({"compile", "--no-prelude-import", "--output=-",
                         target_arg, "--target-cpu-tune=not-a-valid-tune-cpu",
                         "test.carbon"})
            .success);
    EXPECT_THAT(test_error_stream_.TakeStr(),
                HasSubstr("not-a-valid-tune-cpu"));
    test_output_stream_.TakeStr();

    EXPECT_FALSE(
        driver_
            .RunCommand({"compile", "--no-prelude-import", "--output=-",
                         target_arg, "--target-cpu-features=+not-a-feature",
                         "test.carbon"})
            .success);
    EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("+not-a-feature"));
    test_output_stream_.TakeStr();

    EXPECT_FALSE(
        driver_
            .RunCommand({"--no-build-runtimes", "link", "--output=out",
                         target_arg, "--target-cpu=not-a-valid-cpu", "test.o"})
            .success);
    EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("not-a-valid-cpu"));
    test_output_stream_.TakeStr();

    EXPECT_FALSE(
        driver_
            .RunCommand({"--no-build-runtimes", "link", "--output=out",
                         target_arg, "--target-cpu-features=+not-a-feature",
                         "test.o"})
            .success);
    EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("+not-a-feature"));
    test_output_stream_.TakeStr();

    EXPECT_FALSE(driver_
                     .RunCommand({"build-runtimes", target_arg,
                                  "--target-cpu=not-a-valid-cpu"})
                     .success);
    EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("not-a-valid-cpu"));
    test_output_stream_.TakeStr();

    EXPECT_FALSE(driver_
                     .RunCommand({"build-runtimes", target_arg,
                                  "--target-cpu-features=+not-a-feature"})
                     .success);
    EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("+not-a-feature"));
    test_output_stream_.TakeStr();
  }

  // Test RISC-V `--target-cpu`, `--target-cpu-tune`, and
  // `--target-cpu-features` mapping and Clang target validation via `config`
  // (since the RISC-V LLVM codegen backend is not linked by default).
  auto cleanup = ScopedTempWorkingDir();
  MakeTestFile("fn Run() {}", "host_test.carbon");
  if (host_triple.isX86() || host_triple.isAArch64()) {
    llvm::StringRef host_cpu = host_triple.isX86() ? "x86-64-v3" : "armv8.2-a";
    llvm::StringRef host_tune = host_triple.isX86() ? "znver4" : "neoverse-v2";
    llvm::StringRef host_feature = host_triple.isX86() ? "+bmi2" : "+lse";
    std::string cpu_arg = llvm::formatv("--target-cpu={0}", host_cpu).str();
    std::string tune_arg =
        llvm::formatv("--target-cpu-tune={0}", host_tune).str();
    std::string feature_arg =
        llvm::formatv("--target-cpu-features={0}", host_feature).str();

    ASSERT_TRUE(driver_
                    .RunCommand({"compile", "--no-prelude-import", cpu_arg,
                                 tune_arg, feature_arg, "host_test.carbon"})
                    .success)
        << test_error_stream_.TakeStr();
    EXPECT_TRUE(driver_
                    .RunCommand({"--no-build-runtimes", "link",
                                 "--output=host_test_linked", cpu_arg, tune_arg,
                                 feature_arg, "host_test.o", "--", "-lc"})
                    .success)
        << test_error_stream_.TakeStr();
    EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));

    // Also verify `build` with `--target-cpu` and a compile-only `--clang-arg`
    // (such as `-fsanitize=address`, which would fail linking if forwarded to
    // the link step without sanitizer runtimes).
    EXPECT_TRUE(
        driver_
            .RunCommand({"--no-build-runtimes", "build", "--no-use-temp-dir",
                         "--no-prelude-import",
                         "--clang-arg=-fsanitize=address", cpu_arg, tune_arg,
                         feature_arg, "host_test.carbon", "--", "--", "-lc"})
            .success)
        << test_error_stream_.TakeStr();
    EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  }

  for (llvm::StringRef riscv_cpu : {"rv64gc", "rva22u64", "spacemit-x60"}) {
    SCOPED_TRACE(riscv_cpu);
    std::string cpu_arg = llvm::formatv("--target-cpu={0}", riscv_cpu).str();
    EXPECT_TRUE(driver_
                    .RunCommand({"config", "--json",
                                 "--target=riscv64-unknown-linux-gnu", cpu_arg,
                                 "--target-cpu-tune=spacemit-x60",
                                 "--target-cpu-features=+zba,-m"})
                    .success)
        << test_error_stream_.TakeStr();
    EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
    test_output_stream_.TakeStr();
  }
  EXPECT_FALSE(
      driver_
          .RunCommand({"config", "--json", "--target=riscv64-unknown-linux-gnu",
                       "--target-cpu=not-a-valid-cpu"})
          .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("not-a-valid-cpu"));
  test_output_stream_.TakeStr();

  EXPECT_FALSE(
      driver_
          .RunCommand({"config", "--json", "--target=riscv64-unknown-linux-gnu",
                       "--target-cpu-tune=not-a-valid-tune-cpu"})
          .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("not-a-valid-tune-cpu"));
  test_output_stream_.TakeStr();

  EXPECT_FALSE(
      driver_
          .RunCommand({"config", "--json", "--target=riscv64-unknown-linux-gnu",
                       "--target-cpu-features=+f,+zfinx"})
          .success);
  EXPECT_THAT(test_error_stream_.TakeStr(),
              HasSubstr("invalid feature combination"));
  test_output_stream_.TakeStr();
}

}  // namespace
}  // namespace Carbon
