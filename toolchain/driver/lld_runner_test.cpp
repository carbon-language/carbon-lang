// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/lld_runner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <utility>

#include "common/check.h"
#include "common/ostream.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Host.h"
#include "testing/base/capture_std_streams.h"
#include "testing/base/file_helpers.h"
#include "testing/base/global_exe_path.h"
#include "toolchain/driver/clang_runner.h"

namespace Carbon {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::StrEq;

TEST(LldRunnerTest, Version) {
  RawStringOstream test_os;
  const auto install_paths =
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath());
  LldRunner runner(&install_paths, &test_os);

  std::string out;
  std::string err;
  EXPECT_TRUE(Testing::CallWithCapturedOutput(
      out, err, [&] { return runner.ElfLink({"--version"}); }));

  // The arguments to LLD should be part of the verbose log.
  EXPECT_THAT(test_os.TakeStr(), HasSubstr("--version"));

  // Nothing should print to stderr here.
  EXPECT_THAT(err, StrEq(""));

  // We don't care about any particular version, just that it is printed.
  EXPECT_THAT(out, HasSubstr("LLD"));
  // Check that it was in fact the GNU linker.
  EXPECT_THAT(out, HasSubstr("compatible with GNU linkers"));

  // Try the Darwin linker.
  EXPECT_TRUE(Testing::CallWithCapturedOutput(
      out, err, [&] { return runner.MachOLink({"--version"}); }));

  // Again, the arguments to LLD should be part of the verbose log.
  EXPECT_THAT(test_os.TakeStr(), HasSubstr("--version"));

  // Nothing should print to stderr.
  EXPECT_THAT(err, StrEq(""));

  // We don't care about any particular version.
  EXPECT_THAT(out, HasSubstr("LLD"));
  // The Darwin link code path doesn't print anything distinct, so instead check
  // that the GNU output isn't repeated.
  EXPECT_THAT(out, Not(HasSubstr("GNU")));
}

static auto CompileTwoSources(const InstallPaths& install_paths,
                              llvm::StringRef target)
    -> std::pair<std::filesystem::path, std::filesystem::path> {
  std::filesystem::path test_a_file =
      *Testing::WriteTestFile("test_a.cpp", "int test_a() { return 0; }");
  std::filesystem::path test_b_file = *Testing::WriteTestFile(
      "test_b.cpp", "int test_a();\nint main() { return test_a(); }");
  std::filesystem::path test_a_output = *Testing::WriteTestFile("test_a.o", "");
  std::filesystem::path test_b_output = *Testing::WriteTestFile("test_b.o", "");

  // First compile the two source files to `.o` files with Clang.
  RawStringOstream verbose_out;
  auto vfs = llvm::vfs::getRealFileSystem();
  ClangRunner clang(&install_paths, target, vfs, &verbose_out);
  std::string target_arg = llvm::formatv("--target={0}", target).str();
  std::string out;
  std::string err;
  CARBON_CHECK(
      Testing::CallWithCapturedOutput(
          out, err,
          [&] {
            return clang.Run({target_arg, "-fPIE", "-c", test_a_file.string(),
                              "-o", test_a_output.string()});
          }),
      "Verbose output from runner:\n{0}\nStderr:\n{1}\n", verbose_out.TakeStr(),
      err);
  verbose_out.clear();

  CARBON_CHECK(
      Testing::CallWithCapturedOutput(
          out, err,
          [&] {
            return clang.Run({target_arg, "-fPIE", "-c", test_b_file.string(),
                              "-o", test_b_output.string()});
          }),
      "Verbose output from runner:\n{0}\nStderr:\n{1}\n", verbose_out.TakeStr(),
      err);
  verbose_out.clear();

  return {test_a_output, test_b_output};
}

TEST(LldRunnerTest, ElfLinkTest) {
  const auto install_paths =
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath());

  std::filesystem::path test_a_output;
  std::filesystem::path test_b_output;
  std::tie(test_a_output, test_b_output) =
      CompileTwoSources(install_paths, "aarch64-unknown-linux");

  std::filesystem::path test_output = *Testing::WriteTestFile("test.o", "");

  RawStringOstream verbose_out;
  std::string out;
  std::string err;

  LldRunner lld(&install_paths, &verbose_out);

  // Link the two object files together.
  //
  // TODO: Currently, this uses a relocatable link, but it would be better to do
  // a full link to an executable. For that to work, we need at least the
  // C-runtime built artifacts available in the toolchain. We should revisit
  // this once we have those in place. This also prevents us from testing a
  // failed link easily.
  EXPECT_TRUE(Testing::CallWithCapturedOutput(
      out, err,
      [&] {
        return lld.ElfLink({"-m", "aarch64linux", "--relocatable", "-o",
                            test_output.string(), test_a_output.string(),
                            test_b_output.string()});
      }))
      << "Verbose output from runner:\n"
      << verbose_out.TakeStr() << "\n";
  verbose_out.clear();

  // No output should be produced.
  EXPECT_THAT(out, StrEq(""));
  EXPECT_THAT(err, StrEq(""));
}

TEST(LldRunnerTest, MachOLinkTest) {
  const auto install_paths =
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath());

  std::filesystem::path test_a_output;
  std::filesystem::path test_b_output;
  std::tie(test_a_output, test_b_output) =
      CompileTwoSources(install_paths, "arm64-unknown-macosx10.4.0");

  std::filesystem::path test_output = *Testing::WriteTestFile("test.o", "");

  RawStringOstream verbose_out;
  std::string out;
  std::string err;

  // Link the two object files together.
  //
  // This is a somewhat arbitrary command line, and is missing the C-runtimes,
  // but seems to succeed currently. The goal isn't to test any *particular*
  // link, but just than an actual link occurs successfully.
  LldRunner lld(&install_paths, &verbose_out);
  EXPECT_TRUE(Testing::CallWithCapturedOutput(
      out, err,
      [&] {
        return lld.MachOLink({"-arch", "arm64", "-platform_version", "macos",
                              "10.4.0", "10.4.0", "-o", test_output.string(),
                              test_a_output.string(), test_b_output.string()});
      }))
      << "Verbose output from runner:\n"
      << verbose_out.TakeStr() << "\n";
  verbose_out.clear();

  // No output should be produced.
  EXPECT_THAT(out, StrEq(""));
  EXPECT_THAT(err, StrEq(""));

  // Re-do the link, but with only one of the inputs. This should fail due to an
  // unresolved symbol.
  EXPECT_FALSE(Testing::CallWithCapturedOutput(
      out, err,
      [&] {
        return lld.MachOLink({"-arch", "arm64", "-platform_version", "macos",
                              "10.4.0", "10.4.0", "-o", test_output.string(),
                              test_b_output.string()});
      }))
      << "Verbose output from runner:\n"
      << verbose_out.TakeStr() << "\n";
  verbose_out.clear();

  // The missing symbol should be diagnosed on `stderr`.
  EXPECT_THAT(out, StrEq(""));
  EXPECT_THAT(err, HasSubstr("undefined symbol: __Z6test_av"));
}

}  // namespace
}  // namespace Carbon
