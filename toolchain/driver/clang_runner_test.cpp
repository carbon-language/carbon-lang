// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/clang_runner.h"

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

namespace Carbon {
namespace {

using ::testing::HasSubstr;
using ::testing::StrEq;

TEST(ClangRunnerTest, Version) {
  RawStringOstream test_os;
  const auto install_paths =
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath());
  std::string target = llvm::sys::getDefaultTargetTriple();
  auto vfs = llvm::vfs::getRealFileSystem();
  ClangRunner runner(&install_paths, target, vfs, &test_os);

  std::string out;
  std::string err;
  EXPECT_TRUE(Testing::CallWithCapturedOutput(
      out, err, [&] { return runner.Run({"--version"}); }));
  // The arguments to Clang should be part of the verbose log.
  EXPECT_THAT(test_os.TakeStr(), HasSubstr("--version"));

  // No need to flush stderr, just check its contents.
  EXPECT_THAT(err, StrEq(""));

  // Flush and get the captured stdout to test that this command worked.
  // We don't care about any particular version, just that it is printed.
  EXPECT_THAT(out, HasSubstr("clang version"));
  // The target should match what we provided.
  EXPECT_THAT(out, HasSubstr((llvm::Twine("Target: ") + target).str()));
  // Clang's install should be our private LLVM install bin directory.
  EXPECT_THAT(out, HasSubstr(std::string("InstalledDir: ") +
                             install_paths.llvm_install_bin()));
}

// It's hard to write a portable and reliable unittest for all the layers of the
// Clang driver because they work hard to interact with the underlying
// filesystem and operating system. For now, we just check that a link command
// is echoed back with plausible contents.
//
// TODO: We should eventually strive to have a more complete setup that lets us
// test more complete Clang functionality here.
TEST(ClangRunnerTest, LinkCommandEcho) {
  // Just create some empty files to use in a synthetic link command below.
  std::filesystem::path foo_file = *Testing::WriteTestFile("foo.o", "");
  std::filesystem::path bar_file = *Testing::WriteTestFile("bar.o", "");

  const auto install_paths =
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath());
  RawStringOstream verbose_out;
  std::string target = llvm::sys::getDefaultTargetTriple();
  auto vfs = llvm::vfs::getRealFileSystem();
  ClangRunner runner(&install_paths, target, vfs, &verbose_out);
  std::string out;
  std::string err;
  EXPECT_TRUE(Testing::CallWithCapturedOutput(
      out, err,
      [&] {
        return runner.Run(
            {"-###", "-o", "binary", foo_file.string(), bar_file.string()});
      }))
      << "Verbose output from runner:\n"
      << verbose_out.TakeStr() << "\n";
  verbose_out.clear();

  // Because we use `-###' above, we should just see the command that the Clang
  // driver would have run in a subprocess. This will be very architecture
  // dependent and have lots of variety, but we expect to see both file strings
  // in it the command at least.
  EXPECT_THAT(err, HasSubstr(foo_file.string())) << err;
  EXPECT_THAT(err, HasSubstr(bar_file.string())) << err;

  // And no non-stderr output should be produced.
  EXPECT_THAT(out, StrEq(""));
}

TEST(ClangRunnerTest, DashC) {
  std::filesystem::path test_file =
      *Testing::WriteTestFile("test.cpp", "int test() { return 0; }");
  std::filesystem::path test_output = *Testing::WriteTestFile("test.o", "");

  const auto install_paths =
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath());
  RawStringOstream verbose_out;
  std::string target = llvm::sys::getDefaultTargetTriple();
  auto vfs = llvm::vfs::getRealFileSystem();
  ClangRunner runner(&install_paths, target, vfs, &verbose_out);
  std::string out;
  std::string err;
  EXPECT_TRUE(Testing::CallWithCapturedOutput(
      out, err,
      [&] {
        return runner.Run(
            {"-c", test_file.string(), "-o", test_output.string()});
      }))
      << "Verbose output from runner:\n"
      << verbose_out.TakeStr() << "\n";
  verbose_out.clear();

  // No output should be produced.
  EXPECT_THAT(out, StrEq(""));
  EXPECT_THAT(err, StrEq(""));
}

TEST(ClangRunnerTest, BuitinHeaders) {
  std::filesystem::path test_file = *Testing::WriteTestFile("test.c", R"cpp(
#include <stdalign.h>

#ifndef alignas
#error included the wrong header
#endif
  )cpp");
  std::filesystem::path test_output = *Testing::WriteTestFile("test.o", "");

  const auto install_paths =
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath());
  RawStringOstream verbose_out;
  std::string target = llvm::sys::getDefaultTargetTriple();
  auto vfs = llvm::vfs::getRealFileSystem();
  ClangRunner runner(&install_paths, target, vfs, &verbose_out);
  std::string out;
  std::string err;
  EXPECT_TRUE(Testing::CallWithCapturedOutput(
      out, err,
      [&] {
        return runner.Run(
            {"-c", test_file.string(), "-o", test_output.string()});
      }))
      << "Verbose output from runner:\n"
      << verbose_out.TakeStr() << "\n";
  verbose_out.clear();

  // No output should be produced.
  EXPECT_THAT(out, StrEq(""));
  EXPECT_THAT(err, StrEq(""));
}

TEST(ClangRunnerTest, CompileMultipleFiles) {
  const auto install_paths =
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath());

  // Memory leaks and other errors from running Clang can at times only manifest
  // with repeated compilations. Use a lambda to just do a series of compiles.
  auto compile = [&](llvm::StringRef filename, llvm::StringRef source) {
    std::string output_file = std::string(filename.split('.').first) + ".o";
    std::filesystem::path file = *Testing::WriteTestFile(filename, source);
    std::filesystem::path output = *Testing::WriteTestFile(output_file, "");

    RawStringOstream verbose_out;
    std::string target = llvm::sys::getDefaultTargetTriple();
    auto vfs = llvm::vfs::getRealFileSystem();
    ClangRunner runner(&install_paths, target, vfs, &verbose_out);
    std::string out;
    std::string err;
    EXPECT_TRUE(Testing::CallWithCapturedOutput(
        out, err,
        [&] {
          return runner.Run({"-c", file.string(), "-o", output.string()});
        }))
        << "Verbose output from runner:\n"
        << verbose_out.TakeStr() << "\n";
    verbose_out.clear();

    EXPECT_THAT(out, StrEq(""));
    EXPECT_THAT(err, StrEq(""));
  };

  compile("test1.cpp", "int test1() { return 0; }");
  compile("test2.cpp", "int test2() { return 0; }");
  compile("test3.cpp", "int test3() { return 0; }");
}

}  // namespace
}  // namespace Carbon
