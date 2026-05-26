// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/clang_runner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <utility>

#include "common/error_test_helpers.h"
#include "common/ostream.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Host.h"
#include "testing/base/capture_std_streams.h"
#include "testing/base/file_helpers.h"
#include "testing/base/global_exe_path.h"
#include "toolchain/base/install_paths.h"
#include "toolchain/driver/runtimes_cache.h"

namespace Carbon {
namespace {

using ::testing::HasSubstr;
using Testing::IsSuccess;
using ::testing::StrEq;

// NOLINTNEXTLINE(modernize-use-trailing-return-type): Macro based function.
MATCHER_P(TextSymbolNamed, name_matcher, "") {
  llvm::Expected<llvm::StringRef> name = arg.getName();
  if (auto error = name.takeError()) {
    *result_listener << "with an error instead of a name: " << error;
    return false;
  }
  if (!testing::ExplainMatchResult(name_matcher, *name, result_listener)) {
    return false;
  }
  // We have to dig out the section to determine if this was a text symbol.
  auto expected_section_it = arg.getSection();
  if (auto error = expected_section_it.takeError()) {
    *result_listener << "without a section: " << error;
    return false;
  }
  llvm::object::SectionRef section = **expected_section_it;
  if (!section.isText()) {
    *result_listener << "in the non-text section: " << *section.getName();
    return false;
  }
  return true;
}

class ClangRunnerTest : public ::testing::Test {
 public:
  InstallPaths install_paths_ =
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath());
  Runtimes::Cache runtimes_cache_ =
      *Runtimes::Cache::MakeSystem(install_paths_);
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs_ =
      llvm::vfs::getRealFileSystem();
};

TEST_F(ClangRunnerTest, Version) {
  RawStringOstream test_os;
  ClangRunner runner(&install_paths_, vfs_, &test_os);

  std::string out;
  std::string err;
  EXPECT_THAT(
      Testing::CallWithCapturedOutput(
          out, err, [&] { return runner.RunWithNoRuntimes({"--version"}); }),
      IsSuccess(true));
  // The arguments to Clang should be part of the verbose log.
  EXPECT_THAT(test_os.TakeStr(), HasSubstr("--version"));

  // No need to flush stderr, just check its contents.
  EXPECT_THAT(err, StrEq(""));

  // Flush and get the captured stdout to test that this command worked.
  // We don't care about any particular version, just that it is printed.
  EXPECT_THAT(out, HasSubstr("clang version"));
  // The target should match the LLVM default.
  EXPECT_THAT(out, HasSubstr((llvm::Twine("Target: ") +
                              llvm::sys::getDefaultTargetTriple())
                                 .str()));
  // Clang's install should be our private LLVM install bin directory.
  EXPECT_THAT(out, HasSubstr(std::string("InstalledDir: ") +
                             install_paths_.llvm_install_bin().native()));
}

TEST_F(ClangRunnerTest, DashC) {
  std::filesystem::path test_file =
      *Testing::WriteTestFile("test.cpp", "int test() { return 0; }");
  std::filesystem::path test_output = *Testing::WriteTestFile("test.o", "");

  RawStringOstream verbose_out;
  ClangRunner runner(&install_paths_, vfs_, &verbose_out);
  std::string out;
  std::string err;
  EXPECT_THAT(Testing::CallWithCapturedOutput(
                  out, err,
                  [&] {
                    return runner.RunWithNoRuntimes(
                        {"-c", test_file.string(), "-o", test_output.string()});
                  }),
              IsSuccess(true))
      << "Verbose output from runner:\n"
      << verbose_out.TakeStr() << "\n";
  verbose_out.clear();

  // No output should be produced.
  EXPECT_THAT(out, StrEq(""));
  EXPECT_THAT(err, StrEq(""));
}

TEST_F(ClangRunnerTest, BuitinHeaders) {
  std::filesystem::path test_file = *Testing::WriteTestFile("test.c", R"cpp(
#include <stdalign.h>

#ifndef alignas
#error included the wrong header
#endif
  )cpp");
  std::filesystem::path test_output = *Testing::WriteTestFile("test.o", "");

  RawStringOstream verbose_out;
  ClangRunner runner(&install_paths_, vfs_, &verbose_out);
  std::string out;
  std::string err;
  EXPECT_THAT(Testing::CallWithCapturedOutput(
                  out, err,
                  [&] {
                    return runner.RunWithNoRuntimes(
                        {"-c", test_file.string(), "-o", test_output.string()});
                  }),
              IsSuccess(true))
      << "Verbose output from runner:\n"
      << verbose_out.TakeStr() << "\n";
  verbose_out.clear();

  // No output should be produced.
  EXPECT_THAT(out, StrEq(""));
  EXPECT_THAT(err, StrEq(""));
}

TEST_F(ClangRunnerTest, CompileMultipleFiles) {
  // Memory leaks and other errors from running Clang can at times only manifest
  // with repeated compilations. Use a lambda to just do a series of compiles.
  auto compile = [&](llvm::StringRef filename, llvm::StringRef source) {
    std::string output_file = std::string(filename.split('.').first) + ".o";
    std::filesystem::path file = *Testing::WriteTestFile(filename, source);
    std::filesystem::path output = *Testing::WriteTestFile(output_file, "");

    RawStringOstream verbose_out;
    ClangRunner runner(&install_paths_, vfs_, &verbose_out);
    std::string out;
    std::string err;
    EXPECT_THAT(Testing::CallWithCapturedOutput(
                    out, err,
                    [&] {
                      return runner.RunWithNoRuntimes(
                          {"-c", file.string(), "-o", output.string()});
                    }),
                IsSuccess(true))
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

// It's hard to write a portable and reliable unittest for all the layers of the
// Clang driver because they work hard to interact with the underlying
// filesystem and operating system. For now, we just check that a link command
// is echoed back with plausible contents.
//
// TODO: We should eventually strive to have a more complete setup that lets us
// test more complete Clang functionality here.
TEST_F(ClangRunnerTest, LinkCommandEcho) {
  // Just create some empty files to use in a synthetic link command below.
  std::filesystem::path foo_file = *Testing::WriteTestFile("foo.o", "");
  std::filesystem::path bar_file = *Testing::WriteTestFile("bar.o", "");

  RawStringOstream verbose_out;
  ClangRunner runner(&install_paths_, vfs_, &verbose_out);
  std::string out;
  std::string err;
  EXPECT_THAT(
      Testing::CallWithCapturedOutput(
          out, err,
          [&] {
            // Note that we use the target independent run command here because
            // we're just getting the echo-ed output back. For this to actually
            // link, we'd need to have the target-dependent resources, but those
            // are expensive to build so we only want to test them once (above).
            return runner.RunWithNoRuntimes(
                {"-###", "-o", "binary", foo_file.string(), bar_file.string()});
          }),
      IsSuccess(true))
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

TEST_F(ClangRunnerTest, ParamsFile) {
  // Use an overlay file system to ensure the params file expansion goes through
  // the VFS.
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> overlay_fs(
      new llvm::vfs::OverlayFileSystem(vfs_));
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> in_memory_fs(
      new llvm::vfs::InMemoryFileSystem);
  overlay_fs->pushOverlay(in_memory_fs);

  std::filesystem::path params_path = "/params";
  in_memory_fs->addFile(params_path.native(), 0,
                        llvm::MemoryBuffer::getMemBuffer(R"(
--version
)"));

  RawStringOstream verbose_out;
  ClangRunner runner(&install_paths_, overlay_fs, &verbose_out);

  std::string out;
  std::string err;
  EXPECT_THAT(
      Testing::CallWithCapturedOutput(out, err,
                                      [&] {
                                        return runner.RunWithNoRuntimes(
                                            {"@" + params_path.native()});
                                      }),
      IsSuccess(true))
      << "Verbose output:\n"
      << verbose_out.TakeStr();
  verbose_out.clear();

  // Check that the version is printed, as if we directly passed `--version`.
  EXPECT_THAT(err, StrEq(""));
  EXPECT_THAT(out, HasSubstr("clang version"));
}

TEST_F(ClangRunnerTest, Assemble) {
  std::filesystem::path test_file = *Testing::WriteTestFile("test.s", R"asm(
	.file	"test.s"
	.section	.text,"ax",@progbits,unique,1
	.globl	_Z4testv
	.p2align	2
	.type	_Z4testv,@function
_Z4testv:
	.cfi_startproc
	mov	w0, wzr
	ret
.Lfunc_end0:
	.size	_Z4testv, .Lfunc_end0-_Z4testv
	.cfi_endproc
	.section	".note.GNU-stack","",@progbits
	.addrsig)asm");

  std::filesystem::path test_output = *Testing::WriteTestFile("test.o", "");

  RawStringOstream verbose_out;
  ClangRunner runner(&install_paths_, vfs_, &verbose_out);
  std::string out;
  std::string err;
  EXPECT_THAT(
      Testing::CallWithCapturedOutput(
          out, err,
          [&] {
            return runner.RunWithNoRuntimes(
                {"-c", test_file.string(), "--target=aarch64-unknown-linux-gnu",
                 "-o", test_output.string()});
          }),
      IsSuccess(true))
      << "Verbose output from runner:\n"
      << verbose_out.TakeStr() << "\n";
  verbose_out.clear();

  // No output should be produced.
  EXPECT_THAT(out, StrEq(""));
  EXPECT_THAT(err, StrEq(""));
}

}  // namespace
}  // namespace Carbon
