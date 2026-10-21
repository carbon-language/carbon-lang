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
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/driver/codegen_options.h"
#include "toolchain/driver/runtimes_cache.h"

namespace Carbon {
namespace {

using ::testing::_;
using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using Testing::IsError;
using Testing::IsSuccess;
using ::testing::Not;
using ::testing::StrEq;

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

TEST_F(ClangRunnerTest, ComputeRuntimesFeatures) {
  ClangRunner runner(&install_paths_, vfs_);
  auto compute = [&](const CodegenOptions& options) {
    return runner.ComputeRuntimesFeatures(options, Diagnostics::NullConsumer());
  };

  // x86_64: default, microarchitecture level, specific CPU, and tune CPU.
  auto x86_default = compute({.target = "x86_64-unknown-linux-gnu"});
  ASSERT_THAT(x86_default, IsSuccess(_));
  EXPECT_THAT(x86_default->cpu, StrEq("x86-64"));
  EXPECT_THAT(x86_default->tune_cpu, StrEq("generic"));

  auto x86_v3 = compute(
      {.target = "x86_64-unknown-linux-gnu", .target_cpu = "x86-64-v3"});
  ASSERT_THAT(x86_v3, IsSuccess(_));
  EXPECT_THAT(x86_v3->cpu, StrEq("x86-64-v3"));
  EXPECT_THAT(x86_v3->tune_cpu, IsEmpty());
  EXPECT_THAT(x86_v3->target_features, Contains("+avx2"));
  EXPECT_THAT(x86_v3->target_features, Contains("+bmi2"));

  auto x86_v3_tuned = compute({.target = "x86_64-unknown-linux-gnu",
                               .target_cpu = "x86-64-v3",
                               .target_cpu_tune = "znver4"});
  ASSERT_THAT(x86_v3_tuned, IsSuccess(_));
  EXPECT_THAT(x86_v3_tuned->cpu, StrEq("x86-64-v3"));
  EXPECT_THAT(x86_v3_tuned->tune_cpu, StrEq("znver4"));

  auto x86_znver4 =
      compute({.target = "x86_64-unknown-linux-gnu", .target_cpu = "znver4"});
  ASSERT_THAT(x86_znver4, IsSuccess(_));
  EXPECT_THAT(x86_znver4->cpu, StrEq("znver4"));
  EXPECT_THAT(x86_znver4->tune_cpu, IsEmpty());
  EXPECT_THAT(x86_znver4->target_features, Contains("+avx512f"));

  // AArch64: architecture string (-march), CPU name (-mcpu), and tune (-mtune).
  auto aarch64_v9 = compute({.target = "aarch64-unknown-linux-gnu",
                             .target_cpu = "armv9-a",
                             .target_cpu_tune = "neoverse-v2"});
  ASSERT_THAT(aarch64_v9, IsSuccess(_));
  EXPECT_THAT(aarch64_v9->tune_cpu, StrEq("neoverse-v2"));
  EXPECT_THAT(aarch64_v9->target_features, Contains("+v9a"));
  EXPECT_THAT(aarch64_v9->target_features, Contains("+sve2"));

  auto aarch64_neoverse = compute(
      {.target = "aarch64-unknown-linux-gnu", .target_cpu = "neoverse-v2"});
  ASSERT_THAT(aarch64_neoverse, IsSuccess(_));
  EXPECT_THAT(aarch64_neoverse->cpu, StrEq("neoverse-v2"));
  EXPECT_THAT(aarch64_neoverse->tune_cpu, IsEmpty());
  EXPECT_THAT(aarch64_neoverse->target_features, Contains("+v9a"));

  // RISC-V: ISA string (-march), profile (-march), CPU name (-mcpu), and tune
  // (-mtune).
  auto riscv_gc = compute({.target = "riscv64-unknown-linux-gnu",
                           .target_cpu = "rv64gc",
                           .target_cpu_tune = "spacemit-x60"});
  ASSERT_THAT(riscv_gc, IsSuccess(_));
  EXPECT_THAT(riscv_gc->tune_cpu, StrEq("spacemit-x60"));
  EXPECT_THAT(riscv_gc->target_features, Contains("+m"));
  EXPECT_THAT(riscv_gc->target_features, Contains("+a"));
  EXPECT_THAT(riscv_gc->target_features, Contains("+f"));
  EXPECT_THAT(riscv_gc->target_features, Contains("+d"));
  EXPECT_THAT(riscv_gc->target_features, Contains("+c"));

  auto riscv_profile = compute(
      {.target = "riscv64-unknown-linux-gnu", .target_cpu = "rva22u64"});
  ASSERT_THAT(riscv_profile, IsSuccess(_));
  EXPECT_THAT(riscv_profile->target_features, Contains("+zba"));
  EXPECT_THAT(riscv_profile->target_features, Contains("+zbb"));
  EXPECT_THAT(riscv_profile->target_features, Contains("+zbs"));

  auto riscv_cpu = compute(
      {.target = "riscv64-unknown-linux-gnu", .target_cpu = "spacemit-x60"});
  ASSERT_THAT(riscv_cpu, IsSuccess(_));
  EXPECT_THAT(riscv_cpu->cpu, StrEq("spacemit-x60"));
  EXPECT_THAT(riscv_cpu->tune_cpu, IsEmpty());
  EXPECT_THAT(riscv_cpu->target_features, Contains("+v"));

  // Host `native` resolution should canonicalize to a concrete CPU name for
  // both target CPU and tune CPU.
  std::string host_target = llvm::sys::getDefaultTargetTriple();
  llvm::Triple host_triple(host_target);
  if (host_triple.isX86() || host_triple.isAArch64()) {
    std::string host_cpu = llvm::sys::getHostCPUName().str();
    auto native_features = compute({.target = host_target,
                                    .target_cpu = "native",
                                    .target_cpu_tune = "native"});
    ASSERT_THAT(native_features, IsSuccess(_));
    EXPECT_THAT(native_features->cpu, StrEq(host_cpu));
    EXPECT_THAT(native_features->tune_cpu, StrEq(host_cpu));
    EXPECT_THAT(native_features->target_features, Not(IsEmpty()));
  }

  // `--target-cpu-features` across x86_64, AArch64, and RISC-V, including `+`,
  // `-`, and bare feature names.
  {
    auto features = compute({.target = "x86_64-unknown-linux-gnu",
                             .target_cpu = "x86-64-v3",
                             .target_cpu_features = "-avx2,avx512f"});
    ASSERT_THAT(features, IsSuccess(_));
    EXPECT_THAT(features->cpu, StrEq("x86-64-v3"));
    // Enabling `avx512f` after `-avx2` works in order and re-enables `avx2` as
    // an implied prerequisite.
    EXPECT_THAT(features->target_features, Contains("+avx512f"));
    EXPECT_THAT(features->target_features, Contains("+avx2"));
  }
  {
    auto features = compute({.target = "x86_64-unknown-linux-gnu",
                             .target_cpu = "x86-64-v3",
                             .target_cpu_features = "avx512f,-avx512f,-fma"});
    ASSERT_THAT(features, IsSuccess(_));
    EXPECT_THAT(features->target_features, Not(Contains("+avx512f")));
    EXPECT_THAT(features->target_features, Not(Contains("+fma")));
    EXPECT_THAT(features->target_features, Contains("+avx2"));
  }
  {
    auto features = compute({.target = "aarch64-unknown-linux-gnu",
                             .target_cpu = "armv8-a",
                             .target_cpu_features = "sve,+lse,-crc"});
    ASSERT_THAT(features, IsSuccess(_));
    EXPECT_THAT(features->target_features, Contains("+sve"));
    EXPECT_THAT(features->target_features, Contains("+lse"));
    EXPECT_THAT(features->target_features, Not(Contains("+crc")));
  }
  {
    auto features = compute({.target = "riscv64-unknown-linux-gnu",
                             .target_cpu = "rv64gc",
                             .target_cpu_features = "+zba,zbb,-m"});
    ASSERT_THAT(features, IsSuccess(_));
    EXPECT_THAT(features->target_features, Contains("+zba"));
    EXPECT_THAT(features->target_features, Contains("+zbb"));
    EXPECT_THAT(features->target_features, Not(Contains("+m")));
  }

  // Invalid target CPU, tune CPU, or target feature flags should return an
  // error.
  EXPECT_THAT(compute({.target = "x86_64-unknown-linux-gnu",
                       .target_cpu = "invalid-cpu-name"}),
              IsError(_));
  EXPECT_THAT(compute({.target = "x86_64-unknown-linux-gnu",
                       .target_cpu_tune = "invalid-cpu-name"}),
              IsError(_));
  EXPECT_THAT(compute({.target = "x86_64-unknown-linux-gnu",
                       .target_cpu_features = "+not-a-feature"}),
              IsError(_));
  EXPECT_THAT(compute({.target = "riscv64-unknown-linux-gnu",
                       .target_cpu_features = "+f,+zfinx"}),
              IsError(_));
}

}  // namespace
}  // namespace Carbon
