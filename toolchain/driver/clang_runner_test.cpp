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
#include "toolchain/driver/codegen_options.h"
#include "toolchain/driver/runtimes_cache.h"

namespace Carbon {
namespace {

using ::testing::_;
using ::testing::ContainerEq;
using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using Testing::IsError;
using Testing::IsSuccess;
using ::testing::Ne;
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

  // x86_64: default, microarchitecture level, specific CPU, and tune CPU.
  auto x86_default =
      runner.ComputeRuntimesFeatures("x86_64-unknown-linux-gnu", {});
  ASSERT_THAT(x86_default, IsSuccess(_));
  EXPECT_THAT(x86_default->cpu, StrEq("x86-64"));
  EXPECT_THAT(x86_default->tune_cpu, StrEq("generic"));

  auto x86_v3 = runner.ComputeRuntimesFeatures("x86_64-unknown-linux-gnu",
                                               {"-march=x86-64-v3"});
  ASSERT_THAT(x86_v3, IsSuccess(_));
  EXPECT_THAT(x86_v3->cpu, StrEq("x86-64-v3"));
  EXPECT_THAT(x86_v3->tune_cpu, StrEq("generic"));
  EXPECT_THAT(x86_v3->target_features, Contains("+avx2"));
  EXPECT_THAT(x86_v3->target_features, Contains("+bmi2"));

  auto x86_v3_tuned = runner.ComputeRuntimesFeatures(
      "x86_64-unknown-linux-gnu", {"-march=x86-64-v3", "-mtune=znver4"});
  ASSERT_THAT(x86_v3_tuned, IsSuccess(_));
  EXPECT_THAT(x86_v3_tuned->cpu, StrEq("x86-64-v3"));
  EXPECT_THAT(x86_v3_tuned->tune_cpu, StrEq("znver4"));

  auto x86_znver4 = runner.ComputeRuntimesFeatures("x86_64-unknown-linux-gnu",
                                                   {"-march=znver4"});
  ASSERT_THAT(x86_znver4, IsSuccess(_));
  EXPECT_THAT(x86_znver4->cpu, StrEq("znver4"));
  EXPECT_THAT(x86_znver4->tune_cpu, StrEq("znver4"));
  EXPECT_THAT(x86_znver4->target_features, Contains("+avx512f"));

  auto x86_znver4_explicit_tune = runner.ComputeRuntimesFeatures(
      "x86_64-unknown-linux-gnu", {"-march=znver4", "-mtune=znver4"});
  ASSERT_THAT(x86_znver4_explicit_tune, IsSuccess(_));
  EXPECT_THAT(x86_znver4_explicit_tune->cpu, StrEq(x86_znver4->cpu));
  EXPECT_THAT(x86_znver4_explicit_tune->tune_cpu, StrEq(x86_znver4->tune_cpu));

  // AArch64: architecture string (-march), CPU name (-mcpu), and tune (-mtune).
  auto aarch64_v9 = runner.ComputeRuntimesFeatures(
      "aarch64-unknown-linux-gnu", {"-march=armv9-a", "-mtune=neoverse-v2"});
  ASSERT_THAT(aarch64_v9, IsSuccess(_));
  EXPECT_THAT(aarch64_v9->tune_cpu, StrEq("neoverse-v2"));
  EXPECT_THAT(aarch64_v9->target_features, Contains("+v9a"));
  EXPECT_THAT(aarch64_v9->target_features, Contains("+sve2"));

  auto aarch64_neoverse = runner.ComputeRuntimesFeatures(
      "aarch64-unknown-linux-gnu", {"-mcpu=neoverse-v2"});
  ASSERT_THAT(aarch64_neoverse, IsSuccess(_));
  EXPECT_THAT(aarch64_neoverse->cpu, StrEq("neoverse-v2"));
  EXPECT_THAT(aarch64_neoverse->tune_cpu, StrEq("neoverse-v2"));
  EXPECT_THAT(aarch64_neoverse->target_features, Contains("+v9a"));

  // RISC-V: ISA string (-march), profile (-march), CPU name (-mcpu), and tune
  // (-mtune).
  auto riscv_gc = runner.ComputeRuntimesFeatures(
      "riscv64-unknown-linux-gnu", {"-march=rv64gc", "-mtune=spacemit-x60"});
  ASSERT_THAT(riscv_gc, IsSuccess(_));
  EXPECT_THAT(riscv_gc->tune_cpu, StrEq("spacemit-x60"));
  EXPECT_THAT(riscv_gc->target_features, Contains("+m"));
  EXPECT_THAT(riscv_gc->target_features, Contains("+a"));
  EXPECT_THAT(riscv_gc->target_features, Contains("+f"));
  EXPECT_THAT(riscv_gc->target_features, Contains("+d"));
  EXPECT_THAT(riscv_gc->target_features, Contains("+c"));

  auto riscv_profile = runner.ComputeRuntimesFeatures(
      "riscv64-unknown-linux-gnu", {"-march=rva22u64"});
  ASSERT_THAT(riscv_profile, IsSuccess(_));
  EXPECT_THAT(riscv_profile->target_features, Contains("+zba"));
  EXPECT_THAT(riscv_profile->target_features, Contains("+zbb"));
  EXPECT_THAT(riscv_profile->target_features, Contains("+zbs"));

  auto riscv_cpu = runner.ComputeRuntimesFeatures("riscv64-unknown-linux-gnu",
                                                  {"-mcpu=spacemit-x60"});
  ASSERT_THAT(riscv_cpu, IsSuccess(_));
  EXPECT_THAT(riscv_cpu->cpu, StrEq("spacemit-x60"));
  EXPECT_THAT(riscv_cpu->tune_cpu, StrEq("spacemit-x60"));
  EXPECT_THAT(riscv_cpu->target_features, Contains("+v"));

  // Host `native` resolution should canonicalize to a concrete CPU name for
  // both target CPU and tune CPU.
  std::string host_target = llvm::sys::getDefaultTargetTriple();
  llvm::Triple host_triple(host_target);
  if (host_triple.isX86() || host_triple.isAArch64()) {
    std::string host_cpu = llvm::sys::getHostCPUName().str();
    std::string native_flag =
        host_triple.isX86() ? "-march=native" : "-mcpu=native";
    auto native_features =
        runner.ComputeRuntimesFeatures(host_target, {native_flag});
    ASSERT_THAT(native_features, IsSuccess(_));
    EXPECT_THAT(native_features->cpu, StrEq(host_cpu));
    EXPECT_THAT(native_features->tune_cpu, StrEq(host_cpu));
    EXPECT_THAT(native_features->target_features, Not(IsEmpty()));

    auto native_explicit_tune = runner.ComputeRuntimesFeatures(
        host_target, {native_flag, "-mtune=native"});
    ASSERT_THAT(native_explicit_tune, IsSuccess(_));
    EXPECT_THAT(native_explicit_tune->cpu, StrEq(host_cpu));
    EXPECT_THAT(native_explicit_tune->tune_cpu, StrEq(host_cpu));

    auto native_tune_only =
        runner.ComputeRuntimesFeatures(host_target, {"-mtune=native"});
    ASSERT_THAT(native_tune_only, IsSuccess(_));
    EXPECT_THAT(native_tune_only->tune_cpu, StrEq(host_cpu));
  }

  // Full link and compile driver command lines (with object files, `-o`, `-l`
  // flags, C language flags, or `-###`) should still succeed in extracting
  // target features.
  auto link_cmd = runner.ComputeRuntimesFeatures(
      "x86_64-unknown-linux-gnu", {"--driver-mode=g++", "-march=x86-64-v3",
                                   "-o", "out", "-lm", "--", "foo.o"});
  ASSERT_THAT(link_cmd, IsSuccess(_));
  EXPECT_THAT(link_cmd->cpu, StrEq("x86-64-v3"));
  EXPECT_THAT(link_cmd->target_features, Contains("+avx2"));

  auto c_cmd = runner.ComputeRuntimesFeatures("x86_64-unknown-linux-gnu",
                                              {"-march=x86-64-v3", "-std=c11"});
  ASSERT_THAT(c_cmd, IsSuccess(_));
  EXPECT_THAT(c_cmd->cpu, StrEq("x86-64-v3"));

  std::string out;
  std::string err;
  auto hash_cmd = Testing::CallWithCapturedOutput(out, err, [&] {
    return runner.ComputeRuntimesFeatures("x86_64-unknown-linux-gnu",
                                          {"-march=x86-64-v3", "-###"});
  });
  ASSERT_THAT(hash_cmd, IsSuccess(_));
  EXPECT_THAT(hash_cmd->cpu, StrEq("x86-64-v3"));

  // Invalid target CPU, tune CPU, or target feature flags should return an
  // error.
  EXPECT_THAT(runner.ComputeRuntimesFeatures("x86_64-unknown-linux-gnu",
                                             {"-march=invalid-cpu-name"}),
              IsError(_));
  EXPECT_THAT(runner.ComputeRuntimesFeatures("x86_64-unknown-linux-gnu",
                                             {"-mtune=invalid-cpu-name"}),
              IsError(_));
  EXPECT_THAT(runner.ComputeRuntimesFeatures(
                  "x86_64-unknown-linux-gnu",
                  {"-Xclang", "-target-feature", "-Xclang", "+not-a-feature"}),
              IsError(_));
  EXPECT_THAT(runner.ComputeRuntimesFeatures(
                  "riscv64-unknown-linux-gnu",
                  {"-Xclang", "-target-feature", "-Xclang", "+f", "-Xclang",
                   "-target-feature", "-Xclang", "+zfinx"}),
              IsError(_));
}

TEST_F(ClangRunnerTest, EffectiveTargetArgsAndTieBreaking) {
  ClangRunner runner(&install_paths_, vfs_);

  auto compute_effective = [&](const CodegenOptions& options,
                               llvm::ArrayRef<llvm::StringRef> clang_args = {})
      -> ErrorOr<Runtimes::Cache::Features> {
    llvm::SmallVector<std::string> codegen_args = options.GetClangArgs();
    llvm::SmallVector<llvm::StringRef> effective_refs(clang_args.begin(),
                                                      clang_args.end());
    for (llvm::StringRef arg : codegen_args) {
      effective_refs.push_back(arg);
    }
    return runner.ComputeRuntimesFeatures(options.target, effective_refs);
  };

  // `--target-cpu-features` across x86_64, AArch64, and RISC-V, including `+`,
  // `-`, and bare feature names.
  {
    CodegenOptions opts = {.target = "x86_64-unknown-linux-gnu",
                           .target_cpu = "x86-64-v3",
                           .target_cpu_features = "-avx2,avx512f"};
    auto features = compute_effective(opts);
    ASSERT_THAT(features, IsSuccess(_));
    EXPECT_THAT(features->cpu, StrEq("x86-64-v3"));
    // Enabling `avx512f` after `-avx2` works in order and re-enables `avx2` as
    // an implied prerequisite.
    EXPECT_THAT(features->target_features, Contains("+avx512f"));
    EXPECT_THAT(features->target_features, Contains("+avx2"));
  }
  {
    CodegenOptions opts = {.target = "x86_64-unknown-linux-gnu",
                           .target_cpu = "x86-64-v3",
                           .target_cpu_features = "avx512f,-avx512f,-fma"};
    auto features = compute_effective(opts);
    ASSERT_THAT(features, IsSuccess(_));
    EXPECT_THAT(features->target_features, Not(Contains("+avx512f")));
    EXPECT_THAT(features->target_features, Not(Contains("+fma")));
    EXPECT_THAT(features->target_features, Contains("+avx2"));
  }
  {
    CodegenOptions opts = {.target = "aarch64-unknown-linux-gnu",
                           .target_cpu = "armv8-a",
                           .target_cpu_features = "sve,+lse,-crc"};
    auto features = compute_effective(opts);
    ASSERT_THAT(features, IsSuccess(_));
    EXPECT_THAT(features->target_features, Contains("+sve"));
    EXPECT_THAT(features->target_features, Contains("+lse"));
    EXPECT_THAT(features->target_features, Not(Contains("+crc")));
  }
  {
    CodegenOptions opts = {.target = "riscv64-unknown-linux-gnu",
                           .target_cpu = "rv64gc",
                           .target_cpu_features = "+zba,zbb,-m"};
    auto features = compute_effective(opts);
    ASSERT_THAT(features, IsSuccess(_));
    EXPECT_THAT(features->target_features, Contains("+zba"));
    EXPECT_THAT(features->target_features, Contains("+zbb"));
    EXPECT_THAT(features->target_features, Not(Contains("+m")));
  }

  // Non-conflicting Clang flags and Carbon CLI flags combine into an effective
  // set.
  {
    CodegenOptions opts = {.target = "x86_64-unknown-linux-gnu",
                           .target_cpu = "x86-64-v2",
                           .target_cpu_features = "+bmi2"};
    auto features =
        compute_effective(opts, {"-I/some/include", "-mtune=znver4", "-mavx2"});
    ASSERT_THAT(features, IsSuccess(_));
    EXPECT_THAT(features->cpu, StrEq("x86-64-v2"));
    EXPECT_THAT(features->tune_cpu, StrEq("znver4"));
    EXPECT_THAT(features->target_features, Contains("+avx2"));
    EXPECT_THAT(features->target_features, Contains("+bmi2"));
  }
  {
    CodegenOptions opts = {.target = "x86_64-unknown-linux-gnu",
                           .target_cpu_tune = "znver4",
                           .target_cpu_features = "+avx512f"};
    auto features = compute_effective(opts, {"-march=x86-64-v3", "-mbmi2"});
    ASSERT_THAT(features, IsSuccess(_));
    EXPECT_THAT(features->cpu, StrEq("x86-64-v3"));
    EXPECT_THAT(features->tune_cpu, StrEq("znver4"));
    EXPECT_THAT(features->target_features, Contains("+avx2"));
    EXPECT_THAT(features->target_features, Contains("+avx512f"));
  }

  // Conflicting Clang flags are tie-broken in favor of Carbon CLI flags across
  // x86_64, AArch64, and RISC-V.
  {
    // x86_64: `--target-cpu`, `--target-cpu-tune`, and `--target-cpu-features`
    // override `-march=`, `-mtune=`, and `-m<feat>`.
    CodegenOptions opts = {.target = "x86_64-unknown-linux-gnu",
                           .target_cpu = "x86-64-v3",
                           .target_cpu_tune = "generic",
                           .target_cpu_features = "-fma,+aes"};
    auto features = compute_effective(
        opts,
        {"-march=x86-64-v2", "-mtune=znver4", "-mfma", "-mno-aes", "-mlzcnt"});
    ASSERT_THAT(features, IsSuccess(_));
    EXPECT_THAT(features->cpu, StrEq("x86-64-v3"));
    EXPECT_THAT(features->tune_cpu, StrEq("generic"));
    EXPECT_THAT(features->target_features, Not(Contains("+fma")));
    EXPECT_THAT(features->target_features, Contains("+aes"));
    EXPECT_THAT(features->target_features, Contains("+lzcnt"));
  }
  {
    // AArch64: `--target-cpu` overrides both Clang `-march=` and `-mcpu=`,
    // including across `-march=` vs `-mcpu=` spellings.
    CodegenOptions arch_opts = {.target = "aarch64-unknown-linux-gnu",
                                .target_cpu = "armv8.2-a",
                                .target_cpu_features = "-crc"};
    auto arch_features = compute_effective(
        arch_opts, {"-march=armv9-a", "-mcpu=neoverse-v2", "-mcrc"});
    ASSERT_THAT(arch_features, IsSuccess(_));
    EXPECT_THAT(arch_features->cpu, StrEq("generic"));
    EXPECT_THAT(arch_features->target_features, Contains("+v8.2a"));
    EXPECT_THAT(arch_features->target_features, Not(Contains("+v9a")));
    EXPECT_THAT(arch_features->target_features, Not(Contains("+sve2")));
    EXPECT_THAT(arch_features->target_features, Not(Contains("+crc")));

    CodegenOptions cpu_opts = {.target = "aarch64-unknown-linux-gnu",
                               .target_cpu = "cortex-a53"};
    auto cpu_features =
        compute_effective(cpu_opts, {"-march=armv9-a", "-mcpu=neoverse-v2"});
    auto direct_a53 = runner.ComputeRuntimesFeatures(
        "aarch64-unknown-linux-gnu", {"-mcpu=cortex-a53"});
    ASSERT_THAT(cpu_features, IsSuccess(_));
    ASSERT_THAT(direct_a53, IsSuccess(_));
    EXPECT_THAT(cpu_features->cpu, StrEq("cortex-a53"));
    EXPECT_THAT(cpu_features->target_features,
                ContainerEq(direct_a53->target_features));

    CodegenOptions v2_opts = {.target = "aarch64-unknown-linux-gnu",
                              .target_cpu = "neoverse-v2"};
    auto v2_features =
        compute_effective(v2_opts, {"-march=armv8-a", "-mcpu=cortex-a53"});
    auto direct_v2 = runner.ComputeRuntimesFeatures("aarch64-unknown-linux-gnu",
                                                    {"-mcpu=neoverse-v2"});
    ASSERT_THAT(v2_features, IsSuccess(_));
    ASSERT_THAT(direct_v2, IsSuccess(_));
    EXPECT_THAT(v2_features->cpu, StrEq("neoverse-v2"));
    EXPECT_THAT(v2_features->target_features,
                ContainerEq(direct_v2->target_features));
  }
  {
    // RISC-V: `--target-cpu` overrides both Clang `-march=` and `-mcpu=`,
    // including across `-march=` vs `-mcpu=` spellings.
    CodegenOptions arch_opts = {.target = "riscv64-unknown-linux-gnu",
                                .target_cpu = "rv64gc"};
    auto arch_features =
        compute_effective(arch_opts, {"-march=rv64gcv", "-mcpu=spacemit-x60"});
    ASSERT_THAT(arch_features, IsSuccess(_));
    EXPECT_THAT(arch_features->cpu, StrEq("generic-rv64"));
    EXPECT_THAT(arch_features->tune_cpu, StrEq("generic-rv64"));
    EXPECT_THAT(arch_features->target_features, Not(Contains("+v")));

    CodegenOptions cpu_opts = {.target = "riscv64-unknown-linux-gnu",
                               .target_cpu = "sifive-u74"};
    auto cpu_features =
        compute_effective(cpu_opts, {"-march=rv64gcv", "-mcpu=spacemit-x60"});
    auto direct_u74 = runner.ComputeRuntimesFeatures(
        "riscv64-unknown-linux-gnu", {"-mcpu=sifive-u74"});
    ASSERT_THAT(cpu_features, IsSuccess(_));
    ASSERT_THAT(direct_u74, IsSuccess(_));
    EXPECT_THAT(cpu_features->cpu, StrEq("sifive-u74"));
    EXPECT_THAT(cpu_features->target_features,
                ContainerEq(direct_u74->target_features));
  }
}

}  // namespace
}  // namespace Carbon
