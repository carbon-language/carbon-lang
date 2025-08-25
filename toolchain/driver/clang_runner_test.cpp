// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/clang_runner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

#include "common/check.h"
#include "common/ostream.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Host.h"
#include "testing/base/capture_std_streams.h"
#include "testing/base/file_helpers.h"
#include "testing/base/global_exe_path.h"
#include "toolchain/driver/llvm_runner.h"

namespace Carbon {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsSupersetOf;
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
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs_ =
      llvm::vfs::getRealFileSystem();
};

TEST_F(ClangRunnerTest, Version) {
  RawStringOstream test_os;
  ClangRunner runner(&install_paths_, vfs_, &test_os);

  std::string out;
  std::string err;
  EXPECT_TRUE(Testing::CallWithCapturedOutput(out, err, [&] {
    return runner.RunTargetIndependentCommand({"--version"});
  }));
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
  EXPECT_TRUE(Testing::CallWithCapturedOutput(
      out, err,
      [&] {
        return runner.RunTargetIndependentCommand(
            {"-c", test_file.string(), "-o", test_output.string()});
      }))
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
  EXPECT_TRUE(Testing::CallWithCapturedOutput(
      out, err,
      [&] {
        return runner.RunTargetIndependentCommand(
            {"-c", test_file.string(), "-o", test_output.string()});
      }))
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
    EXPECT_TRUE(Testing::CallWithCapturedOutput(
        out, err,
        [&] {
          return runner.RunTargetIndependentCommand(
              {"-c", file.string(), "-o", output.string()});
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

TEST_F(ClangRunnerTest, BuildResourceDir) {
  ClangRunner runner(&install_paths_, vfs_, &llvm::errs(),
                     /*build_runtimes_on_demand=*/true);

  // Note that we can't test arbitrary targets here as we need to be able to
  // compile the builtin functions for the target. We use the default target as
  // the most likely to pass.
  std::string target = llvm::sys::getDefaultTargetTriple();
  llvm::Triple target_triple(target);
  auto tmp_dir = *Filesystem::MakeTmpDir();
  std::filesystem::path resource_dir_path = tmp_dir.abs_path() / "clang";

  auto build_result = runner.BuildTargetResourceDir(target, resource_dir_path,
                                                    tmp_dir.abs_path());
  ASSERT_TRUE(build_result.ok()) << build_result.error();

  // For Linux we can directly check the CRT begin/end object files.
  if (target_triple.isOSLinux()) {
    std::filesystem::path crt_begin_path =
        resource_dir_path / "lib" / target / "clang_rt.crtbegin.o";
    ASSERT_TRUE(std::filesystem::is_regular_file(crt_begin_path));
    auto begin_result =
        llvm::object::ObjectFile::createObjectFile(crt_begin_path.native());
    llvm::object::ObjectFile& crtbegin = *begin_result->getBinary();
    EXPECT_TRUE(crtbegin.isELF());
    EXPECT_TRUE(crtbegin.isObject());
    EXPECT_THAT(crtbegin.getArch(), Eq(target_triple.getArch()));

    llvm::SmallVector<llvm::object::SymbolRef> symbols(crtbegin.symbols());
    // The first symbol should come from the source file.
    EXPECT_THAT(*symbols.front().getName(), Eq("crtbegin.c"));

    // Check for representative symbols of `crtbegin.o` -- we always use
    // `.init_array` in our runtimes build so we have predictable functions.
    EXPECT_THAT(symbols, IsSupersetOf({TextSymbolNamed("__do_init"),
                                       TextSymbolNamed("__do_fini")}));

    std::filesystem::path crt_end_path =
        resource_dir_path / "lib" / target / "clang_rt.crtend.o";
    ASSERT_TRUE(std::filesystem::is_regular_file(crt_end_path));
    auto end_result =
        llvm::object::ObjectFile::createObjectFile(crt_end_path.native());
    llvm::object::ObjectFile& crtend = *end_result->getBinary();
    EXPECT_TRUE(crtend.isELF());
    EXPECT_TRUE(crtend.isObject());
    EXPECT_THAT(crtend.getArch(), Eq(target_triple.getArch()));

    // Just check the source file symbol, not much of interest in the end.
    llvm::object::SymbolRef crtend_front_symbol = *crtend.symbol_begin();
    EXPECT_THAT(*crtend_front_symbol.getName(), Eq("crtend.c"));
  }

  // Across all targets, check that the builtins archive exists, and contains a
  // relevant symbol by running the `llvm-nm` tool over it. Using `nm` rather
  // than directly inspecting the objects is a bit awkward, but lets us easily
  // ignore the wrapping in an archive file.
  std::filesystem::path builtins_path =
      resource_dir_path / "lib" / target / "libclang_rt.builtins.a";
  LLVMRunner llvm_runner(&install_paths_, &llvm::errs());
  std::string out;
  std::string err;
  EXPECT_TRUE(Testing::CallWithCapturedOutput(out, err, [&] {
    return llvm_runner.Run(LLVMTool::Nm, {builtins_path.native()});
  }));

  // Check that we found a definition of `__mulodi4`, a builtin function
  // provided by Compiler-RT, but not `libgcc` historically. Note that on macOS
  // there is a leading `_` due to mangling.
  EXPECT_THAT(out, HasSubstr(target_triple.isMacOSX() ? "T ___mulodi4\n"
                                                      : "T __mulodi4\n"));

  // Check that we don't include the `chkstk` builtins outside of Windows.
  if (!target_triple.isOSWindows()) {
    EXPECT_THAT(out, Not(HasSubstr("chkstk")));
  }
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
  EXPECT_TRUE(Testing::CallWithCapturedOutput(
      out, err,
      [&] {
        // Note that we use the target independent run command here because
        // we're just getting the echo-ed output back. For this to actually
        // link, we'd need to have the target-dependent resources, but those are
        // expensive to build so we only want to test them once (above).
        return runner.RunTargetIndependentCommand(
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

}  // namespace
}  // namespace Carbon
