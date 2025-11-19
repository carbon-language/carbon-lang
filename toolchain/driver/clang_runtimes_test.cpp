// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/clang_runtimes.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <utility>

#include "common/ostream.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "testing/base/capture_std_streams.h"
#include "testing/base/global_exe_path.h"
#include "toolchain/base/llvm_tools.h"
#include "toolchain/driver/clang_runner.h"
#include "toolchain/driver/llvm_runner.h"
#include "toolchain/driver/runtimes_cache.h"
#include "toolchain/install/install_paths.h"

namespace Carbon {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsSupersetOf;

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

class ClangRuntimesTest : public ::testing::Test {
 public:
  InstallPaths install_paths_ =
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath());
  Runtimes::Cache runtimes_cache_ =
      *Runtimes::Cache::MakeSystem(install_paths_);
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs_ =
      llvm::vfs::getRealFileSystem();
};

TEST_F(ClangRuntimesTest, ResourceDir) {
  ClangRunner runner(&install_paths_, vfs_, &llvm::errs());

  // Note that we can't test arbitrary targets here as we need to be able to
  // compile the builtin functions for the target. We use the default target as
  // the most likely to pass.
  std::string target = llvm::sys::getDefaultTargetTriple();
  llvm::Triple target_triple(target);
  Runtimes::Cache::Features features = {.target = target};
  auto runtimes = *runtimes_cache_.Lookup(features);
  llvm::DefaultThreadPool threads(llvm::optimal_concurrency());

  ClangResourceDirBuilder resource_dir_builder(&runner, &threads, target_triple,
                                               &runtimes);
  auto build_result = std::move(resource_dir_builder).Wait();
  ASSERT_TRUE(build_result.ok()) << build_result.error();
  std::filesystem::path resource_dir_path = std::move(*build_result);

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

}  // namespace
}  // namespace Carbon
