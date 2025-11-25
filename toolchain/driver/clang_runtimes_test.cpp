// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/clang_runtimes.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <utility>

#include "common/check.h"
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
  // Helper to get the `llvm-nm` listing of defined symbols for an archive.
  //
  // TODO: It would be nice to use a library API and matchers instead of
  // `llvm-nm` and matching text on the output.
  auto NmListDefinedSymbols(const std::filesystem::path& archive)
      -> std::string {
    LLVMRunner llvm_runner(&install_paths_, &llvm::errs());
    std::string out;
    std::string err;
    bool result = Testing::CallWithCapturedOutput(out, err, [&] {
      return llvm_runner.Run(
          LLVMTool::Nm, {"--format=just-symbols", "--defined-only", "--quiet",
                         archive.native()});
    });
    CARBON_CHECK(result, "Unable to run `llvm-nm`:\n{1}", err);

    return out;
  }

  // Helper to expect a specific symbol in the `llvm-nm` list.
  //
  // This handles platform-specific formatting of symbols.
  auto ExpectSymbol(llvm::StringRef nm_list, llvm::StringRef symbol) -> void {
    std::string symbol_substr = llvm::formatv(
        target_triple_.isMacOSX() ? "\n_{0}\n" : "\n{0}\n", symbol);

    // Do the actual match with `HasSubstr` so it can explain failures.
    EXPECT_THAT(nm_list, HasSubstr(symbol_substr));
  }

  InstallPaths install_paths_ =
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath());
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs_ =
      llvm::vfs::getRealFileSystem();
  // Note that for debugging, you can pass `llvm::errs()` as the vlog stream,
  // but this makes the output both very verbose and hard to use with multiple
  // threads.
  ClangRunner runner_{&install_paths_, vfs_};

  // Note that we can't test arbitrary targets here as we need to be able to
  // compile the builtin functions for the target. We use the default target as
  // the most likely to pass.
  std::string target_ = llvm::sys::getDefaultTargetTriple();
  llvm::Triple target_triple_{target_};

  Runtimes::Cache runtimes_cache_ =
      *Runtimes::Cache::MakeSystem(install_paths_);
  Runtimes::Cache::Features features = {.target = target_};
  Runtimes runtimes_ = *runtimes_cache_.Lookup(features);

  // Note that for debugging it may be useful to replace this with a
  // single-threaded thread pool. However the test will be _much_ slower.
  llvm::DefaultThreadPool threads_{llvm::optimal_concurrency()};
};

TEST_F(ClangRuntimesTest, ResourceDir) {
  ClangResourceDirBuilder resource_dir_builder(&runner_, &threads_,
                                               target_triple_, &runtimes_);
  auto build_result = std::move(resource_dir_builder).Wait();
  ASSERT_TRUE(build_result.ok()) << build_result.error();
  std::filesystem::path resource_dir_path = std::move(*build_result);

  // For Linux we can directly check the CRT begin/end object files.
  if (target_triple_.isOSLinux()) {
    std::filesystem::path crt_begin_path =
        resource_dir_path / "lib" / target_ / "clang_rt.crtbegin.o";
    ASSERT_TRUE(std::filesystem::is_regular_file(crt_begin_path));
    auto begin_result =
        llvm::object::ObjectFile::createObjectFile(crt_begin_path.native());
    llvm::object::ObjectFile& crtbegin = *begin_result->getBinary();
    EXPECT_TRUE(crtbegin.isELF());
    EXPECT_TRUE(crtbegin.isObject());
    EXPECT_THAT(crtbegin.getArch(), Eq(target_triple_.getArch()));

    llvm::SmallVector<llvm::object::SymbolRef> symbols(crtbegin.symbols());
    // The first symbol should come from the source file.
    EXPECT_THAT(*symbols.front().getName(), Eq("crtbegin.c"));

    // Check for representative symbols of `crtbegin.o` -- we always use
    // `.init_array` in our runtimes build so we have predictable functions.
    EXPECT_THAT(symbols, IsSupersetOf({TextSymbolNamed("__do_init"),
                                       TextSymbolNamed("__do_fini")}));

    std::filesystem::path crt_end_path =
        resource_dir_path / "lib" / target_ / "clang_rt.crtend.o";
    ASSERT_TRUE(std::filesystem::is_regular_file(crt_end_path));
    auto end_result =
        llvm::object::ObjectFile::createObjectFile(crt_end_path.native());
    llvm::object::ObjectFile& crtend = *end_result->getBinary();
    EXPECT_TRUE(crtend.isELF());
    EXPECT_TRUE(crtend.isObject());
    EXPECT_THAT(crtend.getArch(), Eq(target_triple_.getArch()));

    // Just check the source file symbol, not much of interest in the end.
    llvm::object::SymbolRef crtend_front_symbol = *crtend.symbol_begin();
    EXPECT_THAT(*crtend_front_symbol.getName(), Eq("crtend.c"));
  }

  // Across all targets, check that the builtins archive exists, and contains a
  // relevant symbol by running the `llvm-nm` tool over it. Using `nm` rather
  // than directly inspecting the objects is a bit awkward, but lets us easily
  // ignore the wrapping in an archive file.
  std::filesystem::path builtins_path =
      resource_dir_path / "lib" / target_ / "libclang_rt.builtins.a";
  std::string builtins_symbols = NmListDefinedSymbols(builtins_path);

  // Check that we found a definition of `__mulodi4`, a builtin function
  // provided by Compiler-RT.
  ExpectSymbol(builtins_symbols, "__mulodi4");

  // Check that we don't include the `chkstk` builtins outside of Windows.
  if (!target_triple_.isOSWindows()) {
    EXPECT_THAT(builtins_symbols, Not(HasSubstr("chkstk")));
  }
}

TEST_F(ClangRuntimesTest, Libunwind) {
  LibunwindBuilder libunwind_builder(&runner_, &threads_, target_triple_,
                                     &runtimes_);
  auto build_result = std::move(libunwind_builder).Wait();
  ASSERT_TRUE(build_result.ok()) << build_result.error();
  std::filesystem::path runtimes_path = std::move(*build_result);

  std::filesystem::path libunwind_path = runtimes_path / "lib/libunwind.a";
  std::string libunwind_symbols = NmListDefinedSymbols(libunwind_path);

  // Check a few of the main exported symbols here. The set here is somewhat
  // arbitrary, but chosen to be among the more stable names and have at least
  // one from most of the object files that should be linked into the archive.
  ExpectSymbol(libunwind_symbols, "_Unwind_Resume");
  ExpectSymbol(libunwind_symbols, "_Unwind_Backtrace");
  ExpectSymbol(libunwind_symbols, "__unw_getcontext");
  ExpectSymbol(libunwind_symbols, "__unw_get_proc_info");
}

}  // namespace
}  // namespace Carbon
