// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/runtimes_cache.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <ratio>
#include <string>
#include <thread>
#include <utility>
#include <variant>

#include "common/check.h"
#include "common/error_test_helpers.h"
#include "common/filesystem.h"
#include "common/ostream.h"
#include "common/raw_string_ostream.h"
#include "common/version.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/SHA256.h"
#include "testing/base/capture_std_streams.h"
#include "testing/base/file_helpers.h"
#include "testing/base/global_exe_path.h"

namespace Carbon {

class RuntimesTestPeer {
 public:
  static auto LockFilePath(Runtimes::Component component) -> std::string {
    return llvm::formatv(Runtimes::LockFileFormat,
                         Runtimes::ComponentPath(component))
        .str();
  }

  static auto BuildImpl(Runtimes& runtimes, Runtimes::Component component,
                        Filesystem::Duration deadline,
                        Filesystem::Duration poll_interval)
      -> ErrorOr<std::variant<std::filesystem::path, Runtimes::Builder>> {
    return runtimes.BuildImpl(component, deadline, poll_interval);
  }

  static auto CacheMinNumEntries() -> int {
    return Runtimes::Cache::MinNumEntries;
  }
  static auto CacheMaxNumEntries() -> int {
    return Runtimes::Cache::MaxNumEntries;
  }
};

namespace {

using ::testing::_;
using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::Eq;
using ::testing::Gt;
using Testing::IsError;
using Testing::IsSuccess;
using ::testing::Lt;
using ::testing::Ne;
using ::testing::Not;
using ::testing::StartsWith;
using ::testing::StrEq;
using ::testing::VariantWith;

class RuntimesCacheTest : public ::testing::Test {
 public:
  RuntimesCacheTest()
      : cache_(*Runtimes::Cache::MakeCustom(install_, tmp_dir_.abs_path())) {}

  auto LookupNRuntimes(int n) -> llvm::SmallVector<Runtimes> {
    llvm::SmallVector<Runtimes> runtimes;
    for (int i : llvm::seq(n)) {
      runtimes.push_back(*cache_.Lookup(
          {.target = llvm::formatv("aarch64-unknown-unknown{0}", i).str()}));
    }
    return runtimes;
  }

  InstallPaths install_ =
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath());
  Filesystem::RemovingDir tmp_dir_ = *Filesystem::MakeTmpDir();
  std::string cache_key_ = "test cache";
  Runtimes::Cache cache_;
};

TEST_F(RuntimesCacheTest, BuildSystemCache) {
  // Create an install with a missing digest.
  auto bad_install_dir = *tmp_dir_.CreateDirectories("bad_install/lib/carbon");
  bad_install_dir.WriteFileFromString("carbon_install.txt", "no digest")
      .Check();
  InstallPaths bad_install =
      InstallPaths::Make((tmp_dir_.abs_path() / "bad_install").native());

  // Create directories to use in various environment variables.
  auto xdg_dir = *tmp_dir_.CreateDirectories("xdg_cache_home");
  std::filesystem::path xdg_path = tmp_dir_.abs_path() / "xdg_cache_home";
  auto test_home = *tmp_dir_.CreateDirectories("test_home");
  std::filesystem::path home_path = tmp_dir_.abs_path() / "test_home";
  auto home_cache_dir = *test_home.CreateDirectories(".cache");
  std::filesystem::path home_cache_path = home_path / ".cache";

  // Save the environment variables we'll override for testing and restore them
  // afterward to avoid test-to-test oddities.
  constexpr const char* XdgCacheEnv = "XDG_CACHE_HOME";
  constexpr const char* HomeEnv = "HOME";
  const char* orig_xdg_cache = getenv(XdgCacheEnv);
  const char* orig_home = getenv(HomeEnv);
  auto restore_env = llvm::make_scope_exit([&] {
    for (const auto [env, orig] : {std::pair{XdgCacheEnv, orig_xdg_cache},
                                   std::pair{HomeEnv, orig_home}}) {
      if (orig) {
        setenv(env, orig, /*overwrite*/ true);
      } else {
        unsetenv(env);
      }
    }
  });

  // Begin testing the basic logic of selecting different roots for the cache.
  setenv(XdgCacheEnv, xdg_path.c_str(), /*overwrite*/ true);
  setenv(HomeEnv, home_path.c_str(), /*overwrite*/ true);

  // First check that even with all the environment set up, when we don't have a
  // digest file available, we bypass those options and use a temporary cache
  // path. This is the only safe approach as without a digest file we can't
  // track whether it is correct to reuse a persistently cached entry.
  auto result = Runtimes::Cache::MakeSystem(bad_install);
  ASSERT_THAT(result, IsSuccess(_));
  EXPECT_THAT(result->path(), Not(StartsWith(home_cache_path)));
  EXPECT_THAT(result->path(), Not(StartsWith(xdg_path)));

  // Once we have a digest, the main XDG cache logic should work.
  result = Runtimes::Cache::MakeSystem(install_);
  ASSERT_THAT(result, IsSuccess(_));
  EXPECT_THAT(result->path(), StartsWith(xdg_path));
  // Destruction shouldn't remove system cache directories.
  result = Error("nothing");
  EXPECT_TRUE(*tmp_dir_.Access("xdg_cache_home"));

  // Remove the XDG cache directory, but leave the environment set. We want to
  // be robust against this, but it isn't important *how* the fallback occurs,
  // it could go to `$HOME/.cache`, or to a temporary directory.
  tmp_dir_.Rmtree("xdg_cache_home").Check();
  EXPECT_THAT(Runtimes::Cache::MakeSystem(install_), IsSuccess(_));

  // Set the XDG environment to the empty string which should trigger using the
  // home directory.
  setenv(XdgCacheEnv, "", /*overwrite*/ true);
  result = Runtimes::Cache::MakeSystem(install_);
  ASSERT_THAT(result, IsSuccess(_));
  EXPECT_THAT(result->path(), StartsWith(home_cache_path));
  // Destruction shouldn't remove system cache directories.
  result = Error("nothing");
  EXPECT_TRUE(*tmp_dir_.Access("test_home"));
  EXPECT_TRUE(*test_home.Access(".cache"));

  // Same as with an empty string, but with a relative path instead.
  setenv(XdgCacheEnv, "relative/cache/home", /*overwrite*/ true);
  result = Runtimes::Cache::MakeSystem(install_);
  ASSERT_THAT(result, IsSuccess(_));
  EXPECT_THAT(result->path(), StartsWith(home_cache_path));
  // Destruction shouldn't remove system cache directories.
  result = Error("nothing");
  EXPECT_TRUE(*tmp_dir_.Access("test_home"));
  EXPECT_TRUE(*test_home.Access(".cache"));

  // Same as with an empty string, but this time with an unset environment
  // variable.
  unsetenv(XdgCacheEnv);
  result = Runtimes::Cache::MakeSystem(install_);
  ASSERT_THAT(result, IsSuccess(_));
  EXPECT_THAT(result->path(), StartsWith(home_cache_path));
  // Destruction shouldn't remove system cache directories.
  result = Error("nothing");
  EXPECT_TRUE(*tmp_dir_.Access("test_home"));
  EXPECT_TRUE(*test_home.Access(".cache"));

  // Now check a bunch of different failure modes for the home directory
  // fallback. These should all end up creating temporary directories which
  // we'll test functionally at the end.
  setenv(HomeEnv, "", /*overwrite*/ true);
  EXPECT_THAT(Runtimes::Cache::MakeSystem(install_), IsSuccess(_));
  setenv(HomeEnv, "relative/home", /*overwrite*/ true);
  EXPECT_THAT(Runtimes::Cache::MakeSystem(install_), IsSuccess(_));

  // Correct the path and make sure it works again.
  setenv(HomeEnv, home_path.c_str(), /*overwrite*/ true);
  result = Runtimes::Cache::MakeSystem(install_);
  ASSERT_THAT(result, IsSuccess(_));
  EXPECT_THAT(result->path(), StartsWith(home_cache_path));

  // Now try removing directories around home.
  test_home.Rmtree(".cache").Check();
  EXPECT_THAT(Runtimes::Cache::MakeSystem(install_), IsSuccess(_));
  tmp_dir_.Rmtree("test_home").Check();
  EXPECT_THAT(Runtimes::Cache::MakeSystem(install_), IsSuccess(_));

  // Finally, double check that these temporary caches still produce a writable
  // directory.
  result = Runtimes::Cache::MakeSystem(install_);
  ASSERT_THAT(result, IsSuccess(_));
  EXPECT_THAT(result->path(), Not(StartsWith(home_cache_path)));
  EXPECT_THAT(result->path(), Not(StartsWith(xdg_path)));
  ASSERT_THAT(Filesystem::Cwd().WriteFileFromString(
                  result->path() / "test_file", "test"),
              IsSuccess(_));
  ASSERT_THAT(Filesystem::Cwd().ReadFileToString(result->path() / "test_file"),
              IsSuccess(StrEq("test")));
}

TEST_F(RuntimesCacheTest, BasicBuild) {
  llvm::SmallVector<std::string> targets = {"aarch64-unknown-unknown",
                                            "x86_64-unknown-unknown"};
  llvm::SmallVector<std::filesystem::path> built_runtimes_paths;

  for (const std::string& target : targets) {
    SCOPED_TRACE(target);
    auto lookup_result = cache_.Lookup({.target = target});
    ASSERT_THAT(lookup_result, IsSuccess(_));
    auto runtimes = *std::move(lookup_result);

    auto build_result = runtimes.Build(Runtimes::ClangResourceDir);
    ASSERT_THAT(build_result, IsSuccess(VariantWith<Runtimes::Builder>(_)));
    auto builder = std::get<Runtimes::Builder>(*std::move(build_result));
    EXPECT_TRUE(builder.path().is_absolute()) << builder.path();

    // Create a file as our "runtime".
    builder.dir().WriteFileFromString("runtime_file", target).Check();
    // Make sure the builder's path finds this file.
    EXPECT_THAT(
        Filesystem::Cwd().ReadFileToString(builder.path() / "runtime_file"),
        IsSuccess(StrEq(target)));

    auto commit_result = std::move(builder).Commit();
    ASSERT_THAT(commit_result, IsSuccess(_));
    std::filesystem::path clang_runtimes_path = *std::move(commit_result);

    EXPECT_THAT(
        runtimes.Build(Runtimes::ClangResourceDir),
        IsSuccess(VariantWith<std::filesystem::path>(Eq(clang_runtimes_path))));
    built_runtimes_paths.push_back(clang_runtimes_path);
  }

  for (const auto& [target, built_runtimes_path] :
       llvm::zip(targets, built_runtimes_paths)) {
    SCOPED_TRACE(target);
    auto lookup_result = cache_.Lookup({.target = target});
    ASSERT_THAT(lookup_result, IsSuccess(_));
    auto runtimes = *std::move(lookup_result);

    EXPECT_THAT(
        runtimes.Build(Runtimes::ClangResourceDir),
        IsSuccess(VariantWith<std::filesystem::path>(Eq(built_runtimes_path))));
  }
}

TEST_F(RuntimesCacheTest, DifferentKeys) {
  const std::string target = "aarch64-unknown-unknown";
  auto runtimes1 = *cache_.Lookup({.target = target});

  // Build a second cache with a different key but pointing at the same
  // directory and target to simulate two versions or builds of the Carbon
  // toolchain.
  auto custom_install_dir =
      *tmp_dir_.CreateDirectories("custom_install/lib/carbon");
  custom_install_dir.WriteFileFromString("carbon_install.txt", "diff digest")
      .Check();
  custom_install_dir.WriteFileFromString("install_digest.txt", "abcd").Check();
  InstallPaths install2 =
      InstallPaths::Make((tmp_dir_.abs_path() / "custom_install").native());
  auto cache2 = *Runtimes::Cache::MakeCustom(install2, tmp_dir_.abs_path());
  auto runtimes2 = *cache2.Lookup({.target = target});

  // The parent paths of these runtimes should be the same.
  EXPECT_THAT(runtimes1.base_path().parent_path(),
              Eq(runtimes2.base_path().parent_path()));

  // But the base paths for these two runtimes should differ due to cache key
  // differences.
  EXPECT_THAT(runtimes1.base_path(), Ne(runtimes2.base_path()));
}

TEST_F(RuntimesCacheTest, ConcurrentBuilds) {
  const std::string target = "aarch64-unknown-unknown";
  auto runtimes1 = *cache_.Lookup({.target = target});

  // Build a second cache and runtimes pointing at the same directory and target
  // to simulate concurrent processes.
  auto cache2 = *Runtimes::Cache::MakeCustom(install_, tmp_dir_.abs_path());
  auto runtimes2 = *cache2.Lookup({.target = target});

  // Start the first build, this will lock the directory.
  auto build_result1 = runtimes1.Build(Runtimes::ClangResourceDir);
  ASSERT_THAT(build_result1, IsSuccess(VariantWith<Runtimes::Builder>(_)));
  auto builder1 = std::get<Runtimes::Builder>(*std::move(build_result1));
  EXPECT_THAT(builder1.dir().WriteFileFromString("runtime_file", "build1"),
              IsSuccess(_));

  // Start the second build in a separate thread so that it can block while we
  // finish the first build. The only result we'll need at the end is the built
  // path.
  std::filesystem::path build2_path;
  auto build2_lambda = [&build2_path, &runtimes2] {
    // Typically building here will try to acquire the same file lock acquired
    // with the first build. However, the file locking is always _advisory_ and
    // may fail. As a consequence we can't make assumptions about whether this
    // blocks or not.
    auto build_result2 = runtimes2.Build(Runtimes::ClangResourceDir);
    ASSERT_THAT(build_result2, IsSuccess(_));
    if (std::holds_alternative<std::filesystem::path>(*build_result2)) {
      // In the common case, we blocked on a file lock and find the first built
      // result directly. Save it.
      build2_path = std::get<std::filesystem::path>(*std::move(build_result2));
    } else {
      // In rare cases, the initial build will fail to acquire the file lock.
      // The entire build process is designed specifically to be resilient to
      // that so we should still succeed, but now we need to handle building in
      // this thread as well. Note that a true failure here may only
      // show up intermittently.
      auto builder2 = std::get<Runtimes::Builder>(*std::move(build_result2));
      builder2.dir().WriteFileFromString("runtime_file", "build2").Check();
      auto commit2_result = std::move(builder2).Commit();
      ASSERT_THAT(commit2_result, IsSuccess(_));
      build2_path = *std::move(commit2_result);
    }
  };
  std::thread build2_thread(build2_lambda);
  // Use a scoped join to avoid leaking the thread as some platforms don't have
  // `std::jthread`.
  auto scoped_join =
      llvm::make_scope_exit([&build2_thread] { build2_thread.join(); });

  // Commit the first built runtime.
  auto commit_result = std::move(builder1).Commit();
  ASSERT_THAT(commit_result, IsSuccess(_));
  std::filesystem::path build1_path = *std::move(commit_result);

  // Even though there may be is another thread running, we should now get
  // non-blocking access directly to the built runtime.
  EXPECT_THAT(runtimes1.Build(Runtimes::ClangResourceDir),
              IsSuccess(VariantWith<std::filesystem::path>(Eq(build1_path))));

  // Now join the second cache's build thread to ensure it completes and verify
  // that it produces the same path fully-built path.
  build2_thread.join();
  scoped_join.release();
  EXPECT_THAT(build2_path, Eq(build1_path));

  // Note that we don't know which build actually ended up committed here so
  // accept either. The first one is much more common, but in rare cases it will
  // fail to acquire its file lock and we will have racing builds. In that case
  // the second build may commit first.
  EXPECT_THAT(*Filesystem::Cwd().ReadFileToString(build1_path / "runtime_file"),
              AnyOf(StrEq("build1"), StrEq("build2")));
}

TEST_F(RuntimesCacheTest, ConcurrentBuildsWithFailedLocking) {
  // This test is very similar to `ConcurrentBuild` in terms of what can happen.
  // But here, we intentionally subvert the file locking and even us
  // synchronization to maximize the chance of racing commits.
  //
  // The goal here is to do two things:
  // 1) Provide more direct stress testing of lock-file-failure modes and racing
  //    commits to catch any consistent bugs that emerge.
  // 2) Ensure that a removed lock file specifically is handled gracefully, both
  // by a build with the file open and locked, and by a racing build.
  const std::string target = "aarch64-unknown-unknown";
  auto runtimes1 = *cache_.Lookup({.target = target});

  // Build a second cache and runtimes pointing at the same directory and target
  // to simulate concurrent processes.
  auto cache2 = *Runtimes::Cache::MakeCustom(install_, tmp_dir_.abs_path());
  auto runtimes2 = *cache2.Lookup({.target = target});

  // Start the first build, this will lock the directory.
  auto build_result1 = runtimes1.Build(Runtimes::ClangResourceDir);
  ASSERT_THAT(build_result1, IsSuccess(VariantWith<Runtimes::Builder>(_)));
  auto builder1 = std::get<Runtimes::Builder>(*std::move(build_result1));
  builder1.dir().WriteFileFromString("runtime_file", "build1").Check();

  // Now sneakily remove the lock file from the runtimes directory in the cache.
  // This is something that could happen, for example from temporary directories
  // being cleaned. The cache should be resilient against this and it gives us a
  // good way to have two racing builds of the same directory.
  std::filesystem::path lock_file_path =
      RuntimesTestPeer::LockFilePath(Runtimes::ClangResourceDir);
  ASSERT_THAT(runtimes1.base_dir().Unlink(lock_file_path), IsSuccess(_));

  // We will synchronize with the thread to ensure we _actually_ have two
  // parallel builds rather than accidentally having a fully serial execution.
  std::mutex m;
  std::condition_variable cv;
  bool build_started = false;

  // Start the second build in a separate thread. The only result we'll need at
  // the end is the built path.
  std::filesystem::path build2_path;
  auto build2_lambda = [&build2_path, &runtimes2, target, &m, &cv,
                        &build_started] {
    auto build_result2 = runtimes2.Build(Runtimes::ClangResourceDir);
    ASSERT_THAT(build_result2, IsSuccess(VariantWith<Runtimes::Builder>(_)));
    auto builder2 = std::get<Runtimes::Builder>(*std::move(build_result2));
    builder2.dir().WriteFileFromString("runtime_file", "build2").Check();

    // Notify the first thread to commit its build and concurrently commit this
    // built runtime. The goal is to get as close as we can to having these
    // commits actually race so that a failure in that mode would emerge as a
    // flake of the test. None of this is providing correctness.
    {
      std::unique_lock lock(m);
      build_started = true;
      cv.notify_one();
    }
    auto commit_result = std::move(builder2).Commit();
    ASSERT_THAT(commit_result, IsSuccess(_));
    build2_path = *std::move(commit_result);

    // Even though there may be another thread running, and even holding a lock
    // file, we should now get non-blocking access directly to the built
    // runtime. This is mostly added for completeness, a held lock is more
    // directly tested in `CurrentBuildsLockTimeout`.
    EXPECT_THAT(runtimes2.Build(Runtimes::ClangResourceDir),
                IsSuccess(VariantWith<std::filesystem::path>(Eq(build2_path))));
  };
  std::thread build2_thread(build2_lambda);
  // Use a scoped join to avoid leaking the thread as some platforms don't have
  // `std::jthread`.
  auto scoped_join =
      llvm::make_scope_exit([&build2_thread] { build2_thread.join(); });

  // As soon as the second thread notifies that its build is started and ready
  // to commit, also commit the first built runtime.
  {
    std::unique_lock lock(m);
    cv.wait(lock, [&build_started] { return build_started; });
  }
  auto commit_result = std::move(builder1).Commit();
  ASSERT_THAT(commit_result, IsSuccess(_));
  std::filesystem::path build1_path = *std::move(commit_result);

  // Even though there may be another thread running, we should now get
  // non-blocking access directly to the built runtime.
  EXPECT_THAT(runtimes1.Build(Runtimes::ClangResourceDir),
              IsSuccess(VariantWith<std::filesystem::path>(Eq(build1_path))));

  // Now join the second cache's build thread to ensure it completes and verify
  // that it produces the same path fully-built path.
  build2_thread.join();
  scoped_join.release();
  EXPECT_THAT(build2_path, Eq(build1_path));

  // Much like the simple concurrent build, we can't know which build finished
  // first so we need to accept either build's runtime file.
  EXPECT_THAT(*Filesystem::Cwd().ReadFileToString(build1_path / "runtime_file"),
              AnyOf(StrEq("build1"), StrEq("build2")));
}

TEST_F(RuntimesCacheTest, ConcurrentBuildsLockTimeout) {
  // Another test designed to be similar to `ConcurrentBuilds` but stressing a
  // failure path. Here, we want to reliably exercise the code path where a lock
  // file is held when a second build begins and it polls and times out. This
  // can happen naturally, even with very large timeouts under sufficient system
  // load. Here, we artificially make it as likely as possible for better stress
  // testing and easier debugging of problems with this situation.
  const std::string target = "aarch64-unknown-unknown";
  auto runtimes1 = *cache_.Lookup({.target = target});

  // Build a second cache and runtimes pointing at the same directory and target
  // to simulate concurrent processes.
  auto cache2 = *Runtimes::Cache::MakeCustom(install_, tmp_dir_.abs_path());
  auto runtimes2 = *cache2.Lookup({.target = target});

  // Start the first build, this will lock the directory.
  auto build_result1 = runtimes1.Build(Runtimes::ClangResourceDir);
  ASSERT_THAT(build_result1, IsSuccess(VariantWith<Runtimes::Builder>(_)));
  auto builder1 = std::get<Runtimes::Builder>(*std::move(build_result1));
  builder1.dir().WriteFileFromString("runtime_file", "build1").Check();

  // Directly simulate a second thread or process timing out on acquiring the
  // file-based advisory lock by giving it an artificially short timeout and
  // running it in the same thread. This should only poll for 50ms before
  // proceeding without the lock.
  //
  // However, note that this is not *guaranteed* -- the first build may have
  // exhausted the much higher default poll timeout and failed to acquire a file
  // lock at all. When that happens, this path may in turn succeed at acquiring
  // the file lock. All of that is fine, and the test even remains effective as
  // either way we have successfully exercised the code path with lock file
  // timeout. The lowered time here just ensures that the test finishes promptly
  // relative to the system load.
  auto build_result2 = RuntimesTestPeer::BuildImpl(
      runtimes2, Runtimes::ClangResourceDir, std::chrono::milliseconds(50),
      std::chrono::milliseconds(10));
  ASSERT_THAT(build_result2, IsSuccess(VariantWith<Runtimes::Builder>(_)));
  auto builder2 = std::get<Runtimes::Builder>(*std::move(build_result2));
  builder2.dir().WriteFileFromString("runtime_file", "build2").Check();

  // Commit the second runtime, as this one *doesn't* hold any lock. This leaves
  // the lock present and held, but creates a valid runtimes directory.
  auto commit2_result = std::move(builder2).Commit();
  ASSERT_THAT(commit2_result, IsSuccess(_));
  std::filesystem::path build2_path = *std::move(commit2_result);

  // Now, even though we still have the lock file held, repeatedly building
  // proceeds without blocking.
  EXPECT_THAT(runtimes2.Build(Runtimes::ClangResourceDir),
              IsSuccess(VariantWith<std::filesystem::path>(Eq(build2_path))));

  // Finally, commit the lock-holding build to ensure it also succeeds, even
  // though it will reliably discard its built cache.
  auto commit1_result = std::move(builder1).Commit();
  ASSERT_THAT(commit1_result, IsSuccess(_));
  std::filesystem::path build1_path = *std::move(commit1_result);

  // And ensure that we got the same path and the second build's contents.
  EXPECT_THAT(build1_path, Eq(build2_path));
  EXPECT_THAT(*Filesystem::Cwd().ReadFileToString(build1_path / "runtime_file"),
              StrEq("build2"));
}

TEST_F(RuntimesCacheTest, Lookup) {
  // Basic successful lookup of a new runtimes.
  auto lookup_result = cache_.Lookup({.target = "aarch64-unknown-unknown"});
  ASSERT_THAT(lookup_result, IsSuccess(_));
  auto runtimes = *std::move(lookup_result);

  auto lock_stat = runtimes.base_dir().Stat(".lock");
  ASSERT_THAT(lock_stat, IsSuccess(_));
  EXPECT_TRUE(lock_stat->is_file());

  // Looking up the same target should return the same runtimes.
  lookup_result = cache_.Lookup({.target = "aarch64-unknown-unknown"});
  ASSERT_THAT(lookup_result, IsSuccess(_));
  auto runtimes2 = *std::move(lookup_result);
  EXPECT_THAT(runtimes2.base_path(), Eq(runtimes.base_path()));

  EXPECT_THAT(runtimes.base_dir().Stat()->unix_inode(),
              Eq(runtimes.base_dir().Stat()->unix_inode()));
}

TEST_F(RuntimesCacheTest, LookupFailsIfCannotCreateDir) {
  // Create a read-only directory with the cache in it to cause failures.
  std::filesystem::path ro_cache_path = tmp_dir_.abs_path() / "ro_cache";
  tmp_dir_.CreateDirectories("ro_cache", /*creation_mode=*/0500).Check();
  auto ro_cache = *Runtimes::Cache::MakeCustom(install_, ro_cache_path);

  auto lookup_result = ro_cache.Lookup({.target = "aarch64-unknown-unknown"});
  EXPECT_THAT(lookup_result, IsError(_));
}

TEST_F(RuntimesCacheTest, LookupWithSmallNumberOfStaleRuntimes) {
  // Lookup two runtimes to populate the cache.
  auto runtimes1 = *cache_.Lookup({.target = "aarch64-unknown-unknown1"});
  auto runtimes2 = *cache_.Lookup({.target = "aarch64-unknown-unknown2"});

  // Get the Unix-like inode of the directories so we can check whether
  // subsequent lookups create a new directory.
  auto runtimes1_inode = runtimes1.base_dir().Stat()->unix_inode();
  auto runtimes2_inode = runtimes2.base_dir().Stat()->unix_inode();

  // Now adjust their age backwards in time by two years to make them very, very
  // stale.
  auto now = Filesystem::Clock::now();
  auto two_years_ago = now - std::chrono::years(2);
  runtimes1.base_dir().UpdateTimes(".lock", two_years_ago).Check();
  runtimes2.base_dir().UpdateTimes(".lock", two_years_ago).Check();

  // Close the runtimes, releasing any locks.
  runtimes1 = {};
  runtimes2 = {};

  // Lookup a new runtime, potentially pruning stale ones.
  auto runtimes3 = *cache_.Lookup({.target = "aarch64-unknown-unknown3"});

  // Redo the previous lookups and ensure they found the original directories as
  // we don't have enough runtimes to prune.
  runtimes1 = *cache_.Lookup({.target = "aarch64-unknown-unknown1"});
  runtimes2 = *cache_.Lookup({.target = "aarch64-unknown-unknown2"});
  EXPECT_THAT(runtimes1.base_dir().Stat()->unix_inode(), Eq(runtimes1_inode));
  EXPECT_THAT(runtimes2.base_dir().Stat()->unix_inode(), Eq(runtimes2_inode));

  // The timestamp on the lock file should also be updated. We can't assume the
  // filesystem clock is monotonic, so it is possible an adjustment occurs while
  // this test is running. We check that the updated time is within 2 days of
  // `now` to minimize flake risks, which should be completely fine to detect
  // bugs as we set the time to 2 years in the past above.
  EXPECT_THAT(
      runtimes1.base_dir().Stat(".lock")->mtime(),
      AllOf(Gt(now - std::chrono::days(2)), Lt(now + std::chrono::days(2))));
  EXPECT_THAT(
      runtimes2.base_dir().Stat(".lock")->mtime(),
      AllOf(Gt(now - std::chrono::days(2)), Lt(now + std::chrono::days(2))));
}

TEST_F(RuntimesCacheTest, LookupWithManyStaleRuntimes) {
  auto runtimes1 = *cache_.Lookup({.target = "aarch64-unknown-unknown-fresh1"});
  auto stale_runtimes = LookupNRuntimes(RuntimesTestPeer::CacheMinNumEntries());
  auto runtimes2 = *cache_.Lookup({.target = "aarch64-unknown-unknown-fresh2"});

  // Get the Unix-like inode of the directories so we can check whether
  // subsequent lookups create a new directory.
  auto runtimes1_inode = runtimes1.base_dir().Stat()->unix_inode();
  auto stale_runtimes_0_inode =
      stale_runtimes[0].base_dir().Stat()->unix_inode();
  auto runtimes2_inode = runtimes2.base_dir().Stat()->unix_inode();

  // Now adjust their age backwards in time by two years to make them very, very
  // stale.
  auto now = Filesystem::Clock::now();
  auto two_years_ago = now - std::chrono::years(2);
  for (auto& stale_runtime : stale_runtimes) {
    stale_runtime.base_dir().UpdateTimes(".lock", two_years_ago).Check();
  }

  // Close the runtimes, releasing any locks.
  runtimes1 = {};
  stale_runtimes.clear();
  runtimes2 = {};

  // Lookup a new runtime, potentially pruning stale ones.
  auto runtimes3 = *cache_.Lookup({.target = "aarch64-unknown-unknown-fresh3"});

  // Re-lookup three of the original runtimes.
  runtimes1 = *cache_.Lookup({.target = "aarch64-unknown-unknown-fresh1"});
  auto stale_runtimes_0 =
      *cache_.Lookup({.target = "aarch64-unknown-unknown0"});
  runtimes2 = *cache_.Lookup({.target = "aarch64-unknown-unknown-fresh2"});

  // The first and last should have been preserved as they were not stale.
  EXPECT_THAT(runtimes1.base_dir().Stat()->unix_inode(), Eq(runtimes1_inode));
  EXPECT_THAT(runtimes2.base_dir().Stat()->unix_inode(), Eq(runtimes2_inode));

  // One of the stale runtimes should be freshly created though.
  EXPECT_THAT(stale_runtimes_0.base_dir().Stat()->unix_inode(),
              Ne(stale_runtimes_0_inode));
}

TEST_F(RuntimesCacheTest, LookupWithTooManyRuntimes) {
  auto runtimes1 = *cache_.Lookup({.target = "aarch64-unknown-unknown-fresh1"});
  auto runtimes2 = *cache_.Lookup({.target = "aarch64-unknown-unknown-fresh2"});
  int n = RuntimesTestPeer::CacheMaxNumEntries();
  auto stale_runtimes = LookupNRuntimes(n);

  // Compute stale target strings.
  auto stale_runtimes_n_1_target =
      llvm::formatv("aarch64-unknown-unknown{0}", n - 1).str();
  auto stale_runtimes_n_2_target =
      llvm::formatv("aarch64-unknown-unknown{0}", n - 2).str();

  // Get the Unix-like inode of the directories so we can check whether
  // subsequent lookups create a new directory.
  auto runtimes1_inode = runtimes1.base_dir().Stat()->unix_inode();
  auto runtimes2_inode = runtimes2.base_dir().Stat()->unix_inode();
  auto stale_runtimes_0_inode =
      stale_runtimes[0].base_dir().Stat()->unix_inode();
  auto stale_runtimes_n_inode =
      stale_runtimes.back().base_dir().Stat()->unix_inode();
  auto stale_runtimes_n_1_inode =
      std::prev(stale_runtimes.end(), 2)->base_dir().Stat()->unix_inode();

  // Now manually set all the timestamps. We do this manually to avoid any
  // reliance on the clock behavior or the amount of time passing between lookup
  // calls.
  auto now = Filesystem::Clock::now();
  runtimes1.base_dir().UpdateTimes(".lock", now).Check();
  runtimes2.base_dir().UpdateTimes(".lock", now).Check();
  // Now set the stale runtimes to times further and further in the past.
  now -= std::chrono::milliseconds(1);
  for (auto [i, stale_runtime] : llvm::enumerate(stale_runtimes)) {
    stale_runtime.base_dir()
        .UpdateTimes(".lock", now - std::chrono::milliseconds(i * i))
        .Check();
  }

  // Close most of the runtimes to release the locks, but keep the oldest stale
  // runtime locked along with a fresh one to exercise the locking path.
  runtimes1 = {};
  auto stale_runtime_n_orig = stale_runtimes.pop_back_val();
  stale_runtimes.clear();

  // Lookup a new runtime, potentially pruning stale ones.
  auto runtimes3 = *cache_.Lookup({.target = "aarch64-unknown-unknown-fresh3"});

  // Re-lookup three of the original runtimes.
  runtimes1 = *cache_.Lookup({.target = "aarch64-unknown-unknown-fresh1"});
  runtimes2 = *cache_.Lookup({.target = "aarch64-unknown-unknown-fresh2"});
  auto stale_runtimes_0 =
      *cache_.Lookup({.target = "aarch64-unknown-unknown0"});
  auto stale_runtimes_n = *cache_.Lookup({.target = stale_runtimes_n_1_target});
  auto stale_runtimes_n_1 =
      *cache_.Lookup({.target = stale_runtimes_n_2_target});

  // The fresh runtimes should be preserved.
  EXPECT_THAT(runtimes1.base_dir().Stat()->unix_inode(), Eq(runtimes1_inode));
  EXPECT_THAT(runtimes2.base_dir().Stat()->unix_inode(), Eq(runtimes2_inode));
  EXPECT_THAT(stale_runtimes_0.base_dir().Stat()->unix_inode(),
              Eq(stale_runtimes_0_inode));

  // THe last stale runtime should have been locked and so should remain.
  EXPECT_THAT(stale_runtimes_n.base_dir().Stat()->unix_inode(),
              Eq(stale_runtimes_n_inode));

  // The next to last should have been pruned and re-created though.
  EXPECT_THAT(stale_runtimes_n.base_dir().Stat()->unix_inode(),
              Ne(stale_runtimes_n_1_inode));
}

}  // namespace
}  // namespace Carbon
