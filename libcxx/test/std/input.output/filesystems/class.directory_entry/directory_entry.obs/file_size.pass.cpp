//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// The string reported on errors changed, which makes those tests fail when run
// against already-released libc++'s.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.15

// <filesystem>

// class directory_entry

// uintmax_t file_size() const;
// uintmax_t file_size(error_code const&) const noexcept;

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "filesystem_test_helper.h"
#include "rapid-cxx-test.h"

#include "test_macros.h"

TEST_SUITE(directory_entry_obs_testsuite)

TEST_CASE(signatures) {
  using namespace fs;
  {
    const fs::directory_entry e = {};
    std::error_code ec;
    static_assert(std::is_same<decltype(e.file_size()), uintmax_t>::value, "");
    static_assert(std::is_same<decltype(e.file_size(ec)), uintmax_t>::value,
                  "");
    static_assert(noexcept(e.file_size()) == false, "");
    static_assert(noexcept(e.file_size(ec)) == true, "");
  }
}

TEST_CASE(basic) {
  using namespace fs;

  scoped_test_env env;
  const path file = env.create_file("file", 42);
  const path dir = env.create_dir("dir");
  const path sym = env.create_symlink("file", "sym");

  {
    directory_entry ent(file);
    uintmax_t expect = file_size(ent);
    TEST_CHECK(expect == 42);

    // Remove the file to show that the results were already in the cache.
    LIBCPP_ONLY(remove(file));

    std::error_code ec = GetTestEC();
    TEST_CHECK(ent.file_size(ec) == expect);
    TEST_CHECK(!ec);
  }
  env.create_file("file", 99);
  {
    directory_entry ent(sym);

    uintmax_t expect = file_size(ent);
    TEST_CHECK(expect == 99);

    LIBCPP_ONLY(remove(ent));

    std::error_code ec = GetTestEC();
    TEST_CHECK(ent.file_size(ec) == 99);
    TEST_CHECK(!ec);
  }
}

TEST_CASE(not_regular_file) {
  using namespace fs;

  scoped_test_env env;
  struct {
    const path p;
    std::errc expected_err;
  } TestCases[] = {
      {env.create_dir("dir"), std::errc::is_a_directory},
#ifndef _WIN32
      {env.create_fifo("fifo"), std::errc::not_supported},
#endif
      {env.create_directory_symlink("dir", "sym"), std::errc::is_a_directory}};

  for (auto const& TC : TestCases) {
    const path& p = TC.p;
    directory_entry ent(p);
    TEST_CHECK(ent.path() == p);
    std::error_code ec = GetTestEC(0);

    std::error_code other_ec = GetTestEC(1);
    uintmax_t expect = file_size(p, other_ec);

    uintmax_t got = ent.file_size(ec);
    TEST_CHECK(got == expect);
    TEST_CHECK(got == uintmax_t(-1));
    TEST_CHECK(ec == other_ec);
    TEST_CHECK(ErrorIs(ec, TC.expected_err));

    ExceptionChecker Checker(p, TC.expected_err, "directory_entry::file_size");
    TEST_CHECK_THROW_RESULT(filesystem_error, Checker, ent.file_size());
  }
}

TEST_CASE(error_reporting) {
  using namespace fs;

  static_test_env static_env;
  scoped_test_env env;

  const path dir = env.create_dir("dir");
  const path file = env.create_file("dir/file", 42);
  const path file_out_of_dir = env.create_file("file2", 101);
  const path sym_out_of_dir = env.create_symlink("dir/file", "sym");
  const path sym_in_dir = env.create_symlink("file2", "dir/sym2");

#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
  const perms old_perms = status(dir).permissions();
#endif

  // test a file which doesn't exist
  {
    directory_entry ent;

    std::error_code ec = GetTestEC();
    ent.assign(static_env.DNE, ec);
    TEST_REQUIRE(ent.path() == static_env.DNE);
    TEST_CHECK(ErrorIs(ec, std::errc::no_such_file_or_directory));

    ec = GetTestEC();
    TEST_CHECK(ent.file_size(ec) == uintmax_t(-1));
    TEST_CHECK(ErrorIs(ec, std::errc::no_such_file_or_directory));

    ExceptionChecker Checker(static_env.DNE,
                             std::errc::no_such_file_or_directory,
                             "directory_entry::file_size");
    TEST_CHECK_THROW_RESULT(filesystem_error, Checker, ent.file_size());
  }
  // test a dead symlink
  {
    directory_entry ent;

    std::error_code ec = GetTestEC();
    uintmax_t expect_bad = file_size(static_env.BadSymlink, ec);
    TEST_CHECK(expect_bad == uintmax_t(-1));
    TEST_CHECK(ErrorIs(ec, std::errc::no_such_file_or_directory));

    ec = GetTestEC();
    ent.assign(static_env.BadSymlink, ec);
    TEST_REQUIRE(ent.path() == static_env.BadSymlink);
    TEST_CHECK(!ec);

    ec = GetTestEC();
    TEST_CHECK(ent.file_size(ec) == expect_bad);
    TEST_CHECK(ErrorIs(ec, std::errc::no_such_file_or_directory));

    ExceptionChecker Checker(static_env.BadSymlink,
                             std::errc::no_such_file_or_directory,
                             "directory_entry::file_size");
    TEST_CHECK_THROW_RESULT(filesystem_error, Checker, ent.file_size());
  }
  // Windows doesn't support setting perms::none to trigger failures
  // reading directories.
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
  // test a file w/o appropriate permissions.
  {
    directory_entry ent;
    uintmax_t expect_good = file_size(file);
    permissions(dir, perms::none);

    std::error_code ec = GetTestEC();
    ent.assign(file, ec);
    TEST_REQUIRE(ent.path() == file);
    TEST_CHECK(ErrorIs(ec, std::errc::permission_denied));

    ec = GetTestEC();
    TEST_CHECK(ent.file_size(ec) == uintmax_t(-1));
    TEST_CHECK(ErrorIs(ec, std::errc::permission_denied));

    ExceptionChecker Checker(file, std::errc::permission_denied, "file_size");
    TEST_CHECK_THROW_RESULT(filesystem_error, Checker, ent.file_size());

    permissions(dir, old_perms);
    ec = GetTestEC();
    TEST_CHECK(ent.file_size(ec) == expect_good);
    TEST_CHECK(!ec);
    TEST_CHECK_NO_THROW(ent.file_size());
  }
  permissions(dir, old_perms);
  // test a symlink w/o appropriate permissions.
  {
    directory_entry ent;
    uintmax_t expect_good = file_size(sym_in_dir);
    permissions(dir, perms::none);

    std::error_code ec = GetTestEC();
    ent.assign(sym_in_dir, ec);
    TEST_REQUIRE(ent.path() == sym_in_dir);
    TEST_CHECK(ErrorIs(ec, std::errc::permission_denied));

    ec = GetTestEC();
    TEST_CHECK(ent.file_size(ec) == uintmax_t(-1));
    TEST_CHECK(ErrorIs(ec, std::errc::permission_denied));

    ExceptionChecker Checker(sym_in_dir, std::errc::permission_denied,
                             "file_size");
    TEST_CHECK_THROW_RESULT(filesystem_error, Checker, ent.file_size());

    permissions(dir, old_perms);
    ec = GetTestEC();
    TEST_CHECK(ent.file_size(ec) == expect_good);
    TEST_CHECK(!ec);
    TEST_CHECK_NO_THROW(ent.file_size());
  }
  permissions(dir, old_perms);
  // test a symlink to a file w/o appropriate permissions
  {
    directory_entry ent;
    uintmax_t expect_good = file_size(sym_out_of_dir);
    permissions(dir, perms::none);

    std::error_code ec = GetTestEC();
    ent.assign(sym_out_of_dir, ec);
    TEST_REQUIRE(ent.path() == sym_out_of_dir);
    TEST_CHECK(!ec);

    ec = GetTestEC();
    TEST_CHECK(ent.file_size(ec) == uintmax_t(-1));
    TEST_CHECK(ErrorIs(ec, std::errc::permission_denied));

    ExceptionChecker Checker(sym_out_of_dir, std::errc::permission_denied,
                             "file_size");
    TEST_CHECK_THROW_RESULT(filesystem_error, Checker, ent.file_size());

    permissions(dir, old_perms);
    ec = GetTestEC();
    TEST_CHECK(ent.file_size(ec) == expect_good);
    TEST_CHECK(!ec);
    TEST_CHECK_NO_THROW(ent.file_size());
  }
#endif
}

TEST_SUITE_END()
