//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class directory_entry

// directory_entry& operator=(directory_entry const&) = default;
// directory_entry& operator=(directory_entry&&) noexcept = default;
// void assign(path const&);
// void replace_filename(path const&);

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

TEST_SUITE(directory_entry_mods_suite)

TEST_CASE(test_path_assign_method) {
  using namespace fs;
  const path p("foo/bar/baz");
  const path p2("abc");
  directory_entry e(p);
  {
    static_assert(std::is_same<decltype(e.assign(p)), void>::value,
                  "return type should be void");
    static_assert(noexcept(e.assign(p)) == false,
                  "operation must not be noexcept");
  }
  {
    TEST_CHECK(e.path() == p);
    e.assign(p2);
    TEST_CHECK(e.path() == p2 && e.path() != p);
    e.assign(p);
    TEST_CHECK(e.path() == p && e.path() != p2);
  }
}

TEST_CASE(test_path_assign_ec_method) {
  using namespace fs;
  const path p("foo/bar/baz");
  const path p2("abc");
  {
    std::error_code ec;
    directory_entry e(p);
    static_assert(std::is_same<decltype(e.assign(p, ec)), void>::value,
                  "return type should be void");
    static_assert(noexcept(e.assign(p, ec)) == false,
                  "operation must not be noexcept");
  }
  {
    directory_entry ent(p);
    std::error_code ec = GetTestEC();
    ent.assign(p2, ec);
    TEST_CHECK(ErrorIs(ec, std::errc::no_such_file_or_directory));
    TEST_CHECK(ent.path() == p2);
  }
}

TEST_CASE(test_assign_calls_refresh) {
  using namespace fs;
  scoped_test_env env;
  const path dir = env.create_dir("dir");
  const path file = env.create_file("dir/file", 42);
  const path sym = env.create_symlink("dir/file", "sym");

  {
    directory_entry ent;
    ent.assign(file);

    // removing the file demonstrates that the values where cached previously.
    LIBCPP_ONLY(remove(file));

    TEST_CHECK(ent.is_regular_file());
  }
  env.create_file("dir/file", 101);
  {
    directory_entry ent;
    ent.assign(sym);

    LIBCPP_ONLY(remove(file));
    LIBCPP_ONLY(remove(sym));

    TEST_CHECK(ent.is_symlink());
    TEST_CHECK(ent.is_regular_file());
  }
}

TEST_CASE(test_assign_propagates_error) {
  using namespace fs;
  scoped_test_env env;
#ifdef _WIN32
  // Windows doesn't support setting perms::none to trigger failures
  // reading directories; test using a special inaccessible directory
  // instead.
  const path dir = GetWindowsInaccessibleDir();
  if (dir.empty())
    TEST_UNSUPPORTED();
  const path file = dir / "inaccessible_file";
  // We can't create files in the inaccessible directory, so this doesn't
  // test exactly the same as the code below.
  const path sym_out_of_dir = env.create_symlink(file, "sym");
  {
    directory_entry ent;
    std::error_code ec = GetTestEC();
    ent.assign(file, ec);
    TEST_CHECK(ErrorIs(ec, std::errc::no_such_file_or_directory));
  }
#else
  const path dir = env.create_dir("dir");
  const path file = env.create_file("dir/file", 42);
  const path sym_out_of_dir = env.create_symlink("dir/file", "sym");
  const path file_out_of_dir = env.create_file("file1");
  const path sym_in_dir = env.create_symlink("file1", "dir/sym1");

  permissions(dir, perms::none);

  {
    directory_entry ent;
    std::error_code ec = GetTestEC();
    ent.assign(file, ec);
    TEST_CHECK(ErrorIs(ec, std::errc::permission_denied));
  }
  {
    directory_entry ent;
    std::error_code ec = GetTestEC();
    ent.assign(sym_in_dir, ec);
    TEST_CHECK(ErrorIs(ec, std::errc::permission_denied));
  }
#endif
  {
    directory_entry ent;
    std::error_code ec = GetTestEC();
    ent.assign(sym_out_of_dir, ec);
    TEST_CHECK(!ec);
  }
}

TEST_SUITE_END()
