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

// const path& path() const noexcept;
// operator const path&() const noexcept;

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"


void test_path_method() {
  using namespace fs;
  const path p("foo/bar/baz.exe");
  const path p2("abc");
  {
    directory_entry nce;
    const directory_entry e("");
    static_assert(std::is_same<decltype(e.path()),  const path&>::value, "");
    static_assert(std::is_same<decltype(nce.path()), const path&>::value, "");
    static_assert(noexcept(e.path()) && noexcept(nce.path()), "");
  }
  {
    directory_entry e(p);
    path const& pref = e.path();
    assert(pref == p);
    assert(&pref == &e.path());
    e.assign(p2);
    assert(pref == p2);
    assert(&pref == &e.path());
  }
}

void test_path_conversion() {
  using namespace fs;
  const path p("foo/bar/baz.exe");
  const path p2("abc");
  {
    directory_entry nce;
    const directory_entry e("");
    // Check conversions exist
    static_assert(std::is_convertible<directory_entry&,        path const&>::value, "");
    static_assert(std::is_convertible<directory_entry const&,  path const&>::value, "");
    static_assert(std::is_convertible<directory_entry &&,      path const&>::value, "");
    static_assert(std::is_convertible<directory_entry const&&, path const&>::value, "");
    // Not convertible to non-const
    static_assert(!std::is_convertible<directory_entry&,        path&>::value, "");
    static_assert(!std::is_convertible<directory_entry const&,  path&>::value, "");
    static_assert(!std::is_convertible<directory_entry &&,      path&>::value, "");
    static_assert(!std::is_convertible<directory_entry const&&, path&>::value, "");
    // conversions are noexcept
    static_assert(noexcept(e.operator fs::path const&()) &&
                  noexcept(e.operator fs::path const&()), "");
  }
  // const
  {
    directory_entry const e(p);
    path const& pref = e;
    assert(&pref == &e.path());
  }
  // non-const
  {
    directory_entry e(p);
    path const& pref = e;
    assert(&pref == &e.path());

    e.assign(p2);
    assert(pref == p2);
    assert(&pref == &e.path());
  }
}

int main(int, char**) {
  test_path_method();
  test_path_conversion();

  return 0;
}
