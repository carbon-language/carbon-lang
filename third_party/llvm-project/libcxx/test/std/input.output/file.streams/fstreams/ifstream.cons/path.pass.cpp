//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-filesystem

// Filesystem is supported on Apple platforms starting with macosx10.15.
// UNSUPPORTED: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14}}

// FILE_DEPENDENCIES: test.dat

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ifstream

// explicit basic_ifstream(const filesystem::path& s,
//     ios_base::openmode mode = ios_base::in);

#include <fstream>
#include <filesystem>
#include <cassert>

#include "test_macros.h"

namespace fs = std::filesystem;

int main(int, char**) {
  {
    fs::path p;
    static_assert(!std::is_convertible<fs::path, std::ifstream>::value,
                  "ctor should be explicit");
    static_assert(std::is_constructible<std::ifstream, fs::path const&,
                                        std::ios_base::openmode>::value,
                  "");
  }
  {
    std::ifstream fs(fs::path("test.dat"));
    double x = 0;
    fs >> x;
    assert(x == 3.25);
  }
  // std::ifstream(const fs::path&, std::ios_base::openmode) is tested in
  // test/std/input.output/file.streams/fstreams/ofstream.cons/string.pass.cpp
  // which creates writable files.

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    std::wifstream fs(fs::path("test.dat"));
    double x = 0;
    fs >> x;
    assert(x == 3.25);
  }
  // std::wifstream(const fs::path&, std::ios_base::openmode) is tested in
  // test/std/input.output/file.streams/fstreams/ofstream.cons/string.pass.cpp
  // which creates writable files.
#endif

  return 0;
}
