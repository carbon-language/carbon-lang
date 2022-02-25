//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iostream>

// istream wcin;

// XFAIL: libcpp-has-no-wide-characters

// UNSUPPORTED: executor-has-no-bash
// FILE_DEPENDENCIES: ../send-stdin.sh
// RUN: %{build}
// RUN: %{exec} bash send-stdin.sh "%t.exe" "1234"

#include <iostream>
#include <cassert>

int main(int, char**) {
    int i;
    std::wcin >> i;
    assert(i == 1234);
    return 0;
}
