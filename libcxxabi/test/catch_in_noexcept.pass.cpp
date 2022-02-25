//===---------------------- catch_in_noexcept.cpp--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: no-exceptions

#include <exception>
#include <stdlib.h>
#include <assert.h>

struct A {};

// Despite being marked as noexcept, this function must have an EHT entry that
// is not 'cantunwind', so that the unwinder can correctly deal with the throw.
void f1() noexcept
{
    try {
        A a;
        throw a;
        assert(false);
    } catch (...) {
        assert(true);
        return;
    }
    assert(false);
}

int main(int, char**)
{
    f1();

    return 0;
}
