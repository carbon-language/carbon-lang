//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Call __clear_and_shrink() and ensure string invariants hold

#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::string l = "Long string so that allocation definitely, for sure, absolutely happens. Probably.";
    std::string s = "short";

    assert(l.__invariants());
    assert(s.__invariants());

    s.__clear_and_shrink();
    assert(s.__invariants());
    assert(s.size() == 0);

    {
    std::string::size_type cap = l.capacity();
    l.__clear_and_shrink();
    assert(l.__invariants());
    assert(l.size() == 0);
    assert(l.capacity() < cap);
    }

    return 0;
}
