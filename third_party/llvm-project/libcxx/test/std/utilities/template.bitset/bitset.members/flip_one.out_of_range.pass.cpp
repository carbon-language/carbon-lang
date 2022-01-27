//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// test bitset<N>& flip(size_t pos);

// Make sure we throw std::out_of_range when calling flip() on an OOB index.

#include <bitset>
#include <cassert>
#include <stdexcept>

int main(int, char**) {
    {
        std::bitset<0> v;
        try { v.flip(0); assert(false); } catch (std::out_of_range const&) { }
    }
    {
        std::bitset<1> v("0");
        try { v.flip(2); assert(false); } catch (std::out_of_range const&) { }
    }
    {
        std::bitset<10> v("0000000000");
        try { v.flip(10); assert(false); } catch (std::out_of_range const&) { }
    }

    return 0;
}
