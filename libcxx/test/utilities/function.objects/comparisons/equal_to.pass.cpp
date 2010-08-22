//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// equal_to

#include <functional>
#include <type_traits>
#include <cassert>

int main()
{
    typedef std::equal_to<int> F;
    const F f = F();
    static_assert((std::is_base_of<std::binary_function<int, int, bool>, F>::value), "");
    assert(f(36, 36));
    assert(!f(36, 6));
}
