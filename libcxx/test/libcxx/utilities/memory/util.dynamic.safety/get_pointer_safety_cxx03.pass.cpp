//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// pointer_safety get_pointer_safety();

#include <memory>
#include <cassert>

int main()
{
  // Test that std::pointer_safety is still offered in C++03 under the old ABI.
#ifndef _LIBCPP_ABI_POINTER_SAFETY_ENUM_TYPE
    std::pointer_safety r = std::get_pointer_safety();
    assert(r == std::pointer_safety::relaxed ||
           r == std::pointer_safety::preferred ||
           r == std::pointer_safety::strict);
#endif
}
