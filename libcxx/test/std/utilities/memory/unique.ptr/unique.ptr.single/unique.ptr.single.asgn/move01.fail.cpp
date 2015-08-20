//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// Test unique_ptr move assignment

#include <memory>

#include "test_macros.h"

// Can't copy from lvalue
int main()
{
    std::unique_ptr<int> s, s2;
#if TEST_STD_VER >= 11
    s2 = s; // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
#else
    s2 = s; // expected-error {{'operator=' is a private member}}
#endif
}
