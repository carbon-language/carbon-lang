//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

// Test that reference wrapper meets the requirements of TriviallyCopyable,
// CopyConstructible and CopyAssignable.

#include <functional>
#include <type_traits>

int main()
{
    typedef std::reference_wrapper<int> T;
    static_assert(std::is_copy_constructible<T>::value, "");
    static_assert(std::is_copy_assignable<T>::value, "");
    // Extension up for standardization: See N4151.
    static_assert(std::is_trivially_copyable<T>::value, "");
}
