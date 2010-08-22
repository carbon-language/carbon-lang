//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// binary_function

#include <functional>
#include <type_traits>

int main()
{
    typedef std::binary_function<int, short, bool> bf;
    static_assert((std::is_same<bf::first_argument_type, int>::value), "");
    static_assert((std::is_same<bf::second_argument_type, short>::value), "");
    static_assert((std::is_same<bf::result_type, bool>::value), "");
}
